#!/usr/bin/env python3

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import urllib.request

__version__ = "3.1.0"

# bits per weight. i guess?

QUANTS = {
    "IQ1_S": 1.56, "IQ2_XXS": 2.06, "IQ2_XS": 2.31,
    "Q2_K": 2.96, "IQ3_XXS": 3.06, "IQ4_XS": 3.85,
    "Q3_K_S": 3.50, "Q3_K_M": 3.91,
    "Q4_0": 4.55, "Q4_K_S": 4.59, "Q4_K_M": 4.85,
    "Q5_0": 5.54, "Q5_K_S": 5.54, "Q5_K_M": 5.69,
    "Q6_K": 6.57, "Q8_0": 8.50, "F16": 16.00,
}

DEFAULT_QUANTS = ["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
RECOMMENDED = "Q4_K_M"

# quality ranking: higher index = lower quality
QUANT_QUALITY_ORDER = [
    "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0",
    "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S",
    "Q2_K", "IQ4_XS", "IQ3_XXS", "IQ2_XS", "IQ2_XXS", "IQ1_S",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# oh god, hardware detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def detect_ram():
    # total system ram in gb. or none if i messed up.
    s = platform.system()
    try:
        if s == "Linux":
            with open("/proc/meminfo") as f:
                for ln in f:
                    if ln.startswith("MemTotal"):
                        return int(ln.split()[1]) / 1_048_576
        elif s == "Darwin":
            o = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True, stderr=subprocess.DEVNULL,
            )
            return int(o.strip()) / (1024**3)
        elif s == "Windows":
            o = subprocess.check_output(
                [
                    "powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
                ],
                text=True, stderr=subprocess.DEVNULL,
            )
            return int(o.strip()) / (1024**3)
    except Exception:
        pass
    return None


def detect_gpus():
    # returns list of {name, vram_gb, kind}. hope this works.
    gpus = []

    # nvidia. expensive and loud.
    if shutil.which("nvidia-smi"):
        try:
            o = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True, stderr=subprocess.DEVNULL,
            )
            for ln in o.strip().splitlines():
                p = [x.strip() for x in ln.split(",")]
                if len(p) >= 2:
                    try:
                        mb = float(p[1])
                    except ValueError:
                        continue
                    gpus.append(
                        dict(name=p[0], vram_gb=round(mb / 1024, 1), kind="nvidia")
                    )
        except Exception:
            pass

    # ── AMD (sysfs on Linux) ──
    if platform.system() == "Linux":
        drm = "/sys/class/drm"
        try:
            for d in sorted(os.listdir(drm)):
                vt = os.path.join(drm, d, "device", "mem_info_vram_total")
                if os.path.isfile(vt):
                    with open(vt) as f:
                        vb = int(f.read().strip())
                    if vb > 0:
                        np_ = os.path.join(drm, d, "device", "product_name")
                        try:
                            with open(np_) as f:
                                nm = f.read().strip()
                        except Exception:
                            nm = f"AMD GPU ({d})"
                        gpus.append(
                            dict(name=nm, vram_gb=round(vb / (1024**3), 1), kind="amd")
                        )
        except Exception:
            pass

    # ── AMD (rocm-smi fallback) ──
    if not any(g["kind"] == "amd" for g in gpus) and shutil.which("rocm-smi"):
        try:
            o = subprocess.check_output(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                text=True, stderr=subprocess.DEVNULL,
            )
            data = json.loads(o)
            for key, val in data.items():
                if isinstance(val, dict):
                    for k, v in val.items():
                        if "total" in k.lower() and "vram" in k.lower():
                            try:
                                vb = int(v)
                                gpus.append(
                                    dict(
                                        name=f"AMD GPU ({key})",
                                        vram_gb=round(vb / (1024**3), 1),
                                        kind="amd",
                                    )
                                )
                            except (ValueError, TypeError):
                                pass
        except Exception:
            pass

    # ── Apple Silicon (unified memory) ──
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        ram = detect_ram()
        if ram:
            try:
                chip = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True, stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                chip = "Apple Silicon"
            gpus.append(dict(name=chip, vram_gb=round(ram, 1), kind="apple"))

    return gpus


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# math. unfortunately necessary.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def default_overhead(kind, mem_gb):
    # memory overhead for kv-cache, cuda/metal context, and os.
    # i have no idea if these numbers are correct.
    if kind == "apple":
        return max(4.0, mem_gb * 0.10)  # macOS keeps its share
    if kind in ("nvidia", "amd"):
        return 2.0  # CUDA/ROCm context + KV cache
    if kind == "hybrid":
        return 4.0  # VRAM overhead + RAM overhead
    return 3.0  # CPU/RAM: OS + runtime


# additional runtime overhead beyond model weights (KV cache, buffers, etc.)
# this is what actually gets used at runtime, separate from the 2GB base overhead
RUNTIME_OVERHEAD_GB = 1.5  # accounts for KV cache (~800MB) + compute buffers (~500MB) + misc


def max_billions(mem_gb, quant, overhead_gb):
    # max model size (billion params) that fits at given quantization.
    # math. unfortunately necessary.
    bpw = QUANTS[quant]
    avail = mem_gb - overhead_gb
    return max(0.0, avail * 8.0 / bpw)


def make_url(max_b):
    # huggingface model search url. pray it doesn't rate limit.
    cap = max(1, math.floor(max_b))
    return (
        f"https://huggingface.co/models"
        f"?num_parameters=max:{cap}B&apps=llama.cpp&sort=trending"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# huggingface api calls. pray it doesn't rate limit.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def parse_model_id(model_arg):
    # parse user input into (repo_id, quant_suffix).
    if ":" in model_arg:
        repo_id, quant_suffix = model_arg.rsplit(":", 1)
        return repo_id, quant_suffix
    return model_arg, None


def fetch_gguf_files(repo_id):
    # fetch gguf file info from huggingface. api, please work.
    api_url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    try:
        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": "gguf-fit/1.1"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return None, str(e)

    gguf_files = []
    for item in data:
        if item.get("type") != "file":
            continue
        path = item.get("path", "")
        if not path.lower().endswith(".gguf"):
            continue
        # skip mmproj files
        if "mmproj" in path.lower():
            continue
        size_bytes = item.get("size", 0)
        size_gb = size_bytes / (1024**3)
        gguf_files.append({
            "name": path,
            "size_gb": round(size_gb, 2),
        })

    if not gguf_files:
        return None, "no GGUF files found in repository"

    return gguf_files, None


def infer_quant_from_filename(filename):
    # extract quantization type from gguf filename. hopefully.
    name = filename.upper()
    # check for Qn_K_M etc first (more specific)
    for q in QUANTS:
        if q in name:
            return q
    # fallback: check for Qn_n patterns
    return None


def is_4bit_quant(quant):
    # check if quantization is ~4-bit. q3, q4 variants.
    return quant and quant.startswith(("Q3_", "Q4_", "IQ3", "IQ4"))


def find_best_fit(gguf_files, mem_available, overhead_gb):
    # this heap of logic somehow finds the best quant. don't ask me how.
    avail = mem_available - overhead_gb
    if avail <= 0:
        return None, None, None

    # first: check if any 4-bit quant fits
    has_4bit = False
    for f in gguf_files:
        quant = infer_quant_from_filename(f["name"])
        if quant and is_4bit_quant(quant) and f["size_gb"] <= avail:
            has_4bit = True
            break

    if not has_4bit:
        return None, None, None

    # 4-bit fits! now find the highest quality quant that fits
    suitable = []
    for f in gguf_files:
        quant = infer_quant_from_filename(f["name"])
        if not quant:
            continue
        if f["size_gb"] <= avail:
            suitable.append((f, quant))

    if not suitable:
        return None, None, None

    # prefer higher quant (better quality) - sort by bpw descending
    suitable.sort(key=lambda x: QUANTS.get(x[1], 0), reverse=True)
    best = suitable[0]
    return best[0], best[1], best[0]["size_gb"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# model discovery. api, i'm begging you.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_useful_type(model_data: dict) -> str:
    # pick the most useful type tag from model metadata.
    # this is subjective and probably wrong, but here we are.
    tags = [t.lower() for t in model_data.get("tags", [])]
    pipeline = model_data.get("pipeline_tag", "")

    # priority order: most specific/useful first
    if "uncensored" in tags or "abliterated" in tags:
        return "uncensored"
    if any(t in tags for t in ["code", "coding", "coder"]):
        return "code"
    if "roleplay" in tags:
        return "roleplay"
    if "agent" in tags or "agentic" in tags:
        return "agent"
    if any(t in tags for t in ["reasoning", "math", "thinking"]):
        return "reasoning"

    # fall back to pipeline tag, but make it readable
    type_map = {
        "text-generation": "chat",
        "image-text-to-text": "chat",
        "text-to-image": "image-gen",
        "image-to-video": "video",
        "text-to-speech": "tts",
    }
    return type_map.get(pipeline, "")


def discover_gguf_models(available_gb, forced_quant=None, limit=20):
    # fetching trending models. api, please don't rate limit me.
    # query trending gguf models - sorted by likes this week
    url = (
        "https://huggingface.co/api/models"
        "?filter=gguf"
        "&sort=likes7d"
        f"&limit={limit * 3}"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gguf-fit/2.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            models = json.loads(resp.read().decode())
    except Exception as e:
        return [], str(e)

    results = []
    seen = set()

    for model in models:
        model_id = model.get("id", "")
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)

        likes = model.get("likes", 0)
        downloads = model.get("downloads", 0)
        # get pipeline_tag and tags from search results
        pipeline_tag = model.get("pipeline_tag", "")
        model_tags = model.get("tags", [])

        # get file listing to find ALL GGUF files for this model
        try:
            files_url = f"https://huggingface.co/api/models/{model_id}?blobs=true"
            req2 = urllib.request.Request(files_url, headers={"User-Agent": "gguf-fit/2.0"})
            with urllib.request.urlopen(req2, timeout=30) as resp2:
                model_info = json.loads(resp2.read().decode())
        except Exception:
            continue

        siblings = model_info.get("siblings", [])
        gguf_files = []
        for f in siblings:
            fname = f.get("rfilename", "")
            size_bytes = f.get("size", 0)
            if not fname.lower().endswith(".gguf"):
                continue
            if "mmproj" in fname.lower():
                continue

            size_gb = size_bytes / (1024**3)
            # skip zero-size files - they're likely API failures or untracked
            if size_gb <= 0:
                continue

            # skip if size > 120GB (sanity check - no GGUF should be >120GB)
            if size_gb > 120:
                continue

            # determine quant level from filename
            quant = infer_quant_from_filename(fname)
            if quant:
                # if user forced a specific quant, filter to only that one
                if forced_quant and forced_quant.upper() != quant.upper():
                    continue
                gguf_files.append({
                    "file": fname,
                    "size_gb": round(size_gb, 2),
                    "quant": quant,
                })

        if not gguf_files:
            continue

        if forced_quant:
            # user forced a quant - just pick the first matching one that fits
            candidates = [f for f in gguf_files if f["size_gb"] <= available_gb]
            if not candidates:
                continue
            best = candidates[0]
        else:
            # find the best quant that fits in available memory
            candidates = []
            for f in gguf_files:
                if f["size_gb"] <= available_gb:
                    # get quality rank (lower = better)
                    quality_rank = QUANT_QUALITY_ORDER.index(f["quant"]) if f["quant"] in QUANT_QUALITY_ORDER else 99
                    candidates.append({
                        **f,
                        "quality_rank": quality_rank,
                    })

            if not candidates:
                continue

            # pick highest quality (lowest rank number)
            best = min(candidates, key=lambda x: x["quality_rank"])

        # store full model data for get_useful_type
        model_data = {
            "tags": model_tags,
            "pipeline_tag": pipeline_tag,
        }
        type_label = get_useful_type(model_data)

        results.append({
            "model": model_id,
            "file": best["file"],
            "size_gb": best["size_gb"],
            "likes": likes,
            "downloads": downloads,
            "tags": model_tags,
            "pipeline_tag": pipeline_tag,
            "type": type_label,
            "quant": best["quant"],
        })

        if len(results) >= limit:
            break

    # sort by weekly likes (trending) as primary signal
    results.sort(key=lambda x: x["likes"], reverse=True)
    # final filter to ensure no zeros slip through
    results = [r for r in results if r["size_gb"] > 0]
    return results, None


def get_recommendations(memory_gb, overhead_gb=2.0, forced_quant=None, limit=10):
    # returns models that won't crash. hopefully.
    available = memory_gb - overhead_gb
    if available <= 0:
        return [], "not enough memory"

    candidates, err = discover_gguf_models(available, forced_quant, limit * 2)
    if err:
        return [], err

    # filter by actual file size - exclude tiny/zero files (likely LFS pointers)
    fits = [m for m in candidates if m["size_gb"] >= 0.1 and m["size_gb"] <= available]

    # deduplicate by model repo - keep the best quant per repo (already done in discover)
    seen = {}
    for m in fits:
        if m["model"] not in seen:
            seen[m["model"]] = m
    fits = list(seen.values())

    return fits[:limit], None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def fmt(b):
    if b <= 0:
        return "—"
    if b < 10:
        return f"~{b:.1f}B"
    return f"~{b:.0f}B"


def print_report(sources, quants, oh_fn):
    COL = 16

    # simplified quants - just show top 3 instead of all
    key_quants = ["Q4_K_M", "Q5_K_M", "Q6_K"]

    print()
    print("  gguf-fit — What GGUF models fit your hardware?")
    print("  " + "━" * 50)
    print()

    # ── Hardware summary ──
    for s in sources:
        label = s["tag"] + ":"
        if s["kind"] in ("nvidia", "amd", "apple"):
            note = " (unified)" if s["kind"] == "apple" else ""
            print(f"  {label:<12} {s['name']} · {s['mem']:.1f} GB{note}")
        elif s["tag"] == "Multi-GPU":
            print(f"  {label:<12} {s['name']} · {s['mem']:.1f} GB total")
        else:
            print(f"  {label:<12} {s['mem']:.1f} GB")
    print()

    # ── Simplified Table ──
    hdr = f"  {'Quant':<10}"
    for s in sources:
        col = f"{s['tag']} ({s['mem']:.0f}GB)"
        hdr += f"{col:>{COL}}"
    print(hdr)
    print("  " + "─" * (10 + COL * len(sources)))

    for q in key_quants:
        row = f"  {q:<10}"
        for s in sources:
            oh = oh_fn(s["kind"], s["mem"])
            mp = max_billions(s["mem"], q, oh)
            cell = fmt(mp)
            if q == "Q4_K_M" and mp > 0:
                cell += " ★"
            row += f"{cell:>{COL}}"
        print(row)

    print()
    print("  ★ = recommended (Q4_K_M)")
    print()

    # ── Links ──
    print("  ─── Hugging Face Links " + "─" * 28)
    print()
    for s in sources:
        oh = oh_fn(s["kind"], s["mem"])
        mp = max_billions(s["mem"], "Q4_K_M", oh)
        if mp < 0.5:
            print(f"  {s['tag']}: too little memory for Q4_K_M")
            print()
            continue
        print(f"  {s['tag']} · Q4_K_M (up to {fmt(mp)}):")
        print(f"    {make_url(mp)}")
        print()

    # Tip
    print("  Tip: use --context N to set context length (adds ~0.25GB per 1K tokens).")
    print("       use --overhead GB for custom overhead.")
    print("       use --all-quants to see every quantization level.")
    print()


def build_sources(ram, gpus, cpu_only=False):
    # build the list of memory sources to evaluate. 
    # this logic is confusing even to me.
    sources = []

    if not cpu_only and gpus:
        # Get main GPU (first discrete GPU)
        discrete = [g for g in gpus if g["kind"] in ("nvidia", "amd")]
        if discrete:
            main_gpu = discrete[0]
            # VRAM column - single main GPU
            sources.append(
                dict(tag="VRAM", name=main_gpu["name"], mem=main_gpu["vram_gb"], kind=main_gpu["kind"])
            )
            # Hybrid column - VRAM + RAM
            if ram:
                hybrid_mem = main_gpu["vram_gb"] + ram
                sources.append(
                    dict(tag="Hybrid", name=f"VRAM+RAM", mem=hybrid_mem, kind="hybrid")
                )

    # Show RAM column unless Apple unified (which IS the RAM)
    has_apple = any(g["kind"] == "apple" for g in gpus) if gpus else False
    if ram and (cpu_only or not has_apple):
        sources.append(dict(tag="RAM", name="System RAM", mem=ram, kind="ram"))

    # Fallback
    if not sources and ram:
        sources.append(dict(tag="RAM", name="System RAM", mem=ram, kind="ram"))

    return sources


def json_report(sources, quants, oh_fn, ram, gpus):
    result = {
        "hardware": {
            "ram_gb": round(ram, 1) if ram else None,
            "gpus": gpus,
        },
        "sources": [],
    }
    for s in sources:
        entry = {
            "label": s["tag"],
            "memory_gb": s["mem"],
            "overhead_gb": round(oh_fn(s["kind"], s["mem"]), 1),
            "quants": {},
        }
        for q in quants:
            oh = oh_fn(s["kind"], s["mem"])
            mp = max_billions(s["mem"], q, oh)
            if mp > 0.5:
                entry["quants"][q] = {
                    "max_billion_params": round(mp, 1),
                    "url": make_url(mp),
                }
        result["sources"].append(entry)
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# mode selection (ram/vram/hybrid)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def find_best_fit_for_mode(gguf_files, vram_gb, vram_overhead, ram_gb, ram_overhead, mode):
    # find best quantization for a given mode.
    # mode: 'ram', 'vram', or 'hybrid'.
    # returns (file, quant, size_gb, max_params_billions) or (none, none, 0, 0)
    # i wrote this at 2am. don't judge.
    if mode == "ram":
        avail = ram_gb - ram_overhead
        if avail <= 0:
            return None, None, 0, 0
        f, quant, size_gb = find_best_fit(gguf_files, ram_gb, ram_overhead)
        if f:
            # calculate max params
            bpw = QUANTS.get(quant, 4.5)
            max_b = (avail) * 8.0 / bpw
            return f, quant, size_gb, max_b
        return None, None, 0, 0

    elif mode == "vram":
        avail = vram_gb - vram_overhead
        if avail <= 0:
            return None, None, 0, 0
        f, quant, size_gb = find_best_fit(gguf_files, vram_gb, vram_overhead)
        if f:
            bpw = QUANTS.get(quant, 4.5)
            max_b = (avail) * 8.0 / bpw
            return f, quant, size_gb, max_b
        return None, None, 0, 0

    elif mode == "hybrid":
        # hybrid: use VRAM for compute + RAM for KV cache / context
        # we can fit larger models since KV cache can go to RAM
        # assume 20% of model must be in VRAM (minimum for GPU compute)
        vram_avail = vram_gb - vram_overhead
        if vram_avail <= 0:
            return None, None, 0, 0
        # find best quant that fits in combined memory
        combined_avail = (vram_gb - vram_overhead) + (ram_gb - ram_overhead)
        if combined_avail <= 0:
            return None, None, 0, 0

        f, quant, size_gb = find_best_fit(gguf_files, vram_gb + ram_gb, vram_overhead + ram_overhead)
        if f and size_gb <= combined_avail:
            # check that at least 20% fits in VRAM
            vram_needed = size_gb * 0.2
            if vram_needed <= vram_avail:
                bpw = QUANTS.get(quant, 4.5)
                max_b = combined_avail * 8.0 / bpw
                return f, quant, size_gb, max_b
        return None, None, 0, 0

    return None, None, 0, 0


def recommend_mode(vram_gb, vram_overhead, ram_gb, ram_overhead, gguf_files):
    # determine recommended mode and quant.
    # 1. check if any 4-bit quant fits in vram
    # 2. if vram fails, check hybrid
    # 3. fall back to ram-only
    # returns (mode, file, quant, size_gb, max_b)
    # this algorithm is probably wrong but it sounds reasonable.
    # try vram first
    vram_file, vram_quant, vram_size, vram_max = find_best_fit_for_mode(
        gguf_files, vram_gb, vram_overhead, ram_gb, ram_overhead, "vram"
    )
    if vram_file:
        return "VRAM", vram_file, vram_quant, vram_size, vram_max

    # try hybrid
    hybrid_file, hybrid_quant, hybrid_size, hybrid_max = find_best_fit_for_mode(
        gguf_files, vram_gb, vram_overhead, ram_gb, ram_overhead, "hybrid"
    )
    if hybrid_file:
        return "Hybrid", hybrid_file, hybrid_quant, hybrid_size, hybrid_max

    # fall back to ram
    ram_file, ram_quant, ram_size, ram_max = find_best_fit_for_mode(
        gguf_files, vram_gb, vram_overhead, ram_gb, ram_overhead, "ram"
    )
    if ram_file:
        return "RAM", ram_file, ram_quant, ram_size, ram_max

    return None, None, None, 0, 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    ap = argparse.ArgumentParser(
        prog="gguf-fit",
        description="Detect hardware & find GGUF models that fit.",
    )
    ap.add_argument(
        "model",
        nargs="?",
        help="Hugging Face model ID (e.g. HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive)",
    )
    ap.add_argument(
        "--cpu", action="store_true", help="only consider system RAM (ignore GPUs)"
    )
    ap.add_argument(
        "--all-quants", action="store_true", help="show all quantization types"
    )
    ap.add_argument(
        "--overhead",
        type=float,
        default=None,
        metavar="GB",
        help="override auto-detected memory overhead (GB)",
    )
    ap.add_argument(
        "--context",
        "--ctx",
        type=int,
        default=None,
        metavar="N",
        help="set context length (adds overhead: ~0.25GB per 1K tokens)",
    )
    ap.add_argument("--json", action="store_true", help="output JSON")
    ap.add_argument(
        "--recommend",
        action="store_true",
        help="show trending GGUF models that fit your hardware",
    )
    ap.add_argument(
        "--quant",
        default=None,
        help="force a specific quantization level (default: auto-select best fit)",
    )
    ap.add_argument(
        "--ram",
        action="store_true",
        help="also show recommendations for system RAM (for CPU inference)",
    )
    ap.add_argument(
        "--ram-only",
        dest="ram_only",
        action="store_true",
        help="Only show RAM-only inference options",
    )
    ap.add_argument(
        "--vram-only",
        dest="vram_only",
        action="store_true",
        help="Only show VRAM-only (GPU) inference options",
    )
    ap.add_argument(
        "--hybrid",
        dest="hybrid_only",
        action="store_true",
        help="Only show Hybrid (VRAM+RAM combined) inference options",
    )
    ap.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    args = ap.parse_args()

    quants = list(QUANTS.keys()) if args.all_quants else DEFAULT_QUANTS

    # ── Detect ──
    ram = detect_ram()
    gpus = [] if args.cpu else detect_gpus()

    if ram is None and not gpus:
        print("Error: could not detect system memory.", file=sys.stderr)
        sys.exit(1)

    sources = build_sources(ram, gpus, cpu_only=args.cpu)

    if not sources:
        print("Error: no memory sources found.", file=sys.stderr)
        sys.exit(1)

    # ── Overhead function ──
    context_len = args.context
    if args.overhead is not None:
        fixed = args.overhead
        oh_fn = lambda kind, mem: fixed
    elif context_len is not None:
        # each 1K tokens adds ~0.25GB for GPU, ~0.5GB for CPU
        ctx_overhead = (context_len / 1024) * 0.25
        def oh_fn(kind, mem):
            base = default_overhead(kind, mem)
            if kind == "apple":
                return base + (context_len / 1024) * 0.3  # metal is a bit more
            elif kind in ("nvidia", "amd"):
                return base + (context_len / 1024) * 0.25
            elif kind == "hybrid":
                return base + (context_len / 1024) * 0.35  # both VRAM and RAM
            else:
                return base + (context_len / 1024) * 0.5  # CPU/RAM
    else:
        oh_fn = default_overhead

    # ── Model-specific output ──
    if args.model:
        repo_id, requested_quant = parse_model_id(args.model)

        # fetch GGUF files
        gguf_files, err = fetch_gguf_files(repo_id)
        if err:
            print(f"Error: {err}", file=sys.stderr)
            sys.exit(1)

        # determine which mode filter was requested
        mode_filter = None
        if args.ram_only:
            mode_filter = "ram"
        elif args.vram_only:
            mode_filter = "vram"
        elif args.hybrid_only:
            mode_filter = "hybrid"

        # get hardware info
        ram_gb = ram if ram else 0
        vram_gb = 0
        vram_overhead = 2.0
        ram_overhead = 3.0

        # find primary GPU (first one or highest VRAM)
        gpu_source = None
        for s in sources:
            if s["tag"] == "VRAM":
                gpu_source = s
                break
            if s["tag"].startswith("GPU") or s["tag"] == "Metal":
                if not gpu_source or s["mem"] > gpu_source["mem"]:
                    gpu_source = s

        if gpu_source:
            vram_gb = gpu_source["mem"]
            vram_overhead = oh_fn(gpu_source["kind"], gpu_source["mem"])
            # add runtime overhead for GPU (KV cache, compute buffers, etc.)
            if gpu_source["kind"] in ("nvidia", "amd"):
                vram_overhead += RUNTIME_OVERHEAD_GB

        # get all three options
        options = []

        # RAM-only option
        if ram_gb > 0:
            ram_file, ram_quant, ram_size, ram_max = find_best_fit_for_mode(
                gguf_files, vram_gb, vram_overhead, ram_gb, ram_overhead, "ram"
            )
            if ram_file:
                options.append({
                    "mode": "RAM",
                    "file": ram_file,
                    "quant": ram_quant,
                    "size": ram_size,
                    "max_b": ram_max,
                })

        # VRAM-only option
        if vram_gb > 0:
            vram_file, vram_quant, vram_size, vram_max = find_best_fit_for_mode(
                gguf_files, vram_gb, vram_overhead, ram_gb, ram_overhead, "vram"
            )
            if vram_file:
                options.append({
                    "mode": "VRAM",
                    "file": vram_file,
                    "quant": vram_quant,
                    "size": vram_size,
                    "max_b": vram_max,
                })

        # Hybrid option
        if vram_gb > 0 and ram_gb > 0:
            hybrid_file, hybrid_quant, hybrid_size, hybrid_max = find_best_fit_for_mode(
                gguf_files, vram_gb, vram_overhead, ram_gb, ram_overhead, "hybrid"
            )
            if hybrid_file:
                options.append({
                    "mode": "Hybrid",
                    "file": hybrid_file,
                    "quant": hybrid_quant,
                    "size": hybrid_size,
                    "max_b": hybrid_max,
                })

        # determine recommended mode (prioritize VRAM > Hybrid > RAM)
        rec_mode = None
        if mode_filter:
            # user selected specific mode, only show that
            filtered = [o for o in options if o["mode"].lower() == mode_filter.lower()]
            if filtered:
                rec_mode = filtered[0]["mode"]
                options = filtered
        else:
            # auto-recommend based on algorithm
            rec_mode, _, _, _, _ = recommend_mode(vram_gb, vram_overhead, ram_gb, ram_overhead, gguf_files)

        if not options:
            print(f"Error: no suitable quantization found for this model on your hardware", file=sys.stderr)
            sys.exit(1)

        print()
        hf_link = f"https://huggingface.co/{repo_id}"
        # show each option
        for opt in options:
            print(f"  {opt['mode']}   {opt['quant']} ({opt['size']:.2f} GB)")

        # show recommendation
        print()
        if rec_mode:
            rec_opt = next((o for o in options if o["mode"] == rec_mode), None)
            if rec_opt:
                if rec_opt["mode"] == "VRAM":
                    reason = f"{rec_opt['quant']} fits in your {vram_gb:.0f}GB GPU"
                elif rec_opt["mode"] == "Hybrid":
                    reason = f"{rec_opt['quant']} uses your {vram_gb:.0f}GB GPU + {ram_gb:.0f}GB RAM"
                else:
                    reason = f"{rec_opt['quant']} fits in your {ram_gb:.0f}GB RAM"
                print(f"  Recommended: {rec_opt['mode']} ({reason})")
                print(f"  {hf_link}")
        print()

        # figure out which quant to use for the server command
        serve_quant = None
        if rec_mode:
            rec_opt = next((o for o in options if o["mode"] == rec_mode), None)
            if rec_opt:
                serve_quant = rec_opt["quant"]
        if not serve_quant:
            serve_quant = options[0]["quant"]

        # build the llama-server command
        hf_tag = f"{repo_id}:{serve_quant}"
        server_cmd = ["llama-server", "-hf", hf_tag]

        print("  ─── Server Command " + "─" * 33)
        print()
        print(f"  llama-server -hf {hf_tag}")
        print()

        # only prompt if stdout is a terminal and not in json mode
        if not sys.stdout.isatty():
            return

        # check if llama-server is even installed before bothering the user
        if not shutil.which("llama-server"):
            print("  (llama-server not found in PATH — install llama.cpp to run)")
            print()
            return

        try:
            answer = input("  Start local server with web UI? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if answer not in ("y", "yes"):
            return

        # hand off to llama-server. goodbye, python. you will not be missed.
        print()
        print(f"  Starting: {' '.join(server_cmd)}")
        print()
        os.execvp("llama-server", server_cmd)
        return

    # ── Recommendations ──
    if args.recommend:
        primary = sources[0]
        overhead = oh_fn(primary["kind"], primary["mem"])

        # add runtime overhead for GPU (KV cache, compute buffers, etc.)
        # this ensures the recommendation fits entirely in VRAM, not just the model weights
        if primary["kind"] in ("nvidia", "amd"):
            overhead += RUNTIME_OVERHEAD_GB

        # if user specified --quant, use it; otherwise auto-select best quant
        forced_quant = args.quant

        recs, err = get_recommendations(primary["mem"], overhead, forced_quant)
        if err:
            print(f"Error: {err}", file=sys.stderr)
            sys.exit(1)

        # get RAM recommendations if requested
        ram_recs = None
        show_ram = args.ram
        if args.ram:
            ram_source = next((s for s in sources if s["tag"] == "RAM"), None)
            if ram_source:
                ram_overhead = oh_fn(ram_source["kind"], ram_source["mem"])
                ram_recs, ram_err = get_recommendations(ram_source["mem"], ram_overhead, forced_quant)
                if ram_err:
                    ram_recs = None

        # output in new format: model name + HF link + which quant fits where
        print()
        print("  gguf-fit — Trending models that fit your hardware")
        print("  " + "━" * 50)
        print()

        for rec in recs:
            print(f"  {rec['model']}")
            print(f"    {rec['quant']} ({rec['size_gb']:.2f} GB)")
            print(f"    https://huggingface.co/{rec['model']}")
            print()

        if show_ram and ram_recs:
            print("  ─── RAM-Only Options " + "─" * 32)
            print()
            for rec in ram_recs:
                print(f"  {rec['model']}")
                print(f"    {rec['quant']} ({rec['size_gb']:.2f} GB)")
                print(f"    https://huggingface.co/{rec['model']}")
                print()

        return

    # ── Output ──
    if args.json:
        data = json_report(sources, quants, oh_fn, ram, gpus)
        json.dump(data, sys.stdout, indent=2)
        print()
    else:
        print_report(sources, quants, oh_fn)


if __name__ == "__main__":
    main()
