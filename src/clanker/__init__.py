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
import pathlib

__version__ = "4.0.0"

# bits per weight. i guess?

QUANTS = {
    # Legacy base types
    "Q4_0": 4.34,
    "Q4_1": 4.78,
    "Q5_0": 5.21,
    "Q5_1": 5.65,
    # K-quants
    "Q2_K": 2.96,
    "Q2_K_S": 2.96,
    "Q2_K_P": 3.5,
    "Q3_K_S": 3.41,
    "Q3_K_M": 3.74,
    "Q3_K_L": 4.03,
    "Q3_K_P": 4.1,
    "Q4_K_S": 4.59,
    "Q4_K_M": 4.85,
    "Q4_K_P": 5.2,
    "Q5_K_S": 5.54,
    "Q5_K_M": 5.69,
    "Q5_K_P": 6.1,
    "Q6_K": 6.57,
    "Q6_K_P": 7.0,
    "Q8_0": 8.50,
    "Q8_K_P": 9.4,
    # IQ series
    "IQ1_S": 1.56,
    "IQ1_M": 1.75,
    "IQ2_XXS": 2.06,
    "IQ2_XS": 2.31,
    "IQ2_S": 2.50,
    "IQ2_M": 2.70,
    "IQ3_XXS": 3.06,
    "IQ3_XS": 3.30,
    "IQ3_S": 3.44,
    "IQ3_M": 3.66,
    "IQ4_XS": 3.85,
    "IQ4_NL": 4.50,
    # Ternary (new)
    "TQ1_0": 1.69,
    "TQ2_0": 2.06,
    # Full precision
    "BF16": 16.00,
    "F16": 16.00,
    "F32": 32.00,
}

DEFAULT_QUANTS = ["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
RECOMMENDED = "Q4_K_M"

# quality ranking: higher index = lower quality
QUANT_QUALITY_ORDER = [
    "F32",
    "BF16",
    "F16",
    "Q8_K_P",
    "Q8_0",
    "Q6_K_P",
    "Q6_K",
    "Q5_K_P",
    "Q5_K_M",
    "Q5_K_S",
    "Q5_1",
    "Q5_0",
    "Q4_K_P",
    "Q4_K_M",
    "Q4_K_S",
    "Q4_1",
    "Q4_0",
    "IQ4_NL",
    "IQ4_XS",
    "Q3_K_P",
    "Q3_K_L",
    "Q3_K_M",
    "Q3_K_S",
    "IQ3_M",
    "IQ3_S",
    "IQ3_XS",
    "IQ3_XXS",
    "Q2_K_P",
    "Q2_K",
    "Q2_K_S",
    "IQ2_M",
    "IQ2_S",
    "IQ2_XS",
    "IQ2_XXS",
    "TQ2_0",
    "IQ1_M",
    "TQ1_0",
    "IQ1_S",
]

# UD aliases — for ranking, map to nearest standard quant
UD_ALIASES = {
    # strip prefix, map to nearest standard quant for ranking
    "UD-IQ1_S": "IQ1_S",
    "UD-IQ1_M": "IQ1_M",
    "UD-IQ2_XXS": "IQ2_XXS",
    "UD-IQ2_XS": "IQ2_XS",
    "UD-IQ2_S": "IQ2_S",
    "UD-IQ2_M": "IQ2_M",
    "UD-IQ3_XXS": "IQ3_XXS",
    "UD-IQ3_XS": "IQ3_XS",
    "UD-IQ3_S": "IQ3_S",
    "UD-IQ3_M": "IQ3_M",
    "UD-IQ4_XS": "IQ4_XS",
    "UD-IQ4_NL": "IQ4_NL",
    "UD-Q2_K_XL": "Q2_K",
    "UD-Q3_K_XL": "Q3_K_M",  # XL = higher quality embeddings
    "UD-Q4_K_XL": "Q4_K_M",
    "UD-Q5_K_XL": "Q5_K_M",
    "UD-Q6_K_XL": "Q6_K",
    "UD-Q8_0_XL": "Q8_0",
}


def resolve_quant(quant_key):
    """Returns the base quant for a UD key, or the key itself."""
    return UD_ALIASES.get(quant_key, quant_key)


def get_bpw(quant_key):
    """Get bpw for a quant key, resolving UD aliases."""
    base = resolve_quant(quant_key)
    return QUANTS.get(base, None)


def is_dynamic_quant(quant_key):
    return quant_key in UD_ALIASES


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
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return int(o.strip()) / (1024**3)
        elif s == "Windows":
            o = subprocess.check_output(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
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
                text=True,
                stderr=subprocess.DEVNULL,
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
                text=True,
                stderr=subprocess.DEVNULL,
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
                    text=True,
                    stderr=subprocess.DEVNULL,
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
RUNTIME_OVERHEAD_GB = (
    1.5  # accounts for KV cache (~800MB) + compute buffers (~500MB) + misc
)


def max_billions(mem_gb, quant, overhead_gb):
    # max model size (billion params) that fits at given quantization.
    # math. unfortunately necessary.
    bpw = get_bpw(quant)
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


def fetch_model_max_context(repo_id):
    # try to get the model's context length from huggingface. why not.
    # first: api call
    try:
        api_url = f"https://huggingface.co/api/models/{repo_id}"
        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": f"clanker/{__version__}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        cfg = data.get("config", {})
        for key in [
            "max_position_embeddings",
            "n_ctx",
            "context_length",
            "max_seq_len",
            "model_max_length",
        ]:
            if key in cfg:
                try:
                    return int(cfg[key])
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass
    # api failed. try raw config.json i guess.
    try:
        cfg_url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
        req = urllib.request.Request(
            cfg_url, headers={"User-Agent": f"clanker/{__version__}"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            cfg = json.loads(resp.read().decode())
        for key in [
            "max_position_embeddings",
            "n_ctx",
            "context_length",
            "max_seq_len",
            "model_max_length",
        ]:
            if key in cfg:
                try:
                    return int(cfg[key])
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass  # whatever, give up
    return None


def compute_max_context(mode, model_size_gb, vram_gb, ram_gb, gpu_kind=None):
    # how many tokens can we even fit. rough math.
    if mode == "VRAM":
        if gpu_kind is None:
            return None
        available = vram_gb
        base = default_overhead(gpu_kind, vram_gb)
        if gpu_kind in ("nvidia", "amd"):
            base += RUNTIME_OVERHEAD_GB
        factor = 0.25 if gpu_kind in ("nvidia", "amd") else 0.30  # gpu
    elif mode == "RAM":
        available = ram_gb
        base = 3.0  # cpu overhead
        factor = 0.5
    elif mode == "Hybrid":
        if gpu_kind is None:
            return None
        available = vram_gb + ram_gb
        vram_base = default_overhead(gpu_kind, vram_gb)
        if gpu_kind in ("nvidia", "amd"):
            vram_base += RUNTIME_OVERHEAD_GB
        ram_base = 3.0
        base = vram_base + ram_base
        factor = 0.35  # split memory
    else:
        return None

    remaining = available - base - model_size_gb
    if remaining <= 0:
        return 0  # no room for context at all
    max_tokens = int((remaining / factor) * 1024)
    return max_tokens


def fetch_gguf_files(repo_id):
    # fetch gguf file info from huggingface. api, please work.
    api_url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    try:
        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": f"clanker/{__version__}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            tree_nodes = json.loads(resp.read().decode())
    except Exception as e:
        return None, str(e)

    gguf_files = []
    for node in tree_nodes:
        if node.get("type") != "file":
            continue
        path = node.get("path", "")
        if not path.lower().endswith(".gguf"):
            continue
        # skip mmproj files
        if "mmproj" in path.lower():
            continue
        size_bytes = item.get("size", 0)
        size_gb = size_bytes / (1024**3)
        gguf_files.append(
            {
                "name": path,
                "size_gb": round(size_gb, 2),
            }
        )

    if not gguf_files:
        return None, "no GGUF files found in repository"

    return gguf_files, None


def infer_quant_from_filename(filename):
    name = os.path.basename(filename).upper()
    name_noext = name.rsplit(".GGUF", 1)[0]

    # Check UD aliases first (most specific)
    for ud_key, base_quant in UD_ALIASES.items():
        if ud_key in name_noext:
            return ud_key  # return the UD key, caller resolves via UD_ALIASES

    # Standard quants — longest match first to avoid Q4 matching Q4_K_M
    for q in sorted(QUANTS.keys(), key=len, reverse=True):
        if q in name_noext:
            return q

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
    suitable.sort(key=lambda x: get_bpw(x[1]) or 0, reverse=True)
    best = suitable[0]
    return best[0], best[1], best[0]["size_gb"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# model discovery. api, i'm begging you.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


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
    print("  clanker — What GGUF models fit your hardware?")
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
                dict(
                    tag="VRAM",
                    name=main_gpu["name"],
                    mem=main_gpu["vram_gb"],
                    kind=main_gpu["kind"],
                )
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
    report_body = {
        "hardware": {
            "ram_gb": round(ram, 1) if ram else None,
            "gpus": gpus,
        },
        "sources": [],
    }
    for s in sources:
        source_info = {
            "label": s["tag"],
            "memory_gb": s["mem"],
            "overhead_gb": round(oh_fn(s["kind"], s["mem"]), 1),
            "quants": {},
        }
        for q in quants:
            oh = oh_fn(s["kind"], s["mem"])
            mp = max_billions(s["mem"], q, oh)
            if mp > 0.5:
                source_info["quants"][q] = {
                    "max_billion_params": round(mp, 1),
                    "url": make_url(mp),
                }
        report_body["sources"].append(source_info)
    return report_body


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# mode selection (ram/vram/hybrid)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def find_best_fit_for_mode(
    gguf_files, vram_gb, vram_overhead, ram_gb, ram_overhead, mode
):
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

        f, quant, size_gb = find_best_fit(
            gguf_files, vram_gb + ram_gb, vram_overhead + ram_overhead
        )
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
# Local storage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_model_dir():
    """Get the directory where models are stored."""
    return pathlib.Path.home() / ".cache" / "huggingface" / "hub"


def get_metadata_file():
    """Get the path to the metadata file."""
    return pathlib.Path.home() / ".clanker" / "metadata.json"


def load_metadata():
    """Load metadata from JSON file."""
    mf = get_metadata_file()
    if mf.exists():
        with open(mf) as f:
            return json.load(f)
    return {}


def save_metadata(metadata):
    """Save metadata to JSON file."""
    mf = get_metadata_file()
    mf.parent.mkdir(parents=True, exist_ok=True)
    with open(mf, "w") as f:
        json.dump(metadata, f, indent=2)


def list_local_models():
    model_dir = get_model_dir()
    if not model_dir.exists():
        return []
    models = []
    for file in model_dir.rglob("*.gguf"):
        # this is dumb but it works
        models.append(str(file.relative_to(model_dir)))
    return models


def handle_ls(args):
    model_dir = get_model_dir()
    if not model_dir.exists():
        print("No local models found.")
        return
    found = False
    for file in model_dir.rglob("*.gguf"):
        if "mmproj" in file.name.lower():
            continue  # hate these mmproj things
        found = True
        relative = file.relative_to(model_dir)
        parts = str(relative).split("/")
        if len(parts) >= 3 and parts[0].startswith("models--"):
            repo_parts = parts[0].split("--")[1:]
            repo = "/".join(repo_parts)
            filename = parts[-1]
            quant = infer_quant_from_filename(filename)
            # trying to make it less ugly
            repo_name = repo.split("/")[-1]
            if filename.startswith(repo_name):
                short_filename = filename[len(repo_name) :].lstrip("-")
            else:
                short_filename = filename
            if quant:
                print(f"{repo}/{short_filename} ({quant})")
            else:
                print(f"{repo}/{short_filename}")
        else:
            # whatever
            print(str(relative))
    if not found:
        print("No local models found.")


def handle_rm(args):
    model = args.model
    model_dir = get_model_dir()
    path = model_dir / model
    if path.exists():
        path.unlink()
        print(f"Removed {model}")
        # whatever, update this junk
        metadata = load_metadata()
        if model in metadata:
            del metadata[model]
            save_metadata(metadata)
    else:
        print(f"Model {model} not found", file=sys.stderr)


def handle_info(args):
    model = args.model
    metadata = load_metadata()
    if model in metadata:
        info = metadata[model]
        print(f"Model: {model}")
        print(f"Size: {info.get('size_gb', 'unknown')} GB")
        print(f"Quant: {info.get('quant', 'unknown')}")
    else:
        print(f"No metadata for {model}")


def handle_cp(args):
    src = args.src
    dest = args.dest
    model_dir = get_model_dir()
    src_path = model_dir / src
    dest_path = model_dir / dest
    if src_path.exists():
        shutil.copy2(src_path, dest_path)
        print(f"Copied {src} to {dest}")
    else:
        print(f"Source {src} not found", file=sys.stderr)


def handle_mv(args):
    src = args.src
    dest = args.dest
    model_dir = get_model_dir()
    src_path = model_dir / src
    dest_path = model_dir / dest
    if src_path.exists():
        src_path.rename(dest_path)
        print(f"Moved {src} to {dest}")
        # update metadata
        metadata = load_metadata()
        if src in metadata:
            metadata[dest] = metadata[src]
            del metadata[src]
            save_metadata(metadata)
    else:
        print(f"Source {src} not found", file=sys.stderr)


def handle_du(args):
    model_dir = get_model_dir()
    total_size = 0
    for file in model_dir.rglob("*.gguf"):
        total_size += file.stat().st_size
    print(f"Total disk usage: {total_size / (1024**3):.2f} GB")


def handle_download(args):
    import huggingface_hub

    repo_id = args.model
    quant = args.quant
    gguf_files, err = fetch_gguf_files(repo_id)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        return
    if not gguf_files:
        print("No GGUF files found", file=sys.stderr)
        return
    if quant:
        file = next((f for f in gguf_files if quant in f["name"]), None)
        if not file:
            print(f"No file with quant {quant} found", file=sys.stderr)
            return
    else:
        file = gguf_files[0]  # download first
    filename = file["name"]
    try:
        local_path = huggingface_hub.hf_hub_download(repo_id, filename)
        print(f"Downloaded to {local_path}")
        # update metadata
        metadata = load_metadata()
        key = repo_id + "/" + filename  # or just filename
        metadata[filename] = {
            "size_gb": file["size_gb"],
            "quant": infer_quant_from_filename(filename),
        }
        save_metadata(metadata)
    except Exception as e:
        print(f"Error downloading: {e}", file=sys.stderr)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    # Check for help
    if len(sys.argv) > 1 and sys.argv[1] in ("help", "--help", "-h"):
        ap = argparse.ArgumentParser(
            prog="clanker",
            description="Detect hardware & find GGUF models that fit.",
        )
        # Add subparsers for help display
        subparsers = ap.add_subparsers(dest="subcommand", help="Available commands")
        ls_parser = subparsers.add_parser("ls", help="List local models or HF models")
        download_parser = subparsers.add_parser("download", help="Download a model")
        rm_parser = subparsers.add_parser("rm", help="Remove local model")
        info_parser = subparsers.add_parser("info", help="Show model details")
        cp_parser = subparsers.add_parser("cp", help="Copy model")
        mv_parser = subparsers.add_parser("mv", help="Move/rename model")
        du_parser = subparsers.add_parser("du", help="Show disk usage")
        ap.add_argument(
            "--version", action="version", version=f"%(prog)s {__version__}"
        )
        # Add the old parser arguments for completeness
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
            "--quant",
            default=None,
            help="force a specific quantization level (default: auto-select best fit)",
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
        ap.print_help()
        return

    # Check for subcommands
    known_subs = ["ls", "download", "rm", "info", "cp", "mv", "du"]
    use_subcommand = len(sys.argv) > 1 and sys.argv[1] in known_subs

    if use_subcommand:
        # Use subcommand parser
        ap = argparse.ArgumentParser(
            prog="clanker",
            description="Detect hardware & find GGUF models that fit.",
        )
        subparsers = ap.add_subparsers(dest="subcommand", help="Available commands")

        # File system commands
        ls_parser = subparsers.add_parser("ls", help="List local models or HF models")
        ls_parser.add_argument(
            "path", nargs="?", default=".", help="Path to list (default: local models)"
        )

        download_parser = subparsers.add_parser("download", help="Download a model")
        download_parser.add_argument("model", help="Hugging Face model ID")
        download_parser.add_argument("quant", nargs="?", help="Quantization level")

        rm_parser = subparsers.add_parser("rm", help="Remove local model")
        rm_parser.add_argument("model", help="Model name or path")

        info_parser = subparsers.add_parser("info", help="Show model details")
        info_parser.add_argument("model", help="Model name or path")

        cp_parser = subparsers.add_parser("cp", help="Copy model")
        cp_parser.add_argument("src", help="Source model")
        cp_parser.add_argument("dest", help="Destination path")

        mv_parser = subparsers.add_parser("mv", help="Move/rename model")
        mv_parser.add_argument("src", help="Source model")
        mv_parser.add_argument("dest", help="Destination path")

        du_parser = subparsers.add_parser("du", help="Show disk usage")
        du_parser.add_argument("path", nargs="?", default=".", help="Path to check")

        ap.add_argument(
            "--version", action="version", version=f"%(prog)s {__version__}"
        )
        args = ap.parse_args()
    else:
        # Backward compatibility: old parser
        ap = argparse.ArgumentParser(
            prog="clanker",
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
            "--quant",
            default=None,
            help="force a specific quantization level (default: auto-select best fit)",
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
        args.subcommand = "check"

    if args.subcommand == "check":
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

        # ── Output ──
        if not args.model:
            if args.json:
                data = json_report(sources, quants, oh_fn, ram, gpus)
                json.dump(data, sys.stdout, indent=2)
                print()
            else:
                print_report(sources, quants, oh_fn)

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
                    options.append(
                        {
                            "mode": "RAM",
                            "file": ram_file,
                            "quant": ram_quant,
                            "size": ram_size,
                            "max_b": ram_max,
                        }
                    )

            # VRAM-only option
            if vram_gb > 0:
                vram_file, vram_quant, vram_size, vram_max = find_best_fit_for_mode(
                    gguf_files, vram_gb, vram_overhead, ram_gb, ram_overhead, "vram"
                )
                if vram_file:
                    options.append(
                        {
                            "mode": "VRAM",
                            "file": vram_file,
                            "quant": vram_quant,
                            "size": vram_size,
                            "max_b": vram_max,
                        }
                    )

            # Hybrid option
            if vram_gb > 0 and ram_gb > 0:
                hybrid_file, hybrid_quant, hybrid_size, hybrid_max = (
                    find_best_fit_for_mode(
                        gguf_files,
                        vram_gb,
                        vram_overhead,
                        ram_gb,
                        ram_overhead,
                        "hybrid",
                    )
                )
                if hybrid_file:
                    options.append(
                        {
                            "mode": "Hybrid",
                            "file": hybrid_file,
                            "quant": hybrid_quant,
                            "size": hybrid_size,
                            "max_b": hybrid_max,
                        }
                    )

            # determine recommended mode (prioritize VRAM > Hybrid > RAM)
            rec_mode = None
            if mode_filter:
                # user selected specific mode, only show that
                filtered = [
                    o for o in options if o["mode"].lower() == mode_filter.lower()
                ]
                if filtered:
                    rec_mode = filtered[0]["mode"]
                    options = filtered
            else:
                # auto-recommend based on algorithm
                rec_mode, _, _, _, _ = recommend_mode(
                    vram_gb, vram_overhead, ram_gb, ram_overhead, gguf_files
                )

            if not options:
                print(
                    f"Error: no suitable quantization found for this model on your hardware",
                    file=sys.stderr,
                )
                sys.exit(1)

            print()
            hf_link = f"https://huggingface.co/{repo_id}"
            # show each option
            for opt in options:
                dynamic_note = ", dynamic" if is_dynamic_quant(opt["quant"]) else ""
                print(
                    f"  {opt['mode']}   {opt['quant']} ({opt['size']:.2f} GB{dynamic_note})"
                )

            # show recommendation
            print()
            if rec_mode:
                rec_opt = next((o for o in options if o["mode"] == rec_mode), None)
                if rec_opt:
                    if is_dynamic_quant(rec_opt["quant"]):
                        ud_note = f" — Unsloth Dynamic, ~{resolve_quant(rec_opt['quant'])} quality"
                    else:
                        ud_note = ""
                    if rec_opt["mode"] == "VRAM":
                        reason = f"{rec_opt['quant']}{ud_note} fits in your {vram_gb:.0f}GB GPU"
                    elif rec_opt["mode"] == "Hybrid":
                        reason = f"{rec_opt['quant']}{ud_note} uses your {vram_gb:.0f}GB GPU + {ram_gb:.0f}GB RAM"
                    else:
                        reason = f"{rec_opt['quant']}{ud_note} fits in your {ram_gb:.0f}GB RAM"
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

            # only prompt if stdout is a terminal
            if not sys.stdout.isatty():
                # still check context and warn, but don't prompt
                requested_ctx = args.context
                native_ctx_val = None
                effective_ctx = None
                if requested_ctx is None:
                    native_ctx_val = fetch_model_max_context(repo_id)
                    if native_ctx_val is not None:
                        effective_ctx = native_ctx_val
                else:
                    effective_ctx = requested_ctx

                if effective_ctx is not None:
                    selected_opt = None
                    if rec_mode:
                        selected_opt = next(
                            (o for o in options if o["mode"] == rec_mode), None
                        )
                    else:
                        selected_opt = options[0] if options else None

                    if selected_opt:
                        selected_mode = selected_opt["mode"]
                        model_size_gb = selected_opt["size"]
                        gpu_kind = gpu_source["kind"] if gpu_source else None
                        max_ctx = compute_max_context(
                            selected_mode, model_size_gb, vram_gb, ram_gb, gpu_kind
                        )
                        if max_ctx is not None and effective_ctx > max_ctx:
                            if max_ctx < 1:
                                print(
                                    "Error: Insufficient memory to run this model with any context length.",
                                    file=sys.stderr,
                                )
                                sys.exit(1)
                            print(
                                f"Warning: Context {effective_ctx:,} exceeds hardware capacity ({max_ctx:,} tokens max).",
                                file=sys.stderr,
                            )
                return

            # check if llama-server is even installed before bothering the user
            if not shutil.which("llama-server"):
                print("  (llama-server not found in PATH — install llama.cpp to run)")
                print()
                return

            # --- context reduction check ---
            override_ctx = None
            requested_ctx = args.context
            native_ctx_val = None
            effective_ctx = None
            if requested_ctx is None:
                native_ctx_val = fetch_model_max_context(repo_id)
                if native_ctx_val is not None:
                    effective_ctx = native_ctx_val
            else:
                effective_ctx = requested_ctx

            if effective_ctx is not None:
                selected_opt = None
                if rec_mode:
                    selected_opt = next(
                        (o for o in options if o["mode"] == rec_mode), None
                    )
                else:
                    selected_opt = options[0] if options else None

                if selected_opt:
                    selected_mode = selected_opt["mode"]
                    model_size_gb = selected_opt["size"]
                    gpu_kind = gpu_source["kind"] if gpu_source else None
                    max_ctx = compute_max_context(
                        selected_mode, model_size_gb, vram_gb, ram_gb, gpu_kind
                    )
                    if max_ctx is not None and effective_ctx > max_ctx:
                        if max_ctx < 1:
                            print(
                                "  Error: Insufficient memory to run this model with any context length."
                            )
                            return
                        print()
                        print("  Context Warning")
                        if native_ctx_val is not None and requested_ctx is None:
                            print(
                                f"  This model supports {native_ctx_val:,} tokens, but your hardware"
                            )
                            print(f"  only fits {max_ctx:,} with the current settings.")
                        else:
                            if requested_ctx is not None:
                                print(
                                    f"  Requested context {requested_ctx:,} exceeds hardware capacity."
                                )
                            else:
                                print(
                                    f"  The model's default context exceeds hardware capacity."
                                )
                            print(f"  Maximum supported: {max_ctx:,} tokens.")
                        print()
                        suggestions = []
                        for cand in [4096, 8192, 16384, 32768, 65536, 131072, 262144]:
                            if cand <= max_ctx:
                                suggestions.append(cand)
                        suggestions = (
                            suggestions[-2:] if len(suggestions) > 2 else suggestions
                        )
                        if suggestions:
                            print("  To increase context at the cost of speed, try:")
                            for s in suggestions:
                                if s >= 1024:
                                    hr = f"~{s // 1024}K"
                                else:
                                    hr = str(s)
                                print(f"    --ctx {s}   ({hr})")
                            print()
                        try:
                            answer = (
                                input(f"  Continue with {max_ctx:,} tokens? [Y/n] ")
                                .strip()
                                .lower()
                            )
                        except (EOFError, KeyboardInterrupt):
                            print()
                            return
                        if answer not in ("y", "yes"):
                            return
                        override_ctx = max_ctx

            if override_ctx is not None:
                server_cmd.extend(["--ctx-size", str(override_ctx)])
                print()
                print(f"  Starting: {' '.join(server_cmd)}")
                print()
                os.execvp("llama-server", server_cmd)

            try:
                answer = (
                    input("  Start local server with web UI? [y/N] ").strip().lower()
                )
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

    if args.subcommand != "check":
        if args.subcommand == "ls":
            handle_ls(args)
        elif args.subcommand == "download":
            handle_download(args)
        elif args.subcommand == "rm":
            handle_rm(args)
        elif args.subcommand == "info":
            handle_info(args)
        elif args.subcommand == "cp":
            handle_cp(args)
        elif args.subcommand == "mv":
            handle_mv(args)
        elif args.subcommand == "du":
            handle_du(args)
        return


if __name__ == "__main__":
    main()
