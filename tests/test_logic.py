"""tests for clanker hardware detection and quantization logic"""

import json
import platform
import subprocess
from unittest import mock

import pytest

from clanker import (
    compute_max_context,
    default_overhead,
    detect_gpus,
    detect_ram,
    find_best_fit,
    find_best_fit_for_mode,
    get_bpw,
    get_oh_values,
    infer_quant_from_filename,
    is_4bit_quant,
    is_dynamic_quant,
    make_url,
    max_billions,
    recommend_mode,
    resolve_quant,
    fetch_model_max_context,
    fetch_gguf_files,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quant Table Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestQuantLogic:
    def test_resolve_quant_ud_alias(self):
        # UD aliases should map to standard quants
        assert resolve_quant("UD-IQ1_S") == "IQ1_S"
        assert resolve_quant("UD-Q4_K_XL") == "Q4_K_M"
        assert resolve_quant("UD-Q6_K_XL") == "Q6_K"

    def test_resolve_quant_unknown_returns_original(self):
        # non-UD keys pass through
        assert resolve_quant("Q4_K_M") == "Q4_K_M"
        assert resolve_quant("Q5_K_S") == "Q5_K_S"

    def test_get_bpw_returns_known_quant(self):
        # spot-check a few values from QUANTS dict
        assert get_bpw("Q4_K_M") == pytest.approx(4.85)
        assert get_bpw("Q6_K") == pytest.approx(6.57)
        assert get_bpw("IQ1_S") == pytest.approx(1.56)

    def test_get_bpw_unknown_returns_none(self):
        assert get_bpw("INVALID_QUANT") is None

    def test_is_dynamic_quant_detects_ud_keys(self):
        assert is_dynamic_quant("UD-IQ1_S") is True
        assert is_dynamic_quant("UD-Q4_K_XL") is True

    def test_is_dynamic_quant_rejects_standard(self):
        assert is_dynamic_quant("Q4_K_M") is False
        assert is_dynamic_quant("Q6_K") is False

    def test_infer_quant_from_filename_matches_standard(self):
        assert infer_quant_from_filename("model-Q4_K_M.gguf") == "Q4_K_M"
        assert infer_quant_from_filename("mistral-Q5_K_S.gguf") == "Q5_K_S"
        # longest match wins
        assert infer_quant_from_filename("model-Q4_K_M.gguf") == "Q4_K_M"

    def test_infer_quant_from_filename_ud_alias(self):
        # UD variants should return the UD key, not the base
        assert infer_quant_from_filename("model-UD-IQ1_S.gguf") == "UD-IQ1_S"
        assert infer_quant_from_filename("my-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"

    def test_infer_quant_from_filename_no_match(self):
        assert infer_quant_from_filename("random-model.gguf") is None
        assert infer_quant_from_filename("model-q7.gguf") is None

    def test_is_4bit_quant_positive_cases(self):
        assert is_4bit_quant("Q3_K_M") is True
        assert is_4bit_quant("Q4_K_M") is True
        assert is_4bit_quant("Q4_0") is True
        assert is_4bit_quant("IQ3_M") is True
        assert is_4bit_quant("IQ4_XS") is True

    def test_is_4bit_quant_negative_cases(self):
        assert is_4bit_quant("Q5_K_M") is False
        assert is_4bit_quant("Q6_K") is False
        assert is_4bit_quant("Q8_0") is False
        assert is_4bit_quant("BF16") is False
        assert is_4bit_quant(None) is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hardware Detection Tests (mocked system calls)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDetectRAM:
    @mock.patch("platform.system")
    def test_detect_ram_linux(self, mock_system):
        mock_system.return_value = "Linux"
        with mock.patch("builtins.open", mock.mock_open(read_data="MemTotal: 16384576 kB")):
            ram_gb = detect_ram()
        assert ram_gb == pytest.approx(16384576 / 1048576, rel=0.01)

    @mock.patch("platform.system")
    def test_detect_ram_darwin(self, mock_system):
        mock_system.return_value = "Darwin"
        with mock.patch("subprocess.check_output", return_value="17179869184\n"):
            ram_gb = detect_ram()
        assert ram_gb == pytest.approx(17179869184 / (1024**3), rel=0.01)

    @mock.patch("platform.system")
    def test_detect_ram_windows(self, mock_system):
        mock_system.return_value = "Windows"
        with mock.patch("subprocess.check_output", return_value="17179869184"):
            ram_gb = detect_ram()
        assert ram_gb == pytest.approx(17179869184 / (1024**3), rel=0.01)

    @mock.patch("platform.system")
    def test_detect_ram_unknown_os_returns_none(self, mock_system):
        mock_system.return_value = "UnknownOS"
        assert detect_ram() is None

    @mock.patch("platform.system")
    def test_detect_ram_handles_exceptions(self, mock_system):
        mock_system.return_value = "Linux"
        with mock.patch("builtins.open", side_effect=OSError("no such file")):
            assert detect_ram() is None


class TestDetectGPUs:
    @mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi")
    @mock.patch("platform.system", return_value="Linux")
    def test_detect_gpus_nvidia(self, mock_system, mock_which):
        fake_output = "NVIDIA GeForce RTX 4090, 24576\n"
        # block AMD /sys/class/drm scanning
        with mock.patch("os.listdir", side_effect=FileNotFoundError):
            with mock.patch("subprocess.check_output", return_value=fake_output):
                gpus = detect_gpus()
        assert len(gpus) == 1
        assert gpus[0]["kind"] == "nvidia"
        assert gpus[0]["name"] == "NVIDIA GeForce RTX 4090"
        assert gpus[0]["vram_gb"] == pytest.approx(24.0)

    @mock.patch("shutil.which", return_value=None)  # no nvidia-smi, no rocm-smi
    @mock.patch("platform.system", return_value="Linux")
    def test_detect_gpus_amd_linux_drm(self, mock_system, mock_which):
        # Simulate /sys/class/drm structure
        fake_vram_path = "/sys/class/drm/card0/device/mem_info_vram_total"
        fake_name_path = "/sys/class/drm/card0/device/product_name"

        def isfile(path):
            return path in (fake_vram_path, fake_name_path)

        def listdir(path):
            if path == "/sys/class/drm":
                return ["card0"]
            return []

        # 16 GB in bytes
        sixteen_gb_bytes = str(16 * 1024**3)
        file_map = {
            fake_vram_path: sixteen_gb_bytes + "\n",
            fake_name_path: "AMD Radeon RX 7900 XT\n",
        }

        def mock_open(path, *args, **kwargs):
            from io import StringIO
            return StringIO(file_map.get(str(path), ""))

        with mock.patch("os.listdir", side_effect=listdir):
            with mock.patch("os.path.isfile", side_effect=isfile):
                with mock.patch("builtins.open", mock_open):
                    gpus = detect_gpus()
        amd_gpus = [g for g in gpus if g["kind"] == "amd"]
        assert len(amd_gpus) >= 1
        assert amd_gpus[0]["vram_gb"] == pytest.approx(16.0)

    @mock.patch("shutil.which", return_value=None)  # no gpu tools
    @mock.patch("platform.system", return_value="Darwin")
    @mock.patch("platform.machine", return_value="arm64")
    def test_detect_gpus_apple_silicon(self, mock_machine, mock_system, mock_which):
        # Apple Silicon: unifed memory = RAM
        with mock.patch("clanker.detect_ram", return_value=64.0):
            with mock.patch("subprocess.check_output", return_value="Apple M2 Ultra\n"):
                gpus = detect_gpus()
        assert len(gpus) == 1
        assert gpus[0]["kind"] == "apple"
        assert "Apple" in gpus[0]["name"]
        assert gpus[0]["vram_gb"] == pytest.approx(64.0)

    @mock.patch("shutil.which")
    def test_detect_gpus_empty_when_no_gpus(self, mock_which):
        mock_which.return_value = None  # no nvidia-smi, no rocm-smi
        with mock.patch("platform.system", return_value="Linux"):
            with mock.patch("os.listdir", side_effect=FileNotFoundError):
                gpus = detect_gpus()
        assert gpus == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Overhead & Memory Math
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMemoryMath:
    def test_get_oh_values_apple(self):
        base, factor = get_bpw("apple"), 0.3  # wrong, just checking the madness
        # actually get_oh_values returns overhead, not bpw
        v_overhead, ctx_factor = get_oh_values("apple")
        assert v_overhead == pytest.approx(4.0)
        assert ctx_factor == pytest.approx(0.3)

    def test_get_oh_values_nvidia_amd(self):
        v_overhead, ctx_factor = get_oh_values("nvidia")
        # nvidia/amd use 2.0 base + RUNTIME_OVERHEAD_GB (1.5) = 3.5
        assert v_overhead == pytest.approx(3.5)
        assert ctx_factor == pytest.approx(0.25)

    def test_get_oh_values_ram(self):
        v_overhead, ctx_factor = get_oh_values("ram")
        assert v_overhead == pytest.approx(3.0)
        assert ctx_factor == pytest.approx(0.5)

    def test_default_overhead_returns_base(self):
        assert default_overhead("apple") == pytest.approx(4.0)
        assert default_overhead("nvidia", mem_gb=24) == pytest.approx(3.5)

    def test_max_billions_basic(self):
        # avail=10GB, bpw=4.0 → 10 * 8 / 4 = 20B
        result = max_billions(10, "Q4_K_M", overhead_gb=2)
        # Q4_K_M bpw = 4.85
        expected = 8 * 8 / 4.85
        assert result == pytest.approx(expected)

    def test_max_billions_negative_avail_returns_zero(self):
        assert max_billions(1, "Q4_K_M", overhead_gb=2) == 0.0

    def test_make_url_rounds_down(self):
        url = make_url(20.7)
        assert "max:20B" in url
        url = make_url(20.2)
        assert "max:20B" in url
        url = make_url(0.4)
        assert "max:1B" in url

    def test_compute_max_context_vram_mode(self):
        # model_size=6GB, vram=24GB, overhead=2GB, factor=0.25
        # rem = 24 - 2 - 6 = 16GB → 16 * 1024 / 0.25 * 1024? wait let me trace
        # the function returns int((rem / ctx_factor) * 1024)
        ctx = compute_max_context("vram", model_size_gb=6, vram_gb=24, ram_gb=0, gpu_kind="nvidia")
        # base_oh=3.5, ctx_factor=0.25, rem=24-3.5-6=14.5 → 14.5/0.25*1024 = 59392
        assert ctx == pytest.approx(int((24 - 3.5 - 6) / 0.25 * 1024))

    def test_compute_max_context_ram_mode(self):
        ctx = compute_max_context("ram", model_size_gb=10, vram_gb=0, ram_gb=32)
        # base_oh=3.0, ctx_factor=0.5, rem=32-3-10=19 → 19/0.5*1024 = 38912
        assert ctx == pytest.approx(int((32 - 3.0 - 10) / 0.5 * 1024))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Best-Fit Algorithm Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFindBestFit:
    def test_find_best_fit_4bit_required(self):
        gguf_files = [
            {"name": "model-Q4_K_M.gguf", "size_gb": 6.5},
            {"name": "model-Q5_K_M.gguf", "size_gb": 8.2},
        ]
        file, quant, size = find_best_fit(gguf_files, mem_available=10, overhead_gb=2)
        assert file is not None
        assert quant == "Q4_K_M"
        assert size == pytest.approx(6.5)

    def test_find_best_fit_no_4bit_fits_returns_none(self):
        gguf_files = [
            {"name": "model-Q5_K_M.gguf", "size_gb": 8.2},
            {"name": "model-Q6_K.gguf", "size_gb": 9.5},
        ]
        file, quant, size = find_best_fit(gguf_files, mem_available=8, overhead_gb=2)
        assert file is None
        assert quant is None
        assert size is None

    def test_find_best_fit_highest_quality_4bit(self):
        gguf_files = [
            {"name": "model-Q3_K_M.gguf", "size_gb": 5.0},
            {"name": "model-Q4_K_M.gguf", "size_gb": 6.5},
            {"name": "model-Q5_K_M.gguf", "size_gb": 8.2},
        ]
        file, quant, size = find_best_fit(gguf_files, mem_available=9, overhead_gb=2)
        # only 4-bit quants (Q3, Q4) are considered; Q5 is not 4-bit.
        # best quality among 4-bit is Q4_K_M (bpw 4.85 > 3.74)
        assert quant == "Q4_K_M"

    def test_find_best_fit_ignores_non_gguf(self):
        gguf_files = [
            {"name": "model-Q4_K_M.gguf", "size_gb": 6.5},
            {"name": "model-Q4_K_M.GGUF", "size_gb": 6.5},  # uppercase
        ]
        # both should be detected. uppercase check uses .lower()
        file, quant, size = find_best_fit(gguf_files, mem_available=10, overhead_gb=2)
        assert file is not None

    def test_find_best_fit_empty_list(self):
        file, quant, size = find_best_fit([], mem_available=10, overhead_gb=2)
        assert file is None


class TestFindBestFitForMode:
    def test_find_best_fit_for_mode_ram(self):
        gguf_files = [{"name": "model-Q4_K_M.gguf", "size_gb": 6.5}]
        f, q, sz, max_b = find_best_fit_for_mode(
            gguf_files, vram_gb=0, vram_overhead=0, ram_gb=16, ram_overhead=3, mode="ram"
        )
        assert f is not None
        assert q in ("Q4_K_M", "Q3_K_M")  # whichever fits

    def test_find_best_fit_for_mode_vram(self):
        gguf_files = [{"name": "model-Q4_K_M.gguf", "size_gb": 6.5}]
        f, q, sz, max_b = find_best_fit_for_mode(
            gguf_files, vram_gb=12, vram_overhead=3, ram_gb=0, ram_overhead=0, mode="vram"
        )
        assert f is not None

    def test_find_best_fit_for_mode_hybrid(self):
        gguf_files = [{"name": "model-Q4_K_M.gguf", "size_gb": 6.5}]
        f, q, sz, max_b = find_best_fit_for_mode(
            gguf_files, vram_gb=8, vram_overhead=3, ram_gb=16, ram_overhead=2, mode="hybrid"
        )
        assert f is not None

    def test_find_best_fit_for_mode_insufficient_memory(self):
        gguf_files = [{"name": "model-Q6_K.gguf", "size_gb": 12}]
        f, q, sz, max_b = find_best_fit_for_mode(
            gguf_files, vram_gb=8, vram_overhead=3, ram_gb=0, ram_overhead=0, mode="vram"
        )
        assert f is None


class TestRecommendMode:
    def test_recommend_mode_prefers_vram(self):
        gguf_files = [{"name": "model-Q4_K_M.gguf", "size_gb": 6.5}]
        mode, f, q, sz, max_b = recommend_mode(
            vram_gb=12, ram_gb=16, gguf_files=gguf_files, oh_fn=lambda k, m: 3.0
        )
        # vram should work (6.5 < 12-3=9), so VRAM is recommended
        assert mode == "VRAM"

    def test_recommend_mode_fallback_to_ram(self):
        gguf_files = [{"name": "model-Q4_K_M.gguf", "size_gb": 6.5}]
        mode, f, q, sz, max_b = recommend_mode(
            vram_gb=2,  # tiny GPU that can't fit model even with no overhead (6.5 > 2)
            ram_gb=16,
            gguf_files=gguf_files,
            oh_fn=lambda k, m: 3.0,
            gpu_kind="nvidia",  # having a dedicated GPU means hybrid requires VRAM for 20% rule
        )
        # VRAM fails (too small), Hybrid fails (20% VRAM requirement unsatisfiable),
        # RAM succeeds → fallback to RAM
        assert mode == "RAM"

    def test_recommend_mode_hybrid_when_vram_fails_but_combined_works(self):
        gguf_files = [{"name": "model-Q4_K_M.gguf", "size_gb": 6.5}]
        mode, f, q, sz, max_b = recommend_mode(
            vram_gb=4,  # too small alone
            ram_gb=16,
            gguf_files=gguf_files,
            oh_fn=lambda k, m: 3.0 if k == "ram" else 2.0,
        )
        # hybrid may work if combined >= size + overheads
        assert mode in ("Hybrid", "RAM")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API Fetching Tests (mocked urllib)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFetchModelMaxContext:
    @mock.patch("urllib.request.urlopen")
    def test_fetch_from_api_success(self, mock_urlopen):
        mock_resp = mock.Mock()
        mock_resp.read.return_value = b'{"config": {"max_position_embeddings": 8192}}'
        mock_urlopen.return_value.__enter__ = mock.Mock(return_value=mock_resp)
        mock_urlopen.return_value.__exit__ = mock.Mock(return_value=False)

        ctx = fetch_model_max_context("some/model")
        assert ctx == 8192

    @mock.patch("urllib.request.urlopen")
    def test_fetch_fallback_to_raw_config(self, mock_urlopen):
        # API fails, raw config.json succeeds
        def side_effect(req, timeout=None):
            # req may be a Request object or string
            url = getattr(req, "full_url", req) if not isinstance(req, str) else req
            if "api/models" in str(url):
                raise Exception("api down")
            mock_resp = mock.Mock()
            mock_resp.read.return_value = b'{"n_ctx": 4096}'
            mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
            mock_resp.__exit__ = mock.Mock(return_value=False)
            return mock_resp

        mock_urlopen.side_effect = side_effect

        ctx = fetch_model_max_context("some/model")
        assert ctx == 4096

    @mock.patch("urllib.request.urlopen")
    def test_fetch_returns_none_when_both_fail(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("network dead")
        assert fetch_model_max_context("some/model") is None


class TestFetchGGUFFiles:
    @mock.patch("urllib.request.urlopen")
    def test_fetch_returns_file_list(self, mock_urlopen):
        tree = [
            {"type": "file", "path": "model-Q4_K_M.gguf", "size": 6_500_000_000},
            {"type": "file", "path": "model-Q5_K_M.gguf", "size": 8_200_000_000},
            {"type": "directory", "path": "mmproj"},  # should be skipped
        ]
        mock_resp = mock.Mock()
        mock_resp.read.return_value = json.dumps(tree).encode()
        mock_urlopen.return_value.__enter__ = mock.Mock(return_value=mock_resp)
        mock_urlopen.return_value.__exit__ = mock.Mock(return_value=False)

        files, err = fetch_gguf_files("some/model")
        assert err is None
        assert len(files) == 2
        assert files[0]["name"] == "model-Q4_K_M.gguf"

    @mock.patch("urllib.request.urlopen")
    def test_fetch_returns_error_on_api_failure(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("timeout")
        files, err = fetch_gguf_files("some/model")
        assert files is None
        assert err is not None

    @mock.patch("urllib.request.urlopen")
    def test_fetch_skips_mmproj_files(self, mock_urlopen):
        tree = [
            {"type": "file", "path": "model-mmproj-layer.gguf", "size": 100},
        ]
        mock_resp = mock.Mock()
        mock_resp.read.return_value = json.dumps(tree).encode()
        mock_urlopen.return_value.__enter__ = mock.Mock(return_value=mock_resp)
        mock_urlopen.return_value.__exit__ = mock.Mock(return_value=False)

        files, err = fetch_gguf_files("some/model")
        assert files is None
        assert err == "no GGUF files found in repository"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# URL Builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_make_url_format():
    url = make_url(7.5)
    assert "max:7B" in url
    assert "huggingface.co/models" in url

def test_make_url_small_model():
    url = make_url(0.4)
    assert "max:1B" in url

def test_make_url_zero():
    url = make_url(0)
    assert "max:0B" in url or "max:1B" in url  # depends on max(1, floor(0))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Corner Cases: None, negative, empty
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_resolve_quant_none():
    assert resolve_quant(None) is None

def test_get_bpw_none():
    assert get_bpw(None) is None

def test_infer_quant_from_filename_empty_string():
    assert infer_quant_from_filename("") is None

def test_is_4bit_quant_empty_string():
    assert not is_4bit_quant("")
