import pytest
from clanker import (
    get_kv_factor,
    get_oh_values,
    compute_max_context,
    default_overhead,
)

class TestKVQuantizationMath:
    def test_get_kv_factor_default(self):
        # f16 is default. factor 1.0 because i said so.
        assert get_kv_factor(None, None) == 1.0
        assert get_kv_factor("f16", "f16") == 1.0

    def test_get_kv_factor_quantized(self):
        # q8_0 is 0.5. q4_0 is 0.28. don't ask where i got these.
        assert get_kv_factor("q8_0", "q8_0") == 0.5
        assert get_kv_factor("q4_0", "q4_0") == 0.28
        # mixed. half of one, half of the other.
        assert get_kv_factor("f16", "q4_0") == pytest.approx((1.0 + 0.28) / 2.0)

    def test_get_oh_values_with_kv_quant(self):
        # nvidia base is 3.5. ctx factor is 0.25. 
        # with q4_0 it should be way smaller.
        base, factor = get_oh_values("nvidia", ctk="q4_0", ctv="q4_0")
        assert base == pytest.approx(3.5)
        assert factor == pytest.approx(0.25 * 0.28)

    def test_compute_max_context_with_kv_quant(self):
        # f16 vs q4_0. q4 should definitely fit more.
        ctx_f16 = compute_max_context("vram", 4, 10, 0, "nvidia", None, None)
        assert ctx_f16 == pytest.approx(10240)
        
        ctx_q4 = compute_max_context("vram", 4, 10, 0, "nvidia", "q4_0", "q4_0")
        assert ctx_q4 == pytest.approx(int(2.5 / (0.25 * 0.28) * 1024))
        assert ctx_q4 > ctx_f16

class TestPresets:
    # testing the math presets set up. i'm not mocking main().
    
    def test_agentic_coding_math(self):
        # agentic-coding = q8_0
        base, factor = get_oh_values("nvidia", ctk="q8_0", ctv="q8_0")
        assert factor == pytest.approx(0.25 * 0.5)

    def test_memory_preset_math(self):
        # memory preset = q4_0
        base, factor = get_oh_values("nvidia", ctk="q4_0", ctv="q4_0")
        assert factor == pytest.approx(0.25 * 0.28)

def test_default_overhead_with_kv():
    # just making sure it doesn't crash
    assert default_overhead("nvidia", ctk="q8_0", ctv="q8_0") == pytest.approx(3.5)
