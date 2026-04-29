"""common test fixtures and helpers"""

import platform
from unittest import mock

import pytest


@pytest.fixture
def mock_platform_linux():
    with mock.patch("platform.system", return_value="Linux"):
        yield


@pytest.fixture
def mock_platform_darwin():
    with mock.patch("platform.system", return_value="Darwin"):
        yield


@pytest.fixture
def mock_platform_windows():
    with mock.patch("platform.system", return_value="Windows"):
        yield


@pytest.fixture
def sample_gguf_files():
    return [
        {"name": "mistral-Q3_K_M.gguf", "size_gb": 4.5},
        {"name": "mistral-Q4_K_M.gguf", "size_gb": 5.8},
        {"name": "mistral-Q5_K_M.gguf", "size_gb": 7.0},
        {"name": "mistral-Q6_K.gguf", "size_gb": 8.5},
    ]


@pytest.fixture
def hardware_gpu():
    return [{"name": "NVIDIA RTX 4090", "vram_gb": 24.0, "kind": "nvidia"}]


@pytest.fixture
def hardware_apple():
    return [{"name": "Apple M2 Ultra", "vram_gb": 64.0, "kind": "apple"}]
