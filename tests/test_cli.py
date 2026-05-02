import sys
from unittest import mock
import pytest
from clanker import main

def test_main_check_no_args(monkeypatch, capsys):
    # mock hardware so we don't actually hit the system. i hate mocking.
    with mock.patch("clanker.detect_ram", return_value=32.0), \
         mock.patch("clanker.detect_gpus", return_value=[{"name": "RTX 4090", "vram_gb": 24.0, "kind": "nvidia"}]), \
         mock.patch("clanker.shutil.which", return_value="/usr/bin/nvidia-smi"):
        
        # mock sys.argv. clanker with no args should just work.
        monkeypatch.setattr(sys, "argv", ["clanker"])
        
        main()
        
        captured = capsys.readouterr()
        assert "clanker — What GGUF models fit your hardware?" in captured.out
        assert "Q4_K_M" in captured.out

def test_main_check_json(monkeypatch, capsys):
    # json output. for the robots.
    with mock.patch("clanker.detect_ram", return_value=16.0), \
         mock.patch("clanker.detect_gpus", return_value=[]):
        
        monkeypatch.setattr(sys, "argv", ["clanker", "--json"])
        main()
        
        captured = capsys.readouterr()
        assert '"hardware":' in captured.out
        assert '"sources":' in captured.out

def test_main_set_ctx(monkeypatch, capsys, tmp_path):
    # isolate home so we don't mess up the actual config.
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)
    
    monkeypatch.setattr(sys, "argv", ["clanker", "set", "ctx", "8192"])
    main()
    
    captured = capsys.readouterr()
    assert "Default context size set to 8192" in captured.out
