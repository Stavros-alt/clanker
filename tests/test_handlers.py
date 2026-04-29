"""tests for file operation handlers (ls, rm, cp, mv, du, info) using real tmp filesystem"""

from pathlib import Path
from unittest import mock

import pytest

from clanker import (
    handle_cp,
    handle_du,
    handle_download,
    handle_ls,
    handle_mv,
    handle_rm,
    handle_set,
    handle_info,
    get_config_file,
    get_metadata_file,
    get_model_dir,
    infer_quant_from_filename,
    list_local_models,
    load_config,
    load_metadata,
    save_config,
    save_metadata,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper: patch HOME to isolate config/metadata in tmp_path
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """redirect ~/.clanker and ~/.cache/huggingface/hub to tmp_path"""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    # patch Path.home to return fake_home
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    return fake_home


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# handle_ls
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHandleLS:
    def test_handle_ls_empty_dir(self, isolated_home, capsys):
        model_dir = get_model_dir()
        model_dir.mkdir(parents=True)
        handle_ls(None)
        captured = capsys.readouterr()
        assert "No local models found" in captured.out

    def test_handle_ls_lists_models_with_quant(self, isolated_home, capsys):
        model_dir = get_model_dir()
        model_dir.mkdir(parents=True)
        # create some files with recognizable quant names
        file1 = model_dir / "models--mistral/Mistral-7B-v0.1-GGUF/Q4_K_M.gguf"
        file1.parent.mkdir(parents=True)
        file1.touch()
        file2 = model_dir / "models--llama/Llama-2-7B-GGUF/Q5_K_M.gguf"
        file2.parent.mkdir(parents=True)
        file2.touch()
        handle_ls(None)
        captured = capsys.readouterr()
        assert "Q4_K_M" in captured.out
        assert "Q5_K_M" in captured.out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# handle_rm
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHandleRm:
    def test_handle_rm_deletes_file(self, isolated_home, capsys):
        model_dir = get_model_dir()
        model_dir.mkdir(parents=True)
        test_file = model_dir / "test.gguf"
        test_file.touch()
        # also create metadata entry
        metadata = load_metadata()
        metadata["test.gguf"] = {"size_gb": 6.5, "quant": "Q4_K_M"}
        save_metadata(metadata)

        args = mock.Mock(model="test.gguf")
        handle_rm(args)

        assert not test_file.exists()
        # metadata entry removed
        new_meta = load_metadata()
        assert "test.gguf" not in new_meta

    def test_handle_rm_missing_file(self, isolated_home, capsys):
        model_dir = get_model_dir()
        model_dir.mkdir(parents=True)
        args = mock.Mock(model="missing.gguf")
        handle_rm(args)
        captured = capsys.readouterr()
        assert "not found" in captured.err


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# handle_info
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHandleInfo:
    def test_handle_info_found(self, isolated_home, capsys):
        metadata = {"model.gguf": {"size_gb": 6.5, "quant": "Q4_K_M"}}
        save_metadata(metadata)
        args = mock.Mock(model="model.gguf")
        handle_info(args)
        captured = capsys.readouterr()
        assert "Model: model.gguf" in captured.out
        assert "6.5 GB" in captured.out

    def test_handle_info_not_found(self, isolated_home, capsys):
        # no metadata
        args = mock.Mock(model="missing.gguf")
        handle_info(args)
        captured = capsys.readouterr()
        assert "No metadata" in captured.out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# handle_cp
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHandleCp:
    def test_handle_cp_copies(self, isolated_home, capsys):
        model_dir = get_model_dir()
        model_dir.mkdir(parents=True)
        src_file = model_dir / "source.gguf"
        src_file.touch()
        dest_file = model_dir / "dest.gguf"
        args = mock.Mock(src="source.gguf", dest="dest.gguf")
        handle_cp(args)
        assert dest_file.exists()
        captured = capsys.readouterr()
        assert "Copied" in captured.out

    def test_handle_cp_source_missing(self, isolated_home, capsys):
        model_dir = get_model_dir()
        model_dir.mkdir(parents=True)
        args = mock.Mock(src="missing.gguf", dest="dest.gguf")
        handle_cp(args)
        captured = capsys.readouterr()
        assert "not found" in captured.err


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# handle_mv
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHandleMv:
    def test_handle_mv_renames(self, isolated_home, capsys):
        model_dir = get_model_dir()
        model_dir.mkdir(parents=True)
        src_file = model_dir / "old.gguf"
        src_file.touch()
        metadata = load_metadata()
        metadata["old.gguf"] = {"size_gb": 5.0}
        save_metadata(metadata)

        args = mock.Mock(src="old.gguf", dest="new.gguf")
        handle_mv(args)

        dest_file = model_dir / "new.gguf"
        assert dest_file.exists()
        assert not src_file.exists()
        new_meta = load_metadata()
        assert "new.gguf" in new_meta
        assert "old.gguf" not in new_meta

    def test_handle_mv_missing(self, isolated_home, capsys):
        model_dir = get_model_dir()
        model_dir.mkdir(parents=True)
        args = mock.Mock(src="missing.gguf", dest="new.gguf")
        handle_mv(args)
        captured = capsys.readouterr()
        assert "not found" in captured.err


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# handle_du
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHandleDu:
    @mock.patch("clanker.print")
    @mock.patch("pathlib.Path.rglob")
    def test_handle_du_calculates_size(self, mock_rglob, mock_print):
        # simulate two gguf files with 2GB and 3GB sizes
        file1 = mock.Mock()
        file1.stat.return_value.st_size = 2 * 1024**3  # 2 GB
        file2 = mock.Mock()
        file2.stat.return_value.st_size = 3 * 1024**3  # 3 GB
        mock_rglob.return_value = [file1, file2]
        with mock.patch("pathlib.Path.exists", return_value=True):
            handle_du(None)
        # check that print was called with something containing "5.00 GB" or "5.0 GB"
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "5.00" in printed or "5.0" in printed

    @mock.patch("builtins.print")
    def test_handle_du_empty_dir(self, mock_print, isolated_home):
        model_dir = get_model_dir()
        model_dir.mkdir(parents=True)  # exists but no gguf files
        handle_du(None)
        printed = " ".join(str(call) for call in mock_print.call_args_list)
        assert "0.00" in printed or "0.0" in printed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# handle_set
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHandleSet:
    def test_handle_set_ctx_integer(self, isolated_home, capsys):
        cfg = load_config()
        assert "ctx" not in cfg
        args = mock.Mock(key="ctx", value="8192")
        handle_set(args)
        new_cfg = load_config()
        assert new_cfg["ctx"] == 8192
        captured = capsys.readouterr()
        assert "set to 8192" in captured.out

    def test_handle_set_invalid_integer(self, isolated_home, capsys):
        args = mock.Mock(key="ctx", value="not-a-number")
        handle_set(args)
        captured = capsys.readouterr()
        assert "not a valid integer" in captured.err

    def test_handle_set_unknown_key(self, isolated_home, capsys):
        args = mock.Mock(key="unknown", value="foo")
        handle_set(args)
        captured = capsys.readouterr()
        assert "unknown key" in captured.err


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# handle_download (mocked HF API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHandleDownload:
    @mock.patch("clanker.fetch_gguf_files")
    @mock.patch("huggingface_hub.hf_hub_download")
    def test_handle_download_success(self, mock_download, mock_fetch, isolated_home, capsys, monkeypatch):
        mock_fetch.return_value = (
            [{"name": "model-Q4_K_M.gguf", "size_gb": 6.5}],
            None
        )
        mock_download.return_value = str(get_model_dir() / "model-Q4_K_M.gguf")
        # actually create the file (and parent dirs) so metadata size check works
        dl_path = get_model_dir() / "model-Q4_K_M.gguf"
        dl_path.parent.mkdir(parents=True, exist_ok=True)
        dl_path.touch()

        args = mock.Mock(model="author/model", quant=None)
        handle_download(args)

        captured = capsys.readouterr()
        assert "Downloaded to" in captured.out

    @mock.patch("clanker.fetch_gguf_files")
    def test_handle_download_fetch_error(self, mock_fetch, isolated_home, capsys):
        mock_fetch.return_value = (None, "API error")
        args = mock.Mock(model="bad/repo", quant=None)
        handle_download(args)
        captured = capsys.readouterr()
        assert "Error: API error" in captured.err

    @mock.patch("clanker.fetch_gguf_files")
    def test_handle_download_no_gguf(self, mock_fetch, isolated_home, capsys):
        mock_fetch.return_value = ([], "no GGUF files found")
        args = mock.Mock(model="empty/repo", quant=None)
        handle_download(args)
        captured = capsys.readouterr()
        assert "no GGUF files found" in captured.err


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config/Path helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_get_model_dir_returns_path():
    p = get_model_dir()
    assert ".cache" in str(p)
    assert "huggingface" in str(p)
    assert "hub" in str(p)

def test_get_config_file_returns_path():
    p = get_config_file()
    assert ".clanker" in str(p)
    assert "config.json" in str(p)

def test_get_metadata_file_returns_path():
    p = get_metadata_file()
    assert ".clanker" in str(p)
    assert "metadata.json" in str(p)

def test_load_save_config_roundtrip(isolated_home):
    data = {"ctx": 12345, "mode": "vram"}
    save_config(data)
    loaded = load_config()
    assert loaded == data

def test_load_save_metadata_roundtrip(isolated_home):
    data = {"model.gguf": {"size_gb": 6.5}}
    save_metadata(data)
    loaded = load_metadata()
    assert loaded == data
