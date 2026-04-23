# clanker

detect hardware and find GGUF models that fit.

## install

```bash
pip install clanker
```

## usage

```bash
clanker
clanker <repo_id>
clanker ls
clanker download <repo_id>
clanker rm <file>
clanker info <file>
clanker cp <src> <dest>
clanker mv <src> <dest>
clanker du
```

## local storage

uses huggingface cache directory.

## requirements

- python 3.10+
- llama-server (optional, for inference)
