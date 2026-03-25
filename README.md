# gguf-fit

tells you what gguf models fit your hardware. that's it.

## install

```bash
pip install gguf-fit
# or
uv pip install gguf-fit
```

## usage

```bash
gguf-fit                                    # show hardware + what fits
gguf-fit unsloth/Qwen3-8B-GGUF             # check a specific model
gguf-fit unsloth/Qwen3-8B-GGUF --context 8192  # with custom context
```

## output

hardware specs, quantization table, huggingface links.

## what it does not do

does not download, run, or train models. use unsloth studio or llama.cpp for that.

## requirements

- python 3.10+
- works on linux, macos, windows
- optional: nvidia-smi (nvidia gpu), rocm-smi (amd gpu)
