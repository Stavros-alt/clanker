# clanker

tells you what gguf models fit your hardware and manages local GGUF model files.

## install

```bash
pip install clanker
# or
uv pip install clanker
```

## usage

```bash
clanker                                    # show hardware + what fits
clanker unsloth/Qwen3-8B-GGUF             # check a specific model
clanker unsloth/Qwen3-8B-GGUF --context 8192  # with custom context

# new file system commands
clanker ls                                 # list local models
clanker download unsloth/Qwen3-8B-GGUF    # download model
clanker rm model.gguf                     # remove local model
clanker info model.gguf                   # show model details
clanker cp src.gguf dest.gguf             # copy model
clanker mv src.gguf dest.gguf             # move/rename model
clanker du                                # show disk usage
```

## output

hardware specs, quantization table, huggingface links.

## local storage

models are stored in the Hugging Face cache directory (~/.cache/huggingface/hub/) for easy integration.

## what it does not do

does not run or train models. use llama.cpp for inference.

## requirements

- python 3.10+
- works on linux, macos, windows
- optional: nvidia-smi (nvidia gpu), rocm-smi (amd gpu)
