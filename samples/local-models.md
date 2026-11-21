# Running Local Models with Ollama

Ollama makes it easy to download and serve open-weight language models through a
local HTTP API. The default endpoint is `http://localhost:11434`. Chat requests
go to `/api/chat`, embeddings to `/api/embed`, and the installed model list to
`/api/tags`.

## Models on a 6 GB laptop GPU

A 6 GB GPU such as an NVIDIA RTX 3050 Laptop can comfortably run quantized models
in the 3B to 4B range, and can run some 7B models with partial CPU offload. The
practical recommendation is to start small and measure tokens per second.

Good choices for a 6 GB GPU:

- **qwen2.5-coder:3b-instruct-q8_0** — about 3.1B parameters, roughly 3.3 GB on
  disk. Fast, strong at code and structured answers, and a good default chat
  model for RAG.
- **Qwen3 4B instruct (4-bit)** — about 3.1B to 4B parameters, roughly 2 to 3 GB.
  A good balance of quality and speed.
- **Llama 3.2 3B Instruct** — about 3.3B parameters, around 2 to 3 GB quantized.
- **Gemma 4 E4B (Q4_K_M)** — about 7.5B parameters, roughly 6.3 GB on disk. It
  offers vision and audio capabilities but is tight on a 6 GB GPU and may need
  CPU offload or a smaller quantization.
- **nomic-embed-text** — about 274 MB, 768-dimensional embeddings. It is the
  recommended embedding model for RAG on this hardware.

A 7B model at 4-bit quantization needs roughly 4 to 5 GB of weights plus context
and runtime overhead, so it can spill to system RAM. A 3B to 4B model leaves more
headroom for the context window and for the rest of the application.

## Context length and memory

Larger context windows consume more VRAM. A 8192-token context is usually enough
for RAG with six retrieved chunks. If a model runs out of memory, reduce the
context length before changing the model. Ollama exposes this through the
`num_ctx` option.

## Choosing an embedding model

RAG needs a dedicated embedding model. Chat models are not reliable embedders.
`nomic-embed-text` produces 768-dimensional vectors and is a strong default.
`all-minilm` is smaller and faster but produces 384-dimensional vectors and may
be less accurate. Whichever model you choose, keep it fixed for the lifetime of
an index: changing it requires re-embedding every document.

## Discovering what fits

`llmfit` inspects the machine's RAM, CPU, and GPU, then scores models for fit,
speed, and quality. Useful commands:

    llmfit system
    llmfit fit
    llmfit recommend
    llmfit info <model>

On a machine with 7.6 GB of RAM and a 6 GB GPU, expect 3B to 4B models to be the
sweet spot. Very large models may run but will be slow and may thrash memory.

## Windows and WSL2 networking

Ollama on Windows listens on `127.0.0.1:11434` by default. A process inside WSL2
cannot reach the Windows loopback address. Setting `OLLAMA_HOST=0.0.0.0:11434`
and restarting Ollama makes it listen on all interfaces; WSL can then connect
through the default gateway IP, which is the Windows host. A Windows Firewall
rule may be required on some machines. Binding to all interfaces exposes Ollama
to the local network, so restrict access on untrusted networks.

## Practical defaults

For a local RAG studio on this hardware, use a 3B to 4B chat model at Q4 or Q8,
`nomic-embed-text` for embeddings, 8192 context tokens, and a top-k of about 6
chunks. Measure latency and answer quality, then adjust chunk size and top-k
before moving to a larger model.
