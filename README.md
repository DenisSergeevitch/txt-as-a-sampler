# Text File Sampler Chat

A CLI tool that constrains a HuggingFace causal language model to only generate words and tokens that appear in a given text file.

## Overview

This tool implements vocabulary-constrained text generation. The model can only output words (and punctuation/whitespace) that exist in your provided corpus file. The output won't necessarily reproduce your file verbatim—it just can't use *new* words beyond what's in your vocabulary file.

Inspired by the "biblically-accurate-sampler" concept, generalized to work with any plaintext corpus in multiple languages (English, Russian, Ukrainian, etc.).

## How It Works

1. **Trie Construction**: Builds a trie of token-ID sequences from every unique word in your text file, plus space-prefixed variants
2. **Constrained Generation**: During generation, greedily chooses the most likely next token that keeps us on a valid trie path
3. **Word Boundaries**: When an allowed token completes, allows punctuation/whitespace or starts a new token

## Installation

```bash
pip install torch transformers
```

## Usage

### Basic Usage

```bash
python text_file_sampler_chat.py --txt ./my_corpus.txt
```

### With Custom Model

```bash
python text_file_sampler_chat.py --model /path/to/model --txt ./my_corpus.txt
```

### All Options

```bash
python text_file_sampler_chat.py \
  --model /path/to/model \
  --txt ./corpus.txt \
  --encoding utf-8 \
  --max-new-tokens 128 \
  --min-token-len 1 \
  --max-unique-tokens 0 \
  --device auto \
  --dtype auto \
  --temperature 0.0 \
  --top-k 0 \
  --system "You can only use words from the provided text." \
  --trust-remote-code
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `./YandexGPT-5-Lite-8B-instruct` | HuggingFace model name or local path |
| `--txt` | (required) | Path to plaintext corpus file |
| `--encoding` | `utf-8` | Text file encoding |
| `--max-new-tokens` | `128` | Max tokens to generate per turn |
| `--min-token-len` | `1` | Drop corpus tokens shorter than this |
| `--max-unique-tokens` | `0` | Cap on unique corpus tokens (0 = no cap) |
| `--device` | `auto` | Device: `auto`, `mps`, `cuda`, `cpu` |
| `--dtype` | `auto` | Data type: `auto`, `fp16`, `bf16`, `fp32` |
| `--temperature` | `0.0` | Sampling temperature (0 = greedy) |
| `--top-k` | `0` | Sample from top-k allowed tokens |
| `--system` | `""` | Optional system prompt |
| `--trust-remote-code` | `false` | Pass trust_remote_code to HF loaders |

## Features

- **Multi-language Support**: Works with English, Russian, Ukrainian, and other languages using Unicode-aware tokenization
- **Streaming Output**: Responses are streamed token-by-token to the terminal
- **Flexible Encoding**: Automatically tries fallback encodings (utf-8-sig, cp1251, koi8-r) if primary fails
- **Repetition Penalty**: Built-in repetition penalty (1.2) to reduce repetitive outputs
- **Device Auto-detection**: Automatically selects MPS (Apple Silicon), CUDA, or CPU

## macOS / Apple Silicon Notes

- If you have Apple Silicon with PyTorch MPS support enabled, the script will use MPS automatically
- Larger models may not fit in memory; start with smaller models (~1B–3B parameters)

## Example Session

```
[info] device=mps dtype=torch.float16 model='./my-model'
[info] loading tokenizer...
[info] loading model...
[info] building trie from './bible.txt' ...
[info] unique corpus tokens added: 12847 subtoken-ids
[info] boundary tokens allowed: 156

Type your message. Ctrl+C to quit.

> Hello, how are you?
I am well, for the Lord is with me...
```

## License

MIT
