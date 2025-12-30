#!/usr/bin/env python3
"""
text_file_sampler_chat.py

A tiny "chat" CLI that constrains a HuggingFace causal LM so it can only emit
words (and punctuation/whitespace tokens) that appear in a given .txt file.

This is adapted from the idea in vgel's "biblically-accurate-sampler" notebook/script,
but generalized to any plaintext corpus (English / Russian / Ukrainian, etc.).

Key idea:
- Build a trie of token-ID sequences corresponding to every unique "word-like token"
  found in your text file, plus a space-prefixed variant (" " + token).
- During generation, greedily choose the most likely next token that keeps us on
  a valid trie path (i.e., we are always spelling an allowed token).
- When we hit the end of an allowed token, allow punctuation/whitespace, or start a new token.

This makes the model's outputs "vocabulary-limited" to whatever is present in your file.
It won't necessarily reproduce your file verbatim; it just can't use *new* words.

Usage:
  python text_file_sampler_chat.py --txt ./my_corpus.txt
  python text_file_sampler_chat.py --model ./YandexGPT-5-Lite-8B-instruct --txt ./my_corpus.txt

Notes for macOS / Metal (MPS):
- If you have Apple Silicon + PyTorch with MPS enabled, the script will use MPS automatically.
- Bigger models may not fit; start with ~1B–3B.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import sys
import unicodedata
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    # Newer Transformers versions
    from transformers import DynamicCache  # type: ignore
except Exception:  # pragma: no cover
    DynamicCache = None  # type: ignore


# -----------------------------
# Text tokenization (corpus -> "word-like" tokens)
# -----------------------------

# Matches:
#  - letters (unicode), with optional internal apostrophes/hyphens (e.g., don't, l'amour, п'єса)
#  - numbers
#  - any single non-whitespace character (punctuation, symbols, etc.)
TOKEN_RE = re.compile(r"[^\W\d_]+(?:[’'\-][^\W\d_]+)*|\d+|[^\s]", re.UNICODE)


def iter_corpus_tokens(path: str, encoding: str = "utf-8") -> Iterator[str]:
    """
    Stream tokens from a text file without loading the whole file.

    For English/Russian/Ukrainian this is usually sufficient and avoids NLTK downloads.
    """
    # Try a couple common fallbacks if user doesn't specify a working encoding.
    encodings_to_try = [encoding]
    if encoding.lower() in ("utf8", "utf-8"):
        encodings_to_try += ["utf-8-sig", "cp1251", "koi8-r"]

    last_err: Optional[Exception] = None
    for enc in encodings_to_try:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                for line in f:
                    for tok in TOKEN_RE.findall(line):
                        yield tok
            return
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Failed to read {path!r} with encodings tried: {encodings_to_try}. "
        f"Last error: {last_err}"
    )


# -----------------------------
# Token trie (allowed token-ID sequences)
# -----------------------------

@dataclasses.dataclass
class TokenTrie:
    nodes: Dict[int, "TokenTrie"] = dataclasses.field(default_factory=dict)
    has_leading_space: bool = False
    is_terminal: bool = False

    def add_phrase(self, phrase: str, tokenizer) -> Set[int]:
        """
        Add one allowed phrase to the trie. Returns the set of token IDs used.
        """
        token_ids = tokenizer.encode(phrase, add_special_tokens=False)
        cur = self
        for t in token_ids:
            cur = cur.nodes.setdefault(int(t), TokenTrie())
        cur.is_terminal = True

        # Track whether the first token of this phrase corresponds to a "space-prefixed" variant.
        if phrase.startswith(" ") and token_ids:
            first = int(token_ids[0])
            self.nodes[first].has_leading_space = True

        return set(int(t) for t in token_ids)

    def get(self, token_id: int) -> Optional["TokenTrie"]:
        return self.nodes.get(int(token_id))

    def is_leading_space_tok(self, token_id: int) -> bool:
        token_id = int(token_id)
        return token_id in self.nodes and self.nodes[token_id].has_leading_space


def build_trie_from_txt(
    tokenizer,
    txt_path: str,
    encoding: str = "utf-8",
    min_token_len: int = 1,
    max_unique_tokens: Optional[int] = None,
) -> Tuple[TokenTrie, Set[int]]:
    """
    Build a trie of allowed words/punct from the file.
    Returns (trie, allowed_subtoken_ids_used).

    allowed_subtoken_ids_used is used to make punctuation/whitespace handling stricter:
    we only allow punctuation/whitespace tokens that appear somewhere in the corpus tokenizations.
    """
    uniq: Set[str] = set()
    for tok in iter_corpus_tokens(txt_path, encoding=encoding):
        if len(tok) < min_token_len:
            continue
        uniq.add(tok)
        if max_unique_tokens is not None and len(uniq) >= max_unique_tokens:
            break

    trie = TokenTrie()
    used_subtoken_ids: Set[int] = set()

    # Add each token both with and without a leading space to help match tokenizers that
    # distinguish "word" vs " word" tokens (e.g., GPT-2/BPE tokenizers).
    for tok in sorted(uniq):
        used_subtoken_ids |= trie.add_phrase(tok, tokenizer)
        used_subtoken_ids |= trie.add_phrase(" " + tok, tokenizer)

    return trie, used_subtoken_ids


# -----------------------------
# Word-boundary / punctuation logic
# -----------------------------

def _is_punct_or_symbol(ch: str) -> bool:
    # Unicode category: P* = punctuation, S* = symbols
    cat = unicodedata.category(ch)
    return bool(cat) and cat[0] in ("P", "S")


def tokens_that_can_end_word(tokenizer, allowed_subtoken_ids: Optional[Set[int]] = None) -> Set[int]:
    """
    Identify token IDs that are "safe boundary" tokens: punctuation, symbols, or whitespace.

    If allowed_subtoken_ids is provided, we intersect with it so we only allow boundary
    tokens that actually appear somewhere in the corpus tokenizations. This keeps the
    promise closer to "only tokens from the file".
    """
    vocab = tokenizer.get_vocab()

    can_end: Set[int] = set()
    for tok_str, tid in vocab.items():
        tid = int(tid)
        s = tokenizer.decode([tid])
        stripped = s.strip()

        # Check for SentencePiece word boundary marker (▁) - always allow it
        if tok_str == "▁" or tok_str == "\u2581":
            can_end.add(tid)
            continue

        if stripped == "":
            # pure whitespace (space, newline, etc.)
            can_end.add(tid)
            continue
        if all(_is_punct_or_symbol(ch) for ch in stripped):
            can_end.add(tid)

    # Also add common whitespace tokens that may decode oddly
    always_allow: Set[int] = set()
    for test_str in [" ", "  ", "\n", "\t"]:
        try:
            test_ids = tokenizer.encode(test_str, add_special_tokens=False)
            for tid in test_ids:
                can_end.add(int(tid))
                always_allow.add(int(tid))
        except Exception:
            pass

    # Keep SentencePiece marker always allowed
    sp_marker = vocab.get("▁") or vocab.get("\u2581")
    if sp_marker is not None:
        always_allow.add(int(sp_marker))

    if allowed_subtoken_ids is not None:
        can_end &= set(int(t) for t in allowed_subtoken_ids)

    # Re-add essential whitespace tokens that shouldn't be filtered
    can_end |= always_allow

    return can_end


def _eos_token_ids(model) -> Set[int]:
    eos = getattr(model.config, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {int(eos)}
    if isinstance(eos, (list, tuple, set)):
        return {int(x) for x in eos}
    return set()


# -----------------------------
# Constrained generation
# -----------------------------

def _get_sp_prefix_tokens(tokenizer) -> Set[int]:
    """
    Get set of token IDs that represent SentencePiece-style space-prefixed tokens (▁word).
    """
    vocab = tokenizer.get_vocab()
    sp_tokens: Set[int] = set()
    for tok_str, tid in vocab.items():
        if tok_str.startswith("▁") or tok_str.startswith("\u2581"):
            sp_tokens.add(int(tid))
    return sp_tokens


@torch.inference_mode()
def constrained_generate_stream(
    model,
    tokenizer,
    prompt_input_ids: Sequence[int],
    max_new_tokens: int,
    token_trie: TokenTrie,
    can_end_word: Set[int],
    temperature: float = 0.0,
    top_k: int = 0,
    repetition_penalty: float = 1.2,
) -> Iterator[str]:
    """
    Stream decoded text pieces as we generate.

    Decoding strategy:
    - If temperature == 0: greedy (pick highest-prob allowed token)
    - Else: sample from the top_k allowed tokens (k defaults to all allowed candidates encountered)
      after applying temperature.

    Important: constraints are applied *before* sampling.
    """
    device = model.device
    eos_ids = _eos_token_ids(model)

    # For SentencePiece tokenizers: track tokens that need space prefix when decoded individually
    sp_prefix_tokens = _get_sp_prefix_tokens(tokenizer)

    # Track generated tokens for repetition penalty
    generated_tokens: List[int] = []

    # First pass: feed the whole prompt
    tokens = torch.tensor([list(prompt_input_ids)], dtype=torch.long, device=device)

    # Use DynamicCache if available; else use the model's native caching.
    kv = DynamicCache() if DynamicCache is not None else None

    # Pointer into the trie for the current "token we're spelling".
    tt_ptr: TokenTrie = token_trie

    for _ in range(max_new_tokens):
        out = model(input_ids=tokens, past_key_values=kv, use_cache=True)
        logits = out.logits[:, -1, :].squeeze(0)

        # Update cache for next step
        kv = out.past_key_values

        # Apply repetition penalty to already generated tokens
        if repetition_penalty != 1.0 and generated_tokens:
            for prev_token in set(generated_tokens):
                if logits[prev_token] > 0:
                    logits[prev_token] = logits[prev_token] / repetition_penalty
                else:
                    logits[prev_token] = logits[prev_token] * repetition_penalty

        # Move logits to CPU for cheap sorting/scanning
        logits_cpu = logits.detach().float().cpu()

        # We'll scan tokens from most-likely downward until we find something allowed.
        # This is simple and robust but not the fastest approach.
        sorted_ids = torch.argsort(logits_cpu, descending=True)

        chosen: Optional[int] = None

        if temperature and temperature > 0:
            # Collect a small candidate set of allowed tokens from the top of the ranking,
            # then sample among them.
            allowed_candidates: List[int] = []
            allowed_logits: List[float] = []

            for tid in sorted_ids.tolist():
                tid = int(tid)
                # Primary path: continue along trie (spelling the current allowed token)
                if tt_ptr.get(tid) is not None:
                    allowed_candidates.append(tid)
                    allowed_logits.append(float(logits_cpu[tid]))
                else:
                    # If we're at a word boundary, we can start a new token or emit punctuation/whitespace/eos.
                    if not (tt_ptr.is_terminal or tt_ptr is token_trie):
                        continue

                    if token_trie.is_leading_space_tok(tid):
                        allowed_candidates.append(tid)
                        allowed_logits.append(float(logits_cpu[tid]))
                    elif tid in can_end_word:
                        allowed_candidates.append(tid)
                        allowed_logits.append(float(logits_cpu[tid]))
                    elif tid in eos_ids:
                        allowed_candidates.append(tid)
                        allowed_logits.append(float(logits_cpu[tid]))

                if top_k and len(allowed_candidates) >= top_k:
                    break
                # Safety cap to avoid pathological slowdowns
                if len(allowed_candidates) >= 512 and not top_k:
                    break

            if not allowed_candidates:
                # Nothing found (very restrictive vocab). Try to end.
                if eos_ids:
                    chosen = next(iter(eos_ids))
                else:
                    return
            else:
                # Sample proportional to exp(logit/temperature)
                logits_tensor = torch.tensor(allowed_logits, dtype=torch.float32)
                logits_tensor = logits_tensor / float(temperature)
                probs = torch.softmax(logits_tensor, dim=0)
                idx = int(torch.multinomial(probs, num_samples=1).item())
                chosen = int(allowed_candidates[idx])
        else:
            # Greedy selection
            for tid in sorted_ids.tolist():
                tid = int(tid)
                # Continue within trie?
                nxt = tt_ptr.get(tid)
                if nxt is not None:
                    tt_ptr = nxt
                    chosen = tid
                    break

                # Only allow starting a new token / punctuation / eos if we're not mid-token.
                if not (tt_ptr.is_terminal or tt_ptr is token_trie):
                    continue

                if token_trie.is_leading_space_tok(tid):
                    tt_ptr = token_trie.get(tid)  # type: ignore[assignment]
                    chosen = tid
                    break
                if tid in can_end_word:
                    tt_ptr = token_trie
                    chosen = tid
                    break
                if tid in eos_ids:
                    return

        if chosen is None:
            return

        # Track for repetition penalty
        generated_tokens.append(chosen)

        # Advance trie pointer if we chose a token that continues a token-in-progress
        # or started a new space-prefixed token.
        if temperature and temperature > 0:
            nxt = tt_ptr.get(chosen)
            if nxt is not None:
                tt_ptr = nxt
            else:
                if (tt_ptr.is_terminal or tt_ptr is token_trie) and token_trie.is_leading_space_tok(chosen):
                    tt_ptr = token_trie.get(chosen)  # type: ignore[assignment]
                elif chosen in can_end_word:
                    tt_ptr = token_trie

        # Prepare next-step input (single token)
        tokens = torch.tensor([[chosen]], dtype=torch.long, device=device)

        # Decode the token, prepending space if it's a SentencePiece ▁-prefixed token
        decoded = tokenizer.decode([chosen])
        if chosen in sp_prefix_tokens:
            decoded = " " + decoded
        yield decoded


# -----------------------------
# Prompt building (chat + fallback)
# -----------------------------

def build_prompt_input_ids(tokenizer, messages: List[dict]) -> List[int]:
    """
    Prefer tokenizer.apply_chat_template if available; otherwise fallback to a simple
    "User: ...\nAssistant: ..." format.
    """
    # Try chat template
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            # Newer HF versions: tokenize=True returns token IDs.
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            # Some versions return a tensor; normalize to list[int]
            if isinstance(ids, torch.Tensor):
                return ids.tolist()
            return list(ids)
        except Exception:
            pass

    # Fallback: plain text prompt
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")
    lines.append("Assistant:")

    prompt = "\n".join(lines)
    enc = tokenizer(prompt, add_special_tokens=False)
    return list(enc["input_ids"])


# -----------------------------
# Main CLI
# -----------------------------

def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    # Prefer MPS on macOS if available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(dtype_str: str, device: str) -> torch.dtype:
    dtype_str = dtype_str.lower()
    if dtype_str in ("auto", ""):
        if device in ("cuda", "mps"):
            return torch.float16
        return torch.float32
    if dtype_str in ("fp16", "float16", "half"):
        return torch.float16
    if dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_str in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Chat with an LLM constrained to a .txt file vocabulary.")
    ap.add_argument("--model", default="./YandexGPT-5-Lite-8B-instruct", help="HuggingFace model name or local path (default: ./YandexGPT-5-Lite-8B-instruct)")
    ap.add_argument("--txt", required=True, help="Path to a plaintext corpus (.txt)")
    ap.add_argument("--encoding", default="utf-8", help="Text file encoding (default: utf-8)")
    ap.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens to generate per turn")
    ap.add_argument("--min-token-len", type=int, default=1, help="Drop corpus tokens shorter than this")
    ap.add_argument("--max-unique-tokens", type=int, default=0, help="Optional cap on unique corpus tokens (0 = no cap)")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"], help="Device selection")
    ap.add_argument("--dtype", default="auto", help="auto|fp16|bf16|fp32")
    ap.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to HF loaders")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; >0 enables sampling")
    ap.add_argument("--top-k", type=int, default=0, help="When sampling: sample among top-k allowed tokens (0 = heuristic cap)")
    ap.add_argument("--system", default="", help="Optional system prompt (recommended: explain the restriction)")
    args = ap.parse_args()

    if not os.path.exists(args.txt):
        print(f"ERROR: txt file not found: {args.txt}", file=sys.stderr)
        return 2

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)

    print(f"[info] device={device} dtype={dtype} model={args.model!r}")
    print(f"[info] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    print(f"[info] loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    model.to(device)

    print(f"[info] building trie from {args.txt!r} ...")
    max_unique = None if args.max_unique_tokens in (0, None) else int(args.max_unique_tokens)
    trie, used_subtoken_ids = build_trie_from_txt(
        tokenizer,
        args.txt,
        encoding=args.encoding,
        min_token_len=int(args.min_token_len),
        max_unique_tokens=max_unique,
    )
    can_end = tokens_that_can_end_word(tokenizer, allowed_subtoken_ids=used_subtoken_ids)

    print(f"[info] unique corpus tokens added: {len(used_subtoken_ids)} subtoken-ids (incl. space-prefixed variants).")
    print(f"[info] boundary tokens allowed: {len(can_end)}")
    print()

    messages: List[dict] = []
    if args.system.strip():
        messages.append({"role": "system", "content": args.system.strip()})

    print("Type your message. Ctrl+C to quit.\n")

    try:
        while True:
            q = input("> ").strip()
            if not q:
                continue
            messages.append({"role": "user", "content": q})

            prompt_ids = build_prompt_input_ids(tokenizer, messages)

            # Add an empty assistant message; we will fill it as we stream.
            messages.append({"role": "assistant", "content": ""})

            response_parts: List[str] = []
            for piece in constrained_generate_stream(
                model=model,
                tokenizer=tokenizer,
                prompt_input_ids=prompt_ids,
                max_new_tokens=int(args.max_new_tokens),
                token_trie=trie,
                can_end_word=can_end,
                temperature=float(args.temperature),
                top_k=int(args.top_k),
            ):
                response_parts.append(piece)
                # Stream to terminal
                print(piece, end="", flush=True)

            resp = "".join(response_parts)
            print()  # newline after assistant turn
            messages[-1]["content"] = resp

    except (KeyboardInterrupt, EOFError):
        print("\n[bye]")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
