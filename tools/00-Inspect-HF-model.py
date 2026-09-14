#!/usr/bin/env python3
"""
HuggingFace Model Inspector (pre-GGUF)

Inspect a HuggingFace model directory BEFORE converting it to GGUF.
Reveals tokenizer configuration, special tokens, chat template, and post-processing behaviour.
Useful to spot tokenizer bugs early (e.g. EOS tokens not marked as CONTROL, missing chat templates, add_bos/add_eos misconfiguration).

Files inspected (if present):

    - config.json
    - tokenizer_config.json
    - special_tokens_map.json
    - tokenizer.json

Usage:
    python3 00-Inspect-HF-model.py <path/to/model/dir>

"""

import json
import sys
from pathlib import Path


def load_json(path):

    """Load eacyh JSON file, returning an error string if it fails."""

    try:

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except FileNotFoundError:
        return None
    except Exception as e:
        return f"[ERROR] {e}"


def section(title):
    print(f"\n--- {title} ---")


def inspect_config(p):

    cfg = load_json(p / "config.json")
    section("config.json")

    if cfg is None:
        print("  [not found]")
        return

    if not isinstance(cfg, dict):
        print(f"  {cfg}")
        return

    keys = (
        "model_type", "architectures", "vocab_size",
        "hidden_size", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "max_position_embeddings", "rope_theta",
        "torch_dtype", "tie_word_embeddings",
        "bos_token_id", "eos_token_id", "pad_token_id",
    )

    for k in keys:
        if k in cfg:
            print(f"  {k} = {cfg[k]}")


def inspect_tokenizer_config(p):

    tok = load_json(p / "tokenizer_config.json")
    section("tokenizer_config.json")

    if tok is None:

        print("  [not found]")
        return

    if not isinstance(tok, dict):

        print(f"  {tok}")
        return

    keys = (
        "tokenizer_class", "model_max_length",
        "bos_token", "eos_token", "unk_token", "pad_token",
        "add_bos_token", "add_eos_token",
        "clean_up_tokenization_spaces",
        "legacy", "split_special_tokens",
    )

    for k in keys:
        if k in tok:
            print(f"  {k} = {tok[k]}")


    # For chat template:    show whether it exists and a short preview
    ct = tok.get("chat_template")

    if ct is None:

        print("  chat_template = [NOT PRESENT]  <-- must supply your own")

    elif isinstance(ct, str):

        preview = ct[:160].replace("\n", "\\n")
        print(f"  chat_template = {preview}{'...' if len(ct) > 160 else ''}")

    elif isinstance(ct, list):

        print(f"  chat_template = [list with {len(ct)} entries]")

        for i, entry in enumerate(ct):

            name = entry.get("name", "?") if isinstance(entry, dict) else "?"
            print(f"    [{i}] name={name}")


def inspect_special_tokens_map(p):

    stm = load_json(p / "special_tokens_map.json")
    section("special_tokens_map.json")

    if stm is None:

        rint("  [not found]")
        return

    if not isinstance(stm, dict):

        rint(f"  {stm}")
        return

    for k, v in stm.items():

        print(f"  {k} = {v}")
        # return

def inspect_tokenizer_json(p):

    tok = load_json(p / "tokenizer.json")
    section("tokenizer.json (summary)")

    if tok is None:

        print("  [not found]")
        return

    if not isinstance(tok, dict):

        print(f"  {tok}")
        return



    # Special / added tokens
    added = tok.get("added_tokens", [])
    special = [t for t in added if t.get("special", False)]

    print(f"  added_tokens: {len(added)} total, {len(special)} marked special")

    for t in added:

        marker = "SPECIAL" if t.get("special") else "USER_DEF"

        print(
            f"    id={t.get('id'):>6}  content={t.get('content')!r:<20}  "
            f"type={marker:<9}  "
            f"lstrip={t.get('lstrip')}  rstrip={t.get('rstrip')}  "
            f"normalized={t.get('normalized')}"
        )



    # Base model
    model = tok.get("model", {}) or {}

    print(f"  model.type     = {model.get('type')}")

    vocab = model.get("vocab", {}) or {}

    print(f"  vocab_size     = {len(vocab)}")



    # Pre-tokenizer / decoder
    pre = tok.get("pre_tokenizer", {}) or {}

    print(f"  pre_tokenizer  = {pre.get('type')}")

    dec = tok.get("decoder", {}) or {}

    print(f"  decoder        = {dec.get('type')}")



    # Normalizer
    norm = tok.get("normalizer", {}) or {}

    print(f"  normalizer     = {norm.get('type')}")



    # Post-processor (BOS/EOS handling)
    post = tok.get("post_processor", {}) or {}

    print(f"  post_processor = {post.get('type')}")



    if post.get("type") == "TemplateProcessing":

        print(f"    single = {post.get('single')}")
        print(f"    pair   = {post.get('pair')}")




def list_files(p):

    section("Files in directory")

    for f in sorted(p.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name:<45} {size_mb:>8.2f} MB")


def inspect(model_dir):

    p = Path(model_dir)

    if not p.is_dir():

        print(f"[ERROR] Directory not found: {p}")
        sys.exit(1)

    print(f"=== HuggingFace model inspection: {p} ===")

    inspect_config(p)
    inspect_tokenizer_config(p)
    inspect_special_tokens_map(p)
    inspect_tokenizer_json(p)
    list_files(p)


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("Usage: python3 00-Inspect-HF-model.py <path/to/model/dir>")
        sys.exit(1)

    inspect(sys.argv[1])
