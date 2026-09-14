#!/usr/bin/env python3
"""
GGUF Model Inspector (post-conversion)

Inspect a GGUF file to verify tokenizer metadata, special tokens, token types, and chat template AFTER converting from HuggingFace models.

Complements 00-Inspect-HF-model.py :
-   if something is wrong here (E.g EOS marked as USER_DEFINED instead of CONTROL), the bug came either from the source model or from the conversion step.

Usage:

    python3 01-Inspect-GGUF.py <path/to/model.gguf>

"""

import sys
from pathlib import Path

try:
    from gguf import GGUFReader

except ImportError:

    print("[ERROR] No se encontró el paquete 'gguf'.")
    print("Instálalo con: pip install gguf")

    sys.exit(1)


# GGUF token_type values (per spec)
TOKEN_TYPES = {
    1: "NORMAL",
    2: "UNKNOWN",
    3: "CONTROL",
    4: "USER_DEFINED",
    5: "UNUSED",
    6: "BYTE",
}


def section(title):
    print(f"\n--- {title} ---")


def inspect_metadata(reader):
    section("Metadata keys")

    for key in sorted(reader.fields.keys()):
        field = reader.fields[key]

        # Skip very long arrays in the summary
        try:
            types = field.types

        except Exception:
            types = None


        print(f"  {key}  (types={types})")


def inspect_special_tokens(reader):

    section("Special token IDs")

    keys = (
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
    )

    for k in keys:

        if k in reader.fields:

            try:
                print(f"  {k} = {reader.fields[k].contents()}")

            except Exception as e:
                print(f"  {k} = [ERROR: {e}]")


def inspect_token_types(reader):

    section("tokenizer.ggml.token_type")
    key = "tokenizer.ggml.token_type"

    if key not in reader.fields:

        print("  [NOT PRESENT]  <-- suspicious for modern models")
        return

    try:

        types = list(reader.fields[key].contents())

    except Exception as e:

        print(f"  [ERROR reading types: {e}]")
        return


    print(f"  total tokens: {len(types)}")
    # return

    # Distribution
    from collections import Counter

    dist = Counter(types)

    print("  distribution:")

    for t, count in sorted(dist.items()):

        label = TOKEN_TYPES.get(t, f"UNKNOWN({t})")
        print(f"    type={t} ({label:<12}) -> {count} tokens")

    # return

    # Show suspicious tokens: should be CONTROL but are not
    print("\n  Checking for EOS/BOS-like tokens not marked as CONTROL (3)...")

    try:

        tokens = list(reader.fields["tokenizer.ggml.tokens"].contents())

    except Exception:
        tokens = []

    suspects = []

    for i, t in enumerate(types):

        if i >= len(tokens):
            break
        tok = tokens[i]

        if not isinstance(tok, str):
            continue

        # Heuristic: tokens that *look* like control tokens
        looks_control = (
            tok.startswith("<") and tok.endswith(">")
            or tok in ("</s>", "<s>", "<eos>", "<bos>", "<pad>", "<unk>")
            or tok.startswith("<|") and tok.endswith("|>")
        )

        if looks_control and t != 3:

            suspects.append((i, tok, t))

    if suspects:

        for i, tok, t in suspects:

            label = TOKEN_TYPES.get(t, f"UNKNOWN({t})")

            print(f"    id={i:>6}  {tok!r:<25}  type={t} ({label})  <-- WRONG")

    else:

        print("    OK,  no suspicious tokens found.")


def inspect_chat_template(reader):

    section("Chat template")
    key = "tokenizer.chat_template"

    if key not in reader.fields:

        print("  [NOT PRESENT]  <-- llama.cpp/Ollama will use a default")
        return

    try:

        tmpl = reader.fields[key].contents()


        # It's usually a single-element list with the string
        if isinstance(tmpl, list) and tmpl:
            tmpl = tmpl[0]

        preview = str(tmpl)[:200].replace("\n", "\\n")

        print(f"  {preview}{'...' if len(str(tmpl)) > 200 else ''}")


    except Exception as e:

        print(f"  [ERROR: {e}]")


def inspect_tokens_sample(reader, n=20):

    section(f"First {n} tokens (id, type, content)")

    try:

        tokens = list(reader.fields["tokenizer.ggml.tokens"].contents())
        types = list(reader.fields["tokenizer.ggml.token_type"].contents())

    except Exception as e:

        print(f"  [ERROR: {e}]")
        return

    for i in range(min(n, len(tokens))):

        tok = tokens[i]
        t = types[i] if i < len(types) else -1
        label = TOKEN_TYPES.get(t, f"UNKNOWN({t})")

        print(f"  id={i:>6}  type={t} ({label:<12})  {tok!r}")


def inspect_file(path):

    print(f"=== GGUF inspection: {path} ===\n")

    reader = GGUFReader(path)

    print("--- Version ---")


    try:
        print(f"  gguf_version = {reader.header.get('version', '?')}")
        print(f"  n_tensors    = {reader.header.get('n_tensors', '?')}")
        print(f"  n_kv         = {reader.header.get('n_kv', '?')}")

    except Exception:
        pass

    inspect_metadata(reader)
    inspect_special_tokens(reader)
    inspect_chat_template(reader)
    inspect_token_types(reader)
    inspect_tokens_sample(reader, n=20)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python3 01-Inspect-GGUF.py <path/to/model.gguf>")
        sys.exit(1)

    p = Path(sys.argv[1])

    if not p.is_file():
        print(f"[ERROR] File not found: {p}")
        sys.exit(1)

    inspect_file(p)
