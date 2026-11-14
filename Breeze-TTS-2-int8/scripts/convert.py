#!/usr/bin/env python
"""Convert BreezeBlue/Breeze-TTS-2 (bf16) to int8 weight-only via torchao.

Reads a local HF snapshot of the bf16 checkpoint, quantises the speech LLM
(backbone + depth decoder) to int8 weight-only, and writes an upload-ready
repo directory: quantised safetensors, config with quantization_config,
tokenizer files, the audio_tokenizer copied as-is, and the inference code
from the breeze-tts checkout.

Quantised: nn.Linear layers in backbone_model.layers and depth_decoder.model.
Not quantised (kept bf16): text_encoder (T5Gemma2), codec_model (Mimi),
embed_text_tokens, backbone_model.embed_tokens, lm_head,
depth_decoder.codebooks_head, all norms.

Requires: torch 2.9.x, transformers 4.57.3, torchao 0.17.0.

Serialisation is pickle (pytorch_model.bin), not safetensors: transformers
4.57.3 can only WRITE torchao safetensors for Float8 configs, and its READ
path expects the ':'-flattened key format from torchao <= 0.14, while
torchao 0.15+ flattens to '._weight_qdata' style keys - so an int8
safetensors checkpoint written today cannot be read back by this
transformers version. The .bin route keeps stock from_pretrained working.

Usage:
    python scripts/convert.py <bf16_snapshot_dir> <out_repo_dir> --code-dir <breeze-tts checkout>
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MODULES_TO_NOT_CONVERT = [
    "text_encoder",
    "text_encoder_proj",
    "codec_model",
    "embed_text_tokens",
    "lm_head",
    "backbone_model.embed_tokens",
    "depth_decoder.codebooks_head",
]

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
CODE_DIRS = ["models", "breeze_infer", "configs"]
CODE_FILES = ["infer.py", "requirements.txt", "LICENSE", "MODEL_LICENSE"]


def copy_code(code_dir: Path, out: Path) -> None:
    for name in CODE_DIRS:
        src = code_dir / name
        dst = out / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in CODE_FILES:
        src = code_dir / name
        if src.is_file():
            shutil.copy2(src, out / name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", type=Path, help="bf16 HF snapshot dir")
    ap.add_argument("out", type=Path, help="output repo dir")
    ap.add_argument(
        "--code-dir",
        type=Path,
        required=True,
        help="breeze-tts checkout providing the models/ and breeze_infer/ packages",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(args.code_dir))

    import torch
    from transformers import TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig

    from models.breeze import BreezeForConditionalGeneration

    quant_config = TorchAoConfig(
        quant_type=Int8WeightOnlyConfig(version=2),
        modules_to_not_convert=MODULES_TO_NOT_CONVERT,
    )

    print("loading + quantising on CPU ...")
    model = BreezeForConditionalGeneration.from_pretrained(
        args.snapshot,
        dtype=torch.bfloat16,
        quantization_config=quant_config,
    )
    model.eval()

    quantised = [
        n for n, p in model.named_parameters() if type(p).__name__ == "Int8Tensor"
    ]
    print(f"{len(quantised)} weights quantised to Int8Tensor")
    for prefix in ("backbone_model.", "depth_decoder."):
        n = sum(1 for q in quantised if q.startswith(prefix))
        print(f"  {prefix}* : {n}")
    bad = [q for q in quantised if not q.startswith(("backbone_model.layers.", "depth_decoder.model."))]
    if bad:
        raise SystemExit(f"unexpected quantised weights: {bad[:10]}")
    if not quantised:
        raise SystemExit("nothing was quantised")

    args.out.mkdir(parents=True, exist_ok=True)
    print("saving quantised checkpoint (pickle .bin, see module docstring) ...")
    model.save_pretrained(args.out, safe_serialization=False, max_shard_size="5GB")

    for name in TOKENIZER_FILES:
        shutil.copy2(args.snapshot / name, args.out / name)

    dst_tok = args.out / "audio_tokenizer"
    if not dst_tok.exists():
        print("copying audio_tokenizer ...")
        # snapshot files are symlinks into the HF blob store; dereference them
        shutil.copytree(args.snapshot / "audio_tokenizer", dst_tok, symlinks=False)

    copy_code(args.code_dir, args.out)
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
