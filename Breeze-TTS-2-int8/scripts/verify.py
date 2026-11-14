#!/usr/bin/env python
"""End-to-end verification of the int8 Breeze-TTS-2 repo on MPS (or CPU/CUDA).

Loads the quantised checkpoint with stock transformers from_pretrained via the
bundled inference code, renders a voice-clone request twice (cold, then warm),
reports wall time and RTF for the warm render, and optionally transcribes the
output with an external command for a word-match check.

Usage:
    python scripts/verify.py [repo_dir] \
        --ref-audio ref.wav --ref-text "..." --text "..." \
        --output out.wav [--device mps] [--seed 42] \
        [--transcribe-cmd "/path/to/transcriber {wav}"]
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


def render(runtime, tokenizer, audio_tokenizer, model, args, set_all_seeds, prepare_inputs, get_template):
    request = {
        "id": "verify",
        "text": args.text,
        "instruction": "Speak clearly and naturally.",
        "speaker": "S0",
        "ref_audio_path": str(args.ref_audio),
        "ref_text": args.ref_text.strip(),
    }
    set_all_seeds(args.seed)
    inputs = prepare_inputs(
        tokenizer,
        audio_tokenizer,
        model,
        [request],
        get_template("ref_edit_tata"),
        guidance_scale=1.0,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )
    chunks = []
    for chunk in runtime.iter_audio_chunks(inputs, request_id="verify"):
        chunks.append(chunk.audio)
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    ap.add_argument("--ref-audio", type=Path, required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--transcribe-cmd", default=None, help="command template, {wav} replaced with output path")
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo))

    import numpy as np
    import soundfile as sf

    from breeze_infer.runtime import (
        load_runtime,
        resolve_dtype,
        set_all_seeds,
        update_generation_config_for_breeze,
    )
    from breeze_infer.templates import get_template, prepare_inputs
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

    dtype = resolve_dtype(args.device, "bfloat16")
    print(f"loading {args.repo} on {args.device} ({dtype}) ...")
    t0 = time.perf_counter()
    tokenizer, model, audio_tokenizer = load_runtime(
        args.repo, device=args.device, attn_implementation="eager", dtype=dtype
    )
    update_generation_config_for_breeze(model)
    n_int8 = sum(1 for _, p in model.named_parameters() if type(p).__name__ == "Int8Tensor")
    print(f"loaded in {time.perf_counter() - t0:.1f}s, {n_int8} int8 weights")
    if n_int8 == 0:
        raise SystemExit("FAIL: no quantised weights found - wrong checkpoint?")

    config = FastStreamingConfig(max_new_tokens=1500, max_seq_len=2048, repetition_penalty=1.1)
    runtime = FastBreezeStreamingRuntime(model, audio_tokenizer, config, tokenizer=tokenizer)

    print("cold render ...")
    render(runtime, tokenizer, audio_tokenizer, model, args, set_all_seeds, prepare_inputs, get_template)

    print("warm render ...")
    t1 = time.perf_counter()
    chunks = render(runtime, tokenizer, audio_tokenizer, model, args, set_all_seeds, prepare_inputs, get_template)
    wall = time.perf_counter() - t1

    audio = np.concatenate(chunks)
    duration = len(audio) / runtime.sample_rate
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, runtime.sample_rate, subtype="PCM_16")
    print(f"saved {args.output}")
    print(f"duration {duration:.2f}s, warm wall {wall:.1f}s, RTF {wall / duration:.2f}")

    if args.transcribe_cmd:
        cmd = args.transcribe_cmd.replace("{wav}", str(args.output))
        out = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=300)
        transcript = out.stdout.strip()
        print(f"transcription: {transcript}")
        target = {w.strip(".,:;!?'\"").lower() for w in args.text.split()}
        got = {w.strip(".,:;!?'\"").lower() for w in transcript.split()}
        matched = len(target & got)
        print(f"word match: {matched}/{len(target)}")
        if matched < 0.8 * len(target):
            raise SystemExit("FAIL: transcription does not match target text")


if __name__ == "__main__":
    main()
