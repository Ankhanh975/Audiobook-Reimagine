---
base_model: BreezeBlue/Breeze-TTS-2
license: other
license_name: breezeblue-research-and-non-commercial-license
license_link: >-
  https://huggingface.co/BreezeBlue/Breeze-TTS-2/commit/a3bd0a6e83cd2d046ce783df2f7cb84292869ef7#d2h-740129
pipeline_tag: text-to-speech
language:
- en
tags:
- text-to-speech
- voice-cloning
- quantized
- int8
- torchao
---

# Breeze-TTS-2 int8

Int8 weight-only quantisation of [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2), a 3.5B voice-cloning TTS model. Official code: https://github.com/breezeblue-ai/breeze-tts

## What is quantised

- Quantised to int8 (torchao `Int8WeightOnlyConfig(version=2)`, per-row scales): all `nn.Linear` layers in the speech backbone (`backbone_model.layers`, Qwen3 architecture, 1.4B params) and the depth decoder (`depth_decoder.model`, 0.4B params) - 281 weights total
- Kept bf16: T5Gemma2 text encoder, Mimi codec (`codec_model`), text/audio embeddings, `lm_head`, `depth_decoder.codebooks_head`, all norms
- `audio_tokenizer/` is the unmodified bf16 copy from the base checkpoint

Weights: 5.2 GB vs 7.0 GB bf16. Per-layer mean relative error vs bf16 is ~1%.

## Format

Weights are pickle shards (`pytorch_model-*.bin`), not safetensors. transformers 4.57.3 only writes torchao safetensors for Float8 configs, and its reader expects the `:`-flattened key format of torchao <= 0.14 while torchao 0.15+ writes `._weight_qdata` keys - an int8 safetensors checkpoint written with this pairing cannot be read back. The `.bin` route keeps stock `from_pretrained` working. Loading a pickle checkpoint requires torchao to be installed (its tensor classes are registered as safe globals for `torch.load`).

## Requirements

Built and verified with torch 2.9.1, transformers 4.57.3, torchao 0.17.0, qwen-tts 0.1.1 (torchao 0.18+ needs torch >= 2.11). The inference code from the official repo is bundled (`models/`, `breeze_infer/`, `infer.py`, Apache-2.0 - see `LICENSE`); the checkpoint has no `trust_remote_code` wiring, so it must be loaded through that code.

```sh
pip install -r requirements.txt torchao==0.17.0
```

## Load

CLI (from the repo root):

```sh
python infer.py . \
  --text "The committee has voted: all meetings are now conducted underwater." \
  --ref-audio reference.wav --ref-text "transcript of the reference audio" \
  --device mps --dtype bfloat16 --seed 42 --output out.wav
```

Python (works on `mps`, `cpu` and `cuda`):

```python
import sys
from pathlib import Path

repo = Path("/path/to/Breeze-TTS-2-int8")
sys.path.insert(0, str(repo))

import torch
from breeze_infer.runtime import load_runtime, update_generation_config_for_breeze

tokenizer, model, audio_tokenizer = load_runtime(
    repo, device="mps", attn_implementation="eager", dtype=torch.bfloat16
)
update_generation_config_for_breeze(model)
```

`from_pretrained` picks up the `quantization_config` in `config.json` and restores the int8 tensors; no extra arguments needed.

## Verification (M5 Max, macOS, MPS)

`scripts/verify.py` renders a seed-42 voice clone twice (cold, warm) and transcribes the result:

- Transcription word match vs target text: 16/16
- 6.24 s of audio, warm wall time 28.3 s, RTF 4.5
- Repo on disk 5.5 GB (bf16 checkpoint: 7.0 GB weights + 0.7 GB audio tokenizer)

## MPS caveats

- torchao has no fused int8 matmul kernel on MPS; each linear dequantises to bf16 before the matmul. Generation is ~2.5x slower than the bf16 checkpoint (warm RTF ~4.5 vs ~1.8). The quantisation buys disk and memory, not speed, on Apple silicon.
- MPS is not an officially validated torchao platform; validated here empirically (see verification above).
- Outputs differ sample-for-sample from bf16 at the same seed (quantisation shifts sampled logits) but speaker cloning and text fidelity hold.

## Reproduce

```sh
python scripts/convert.py <bf16_snapshot_dir> <out_dir> --code-dir <breeze-tts checkout>
python scripts/verify.py <out_dir> --ref-audio ref.wav --ref-text "..." --text "..." --output out.wav
```

## Licence

Weights inherit the BreezeBlue Research and Non-Commercial License (`MODEL_LICENSE`). Bundled inference code is Apache-2.0 (`LICENSE`).
