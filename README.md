# Audiobook Reimagine

Local text-to-speech experiments using the Voxtral TTS runtime on Windows.

## Voxtral TTS

The implementation is in [`Voxtral TTS/voxtral_tts.py`](<Voxtral TTS/voxtral_tts.py>). It uses the bundled `crispasr.exe` runtime, downloads the required model into `Voxtral TTS/models`, and writes WAV files to `Voxtral TTS/outputs`.

### Requirements

- Windows
- Python 3.10 or newer
- PowerShell
- The bundled `crispasr-windows-x86_64-cuda` runtime and a compatible NVIDIA CUDA setup

The first run may download a model. The runtime accepts the `cc-by-nc-4.0` license automatically through the command used by this project.

### Python usage

Run this from the `Voxtral TTS` directory:

```powershell
python -c "from voxtral_tts import synthesize; print(synthesize('Hello from Voxtral TTS.'))"
```

You can select a voice and output path:

```python
from voxtral_tts import synthesize

output = synthesize(
    "This is a generated audiobook sample.",
    voice="neutral_female",
    output_path="outputs/sample.wav",
)
print(output)
```

Available bundled voices include `neutral_female`, `neutral_male`, `casual_female`, `casual_male`, `cheerful_female`, and multilingual presets.

### PowerShell usage

Run the interactive prompt:

```powershell
.\run_voxtral_tts.ps1
```

Generate numbers one through one thousand in 50-number WAV chunks:

```powershell
.\run_voxtral_tts.ps1 -ReadOneToThousand
```

Generated models, caches, and audio outputs are excluded from version control by [`.gitignore`](.gitignore).
