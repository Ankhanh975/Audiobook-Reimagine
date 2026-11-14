# Audiobook Reimagine

Turn a classic novel into a cinematic YouTube experience.

**Audiobook Reimagine** is an experimental project for transforming public-domain
literature into narrated, illustrated videos. Instead of producing only an
audiobook, the project approaches each story like a film director: interpret the
text, design a visual language, cast voices, shape the pacing, and assemble the
result into a watchable scene-by-scene production.

## Vision

Given a novel, the pipeline should:

1. **Understand the story** — identify chapters, scenes, characters, settings,
   dialogue, mood, and important visual moments.
2. **Direct the adaptation** — create a scene treatment, shot list, narration
   script, character descriptions, and visual continuity rules.
3. **Create the voices** — generate expressive narration and character dialogue
   with consistent voices and delivery.
4. **Create the images** — generate cinematic establishing shots, character
   moments, locations, and transitions that match the direction.
5. **Produce the video** — synchronize narration, images, music, sound design,
   subtitles, and pacing into a polished YouTube episode.

The goal is not to replace the original novel. It is to create a new audiovisual
interpretation that helps an audience experience a classic story.

## Current status

This repository is an early prototype. The current working capability is local
text-to-speech on Windows using the Voxtral runtime. The image-generation,
directing, editing, and final video-export stages are planned parts of the
project.

The included source novel is **The Possessed** by Fyodor Dostoevsky, translated
by Constance Garnett and published in 1916:

- [`novels/The Possessed.txt`](novels/The%20Possessed.txt)

## Repository layout

```text
.
├── novels/
│   └── The Possessed.txt       # Source literature
├── Voxtral TTS/                # Local Windows narration experiments
│   ├── voxtral_tts.py          # Python synthesis interface
│   ├── run_voxtral_tts.ps1    # Interactive PowerShell runner
│   └── outputs/                # Generated WAV files
├── breeze-tts/                 # Breeze TTS inference source
└── Breeze-TTS-2/               # Local Breeze TTS model assets and samples
```

## Quick start: generate narration

### Requirements

- Windows
- Python 3.10 or newer
- PowerShell
- A compatible NVIDIA CUDA setup
- The bundled `crispasr-windows-x86_64-cuda` runtime

The Voxtral runtime may download its model on the first run. Model caches and
generated audio are kept under `Voxtral TTS/`.

### Python API

From the `Voxtral TTS` directory:

```powershell
cd "Voxtral TTS"
python -c "from voxtral_tts import synthesize; print(synthesize('The story begins in a quiet city.', voice='neutral_female'))"
```

Or use Python directly:

```python
from voxtral_tts import synthesize

output = synthesize(
    "The story begins in a quiet city.",
    voice="neutral_female",
    output_path="outputs/scene-001.wav",
)
print(output)
```

### PowerShell runner

Run the interactive prompt:

```powershell
cd "Voxtral TTS"
.\run_voxtral_tts.ps1
```

The runner writes WAV files to `Voxtral TTS/outputs/`. It can also generate a
number-reading test in 50-number chunks:

```powershell
.\run_voxtral_tts.ps1 -ReadOneToThousand
```

Available voice presets include `neutral_female`, `neutral_male`,
`casual_female`, `casual_male`, `cheerful_female`, and multilingual presets.
See [`Voxtral TTS/README.md`](Voxtral%20TTS/README.md) for runtime details.

## Planned production workflow

The intended end-to-end workflow is:

```text
Novel
  -> chapter and scene breakdown
  -> director's treatment
  -> narration and dialogue script
  -> voice generation
  -> image and shot generation
  -> timeline assembly
  -> subtitles, music, and sound design
  -> YouTube-ready video
```

Each scene should have a reproducible manifest containing its source passage,
narration text, cast, voice direction, image prompts, shot duration, aspect
ratio, and output files. This will make it possible to revise a scene without
regenerating an entire episode.

## Design principles

- **Respect the source** — preserve the author's meaning while making adaptation
  choices explicit.
- **Direct for the screen** — use composition, movement, pacing, lighting, and
  sound rather than presenting a sequence of unrelated images.
- **Maintain continuity** — keep characters, locations, costumes, palette, and
  visual style consistent across scenes.
- **Prefer local and reproducible tools** — keep inputs, prompts, manifests, and
  outputs organized so productions can be repeated and improved.
- **Publish responsibly** — identify generated material and respect copyright,
  voice, and model licenses.

## Responsible use

Only use texts, images, voices, music, and other assets that you have the right
to use. Public-domain status can vary by country and by translation. Verify the
rights for the specific edition and translation before publishing. Also review
the licenses for the included TTS runtimes, model weights, and generated
outputs; some models are restricted to research or non-commercial use.

## Roadmap

- [ ] Define a scene and shot manifest format
- [ ] Build chapter and scene extraction tools
- [ ] Add director-style narration and visual treatments
- [ ] Add consistent character and location references
- [ ] Add image-generation adapters
- [ ] Add timeline assembly with subtitles and audio mixing
- [ ] Add YouTube episode packaging and metadata
- [ ] Produce a complete pilot episode

## License

This repository contains experimental integration code and third-party model
assets. Check each component's license before using or redistributing it:

- [`Voxtral TTS/LICENSE`](Voxtral%20TTS/LICENSE)
- [`breeze-tts/LICENSE`](breeze-tts/LICENSE)
- [`Breeze-TTS-2/LICENSE`](Breeze-TTS-2/LICENSE)
