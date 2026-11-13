"""Local Python interface for the bundled Voxtral TTS runtime."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNTIME_DIR = PROJECT_ROOT / "crispasr-windows-x86_64-cuda"
BINARY = RUNTIME_DIR / "crispasr.exe"


def synthesize(
    text: str,
    output_path: Optional[str | Path] = None,
    voice: str = "neutral_female",
    seed: int = 42,
) -> Path:
    """Generate a WAV file from text using the local Voxtral runtime."""
    if not text.strip():
        raise ValueError("text must not be empty")
    if not BINARY.is_file():
        raise FileNotFoundError(f"Voxtral runtime not found: {BINARY}")

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    target = (
        Path(output_path)
        if output_path is not None
        else OUTPUTS_DIR / f"voxtral-{datetime.now():%Y%m%d-%H%M%S}.wav"
    )
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(BINARY),
        "--backend",
        "voxtral-tts",
        "--cache-dir",
        str(MODELS_DIR),
        "-m",
        "auto",
        "--auto-download",
        "--accept-license",
        "cc-by-nc-4.0",
        "--tts",
        text,
        "--voice",
        voice,
        "--seed",
        str(seed),
        "--tts-output",
        str(target),
    ]
    result = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"Voxtral TTS failed with exit code {result.returncode}: {details}"
        )
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f"Voxtral TTS did not create a valid output: {target}")
    return target


__all__ = ["synthesize"]


VOICE_PARAGRAPHS = {
    "neutral_female": "The morning light moved softly across the room, and the quiet city began another ordinary day.",
    "neutral_male": "The morning light moved softly across the room, and the quiet city began another ordinary day.",
    "casual_female": "I took a slow walk through the neighborhood, listening to the birds and enjoying the fresh air.",
    "casual_male": "I took a slow walk through the neighborhood, listening to the birds and enjoying the fresh air.",
    "cheerful_female": "What a wonderful day to explore something new, share a few laughs, and make a happy memory.",
    "fr_female": "La lumière du matin glissait doucement dans la pièce, et la ville commençait une nouvelle journée.",
    "fr_male": "La lumière du matin glissait doucement dans la pièce, et la ville commençait une nouvelle journée.",
    "de_female": "Das Morgenlicht fiel sanft durch das Fenster, und die Stadt erwachte zu einem neuen Tag.",
    "de_male": "Das Morgenlicht fiel sanft durch das Fenster, und die Stadt erwachte zu einem neuen Tag.",
    "es_female": "La luz de la mañana entraba suavemente por la ventana, y la ciudad despertaba poco a poco.",
    "es_male": "La luz de la mañana entraba suavemente por la ventana, y la ciudad despertaba poco a poco.",
    "it_female": "La luce del mattino entrava dolcemente dalla finestra, mentre la città iniziava una nuova giornata.",
    "it_male": "La luce del mattino entrava dolcemente dalla finestra, mentre la città iniziava una nuova giornata.",
    "pt_female": "A luz da manhã entrava suavemente pela janela, enquanto a cidade começava um novo dia.",
    "pt_male": "A luz da manhã entrava suavemente pela janela, enquanto a cidade começava um novo dia.",
    "nl_female": "Het zachte ochtendlicht viel door het raam terwijl de stad langzaam aan een nieuwe dag begon.",
    "nl_male": "Het zachte ochtendlicht viel door het raam terwijl de stad langzaam aan een nieuwe dag begon.",
    "ar_male": "كان ضوء الصباح يتسلل بهدوء عبر النافذة، وكانت المدينة تستيقظ ليوم جديد.",
    "hi_female": "सुबह की रोशनी खिड़की से धीरे-धीरे अंदर आई और शहर एक नए दिन के लिए जाग उठा।",
    "hi_male": "सुबह की रोशनी खिड़की से धीरे-धीरे अंदर आई और शहर एक नए दिन के लिए जाग उठा।",
}


def test_all_voices(output_dir: Optional[str | Path] = None) -> list[Path]:
    """Generate a natural-language sample for every bundled preset voice."""
    destination = Path(output_dir) if output_dir is not None else OUTPUTS_DIR / "voice-tests"
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.mkdir(parents=True, exist_ok=True)

    outputs = []
    for voice, paragraph in VOICE_PARAGRAPHS.items():
        outputs.append(
            synthesize(
                paragraph,
                output_path=destination / f"{voice}.wav",
                voice=voice,
            )
        )
    return outputs


__all__.append("test_all_voices")
