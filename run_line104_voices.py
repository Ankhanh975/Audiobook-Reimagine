from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "Breeze-TTS-2-int8"
INFERENCE_REPO = MODEL
NOVEL = ROOT / "novels" / "The Possessed.txt"
OUTPUT = MODEL / "line-104-voices-cfg2"
CFG_SCALE = "2"

VOICE_DESCRIPTIONS = {
    "affectionate-literary-narrator": (
        "A refined, warm literary narrator with gentle affection for the "
        "characters, clear diction, measured pacing, and subtle irony."
    ),
    "dry-satirical-observer": (
        "A dry, observant satirical narrator with restrained amusement, crisp "
        "articulation, understated comic timing, and an intelligent skeptical tone."
    ),
    "grand-theatrical-patriot": (
        "A grand, theatrical male patriot who takes his public role very "
        "seriously, speaking with proud conviction, dramatic emphasis, and "
        "slightly excessive self-importance."
    ),
    "scholarly-storyteller": (
        "A cultured older professor telling an anecdote to an attentive "
        "audience, thoughtful and articulate, with deliberate pacing and "
        "nuanced historical gravitas."
    ),
    "warm-amused-witness": (
        "A warm, humane eyewitness who respects the subject but cannot hide a "
        "fond smile at his vanity, conversational, expressive, and gently playful."
    ),
}


def main() -> None:
    lines = NOVEL.read_text(encoding="utf-8").splitlines()
    text = lines[103]
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for name, description in VOICE_DESCRIPTIONS.items():
        target = OUTPUT / f"{name}.wav"
        command = [
            "py",
            "-3.10",
            "infer.py",
            str(MODEL),
            "--text",
            text,
            "--instruction",
            description,
            "--cfg-scale",
            CFG_SCALE,
            "--no-fast-all",
            "--output",
            str(target),
        ]
        subprocess.run(command, cwd=INFERENCE_REPO, check=True)
        print(f"created: {target}")


if __name__ == "__main__":
    main()
