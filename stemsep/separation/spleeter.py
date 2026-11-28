"""Backend Spleetera uruchamiany jako podproces w osobnym środowisku.

Spleeter wymaga starego TensorFlow, który nie może współistnieć z PyTorchem w jednym
środowisku, dlatego separacja odbywa się w backends/spleeter/worker.py, a wyniki
wracają przez pliki .npy w katalogu tymczasowym.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from stemsep.separation.base import BaseSeparator, SeparationResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backends" / "spleeter"
WORKER = BACKEND_DIR / "worker.py"
# Wagi trzymamy w cache użytkownika, a nie w repozytorium (podobnie jak Demucs)
MODEL_CACHE = Path.home() / ".cache" / "spleeter"

STEMS = {
    "2stems": ["vocals", "accompaniment"],
    "4stems": ["vocals", "drums", "bass", "other"],
    "5stems": ["vocals", "drums", "bass", "piano", "other"],
}


class SpleeterSeparatorError(RuntimeError):
    """Błąd zgłaszany, gdy podproces Spleetera zakończy się niepowodzeniem."""


class SpleeterSeparator(BaseSeparator):
    """Separacja pretrenowanym modelem Spleeter (Deezer) przez osobny proces."""

    def __init__(self, variant: str = "4stems"):
        if variant not in STEMS:
            raise ValueError(f"Unknown Spleeter variant {variant!r}, choose from {list(STEMS)}")
        self.variant = variant
        self.name = f"spleeter-{variant}"

    @property
    def stem_names(self) -> list[str]:
        return list(STEMS[self.variant])

    @property
    def sample_rate(self) -> int:
        return 44100

    def separate(self, input_path: Path) -> SeparationResult:
        with tempfile.TemporaryDirectory(prefix="stemsep-spleeter-") as tmp:
            out_dir = Path(tmp)
            self._run_worker(Path(input_path).resolve(), out_dir)
            return self._read_output(out_dir)

    def separate_waveform(self, audio: np.ndarray, sample_rate: int) -> SeparationResult:
        # Worker przyjmuje tylko pliki, więc zapisujemy tablicę do tymczasowego WAV
        # (float, bez strat); resampling i stereo załatwia worker.
        audio = np.atleast_2d(audio)
        with tempfile.TemporaryDirectory(prefix="stemsep-spleeter-") as tmp:
            input_path = Path(tmp) / "input.wav"
            sf.write(input_path, audio.T, sample_rate, subtype="FLOAT")
            out_dir = Path(tmp) / "out"
            self._run_worker(input_path, out_dir)
            return self._read_output(out_dir)

    def _run_worker(self, input_path: Path, out_dir: Path) -> None:
        if shutil.which("uv") is None:
            raise SpleeterSeparatorError("uv is required to run the Spleeter backend")

        env = os.environ.copy()
        # VIRTUAL_ENV głównego projektu myliłby uv przy wyborze środowiska Spleetera
        env.pop("VIRTUAL_ENV", None)
        env["MODEL_PATH"] = str(MODEL_CACHE)
        # Wycisza logi TensorFlow (zostawia tylko błędy)
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        cmd = [
            "uv",
            "run",
            "--project",
            str(BACKEND_DIR),
            "python",
            str(WORKER),
            self.variant,
            str(input_path),
            str(out_dir),
        ]
        # check=False: kod wyjścia sprawdzamy sami, żeby dołączyć stderr do błędu
        proc = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            # Ostatnie linie stderr zwykle zawierają właściwą przyczynę błędu
            tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
            raise SpleeterSeparatorError(f"Spleeter worker failed:\n{tail}")

    def _read_output(self, out_dir: Path) -> SeparationResult:
        meta = json.loads((out_dir / "meta.json").read_text())
        stems = {name: np.load(out_dir / f"{name}.npy") for name in meta["stems"]}
        return SeparationResult(stems=stems, sample_rate=meta["sample_rate"])
