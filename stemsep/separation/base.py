"""Wspólny interfejs dla wszystkich backendów separacji (Demucs, Spleeter, własny model)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class SeparationResult:
    """Wynik separacji: stemy jednego utworu i ich częstotliwość próbkowania."""

    stems: dict[str, np.ndarray]  # nazwa stemu -> tablica o kształcie (kanały, próbki)
    sample_rate: int

    def save(self, out_dir: Path) -> dict[str, Path]:
        """Zapisuje każdy stem jako 32-bitowy WAV float (bez przesterowania i kwantyzacji)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, audio in self.stems.items():
            path = out_dir / f"{name}.wav"
            sf.write(path, audio.T, self.sample_rate, subtype="FLOAT")
            paths[name] = path
        return paths


class BaseSeparator(ABC):
    """Kontrakt dla modeli separacji; CLI, API i ewaluacja nie zależą od konkretnego modelu."""

    name: str

    @property
    @abstractmethod
    def stem_names(self) -> list[str]:
        """Nazwy stemów zwracanych przez model, np. ["drums", "bass", "other", "vocals"]."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Częstotliwość próbkowania modelu; wejście jest do niej resamplowane."""

    @abstractmethod
    def separate_waveform(self, audio: np.ndarray, sample_rate: int) -> SeparationResult:
        """Separuje miks z pamięci o kształcie (kanały, próbki)."""

    @abstractmethod
    def separate(self, input_path: Path) -> SeparationResult:
        """Separuje plik audio (dowolny format obsługiwany przez ffmpeg)."""
