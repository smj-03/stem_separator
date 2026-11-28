from pathlib import Path

from demucs.apply import apply_model
from demucs.audio import AudioFile, convert_audio
from demucs.pretrained import get_model
import numpy as np
import torch

from stemsep.separation.base import BaseSeparator, SeparationResult


def default_device() -> str:
    """Wybiera domyślne urządzenie: CUDA, jeśli jest dostępna, w przeciwnym razie CPU."""
    # MPS trzeba włączyć jawnie (device="mps"): htdemucs korzysta z operacji STFT
    # i liczb zespolonych, które MPS obsługuje słabo, więc na Macu domyślnie jest CPU.
    return "cuda" if torch.cuda.is_available() else "cpu"


class DemucsSeparator(BaseSeparator):
    """Separacja pretrenowanym modelem Demucs (domyślnie htdemucs: 4 stemy, 44,1 kHz, stereo)."""

    def __init__(
        self,
        model_name: str = "htdemucs",
        device: str | None = None,
        shifts: int = 1,
        overlap: float = 0.25,
        progress: bool = False,
    ):
        self.name = model_name
        self.device = device or default_device()
        self.shifts = shifts
        self.overlap = overlap
        self.progress = progress
        self.model = get_model(model_name)
        self.model.eval()

    @property
    def stem_names(self) -> list[str]:
        return list(self.model.sources)

    @property
    def sample_rate(self) -> int:
        return self.model.samplerate

    def separate(self, input_path: Path) -> SeparationResult:
        wav = AudioFile(input_path).read(
            streams=0,
            samplerate=self.model.samplerate,
            channels=self.model.audio_channels,
        )
        return self._separate_tensor(wav)

    def separate_waveform(self, audio: np.ndarray, sample_rate: int) -> SeparationResult:
        wav = torch.as_tensor(audio, dtype=torch.float32)
        if wav.dim() == 1:
            wav = wav[None]
        wav = convert_audio(wav, sample_rate, self.model.samplerate, self.model.audio_channels)
        return self._separate_tensor(wav)

    def _separate_tensor(self, wav: torch.Tensor) -> SeparationResult:
        # Normalizacja jak w oficjalnym CLI Demucsa.
        # eps chroni przed dzieleniem przez 0, gdy wejście jest ciszą.
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        wav = (wav - mean) / std

        with torch.no_grad():
            sources = apply_model(
                self.model,
                wav[None],
                device=self.device,
                shifts=self.shifts,
                overlap=self.overlap,
                split=True,
                progress=self.progress,
            )[0]
        sources = sources * std + mean

        stems = {n: s.cpu().numpy() for n, s in zip(self.model.sources, sources)}
        return SeparationResult(stems=stems, sample_rate=self.model.samplerate)
