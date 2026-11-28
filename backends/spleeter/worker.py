"""Worker Spleetera uruchamiany w osobnym środowisku z TensorFlow.

Użycie: python worker.py <wariant> <plik_wejściowy> <katalog_wyjściowy>
Wariant to 2stems, 4stems lub 5stems. Każdy stem zapisywany jest jako <nazwa>.npy
(float32, kształt (kanały, próbki)), a meta.json zawiera częstotliwość próbkowania.
"""

import json
from pathlib import Path
import sys

import numpy as np
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator

# Modele Spleetera są trenowane na 44,1 kHz
SAMPLE_RATE = 44100


def main(variant: str, input_path: Path, out_dir: Path) -> None:
    separator = Separator(f"spleeter:{variant}", multiprocess=False)

    # AudioAdapter czyta plik przez ffmpeg i od razu resampluje; kształt (próbki, kanały)
    waveform, _ = AudioAdapter.default().load(str(input_path), sample_rate=SAMPLE_RATE)
    if waveform.shape[1] == 1:
        # Spleeter oczekuje stereo, więc mono powielamy na dwa kanały
        waveform = np.repeat(waveform, 2, axis=1)

    stems = separator.separate(waveform)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, audio in stems.items():
        # Transpozycja do (kanały, próbki), tak jak w SeparationResult
        np.save(out_dir / f"{name}.npy", audio.T.astype(np.float32))
    meta = {"sample_rate": SAMPLE_RATE, "stems": list(stems)}
    (out_dir / "meta.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
