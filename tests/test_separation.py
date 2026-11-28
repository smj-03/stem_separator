import numpy as np
import pytest
import soundfile as sf

from stemsep.separation import SeparationResult

SR = 44100


@pytest.fixture
def short_wav(tmp_path):
    path = tmp_path / "noise.wav"
    rng = np.random.default_rng(0)
    audio = rng.uniform(-0.1, 0.1, size=(SR * 3, 2)).astype("float32")
    sf.write(path, audio, SR)
    return path


def test_result_save_roundtrips_float_audio(tmp_path):
    stem = np.full((2, SR), 1.5, dtype="float32")  # > 1.0 zostałoby obcięte w 16-bitowym PCM
    paths = SeparationResult({"vocals": stem}, SR).save(tmp_path / "out")
    data, sr = sf.read(paths["vocals"], always_2d=True)
    assert sr == SR
    np.testing.assert_allclose(data.T, stem)


# --- Poniższe testy ładują prawdziwe wagi Demucsa (pobierane raz, ~80 MB) ---


@pytest.fixture(scope="session")
def separator():
    from stemsep.separation.demucs import DemucsSeparator

    return DemucsSeparator("htdemucs", device="cpu")


@pytest.mark.slow
def test_demucs_returns_four_stereo_stems(separator, short_wav):
    result = separator.separate(short_wav)
    assert set(result.stems) == {"drums", "bass", "other", "vocals"}
    for audio in result.stems.values():
        assert audio.shape == (2, SR * 3)
        assert np.isfinite(audio).all()


@pytest.mark.slow
def test_demucs_handles_mono_array_and_resampling(separator):
    mono_22k = np.zeros(22050 * 2, dtype="float32")  # cisza, mono, inne próbkowanie
    result = separator.separate_waveform(mono_22k, 22050)
    assert result.sample_rate == separator.sample_rate
    for audio in result.stems.values():
        assert audio.shape[0] == 2
        assert np.isfinite(audio).all()  # cisza nie może dać NaN
