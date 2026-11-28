from stemsep.separation.base import BaseSeparator, SeparationResult

__all__ = ["BaseSeparator", "SeparationResult", "get_separator"]


def get_separator(model: str, device: str | None = None) -> BaseSeparator:
    """Tworzy separator na podstawie nazwy modelu.

    "spleeter-<wariant>" (np. spleeter-4stems) uruchamia Spleetera, każda inna nazwa
    jest traktowana jako model Demucs (np. htdemucs, htdemucs_ft, mdx_extra).
    Importy są leniwe, żeby wybór Spleetera nie ładował PyTorcha i odwrotnie.
    """
    if model.startswith("spleeter"):
        from stemsep.separation.spleeter import SpleeterSeparator

        variant = model.removeprefix("spleeter").lstrip("-:") or "4stems"
        # Spleeter działa wyłącznie na CPU w osobnym procesie, parametr device pomijamy
        return SpleeterSeparator(variant)

    from stemsep.separation.demucs import DemucsSeparator

    return DemucsSeparator(model, device=device, progress=True)
