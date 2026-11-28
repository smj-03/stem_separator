"""Interfejs wiersza poleceń (CLI) do separacji stemów."""

from pathlib import Path
import time
from typing import Annotated

import typer

from stemsep.separation import get_separator

# Teksty pomocy (--help) przekazujemy przez help=, a nie przez docstringi,
# żeby docstringi mogły być po polsku, a CLI zostało po angielsku.
app = typer.Typer(no_args_is_help=True)


@app.callback(help="Music stem separation toolkit.")
def main() -> None:
    """Główna grupa poleceń CLI."""


@app.command(help="Split a song into stems and write them as WAV files.")
def separate(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    model: Annotated[
        str,
        typer.Option(
            help="Demucs model (htdemucs, htdemucs_ft, mdx_extra, ...) "
            "or Spleeter variant (spleeter-2stems, spleeter-4stems, spleeter-5stems)."
        ),
    ] = "htdemucs",
    device: Annotated[
        str | None, typer.Option(help="cpu, mps or cuda (default: auto; Demucs only).")
    ] = None,
    out: Annotated[Path, typer.Option(help="Output root directory.")] = Path("data/processed"),
) -> None:
    """Separuje utwór na stemy i zapisuje je jako pliki WAV."""
    separator = get_separator(model, device=device)

    # Mierzymy sam czas separacji (bez ładowania modelu) - przyda się do porównań.
    # Uwaga: dla Spleetera czas obejmuje też start podprocesu i ładowanie TensorFlow.
    start = time.perf_counter()
    result = separator.separate(input_path)
    elapsed = time.perf_counter() - start

    # Wynik trafia do <out>/<model>/<nazwa utworu>/<stem>.wav
    paths = result.save(out / separator.name / input_path.stem)
    for name, path in paths.items():
        typer.echo(f"{name}: {path}")
    # Spleeter nie ma atrybutu device (zawsze CPU w osobnym procesie)
    device_used = getattr(separator, "device", "cpu")
    typer.echo(f"Separated with {separator.name} on {device_used} in {elapsed:.1f}s")


if __name__ == "__main__":
    app()
