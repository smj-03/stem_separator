"""Interfejs wiersza poleceń (CLI) do separacji stemów."""

from pathlib import Path
import time
from typing import Annotated

import typer

from stemsep.separation.demucs import DemucsSeparator

# Teksty pomocy (--help) przekazujemy przez help=, a nie przez docstringi,
# żeby docstringi mogły być po polsku, a CLI zostało po angielsku.
app = typer.Typer(no_args_is_help=True)


@app.callback(help="Music stem separation toolkit.")
def main() -> None:
    """Główna grupa poleceń CLI."""


@app.command(help="Split a song into stems and write them as WAV files.")
def separate(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    model: Annotated[str, typer.Option(help="Demucs model name.")] = "htdemucs",
    device: Annotated[str | None, typer.Option(help="cpu, mps or cuda (default: auto).")] = None,
    out: Annotated[Path, typer.Option(help="Output root directory.")] = Path("data/processed"),
) -> None:
    """Separuje utwór na stemy i zapisuje je jako pliki WAV."""
    separator = DemucsSeparator(model, device=device, progress=True)

    # Mierzymy sam czas separacji (bez ładowania modelu) - przyda się do porównań.
    start = time.perf_counter()
    result = separator.separate(input_path)
    elapsed = time.perf_counter() - start

    # Wynik trafia do <out>/<model>/<nazwa utworu>/<stem>.wav
    paths = result.save(out / separator.name / input_path.stem)
    for name, path in paths.items():
        typer.echo(f"{name}: {path}")
    typer.echo(f"Separated on {separator.device} in {elapsed:.1f}s")


if __name__ == "__main__":
    app()
