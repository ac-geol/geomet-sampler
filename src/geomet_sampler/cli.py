"""Command line interface.

A thin wrapper. Every command reads config, calls library functions, writes output.
No logic lives here: if a command body starts doing arithmetic or filtering, it belongs
in the library instead, where a UI could reach it too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .config import load_config
from .errors import GeometSamplerError
from .init_config import build_draft
from .io.writers import present, write_csv
from .pipeline import gap_frame, run_pipeline, summary_frame
from .report.excel import write_outputs, write_validation_report
from .report.plots import write_plots
from .validate.checks import report_frame, summarise

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Select drill core intervals for geometallurgical testwork.",
)

ConfigOption = Annotated[
    Path, typer.Option("--config", "-c", exists=True, dir_okay=False, help="Project YAML.")
]
OutOption = Annotated[Path | None, typer.Option("--out", "-o", help="Write the table here.")]
ForceOption = Annotated[
    bool, typer.Option("--force", help="Continue even when validation reports errors.")
]
CacheOption = Annotated[
    bool, typer.Option("--use-cache", help="Reuse cached stage output for this config hash.")
]


def _load(config: Path):
    try:
        return load_config(config)
    except GeometSamplerError as exc:  # pragma: no cover - surfaced to the user
        raise typer.BadParameter(str(exc)) from exc


def _run(config: Path, through: str, force: bool, use_cache: bool = False):
    cfg = _load(config)
    try:
        return cfg, run_pipeline(cfg, through=through, force=force, use_cache=use_cache)
    except GeometSamplerError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


def _emit(frame, cfg, out: Path | None, label: str) -> None:
    if out is None:
        out = cfg.project.output_dir / f"{label}.csv"
    write_csv(present(frame, cfg), out)
    typer.echo(f"{len(frame)} rows -> {out}")


@app.command()
def validate(config: ConfigOption, force: ForceOption = False) -> None:
    """Load every source, run all checks, and write the validation report."""
    _, state = _run(config, "load", force=True)
    path = write_validation_report(state)
    counts = summarise(state.report.issues)
    typer.echo(report_frame(state.report.issues).to_string(index=False, max_colwidth=90))
    typer.echo(f"\n{counts['ERROR']} errors, {counts['WARN']} warnings -> {path}")
    if counts["ERROR"] and not force:
        raise typer.Exit(code=1)


@app.command()
def desurvey(config: ConfigOption, out: OutOption = None, force: ForceOption = False) -> None:
    """Desurvey the holes and write the interval table with 3D coordinates."""
    cfg, state = _run(config, "geometry", force)
    _emit(state.intervals, cfg, out, "desurveyed")


@app.command()
def candidates(
    config: ConfigOption,
    out: OutOption = None,
    force: ForceOption = False,
    use_cache: CacheOption = False,
) -> None:
    """Segment into runs and generate candidate composites."""
    cfg, state = _run(config, "candidates", force, use_cache)
    _emit(state.candidates, cfg, out, "candidates")


@app.command()
def allocate(
    config: ConfigOption,
    out: OutOption = None,
    force: ForceOption = False,
    use_cache: CacheOption = False,
) -> None:
    """Compute per-domain sample targets from scheduled tonnage."""
    cfg, state = _run(config, "allocate", force, use_cache)
    _emit(state.allocation, cfg, out, "allocation")


@app.command()
def select(
    config: ConfigOption,
    out: OutOption = None,
    force: ForceOption = False,
    use_cache: CacheOption = False,
) -> None:
    """Run the full pipeline and write the picks workbook."""
    cfg, state = _run(config, "select", force, use_cache)
    written = write_outputs(state)
    if out is not None and "workbook" in written:
        written["workbook"].replace(out)
        written["workbook"] = out
    where = written.get("workbook", cfg.project.output_dir)
    typer.echo(f"{len(state.selected)} composites -> {where}")


@app.command()
def run(config: ConfigOption, force: ForceOption = False) -> None:
    """Full pipeline: validate, desurvey, composite, allocate, select, report."""
    cfg, state = _run(config, "select", force)
    written = write_outputs(state)
    write_validation_report(state)
    if cfg.reporting.include_plots:
        for path in write_plots(state):
            written[path.stem] = path

    typer.echo(summary_frame(state.summary).to_string(index=False, max_colwidth=70))
    gaps = gap_frame(state.gaps)
    if not gaps.empty:
        typer.secho(f"\n{len(gaps)} gap register entries, see Gap_Register", fg=typer.colors.YELLOW)
    typer.echo("\nwritten:")
    for name, path in written.items():
        typer.echo(f"  {name:28} {path}")


@app.command()
def init(
    out: Annotated[Path, typer.Option("--out", "-o", help="Where to write the draft config.")],
    collar: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    survey: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    assay: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    litho: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    block_model: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    name: str = "New project",
) -> None:
    """Inspect data files and write a draft config with # CONFIRM on every guess."""
    paths = {
        label: path
        for label, path in (
            ("collar", collar),
            ("survey", survey),
            ("assay", assay),
            ("litho", litho),
            ("block_model", block_model),
        )
        if path is not None
    }
    if not paths:
        raise typer.BadParameter("supply at least one input file to inspect")

    text, inferences, notes = build_draft(paths, project_name=name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    typer.echo(f"draft config -> {out}\n")
    for inference in inferences:
        found = {k: v for k, v in inference.mapping.items() if v}
        typer.echo(
            f"  {inference.label:12} {len(inference.columns):3} columns, "
            f"{len(found)} mapped, {len(inference.elements)} elements, "
            f"{len(inference.attributes)} attribute candidates"
        )
    typer.echo("\nnotes:")
    for note in notes:
        typer.echo(f"  - {note}")
    typer.secho(
        "\nEvery # CONFIRM line is a guess. Review the file before running the pipeline.",
        fg=typer.colors.YELLOW,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
