"""Presentation and file output.

Internally grades are in ppm and columns carry canonical prefixes. Nothing in the
pipeline should care, but a geologist reading the workbook does, so the translation back
to the user's declared units and readable headers happens here and nowhere else.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import models as M
from ..config import Config
from ..units import ppm_to_units

#: Column prefixes that hold a grade in ppm, mapped to the reported header form and to
#: the suffix used if two of them would otherwise claim the same header.
GRADE_PREFIXES = {
    "elem_": ("{name}_{units}", ""),
    "wtd_": ("{name}_{units}", "_wtd"),
    "bm_elem_": ("bm_{name}_{units}", ""),
}


def present(frame: pd.DataFrame, cfg: Config, *, round_to: int = 3) -> pd.DataFrame:
    """Convert grades back to declared units and give columns readable names."""
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    out = frame.copy()
    renames: dict[str, str] = {}
    suffixes: dict[str, str] = {}

    for name, spec in cfg.elements.items():
        units = spec.drillhole.units or spec.block_model.units
        if units is None:
            continue
        for prefix, (template, suffix) in GRADE_PREFIXES.items():
            source = f"{prefix}{name}"
            if source not in out.columns:
                continue
            declared = (
                spec.block_model.units if prefix == "bm_elem_" and spec.block_model.units else units
            )
            out[source] = ppm_to_units(pd.to_numeric(out[source], errors="coerce"), declared)
            renames[source] = template.format(name=name, units=declared.value)
            suffixes[source] = suffix

    for role in cfg.attributes:
        for prefix, label in ((M.attr_col(role), role), (M.bm_attr_col(role), f"bm_{role}")):
            if prefix in out.columns:
                renames[prefix] = label

    if M.INTERVAL_IDS_COL in out.columns:
        out[M.INTERVAL_IDS_COL] = out[M.INTERVAL_IDS_COL].map(
            lambda v: ", ".join(map(str, v)) if isinstance(v, tuple | list) else v
        )

    out = out.rename(columns=_disambiguate(renames, suffixes))
    numeric = out.select_dtypes("float")
    out[numeric.columns] = numeric.round(round_to)
    return out


def _disambiguate(renames: dict[str, str], suffixes: dict[str, str]) -> dict[str, str]:
    """Keep headers unique.

    An interval's own assay and a composite's length-weighted mean of the same element
    both want to be called ``Zn_pct``. They never share a table today, but two columns
    with one name would corrupt the sheet silently if they ever did.
    """
    claimed: dict[str, int] = {}
    for target in renames.values():
        claimed[target] = claimed.get(target, 0) + 1
    return {
        source: (target + suffixes.get(source, "") if claimed[target] > 1 else target)
        for source, target in renames.items()
    }


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def write_workbook(sheets: dict[str, pd.DataFrame], path: Path) -> Path:
    """Write a formatted workbook, one sheet per table, in the given order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        book = writer.book
        header = book.add_format(
            {"bold": True, "bg_color": "#1F3864", "font_color": "white", "border": 1}
        )
        for name, frame in sheets.items():
            frame = pd.DataFrame() if frame is None else frame
            frame.to_excel(writer, sheet_name=name[:31], index=False, startrow=1, header=False)
            sheet = writer.sheets[name[:31]]
            if frame.empty:
                sheet.write(0, 0, f"no {name.lower()} rows", header)
                continue
            for column, value in enumerate(frame.columns):
                sheet.write(0, column, str(value), header)
                width = max(len(str(value)), _content_width(frame[value])) + 2
                sheet.set_column(column, column, min(width, 46))
            sheet.freeze_panes(1, 0)
            sheet.autofilter(0, 0, len(frame), len(frame.columns) - 1)
    return path


def _content_width(series: pd.Series) -> int:
    if series.empty:
        return 0
    widest = series.head(200).astype(str).str.len().max()
    return 0 if pd.isna(widest) else int(widest)
