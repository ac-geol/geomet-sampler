"""Text to numbers, with every value that is not a plain number accounted for.

Readers load files as text. This is the one place text becomes a number, so the rules
for blanks, declared null tokens and below-detection values are applied once and
reported once.

- An empty cell is blank. It is the only text read as blank without being declared.
- Text listed in a source's ``null_values`` is blank, and the count is reported (INFO).
  A token that is itself a number, such as ``-99``, matches by value, so ``-99.00``
  matches too.
- In grade columns only, ``<x`` is below detection and reads as half the limit (INFO).
- In grade columns only, a negative number is an ERROR unless the source declares
  ``negative_is_below_detection``, in which case ``-x`` reads as half of ``x`` (INFO).
  Databases use negatives both for detection limits and for sentinels like ``-99``, and
  the two cannot be told apart from the values.
- Anything else that will not parse is an ERROR naming the column, the count, examples
  and file lines.

A value that raises an ERROR loads as blank, so a run forced past the errors never
averages a sentinel into a composite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..models import Issue, Severity

#: ``<0.01``, ``< 0.01``, ``<1e-3``: a below-detection value and its limit
BELOW_DETECTION = re.compile(r"^<\s*(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
#: How many distinct example values an issue quotes
EXAMPLE_COUNT = 5
#: How many file lines an issue lists in its detail
LINE_COUNT = 20
#: A CSV's first line is the header, so data row ``i`` is on file line ``i + 2``
HEADER_LINES = 1


@dataclass(frozen=True)
class NumberRules:
    """What one source declares about the text in its numeric columns."""

    #: config name of the source, e.g. ``assay``; messages point at ``sources.<source>``
    source: str
    file: str
    null_values: tuple[str, ...] = ()
    negative_is_below_detection: bool = False


def parse_numbers(
    text: pd.Series, column: str, rules: NumberRules
) -> tuple[pd.Series, list[Issue]]:
    """Parse a numeric column such as a depth, coordinate or density.

    ``column`` is the user's column name, used in messages.
    """
    s, blank, declared = _classify(text, rules)
    values = _to_float(s.where(~blank & ~declared))
    unparsed = values.isna() & ~blank & ~declared
    issues = _declared_issue(s, declared, column, rules)
    issues += _unparseable_issue(s, unparsed, column, rules)
    return values, issues


def parse_grades(text: pd.Series, column: str, rules: NumberRules) -> tuple[pd.Series, list[Issue]]:
    """Parse a grade column: as :func:`parse_numbers`, plus detection limits and negatives.

    Values are returned in the column's own units; unit conversion is the caller's job.
    """
    s, blank, declared = _classify(text, rules)
    values = _to_float(s.where(~blank & ~declared))

    limit = s.str.extract(BELOW_DETECTION, expand=True)
    below = limit[0].notna() & ~declared
    values = values.where(~below, _to_float(limit[0] + limit[1].fillna("")) / 2.0)

    unparsed = values.isna() & ~blank & ~declared
    issues = _declared_issue(s, declared, column, rules)
    issues += _unparseable_issue(s, unparsed, column, rules)
    if below.any():
        issues.append(
            Issue(
                Severity.INFO,
                "below_detection",
                f"{int(below.sum())} values in {column} ({rules.file}) are below detection "
                f"and read as half the limit: {_value_counts(s, below)}",
                source=rules.source,
                count=int(below.sum()),
                detail=_detail(s, below, column),
            )
        )

    negative = values < 0
    if negative.any() and rules.negative_is_below_detection:
        values = values.where(~negative, values.abs() / 2.0)
        issues.append(
            Issue(
                Severity.INFO,
                "below_detection",
                f"{int(negative.sum())} negative values in {column} ({rules.file}) read as "
                f"below detection at half the limit (negative_is_below_detection): "
                f"{_value_counts(s, negative)}. A sentinel such as -99 belongs in null_values.",
                source=rules.source,
                count=int(negative.sum()),
                detail=_detail(s, negative, column),
            )
        )
    elif negative.any():
        values = values.where(~negative)
        issues.append(
            Issue(
                Severity.ERROR,
                "grade_negative",
                f"{int(negative.sum())} negative values in {column} ({rules.file}): "
                f"{_value_counts(s, negative)}, {_lines(negative)}. Declare sentinels "
                f"such as -99 in sources.{rules.source}.null_values, or set "
                f"sources.{rules.source}.negative_is_below_detection: true if a negative "
                "value is a detection limit.",
                source=rules.source,
                count=int(negative.sum()),
                detail=_detail(s, negative, column),
            )
        )
    return values, issues


# ---------------------------------------------------------------------- helpers


def _classify(text: pd.Series, rules: NumberRules) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Stripped text, which cells are empty, and which match a declared null token."""
    s = text.astype(str).str.strip()
    blank = s.eq("")
    words = {t.strip().upper() for t in rules.null_values}
    numbers = _to_float(pd.Series(sorted(words), dtype=object)).dropna()
    declared = ~blank & s.str.upper().isin(words)
    if not numbers.empty:
        declared |= ~blank & _to_float(s).isin(numbers.to_numpy())
    return s, blank, declared


def _to_float(s: pd.Series) -> pd.Series:
    """Plain numeric parse. ``inf`` and ``nan`` written as text are not numbers here."""
    values = pd.to_numeric(s, errors="coerce").astype(float)
    return values.where(np.isfinite(values))


def _declared_issue(
    s: pd.Series, declared: pd.Series, column: str, rules: NumberRules
) -> list[Issue]:
    if not declared.any():
        return []
    return [
        Issue(
            Severity.INFO,
            "null_value",
            f"{int(declared.sum())} values in {column} ({rules.file}) match "
            f"sources.{rules.source}.null_values and read as blank: "
            f"{_value_counts(s, declared)}",
            source=rules.source,
            count=int(declared.sum()),
        )
    ]


def _unparseable_issue(
    s: pd.Series, unparsed: pd.Series, column: str, rules: NumberRules
) -> list[Issue]:
    if not unparsed.any():
        return []
    return [
        Issue(
            Severity.ERROR,
            "value_unparseable",
            f"{int(unparsed.sum())} values in {column} ({rules.file}) are not numbers: "
            f"{_value_counts(s, unparsed)}, {_lines(unparsed)}. If they mean 'no value', "
            "declare them in "
            f"sources.{rules.source}.null_values; otherwise correct the export.",
            source=rules.source,
            count=int(unparsed.sum()),
            detail=_detail(s, unparsed, column),
        )
    ]


def _value_counts(s: pd.Series, mask: pd.Series) -> str:
    """Distinct values with counts, most common first, ties in file order."""
    counts = s[mask].value_counts(sort=False)
    counts = counts.sort_values(ascending=False, kind="stable").head(EXAMPLE_COUNT)
    text = ", ".join(f"{v!r} x{n}" for v, n in counts.items())
    more = s[mask].nunique() - len(counts)
    return text + (f" and {more} more" if more > 0 else "")


def _file_lines(mask: pd.Series) -> list[int]:
    """File line numbers of the flagged rows. Rows are in file order, as read."""
    return (np.flatnonzero(mask.to_numpy()) + 1 + HEADER_LINES).tolist()


def _lines(mask: pd.Series) -> str:
    """The first few file lines, for the message: the report table shows no detail."""
    lines = _file_lines(mask)
    shown = ", ".join(str(n) for n in lines[:EXAMPLE_COUNT])
    return f"file line{'s' if len(lines) > 1 else ''} {shown}" + (
        f" and {len(lines) - EXAMPLE_COUNT} more" if len(lines) > EXAMPLE_COUNT else ""
    )


def _detail(s: pd.Series, mask: pd.Series, column: str) -> dict:
    return {"column": column, "file_lines": _file_lines(mask)[:LINE_COUNT]}
