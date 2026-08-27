# Contributing

## Setup

```bash
uv sync
```

## Working agreements

These come from `CLAUDE.md` and apply to any change, human or AI-assisted:

- Python 3.12, managed with `uv`. Never `pip install` directly.
- `src/` layout. All logic lives in the library; the CLI (`cli.py`) is a thin wrapper
  that reads config, calls library functions, writes output — nothing else.
- Type hints on all public functions.
- pandas for tabular work. Polars is a discussion, not a drop-in.
- **No user column name may appear anywhere in the library.** Readers map user columns
  onto the canonical schema (`models.py`) at load; everything downstream uses canonical
  names only. If you find yourself writing `df["ROCKTYPE"]` outside `io/readers.py`,
  that's a bug.
- Attributes and elements are user-declared dictionaries, not fixed fields. Don't write
  code that enumerates a known list of rock types, weathering states or elements — that
  vocabulary belongs in config.
- Prefer plain functions over classes; use dataclasses for structured records.
- Write the test with the function, not afterwards.

## Before opening a PR

```bash
uv run ruff check --fix .
uv run ruff format .
uv run pytest --cov
```

CI runs the same three commands. Coverage is 100% on `desurvey`, `segment` and `mass` —
those modules encode the correctness guarantees (desurvey geometry, the no-composite-
crosses-a-hard-break rule, and physically collectable mass) and should stay there.

## The rule that matters most

A composite must never span a hard break — not a lithology contact, not a weathering
front, not a hole boundary, not a core gap. This is enforced structurally: composites are
generated only inside pre-computed runs (`segment/runs.py`), never filtered out
afterwards. `tests/test_segment.py` asserts this property directly against generated
candidates. Do not weaken it, and do not "fix" a failure by filtering the output — fix
the run boundaries instead.

## Adding a new attribute role or element

You shouldn't need to touch library code at all. Add the role to `attributes:` or
`elements:` in the project config, reference it in `hard_break_on` / `allocation_key` /
`grade_bins`, and it flows through automatically. If a code change turns out to be
required to support a new role, that's a sign something enumerates a fixed vocabulary
where it shouldn't — see the working agreement above.

## Reporting a bug

Please include the config (redact paths/site names if needed) and, if possible, a
minimal CSV that reproduces it. Since conventions (units, dip sign) fail silently rather
than loudly, "the numbers look wrong" bugs are much easier to fix with the actual input
than with a description of the output.
