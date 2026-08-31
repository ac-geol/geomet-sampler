# CLAUDE.md

Project context for Claude Code. Read `SPEC.md` for the full specification.

## What this is

A Python tool that selects drill core intervals for geometallurgical testwork. It reads
drillhole data and a period-coded block model, works out which geological domains the mine
plan will actually mine, checks what testwork already exists, and recommends the exact core
intervals to sample next.

The user is a senior geologist. Correctness and auditability matter more than cleverness.
Every recommendation must be explainable to a reviewer.

## Working agreements

- Python 3.12, managed with `uv`. Never use `pip install` directly.
- `src/` layout. All logic lives in the library. The CLI is a thin wrapper.
- **Never put logic in CLI command bodies.** A command reads config, calls library
  functions, writes output. Nothing else.
- Type hints on all public functions. Run `ruff check` and `ruff format` before finishing.
- pandas for tabular work. Do not introduce polars without discussion.
- **No user column name may appear anywhere in the library.** Readers map user columns onto
  a fixed canonical schema at load. Everything downstream uses canonical names only. If you
  find yourself writing `df["ROCKTYPE"]` outside `io/readers.py`, stop.
- Attributes and elements are **user-declared dictionaries**, not fixed fields. Never write
  code that enumerates a known list of rock types, weathering states or elements.
- Prefer plain functions over classes. Use dataclasses for structured records.
- Write the test with the function, not afterwards.

## Non-negotiable correctness rules

These are the things that make the output wrong in ways that are hard to spot:

1. **A composite must never span a hard break.** Not lithology, not weathering, not a hole
   boundary, not a core gap. This is enforced structurally by generating composites only
   within pre-computed runs, never by filtering afterwards. There is a dedicated test for
   this. Do not weaken it.
2. **Every composite must trace back to its constituent sample intervals.** The pick list
   is the actual deliverable. A composite without its interval list is useless.
3. **Mass must be checked before a candidate is valid.** A recommendation the user cannot
   physically collect is worse than no recommendation.
4. **Selection must be deterministic** under a fixed seed. Ties break on explicit,
   documented criteria.
5. **Never silently drop records.** Anything excluded goes to the validation report or the
   gap register with a reason.
6. **Logged lithology defines compositing boundaries. The block model defines tonnage and
   period.** Where they disagree, report `domain_match = False` and keep going. Do not
   resolve the conflict silently.
7. **Normalise units and conventions at load, once.** Dips to negative-down, lengths to
   metres, densities to t/m3, grades to their declared units. No module downstream of the
   reader does any convention handling. Conventions fail silently and are a larger risk than
   column names, so the validation suite includes the toe-above-collar check and the
   off-model-extent check specifically to catch them.
8. **Never guess a mapping at runtime.** Guessing belongs only in `geomet-sampler init`,
   which writes a draft config for human review with `# CONFIRM` on every inference. The
   pipeline itself either has an explicit mapping or errors.

## Domain glossary

- **Collar**: the drillhole's surface location and orientation
- **Survey**: downhole dip and azimuth measurements at depth
- **Desurvey**: converting downhole depths to true 3D coordinates using survey data
- **Minimum curvature**: the standard desurvey method, fits a circular arc between stations
- **Interval**: a from/to depth range in a hole, with attributes
- **Composite**: several adjacent intervals combined into one testwork sample
- **Domain / geomet domain**: a grouping of material expected to behave similarly in the
  plant
- **Weathering**: oxide / transition / fresh. Drives flotation response more strongly than
  lithology does.
- **Period**: mine plan production period, from the block model
- **Comminution**: crushing and grinding testwork (Bond Wi, SMC, Ai). Needs 10 to 25 kg.
- **Flotation**: recovery testwork. Bench 1 to 2 kg per test, locked cycle 10 to 20 kg.
- **HQ / NQ / BQ / PQ**: core diameters. HQ 63.5 mm, NQ 47.6 mm, BQ 36.5 mm, PQ 85 mm.
- **Half core**: half the core was already sampled for assay, so half remains

## Reference data quirks

`data/` is git-ignored and ships with nothing; build it with
`uv run python scripts/make_example_data.py`. The generated files are a
**schema example only**. Real users will have entirely
different column names, units and conventions. Never treat these names as defaults in code;
they belong in the example config and nowhere else. Handle the messiness in the readers, do
not clean the files by hand:

- `DEMO_Samples_BD_0001.csv` has a UTF-8 BOM on the `HoleID` header
- `DEMO_Survey_0001.csv` is fully quoted, all fields read as strings
- All files use CRLF line endings
- Lithology file uses lowercase `holeid`, `from`, `to`; other files use `HoleID`
- 60 collars, 52 have assays, 55 have lithology. Partial coverage is normal.
- The example block model is in a local grid, not UTM, and does not correspond to the
  drilling. It is a schema example only.
- The example block model carries a PERIOD column, populated for the blocks inside the
  pit stages (periods 1 to 4) and blank elsewhere.

## Commands

```bash
uv sync                          # install
uv run pytest                    # tests
uv run pytest --cov              # coverage
uv run ruff check --fix .
uv run ruff format .
uv run python scripts/make_example_data.py   # build data/ (not tracked)
uv run geomet-sampler init --collar ... --out config/project.yaml
uv run geomet-sampler run --config config/example_project.yaml
```

## A useful smoke test

Take the reference dataset, rename every column to something arbitrary, change dips to
positive-down and lengths to feet, then write a config for it. The tool must produce
identical output to the original run. If it does not, a convention or a column name has
leaked into the library.

## Build order

Follow the milestones in `SPEC.md` section 8. Do not start M4 before M2 tests pass.
Desurvey errors propagate silently into every downstream result, so it gets verified
against hand calculations before anything is built on top of it.

## When you are unsure

Ask rather than assume, particularly about:

- Anything involving geological interpretation or what constitutes a valid domain
- Testwork mass requirements
- Whether a data quality issue should be an ERROR or a WARN

Do not invent default values for geological or metallurgical parameters. Put them in config
with a comment marking them as placeholders needing user confirmation.
