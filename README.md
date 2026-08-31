# geomet-sampler

[![CI](https://github.com/ac-geol/geomet-sampler/actions/workflows/ci.yml/badge.svg)](https://github.com/ac-geol/geomet-sampler/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

Selects drill core intervals for geometallurgical testwork.

It reads drillhole data and a period-coded block model, works out which geological
domains the mine plan will actually mine, checks what testwork already exists, and
recommends the exact core intervals to pull next.

The question it answers: **given what I already have and what I am going to mine, which
exact core intervals should I sample?**

Output is a ranked recommendation with full reasoning, meant to be reviewed and
overridden, not a decision. Every composite traces back to the sample intervals a core
shed technician has to physically cut.

## Install

Python 3.12, managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run

```bash
uv run geomet-sampler run --config config/example_project.yaml
```

That writes `picks.xlsx`, the CSV set, the validation report and the charts into the
configured `output_dir`.

Stages can also be run individually. Each caches its result under `output_dir/.cache`,
keyed by a hash of the config, so a later stage can be rerun without repeating the
geometry and candidate work:

The example dataset is not stored in the repository. Generate it first:

```bash
uv run python scripts/make_example_data.py
```

```bash
uv run geomet-sampler validate   --config config/example_project.yaml
uv run geomet-sampler desurvey   --config config/example_project.yaml --out desurveyed.csv
uv run geomet-sampler candidates --config config/example_project.yaml --out candidates.csv
uv run geomet-sampler allocate   --config config/example_project.yaml --out allocation.csv
uv run geomet-sampler select     --config config/example_project.yaml --out picks.xlsx
```

Validation ERRORs stop the run. `--force` continues anyway, which is occasionally what
you want and never the default.

## Starting a new project

`init` inspects your files and writes a draft config with `# CONFIRM` on every guess:

```bash
uv run geomet-sampler init --out config/project.yaml \
    --collar data/collar.csv --survey data/survey.csv \
    --assay data/assay.csv --litho data/litho.csv \
    --block-model data/bm.csv
```

This is the only part of the tool that guesses anything. The pipeline itself either has
an explicit mapping or errors, because a mapping guessed at runtime is a mapping nobody
checked. The draft always needs review; you must still supply the domain lookup, the
hole availability file, and the testwork mass parameters by hand.

## Output

`picks.xlsx`, in the order a reviewer reads it:

| Sheet | What it is |
|---|---|
| `Summary` | run parameters, counts, config hash |
| `Allocation` | domain, tonnes, %, weighted %, target, existing, deficit, achieved |
| `Composites` | one row per selected composite, with attributes and the reason it was chosen |
| `Pick_List` | one row per interval to cut, sorted by hole then depth. The core shed worksheet. |
| `Gap_Register` | what could not be covered, and why |
| `Validation` | every data quality issue found |
| `Candidates` | all candidates with scores, for audit |

Plus PNG charts: allocation target vs achieved, plan view, long section by period, and
mine plan grade against selected sample grade.

## How it works

1. **Load and validate.** Readers map your column names onto a fixed canonical schema and
   normalise units and conventions once. Everything downstream works in canonical names
   and metres, negative-down dips, t/m3 and ppm.
2. **Desurvey.** Minimum curvature, verified against hand calculations.
3. **Interval framework.** Assay intervals split wherever a logged contact falls inside
   one. Each framework interval keeps its parent sample ID.
4. **Block model assignment.** Nearest centroid by KD-tree, then an explicit containment
   test against the block's own dimensions. Off-model intervals are flagged, not guessed.
5. **Segmentation into runs.** A run is a stretch with no hard break in it.
6. **Mass estimation.** From core diameter, remaining fraction, density and a loss factor.
7. **Candidate composites.** Generated only within runs, subject to length, mass, grade
   bin and variability constraints.
8. **Allocation.** Scheduled block tonnage, weighted toward early periods, split into
   per-domain targets by proportional or Neyman allocation, less existing testwork.
9. **Selection.** Greedy scoring with a swap improvement pass. Every pick records why.

## Design rules

These are the things that make the output wrong in ways that are hard to spot.

- **No user column name appears anywhere in the library.** Readers map user columns onto
  the canonical schema at load; if `df["ROCKTYPE"]` appears outside `io/readers.py`,
  that is a bug. There is a test that renames every column, converts to feet and inverts
  the dip convention, and asserts the output is unchanged.
- **A composite can never span a hard break.** Not a lithology contact, not a weathering
  front, not a hole boundary, not a core gap. This is structural: composites are
  generated only inside pre-computed runs, never filtered afterwards. There is a
  dedicated test that checks the property on generated candidates.
- **Every composite traces back to its intervals.** A composite without its interval list
  cannot be turned into a core shed instruction.
- **Mass is checked before a candidate is valid.** A recommendation nobody can physically
  collect is worse than no recommendation.
- **Selection is deterministic** under a fixed seed, with documented tie-breaking.
- **Nothing is dropped silently.** Anything excluded goes to the validation report or the
  gap register with a reason.
- **Attributes and elements are user-declared dictionaries**, not fixed fields. No code
  enumerates a known list of rock types, weathering states or elements. Adding an
  `alteration` role needs a config change, not a code change.
- **Logged lithology defines compositing boundaries; the block model defines tonnage and
  period.** Where they disagree the interval is marked `domain_match = False` and kept.
  The conflict is reported, never resolved silently.

## Conventions

Column names fail loudly. Conventions fail silently, which makes them the larger risk, so
all of them are declared explicitly in config with no guessing default, and two checks
exist specifically to catch a wrong one:

- after desurvey, any hole whose toe is above its collar RL is an ERROR, which catches an
  inverted dip convention immediately;
- if most intervals fall outside the block model extents, that is an ERROR about
  coordinate systems and units rather than a note about off-model drilling.

## Development

```bash
uv run pytest
uv run pytest --cov
uv run ruff check --fix .
uv run ruff format .
```

Coverage is 100% on `desurvey`, `segment` and `mass`, and above 90% overall.

## The example data

No data ships with this repository. `data/` is git-ignored unconditionally, so a real
export dropped into it can never be committed. Build the example dataset yourself:

```bash
uv run python scripts/make_example_data.py
```

That writes a synthetic dataset into `data/`. It is a schema example only: it stands in
for a real export and deliberately reproduces the messiness the readers must absorb
without hand-editing, including a UTF-8 BOM on the assay header, a fully quoted survey
file, CRLF line endings, and lithology headers cased differently from every other file.
The generator is seeded, so the output is identical on every machine.

The numbers are invented and must never be quoted as a real sampling result. Real
projects have entirely different column names, units and conventions. Nothing in `data/`
is a default anywhere in the code; those names live in the example config and nowhere
else.

## Known placeholders

Parameters that must be confirmed before the output is trusted, marked in the config:

- `mass.loss_factor` (0.90) should be calibrated against recovered mass from a past
  programme.
- `compositing.target_mass_kg` and `min_mass_kg` must come from your actual testwork
  package requirement.
- `domaining.grade_bins.edges` must come from your own grade distribution.
- `allocation.period_weights` encode how strongly the programme favours early production.

See `SPEC.md` section 9 for the open design decisions.

## Status

v1 (milestones M1–M5 from `SPEC.md`, plus M6 reporting). Milestone M7, an optional UI
calling this library unchanged, is not built.

This is a v1 built from a specification for a real but niche workflow. It has not yet
been run against a full production dataset or reviewed by a second geologist. Treat
output as a starting recommendation to check, not a validated result — see
[Known placeholders](#known-placeholders) above for the parameters that most need
calibrating before that.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for working agreements, the correctness rule that
matters most, and how to add a new attribute or element without touching library code.

## License

[MIT](LICENSE)
