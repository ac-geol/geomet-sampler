# Geometallurgical Sample Selection Tool: Specification

Version 0.1 (draft for development)

---

## 1. Purpose

Select an optimal set of drill core samples for geometallurgical testwork, such that the
resulting testwork programme is representative of the material scheduled in the mine plan,
with weighting toward earlier production periods, and such that every recommended sample is
physically collectable from available core.

The tool answers: **"Given what I already have and what I am going to mine, which exact
core intervals should I pull next?"**

---

## 2. Scope

### In scope (v1)

- Load, validate and merge collar, survey, assay and lithology interval data
- Desurvey holes (minimum curvature) to obtain true 3D interval geometry
- Assign block model attributes and mine plan period to each drillhole interval
- Segment holes into domain-coherent runs
- Generate candidate composites within runs, subject to mass and grade constraints
- Compute per-domain sample targets from mine plan tonnage, adjusted for existing testwork
- Select composites via greedy scoring with a swap-improvement pass
- Export a composite register and an interval-level core shed pick list

### Out of scope (v1)

- Graphical user interface (deferred to M6)
- 3D visualisation
- Direct database or acQuire/Fusion connectivity (CSV only in v1)
- Predictive geometallurgical modelling of test outcomes
- Sub-blocked block model support (parent blocks only in v1, see Open Decisions)

### Non-goals

- The tool does not decide what tests to run. It allocates samples to domains given a
  user-specified testwork package and its mass requirement.
- The tool does not replace geological judgement. Output is a ranked recommendation with
  full reasoning, intended to be reviewed and overridden.

---

## 3. Inputs

All inputs are CSV in v1. **No user column name appears anywhere in the library.** Readers
map user columns to a fixed canonical schema at load time; every downstream module works
only in canonical names.

### 3.1 Canonical schema

These names are fixed in code. The config maps user columns onto them.

**Collar**: `hole_id`, `east`, `north`, `rl`, `total_depth`
**Survey**: `hole_id`, `depth`, `dip`, `azimuth`
**Interval (assay)**: `hole_id`, `sample_id`, `from_m`, `to_m`, `density`, `low_recovery`
**Interval (litho)**: `hole_id`, `from_m`, `to_m`, `logged_code`
**Block**: `x`, `y`, `z`, `dx`, `dy`, `dz`, `density`, `period`

Plus two open namespaces, populated from user config:

- `attributes[<role>]`: categorical fields the user declares, e.g. `attributes["weathering"]`
- `elements[<name>]`: numeric grade fields the user declares, e.g. `elements["Zn"]`

Role names are invented by the user, not fixed by the tool. `hard_break_on` and
`allocation_key` reference those role names.

### 3.2 Column tiers

**Tier 1: required structural.** Must be mapped. Missing mapping is a hard error listing
the columns actually present in the file.

- Collar: `hole_id`, `east`, `north`, `rl`
- Survey: `hole_id`, `depth`, `dip`, `azimuth`
- Intervals: `hole_id`, `from_m`, `to_m`
- Block model: `x`, `y`, `z`, `dx`, `dy`, `dz`, `period`

**Tier 2: semantic optional.** Mapped if present, with a defined fallback. The fallback used
is logged and reported in the run summary.

| Field | If absent | Fallback |
|---|---|---|
| `sample_id` | generated as `{hole_id}_{from_m}` | flagged WARN, pick list is degraded |
| `density` (interval) | constant from config | required if no column |
| `density` (block) | constant from config | required for tonnage |
| `low_recovery` | no recovery-based break | INFO |
| `total_depth` | max survey or interval depth | INFO |

**Tier 3: user-declared attributes and elements.** Arbitrary in number and name. Declared as
role-to-source dictionaries. Nothing in the code enumerates them.

### 3.3 Conventions

Column names fail loudly. Conventions fail silently, and are the larger risk.

| Setting | Options | Consequence if wrong |
|---|---|---|
| `dip_convention` | `negative_down`, `positive_down` | Hole desurveys above surface |
| `length_units` | `m`, `ft` | Everything scales wrong, masses 35x out |
| `azimuth_reference` | `grid`, `true` (+ declination) | Systematic rotation of all traces |
| `density_units` | `t_m3`, `g_cm3`, `lb_ft3` | Mass estimates wrong |
| Element `units` | `pct`, `ppm`, `gpt`, `oz_t` | Drillhole vs block model comparison invalid |

All are declared explicitly in config. There is no default that silently guesses. Validation
includes a sanity check: after desurvey, any hole whose toe is above its collar RL is an
ERROR, which catches an inverted dip convention immediately.

### 3.4 Required files

- **Collar**, **Survey**, **Assay/interval**, **Lithology** (section 3.1 canonical fields)
- **Domain lookup**: maps `logged_code` to `geomet_domain`. Required, because logged codes
  (24 in the reference data) cannot be reconciled with block model domains automatically.

  ```csv
  logged_code,geomet_domain
  BZPZ,BOUNDARY_ZONE
  TSBF,TOM_SILL
  ```

- **Block model**: must include a mine plan period field. Blocks with null or excluded
  period are treated as not scheduled and excluded from allocation targets.
- **Hole availability**: controls candidate supply.

  ```csv
  hole_id,availability,core_diameter_mm,remaining_fraction,notes
  BX23-001,AVAILABLE,63.5,0.5,
  DDH-79-1,UNAVAILABLE,,,core disposed 1998
  PLAN-001,PLANNED,63.5,1.0,scheduled Q3 2026
  ```

  States: `AVAILABLE`, `UNAVAILABLE`, `PLANNED`. `remaining_fraction` is the fraction of
  original core still in the tray. Holes absent from this file default to `UNAVAILABLE`
  (fail closed). Column names here are also configurable.

### 3.5 Optional files

- **Existing testwork**: prior samples, so the tool allocates the deficit rather than the
  total. If `geomet_domain` is blank, the tool derives it by re-running the interval
  pipeline against the recorded depths.
- **Planned holes**: collar and planned survey for holes not yet drilled. Candidates from
  these carry `is_planned = True` and are reported separately.

### 3.6 Reference dataset quirks

The supplied example files are a schema example only. They demonstrate the messiness the
readers must absorb without hand-editing:

- UTF-8 BOM on the assay `HoleID` header
- Survey file fully quoted, all fields parse as strings
- CRLF line endings throughout
- Lithology file uses lowercase `holeid`, `from`, `to`; other files use `HoleID`
- 559 collars, 432 with assays, 347 with lithology
- No sample ID column (Tier 2 fallback applies)
- Block model in a local grid, not corresponding to the drilling, and with no period field

---

## 4. Configuration

Single YAML file per project. This file plus the input CSVs is a complete, reproducible
record of how a sample list was produced. A config hash is written into every output.

### 4.1 Schema

```yaml
project:
  name: MPA Geomet Programme 2026
  crs: EPSG:26909
  output_dir: ./output

conventions:
  length_units: m               # m | ft
  dip_convention: negative_down # negative_down | positive_down
  azimuth_reference: grid       # grid | true
  magnetic_declination_deg: 0.0 # applied only if azimuth_reference: true
  density_units: t_m3           # t_m3 | g_cm3 | lb_ft3

# ---------------------------------------------------------------- data sources
sources:

  collar:
    path: data/MPA_Collar_20240227.csv
    encoding: auto              # auto | utf-8 | utf-8-sig | cp1252
    columns:
      hole_id: HoleID
      east: Easting
      north: Northing
      rl: Elevation
      total_depth: Length_m     # null -> derived, WARN
    carry_through:              # kept for filtering and reporting only
      - HoleType
      - Prospect
      - Year

  survey:
    path: data/MPA_Survey_20240227.csv
    columns:
      hole_id: HoleID
      depth: Depth_m
      dip: Dip
      azimuth: Azimuth

  assay:
    path: data/MPA_Samples_BD_20240227.csv
    columns:
      hole_id: HoleID
      from_m: From_m
      to_m: To_m
      sample_id: null           # null -> generated, WARN
    density:
      source: BD_tonnes_m3
      fallback_constant: null   # used only if source is null
    flags:
      low_recovery:
        source: "LowRecovery_<=85pct"
        type: boolean           # boolean | threshold
        true_values: [Y, "1", TRUE, "true"]
        # if type: threshold ->
        # threshold: 85.0
        # direction: below

  litho:
    path: data/MPA_Interp_20240227.csv
    columns:
      hole_id: holeid
      from_m: from
      to_m: to
      logged_code: Code

  domain_lookup:
    path: data/domain_lookup.csv
    columns:
      logged_code: logged_code
      geomet_domain: geomet_domain

  block_model:
    path: data/block_model_periods.csv
    columns:
      x: X
      y: Y
      z: Z
      dx: DX
      dy: DY
      dz: DZ
      period: PERIOD
    density:
      source: DENSITY
      fallback_constant: 2.70

  availability:
    path: data/hole_availability.csv
    columns:
      hole_id: hole_id
      availability: availability
      core_diameter_mm: core_diameter_mm
      remaining_fraction: remaining_fraction
    value_map:                  # maps site vocabulary onto the three states
      AVAILABLE: [AVAILABLE, YES, Y, IN STORAGE]
      UNAVAILABLE: [UNAVAILABLE, NO, N, DISPOSED, CONSUMED]
      PLANNED: [PLANNED, PROPOSED, FUTURE]

  existing_testwork:
    path: data/existing_testwork.csv    # null to skip
    columns:
      sample_id: sample_id
      hole_id: HoleID
      from_m: from_m
      to_m: to_m
      geomet_domain: geomet_domain      # null -> derived
      test_package: test_package

  planned_collar: null
  planned_survey: null

# ------------------------------------------------------- user-declared fields
# Role names on the left are invented by the user and referenced by
# domaining.hard_break_on and domaining.allocation_key. Nothing in the code
# enumerates these.
attributes:
  weathering:
    block_model: WEATHERING
    drillhole: null             # not logged in this dataset
  rock_type:
    block_model: ROCKTYPE
    drillhole: null
  resource_class:
    block_model: CLASS
    drillhole: null
  # example of a site-specific addition requiring no code change:
  # alteration:
  #   block_model: ALTN
  #   drillhole: Alteration_Code

elements:
  Zn:
    primary: true               # exactly one element must be primary
    drillhole: {field: Zn_pct, units: pct}
    block_model: {field: null,  units: null}
  Pb:
    drillhole: {field: Pb_pct, units: pct}
    block_model: {field: null,  units: null}
  Ag:
    drillhole: {field: Ag_ppm, units: ppm}
    block_model: {field: AG_GPT, units: gpt}
  # units are normalised internally; ppm and gpt are equivalent, pct = ppm / 10000

# ---------------------------------------------------------------- processing
desurvey:
  method: minimum_curvature
  fill_collar_survey: true      # insert station at depth 0 from collar orientation

domaining:
  hard_break_on:                # role names from `attributes`, plus geomet_domain
    - geomet_domain
    - weathering
  max_interval_gap_m: 0.10
  break_on_low_recovery: true
  break_on_missing_primary_grade: true

  allocation_key:               # role names from `attributes`, plus grade_bin
    - rock_type
    - weathering
    - grade_bin
  grade_bins:
    element: Zn                 # must exist in `elements`
    source: block_model         # block_model | drillhole
    edges: [0.0, 2.0, 5.0, 999.0]
    labels: [LOW, MED, HIGH]

compositing:
  min_length_m: 6.0
  max_length_m: 30.0
  target_mass_kg: 50.0
  min_mass_kg: 40.0
  allow_multi_hole: false
  max_cv_primary: 1.0
  require_grade_bin_match: true

mass:
  default_core_diameter_mm: 63.5
  default_remaining_fraction: 0.5
  loss_factor: 0.90             # PLACEHOLDER, calibrate against past programmes

allocation:
  method: neyman                # proportional | neyman | manual
  total_samples: 40
  min_samples_per_domain: 2
  min_domain_tonnage_pct: 1.0
  risk_multipliers:
    VOLCANIC-OXIDE-HIGH: 1.5
  period_weights:
    1: 3.0
    2: 2.5
    3: 2.0
    4: 1.5
    default: 1.0

selection:
  method: greedy_with_swap
  max_samples_per_hole: 3
  min_separation_m: 50.0
  swap_iterations: 200
  random_seed: 42

reporting:
  formats: [xlsx, csv]
  include_gap_register: true
  include_plots: true
```

### 4.2 Config validation rules

Enforced by pydantic models at load, before any data is read:

- Every Tier 1 field mapped to a non-null column name
- Exactly one element flagged `primary: true`
- `grade_bins.element` exists in `elements`
- If `grade_bins.source: block_model`, that element has a non-null `block_model.field`
- Every name in `hard_break_on` and `allocation_key` resolves to a declared attribute role,
  or is one of the built-ins `geomet_domain` / `grade_bin` / `period`
- Every attribute referenced in `allocation_key` has a non-null `block_model` source,
  since allocation is computed from block model tonnage
- Every attribute in `hard_break_on` resolves from either source
- `len(grade_bins.edges) == len(grade_bins.labels) + 1`
- `min_mass_kg <= target_mass_kg`, `min_length_m < max_length_m`
- If any `density.source` is null, `fallback_constant` must be set

After the files are opened, a second pass checks every mapped column actually exists,
reporting all missing mappings at once with the available columns listed, rather than
failing on the first one.

### 4.3 Config generation

```bash
geomet-sampler init --collar data/collar.csv --survey data/survey.csv \
                    --assay data/assay.csv --litho data/litho.csv \
                    --block-model data/bm.csv --out config/project.yaml
```

Inspects headers and data, fuzzy-matches likely mappings, and writes a draft config with
`# CONFIRM` comments on every guess. Guessing rules:

- Structural fields by name similarity against a synonym table
  (`hole_id`: holeid, bhid, dhid, drillhole, hole; `from_m`: from, depth_from, start)
- Numeric columns with a unit suffix become candidate elements (`_pct`, `_ppm`, `_gpt`)
- Low-cardinality string columns (under 50 distinct values) become candidate attributes
- Density guessed from name plus a value range check (1.5 to 6.0)
- `dip_convention` guessed from the sign of the majority of dip values, always marked
  `# CONFIRM`

The generated config is a starting point requiring review, never used unedited. Every
guessed value is marked, and `init` prints a summary of what it inferred.

---

## 5. Processing pipeline

### 5.1 Load and validate (`io/`, `validate/`)

Checks, each producing a structured issue record (severity: ERROR / WARN / INFO):

- Required columns present after mapping
- Collar hole IDs unique
- Every assay/litho/survey hole ID exists in collar (orphan check)
- `from < to` on all intervals
- No overlapping intervals within a hole
- Interval gaps reported (INFO below tolerance, WARN above)
- Intervals beyond collar total depth
- Survey depth monotonic increasing per hole
- Azimuth in [0, 360), dip within valid range for the declared `dip_convention`
- Density present and within a plausible range after unit normalisation (1.5 to 6.0 t/m3)
- Block model: no duplicate centroids, period field populated
- Every value in the litho `logged_code` field resolves in the domain lookup (unmapped
  codes are an ERROR, since silently dropping them distorts allocation)
- Every value in the availability field resolves through `value_map`
- **Convention sanity check**: after desurvey, any hole whose toe RL exceeds its collar RL
  is an ERROR. This catches an inverted `dip_convention` before it propagates.
- **Extent sanity check**: fraction of desurveyed intervals falling outside the block model
  extents. Above a threshold (default 50%) this is an ERROR, since it usually means a
  coordinate system or unit mismatch rather than genuine off-model drilling.

**Fail on ERROR by default**, with `--force` to continue. Write `validation_report.csv`.

All checks operate on canonical field names. Where an issue must name a source column for
the user's benefit, the reader's mapping table is used to translate back.

### 5.2 Desurvey (`desurvey/`)

Minimum curvature between consecutive survey stations.

Dips are normalised to `negative_down` at load, per `conventions.dip_convention`. All
lengths are normalised to metres per `conventions.length_units`. Azimuths have declination
applied if `azimuth_reference: true`. The desurvey module assumes normalised input and does
no convention handling itself.

Given stations 1 and 2 at depths d1, d2 with dips a1, a2 (as inclination from horizontal,
negative down) and azimuths b1, b2:

```
I = 90 + dip                      # convert to inclination from vertical-down convention
DL = arccos( cos(I2 - I1) - sin(I1) * sin(I2) * (1 - cos(B2 - B1)) )
RF = 1.0 if DL == 0 else (2 / DL) * tan(DL / 2)
dNorth = (MD/2) * (sin(I1)cos(B1) + sin(I2)cos(B2)) * RF
dEast  = (MD/2) * (sin(I1)sin(B1) + sin(I2)sin(B2)) * RF
dZ     = (MD/2) * (cos(I1) + cos(I2)) * RF
```

Requirements:

- Insert a station at depth 0 from collar dip/azimuth if absent (`fill_collar_survey`)
- Extrapolate the last station to total depth holding orientation constant
- Interpolate to arbitrary depth (needed for interval from/to and midpoints)
- Return, per interval: `x_from, y_from, z_from, x_to, y_to, z_to, x_mid, y_mid, z_mid`

**Test with hand-calculated cases**: a vertical hole, a 45 degree hole with no deviation,
and a known dogleg with published expected coordinates.

### 5.3 Interval merge (`intervals/`)

Build a single interval table on a common framework.

1. Take assay intervals as the base framework
2. Split assay intervals wherever a lithology boundary falls inside them
3. Attach lithology code and mapped `geomet_domain` to each resulting interval
4. Carry grades and density from the parent assay interval (no re-averaging needed,
   grade is constant within the parent)
5. Record `parent_sample_id` so the physical sample is always traceable

For `PLANNED` holes there are no assays. Instead, discretise the planned trace at a fixed
interval (default 1.0 m) and populate attributes entirely from the block model.

### 5.4 Block model assignment (`blockmodel/`)

1. Build a `scipy.spatial.cKDTree` on block centroids
2. Query nearest block for each interval midpoint
3. Verify the midpoint falls within that block's extents using DX/DY/DZ
4. If outside, the interval is flagged `outside_model = True` and excluded from candidates

Attach: `period`, every declared attribute role with a `block_model` source, and every
declared element with a `block_model` field. The set is read from config, not hardcoded.

Note: for coarse blocks, midpoint assignment is adequate. If interval length approaches
block height, consider sub-sampling the interval and taking the modal block.

Record `domain_match` by comparing the logged `geomet_domain` against the modelled
allocation domain.
This is a QC output, never a filter. Where they disagree the sample is still physically
valid but may not represent the domain the model assumes.

### 5.5 Segmentation into runs (`segment/`)

Within each hole, walk intervals in depth order and start a new run when any of:

- Any `hard_break_on` attribute changes
- Depth gap to previous interval exceeds `max_interval_gap_m`
- Low recovery flag set and `break_on_low_recovery` is true
- Primary grade missing and `break_on_missing_primary_grade` is true
- Interval flagged `outside_model`
- Hole changes

Output: run ID per interval, plus a run summary (length, domain, period, mass).

Runs shorter than the minimum mass requirement generate no candidates and are written to
the **gap register**.

### 5.6 Mass estimation (`mass/`)

Per interval:

```
area_m2   = pi * (core_diameter_mm / 2000) ** 2
mass_kg   = length_m * area_m2 * remaining_fraction * density_t_m3 * 1000 * loss_factor
```

Sanity values at SG 2.7, half core, loss factor 1.0:

| Core size | Diameter | kg/m (half core) | m for 50 kg |
|---|---|---|---|
| BQ | 36.5 mm | 1.41 | 35.4 |
| NQ | 47.6 mm | 2.40 | 20.8 |
| HQ | 63.5 mm | 4.27 | 11.7 |
| PQ | 85.0 mm | 7.66 | 6.5 |

These numbers make the compositing requirement unavoidable, and they mean core size is a
first-order control on which holes can supply a full testwork package.

### 5.7 Candidate composite generation (`composite/`)

For each run, generate all contiguous windows of intervals satisfying:

- Total length between `min_length_m` and `max_length_m`
- Total estimated mass at least `min_mass_kg`
- Length-weighted mean primary grade falls in a defined grade bin
  (and matches the run's assigned bin if `require_grade_bin_match`)
- Coefficient of variation of primary grade at or below `max_cv_primary`

Implementation: sliding window with prefix sums, stopping expansion once `max_length_m` or
mass target plus tolerance is exceeded. Do not enumerate all O(n^2) windows on long runs.

Each candidate carries:

```
candidate_id, hole_id, run_id, from_m, to_m, length_m, n_intervals,
est_mass_kg, geomet_domain, period, domain_match, is_planned,
attributes[<role>] for every declared attribute role,
wtd_<element> for every declared element, cv_primary,
x_mid, y_mid, z_mid

Attribute and element columns are generated from config at runtime. Nothing in the code
enumerates them.
```

De-duplicate heavily overlapping candidates before selection (keep the best-scoring
representative per start interval) to keep the greedy loop tractable.

### 5.8 Allocation targets (`allocate/`)

1. Aggregate scheduled block model tonnage by allocation domain and period:
   `tonnes = DX * DY * DZ * DENSITY`
2. Apply period weights to get **weighted tonnage**:
   `wt_tonnes = tonnes * period_weight(period)`
3. Compute per-domain targets:

   - **proportional**: `n_d = total_samples * wt_tonnes_d / sum(wt_tonnes)`
   - **neyman**: `n_d = total_samples * (wt_tonnes_d * s_d * r_d) / sum(wt_tonnes * s * r)`
     where `s_d` is the standard deviation of the primary grade within the domain (from the
     block model) and `r_d` is the user risk multiplier, default 1.0

4. Apply floors: any domain above `min_domain_tonnage_pct` receives at least
   `min_samples_per_domain`
5. Subtract existing testwork coverage per domain to get the **deficit**
6. Rescale so the sum of deficits equals `total_samples`, then round using largest
   remainder so the total is exact

Report the full table: domain, tonnes, tonnage %, weighted %, target, existing, deficit.

Rationale for Neyman rather than strict proportional: a domain that is 5% of tonnage but
highly variable, or that drives a recovery penalty, needs more samples per tonne than a
homogeneous bulk domain. Proportional allocation is available for when a strict
tonnage-share argument is required.

### 5.9 Selection (`select/`)

Greedy with swap improvement.

**Score for candidate c given current selection S:**

```
score(c) = deficit_weight(domain(c))
         * period_weight(period(c))
         * quality(c)
         * separation_penalty(c, S)
         * hole_penalty(c, S)
         * planned_penalty(c)
```

- `deficit_weight` = 0 if the domain's remaining deficit is 0, else remaining deficit
- `quality` rewards mass margin above target, penalises high CV and `domain_match = False`
- `separation_penalty` = 0 if any selected sample is within `min_separation_m` in 3D,
  else 1. Optionally a smooth ramp rather than a hard cut.
- `hole_penalty` = 0 if the hole is already at `max_samples_per_hole`, else 1
- `planned_penalty` down-weights planned holes so real core is preferred where available

**Loop:**

1. Compute deficits from allocation
2. Score all candidates
3. Select the highest scorer; break ties by lowest CV, then by hole ID for determinism
4. Decrement that domain's deficit, mark overlapping candidates in the same hole as unusable
5. Repeat until all deficits are zero or no candidate scores above zero

**Swap pass:** for `swap_iterations`, attempt to replace a selected composite with an
unselected one and keep the change if the global objective improves. Global objective:

```
objective = -sum(|achieved_d - target_d|) * W1     # allocation fit
          + mean_min_separation * W2               # spatial spread
          - sum(cv_primary) * W3                   # sample homogeneity
          + domain_match_rate * W4                 # logged/modelled agreement
```

Every selection must record a human-readable `reason` string, for example:
`"fills VOLCANIC-OXIDE-HIGH deficit (3 remaining), period 2 (weight 2.5), 52.1 kg, CV 0.41"`.

Start with greedy. It is interpretable, has no solver dependency, runs in seconds, and for
coverage problems is typically within a few percent of optimal. A MILP formulation can be
swapped in behind the same interface if greedy proves inadequate.

### 5.10 Reporting (`report/`)

**Excel workbook**, sheets:

1. `Summary`: run parameters, counts, config hash
2. `Allocation`: domain, tonnes, %, weighted %, target, existing, deficit, achieved
3. `Composites`: one row per selected composite, with all attributes and the reason string
4. `Pick_List`: one row per interval within each selected composite, sorted by hole then
   depth. This is the core shed worksheet.
5. `Gap_Register`: domains with unmet deficits and the reason (no available core, runs too
   short, mass shortfall, grade bin mismatch)
6. `Validation`: issues from step 5.1
7. `Candidates`: all candidates with scores, for audit

**Pick list format:**

```
composite_id | hole_id | sample_id | from_m | to_m | length_m | litho | Zn_pct | est_mass_kg
GM-001       | BX23-014| 40213     | 112.30 | 113.60| 1.30    | TSBF  | 4.82   | 5.6
GM-001       | BX23-014| 40214     | 113.60 | 115.00| 1.40    | TSBF  | 5.11   | 6.0
```

**Plots** (matplotlib, PNG):

- Allocation target vs achieved bar chart by domain
- Plan view of selected composites coloured by domain
- Long section coloured by period
- Grade distribution: mine plan vs selected samples per domain

---

## 6. Architecture

```
geomet-sampler/
├── pyproject.toml            # uv, Python 3.12
├── CLAUDE.md
├── README.md
├── config/
│   └── example_project.yaml
├── src/geomet_sampler/
│   ├── __init__.py
│   ├── cli.py                # typer entry point
│   ├── config.py             # pydantic models for the YAML schema
│   ├── io/
│   │   ├── readers.py        # CSV loading, encoding and column mapping
│   │   └── writers.py        # Excel and CSV output
│   ├── validate/
│   │   └── checks.py
│   ├── desurvey/
│   │   └── minimum_curvature.py
│   ├── intervals/
│   │   └── merge.py          # framework building, litho splitting
│   ├── blockmodel/
│   │   └── assign.py         # cKDTree lookup
│   ├── segment/
│   │   └── runs.py
│   ├── mass/
│   │   └── estimate.py
│   ├── composite/
│   │   └── candidates.py
│   ├── allocate/
│   │   └── targets.py
│   ├── select/
│   │   ├── greedy.py
│   │   └── objective.py
│   ├── report/
│   │   ├── excel.py
│   │   └── plots.py
│   └── models.py             # dataclasses / typed schemas passed between stages
└── tests/
    ├── conftest.py
    ├── data/                 # small synthetic fixtures committed to the repo
    ├── test_desurvey.py      # hand-calculated expected values
    ├── test_intervals.py
    ├── test_segment.py       # asserts no composite crosses a hard break
    ├── test_mass.py
    ├── test_composite.py
    ├── test_allocate.py
    ├── test_select.py
    └── test_end_to_end.py
```

### Layering rule

The **library** contains all logic as plain functions. No printing, no user interaction, no
file paths hardcoded. The **CLI** is a thin wrapper that reads config, calls library
functions, writes output. A future UI becomes a third caller of the same library.

Do not put logic in CLI command bodies or, later, in UI callbacks.

### Stack

- Python 3.12
- `uv` for environment and dependency management
- `pandas` for tabular work (team familiarity; polars is a later optimisation if the block
  model joins become slow)
- `numpy` for desurvey and scoring maths
- `scipy` for cKDTree
- `pydantic` and `pydantic-settings` for config validation
- `typer` for the CLI
- `openpyxl` or `xlsxwriter` for Excel output
- `matplotlib` for plots
- `pytest` and `pytest-cov` for tests
- `ruff` for lint and format

### CLI surface

```bash
geomet-sampler validate  --config project.yaml
geomet-sampler desurvey  --config project.yaml --out desurveyed.csv
geomet-sampler candidates --config project.yaml --out candidates.csv
geomet-sampler allocate  --config project.yaml --out allocation.csv
geomet-sampler select    --config project.yaml --out picks.xlsx
geomet-sampler run       --config project.yaml     # full pipeline
```

Each stage caches its output to `output_dir` so later stages can be rerun without
repeating the expensive steps.

---

## 7. Testing strategy

- **Desurvey**: hand-calculated vertical hole, inclined hole with no deviation, and a
  published dogleg example. Assert to 0.01 m.
- **Segmentation**: property test asserting no generated composite spans more than one
  value of any `hard_break_on` attribute. This is the correctness guarantee that matters
  most, so it gets an explicit test.
- **Mass**: assert against the kg/m table in section 5.6.
- **Allocation**: synthetic block model with known tonnage split; assert proportional
  allocation reproduces the split exactly and that rounding sums to `total_samples`.
- **Selection**: synthetic case with a known optimal answer; assert greedy finds it, and
  assert determinism under a fixed seed.
- **End to end**: run the full pipeline on committed synthetic fixtures, assert output
  schema and that no pick list interval appears in two composites.

Target: 80% line coverage on the library, 100% on `desurvey`, `segment` and `mass`.

---

## 8. Milestones

| M | Deliverable | Definition of done |
|---|---|---|
| M1 | Config, IO, validation | Reference dataset loads, validation report generated, all schema checks pass |
| M2 | Desurvey and interval merge | Desurveyed interval table with XYZ, hand-calc tests pass |
| M3 | Block model and period assignment | Every interval carries bm_domain, period, domain_match |
| M4 | Segmentation, mass, candidates | Candidate table generated, no candidate crosses a hard break |
| M5 | Allocation and selection | Full pipeline produces picks.xlsx with pick list and gap register |
| M6 | Reporting and plots | Charts and formatted workbook |
| M7 | UI (optional) | Streamlit or FastAPI front end calling the library unchanged |

M1 to M5 is the usable product. M6 makes it presentable. M7 makes it distributable.

---

## 9. Open decisions

1. **Sub-blocked block models.** v1 assumes regular parent blocks. If production models are
   sub-blocked, the cKDTree nearest-centroid approach needs a containment test against
   variable block sizes, and tonnage aggregation must respect sub-block volumes.
2. **Multi-hole composites.** Off by default. If the gap register shows this is the only
   way to cover thin domains, enable per domain and flag the resulting samples clearly.
3. **Sample ID source.** The reference assay file has no sample ID column. If the source
   database has one, it must be carried through, as the pick list is worthless without it.
4. **Grade bin definition.** Currently on the primary element only. Multi-element binning
   (for example Zn bin x Pb:Zn ratio) may be needed for flotation representativity.
5. **Core loss factor.** 0.90 is a placeholder. Should be calibrated against actual
   recovered mass from previous programmes.
6. **Period granularity.** Annual periods assumed. If the schedule is monthly or quarterly,
   period weights need to be defined at that resolution or aggregated up.
7. **Existing testwork with partial coverage.** A prior sample that only had comminution
   done should probably count partially toward a comminution plus flotation deficit.
   Currently counted as full coverage.
8. **Attribute role vocabulary.** Role names are free-form. Consider a small reserved set
   (`weathering`, `rock_type`) that reporting can style specially, while still allowing
   arbitrary additions.
9. **Multi-file inputs.** Some sites split drilling across several exports by campaign. v1
   takes one path per source. A list of paths with a concatenation step is a small change
   if needed.
10. **Non-CSV sources.** The reader layer is deliberately isolated so that acQuire, Fusion
    or a database connector can be added behind the same interface without touching the
    pipeline.
