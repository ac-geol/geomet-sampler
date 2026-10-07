# TODO

Planned upgrades and follow-ups. This is a working list, not a design record: the
reasoning behind each design question lives in [SPEC.md section 9](SPEC.md#9-open-decisions),
and the parameters that need calibrating are listed under "Known placeholders" in the
[README](../README.md#known-placeholders). When an item here is decided, record the
decision in `SPEC.md` and delete the line from this file.

## Next

- [ ] Run `geomet-sampler init --samples` and then `validate` on a real desurveyed export.
      This tests the column guessing and the 5% geometry tolerance. Keep the export in
      `data/`, which git ignores.
- [ ] CI runs on `ubuntu-latest`, which moves to Ubuntu 26 from 2026-10-19. If CI breaks,
      pin the runner in `.github/workflows/ci.yml`.

## Input robustness (review of 2026-10-06)

Found by loading altered copies of the example data. Column names and units generalise;
the values inside the files do not yet. The first four are fixes, in priority order.
Numeric parsing (null tokens, detection limits, negative grades) is done: SPEC 3.3a.

- [ ] Check hole IDs across files: availability, existing testwork and planned holes
      against the drilling. Lowercase IDs in the availability file made every hole
      UNAVAILABLE, and the only sign was a hole count in the gap register.
- [ ] Enforce `COLLAR_REQUIRED`, `SURVEY_REQUIRED` and the other lists in `models.py`
      (only the samples reader does today; an unmapped `rl` raises a bare `KeyError`),
      and reject mapping keys that are not canonical names (`total_dept:` is ignored).
- [ ] Add a gap entry when a run yields no candidates, counting each rejection reason:
      mean outside the grade bin edges, CV above `max_cv_primary`, bin mismatch with the
      run. These windows are discarded in `composite/candidates.py` with no record.
- [ ] ERROR when a declared `attributes.<role>.drillhole` column is in no file. Today it is
      skipped, and hard breaks fall back to the modelled value without saying so. Treat
      a blank logged value as missing, not as a value of `""` that creates a break.
- [ ] Blank logged codes: the litho file raises `logged_code_unmapped: ['']` while the
      samples table treats them as unlogged. The two layouts must agree.
- [ ] Block model periods: `excluded_periods: ["0"]` does not exclude `0.0`, and codes
      such as `0` or `-99` count as scheduled tonnage unless listed. Match integral
      floats as `period_weight` does, and report tonnage by period so a sentinel is
      visible.
- [ ] Unparseable coordinates: a block X that will not parse crashes in scipy, and a
      collar easting that will not parse makes the hole look "outside model". Report
      both by name.
- [ ] Over-limit values (`>30`) are an ERROR. Decide how to read them: the limit, or
      blank and reported.
- [ ] Core diameter written as a size code (`HQ`, `NQ`) is an ERROR. Map codes to mm in
      config, rather than declaring them as null and silently using the default diameter.
- [ ] `init` could list the non-numeric tokens it finds in each numeric column, as a
      draft `null_values` marked `# CONFIRM`.
- [ ] Read options per source: `delimiter`, `skip_rows` (header blocks in Datamine,
      Surpac and Vulcan exports) and `decimal` (decimal-comma files).
- [ ] Length units per source. Depths in feet with coordinates in metres is common, and
      today it shows only as a `beyond_total_depth` WARN.
- [ ] Azimuth reference: `true` adds magnetic declination, which converts magnetic to
      true, not true to grid. Needs grid convergence and local-grid rotation. Geological
      decision on what to support.
- [ ] Block models with one block size in a header rather than `dx/dy/dz` columns, and
      rotated models.
- [ ] Holes whose orientation is only in the collar file (dip and azimuth, no survey
      rows). Today they are dropped with a WARN.
- [ ] Samples table with only midpoint, or only from and to, coordinates. Today all nine
      are required.

## Waiting on real data

- [ ] Assign long intervals to more than one block. Today each interval takes the block
      that contains its midpoint. SPEC 5.4 suggests sub-sampling the interval and taking
      the modal block once interval length approaches block height. Decide whether this is
      needed once real interval lengths and block sizes are known.
- [ ] Sub-blocked block models (SPEC 9.1).
- [ ] Carry a sample ID column through to the pick list if the source database has one
      (SPEC 9.3).
- [ ] Calibrate `mass.loss_factor` against recovered mass from a past programme
      (SPEC 9.5).
- [ ] Period granularity: monthly or quarterly schedules (SPEC 9.6).

## Design questions (see SPEC section 9)

- [ ] Multi-hole composites, enabled per domain and flagged clearly (9.2).
- [ ] Grade bins on more than one element, for example Zn bin x Pb:Zn ratio (9.4).
- [ ] Count prior testwork partially by test type, and by the fraction of core used,
      instead of all or nothing (9.7).
- [ ] A small reserved set of attribute roles that reporting can style (9.8).
- [ ] A list of paths per source, concatenated at load (9.9).
- [ ] Non-CSV sources such as acQuire, Fusion or a database, added behind the reader
      interface (9.10).

## Later

- [ ] M7: an optional UI (Streamlit or FastAPI) that calls the library unchanged
      (SPEC section 8).
- [ ] 3D visualisation (out of scope for v1, SPEC section 2).
