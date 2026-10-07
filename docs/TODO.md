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
