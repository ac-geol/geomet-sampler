"""A run must name the exact config and files it came from, and caches must respect that."""

from __future__ import annotations

import shutil

from geomet_sampler.config import load_config
from geomet_sampler.pipeline import cache_dir, run_pipeline

from .fixtures import write_fixture_project


def _copy(project_root, tmp_path):
    root = tmp_path / "project"
    shutil.copytree(project_root, root, ignore=shutil.ignore_patterns("output"))
    return root


def test_every_configured_input_is_hashed(cfg):
    expected = {
        name for name in type(cfg.sources).model_fields if getattr(cfg.sources, name) is not None
    }
    assert set(cfg.input_hashes) == expected
    assert all(len(v) == 16 for v in cfg.input_hashes.values())


def test_changing_a_data_file_changes_the_run_hash_but_not_the_config_hash(project_root, tmp_path):
    root = _copy(project_root, tmp_path)
    before = load_config(root / "project.yaml")
    availability = root / "data" / "availability.csv"
    availability.write_text(availability.read_text().replace("DISPOSED", "IN STORAGE"))
    after = load_config(root / "project.yaml")

    assert before.config_hash == after.config_hash
    assert before.input_hashes["availability"] != after.input_hashes["availability"]
    assert before.run_hash != after.run_hash


def test_the_cache_is_not_reused_after_the_data_changes(project_root, tmp_path):
    """--use-cache must never hand back a result computed from different files."""
    root = _copy(project_root, tmp_path)
    first = load_config(root / "project.yaml")
    run_pipeline(first, through="geometry")
    assert (cache_dir(first) / "geometry.pkl").exists()

    availability = root / "data" / "availability.csv"
    availability.write_text(availability.read_text().replace("DISPOSED", "IN STORAGE"))
    second = load_config(root / "project.yaml")
    assert cache_dir(second) != cache_dir(first)
    assert not (cache_dir(second) / "geometry.pkl").exists()


def test_the_summary_names_every_input_file_and_its_hash(pipeline):
    summary = pipeline.summary
    assert summary["run_hash"] == pipeline.cfg.run_hash
    for name, digest in pipeline.cfg.input_hashes.items():
        path = getattr(pipeline.cfg.sources, name).path
        assert summary[f"input:{name}"] == f"{path.name} sha256:{digest}"


def test_identical_projects_in_different_places_share_a_run_hash(tmp_path):
    a = load_config(write_fixture_project(tmp_path / "a"))
    b = load_config(write_fixture_project(tmp_path / "b"))
    assert a.run_hash == b.run_hash
