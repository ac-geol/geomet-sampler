"""Shared fixtures.

The fixture project is written once per session into a temporary directory. Building it
from seeded code rather than committed CSVs keeps the schema and the fixture in step.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geomet_sampler.config import Config, load_config
from geomet_sampler.pipeline import PipelineState, run_pipeline

from .fixtures import write_fixture_project


@pytest.fixture(scope="session")
def project_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("fixture_project")
    write_fixture_project(root)
    return root


@pytest.fixture
def cfg(project_root: Path) -> Config:
    return load_config(project_root / "project.yaml")


@pytest.fixture(scope="session")
def pipeline(project_root: Path) -> PipelineState:
    """Full pipeline run, shared across tests that only inspect the result."""
    return run_pipeline(load_config(project_root / "project.yaml"))
