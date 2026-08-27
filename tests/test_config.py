"""Config validation: everything checkable before a data file is opened."""

from __future__ import annotations

import pytest
import yaml

from geomet_sampler.config import load_config


def write(tmp_path, cfg_dict) -> str:
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")
    return path


@pytest.fixture
def base(project_root):
    return yaml.safe_load((project_root / "project.yaml").read_text())


def test_the_fixture_config_is_valid(project_root):
    cfg = load_config(project_root / "project.yaml")
    assert cfg.primary_element == "Zn"
    assert cfg.config_hash


def test_config_hash_changes_when_the_config_changes(project_root, tmp_path, base):
    original = load_config(project_root / "project.yaml")
    base["allocation"]["total_samples"] = 999
    changed = load_config(write(tmp_path, base))
    assert changed.config_hash != original.config_hash


def test_exactly_one_primary_element_is_required(tmp_path, base):
    base["elements"]["Zn"]["primary"] = False
    with pytest.raises(ValueError, match="exactly one element must be flagged primary"):
        load_config(write(tmp_path, base))


def test_two_primary_elements_are_rejected(tmp_path, base):
    base["elements"]["Pb"] = {
        "primary": True,
        "drillhole": {"field": "Pb_pct", "units": "pct"},
    }
    with pytest.raises(ValueError, match="exactly one element"):
        load_config(write(tmp_path, base))


def test_grade_bin_element_must_be_declared(tmp_path, base):
    base["domaining"]["grade_bins"]["element"] = "Au"
    with pytest.raises(ValueError, match="not a declared element"):
        load_config(write(tmp_path, base))


def test_block_model_grade_bins_require_a_block_model_field(tmp_path, base):
    base["domaining"]["grade_bins"]["source"] = "block_model"
    base["elements"]["Zn"]["block_model"] = {"field": None, "units": None}
    with pytest.raises(ValueError, match="has no block_model field"):
        load_config(write(tmp_path, base))


def test_bin_edges_must_be_one_longer_than_labels(tmp_path, base):
    base["domaining"]["grade_bins"]["labels"] = ["LOW", "MED", "HIGH"]
    with pytest.raises(ValueError, match="len\\(edges\\)"):
        load_config(write(tmp_path, base))


def test_hard_break_on_must_name_a_declared_role(tmp_path, base):
    base["domaining"]["hard_break_on"] = ["geomet_domain", "not_a_role"]
    with pytest.raises(ValueError, match="not a declared attribute role"):
        load_config(write(tmp_path, base))


def test_allocation_key_roles_need_a_block_model_source(tmp_path, base):
    base["attributes"]["rock_type"]["block_model"] = None
    base["attributes"]["rock_type"]["drillhole"] = "SomeColumn"
    with pytest.raises(ValueError, match="must have a block_model source"):
        load_config(write(tmp_path, base))


def test_geomet_domain_cannot_be_an_allocation_key(tmp_path, base):
    """Logged domains cannot be aggregated over block tonnage."""
    base["domaining"]["allocation_key"] = ["geomet_domain", "grade_bin"]
    with pytest.raises(ValueError, match="cannot be aggregated over block model tonnage"):
        load_config(write(tmp_path, base))


def test_domain_match_attribute_must_resolve(tmp_path, base):
    base["domaining"]["domain_match_attribute"] = "nonsense"
    with pytest.raises(ValueError, match="not a declared attribute role"):
        load_config(write(tmp_path, base))


def test_min_mass_may_not_exceed_target_mass(tmp_path, base):
    base["compositing"]["min_mass_kg"] = 80.0
    with pytest.raises(ValueError, match="min_mass_kg must not exceed"):
        load_config(write(tmp_path, base))


def test_min_length_must_be_below_max_length(tmp_path, base):
    base["compositing"]["min_length_m"] = 50.0
    with pytest.raises(ValueError, match="min_length_m must be less than"):
        load_config(write(tmp_path, base))


def test_density_needs_a_source_or_a_constant(tmp_path, base):
    base["sources"]["assay"]["density"] = {"source": None, "fallback_constant": None}
    with pytest.raises(ValueError, match="`source` or `fallback_constant`"):
        load_config(write(tmp_path, base))


def test_an_element_field_must_declare_its_units(tmp_path, base):
    base["elements"]["Zn"]["drillhole"] = {"field": "Zn_pct", "units": None}
    with pytest.raises(ValueError, match="requires explicit `units`"):
        load_config(write(tmp_path, base))


def test_a_typo_in_a_config_key_is_rejected_rather_than_ignored(tmp_path, base):
    base["compositing"]["target_mass_kilos"] = 50.0
    with pytest.raises(ValueError, match="target_mass_kilos"):
        load_config(write(tmp_path, base))


def test_site_vocabulary_is_not_mangled_into_booleans(tmp_path, base):
    """YAML 1.1 would turn Y, YES and NO into booleans and break the value map."""
    base["sources"]["availability"]["value_map"] = {
        "AVAILABLE": ["YES", "Y", "IN STORAGE"],
        "UNAVAILABLE": ["NO", "N"],
        "PLANNED": ["PLANNED"],
    }
    path = tmp_path / "project.yaml"
    path.write_text(
        yaml.safe_dump(base).replace("- 'YES'", "- YES").replace("- 'NO'", "- NO"),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.sources.availability.value_map["AVAILABLE"] == ["YES", "Y", "IN STORAGE"]


def test_period_weights_accept_integer_keys_and_fall_back_to_default(project_root):
    cfg = load_config(project_root / "project.yaml")
    assert cfg.allocation.period_weight("1") == 2.0
    assert cfg.allocation.period_weight(1) == 2.0
    assert cfg.allocation.period_weight("99") == 1.0
    assert cfg.allocation.period_weight(None) == 1.0
