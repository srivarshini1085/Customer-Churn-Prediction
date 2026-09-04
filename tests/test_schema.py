"""The input schema must stay in lock-step with the model's feature columns."""

from __future__ import annotations

from churn import config
from churn.data import clean, load_raw
from churn.schema import GROUPS, INPUT_FIELDS, defaults


def test_schema_covers_exactly_the_feature_columns():
    assert {f.name for f in INPUT_FIELDS} == set(config.FEATURE_COLUMNS)
    assert len(INPUT_FIELDS) == 19


def test_every_field_belongs_to_a_known_group():
    assert {f.group for f in INPUT_FIELDS} == set(GROUPS)


def test_choice_defaults_are_valid_options():
    for field in INPUT_FIELDS:
        if field.choices:
            assert str(field.default) in field.choices


def test_defaults_are_a_scorable_record():
    record = defaults()
    assert set(record) == set(config.FEATURE_COLUMNS)


def test_choice_options_match_values_seen_in_the_data():
    x, _ = clean(load_raw())
    for field in INPUT_FIELDS:
        if field.kind != "choice" or field.name == "SeniorCitizen":
            continue
        assert set(x[field.name].astype(str).unique()) <= set(field.choices)
