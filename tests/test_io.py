"""Parser tests: the real export columns map to the canonical schema."""

import math

from redoxquant import read_csv, schema


def test_parses_all_rows(demo_csv_path):
    df = read_csv(demo_csv_path)
    assert len(df) == 12
    assert list(df.columns) == schema.CANONICAL_COLUMNS


def test_compensation_parsed_from_x_prefix(demo_csv_path):
    df = read_csv(demo_csv_path)
    assert df[schema.SIGNAL_COMPENSATION].eq(1.003).all()


def test_concentration_value_and_unit_split(demo_csv_path):
    df = read_csv(demo_csv_path)
    first = df.iloc[0]
    assert math.isclose(first[schema.CONCENTRATION], 39.0)
    assert math.isclose(first[schema.ADJUSTED_CONCENTRATION], 38.8)
    assert first[schema.UNIT] in {"µg/ml", "ug/ml"}


def test_duration_parsed_to_seconds(demo_csv_path):
    df = read_csv(demo_csv_path)
    assert df[schema.DURATION_S].eq(33).all()


def test_solution_type_and_tags(demo_csv_path):
    df = read_csv(demo_csv_path)
    assert df[schema.SOLUTION_TYPE].eq(schema.SolutionType.SAMPLE.value).all()
    assert set(df[schema.TAG].dropna()) == {"mid", "hi"}
