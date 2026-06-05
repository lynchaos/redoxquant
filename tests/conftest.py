from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def demo_csv_path():
    return DATA_DIR / "demo_tocilizumab.csv"


@pytest.fixture
def anchor_points():
    """Real (concentration, signal) pairs from the Amperia demo export.

    Twelve sample rows plus the three summary points (lo/mid/hi). Used to
    refit the descending standard curve the instrument applied.
    """
    sample_rows = [
        (39.0, 1187.4), (39.1, 1184.1), (37.2, 1222.2), (37.0, 1225.1),
        (38.0, 1206.0), (37.2, 1222.2), (36.8, 1229.5), (37.5, 1215.1),
        (81.9, 784.7), (80.9, 789.1), (76.5, 810.9), (76.5, 811.1),
    ]
    summary = [(9.45, 2547.8), (38.0, 1206.0), (77.1, 807.8)]
    return sample_rows + summary
