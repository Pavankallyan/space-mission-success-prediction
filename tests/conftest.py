"""Shared fixtures for the pipeline tests."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from pipeline import FEATURE_COLUMNS, TARGET_COLUMN


@pytest.fixture
def synthetic_missions():
    """Small synthetic mission dataset mirroring the modeling CSV schema."""
    rng = np.random.default_rng(7)
    n = 80
    df = pd.DataFrame(
        {
            "company": rng.integers(0, 10, n),
            "loc_city": rng.integers(0, 20, n),
            "loc_state": rng.integers(0, 8, n),
            "loc_country": rng.integers(0, 5, n),
            "rocket": rng.integers(1, 12, n).astype(float),
            "year": rng.integers(2000, 2025, n).astype(float),
            "status_rocket": rng.integers(0, 2, n),
        }
    )
    # success mostly when the rocket status is good -> learnable signal
    df[TARGET_COLUMN] = (
        (df["status_rocket"] == 1) & (rng.random(n) > 0.2)
    ).astype(int)
    # a few missing years, like the real modeling dataset
    df.loc[df.sample(5, random_state=7).index, "year"] = np.nan
    return df
