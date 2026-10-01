"""Shared fixtures: a tiny synthetic raw dataset so unit tests never need the network."""
import numpy as np
import pandas as pd
import pytest

from heart import config


@pytest.fixture
def raw_df():
    """12 rows in the raw UCI layout, incl. NaNs, a duplicate and an out-of-range row."""
    rows = [
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 2],
        [67, 1, 4, 120, 229, 0, 2, 129, 1, 2.6, 2, 2, 7, 1],
        [37, 1, 3, 130, 250, 0, 0, 187, 0, 3.5, 3, 0, 3, 0],
        [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, 0, 3, 0],
        [56, 1, 2, 120, 236, 0, 0, 178, 0, 0.8, 1, 0, 3, 0],
        [62, 0, 4, 140, 268, 0, 2, 160, 0, 3.6, 3, 2, 3, 3],
        [57, 0, 4, 120, 354, 0, 0, 163, 1, 0.6, 1, 0, 3, 0],
        [63, 1, 4, 130, 254, 0, 2, 147, 0, 1.4, 2, 1, 7, 2],
        [53, 1, 4, 140, 203, 1, 2, 155, 1, 3.1, 3, np.nan, 7, 1],   # missing ca
        [57, 1, 4, 140, 192, 0, 0, 148, 0, 0.4, 2, 0, np.nan, 0],   # missing thal
        [44, 1, 2, 999, 250, 0, 0, 170, 0, 0.0, 1, 0, 3, 0],        # trestbps out of range
    ]
    df = pd.DataFrame(rows, columns=config.RAW_COLUMNS)
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)        # + 1 exact duplicate


@pytest.fixture
def sample_patient():
    return {"age": 62, "sex": 1, "cp": 4, "trestbps": 150, "chol": 280, "fbs": 0,
            "restecg": 2, "thalach": 115, "exang": 1, "oldpeak": 2.8, "slope": 2,
            "ca": 2, "thal": 7}
