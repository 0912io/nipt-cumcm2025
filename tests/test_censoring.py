import numpy as np
import pandas as pd

from src.revised.censoring import build_woman_hits


def _visits(rows):
    frame = pd.DataFrame(rows)
    frame["exam_date"] = pd.to_datetime(frame["exam_date"])
    frame["height_cm"] = 160
    frame["weight_kg"] = 70
    frame["age"] = 30
    frame["bmi"] = 27.3
    return frame


def test_three_censoring_kinds_and_replicates_are_one_draw():
    visits = _visits(
        [
            {"woman_id": "L", "draw": 1, "ga_weeks": 12.0, "y_frac": 0.05, "exam_date": "2023-01-01"},
            {"woman_id": "I", "draw": 1, "ga_weeks": 11.0, "y_frac": 0.02, "exam_date": "2023-02-01"},
            {"woman_id": "I", "draw": 2, "ga_weeks": 15.0, "y_frac": 0.06, "exam_date": "2023-03-01"},
            {"woman_id": "R", "draw": 1, "ga_weeks": 12.0, "y_frac": 0.02, "exam_date": "2023-01-01"},
            {"woman_id": "R", "draw": 2, "ga_weeks": 20.0, "y_frac": 0.03, "exam_date": "2023-04-01"},
            {"woman_id": "M", "draw": 1, "ga_weeks": 12.0, "y_frac": 0.03, "exam_date": "2023-01-01"},
            {"woman_id": "M", "draw": 1, "ga_weeks": 12.0, "y_frac": 0.06, "exam_date": "2023-01-01"},
            {"woman_id": "D", "draw": 1, "ga_weeks": 12.0, "y_frac": 0.01, "exam_date": "2023-01-01"},
            {"woman_id": "D", "draw": 1, "ga_weeks": 12.0, "y_frac": 0.02, "exam_date": "2023-01-04"},
            {"woman_id": "D", "draw": 2, "ga_weeks": 16.0, "y_frac": 0.08, "exam_date": "2023-02-01"},
        ]
    )
    hits = build_woman_hits(visits).set_index("woman_id")
    assert set(hits.index) == {"L", "I", "R", "M", "D"}
    assert hits.loc["L", "kind"] == "left"
    assert hits.loc["L", "exact"] == False
    assert hits.loc["L", "right"] == 12
    assert hits.loc["I", "kind"] == "interval"
    assert hits.loc["I", "left"] == 11
    assert hits.loc["I", "right"] == 15
    assert hits.loc["R", "kind"] == "right"
    assert np.isinf(hits.loc["R", "right"])
    assert hits.loc["M", "kind"] == "left"
    assert hits.loc["M", "n_draws"] == 1
    assert hits.loc["D", "kind"] == "interval"
    assert hits.loc["D", "n_draws"] == 2
    assert hits.loc["D", "left"] == 12
    assert hits.loc["D", "right"] == 16
