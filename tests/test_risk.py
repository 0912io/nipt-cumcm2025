import numpy as np

from src.revised.risk import decision_objective, raw_time_risk, time_risk


def _slope(function, time):
    return (function(time + 0.01) - function(time - 0.01)) / 0.02


def test_time_risk_is_normalized_and_steeper_later():
    assert abs(float(np.asarray(time_risk(10))) - 0) < 1e-12
    assert abs(float(np.asarray(time_risk(40))) - 1) < 1e-12
    assert float(time_risk(12)) < float(time_risk(20)) < float(time_risk(30))
    assert _slope(raw_time_risk, 30) > _slope(raw_time_risk, 20) > _slope(raw_time_risk, 11)
    assert _slope(time_risk, 30) > _slope(time_risk, 20) > _slope(time_risk, 11)


def test_equal_success_probability_prefers_earlier_time():
    early = float(decision_objective(11, 0.6))
    middle = float(decision_objective(18, 0.6))
    late = float(decision_objective(24, 0.6))
    assert early < middle < late
    assert float(decision_objective(18, 0.9)) < float(decision_objective(18, 0.4))
