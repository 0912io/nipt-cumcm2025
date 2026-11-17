import math

import pytest

from src.revised.ga import parse_exam_date, parse_gestational_age, parse_pregnancy_count


def test_gestational_age_is_weeks_not_days():
    parsed = parse_gestational_age("11w+6")
    assert parsed == pytest.approx(11 + 6 / 7)
    assert parsed != 11 * 7 + 6
    assert parse_gestational_age("11W+6") == pytest.approx(parsed)
    assert parse_gestational_age("  11w + 6 ") == pytest.approx(parsed)
    assert parse_gestational_age("11w") == 11
    assert parse_gestational_age("11w+0") == 11
    assert parse_gestational_age("23w+4") == pytest.approx(23 + 4 / 7)


def test_invalid_gestational_age_is_missing():
    assert math.isnan(parse_gestational_age("11w+7"))
    assert math.isnan(parse_gestational_age(""))
    assert math.isnan(parse_gestational_age("不是孕周"))
    assert math.isnan(parse_gestational_age(83))
    assert math.isnan(parse_gestational_age(None))


def test_numeric_weeks_stay_weeks():
    assert parse_gestational_age(16.5) == pytest.approx(16.5)


def test_exam_date_and_pregnancy_code():
    assert parse_exam_date(20230401) == parse_exam_date("20230401")
    assert parse_pregnancy_count("≥3") == 3
    assert parse_pregnancy_count(2) == 2
    assert parse_pregnancy_count("1") == 1
    assert math.isnan(parse_pregnancy_count("  "))
