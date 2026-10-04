"""Check the research identities where conditional probabilities are undefined."""
import math

import pytest

from rag_baselines.math_analysis import evidence_decomposition, group_input_bound


def test_correct_answer_without_reference_evidence_is_preserved():
    result = evidence_decomposition([(1, 1, 1), (1, 0, 1), (0, 0, 0), (1, 1, 0)])
    assert result["candidate_rate"] * result["evidence_given_candidate"] == result["evidence_rate"]
    reconstructed = (result["evidence_rate"] * result["correct_given_evidence"]
                     + (1 - result["evidence_rate"]) * result["correct_without_evidence"])
    assert reconstructed == result["answer_em"] == 0.5
    assert result["uncovered_em_contribution"] == 0.25


@pytest.mark.parametrize("rows,missing,expected", [
    ([(0, 0, 1)], ["evidence_given_candidate", "correct_given_evidence"], 1.0),
    ([(1, 1, 1)], ["correct_without_evidence"], 1.0),
])
def test_zero_mass_conditionals_remain_unknown(rows, missing, expected):
    result = evidence_decomposition(rows)
    assert all(result[name] is None for name in missing)
    assert result["covered_em_contribution"] + result["uncovered_em_contribution"] == expected


@pytest.mark.parametrize("rows", [[], [(0, 1, 1)], [(1, 1, 0.5)]])
def test_incompatible_observations_are_rejected(rows):
    with pytest.raises(ValueError):
        evidence_decomposition(rows)


def test_ragged_group_bound_cannot_assess_more_than_the_corpus():
    result = group_input_bound(11, 4, 3, 10, 20)
    assert result["groups"] == 3
    assert result["inspected_passages_upper_bound"] == 11
    assert result["total_tokens_upper_bound"] == 250
    with pytest.raises(ValueError, match="available groups"):
        group_input_bound(11, 4, 4, 10, 20)
    with pytest.raises(ValueError, match="finite"):
        group_input_bound(11, 4, 2, math.nan, 20)


def test_continuous_optimum_is_a_lower_bound_of_the_discrete_input_model():
    n, selected, a, b = 2000, 2, 40, 150
    continuous_lower_bound = 2 * math.sqrt(a * b * selected * n)
    candidates = [group_input_bound(n, g, selected, a, b)["total_tokens_upper_bound"]
                  for g in range(1, n // selected + 1)]
    assert min(candidates) >= continuous_lower_bound
    assert min(candidates) == 9800
