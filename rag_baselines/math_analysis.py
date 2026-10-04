"""Accounting identities and a stated two-stage input-volume bound; no inference."""
import math


def evidence_decomposition(observations):
    """Decompose binary (candidate coverage, selected coverage, answer EM) rows.

    Coverage means that at least one complete reference evidence set is present.
    Answers may be correct without reference coverage. Undefined conditional
    rates remain None; their zero-mass weighted contributions are exactly zero.
    """
    observations = list(observations)
    if not observations:
        raise ValueError("At least one observation is required.")
    for row in observations:
        if len(row) != 3 or any(type(value) not in (bool, int) or value not in (0, 1) for value in row):
            raise ValueError("Coverage and EM observations must be binary triples.")
        if row[1] > row[0]:
            raise ValueError("Selected evidence cannot cover more than its candidate pool.")
    count = len(observations)
    candidates = sum(row[0] for row in observations)
    evidence = sum(row[1] for row in observations)
    covered_correct = sum(row[1] and row[2] for row in observations)
    uncovered_correct = sum(not row[1] and row[2] for row in observations)
    return {
        "observations": count,
        "candidate_hits": candidates,
        "evidence_hits": evidence,
        "covered_correct": covered_correct,
        "uncovered_correct": uncovered_correct,
        "candidate_rate": candidates / count,
        "evidence_rate": evidence / count,
        "evidence_given_candidate": evidence / candidates if candidates else None,
        "correct_given_evidence": covered_correct / evidence if evidence else None,
        "correct_without_evidence": uncovered_correct / (count - evidence) if evidence < count else None,
        "covered_em_contribution": covered_correct / count,
        "uncovered_em_contribution": uncovered_correct / count,
        "answer_em": sum(row[2] for row in observations) / count,
    }


def group_input_bound(n, group_size, selected_groups, summary_tokens, passage_tokens):
    """Bound input volume for packed groups of at most group_size passages.

    Every group summary is assessed; selected groups' originals are assessed
    once. The bound assumes constant input lengths including allocated instructions;
    using sample averages does not make it an upper bound on actual API usage. Shared prompts,
    retries, protocol overhead and variable lengths need separate measurement.
    This is neither an accuracy model nor a latency/FLOP prediction.
    """
    if any(type(value) is not int or value < 1 for value in (n, group_size, selected_groups)):
        raise ValueError("Corpus, group size and selected count must be positive integers.")
    if group_size > n:
        raise ValueError("Group size cannot exceed corpus size.")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in (summary_tokens, passage_tokens)):
        raise ValueError("Token constants must be finite and positive.")
    groups = (n + group_size - 1) // group_size
    if selected_groups > groups:
        raise ValueError("Selected group count exceeds available groups.")
    passages = min(n, selected_groups * group_size)
    return {
        "groups": groups,
        "inspected_passages_upper_bound": passages,
        "summary_tokens_total": summary_tokens * groups,
        "passage_tokens_upper_bound": passage_tokens * passages,
        "total_tokens_upper_bound": summary_tokens * groups + passage_tokens * passages,
    }
