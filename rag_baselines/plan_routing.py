"""Source-grounded, bounded one/two-edge plans; no query/gold-aware ingestion.

Facts are proposed by an offline model. Exact quotes certify provenance, not
semantic truth or completeness. Entity joins use exact normalized names only.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
import re
import time
import unicodedata

from .jev import call_usage


STOP = set("a an the of in on at for to and or is are was were be been do does did who what when where which how many much has have had by with from as it its that this their his her he she they you i than both name born".split())
PLAN_TASK = "Does this plan contain facts useful for answering the question? Partial evidence counts as useful; related subject matter alone does not."


def normalized(text):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


def terms(text):
    return set(normalized(text).split()) - STOP


def validated_facts(body, passages, max_facts):
    """Reject fabricated quotes/IDs, preserve rejection reasons instead of clipping."""
    if not isinstance(body, dict) or not isinstance(body.get("passages"), list):
        raise ValueError("Fact extraction must return a passages list.")
    originals = {p["id"]: p for p in passages}
    returned = [p.get("id") for p in body["passages"] if isinstance(p, dict)]
    if len(returned) != len(set(returned)) or set(returned) != set(originals):
        raise ValueError("Extraction returned missing/duplicate/unexpected passage IDs.")
    accepted, rejected = [], []
    for row in body["passages"]:
        facts = row.get("facts")
        if not isinstance(facts, list):
            raise ValueError("Extraction facts must be a list.")
        p = originals[row["id"]]
        for position, f in enumerate(facts):
            reason = None
            if position >= max_facts:
                reason = "exceeds_explicit_fact_budget"
            elif not isinstance(f, dict) or any(not isinstance(f.get(k), str) or not f[k].strip()
                                            for k in ("subject", "relation", "object", "quote")):
                reason = "missing_fact_fields"
            elif any(len(f[k]) > 160 for k in ("subject", "relation", "object")):
                reason = "fact_field_too_long"
            elif not isinstance(f.get("qualifiers", ""), str) or len(f.get("qualifiers", "")) > 160:
                reason = "invalid_qualifiers"
            elif f["quote"] not in p["content"]:
                reason = "quote_not_verbatim"
            elif not normalized(f["subject"]) or not normalized(f["object"]):
                reason = "empty_normalized_entity"
            elif normalized(f["object"]) not in normalized(f["quote"]):
                reason = "object_not_in_quote"
            elif normalized(f["subject"]) not in normalized(p["title"] + " " + f["quote"]):
                reason = "subject_not_source_grounded"
            if reason:
                rejected.append({"passage_id": p["id"], "position": position, "reason": reason, "fact": f})
                continue
            start = p["content"].index(f["quote"])
            clean = {k: f[k].strip() for k in ("subject", "relation", "object")}
            clean.update({"qualifiers": f.get("qualifiers", "").strip(), "quote": f["quote"],
                          "passage_id": p["id"], "start": start, "end": start + len(f["quote"])})
            clean["id"] = hashlib.sha256(json.dumps(clean, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
            accepted.append(clean)
    return list({f["id"]: f for f in accepted}.values()), rejected


class PlanIndex:
    def __init__(self, store, facts, policy):
        self.store, self.policy = store, policy
        self.facts = {f["id"]: f for f in facts}
        if len(self.facts) != len(facts):
            raise ValueError("Fact IDs must be unique.")
        self.adjacency, self.first_words = defaultdict(list), defaultdict(set)
        self.title_entities = defaultdict(set)
        frequencies = Counter()
        self.fact_terms, self.relation_terms = {}, {}
        for fact in facts:
            original = store.get(fact["passage_id"])
            if original["content"][fact["start"]:fact["end"]] != fact["quote"]:
                raise ValueError("Stored fact coordinates disagree with the original passage.")
            subject, obj = normalized(fact["subject"]), normalized(fact["object"])
            if not subject or not obj:
                raise ValueError("Empty normalized entity.")
            for entity in set((subject, obj)):
                self.adjacency[entity].append(fact["id"])
                alias = re.sub(r"^(the|an|a) ", "", entity)
                for name in set((entity, alias)):
                    words = name.split()
                    if words and terms(name) and not all(w.isdigit() for w in words):
                        self.first_words[words[0]].add((name, entity))
            self.title_entities[normalized(original["title"])].add(subject)
            words = terms(" ".join(fact[k] for k in ("subject", "relation", "object", "qualifiers")))
            self.fact_terms[fact["id"]] = words
            self.relation_terms[fact["id"]] = terms(fact["relation"] + " " + fact["qualifiers"])
            frequencies.update(words)
        self.idf = {w: math.log(1 + len(facts)/(1+n)) for w, n in frequencies.items()}
        self.title_words = {title: terms(title) for title in self.title_entities}

    def candidates(self, question):
        """Finite exact-entity paths; lexical ranking/caps can lose gold evidence."""
        q, qterms = normalized(question), terms(question)
        matches = {}
        for word in set(q.split()):
            for alias, entity in self.first_words.get(word, ()):
                if f" {alias} " in f" {q} ":
                    matches[entity] = max(matches.get(entity, 0), len(alias.split()))
        seeds = sorted(matches, key=lambda e: (-matches[e], -len(e), e))[:self.policy["max_seed_entities"]]
        seed_mode = "exact_entity"
        if not seeds:
            titles = sorted(self.title_words, key=lambda t: (-len(self.title_words[t] & qterms), t))
            seeds = [e for title in titles[:self.policy["fallback_title_count"]]
                     if self.title_words[title] & qterms for e in sorted(self.title_entities[title])]
            seeds = list(dict.fromkeys(seeds))[:self.policy["max_seed_entities"]]
            seed_mode = "lexical_title_fallback" if seeds else "no_seed"
        def fact_score(fid):
            return sum(self.idf[w] for w in self.fact_terms[fid] & qterms) + 2*sum(
                self.idf.get(w, 0) for w in self.relation_terms[fid] & qterms)
        plans, degree_capped, enumeration_capped = {}, [], False
        def neighbors(entity):
            values = self.adjacency.get(entity, [])
            if len(values) > self.policy["max_entity_degree"]:
                degree_capped.append(entity)
            return sorted(values, key=lambda fid: (-fact_score(fid), fid))[:self.policy["max_entity_degree"]]
        def add(ids, seed):
            nonlocal enumeration_capped
            key = tuple(sorted(ids))
            if key not in plans and len(plans) >= self.policy["max_enumerated_plans"]:
                enumeration_capped = True
                return
            score = sum(fact_score(fid) for fid in ids)/math.sqrt(len(ids))
            if key not in plans:
                plans[key] = {"id": hashlib.sha256("|".join(key).encode()).hexdigest()[:16],
                              "fact_ids": list(ids), "seed": seed, "lexical_score": score,
                              "passage_ids": list(dict.fromkeys(self.facts[fid]["passage_id"] for fid in ids))}
        for seed in seeds:
            for fid in neighbors(seed):
                add([fid], seed)
                f = self.facts[fid]
                endpoint = normalized(f["object"] if normalized(f["subject"]) == seed else f["subject"])
                # Do not traverse a generic/high-degree object as a semantic entity.
                if endpoint != seed and len(self.adjacency.get(endpoint, [])) <= self.policy["max_entity_degree"]:
                    for next_fid in neighbors(endpoint):
                        if next_fid != fid:
                            add([fid, next_fid], seed)
        ordered = sorted(plans.values(), key=lambda p: (-p["lexical_score"], len(p["fact_ids"]), p["id"]))
        chosen = ordered[:self.policy["max_candidate_plans"]]
        return chosen, {"seed_mode": seed_mode, "seeds": seeds, "enumerated_plans": len(ordered),
                        "candidate_plans": len(chosen), "candidate_cap_active": len(ordered) > len(chosen),
                        "enumeration_cap_active": enumeration_capped,
                        "degree_capped_entities": sorted(set(degree_capped)),
                        "candidate_passage_ids": list(dict.fromkeys(pid for p in chosen for pid in p["passage_ids"]))}

    def card(self, plan):
        return [{k: self.facts[fid][k] for k in ("subject", "relation", "object", "qualifiers")}
                for fid in plan["fact_ids"]]

    def packed_view(self, result):
        """A separate generator view; the retrieval contract remains full originals."""
        view = deepcopy(result)
        traces = []
        for passage in view["evidence"]:
            original = self.store.get(passage["id"])
            spans = []
            for fid in result["selected_fact_ids"]:
                f = self.facts[fid]
                if f["passage_id"] != passage["id"]:
                    continue
                # Preserve local context as verbatim characters, without generated paraphrases.
                margin = self.policy["quote_context_characters"]
                start, end = max(0, f["start"]-margin), min(len(original["content"]), f["end"]+margin)
                while start > 0 and original["content"][start-1] not in ".!?\n":
                    start -= 1
                while end < len(original["content"]) and original["content"][end-1] not in ".!?\n":
                    end += 1
                spans.append((start, end))
            merged = []
            for start, end in sorted(set(spans)):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            if not merged:
                raise ValueError("Selected original passage has no source-grounded fact span.")
            quotes = [original["content"][start:end] for start, end in merged]
            passage["content"] = original["title"] + ".\n" + "\n[…]\n".join(quotes)
            passage["generator_view"] = "verbatim_spans_with_original_title"
            traces.append({"passage_id": passage["id"], "title": original["title"],
                           "passage_character_spans": [[a,b] for a,b in merged], "quotes": quotes,
                           "original_source_spans": original["source_spans"]})
        view["packing_trace"] = traces
        return view


class PlanRetrieval:
    def __init__(self, index, client=None, *, provider="code", prices=None, audit=None):
        self.index, self.store, self.client = index, index.store, client
        self.provider, self.prices, self.audit = provider, prices, audit

    def retrieve(self, question, method):
        started = time.perf_counter()
        plans, diagnostics = self.index.candidates(question)
        candidate_seconds = time.perf_counter()-started
        calls = []
        if self.audit:
            self.audit.event("plan_candidates", "generated", {"plans": plans, "diagnostics": diagnostics})
        if plans and self.client is not None:
            questions = {p["id"]: {"type": "noul", "instructions": {
                "question": PLAN_TASK, "source_backed_plan": self.index.card(p)}} for p in plans}
            decisions = self.client.decide({"user_question": question}, questions, calls=calls,
                max_calls=self.index.policy["max_decision_calls"], stage="plan_selection",
                on_record=(lambda row: self.audit.event("plan_selection", "decision_call", row)) if self.audit else None)
            scores = {key: row["noul"] for key, row in decisions.items()}
            ordered = sorted(plans, key=lambda p: (-scores[p["id"]], -p["lexical_score"], p["id"]))
        else:
            scores = {p["id"]: p["lexical_score"] for p in plans}
            ordered = plans
        selection = ordered[:self.index.policy["selected_plans"]]
        fids = list(dict.fromkeys(fid for p in selection for fid in p["fact_ids"]))
        pids = list(dict.fromkeys(self.index.facts[fid]["passage_id"] for fid in fids))[:self.index.policy["evidence_k"]]
        fids = [fid for fid in fids if self.index.facts[fid]["passage_id"] in pids]
        usage, cost = call_usage(calls, self.prices)
        result = {"schema_version": 1, "method": method, "question": question,
                  "corpus_sha256": self.store.sha256,
                  "candidates": [{"id": pid} for pid in diagnostics["candidate_passage_ids"]],
                  "evidence": [deepcopy(self.store.get(pid)) for pid in pids],
                  "selected_fact_ids": fids, "plans": plans, "plan_scores": scores,
                  "selected_plan_ids": [p["id"] for p in selection], "diagnostics": diagnostics,
                  "provider": self.provider, "api_usage": usage if self.provider == "jev" else None,
                  "decision_usage": usage, "api_cost_usd": cost if self.provider == "jev" else 0.0,
                  "timings": {"candidate_seconds": candidate_seconds, "total_seconds": time.perf_counter()-started},
                  "calls": calls, "logical_judgment_stages": int(bool(calls))}
        result["seconds"]={"candidate_generation":candidate_seconds,
                           "retrieval_total":result["timings"]["total_seconds"]}
        self.store.validate_evidence(result)
        return result
