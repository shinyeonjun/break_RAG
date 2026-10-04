"""Pinned KILT download, gold-span mapping, and common corpus preparation.

Labels stay in separate files; neither retriever receives them online.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import random
import re
import time

import requests


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_wikipedia(root, workers=6, progress=print):
    """Resume bounded HTTP ranges of the official 35-GiB source on drive D.

    The final SHA-256 is a local integrity/reproduction hash, not a publisher
    checksum. Server size and ETag are checked against our pinned manifest.
    """
    root = Path(root)
    source = read_json(root / "config/baselines.json")["wikipedia"]
    directory = root / "data/raw/kilt_wikipedia"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "kilt_knowledgesource.json"
    state_path = directory / "download.json"
    block_bytes = 64 * 1024 * 1024
    identity = {key: source[key] for key in ("url", "bytes", "etag")}
    state = read_json(state_path) if state_path.exists() else {**identity, "completed": {}}
    if any(state.get(key) != value for key, value in identity.items()):
        raise ValueError("Download state differs from pinned source; use a separate raw directory.")
    if state.get("sha256") and destination.exists() and destination.stat().st_size == source["bytes"]:
        return destination
    head = requests.head(source["url"], timeout=30)
    head.raise_for_status()
    if int(head.headers["Content-Length"]) != source["bytes"] or head.headers.get("ETag") != source["etag"]:
        raise ValueError("Official source changed (size/ETag); do not mix versions.")
    if not destination.exists():
        with destination.open("wb") as stream:
            stream.truncate(source["bytes"])
    if destination.stat().st_size != source["bytes"]:
        raise ValueError("Partial file has the wrong size.")
    count = math.ceil(source["bytes"] / block_bytes)

    def fetch(index):
        start = index * block_bytes
        end = min(source["bytes"], start + block_bytes) - 1
        for attempt in range(4):
            try:
                with requests.get(source["url"], headers={"Range": f"bytes={start}-{end}"},
                                  timeout=(15, 60)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{source['bytes']}":
                        raise ValueError("Server did not honor the requested range.")
                    payload = response.content
                if len(payload) != end - start + 1:
                    raise ValueError("Incomplete HTTP range.")
                with destination.open("r+b") as stream:
                    stream.seek(start)
                    stream.write(payload)
                return str(index), hashlib.sha256(payload).hexdigest()
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    pending = [index for index in range(count) if str(index) not in state["completed"]]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, index) for index in pending]
        for future in as_completed(futures):
            index, checksum = future.result()
            state["completed"][index] = checksum
            write_json(state_path, state)
            completed = len(state["completed"])
            if progress and (completed % 8 == 0 or completed == count):
                progress(f"Wikipedia download: {completed}/{count} ranges ({completed / count:.1%})")
    state["sha256"] = file_sha256(destination)
    write_json(state_path, state)
    return destination


def chunk_article(article, words_per_passage=100):
    """BERGEN KILT100w text, plus original paragraph/character coordinates."""
    if words_per_passage <= 0:
        raise ValueError("words_per_passage must be positive.")
    paragraphs = article["text"]
    if isinstance(paragraphs, dict):
        paragraphs = paragraphs["paragraph"]
    words = []
    for paragraph_id, paragraph in enumerate(paragraphs):
        if "BULLET::::" in paragraph:
            continue
        for match in re.finditer(r"\S+", paragraph):
            words.append((match.group().replace("Section::::", "Section:"),
                          paragraph_id, match.start(), match.end()))
    passages = []
    wiki_id = str(article["wikipedia_id"])
    for offset in range(0, len(words), words_per_passage):
        selected = words[offset:offset + words_per_passage]
        spans = []
        for _, paragraph_id, start, end in selected:
            if spans and spans[-1]["paragraph_id"] == paragraph_id:
                spans[-1]["end_character"] = end
            else:
                spans.append({"paragraph_id": paragraph_id, "start_character": start, "end_character": end})
        passages.append({
            "id": f"{wiki_id}:{offset // words_per_passage}",
            "wikipedia_id": wiki_id,
            "title": article["wikipedia_title"],
            "content": article["wikipedia_title"] + ". " + " ".join(w[0] for w in selected),
            "source_spans": spans,
        })
    return passages


def provenance_options(question):
    """Alternative annotations; each annotation's spans are jointly required."""
    return [output["provenance"] for output in question["output"]
            if output["answer"].strip() and output.get("provenance")
            and all(p.get("bleu_score", 0) >= 0.99 and p.get("start_paragraph_id", -1) >= 0
                    for p in output["provenance"])]


def answer_provenances(question):
    return [p for option in provenance_options(question) for p in option]


def prepare_question_candidates(root):
    """Freeze deterministic train/dev and validation/eval candidate order."""
    import pyarrow.parquet as pq
    root = Path(root)
    config = read_json(root / "config/baselines.json")
    plans = {}
    tasks = config["qa"]["tasks"]
    for split, filename in (("dev", "train"), ("eval", "validation")):
        plans[split] = []
        for task_number, (task, task_config) in enumerate(tasks.items()):
            path = root / f"data/raw/kilt_tasks/{task}/{filename}-00000-of-00001.parquet"
            rows = pq.read_table(path).to_pylist()
            eligible = sorted([row for row in rows if any(
                len({p["wikipedia_id"] for p in option}) >= task_config["min_articles"]
                for option in provenance_options(row))], key=lambda row: row["id"])
            random.Random(config["seed"] + (split == "eval") + 100 * task_number).shuffle(eligible)
            target = task_config[f"{split}_count"]
            if len(eligible) < target:
                raise ValueError(f"Too few eligible {task}/{split} examples.")
            for row in eligible[:target + 100]:
                plans[split].append({**row, "original_id": row["id"], "id": f"{task}:{row['id']}", "task": task})
    write_json(root / "data/prepared/question_candidates.json", plans)
    return plans


def extract_articles(root, candidates, progress=print):
    """Read original JSONL once; retain needed articles and a uniform article sample.

    Background sampling is uniform over configured article indices, followed by
    uniform passage sampling in that pool. It is NOT uniform over all passages.
    """
    root = Path(root)
    config = read_json(root / "config/baselines.json")
    output = root / "data/prepared/articles.jsonl"
    wanted = {str(p["wikipedia_id"]) for rows in candidates.values()
              for row in rows for p in answer_provenances(row)}
    candidate_hash = hashlib.sha256(json.dumps(candidates, sort_keys=True).encode()).hexdigest()
    cache_meta = root / "data/prepared/articles_meta.json"
    if output.exists() and cache_meta.exists():
        meta = read_json(cache_meta)
        if meta["candidate_hash"] == candidate_hash and meta["config"] == config["wikipedia"]:
            return read_jsonl(output), meta
        raise ValueError("Article extraction cache differs from configuration.")
    wiki = config["wikipedia"]
    background = set(random.Random(config["seed"]).sample(range(wiki["articles"]), wiki["background_article_count"]))
    retained = []
    found = set()
    original_path = root / "data/raw/kilt_wikipedia/kilt_knowledgesource.json"
    # IDs occur before the large text field in the pinned official format.
    id_pattern = re.compile(rb'"wikipedia_id"\s*:\s*"?(\d+)"?')
    with original_path.open("rb") as stream:
        for index, line in enumerate(stream):
            match = id_pattern.search(line[:512])
            if match is None:
                raise ValueError(f"Unexpected source record at line {index}.")
            wiki_id = match[1].decode()
            if wiki_id in wanted or index in background:
                row = json.loads(line)
                retained.append({key: row[key] for key in ("wikipedia_id", "wikipedia_title", "text")})
                found.add(wiki_id)
            if progress and (index + 1) % 500000 == 0:
                progress(f"Article scan: {index + 1:,}/{wiki['articles']:,}")
    actual_count = index + 1
    if actual_count != wiki["articles"]:
        raise ValueError("Source article count changed; random sample is invalid.")
    meta = {"candidate_hash": candidate_hash, "config": wiki, "source_articles": actual_count,
            "retained_articles": len(retained), "missing_requested_ids": sorted(wanted - found),
            "source_sha256": read_json(original_path.parent / "download.json")["sha256"]}
    write_jsonl(output, retained)
    write_json(cache_meta, meta)
    return retained, meta


def map_provenance(provenance, article, passages):
    """Return ALL chunks intersecting annotated nonspace characters, or fail.

    A span crossing a chunk boundary needs both chunks. This is stricter than
    treating a hit on the right Wikipedia page as a complete evidence hit.
    """
    paragraphs = article["text"]
    if isinstance(paragraphs, dict):
        paragraphs = paragraphs["paragraph"]
    first, last = provenance["start_paragraph_id"], provenance["end_paragraph_id"]
    if not 0 <= first <= last < len(paragraphs):
        return None
    result = set()
    for paragraph_id in range(first, last + 1):
        text = paragraphs[paragraph_id]
        start = provenance["start_character"] if paragraph_id == first else 0
        end = provenance["end_character"] if paragraph_id == last else len(text)
        if not 0 <= start <= end <= len(text):
            return None
        needed = {i for i in range(start, end) if not text[i].isspace()}
        covered = set()
        for passage in passages:
            for span in passage["source_spans"]:
                if span["paragraph_id"] != paragraph_id:
                    continue
                overlap = needed.intersection(range(span["start_character"], span["end_character"]))
                if overlap:
                    result.add(passage["id"])
                    covered.update(overlap)
        if needed - covered:
            return None
    return sorted(result) or None


def build_common_corpora(root, candidates, articles, source_meta):
    """Gold-preserving nested N0/5N0/20N0, shared verbatim by A/B/later C."""
    root = Path(root)
    config = read_json(root / "config/baselines.json")
    by_article = {str(article["wikipedia_id"]): article for article in articles}
    chunks = {wiki_id: chunk_article(article, config["words_per_passage"])
              for wiki_id, article in by_article.items()}
    labels, online, exclusions = {}, {}, []
    for split, rows in candidates.items():
        labels[split], online[split] = [], []
        targets = {task: task_config[f"{split}_count"] for task, task_config in config["qa"].get("tasks", {"nq": config["qa"]}).items()}
        counts = defaultdict(int)
        for row in rows:
            task = row.get("task", "nq")
            if counts[task] >= targets[task]:
                continue
            groups = []
            for option in provenance_options(row):
                minimum = config["qa"].get("tasks", {}).get(task, {}).get("min_articles", 1)
                if len({p["wikipedia_id"] for p in option}) < minimum:
                    continue
                mapped = []
                for provenance in option:
                    wiki_id = str(provenance["wikipedia_id"])
                    mapped.append(map_provenance(provenance, by_article[wiki_id], chunks[wiki_id]) if wiki_id in by_article else None)
                if mapped and all(mapped):
                    group = sorted({pid for part in mapped for pid in part})
                    if group not in groups:
                        groups.append(group)
            if not groups:
                exclusions.append({"split": split, "id": row["id"], "reason": "no fully mappable answer span"})
                continue
            online[split].append({"id": row["id"], "question": row["input"], "task": task})
            labels[split].append({"id": row["id"], "task": task, "answers": sorted({out["answer"] for out in row["output"] if out["answer"].strip()}),
                                  "evidence_groups": groups})
            counts[task] += 1
        if dict(counts) != targets:
            raise ValueError(f"Too few mappable {split} questions: {dict(counts)}; need {targets}.")
    mandatory = {pid for rows in labels.values() for row in rows for group in row["evidence_groups"] for pid in group}
    all_passages = {p["id"]: p for rows in chunks.values() for p in rows}
    n0 = max(1000, math.ceil(len(mandatory) / 1000) * 1000)
    background_ids = sorted(set(all_passages) - mandatory)
    random.Random(config["seed"]).shuffle(background_ids)
    if len(all_passages) < 20 * n0:
        raise ValueError(f"Background sample too small: {len(all_passages)} < {20*n0}.")
    # Prefixes of one frozen background order make the three corpora nested.
    manifest = {"config": config, "source": source_meta, "mandatory_passages": len(mandatory),
                "n0": n0, "exclusions": exclusions, "corpora": {},
                "question_counts": {split: dict(Counter(row["task"] for row in rows)) for split, rows in online.items()},
                "sampling": "uniform article indices, then uniform passages within retained article pool; all gold spans forced in"}
    directory = root / "data/prepared"
    for multiplier in (1, 5, 20):
        size = multiplier * n0
        ids = sorted(mandatory | set(background_ids[:size - len(mandatory)]))
        path = directory / f"corpus_{size}.jsonl"
        write_jsonl(path, (all_passages[pid] for pid in ids))
        manifest["corpora"][str(size)] = {"path": str(path.relative_to(root)).replace("\\", "/"),
                                           "sha256": file_sha256(path), "count": size}
    for split in labels:
        write_jsonl(directory / f"questions_{split}.jsonl", online[split])
        write_jsonl(directory / f"labels_{split}.jsonl", labels[split])
        manifest[f"questions_{split}_sha256"] = file_sha256(directory / f"questions_{split}.jsonl")
        manifest[f"labels_{split}_sha256"] = file_sha256(directory / f"labels_{split}.jsonl")
    write_json(directory / "manifest.json", manifest)
    return manifest
