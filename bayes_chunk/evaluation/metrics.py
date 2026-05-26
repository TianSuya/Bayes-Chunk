from collections import Counter
import json
import math
import re
from pathlib import Path
from typing import Iterable, List, Sequence


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def sentence_bleu(reference: str, prediction: str, max_order: int = 4) -> float:
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0

    precisions = []
    for order in range(1, max_order + 1):
        ref_counts = ngram_counts(ref_tokens, order)
        pred_counts = ngram_counts(pred_tokens, order)
        overlap = sum((pred_counts & ref_counts).values())
        total = sum(pred_counts.values())
        precisions.append((overlap + 1.0) / (total + 1.0))

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)
    brevity_penalty = 1.0
    if len(pred_tokens) < len(ref_tokens):
        brevity_penalty = math.exp(1 - len(ref_tokens) / max(1, len(pred_tokens)))
    return brevity_penalty * geo_mean


def lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0
    lcs = lcs_length(ref_tokens, pred_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_f1(reference: str, prediction: str) -> float:
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum((ref_counts & pred_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def evaluate_records(records: Sequence[dict]) -> dict:
    bleu = []
    rouge_l = []
    f1 = []
    exact = []
    for record in records:
        answer = str(record.get("answer", ""))
        prediction = str(record.get("original_prediction", ""))
        bleu.append(sentence_bleu(answer, prediction))
        rouge_l.append(rouge_l_f1(answer, prediction))
        f1.append(token_f1(answer, prediction))
        exact.append(float(normalize_text(answer) == normalize_text(prediction)))
    return {
        "count": len(records),
        "bleu": mean(bleu),
        "rouge_l": mean(rouge_l),
        "token_f1": mean(f1),
        "exact_match": mean(exact),
    }


def evaluate_results_file(results_path: str | Path, output_path: str | Path | None = None) -> dict:
    with Path(results_path).open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    metrics = evaluate_records(records)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
