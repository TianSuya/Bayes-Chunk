import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "segmented_editevery.json"
OUT = ROOT / "website" / "static" / "data" / "cases.json"
OUT_JS = ROOT / "website" / "static" / "data" / "cases.js"


CASE_META = {
    245: {
        "title": "Preserves peptidoglycan terms",
        "tag": "Term boundary",
        "takeaway": "Bayes-Chunk preserves the biological term peptidoglycan inside the cell-wall discussion.",
        "highlight": "peptidoglycan",
    },
    212: {
        "title": "Keeps glycolysis compounds intact",
        "tag": "Process step",
        "takeaway": "Bayes-Chunk keeps biochemical entities readable while separating the ATP accounting steps.",
        "highlight": "glyceraldehyde-3-phosphate",
    },
    213: {
        "title": "Keeps isotope names readable",
        "tag": "Symbol sequence",
        "takeaway": "Bayes-Chunk groups the decay-chain sequence around semantic transition points.",
        "highlight": "polonium-218",
    },
    215: {
        "title": "Keeps chemical identities intact",
        "tag": "Formula identity",
        "takeaway": "Bayes-Chunk separates ammonium and nitrate reasoning without breaking chemical identities.",
        "highlight": "NH4NO3",
    },
    363: {
        "title": "Separates formula, values, and computation",
        "tag": "Numeric formula",
        "takeaway": "Bayes-Chunk aligns boundaries with the Clausius-Clapeyron equation, given values, and final computation.",
        "highlight": "ln(P2/P1) = -ΔHvap/R",
    },
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\ufffd", "")).strip()


def split_words(text: str):
    return re.findall(r"\S+\s*", text.replace("\ufffd", ""))


def clean_segment(text: str) -> str:
    return text.replace("\ufffd", "")


def segment_lengths(segments):
    return [len(split_words(seg)) for seg in segments]


def cumulative_boundaries(lengths):
    total = 0
    bounds = []
    for length in lengths[:-1]:
        total += length
        bounds.append(total)
    return bounds


def synthetic_surprisal(words, bayes_bounds):
    """Small deterministic curve for visual storytelling.

    The real paper computes token-level surprisal with the LM. The website data
    keeps this deployable by using a stable proxy curve shaped by the stored
    Bayes boundaries; real values can be swapped in without changing the UI.
    """
    values = []
    boundary_set = set(bayes_bounds)
    for i, raw in enumerate(words):
        word = raw.strip()
        base = 1.8 + 0.45 * math.sin(i * 0.43) + 0.25 * math.cos(i * 0.19)
        if re.search(r"[=^*/×+-]|Δ|γ|²|\\d", word):
            base += 1.0
        if len(word) > 11:
            base += 0.55
        if i in boundary_set or i + 1 in boundary_set:
            base += 2.2
        values.append(round(max(base, 0.35), 2))
    return values


def as_case(index, item):
    meta = CASE_META[index]
    answer = item["original_answer"]
    words = split_words(answer)
    fixed_lengths = segment_lengths(item["fixed_segments"])
    bayes_lengths = segment_lengths(item["bayes_segments"])
    fixed_bounds = cumulative_boundaries(fixed_lengths)
    bayes_bounds = cumulative_boundaries(bayes_lengths)
    return {
        "source_index": index,
        "source_id": item["id"],
        "title": meta["title"],
        "tag": meta["tag"],
        "takeaway": meta["takeaway"],
        "highlight": meta["highlight"],
        "question": item["original_question"],
        "answer_preview": normalize_space(answer)[:280] + "...",
        "fixed_segments": [clean_segment(x) for x in item["fixed_segments"]],
        "bayes_segments": [clean_segment(x) for x in item["bayes_segments"]],
        "fixed_boundaries": fixed_bounds,
        "bayes_boundaries": bayes_bounds,
        "tokens": [w.strip() for w in words],
        "surprisal": synthetic_surprisal(words, bayes_bounds),
        "stats": {
            "tokens": len(words),
            "fixed_segments": len(item["fixed_segments"]),
            "bayes_segments": len(item["bayes_segments"]),
        },
    }


def main():
    with SOURCE.open(encoding="utf-8") as f:
        data = json.load(f)

    cases = [as_case(index, data[index]) for index in CASE_META]
    payload = {
        "note": "Demo data derived from data/segmented_editevery.json. Surprisal values are deterministic display proxies; replace with model-computed values when available.",
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with OUT_JS.open("w", encoding="utf-8") as f:
        f.write("window.ANYEDIT_CASES = ")
        json.dump(payload, f, ensure_ascii=False)
        f.write(";\n")
    print(f"Wrote {OUT.relative_to(ROOT)} with {len(cases)} cases")


if __name__ == "__main__":
    main()
