"""
preprocess_ramdocs.py
─────────────────────
Chuyển đổi RAMDocs_test.jsonl → pipeline input schema (schema_v1_method1_merged).

RAMDocs raw format (mỗi dòng):
{
  "question":        str,
  "documents":       [{"text": str, "type": "correct"|"misinfo"|"noise", "answer": str}, ...],
  "disambig_entity": [str, ...],
  "gold_answers":    [str, ...],
  "wrong_answers":   [str, ...]
}

Output (4 file):
  queries.jsonl    — Query[]          — feed vào pipeline
  documents.jsonl  — Document[]       — feed vào pipeline (Phase 1 input)
  claims.jsonl     — Claim[]          — feed vào pipeline (Phase 1 output / Phase 2 input)
  metadata.jsonl   — ground truth     — CHỈ dùng evaluation, KHÔNG feed pipeline

Claim extraction (Phase 1 mock):
  Mỗi Document → split thành sentences → mỗi sentence là 1 (evidence, claim_text) pair:
    evidence   = raw sentence trích thẳng từ Document.text (không sửa)
    claim_text = sentence được normalize thành atomic declarative statement
  Phase 2 canonical được chạy inline:
    - Gộp claims có overlap cao VÀ cùng temporal/numerical values
    - Giữ riêng nếu khác số (nguồn gốc conflict)
"""

import json
import hashlib
import re
import argparse
from pathlib import Path
from copy import deepcopy
from typing import Iterator
import sys


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

MIN_SENTENCE_LEN   = 10    # chars — bỏ sentences quá ngắn
MAX_DOC_LENGTH     = 512   # chars — truncate doc trước khi extract
CANONICAL_OVERLAP  = 0.55  # cosine sim proxy threshold cho Phase 2 merge
TAU_CONFIDENCE = {         # claim_confidence range theo doc_type
    "correct": (0.80, 0.95),
    "misinfo": (0.55, 0.75),
    "noise":   (0.40, 0.60),
    "unknown": (0.50, 0.70),
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def stable_id(prefix: str, *parts) -> str:
    """ID deterministic từ content — reproducible across runs."""
    raw = "_".join(str(p) for p in parts)
    h = hashlib.md5(raw.encode()).hexdigest()[:6]
    return f"{prefix}_{h}"


def normalize_text(text: str) -> str:
    """Strip whitespace thừa + normalize unicode quotes. Không lowercase."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text


def split_sentences(text: str) -> list[str]:
    """Split document text thành sentences. Giữ nguyên casing."""
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if len(s.strip()) >= MIN_SENTENCE_LEN]


def extract_numbers(text: str) -> set[str]:
    """Extract tất cả số (kể cả năm) từ text."""
    return set(re.findall(r'\b\d+\b', text))


def text_overlap(a: str, b: str) -> float:
    """Jaccard overlap trên word tokens — proxy cho cosine similarity."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def to_declarative(sentence: str) -> str:
    """
    Mock LLM rewrite: chuyển sentence → atomic declarative statement.
    Real pipeline: dùng LLM prompt. Mock: clean trailing noise + ensure period.
    INVARIANT: claim_text phải entail được từ evidence (sentence gốc).
    """
    s = sentence.strip()
    # Bỏ các prefix không mang thông tin
    s = re.sub(r'^(According to [^,]+,\s*|It is reported that\s*|Sources say\s*)', '', s, flags=re.IGNORECASE)
    s = s.strip()
    if s and not s.endswith('.'):
        s += '.'
    # Capitalize đầu câu
    if s:
        s = s[0].upper() + s[1:]
    return s


# ══════════════════════════════════════════════════════════════════════════════
# QUERY BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_query(raw: dict, query_idx: int) -> dict:
    """RAMDocs question → Query schema."""
    return {
        "query_id":   f"q_{query_idx:03d}",
        "user_query": normalize_text(raw["question"]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_documents(raw: dict, query_idx: int, skip_noise: bool = False) -> list[dict]:
    """
    RAMDocs documents list → Document[] schema.

    source_id encode vị trí (query_idx + doc_idx), KHÔNG encode doc_type
    → pipeline không leak ground truth qua source_id.

    retrieval_score: RAMDocs không có score thực → rank-based proxy [0.5, 1.0].
    """
    docs = []
    total = len(raw["documents"])

    for doc_idx, raw_doc in enumerate(raw["documents"]):
        doc_type = raw_doc.get("type", "unknown")

        if skip_noise and doc_type == "noise":
            continue

        text = normalize_text(raw_doc["text"])
        truncated = False
        if len(text) > MAX_DOC_LENGTH:
            text = text[:MAX_DOC_LENGTH].rsplit(" ", 1)[0] + " [TRUNCATED]"
            truncated = True

        # Rank-based proxy: rank 1 → 1.0, last → 0.5
        retrieval_score = round(1.0 - (doc_idx / max(total - 1, 1)) * 0.5, 4) if total > 1 else 1.0

        doc_id    = stable_id("d", query_idx, doc_idx, text[:40])
        source_id = f"s_q{query_idx:03d}_{doc_idx:02d}"

        docs.append({
            "doc_id":          doc_id,
            "source_id":       source_id,
            "text":            text,
            "retrieval_score": retrieval_score,
            "metadata": {
                "query_id":       f"q_{query_idx:03d}",
                "rank":           doc_idx + 1,
                "url":            None,
                "published_date": None,
                "_truncated":     truncated,   # internal flag, không thuộc schema
            },
        })

    return docs


# ══════════════════════════════════════════════════════════════════════════════
# CLAIM BUILDER — Phase 1 + Phase 2
# ══════════════════════════════════════════════════════════════════════════════

def build_claims_from_doc(
    doc: dict,
    doc_type: str,
    query_id: str,
    query_idx: int,
    doc_idx: int,
) -> list[dict]:
    """
    Phase 1 — Claim Extraction (mock):
      Document.text → split sentences → mỗi sentence cho ra 1 Claim:
        evidence   = raw sentence (trích thẳng, không sửa)
        claim_text = to_declarative(sentence) — mock LLM rewrite

    claim_confidence: phản ánh doc_type (correct > misinfo > noise).
    credibility_score: null — sẽ được compute tại Phase 5.
    """
    sentences = split_sentences(doc["text"])
    if not sentences:
        return []

    lo, hi = TAU_CONFIDENCE.get(doc_type, TAU_CONFIDENCE["unknown"])
    # Dùng hash để confidence deterministic (không dùng random)
    claims = []
    for sent_idx, sentence in enumerate(sentences):
        # Deterministic confidence từ hash của sentence
        h_val = int(hashlib.md5(sentence.encode()).hexdigest(), 16)
        confidence = round(lo + (h_val % 1000) / 1000 * (hi - lo), 4)

        # Mock embedding: None — real pipeline dùng BGE-M3
        # Để None thay vì random vector — tránh mislead downstream
        claim_id = f"c_q{query_idx:03d}_d{doc_idx:02d}_s{sent_idx:02d}"

        claims.append({
            "claim_id":            claim_id,
            "claim_text":          to_declarative(sentence),
            "evidence":            sentence,          # raw span từ Document.text
            "doc_id":              doc["doc_id"],
            "source_id":           doc["source_id"],
            "claim_embedding":     None,              # populated bởi BGE-M3 encoder sau
            "retrieval_relevance": doc["retrieval_score"],
            "claim_confidence":    confidence,
            "credibility_score":   None,              # populated tại Phase 5
            "is_representative":   True,              # sẽ cập nhật Phase 2
            "merged_claim_ids":    None,
            # Internal fields — dùng trong Phase 2, stripped trước khi save
            "_query_id":           query_id,
            "_doc_type":           doc_type,
        })

    return claims


def run_phase2_canonical(claims: list[dict]) -> list[dict]:
    """
    Phase 2 — Claim Canonical:
      - Group by query_id (chỉ canonical trong cùng query)
      - Nếu text_overlap > CANONICAL_OVERLAP VÀ cùng số → merge (đánh dấu representative)
      - Nếu khác số dù overlap cao → KHÔNG merge (đây là nguồn conflict)
    """
    merged_indices = set()
    result = []

    for i, ci in enumerate(claims):
        if i in merged_indices:
            continue
        group_ids = [ci["claim_id"]]

        for j, cj in enumerate(claims):
            if j <= i or j in merged_indices:
                continue
            if ci["_query_id"] != cj["_query_id"]:
                continue

            overlap = text_overlap(ci["claim_text"], cj["claim_text"])
            nums_i  = extract_numbers(ci["claim_text"])
            nums_j  = extract_numbers(cj["claim_text"])

            # Phase 2 split rule: sim cao nhưng khác số → KHÔNG merge
            if overlap > CANONICAL_OVERLAP and nums_i == nums_j:
                merged_indices.add(j)
                group_ids.append(cj["claim_id"])

        ci_out = deepcopy(ci)
        if len(group_ids) > 1:
            ci_out["merged_claim_ids"] = group_ids
        result.append(ci_out)

    return result


def build_claims(
    raw: dict,
    docs: list[dict],
    query_idx: int,
    doc_type_map: dict,   # doc_id → doc_type
    skip_noise: bool = False,
) -> list[dict]:
    """
    Tạo claims từ tất cả documents của 1 query, rồi chạy Phase 2 canonical.
    """
    query_id = f"q_{query_idx:03d}"
    all_claims = []

    for doc_idx, doc in enumerate(docs):
        doc_type = doc_type_map.get(doc["doc_id"], "unknown")
        if skip_noise and doc_type == "noise":
            continue
        claims = build_claims_from_doc(doc, doc_type, query_id, query_idx, doc_idx)
        all_claims.extend(claims)

    # Phase 2 canonical
    all_claims = run_phase2_canonical(all_claims)

    # Strip internal fields
    for c in all_claims:
        c.pop("_query_id", None)
        c.pop("_doc_type", None)

    return all_claims


# ══════════════════════════════════════════════════════════════════════════════
# METADATA BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_metadata(raw: dict, query_idx: int, docs: list[dict]) -> dict:
    """
    Ground truth — KHÔNG feed vào pipeline. Chỉ dùng evaluation.
    Map doc_id → doc_type/answer để đánh giá Phase 5 suppression accuracy.
    """
    raw_docs = raw.get("documents", [])
    doc_labels = []
    for doc, raw_doc in zip(docs, raw_docs):
        doc_labels.append({
            "doc_id":   doc["doc_id"],
            "doc_type": raw_doc.get("type", "unknown"),
            "answer":   raw_doc.get("answer", None),
        })

    return {
        "query_id":        f"q_{query_idx:03d}",
        "user_query":      normalize_text(raw["question"]),
        "disambig_entity": raw.get("disambig_entity", []),
        "gold_answers":    raw.get("gold_answers", []),
        "wrong_answers":   raw.get("wrong_answers", []),
        "doc_labels":      doc_labels,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class RAMDocsValidator:
    """
    Validate 4 output files sau preprocess.
    Checks:
      1. Mỗi query có ít nhất 1 correct document
      2. Mỗi document có ít nhất 1 claim được extract
      3. Tất cả claims có evidence field và evidence là substring của doc text
      4. Không có duplicate claim_id
      5. Doc text quá ngắn (< 20 chars)
    """

    def __init__(self, output_dir: str):
        self.dir = Path(output_dir)

    def _load(self, fname: str) -> list[dict]:
        return [json.loads(l) for l in open(self.dir / fname, encoding="utf-8") if l.strip()]

    def validate(self) -> list[str]:
        warnings = []

        meta_records  = self._load("metadata.jsonl")
        docs_records  = self._load("documents.jsonl")
        claim_records = self._load("claims.jsonl")

        # Build lookups
        doc_type_map  = {}
        for meta in meta_records:
            for lbl in meta["doc_labels"]:
                doc_type_map[lbl["doc_id"]] = lbl["doc_type"]

        # claims grouped by query
        claims_by_query: dict[str, list] = {}
        for c in claim_records:
            qid = "_".join(c["claim_id"].split("_")[:2])  # c_q001 → q_001
            # Re-extract query_id từ doc_id pattern
            # Dùng doc_id để lookup
            pass

        doc_text_map  = {}
        docs_by_query: dict[str, list] = {}
        for rec in docs_records:
            qid = rec["query_id"]
            docs_by_query[qid] = rec["documents"]
            for d in rec["documents"]:
                doc_text_map[d["doc_id"]] = d["text"]

        claims_by_doc: dict[str, list] = {}
        all_claim_ids = []
        for c in claim_records:
            claims_by_doc.setdefault(c["doc_id"], []).append(c)
            all_claim_ids.append(c["claim_id"])

        # Check 1: duplicate claim_id
        if len(all_claim_ids) != len(set(all_claim_ids)):
            warnings.append("DUPLICATE claim_ids detected across dataset.")

        for meta, docs_rec in zip(meta_records, docs_records):
            qid  = meta["query_id"]
            docs = docs_rec["documents"]

            # Check 2: ít nhất 1 correct doc
            correct_count = sum(
                1 for d in meta["doc_labels"] if d["doc_type"] == "correct"
            )
            if correct_count == 0:
                warnings.append(f"[{qid}] No correct document — arbitration has no ground truth.")

            for doc in docs:
                did = doc["doc_id"]

                # Check 3: doc text quá ngắn
                if len(doc["text"]) < 20:
                    warnings.append(f"[{qid}][{did}] Document text very short: '{doc['text'][:50]}'")

                # Check 4: doc có claims không
                if did not in claims_by_doc or not claims_by_doc[did]:
                    warnings.append(f"[{qid}][{did}] No claims extracted — document may be too short or all sentences filtered.")

                # Check 5: evidence là substring của doc text
                for c in claims_by_doc.get(did, []):
                    if not c.get("evidence"):
                        warnings.append(f"[{qid}][{c['claim_id']}] Missing evidence field.")
                    elif c["evidence"] not in doc_text_map.get(did, ""):
                        warnings.append(
                            f"[{qid}][{c['claim_id']}] evidence not found in doc text "
                            f"(may be truncated). evidence[:50]='{c['evidence'][:50]}'"
                        )

        return warnings


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PREPROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class RAMDocsPreprocessor:

    def __init__(
        self,
        input_path: str,
        output_dir: str  = "data/preprocessed",
        max_doc_length: int  = 512,
        skip_noise_docs: bool = False,
    ):
        self.input_path     = Path(input_path)
        self.output_dir     = Path(output_dir)
        self.max_doc_length = max_doc_length
        self.skip_noise     = skip_noise_docs
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_raw(self) -> Iterator[dict]:
        with open(self.input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def run(self) -> dict:
        stats = {
            "total_queries":   0,
            "total_docs":      0,
            "total_claims":    0,
            "truncated_docs":  0,
            "skipped_noise":   0,
            "canonical_merges": 0,
            "doc_type_counts": {"correct": 0, "misinfo": 0, "noise": 0},
        }

        q_path = self.output_dir / "queries.jsonl"
        d_path = self.output_dir / "documents.jsonl"
        c_path = self.output_dir / "claims.jsonl"
        m_path = self.output_dir / "metadata.jsonl"

        with (
            open(q_path, "w", encoding="utf-8") as fq,
            open(d_path, "w", encoding="utf-8") as fd,
            open(c_path, "w", encoding="utf-8") as fc,
            open(m_path, "w", encoding="utf-8") as fm,
        ):
            for query_idx, raw in enumerate(self._load_raw()):
                # ── Query ──────────────────────────────────────────────────
                query = build_query(raw, query_idx)

                # ── Documents ──────────────────────────────────────────────
                docs = build_documents(raw, query_idx, skip_noise=self.skip_noise)

                # Build doc_type_map từ raw (dùng để assign confidence)
                raw_docs = raw.get("documents", [])
                doc_type_map = {}
                for doc, raw_doc in zip(docs, raw_docs):
                    doc_type_map[doc["doc_id"]] = raw_doc.get("type", "unknown")

                # ── Claims (Phase 1 + Phase 2) ─────────────────────────────
                claims = build_claims(
                    raw, docs, query_idx, doc_type_map,
                    skip_noise=self.skip_noise,
                )

                # ── Metadata (ground truth) ────────────────────────────────
                meta = build_metadata(raw, query_idx, docs)

                # ── Write ──────────────────────────────────────────────────
                fq.write(json.dumps(query,  ensure_ascii=False) + "\n")
                fd.write(json.dumps({
                    "query_id":  query["query_id"],
                    "documents": docs,
                }, ensure_ascii=False) + "\n")
                for claim in claims:
                    fc.write(json.dumps(claim, ensure_ascii=False) + "\n")
                fm.write(json.dumps(meta, ensure_ascii=False) + "\n")

                # ── Stats ──────────────────────────────────────────────────
                stats["total_queries"] += 1
                stats["total_docs"]    += len(docs)
                stats["total_claims"]  += len(claims)
                stats["canonical_merges"] += sum(
                    1 for c in claims if c.get("merged_claim_ids")
                )

                for raw_doc in raw_docs:
                    t = raw_doc.get("type", "unknown")
                    if t in stats["doc_type_counts"]:
                        stats["doc_type_counts"][t] += 1
                    if t == "noise" and self.skip_noise:
                        stats["skipped_noise"] += 1
                    if len(raw_doc.get("text", "")) > self.max_doc_length:
                        stats["truncated_docs"] += 1

        return stats


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = Path(__file__).parent 
if DATA_DIR not in sys.path:
    sys.path.insert(0,DATA_DIR)

INPUT_DIR = DATA_DIR / "RAMDocs_test.jsonl"
OUTPUT_DIR = DATA_DIR / "preprocessed"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess RAMDocs → pipeline schema")
    parser.add_argument("--input" ,  default=str(INPUT_DIR), help="Path to input RAMDocs file")
    parser.add_argument("--output_dir",  default=str(OUTPUT_DIR))
    parser.add_argument("--max_doc_len", default=512, type=int)
    parser.add_argument("--skip_noise",  action="store_true")
    parser.add_argument("--validate",    action="store_true")
    args = parser.parse_args()

    print(f"[preprocess] Input:      {args.input}")
    print(f"[preprocess] Output dir: {args.output_dir}")
    print(f"[preprocess] max_doc_len={args.max_doc_len}  skip_noise={args.skip_noise}")

    preprocessor = RAMDocsPreprocessor(
        input_path=args.input,
        output_dir=args.output_dir,
        max_doc_length=args.max_doc_len,
        skip_noise_docs=args.skip_noise,
    )
    stats = preprocessor.run()

    print("\n[preprocess] Done.")
    print(f"  Queries:          {stats['total_queries']}")
    print(f"  Documents:        {stats['total_docs']}")
    print(f"  Claims:           {stats['total_claims']}")
    print(f"  Canonical merges: {stats['canonical_merges']}")
    print(f"  Truncated docs:   {stats['truncated_docs']}")
    print(f"  Doc types:        {stats['doc_type_counts']}")
    if args.skip_noise:
        print(f"  Skipped noise:    {stats['skipped_noise']}")

    if args.validate:
        print("\n[validate] Running validation...")
        v = RAMDocsValidator(args.output_dir)
        warnings = v.validate()
        if warnings:
            print(f"  {len(warnings)} warning(s):")
            for w in warnings:
                print(f"    ⚠  {w}")
        else:
            print("  All checks passed.")
