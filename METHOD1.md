# METHOD1.md — Conflict-Aware Evidence Arbitration with Factoid-Level Resolution

> **Đây là file hướng dẫn kỹ thuật cho Method 1.**
> Đọc toàn bộ file này trước khi implement bất kỳ module nào.
> Mọi quyết định về architecture, naming, data flow đều phải tuân theo file này.
> File này là **companion** của `CLAUDE.md` — không thay thế, chỉ bổ sung chi tiết cho Method 1.

---

## 0. TỔNG QUAN METHOD 1

**Method name:** Conflict-Aware Evidence Arbitration with Factoid-Level Resolution

**Paradigm:** `arbitrate-then-verify-then-generate`

**Đóng góp chính so với ArbGraph gốc:**
1. Bổ sung **factoid-level conflict localization** — không chỉ biết claim nào mâu thuẫn mà còn biết *slot nào* (temporal / numerical / entity / relation / location) bị conflict
2. Bổ sung **iterative retrieval loop** — dùng conflict localization để formulate targeted queries và retrieve thêm evidence
3. **Claim canonical** — merge claims tương đồng nhưng tách claims có temporal/numerical khác nhau để bảo toàn conflict information

**Luồng xử lý tổng quan:**
```
Documents
  → [Phase 1] Claim Extraction          (LLM + BGE-M3)
  → [Phase 2] Claim Canonical            (similarity merge + temporal/numerical split)
  → [Phase 3] Evidence Graph             (NLI-based edges: support/contradict/neutral)
  → [Phase 4] Query-Aware Retrieval      (Hybrid BM25+Dense + RRF + Balanced top-k)
  → [Phase 5] Conflict Zone              (Arbitration credibility + Factoid localization)
  → [Phase 6] Iterative Loop             (Conflict-aware query → retrieve → update graph)
  → [Phase 7] Generation                 (Grounded answer với conflict summary)
```

---

## 1. CẤU TRÚC THƯ MỤC (Method 1 specific)

```
project_root/
├── METHOD1.md                   ← file này
├── pipeline.py                  ← end-to-end orchestrator
├── schema/
│   └── models.py                ← Pydantic models (extend schema_v1.json)
│       ├── Document
│       ├── Claim
│       ├── Edge
│       ├── ConflictRegion
│       ├── ActionLabel
│       ├── FactoidSlots         ← MỚI: typed slot schema
│       ├── ConflictLocalization ← MỚI: factoid conflict result
│       └── LoopResult           ← MỚI: iterative loop output
├── claims/
│   ├── extractor.py             ← LLM-based atomic claim extraction
│   └── embedder.py              ← BGE-M3 embedding + Claim Canonical
├── graph/
│   └── graph_builder.py         ← NLI inference + pair gen + graph construction
├── retrieval/
│   └── hybrid_retriever.py      ← BM25 + Dense + RRF + BalancedTopKSelector
├── conflict/
│   └── conflict_zone.py         ← CredibilityArbitrator + FactoidDecomposer + ConflictZoneAnalyzer
├── loop/
│   └── iterative_loop.py        ← ConflictQueryFormulator + GraphUpdater + IterativeLoop
├── answer/
│   └── generator.py             ← EvidenceSelector + AnswerGenerator
└── tests/
    └── test_pipeline_smoke.py   ← 19 smoke tests (không cần LLM/GPU)
```

---

## 2. DATA SCHEMAS (Method 1 Extensions)

> Các schemas bên dưới **extend** schema_v1.json từ CLAUDE.md.
> Không thay đổi các schemas gốc (Document, Claim, Edge, ConflictRegion, ActionLabel).

### 2.1 FactoidSlots — typed slots của một claim

```json
{
  "temporal":        "string | null",   // năm, ngày, tháng
  "numerical":       "string | null",   // số, %, đơn vị
  "entity_subject":  "string | null",   // chủ thể chính
  "entity_object":   "string | null",   // đối tượng chính
  "relation":        "string | null",   // động từ / quan hệ
  "location":        "string | null"    // địa điểm
}
```

**Extraction strategy theo slot type:**

| Slot | Method | Lý do |
|------|--------|-------|
| `temporal` | Regex `\b(1[0-9]{3}\|20[0-9]{2})\b` | Deterministic, nhanh |
| `numerical` | Regex number pattern | Deterministic |
| `entity_subject` | LLM prompt | Cần ngữ nghĩa |
| `entity_object` | LLM prompt | Cần ngữ nghĩa |
| `relation` | LLM prompt | Cần ngữ nghĩa |
| `location` | LLM prompt | Cần ngữ nghĩa |

> **Rule:** Chỉ gọi LLM cho entity/relation/location. Temporal và numerical dùng rule-based để tránh LLM call không cần thiết.

### 2.2 ConflictLocalization — kết quả factoid-level analysis

```json
{
  "claim_i_id":          "string",   // claim_id của claim thứ nhất
  "claim_j_id":          "string",   // claim_id của claim thứ hai
  "slot":                "string",   // slot type bị conflict đầu tiên
  "value_i":             "string",   // giá trị từ claim_i
  "value_j":             "string",   // giá trị từ claim_j
  "conflict_intensity":  "float",    // |conflict_slots| / total_slots, [0.0, 1.0]
  "credibility_i":       "float",    // claim_credibility_score của claim_i
  "credibility_j":       "float"     // claim_credibility_score của claim_j
}
```

**Công thức conflict_intensity:**
```
conflict_slots = {slot: val_i ≠ val_j, val_i ≠ null, val_j ≠ null}
total_slots    = số slots có ít nhất một giá trị không null
conflict_intensity = |conflict_slots| / total_slots
```

> **Lưu ý:** Không dùng XOR(pi, pj) trực tiếp như ECON paper vì hai claims có thể conflict ở cùng slot (cả hai đều có giá trị nhưng khác nhau). Dùng set comparison thay thế.

### 2.3 LoopResult — output của iterative loop

```json
{
  "query_id":                "string",
  "iterations_run":          "int",
  "resolved":                "bool",
  "validated_claims":        "list[Claim]",
  "conflict_localizations":  "list[ConflictLocalization]",
  "final_answer":            "string | null"
}
```

---

## 3. CHI TIẾT TỪNG PHASE

---

### PHASE 1 — Claim Extraction
**File:** `claims/extractor.py` | **Class:** `ClaimExtractor`

#### Mô tả
Dùng LLM để viết lại content của document thành các **atomic statements dạng declarative**.
Mỗi claim phải: (1) self-contained, (2) ≤ 35 words, (3) không phải opinion/question.

#### Prompt template
```
System:
You are an expert at extracting atomic factual claims from text.
An atomic claim is a single, self-contained factual statement
that can be verified independently.

User:
Extract all atomic claims from the following document passage.
Return ONLY a JSON array of strings, one claim per element.
Do not include opinions, questions, or non-factual statements.
Each claim must be under 30 words and self-contained.

Document: {document_text}
```

#### Interface
```python
class ClaimExtractor:
    def extract(self, document: Document) -> list[Claim]:
        """
        Returns list[Claim] với embedding=None.
        Raises ValueError nếu document.text rỗng.
        Raises RuntimeError sau max_retries thất bại.
        """

    def extract_batch(self, documents: list[Document]) -> list[Claim]:
        """Batch extraction. Log error per doc, không crash toàn batch."""
```

#### Config keys (`configs/phase1_claims.yaml`)
```yaml
claim_extraction:
  model: "gpt-4o-mini"
  max_retries: 3
  backoff_base: 2.0
  max_claim_length_words: 35
```

#### Completion criteria
- [ ] Atomicity rate ≥ 97% (manual sample 100)
- [ ] Duplicate rate < 5% (cosine sim > 0.95)
- [ ] Invalid rate < 3%

---

### PHASE 2 — Claim Canonical
**File:** `claims/embedder.py` | **Classes:** `ClaimEmbedder`, `ClaimCanonical`

#### Mô tả
**ClaimEmbedder:** Encode claims bằng BGE-M3 (normalized, cosine-ready).

**ClaimCanonical:** Merge claims diễn đạt cùng sự kiện, nhưng **KHÔNG merge** claims có temporal/numerical values khác nhau — đây là nguồn gốc của conflict.

#### Điều kiện merge/tách

```
sim(ci, cj) > threshold AND vals_i == vals_j   → MERGE
sim(ci, cj) > threshold AND vals_i ≠ vals_j    → KHÔNG MERGE (giữ cả hai)
sim(ci, cj) ≤ threshold                         → KHÔNG MERGE
```

Trong đó `vals` = set các temporal/numerical tokens extract bằng regex.

#### Ví dụ
```
c1: "Điện Biên Phủ diễn ra năm 1954."  [sim=0.94 với c2]
c2: "Trận Điện Biên Phủ xảy ra vào 1954."
→ vals = {"1954"} cho cả hai → MERGE vào c1

c3: "Điện Biên Phủ diễn ra năm 1953."  [sim=0.91 với c1]
→ vals_c1 = {"1954"} ≠ vals_c3 = {"1953"} → KHÔNG MERGE
```

#### Interface
```python
class ClaimEmbedder:
    def embed(self, claims: list[Claim]) -> list[Claim]:
        """Populate embedding field in-place. Normalize to unit vector."""

class ClaimCanonical:
    def canonicalize(self, claims: list[Claim]) -> list[Claim]:
        """
        Trả về representative claims sau merge.
        Raises ValueError nếu bất kỳ claim nào thiếu embedding.
        """
```

#### Config keys
```yaml
claim_canonical:
  similarity_threshold: 0.92    # cosine threshold để candidate merge
  embedding_model: "BAAI/bge-m3"
  batch_size: 64
```

---

### PHASE 3 — Evidence Graph Construction
**File:** `graph/graph_builder.py` | **Classes:** `NLIInference`, `PairGenerator`, `ClaimGraphBuilder`

#### Mô tả
Xây dựng G = (V, E):
- **Node** = claim representative
- **Edge** = relation {support, contradiction, neutral} + nli_score

#### Quy trình tạo edge
```
1. PairGenerator: chỉ pair claims có cosine sim > threshold (tránh O(n²) full pairing)
2. NLIInference: predict (relation, confidence) cho mỗi pair
3. Quality check: loại edge nếu nli_score < τ_nli
4. Thêm edge vào graph với attributes
```

#### NLI label mapping
```python
LABEL_MAP = {
    "entailment":    "support",
    "contradiction": "contradiction",
    "neutral":       "neutral",
}
```

#### Interface
```python
class NLIInference:
    def predict(self, claim_a: str, claim_b: str) -> tuple[str, float]:
        """Returns (relation_label, confidence_score)."""

class PairGenerator:
    def generate(self, claims: list[Claim]) -> list[tuple[Claim, Claim]]:
        """Returns pairs với sim > threshold."""

class ClaimGraphBuilder:
    def build(self, claims: list[Claim]) -> tuple[nx.DiGraph, list[Edge]]:
        """Returns (graph, edges) theo schema_v1."""

    def serialize(self, graph: nx.DiGraph, query_id: str) -> dict:
        """Serialize to dict với keys: query_id, nodes, edges."""
```

#### Config keys
```yaml
graph:
  pair_similarity_threshold: 0.3    # cosine sim để generate pairs
  nli_model: "cross-encoder/nli-deberta-v3-base"
  nli_edge_threshold: 0.5           # loại edge nếu score dưới này
```

---

### PHASE 4 — Query-Aware Retrieval
**File:** `retrieval/hybrid_retriever.py` | **Classes:** `HybridRetriever`, `BalancedTopKSelector`

#### Mô tả
**HybridRetriever:** BM25 + Dense với RRF fusion.

```
claim_relevance_score(q, ri) = α · dense_score(q, ri) + (1-α) · bm25_score(q, ri)
RRF: score(d) = Σ 1/(k + rank_i(d))   với k=60
```

**BalancedTopKSelector:** Đảm bảo top-k có đủ conflict pairs — tránh chỉ retrieve claims tương đồng nhau.

#### Balanced selection logic
```
Pass 1: Greedy selection theo hybrid score
Pass 2: Nếu thiếu conflict pairs:
  - Tìm conflict partners của claims đã chọn
  - Swap ra non-conflict claim cuối cùng
  - Thêm vào conflict partner
```

#### Interface
```python
class HybridRetriever:
    def index(self, claims: list[Claim]) -> None:
        """Build BM25 index + store dense embeddings."""

    def retrieve(self, query: str, top_k: int | None = None
                 ) -> list[tuple[Claim, float]]:
        """Returns (Claim, relevance_score) sorted descending."""

class BalancedTopKSelector:
    def select(self,
               ranked_claims: list[tuple[Claim, float]],
               edge_index: dict[tuple[str, str], str]
               ) -> list[Claim]:
        """Select top-k với đảm bảo min_conflict_pairs."""
```

#### Config keys
```yaml
retrieval:
  top_k: 10
  alpha: 0.5                  # dense weight (1-alpha = BM25 weight)
  rrf_k: 60
  min_conflict_pairs: 1       # minimum conflict pairs trong top-k
  embedding_model: "BAAI/bge-m3"
```

---

### PHASE 5 — Conflict Zone
**File:** `conflict/conflict_zone.py` | **Classes:** `CredibilityArbitrator`, `FactoidDecomposer`, `ConflictZoneAnalyzer`

#### 5.1 — Claim Credibility Scoring (ArbGraph-style)

**Công thức:**
```
claim_credibility_score(ri) =
    Σ edge_score(rj→ri) [relation = support]
  - Σ edge_score(rk→ri) [relation = contradiction]
```

**Iterative update với damping:**
```
new_score(v) = damping · raw_score(v) + (1 - damping) · old_score(v)
```

Dừng khi max_delta < convergence_threshold hoặc đạt max_iterations.

**Output phân loại:**
```
credibility_score > 0  → validated
credibility_score ≤ 0  → suppressed
```

#### 5.2 — Factoid Decomposition

Với mỗi conflict pair (ri, rj), decompose từng claim thành FactoidSlots:

```
claim_i → Si = {temporal, numerical, entity_subject, entity_object, relation, location}
claim_j → Sj = {temporal, numerical, entity_subject, entity_object, relation, location}
```

**Aligned pairs theo schema:**
```
[(Si.temporal, Sj.temporal), (Si.entity_subject, Sj.entity_subject), ...]
```

**Gán nhãn per pair:**
```
temporal/numerical: string compare → contradict nếu val_i ≠ val_j
entity/relation/location: NLI hoặc string match sau normalize
```

**Binary vectors:**
```
pi[k] = 1 nếu slot k của claim_i bị conflict, else 0
pj[k] = 1 nếu slot k của claim_j bị conflict, else 0
```

**Conflict intensity:**
```
conflict_intensity = |conflict_slots| / total_slots
```

#### 5.3 — Output Phase 5

```python
@dataclass
class ConflictAnalysisResult:
    credibility_scores:      dict[str, float]          # claim_id → score
    conflict_localizations:  list[ConflictLocalization]
    validated_claim_ids:     list[str]
    suppressed_claim_ids:    list[str]
```

#### Interface
```python
class CredibilityArbitrator:
    def compute(self, graph: nx.DiGraph) -> dict[str, float]:
        """Returns claim_id → credibility_score."""

class FactoidDecomposer:
    def decompose(self, claim_text: str) -> FactoidSlots:
        """Extract typed slots từ claim text."""

class ConflictZoneAnalyzer:
    def analyze(self,
                graph: nx.DiGraph,
                claims: list[Claim]) -> ConflictAnalysisResult:
        """Run full conflict zone analysis."""
```

#### Config keys
```yaml
conflict_zone:
  arbitration_max_iter: 10
  arbitration_convergence: 0.01
  arbitration_damping: 0.85
  credibility_threshold: 0.0      # cutoff validated vs suppressed
  use_llm_for_entities: true      # False = chỉ dùng rule-based
```

---

### PHASE 6 — Iterative Retrieval Loop
**File:** `loop/iterative_loop.py` | **Classes:** `ConflictQueryFormulator`, `GraphUpdater`, `IterativeLoop`

#### Mô tả
Loop được trigger khi tồn tại conflict pairs chưa resolve.
Mỗi iteration: formulate targeted query → retrieve → update graph → re-analyze.

#### Targeted query templates theo slot type

| Slot | Query template |
|------|---------------|
| `temporal` | `"Verify: {query} Source A: {val_i}, Source B: {val_j}. [temporal conflict] Which year is correct?"` |
| `numerical` | `"Verify numerical: {query} One source: {val_i}, another: {val_j}."` |
| `entity_*` | `"Disambiguate entity: {query} Conflicting: '{val_i}' vs '{val_j}'."` |
| `location` | `"Verify location: {query} Source A: {val_i}, Source B: {val_j}."` |
| `relation` | `"Fact-check: {query} Conflicting claims: '{text_i}' vs '{text_j}'."` |

#### Stopping conditions

```
Condition A: max(conflict_intensity) < θ_intensity
             → "resolved" → break

Condition B: iterations ≥ MAX_ITERATIONS
             → "unresolvable" → break, flag as genuine disagreement

Condition C: new documents = 0 (không có evidence mới)
             → break

Condition D: intensity_reduction < min_intensity_reduction
             → convergence → break
```

**Quyết định generate:**
```
(credibility_score cao, conflict_intensity cao) → generate với conflict summary
(credibility_score thấp, conflict_intensity thấp) → generate với validated claims
intensity < θ_intensity → generate (resolved)
```

#### Interface
```python
class ConflictQueryFormulator:
    def formulate(self,
                  original_query: str,
                  localization: ConflictLocalization,
                  claim_texts: dict[str, str]) -> str:
        """Returns targeted query string theo slot type."""

class GraphUpdater:
    def update(self,
               graph: nx.DiGraph,
               new_docs: list[Document],
               existing_claims: list[Claim]) -> tuple[nx.DiGraph, list[Claim]]:
        """Extract + embed + canonical new claims, insert vào graph."""

class IterativeLoop:
    MAX_ITERATIONS = 3
    CONVERGENCE_THRESHOLD = 0.1

    def run(self,
            query: str,
            query_id: str,
            initial_docs: list[Document]) -> LoopResult:
        """Full iterative loop. Returns LoopResult."""
```

#### Config keys
```yaml
iterative_loop:
  max_iterations: 3
  intensity_threshold: 0.1
  min_intensity_reduction: 0.1
```

---

### PHASE 7 — Final Answer Generation
**File:** `answer/generator.py` | **Classes:** `EvidenceSelector`, `AnswerGenerator`

#### Evidence ranking priority (WBS 34)
```
1. Claims từ validated set với credibility cao nhất
2. Claims có source_credibility cao
3. Claims có retrieval_relevance cao với query
```

**Scoring:**
```
final_score(c) = credibility_weight · (0.7 · cred_norm + 0.3 · src_cred)
              + relevance_weight · retrieval_relevance
```

#### Generation prompt (WBS 35)
```
You are a precise, factual assistant. Answer the question based ONLY
on the provided verified claims. If evidence is conflicting or
insufficient, explicitly state your uncertainty.

Question: {query}

Verified Claims (ordered by confidence):
{claims_list}

{conflict_section}   ← chỉ có nếu resolved=False

Instructions:
- Ground every statement in the provided claims
- If claims conflict, present both perspectives with attribution
- Express uncertainty where evidence is incomplete
- Do NOT add information not present in the claims
```

#### Conflict section template (khi unresolved)
```
Unresolved Conflicts:
- [temporal] '1954' (conf: 1.76) vs '1953' (conf: -2.56)
Note: The above conflicts could not be resolved. Present both perspectives.
```

#### Config keys
```yaml
generation:
  model: "gpt-4o-mini"
  max_retries: 3
  max_evidence_claims: 10
  credibility_weight: 0.6
  relevance_weight: 0.4
```

---

## 4. HYPERPARAMETERS TỔNG HỢP

```yaml
# configs/method1_base.yaml

seed: 42
device: "cpu"   # hoặc "cuda"

# Phase 1-2: Claims
claim_extraction:
  model: "gpt-4o-mini"
  max_retries: 3
  max_claim_length_words: 35

claim_canonical:
  similarity_threshold: 0.92
  embedding_model: "BAAI/bge-m3"
  batch_size: 64

# Phase 3: Graph
graph:
  pair_similarity_threshold: 0.3
  nli_model: "cross-encoder/nli-deberta-v3-base"
  nli_edge_threshold: 0.5

# Phase 4: Retrieval
retrieval:
  top_k: 10
  alpha: 0.5
  rrf_k: 60
  min_conflict_pairs: 1

# Phase 5: Conflict Zone
conflict_zone:
  arbitration_max_iter: 10
  arbitration_convergence: 0.01
  arbitration_damping: 0.85
  credibility_threshold: 0.0
  use_llm_for_entities: true

# Phase 6: Loop
iterative_loop:
  max_iterations: 3
  intensity_threshold: 0.1
  min_intensity_reduction: 0.1

# Phase 7: Generation
generation:
  model: "gpt-4o-mini"
  max_evidence_claims: 10
  credibility_weight: 0.6
  relevance_weight: 0.4
```

---

## 5. EVALUATION METRICS (Method 1 specific)

Method 1 dùng **4 nhóm metrics từ ArbGraph** + **Nhóm 5 mới** cho conflict resolution.

### Nhóm 1 — Generative Informativeness
| Metric | Mô tả | Tool |
|--------|-------|------|
| FR (Fact Recall) | % ground-truth atomic facts covered by answer | LongFact + SAFE |
| ID (Information Density) | atomic facts / token | LongFact + SAFE |

### Nhóm 2 — Factual Grounding
| Metric | Mô tả | Tool |
|--------|-------|------|
| Faithfulness | % generated claims supported by retrieved evidence | RAGChecker |
| Context Utilization | % ground-truth facts captured by retrieval | RAGChecker |

### Nhóm 3 — Robustness to Noisy Retrieval
| Metric | Mô tả | Tool |
|--------|-------|------|
| Noise-S | Sensitivity to topically related misleading evidence | RAGChecker |
| Noise-I | Sensitivity to irrelevant distracting evidence | RAGChecker |

### Nhóm 4 — Knowledge Attribution
| Metric | Mô tả | Tool |
|--------|-------|------|
| Hallucination | % generated claims unsupported by evidence | RAGChecker |
| Self-knowledge | % correct claims from parametric memory (not context) | RAGChecker |

### Nhóm 5 — Conflict Resolution Quality ← MỚI
| Metric | Mô tả | Ground truth source |
|--------|-------|-------------------|
| CP-P | Conflict Pair Precision | RAMDocs / ECON |
| CP-R | Conflict Pair Recall | RAMDocs / ECON |
| CP-F1 | Conflict Pair F1 | RAMDocs / ECON |
| Slot-F1 | Per-type slot localization F1 | ECON binary vectors |
| Intensity-MAE | MAE giữa predicted và ground truth intensity | ECON |

> **Priority:** Metric quan trọng nhất là **CP-F1** và **FR** — hai metrics phân biệt Method 1 với baseline.

---

## 6. DATASETS

### Recommended (theo ranking phù hợp)

| Rank | Dataset | Lý do | Phase dùng |
|------|---------|-------|-----------|
| 1 | **RAMDocs** | Ambiguity + misinfo + noise cùng lúc | Main eval |
| 2 | **ASQA** | Multi-source, entity ambiguity, NER-friendly | Dev + ablation |
| 3 | **ECON dataset** | Có binary vectors + intensity scores | Nhóm 5 metrics |
| 4 | **ConflictBank** | Typed conflicts (temporal/semantic/misinfo) | Ablation per type |
| 5 | **LongFact** | Evaluate generation quality sau resolve | Nhóm 1 metrics |

### Không nên dùng cho main eval
- **TriviaQA**, **NQ**: single-hop, ít conflict tự nhiên
- **ABG-CoQA**: conversational/referential ambiguity, không phải inter-document conflict

---

## 7. FALLBACK STRATEGY

| Component | Primary | Fallback | Trigger |
|-----------|---------|----------|---------|
| Claim extraction | GPT-4o-mini | Llama-3-8B local | API unavailable |
| Entity extraction (factoid) | LLM prompt | spaCy NER | LLM cost > budget |
| NLI edge labeling | DeBERTa NLI | LLM judge (uncertain pairs) | NLI score 0.4–0.6 |
| Claim canonical | BGE-M3 | MiniLM-L6 | BGE unavailable |
| Iterative loop | Full 3 iterations | 1-pass targeted retrieval | Loop không converge |
| Credibility update | Dynamic iterative | Static (sum of support edges) | Instability |

---

## 8. ĐIỂM KHÁC BIỆT SO VỚI ARBGRAPH GỐC

| Aspect | ArbGraph | Method 1 |
|--------|----------|----------|
| Conflict resolution level | Claim-level | Claim + Factoid-level |
| Iterative loop | Không có | Có (max 3 iterations) |
| Targeted retrieval | Không có | Có (slot-specific query) |
| Claim canonical | Không có | Có (temporal/numerical split) |
| Conflict localization | Không có | Typed slots + intensity score |
| Stopping condition | N/A | intensity < θ OR no new evidence |
| Training required | Không | Không |

---

## 9. FREQUENTLY ASKED QUESTIONS

**Q: Tại sao không train GNN như Hướng 2?**
A: Method 1 giữ training-free để: (1) universally applicable trên mọi LLM backbone, (2) contribution rõ ràng từ arbitration mechanism, không bị confound bởi training. Nếu cần learned weights, xem Method 2.

**Q: Claim canonical có làm mất conflict không?**
A: Không — đây là lý do tại sao có điều kiện tách temporal/numerical. Claims với năm/số khác nhau KHÔNG được merge dù semantic rất gần.

**Q: Khi nào dùng LLM judge cho NLI edge?**
A: Chỉ khi NLI score nằm trong khoảng 0.4–0.6 (uncertain). Không dùng LLM judge cho toàn bộ dataset.

**Q: Tại sao Balanced top-k thay vì chỉ lấy top-k theo score?**
A: Pure top-k có thể chọn toàn bộ claims tương đồng nhau, bỏ sót conflict pairs. Balanced selection đảm bảo conflict evidence được đưa vào arbitration.

**Q: conflict_intensity thấp có nghĩa là gì?**
A: Có thể là: (a) conflict thực sự không nghiêm trọng — chỉ một slot nhỏ khác, (b) hai claims không overlap nhiều về slots → không đủ info để judge. Cần kết hợp với credibility_gap để quyết định.

**Q: Loop dừng khi nào nếu conflict là genuine disagreement?**
A: Sau MAX_ITERATIONS=3, loop dừng và flag `resolved=False`. Generation sẽ present cả hai perspectives với attribution thay vì chọn một bên.

---

*Last updated: 30/05/2026 — Version 1.0*
*File này là companion của CLAUDE.md, dành riêng cho Method 1.*
*Mọi thay đổi architecture phải được ghi vào Decisions Log.*
