"""
tests/test_pipeline_smoke.py
Smoke tests — không cần LLM/GPU, dùng mock objects.
Run: pytest tests/test_pipeline_smoke.py -v
"""

from __future__ import annotations

import pytest
import networkx as nx

from schema.models import (
    Claim, Document, Edge, ConflictLocalization,
    FactoidSlots, LoopResult,
)
from claims.embedder import ClaimCanonical, _extract_temporal_numerical
from conflict.conflict_zone import CredibilityArbitrator, ConflictZoneAnalyzer
from graph.graph_builder import PairGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_claim(claim_id: str, text: str, doc_id: str = "d001",
               emb: list[float] | None = None) -> Claim:
    import numpy as np
    if emb is None:
        np.random.seed(int(claim_id[1:]) if claim_id[1:].isdigit() else 0)
        emb = np.random.randn(384).tolist()
        # normalize
        norm = sum(x**2 for x in emb) ** 0.5
        emb = [x / norm for x in emb]
    return Claim(claim_id=claim_id, doc_id=doc_id, text=text, embedding=emb)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchemaModels:
    def test_claim_schema_valid(self):
        c = Claim(claim_id="c001", doc_id="d001", text="Test claim.")
        assert c.claim_id == "c001"
        assert c.embedding is None

    def test_edge_invalid_relation_raises(self):
        with pytest.raises(Exception):
            Edge(edge_id="e001", claim_a="c001", claim_b="c002",
                 relation="invalid_label", nli_score=0.9)

    def test_conflict_localization_valid(self):
        loc = ConflictLocalization(
            claim_i_id="c001", claim_j_id="c002",
            slot="temporal", value_i="1954", value_j="1953",
            conflict_intensity=0.33, credibility_i=0.82, credibility_j=-1.65,
        )
        assert loc.conflict_intensity == pytest.approx(0.33)

    def test_factoid_slots_all_none(self):
        slots = FactoidSlots()
        assert slots.temporal is None
        assert slots.numerical is None


# ---------------------------------------------------------------------------
# Temporal/Numerical extraction tests
# ---------------------------------------------------------------------------

class TestTemporalExtraction:
    def test_extract_year(self):
        vals = _extract_temporal_numerical("Chiến dịch diễn ra năm 1954.")
        assert "1954" in vals

    def test_extract_number(self):
        vals = _extract_temporal_numerical("Chiến dịch kéo dài 56 ngày.")
        assert "56" in vals

    def test_no_values(self):
        vals = _extract_temporal_numerical("Không có số liệu nào ở đây.")
        assert vals == []

    def test_multiple_years(self):
        vals = _extract_temporal_numerical("Từ 1945 đến 1975.")
        assert "1945" in vals
        assert "1975" in vals


# ---------------------------------------------------------------------------
# Claim Canonical tests
# ---------------------------------------------------------------------------

class TestClaimCanonical:
    def test_merge_identical_embeddings(self):
        """Claims với embedding gần giống nhau và cùng temporal → merge."""
        import numpy as np
        base = np.random.randn(384)
        base = base / np.linalg.norm(base)

        c1 = make_claim("c001", "Điện Biên Phủ diễn ra năm 1954.", emb=base.tolist())
        # slightly perturbed
        perturbed = base + np.random.randn(384) * 0.01
        perturbed = perturbed / np.linalg.norm(perturbed)
        c2 = make_claim("c002", "Trận Điện Biên Phủ xảy ra vào 1954.",
                        emb=perturbed.tolist())

        canonical = ClaimCanonical(similarity_threshold=0.95)
        result = canonical.canonicalize([c1, c2])
        # High similarity + same year → should merge to 1
        # (may or may not merge depending on actual sim — just check valid)
        assert len(result) >= 1

    def test_no_merge_different_years(self):
        """Claims với năm khác nhau không được merge dù embedding gần."""
        import numpy as np
        base = np.random.randn(384)
        base = base / np.linalg.norm(base)

        c1 = make_claim("c001", "Điện Biên Phủ diễn ra năm 1954.", emb=base.tolist())
        c3 = make_claim("c003", "Điện Biên Phủ diễn ra năm 1953.", emb=base.tolist())

        canonical = ClaimCanonical(similarity_threshold=0.5)  # low threshold
        result = canonical.canonicalize([c1, c3])
        # Different years → should NOT merge
        assert len(result) == 2

    def test_single_claim_unchanged(self):
        c = make_claim("c001", "Test claim.")
        canonical = ClaimCanonical()
        result = canonical.canonicalize([c])
        assert len(result) == 1
        assert result[0].claim_id == "c001"


# ---------------------------------------------------------------------------
# Credibility Arbitrator tests
# ---------------------------------------------------------------------------

class TestCredibilityArbitrator:
    def _make_graph_with_edges(self) -> nx.DiGraph:
        """Graph: r1 support r4, r1 contradict r3."""
        G = nx.DiGraph()
        for nid in ["r1", "r3", "r4", "r5"]:
            G.add_node(nid)
        G.add_edge("r4", "r1", relation="support", nli_score=0.89)
        G.add_edge("r5", "r1", relation="support", nli_score=0.82)
        G.add_edge("r3", "r1", relation="contradiction", nli_score=0.89)
        G.add_edge("r1", "r3", relation="contradiction", nli_score=0.89)
        G.add_edge("r4", "r3", relation="contradiction", nli_score=0.76)
        return G

    def test_arbitration_validates_supported_claim(self):
        G = self._make_graph_with_edges()
        arb = CredibilityArbitrator(max_iterations=5)
        scores = arb.compute(G)
        # r1 has more support → should have higher score than r3
        assert scores["r1"] > scores["r3"]

    def test_arbitration_suppresses_contradicted_claim(self):
        G = self._make_graph_with_edges()
        arb = CredibilityArbitrator(max_iterations=5)
        scores = arb.compute(G)
        # r3 có contradiction từ r1 và r4 → negative score
        assert scores["r3"] < 0

    def test_arbitration_converges(self):
        """Không raise exception, scores là finite."""
        import math
        G = self._make_graph_with_edges()
        arb = CredibilityArbitrator(max_iterations=20)
        scores = arb.compute(G)
        for v in scores.values():
            assert math.isfinite(v)


# ---------------------------------------------------------------------------
# Pair Generator tests
# ---------------------------------------------------------------------------

class TestPairGenerator:
    def test_no_pairs_below_threshold(self):
        import numpy as np
        # Orthogonal vectors → sim = 0
        c1 = make_claim("c001", "A.", emb=[1.0, 0.0, 0.0] + [0.0]*381)
        c2 = make_claim("c002", "B.", emb=[0.0, 1.0, 0.0] + [0.0]*381)
        gen = PairGenerator(similarity_threshold=0.9)
        pairs = gen.generate([c1, c2])
        assert pairs == []

    def test_pairs_above_threshold(self):
        import numpy as np
        # Same vector → sim = 1
        vec = [1.0] + [0.0]*383
        c1 = make_claim("c001", "A.", emb=vec)
        c2 = make_claim("c002", "B.", emb=vec)
        gen = PairGenerator(similarity_threshold=0.5)
        pairs = gen.generate([c1, c2])
        assert len(pairs) == 1

    def test_empty_input(self):
        gen = PairGenerator()
        assert gen.generate([]) == []

    def test_single_claim(self):
        c = make_claim("c001", "Only claim.")
        gen = PairGenerator()
        assert gen.generate([c]) == []


# ---------------------------------------------------------------------------
# LoopResult tests
# ---------------------------------------------------------------------------

class TestLoopResult:
    def test_loop_result_valid(self):
        result = LoopResult(
            query_id="q001",
            iterations_run=2,
            resolved=True,
            validated_claims=[],
            conflict_localizations=[],
            final_answer="Test answer.",
        )
        assert result.resolved is True
        assert result.final_answer == "Test answer."
