"""
graph/graph_builder.py
Phase 3 — Evidence Graph Construction
WBS 19: Claim pair generation
WBS 20: NLI relation inference
WBS 21: Graph construction với networkx
Owner: C
"""

from __future__ import annotations

import logging
from itertools import combinations

import networkx as nx
import numpy as np
from transformers import pipeline

from schema.models import Claim, Edge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLI Inference (WBS 20)
# ---------------------------------------------------------------------------

class NLIInference:
    """NLI-based relation inference giữa các claim pairs.

    Args:
        model_name: HuggingFace cross-encoder NLI model.
        threshold: Confidence threshold — dưới ngưỡng sẽ label là "neutral".
        device: "cpu" hoặc "cuda".
    """

    LABEL_MAP = {
        "entailment": "support",
        "contradiction": "contradiction",
        "neutral": "neutral",
    }

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self._pipeline = None

    @property
    def nli_pipeline(self):
        if self._pipeline is None:
            logger.info("Loading NLI model: %s", self.model_name)
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
            )
        return self._pipeline

    def predict(self, claim_a: str, claim_b: str) -> tuple[str, float]:
        """Predict relation giữa hai claims.

        Args:
            claim_a: Premise claim text.
            claim_b: Hypothesis claim text.

        Returns:
            Tuple (relation_label, confidence_score).
            relation_label ∈ {"support", "contradiction", "neutral"}.
        """
        try:
            result = self.nli_pipeline(
                claim_a,
                candidate_labels=["entailment", "contradiction", "neutral"],
                hypothesis_template="{}",
            )
            # result["labels"][0] là label có score cao nhất
            top_label = result["labels"][0]
            top_score = result["scores"][0]

            if top_score < self.threshold:
                return "neutral", top_score

            mapped = self.LABEL_MAP.get(top_label, "neutral")
            return mapped, top_score

        except Exception as exc:
            logger.warning("NLI prediction failed: %s. Defaulting to neutral.", exc)
            return "neutral", 0.0

    def predict_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[tuple[str, float]]:
        """Predict relations cho nhiều pairs.

        Args:
            pairs: List of (claim_a_text, claim_b_text).

        Returns:
            List of (relation_label, confidence_score).
        """
        return [self.predict(a, b) for a, b in pairs]


# ---------------------------------------------------------------------------
# Pair Generator (WBS 19)
# ---------------------------------------------------------------------------

class PairGenerator:
    """Generate candidate claim pairs dựa trên cosine similarity.

    Chỉ pair claims có sim > threshold để tránh O(n²) full pairing.

    Args:
        similarity_threshold: Cosine similarity tối thiểu để tạo pair.
    """

    def __init__(self, similarity_threshold: float = 0.3) -> None:
        self.similarity_threshold = similarity_threshold

    def generate(self, claims: list[Claim]) -> list[tuple[Claim, Claim]]:
        """Trả về list các claim pairs đủ điều kiện.

        Args:
            claims: Claims với embedding đã được populate.

        Returns:
            List of (claim_a, claim_b) pairs.
        """
        if len(claims) < 2:
            return []

        emb = np.array([c.embedding for c in claims], dtype=np.float32)
        sim = emb @ emb.T      # normalized embeddings → cosine sim

        pairs = []
        for i, j in combinations(range(len(claims)), 2):
            if sim[i, j] >= self.similarity_threshold:
                pairs.append((claims[i], claims[j]))

        logger.debug(
            "Generated %d pairs from %d claims (threshold=%.2f)",
            len(pairs),
            len(claims),
            self.similarity_threshold,
        )
        return pairs


# ---------------------------------------------------------------------------
# ClaimGraphBuilder (WBS 21)
# ---------------------------------------------------------------------------

class ClaimGraphBuilder:
    """Xây dựng evidence graph từ claims và NLI-inferred edges.

    Args:
        nli: NLIInference instance.
        pair_generator: PairGenerator instance.
        nli_threshold: Edge bị loại bỏ nếu NLI score < threshold.
    """

    def __init__(
        self,
        nli: NLIInference,
        pair_generator: PairGenerator,
        nli_threshold: float = 0.5,
    ) -> None:
        self.nli = nli
        self.pair_generator = pair_generator
        self.nli_threshold = nli_threshold
        self._edge_counter = 0

    def build(self, claims: list[Claim]) -> tuple[nx.DiGraph, list[Edge]]:
        """Xây dựng evidence graph.

        Node attributes: claim_id, text, embedding, retrieval_relevance,
                         claim_confidence, source_credibility
        Edge attributes: relation, nli_score, source

        Args:
            claims: List of Claim objects với embedding populated.

        Returns:
            Tuple (nx.DiGraph, list[Edge]) theo schema_v1.
        """
        G = nx.DiGraph()

        # Thêm nodes
        for c in claims:
            G.add_node(
                c.claim_id,
                text=c.text,
                embedding=c.embedding,
                retrieval_relevance=c.retrieval_relevance,
                claim_confidence=c.claim_confidence,
                source_credibility=c.source_credibility,
                doc_id=c.doc_id,
            )

        # Generate pairs và infer relations
        pairs = self.pair_generator.generate(claims)
        edges: list[Edge] = []

        for claim_a, claim_b in pairs:
            relation, score = self.nli.predict(claim_a.text, claim_b.text)

            # Quality check: loại bỏ nếu score dưới threshold
            if score < self.nli_threshold:
                continue

            self._edge_counter += 1
            edge_id = f"e{self._edge_counter:06d}"
            edge = Edge(
                edge_id=edge_id,
                claim_a=claim_a.claim_id,
                claim_b=claim_b.claim_id,
                relation=relation,
                nli_score=score,
                source="nli_model",
            )
            edges.append(edge)

            G.add_edge(
                claim_a.claim_id,
                claim_b.claim_id,
                edge_id=edge_id,
                relation=relation,
                nli_score=score,
                source="nli_model",
            )

        logger.info(
            "Graph built: %d nodes, %d edges (%d contradiction, %d support, %d neutral)",
            G.number_of_nodes(),
            G.number_of_edges(),
            sum(1 for _, _, d in G.edges(data=True) if d["relation"] == "contradiction"),
            sum(1 for _, _, d in G.edges(data=True) if d["relation"] == "support"),
            sum(1 for _, _, d in G.edges(data=True) if d["relation"] == "neutral"),
        )
        return G, edges

    def serialize(self, graph: nx.DiGraph, query_id: str) -> dict:
        """Serialize graph thành dict theo schema_v1 format.

        Args:
            graph: nx.DiGraph.
            query_id: Associated query ID.

        Returns:
            Dict với keys "query_id", "nodes", "edges".
        """
        nodes = [
            {"claim_id": n, **graph.nodes[n]}
            for n in graph.nodes
        ]
        edges = [
            {
                "from": u,
                "to": v,
                **graph.edges[u, v],
            }
            for u, v in graph.edges
        ]
        return {
            "query_id": query_id,
            "nodes": nodes,
            "edges": edges,
        }
