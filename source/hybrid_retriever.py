"""
retrieval/hybrid_retriever.py
Phase 4 — Query-Aware Retrieval
WBS 13: Hybrid retrieval (BM25 + Dense) với RRF
WBS: Balanced top-k selection theo edge type distribution
Owner: B
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from schema.models import Claim

logger = logging.getLogger(__name__)

RRF_K = 60      # RRF constant (WBS 13)


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Hybrid retrieval combining BM25 + Dense với RRF fusion.

    Args:
        embedding_model: SentenceTransformer model name.
        alpha: Weight cho dense score (1-alpha = BM25 weight).
        top_k: Số claims trả về.
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-m3",
        alpha: float = 0.5,
        top_k: int = 10,
    ) -> None:
        self.alpha = alpha
        self.top_k = top_k
        self.embedding_model_name = embedding_model
        self._dense_model: SentenceTransformer | None = None
        self._claims: list[Claim] = []
        self._bm25: BM25Okapi | None = None
        self._emb_matrix: np.ndarray | None = None

    @property
    def dense_model(self) -> SentenceTransformer:
        if self._dense_model is None:
            self._dense_model = SentenceTransformer(self.embedding_model_name)
        return self._dense_model

    def index(self, claims: list[Claim]) -> None:
        """Build BM25 index và store dense embeddings.

        Args:
            claims: List of claims với embedding populated.

        Raises:
            ValueError: Nếu claims không có embedding.
        """
        self._claims = claims

        # BM25 index
        tokenized = [c.text.lower().split() for c in claims]
        self._bm25 = BM25Okapi(tokenized)

        # Dense matrix
        for c in claims:
            if c.embedding is None:
                raise ValueError(f"Claim {c.claim_id} missing embedding")
        self._emb_matrix = np.array(
            [c.embedding for c in claims], dtype=np.float32
        )

        logger.info("Indexed %d claims for hybrid retrieval", len(claims))

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[Claim, float]]:
        """Retrieve top-k claims cho query bằng hybrid RRF.

        claim_relevance_score(q, ri) = α * dense_score + (1-α) * bm25_score
        Kết hợp ranking bằng RRF: score(d) = Σ 1/(k + rank_i(d))

        Args:
            query: Query string.
            top_k: Override số claims trả về.

        Returns:
            List of (Claim, relevance_score) đã sort giảm dần.
        """
        if not self._claims or self._bm25 is None:
            raise RuntimeError("Call index() before retrieve()")

        k = top_k or self.top_k

        # BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)

        # Dense scores
        q_emb = self.dense_model.encode([query], normalize_embeddings=True)
        dense_scores = (self._emb_matrix @ q_emb.T).flatten()

        # RRF combination
        bm25_ranks = np.argsort(-bm25_scores)
        dense_ranks = np.argsort(-dense_scores)

        rrf_scores = np.zeros(len(self._claims))
        for rank, idx in enumerate(bm25_ranks):
            rrf_scores[idx] += 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(dense_ranks):
            rrf_scores[idx] += 1.0 / (RRF_K + rank + 1)

        # Hybrid score = α * dense + (1-α) * bm25_normalized + rrf
        bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)
        dense_norm = (dense_scores + 1) / 2          # [-1,1] → [0,1]
        hybrid = self.alpha * dense_norm + (1 - self.alpha) * bm25_norm + rrf_scores

        top_indices = np.argsort(-hybrid)[:k]
        results = [(self._claims[i], float(hybrid[i])) for i in top_indices]

        # Update retrieval_relevance field
        for claim, score in results:
            claim.retrieval_relevance = score

        return results


# ---------------------------------------------------------------------------
# Balanced Top-K Selector
# ---------------------------------------------------------------------------

class BalancedTopKSelector:
    """Chọn top-k claims đảm bảo tỷ lệ edge types được cân bằng.

    Tránh việc chỉ retrieve claims tương đồng nhau mà bỏ sót conflict pairs.

    Args:
        target_k: Tổng số claims cần chọn.
        min_conflict_pairs: Số lượng conflict pairs tối thiểu trong top-k.
    """

    def __init__(
        self,
        target_k: int = 10,
        min_conflict_pairs: int = 1,
    ) -> None:
        self.target_k = target_k
        self.min_conflict_pairs = min_conflict_pairs

    def select(
        self,
        ranked_claims: list[tuple[Claim, float]],
        edge_index: dict[tuple[str, str], str],   # (claim_id_a, claim_id_b) → relation
    ) -> list[Claim]:
        """Chọn top-k claims với đảm bảo conflict pair coverage.

        Args:
            ranked_claims: (Claim, score) sorted by relevance.
            edge_index: Dict mapping claim pair → relation type.

        Returns:
            List of selected Claims.
        """
        selected: list[Claim] = []
        selected_ids: set[str] = set()
        conflict_pairs_found = 0

        # Pass 1: greedy selection theo score
        for claim, _ in ranked_claims:
            if len(selected) >= self.target_k:
                break
            selected.append(claim)
            selected_ids.add(claim.claim_id)

        # Count conflict pairs trong selection hiện tại
        for (a, b), rel in edge_index.items():
            if rel == "contradiction" and a in selected_ids and b in selected_ids:
                conflict_pairs_found += 1

        # Pass 2: nếu thiếu conflict pairs, swap in conflict neighbors
        if conflict_pairs_found < self.min_conflict_pairs:
            selected_ids_in_conflict: set[str] = {
                cid
                for (a, b), rel in edge_index.items()
                if rel == "contradiction"
                for cid in (a, b)
                if cid in selected_ids
            }

            # Tìm conflict partners của các claims đã được chọn
            for cid in list(selected_ids):
                for (a, b), rel in edge_index.items():
                    if rel != "contradiction":
                        continue
                    partner = b if a == cid else (a if b == cid else None)
                    if partner and partner not in selected_ids:
                        # Swap ra claim cuối cùng không liên quan conflict
                        non_conflict = [
                            c for c in selected
                            if c.claim_id not in selected_ids_in_conflict
                        ]
                        if non_conflict and len(selected) >= self.target_k:
                            selected.remove(non_conflict[-1])
                            selected_ids.discard(non_conflict[-1].claim_id)

                        # Tìm partner claim trong ranked list
                        partner_claim = next(
                            (c for c, _ in ranked_claims if c.claim_id == partner),
                            None,
                        )
                        if partner_claim:
                            selected.append(partner_claim)
                            selected_ids.add(partner)
                            conflict_pairs_found += 1
                            break

        logger.debug(
            "BalancedTopK: selected %d claims, %d conflict pairs",
            len(selected),
            conflict_pairs_found,
        )
        return selected[: self.target_k]
