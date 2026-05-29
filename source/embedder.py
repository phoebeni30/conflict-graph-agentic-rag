"""
claims/embedder.py + canonical.py
Phase 1 — Claim Embedding & Canonical (Phase 2)
WBS 17: BGE/SBERT embedding
WBS: Claim canonical — merge similar claims, tách claims có temporal/numerical
Owner: B
"""

from __future__ import annotations

import logging
import re
from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer

from schema.models import Claim

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns để detect temporal / numerical values trong claim text
# ---------------------------------------------------------------------------

_YEAR_PATTERN = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_NUMBER_PATTERN = re.compile(r"\b\d+([.,]\d+)?(%|km|kg|m|s|ly)?\b")


def _extract_temporal_numerical(text: str) -> list[str]:
    """Trả về list các temporal/numerical tokens trong text."""
    years = _YEAR_PATTERN.findall(text)
    numbers = [m.group(0) for m in _NUMBER_PATTERN.finditer(text)]
    return list(set(years + numbers))


# ---------------------------------------------------------------------------
# ClaimEmbedder (WBS 17)
# ---------------------------------------------------------------------------

class ClaimEmbedder:
    """Encode claims thành dense vectors bằng SentenceTransformer.

    Args:
        model_name: HuggingFace model identifier.
        batch_size: Batch size cho encoding.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, claims: list[Claim]) -> list[Claim]:
        """Populate embedding field in-place và trả về claims.

        Args:
            claims: List of Claim objects.

        Returns:
            Same list với embedding field được populate.
        """
        if not claims:
            return claims

        texts = [c.text for c in claims]
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        for claim, emb in zip(claims, embeddings):
            claim.embedding = emb.tolist()

        logger.info("Embedded %d claims with model=%s", len(claims), self.model_name)
        return claims


# ---------------------------------------------------------------------------
# ClaimCanonical (Phase 2)
# ---------------------------------------------------------------------------

class ClaimCanonical:
    """Merge claims diễn đạt cùng sự kiện, tách claims có temporal/numerical khác nhau.

    Args:
        similarity_threshold: Cosine similarity ngưỡng để candidate merge.
    """

    def __init__(self, similarity_threshold: float = 0.92) -> None:
        self.similarity_threshold = similarity_threshold

    def canonicalize(self, claims: list[Claim]) -> list[Claim]:
        """Trả về danh sách representative claims sau khi merge.

        Logic:
          1. Tính pairwise cosine similarity giữa các claim embeddings.
          2. Nếu sim(ci, cj) > threshold:
             - Kiểm tra temporal/numerical values.
             - Nếu cùng values → merge (giữ claim có claim_id nhỏ hơn).
             - Nếu khác values → KHÔNG merge (giữ cả hai).

        Args:
            claims: List of Claim objects với embedding đã được populate.

        Returns:
            Reduced list of representative claims.

        Raises:
            ValueError: Nếu bất kỳ claim nào thiếu embedding.
        """
        for c in claims:
            if c.embedding is None:
                raise ValueError(
                    f"Claim {c.claim_id} has no embedding. Run ClaimEmbedder first."
                )

        if len(claims) <= 1:
            return claims

        emb_matrix = np.array([c.embedding for c in claims], dtype=np.float32)
        # cosine similarity matrix (embeddings đã normalized)
        sim_matrix = emb_matrix @ emb_matrix.T

        # Union-Find để group merge candidates
        parent = list(range(len(claims)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        for i, j in combinations(range(len(claims)), 2):
            if sim_matrix[i, j] < self.similarity_threshold:
                continue

            vals_i = set(_extract_temporal_numerical(claims[i].text))
            vals_j = set(_extract_temporal_numerical(claims[j].text))

            # Điều kiện tách: nếu có temporal/numerical khác nhau → không merge
            if vals_i and vals_j and vals_i != vals_j:
                logger.debug(
                    "Keeping separate: '%s' vs '%s' (different values: %s vs %s)",
                    claims[i].text[:50],
                    claims[j].text[:50],
                    vals_i,
                    vals_j,
                )
                continue

            # Merge: group i và j
            union(i, j)

        # Chọn representative: claim đầu tiên trong mỗi group
        groups: dict[int, int] = {}    # root → first index
        for idx in range(len(claims)):
            root = find(idx)
            if root not in groups:
                groups[root] = idx

        representatives = [claims[idx] for idx in sorted(groups.values())]

        logger.info(
            "Canonical: %d claims → %d representatives (threshold=%.2f)",
            len(claims),
            len(representatives),
            self.similarity_threshold,
        )
        return representatives
