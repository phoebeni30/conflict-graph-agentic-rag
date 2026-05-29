"""
conflict/conflict_zone.py
Phase 5 — Conflict Zone
5.1: Claim credibility scoring qua ArbGraph-style arbitration
5.2: Factoid decomposition + conflict localization
Owner: C
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from schema.models import Claim, ConflictLocalization, FactoidSlots

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5.1 — Claim Credibility Arbitration
# ---------------------------------------------------------------------------

class CredibilityArbitrator:
    """Tính claim_credibility_score qua propagation trên evidence graph.

    Giống ArbGraph: propagate credibility signals qua support/contradiction edges.
    claim_credibility_score(ri) = Σ edge_score(support) - Σ edge_score(contradict)

    Iterative update cho đến khi hội tụ hoặc max_iterations.

    Args:
        max_iterations: Số vòng lặp tối đa.
        convergence_threshold: Dừng nếu max delta < threshold.
        damping: Damping factor để tránh oscillation.
    """

    def __init__(
        self,
        max_iterations: int = 10,
        convergence_threshold: float = 0.01,
        damping: float = 0.85,
    ) -> None:
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.damping = damping

    def compute(self, graph: nx.DiGraph) -> dict[str, float]:
        """Tính credibility score cho tất cả nodes trong graph.

        Args:
            graph: Evidence graph với edge attribute "relation" và "nli_score".

        Returns:
            Dict mapping claim_id → credibility_score.
        """
        scores: dict[str, float] = {n: 0.0 for n in graph.nodes}

        for iteration in range(self.max_iterations):
            new_scores = dict(scores)

            for node in graph.nodes:
                support_sum = 0.0
                contradict_sum = 0.0

                # Incoming edges
                for pred in graph.predecessors(node):
                    edge_data = graph.edges[pred, node]
                    rel = edge_data.get("relation", "neutral")
                    nli_score = edge_data.get("nli_score", 0.0)
                    # Weight bởi credibility của source node
                    src_cred = max(scores.get(pred, 0.0), 0.0) + 1.0   # +1 để tránh zero

                    if rel in ("support", "entailment"):
                        support_sum += nli_score * src_cred
                    elif rel == "contradiction":
                        contradict_sum += nli_score * src_cred

                raw = support_sum - contradict_sum
                new_scores[node] = self.damping * raw + (1 - self.damping) * scores[node]

            # Convergence check
            max_delta = max(
                abs(new_scores[n] - scores[n]) for n in scores
            )
            scores = new_scores

            logger.debug("Arbitration iter %d: max_delta=%.4f", iteration + 1, max_delta)
            if max_delta < self.convergence_threshold:
                logger.info("Arbitration converged at iteration %d", iteration + 1)
                break

        return scores


# ---------------------------------------------------------------------------
# 5.2 — Factoid Decomposition
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_NUMBER_RE = re.compile(r"\b\d+([.,]\d+)?(%|km|kg|m|s|days|years|months)?\b")


class FactoidDecomposer:
    """Decompose một claim thành typed slots (temporal, numerical, entity, ...).

    Strategy:
    - Temporal/Numerical: rule-based regex (fast, deterministic)
    - Entity/Relation/Location: LLM prompt (flex, handles Vietnamese)

    Args:
        use_llm_for_entities: Nếu True, dùng LLM để extract entity/relation/location.
        llm_model: Model name nếu dùng LLM.
    """

    def __init__(
        self,
        use_llm_for_entities: bool = True,
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self.use_llm_for_entities = use_llm_for_entities
        self.llm_model = llm_model

    def decompose(self, claim_text: str) -> FactoidSlots:
        """Extract typed slots từ claim text.

        Args:
            claim_text: Text của một claim.

        Returns:
            FactoidSlots với các fields được populate.
        """
        slots = FactoidSlots()

        # Rule-based: temporal
        years = _YEAR_RE.findall(claim_text)
        if years:
            slots.temporal = years[0]   # lấy năm đầu tiên

        # Rule-based: numerical (loại trừ năm đã extract)
        numbers = [
            m.group(0) for m in _NUMBER_RE.finditer(claim_text)
            if m.group(0) not in years
        ]
        if numbers:
            slots.numerical = numbers[0]

        # LLM-based: entity, relation, location
        if self.use_llm_for_entities:
            try:
                llm_slots = self._extract_entities_llm(claim_text)
                slots.entity_subject = llm_slots.get("entity_subject")
                slots.entity_object = llm_slots.get("entity_object")
                slots.relation = llm_slots.get("relation")
                slots.location = llm_slots.get("location")
            except Exception as exc:
                logger.warning("LLM entity extraction failed: %s", exc)

        return slots

    def _extract_entities_llm(self, text: str) -> dict:
        """Gọi LLM để extract entity_subject, entity_object, relation, location."""
        import json
        import openai

        prompt = (
            "Extract the following from this claim and return as JSON:\n"
            '{"entity_subject": "<main subject or null>",\n'
            ' "entity_object": "<main object or null>",\n'
            ' "relation": "<main verb/relation or null>",\n'
            ' "location": "<location mentioned or null>"}\n\n'
            f"Claim: {text}\n"
            "Return ONLY valid JSON, no explanation."
        )
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)


# ---------------------------------------------------------------------------
# 5.3 — Conflict Localization
# ---------------------------------------------------------------------------

@dataclass
class ConflictAnalysisResult:
    """Kết quả phân tích conflict zone."""
    credibility_scores: dict[str, float]           # claim_id → score
    conflict_localizations: list[ConflictLocalization]
    validated_claim_ids: list[str]                 # score > 0
    suppressed_claim_ids: list[str]                # score <= 0


class ConflictZoneAnalyzer:
    """Phân tích conflict zone: arbitration + factoid localization.

    Args:
        arbitrator: CredibilityArbitrator instance.
        decomposer: FactoidDecomposer instance.
        credibility_threshold: Score threshold để phân loại validated vs suppressed.
    """

    def __init__(
        self,
        arbitrator: CredibilityArbitrator,
        decomposer: FactoidDecomposer,
        credibility_threshold: float = 0.0,
    ) -> None:
        self.arbitrator = arbitrator
        self.decomposer = decomposer
        self.credibility_threshold = credibility_threshold

    def analyze(
        self,
        graph: nx.DiGraph,
        claims: list[Claim],
    ) -> ConflictAnalysisResult:
        """Chạy toàn bộ conflict zone analysis.

        Args:
            graph: Evidence graph từ Phase 3.
            claims: Top-k claims từ Phase 4.

        Returns:
            ConflictAnalysisResult với credibility scores và localizations.
        """
        claim_map = {c.claim_id: c for c in claims}

        # 5.1 — Arbitration
        cred_scores = self.arbitrator.compute(graph)

        # 5.2 — Factoid localization cho các conflict pairs
        localizations: list[ConflictLocalization] = []

        contradiction_pairs = [
            (u, v)
            for u, v, d in graph.edges(data=True)
            if d.get("relation") == "contradiction"
            and u in claim_map and v in claim_map
        ]

        for claim_id_i, claim_id_j in contradiction_pairs:
            claim_i = claim_map[claim_id_i]
            claim_j = claim_map[claim_id_j]

            loc = self._localize_conflict(
                claim_i,
                claim_j,
                cred_scores.get(claim_id_i, 0.0),
                cred_scores.get(claim_id_j, 0.0),
            )
            if loc:
                localizations.append(loc)

        # Phân loại validated vs suppressed
        validated = [
            cid for cid, score in cred_scores.items()
            if score > self.credibility_threshold
        ]
        suppressed = [
            cid for cid, score in cred_scores.items()
            if score <= self.credibility_threshold
        ]

        logger.info(
            "Conflict zone: %d validated, %d suppressed, %d localizations",
            len(validated),
            len(suppressed),
            len(localizations),
        )

        return ConflictAnalysisResult(
            credibility_scores=cred_scores,
            conflict_localizations=localizations,
            validated_claim_ids=validated,
            suppressed_claim_ids=suppressed,
        )

    def _localize_conflict(
        self,
        claim_i: Claim,
        claim_j: Claim,
        cred_i: float,
        cred_j: float,
    ) -> Optional[ConflictLocalization]:
        """Decompose hai claims và tìm slot bị conflict.

        Returns:
            ConflictLocalization hoặc None nếu không tìm thấy conflict slot.
        """
        slots_i = self.decomposer.decompose(claim_i.text)
        slots_j = self.decomposer.decompose(claim_j.text)

        schema_fields = [
            "temporal", "numerical", "entity_subject",
            "entity_object", "relation", "location",
        ]

        conflict_slots: list[str] = []
        total_slots = 0
        first_conflict_slot: Optional[str] = None
        first_val_i: Optional[str] = None
        first_val_j: Optional[str] = None

        for field_name in schema_fields:
            val_i = getattr(slots_i, field_name)
            val_j = getattr(slots_j, field_name)

            # Bỏ qua nếu cả hai null
            if val_i is None and val_j is None:
                continue

            total_slots += 1
            if val_i != val_j and not (val_i is None or val_j is None):
                conflict_slots.append(field_name)
                if first_conflict_slot is None:
                    first_conflict_slot = field_name
                    first_val_i = val_i
                    first_val_j = val_j

        if not conflict_slots or total_slots == 0:
            return None

        intensity = len(conflict_slots) / total_slots

        logger.debug(
            "Conflict localized: [%s] '%s' vs '%s' (intensity=%.2f)",
            first_conflict_slot,
            first_val_i,
            first_val_j,
            intensity,
        )

        return ConflictLocalization(
            claim_i_id=claim_i.claim_id,
            claim_j_id=claim_j.claim_id,
            slot=first_conflict_slot or "unknown",
            value_i=str(first_val_i),
            value_j=str(first_val_j),
            conflict_intensity=intensity,
            credibility_i=cred_i,
            credibility_j=cred_j,
        )
