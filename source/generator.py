"""
answer/generator.py
Phase 7 — Final Answer Generation
WBS 34: Evidence selector
WBS 35: Grounded answer generation
Owner: E
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from schema.models import Claim, ConflictLocalization, LoopResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates (WBS 35)
# ---------------------------------------------------------------------------

GROUNDED_PROMPT = """You are a precise, factual assistant. Answer the question based ONLY on the provided verified claims. If the evidence is conflicting or insufficient, explicitly state your uncertainty.

Question: {query}

Verified Claims (ordered by confidence):
{claims_list}

{conflict_section}

Instructions:
- Ground every statement in the provided claims
- If claims conflict, present both perspectives with attribution
- Express uncertainty where evidence is incomplete
- Do NOT add information not present in the claims

Answer:"""

CONFLICT_SECTION_TEMPLATE = """Unresolved Conflicts:
{conflicts}
Note: The above conflicts could not be resolved. Present both perspectives."""


# ---------------------------------------------------------------------------
# Evidence Selector (WBS 34)
# ---------------------------------------------------------------------------

class EvidenceSelector:
    """Chọn và rank evidence claims cho generation.

    Ranking priority (WBS 34):
    1. Claims từ validated set với credibility cao nhất
    2. Claims có source_credibility cao
    3. Claims liên quan trực tiếp đến query (retrieval_relevance cao)

    Args:
        max_claims: Số claims tối đa đưa vào prompt.
        credibility_weight: Weight cho credibility trong ranking.
        relevance_weight: Weight cho relevance trong ranking.
    """

    def __init__(
        self,
        max_claims: int = 10,
        credibility_weight: float = 0.6,
        relevance_weight: float = 0.4,
    ) -> None:
        self.max_claims = max_claims
        self.credibility_weight = credibility_weight
        self.relevance_weight = relevance_weight

    def select(
        self,
        validated_claims: list[Claim],
        credibility_scores: Optional[dict[str, float]] = None,
    ) -> list[Claim]:
        """Chọn và rank claims cho generation.

        Args:
            validated_claims: Claims từ validated set.
            credibility_scores: Optional override cho credibility scores.

        Returns:
            Ranked list of claims (tối đa max_claims).
        """
        def score(claim: Claim) -> float:
            cred = (
                credibility_scores.get(claim.claim_id, 0.0)
                if credibility_scores else claim.claim_confidence
            )
            # Normalize credibility (có thể âm)
            cred_norm = max(0.0, min(1.0, (cred + 2.0) / 4.0))
            rel = max(0.0, claim.retrieval_relevance)
            src = max(0.0, claim.source_credibility if claim.source_credibility >= 0 else 0.5)

            return (
                self.credibility_weight * (0.7 * cred_norm + 0.3 * src)
                + self.relevance_weight * rel
            )

        ranked = sorted(validated_claims, key=score, reverse=True)
        selected = ranked[: self.max_claims]

        logger.info(
            "Evidence selected: %d/%d claims for generation",
            len(selected), len(validated_claims),
        )
        return selected


# ---------------------------------------------------------------------------
# AnswerGenerator (WBS 35)
# ---------------------------------------------------------------------------

class AnswerGenerator:
    """Generate grounded final answer từ validated claims.

    Args:
        model: OpenAI model name.
        max_retries: Retry attempts.
        evidence_selector: EvidenceSelector instance.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        evidence_selector: Optional[EvidenceSelector] = None,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.evidence_selector = evidence_selector or EvidenceSelector()

    def generate(
        self,
        query: str,
        loop_result: LoopResult,
        credibility_scores: Optional[dict[str, float]] = None,
    ) -> str:
        """Generate final answer.

        Args:
            query: Original user query.
            loop_result: Output từ IterativeLoop.
            credibility_scores: Optional credibility scores từ arbitration.

        Returns:
            Generated answer string.
        """
        # Select evidence
        selected_claims = self.evidence_selector.select(
            loop_result.validated_claims,
            credibility_scores,
        )

        if not selected_claims:
            return (
                "Insufficient evidence to answer this question reliably. "
                "The retrieved documents do not contain enough verified information."
            )

        # Format claims list
        claims_list = "\n".join(
            f"{i+1}. [{c.doc_id}] {c.text}"
            for i, c in enumerate(selected_claims)
        )

        # Format conflict section nếu có unresolved conflicts
        conflict_section = ""
        if not loop_result.resolved and loop_result.conflict_localizations:
            conflicts_text = "\n".join(
                f"- [{loc.slot}] '{loc.value_i}' (confidence: {loc.credibility_i:.2f}) "
                f"vs '{loc.value_j}' (confidence: {loc.credibility_j:.2f})"
                for loc in loop_result.conflict_localizations[:3]
            )
            conflict_section = CONFLICT_SECTION_TEMPLATE.format(
                conflicts=conflicts_text
            )

        prompt = GROUNDED_PROMPT.format(
            query=query,
            claims_list=claims_list,
            conflict_section=conflict_section,
        )

        answer = self._call_llm_with_retry(prompt)

        logger.info(
            "Generated answer for query_id=%s (resolved=%s, %d claims used)",
            loop_result.query_id,
            loop_result.resolved,
            len(selected_claims),
        )
        return answer

    def _call_llm_with_retry(self, prompt: str) -> str:
        """LLM call với exponential backoff."""
        import openai

        for attempt in range(self.max_retries):
            try:
                client = openai.OpenAI()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1024,
                )
                return response.choices[0].message.content or ""

            except Exception as exc:
                wait = 2.0 ** attempt
                logger.warning(
                    "LLM generation failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1, self.max_retries, exc, wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"Answer generation failed after {self.max_retries} retries")
