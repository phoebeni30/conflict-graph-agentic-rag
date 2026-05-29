"""
claims/extractor.py
Phase 1 — Claim Extraction
WBS 15: LLM-based atomic claim extraction từ documents.
Owner: B
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from schema.models import Claim, Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template (WBS 15)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT_SYSTEM = (
    "You are an expert at extracting atomic factual claims from text. "
    "An atomic claim is a single, self-contained factual statement that can be "
    "verified independently. Do not combine multiple facts into one claim."
)

EXTRACTION_PROMPT_USER = (
    "Extract all atomic claims from the following document passage. "
    "Return ONLY a JSON array of strings, one claim per element. "
    "Do not include opinions, questions, or non-factual statements. "
    "Each claim must be under 30 words and self-contained.\n\n"
    "Document: {document_text}"
)


# ---------------------------------------------------------------------------
# ClaimExtractor
# ---------------------------------------------------------------------------

class ClaimExtractor:
    """Extract atomic claims từ Document bằng LLM.

    Args:
        model: OpenAI model name.
        max_retries: Số lần retry khi LLM call thất bại.
        backoff_base: Base seconds cho exponential backoff.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        backoff_base: float = 2.0,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._claim_counter = 0

    def extract(self, document: Document) -> list[Claim]:
        """Extract atomic claims từ một document.

        Args:
            document: Input document theo schema_v1.

        Returns:
            List of Claim objects. Embeddings chưa được populate (None).

        Raises:
            ValueError: Nếu document.text rỗng.
            RuntimeError: Nếu LLM call thất bại sau max_retries.
        """
        if not document.text.strip():
            raise ValueError(f"document.text is empty for doc_id={document.doc_id}")

        raw_texts = self._call_llm_with_retry(document.text)
        claims = []
        for text in raw_texts:
            text = text.strip()
            if not text or len(text.split()) > 35:          # bỏ qua claim quá dài
                continue
            self._claim_counter += 1
            claim_id = f"c{self._claim_counter:06d}"
            claims.append(
                Claim(
                    claim_id=claim_id,
                    doc_id=document.doc_id,
                    text=text,
                    source_credibility=document.credibility_score,
                )
            )

        logger.info(
            "Extracted %d claims from doc_id=%s", len(claims), document.doc_id
        )
        return claims

    def extract_batch(self, documents: list[Document]) -> list[Claim]:
        """Extract claims từ nhiều documents.

        Args:
            documents: List of Document objects.

        Returns:
            Flat list of all Claim objects.
        """
        all_claims: list[Claim] = []
        for doc in documents:
            try:
                all_claims.extend(self.extract(doc))
            except Exception as exc:
                logger.error(
                    "Failed to extract claims from doc_id=%s: %s", doc.doc_id, exc
                )
        return all_claims

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm_with_retry(self, text: str) -> list[str]:
        """Gọi LLM với exponential backoff retry.

        Returns:
            List of claim strings.

        Raises:
            RuntimeError: Sau max_retries thất bại.
        """
        import openai

        for attempt in range(self.max_retries):
            try:
                client = openai.OpenAI()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": EXTRACTION_PROMPT_SYSTEM},
                        {
                            "role": "user",
                            "content": EXTRACTION_PROMPT_USER.format(
                                document_text=text[:4000]   # truncate để tránh vượt context
                            ),
                        },
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                parsed = json.loads(content)

                # LLM có thể trả về {"claims": [...]} hoặc trực tiếp [...]
                if isinstance(parsed, list):
                    return parsed
                for key in ("claims", "results", "output"):
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                logger.warning("Unexpected LLM response format: %s", content[:200])
                return []

            except Exception as exc:
                wait = self.backoff_base ** attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"LLM extraction failed after {self.max_retries} retries"
        )
