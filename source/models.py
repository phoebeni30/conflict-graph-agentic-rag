"""
schema/models.py
Pydantic models theo schema_v1.json — FROZEN after 24/05/2026.
Không thay đổi field names/types. Tạo schema_v2.py nếu cần migrate.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class Document(BaseModel):
    doc_id: str
    source: str
    text: str
    title: str = ""
    url: Optional[str] = None
    timestamp: Optional[str] = None          # ISO 8601
    credibility_score: float = -1.0          # -1.0 = chưa tính


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------

class Claim(BaseModel):
    claim_id: str
    doc_id: str
    text: str
    embedding: Optional[list[float]] = None
    retrieval_relevance: float = -1.0        # 0.0–1.0
    claim_confidence: float = -1.0           # 0.0–1.0
    source_credibility: float = -1.0         # kế thừa từ doc


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

class Edge(BaseModel):
    edge_id: str
    claim_a: str                             # claim_id
    claim_b: str                             # claim_id
    relation: str                            # support | contradiction | entailment | neutral
    nli_score: float
    source: str = "nli_model"               # nli_model | rule | llm_judge

    @field_validator("relation")
    @classmethod
    def validate_relation(cls, v: str) -> str:
        allowed = {"support", "contradiction", "entailment", "neutral"}
        if v not in allowed:
            raise ValueError(f"relation must be one of {allowed}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# ConflictRegion
# ---------------------------------------------------------------------------

class ConflictRegion(BaseModel):
    region_id: str
    query_id: str
    claims: list[str]                        # list of claim_id
    edges: list[str]                         # list of edge_id
    predicted_state: str = "Underdetermined" # Resolvable | Underdetermined | Contextual
    probabilities: dict[str, float] = Field(
        default_factory=lambda: {
            "Resolvable": 0.0,
            "Underdetermined": 1.0,
            "Contextual": 0.0,
        }
    )

    @field_validator("predicted_state")
    @classmethod
    def validate_state(cls, v: str) -> str:
        allowed = {"Resolvable", "Underdetermined", "Contextual"}
        if v not in allowed:
            raise ValueError(f"predicted_state must be one of {allowed}")
        return v


# ---------------------------------------------------------------------------
# ActionLabel
# ---------------------------------------------------------------------------

class ActionLabel(BaseModel):
    region_id: str
    query_id: str
    action: str                              # NO_RETRIEVE | VERIFY | DISAMBIGUATE | SUPPORT | COUNTER
    probability: float = 1.0
    targeted_query: Optional[str] = None

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        allowed = {"NO_RETRIEVE", "VERIFY", "DISAMBIGUATE", "SUPPORT", "COUNTER"}
        if v not in allowed:
            raise ValueError(f"action must be one of {allowed}")
        return v


# ---------------------------------------------------------------------------
# Factoid (extension cho Phase 5 — Conflict Zone)
# ---------------------------------------------------------------------------

class FactoidSlots(BaseModel):
    """Typed slots extract từ một claim theo schema của Hướng 1."""
    temporal: Optional[str] = None
    numerical: Optional[str] = None
    entity_subject: Optional[str] = None
    entity_object: Optional[str] = None
    relation: Optional[str] = None
    location: Optional[str] = None


class ConflictLocalization(BaseModel):
    """Output của factoid-level conflict analysis giữa một conflict pair."""
    claim_i_id: str
    claim_j_id: str
    slot: str                                # slot type bị conflict
    value_i: str
    value_j: str
    conflict_intensity: float                # |conflict_slots| / total_slots
    credibility_i: float
    credibility_j: float


# ---------------------------------------------------------------------------
# LoopResult
# ---------------------------------------------------------------------------

class LoopResult(BaseModel):
    query_id: str
    iterations_run: int
    resolved: bool
    validated_claims: list[Claim]
    conflict_localizations: list[ConflictLocalization]
    final_answer: Optional[str] = None
