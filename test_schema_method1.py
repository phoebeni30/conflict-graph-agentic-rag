"""
tests/test_schema_method1.py

Validate Method 1 Extension Schemas (schema_v1_method1.json).
Chạy: pytest tests/test_schema_method1.py -v

Coverage:
  - FactoidSlots: valid construction, null policy, at-least-one-slot constraint
  - ConflictLocalization: valid construction, ID format, enum constraints,
      value_i ≠ value_j, primary_slot in all_conflict_slots, intensity range,
      claim_i ≠ claim_j
  - LoopResult: valid construction, ID format, iterations ceiling (MAX=3),
      stop_reason enum, resolved ↔ empty localizations, validated ∩ suppressed = ∅
  - ClaimCanonicalMeta: valid construction, similarity_score range
  - CredibilityUpdate: valid construction, iteration >= 1
  - ASQA-specific fixtures: entity ambiguity conflict (Sound of Silence example)
  - JSONL serialize / deserialize round-trip cho tất cả 5 schemas
  - Data flow integration: Claim → FactoidSlots → ConflictLocalization → LoopResult
"""

import json
from pathlib import Path
from typing import Optional

import pytest
from pydantic import ValidationError

from schema.models import (
    ActionLabel,
    Claim,
    ClaimCanonicalMeta,
    ConflictLocalization,
    CredibilityUpdate,
    Document,
    Edge,
    FactoidSlots,
    LoopResult,
)

VALID_SLOT_TYPES   = ["temporal", "numerical", "entity_subject", "entity_object", "relation", "location"]
VALID_STOP_REASONS = ["intensity_below_threshold", "max_iterations_reached", "no_new_evidence", "convergence"]


# ===========================================================================
# FIXTURES — ASQA-based examples
# ASQA conflict type: entity ambiguity (cùng tên, khác entity)
# Example: "Who is the original artist of Sound of Silence?"
#   → Dami Im (Eurovision 2016) vs Simon & Garfunkel (1964)
# ===========================================================================

@pytest.fixture
def asqa_doc_dami_im():
    """ASQA document về Dami Im's Sound of Silence (2016)."""
    return {
        "doc_id": "d001",
        "source": "ASQA",
        "text": (
            "Sound of Silence is a song performed by Australian recording artist Dami Im "
            "at the Eurovision Song Contest 2016. It was written by Anthony Egizii and "
            "David Musumeci of DNA Songs."
        ),
        "title": "Sound of Silence (Dami Im song)",
        "url": "https://en.wikipedia.org/wiki/Sound_of_Silence_(Dami_Im_song)",
        "timestamp": "2024-01-10T00:00:00Z",
        "credibility_score": 0.90,
    }


@pytest.fixture
def asqa_doc_simon_garfunkel():
    """ASQA document về Simon & Garfunkel's The Sound of Silence (1964)."""
    return {
        "doc_id": "d002",
        "source": "ASQA",
        "text": (
            "The Sound of Silence is a song by American folk rock duo Simon & Garfunkel. "
            "Written by Paul Simon, it was first recorded in March 1964. "
            "The song became a hit after being remixed and released in 1965."
        ),
        "title": "The Sound of Silence",
        "url": "https://en.wikipedia.org/wiki/The_Sound_of_Silence",
        "timestamp": "2024-01-10T00:00:00Z",
        "credibility_score": 0.92,
    }


@pytest.fixture
def asqa_claim_dami_im():
    """Claim: Dami Im là nghệ sĩ biểu diễn Sound of Silence."""
    return {
        "claim_id": "c001",
        "doc_id": "d001",
        "text": "Sound of Silence was performed by Dami Im at Eurovision 2016.",
        "embedding": None,
        "retrieval_relevance": 0.91,
        "claim_confidence": 0.88,
        "source_credibility": 0.90,
    }


@pytest.fixture
def asqa_claim_simon_garfunkel():
    """Claim: Simon & Garfunkel là nghệ sĩ gốc của Sound of Silence."""
    return {
        "claim_id": "c002",
        "doc_id": "d002",
        "text": "The Sound of Silence was written and originally recorded by Simon & Garfunkel in 1964.",
        "embedding": None,
        "retrieval_relevance": 0.89,
        "claim_confidence": 0.92,
        "source_credibility": 0.92,
    }


@pytest.fixture
def asqa_edge_entity_conflict():
    """Edge: contradiction giữa hai claims về Sound of Silence."""
    return {
        "edge_id": "e001",
        "claim_a": "c001",
        "claim_b": "c002",
        "relation": "contradiction",
        "nli_score": 0.87,
        "source": "nli_model",
    }


@pytest.fixture
def valid_factoid_slots_dami_im():
    """FactoidSlots cho claim về Dami Im — entity conflict chính."""
    return {
        "claim_id": "c001",
        "temporal": "2016",
        "numerical": None,
        "entity_subject": "Dami Im",
        "entity_object": "Sound of Silence",
        "relation": "performed",
        "location": "Eurovision Song Contest",
    }


@pytest.fixture
def valid_factoid_slots_simon_garfunkel():
    """FactoidSlots cho claim về Simon & Garfunkel."""
    return {
        "claim_id": "c002",
        "temporal": "1964",
        "numerical": None,
        "entity_subject": "Simon & Garfunkel",
        "entity_object": "The Sound of Silence",
        "relation": "written and recorded",
        "location": None,
    }


@pytest.fixture
def valid_conflict_localization():
    """ConflictLocalization: entity_subject conflict (Dami Im vs Simon & Garfunkel)."""
    return {
        "localization_id": "l001",
        "query_id": "q001",
        "claim_i_id": "c002",           # higher credibility (Simon & Garfunkel)
        "claim_j_id": "c001",           # lower credibility (Dami Im)
        "primary_slot": "entity_subject",
        "value_i": "Simon & Garfunkel",
        "value_j": "Dami Im",
        "conflict_intensity": 0.4,      # 2 conflict slots / 5 non-null slots
        "credibility_i": 1.20,
        "credibility_j": -0.35,
        "all_conflict_slots": ["entity_subject", "temporal"],
    }


@pytest.fixture
def valid_loop_result_resolved():
    """LoopResult: resolved sau 1 iteration."""
    return {
        "loop_id": "loop001",
        "query_id": "q001",
        "iterations_run": 1,
        "stop_reason": "intensity_below_threshold",
        "resolved": True,
        "validated_claim_ids": ["c002"],
        "suppressed_claim_ids": ["c001"],
        "conflict_localizations": [],    # empty khi resolved=True
        "final_answer": (
            "The original artist of 'The Sound of Silence' is Simon & Garfunkel, "
            "who wrote and recorded the song in 1964. Note: Dami Im also performed "
            "a cover version at Eurovision 2016."
        ),
        "credibility_scores": {"c001": -0.35, "c002": 1.20},
    }


@pytest.fixture
def valid_loop_result_unresolved():
    """LoopResult: unresolved sau MAX_ITERATIONS=3."""
    return {
        "loop_id": "loop002",
        "query_id": "q002",
        "iterations_run": 3,
        "stop_reason": "max_iterations_reached",
        "resolved": False,
        "validated_claim_ids": ["c001", "c002"],
        "suppressed_claim_ids": [],
        "conflict_localizations": ["l001"],  # conflict còn tồn tại
        "final_answer": None,
        "credibility_scores": {"c001": 0.45, "c002": 0.50},
    }


@pytest.fixture
def valid_canonical_meta():
    """ClaimCanonicalMeta: merge 2 paraphrase claims về Dami Im."""
    return {
        "representative_claim_id": "c001",
        "merged_claim_ids": ["c005", "c006"],
        "merge_reason": "high_similarity_same_temporal",
        "similarity_score": 0.96,
    }


@pytest.fixture
def valid_credibility_update():
    """CredibilityUpdate: log iteration 1 cho c002."""
    return {
        "claim_id": "c002",
        "query_id": "q001",
        "iteration": 1,
        "score_before": 0.0,
        "score_after": 1.20,
        "support_sum": 1.71,
        "contradict_sum": 0.51,
    }


# ===========================================================================
# 1. FACTOID SLOTS
# ===========================================================================

class TestFactoidSlots:

    def test_valid_full(self, valid_factoid_slots_dami_im):
        slots = FactoidSlots(**valid_factoid_slots_dami_im)
        assert slots.claim_id == "c001"
        assert slots.entity_subject == "Dami Im"
        assert slots.temporal == "2016"

    def test_valid_minimal_one_slot(self):
        """Chỉ cần 1 slot non-null là valid."""
        slots = FactoidSlots(
            claim_id="c001",
            temporal="1964",
            numerical=None,
            entity_subject=None,
            entity_object=None,
            relation=None,
            location=None,
        )
        assert slots.temporal == "1964"
        assert slots.entity_subject is None

    def test_all_null_slots_raises(self):
        """Tất cả slots null → ValidationError."""
        with pytest.raises(ValidationError):
            FactoidSlots(
                claim_id="c001",
                temporal=None,
                numerical=None,
                entity_subject=None,
                entity_object=None,
                relation=None,
                location=None,
            )

    def test_invalid_claim_id_format_raises(self, valid_factoid_slots_dami_im):
        valid_factoid_slots_dami_im["claim_id"] = "claim1"
        with pytest.raises(ValidationError):
            FactoidSlots(**valid_factoid_slots_dami_im)

    def test_missing_claim_id_raises(self, valid_factoid_slots_dami_im):
        del valid_factoid_slots_dami_im["claim_id"]
        with pytest.raises(ValidationError):
            FactoidSlots(**valid_factoid_slots_dami_im)

    def test_temporal_is_string_not_int(self, valid_factoid_slots_dami_im):
        """temporal phải là string để preserve format gốc."""
        slots = FactoidSlots(**valid_factoid_slots_dami_im)
        assert isinstance(slots.temporal, str)
        assert slots.temporal == "2016"

    def test_null_optional_slots_preserved(self, valid_factoid_slots_dami_im):
        slots = FactoidSlots(**valid_factoid_slots_dami_im)
        assert slots.numerical is None

    def test_asqa_entity_conflict_slots(
        self, valid_factoid_slots_dami_im, valid_factoid_slots_simon_garfunkel
    ):
        """ASQA: entity_subject conflict — Dami Im vs Simon & Garfunkel."""
        slots_i = FactoidSlots(**valid_factoid_slots_dami_im)
        slots_j = FactoidSlots(**valid_factoid_slots_simon_garfunkel)
        assert slots_i.entity_subject != slots_j.entity_subject
        assert slots_i.temporal != slots_j.temporal


# ===========================================================================
# 2. CONFLICT LOCALIZATION
# ===========================================================================

class TestConflictLocalization:

    def test_valid_full(self, valid_conflict_localization):
        loc = ConflictLocalization(**valid_conflict_localization)
        assert loc.localization_id == "l001"
        assert loc.primary_slot == "entity_subject"
        assert loc.conflict_intensity == pytest.approx(0.4)

    def test_invalid_localization_id_raises(self, valid_conflict_localization):
        valid_conflict_localization["localization_id"] = "loc1"
        with pytest.raises(ValidationError):
            ConflictLocalization(**valid_conflict_localization)

    @pytest.mark.parametrize("slot", VALID_SLOT_TYPES)
    def test_all_valid_slot_types(self, valid_conflict_localization, slot):
        valid_conflict_localization["primary_slot"] = slot
        valid_conflict_localization["all_conflict_slots"] = [slot]
        loc = ConflictLocalization(**valid_conflict_localization)
        assert loc.primary_slot == slot

    def test_invalid_primary_slot_raises(self, valid_conflict_localization):
        valid_conflict_localization["primary_slot"] = "year"   # không trong enum
        with pytest.raises(ValidationError):
            ConflictLocalization(**valid_conflict_localization)

    def test_invalid_slot_in_all_conflict_slots_raises(self, valid_conflict_localization):
        valid_conflict_localization["all_conflict_slots"] = ["entity_subject", "year"]
        with pytest.raises(ValidationError):
            ConflictLocalization(**valid_conflict_localization)

    def test_same_value_i_and_j_raises(self, valid_conflict_localization):
        """value_i == value_j không có conflict → ValidationError."""
        valid_conflict_localization["value_i"] = "Simon & Garfunkel"
        valid_conflict_localization["value_j"] = "Simon & Garfunkel"
        with pytest.raises(ValidationError):
            ConflictLocalization(**valid_conflict_localization)

    def test_same_claim_i_and_j_raises(self, valid_conflict_localization):
        """Self-loop localization không được phép."""
        valid_conflict_localization["claim_j_id"] = valid_conflict_localization["claim_i_id"]
        with pytest.raises(ValidationError):
            ConflictLocalization(**valid_conflict_localization)

    def test_intensity_zero_raises(self, valid_conflict_localization):
        """conflict_intensity = 0.0 không có nghĩa lý → ValidationError."""
        valid_conflict_localization["conflict_intensity"] = 0.0
        with pytest.raises(ValidationError):
            ConflictLocalization(**valid_conflict_localization)

    def test_intensity_above_one_raises(self, valid_conflict_localization):
        valid_conflict_localization["conflict_intensity"] = 1.5
        with pytest.raises(ValidationError):
            ConflictLocalization(**valid_conflict_localization)

    def test_intensity_exactly_one_valid(self, valid_conflict_localization):
        """conflict_intensity = 1.0 hợp lệ (tất cả slots đều conflict)."""
        valid_conflict_localization["conflict_intensity"] = 1.0
        valid_conflict_localization["all_conflict_slots"] = VALID_SLOT_TYPES
        valid_conflict_localization["primary_slot"] = "temporal"
        loc = ConflictLocalization(**valid_conflict_localization)
        assert loc.conflict_intensity == 1.0

    def test_primary_slot_not_in_all_conflict_slots_raises(self, valid_conflict_localization):
        """primary_slot phải nằm trong all_conflict_slots."""
        valid_conflict_localization["primary_slot"] = "temporal"
        valid_conflict_localization["all_conflict_slots"] = ["entity_subject"]  # temporal thiếu
        with pytest.raises(ValidationError):
            ConflictLocalization(**valid_conflict_localization)

    def test_credibility_can_be_negative(self, valid_conflict_localization):
        """credibility_score có thể âm (suppressed claim)."""
        valid_conflict_localization["credibility_j"] = -2.56
        loc = ConflictLocalization(**valid_conflict_localization)
        assert loc.credibility_j == pytest.approx(-2.56)

    def test_invalid_query_id_format_raises(self, valid_conflict_localization):
        valid_conflict_localization["query_id"] = "query1"
        with pytest.raises(ValidationError):
            ConflictLocalization(**valid_conflict_localization)

    def test_asqa_entity_conflict_example(self, valid_conflict_localization):
        """ASQA: Sound of Silence entity conflict được localized đúng."""
        loc = ConflictLocalization(**valid_conflict_localization)
        assert loc.primary_slot == "entity_subject"
        assert loc.value_i == "Simon & Garfunkel"
        assert loc.value_j == "Dami Im"
        assert "entity_subject" in loc.all_conflict_slots
        assert loc.credibility_i > loc.credibility_j   # Simon & Garfunkel credible hơn


# ===========================================================================
# 3. LOOP RESULT
# ===========================================================================

class TestLoopResult:

    def test_valid_resolved(self, valid_loop_result_resolved):
        result = LoopResult(**valid_loop_result_resolved)
        assert result.resolved is True
        assert result.iterations_run == 1
        assert result.conflict_localizations == []
        assert result.final_answer is not None

    def test_valid_unresolved(self, valid_loop_result_unresolved):
        result = LoopResult(**valid_loop_result_unresolved)
        assert result.resolved is False
        assert result.iterations_run == 3
        assert len(result.conflict_localizations) > 0
        assert result.final_answer is None

    def test_invalid_loop_id_raises(self, valid_loop_result_resolved):
        valid_loop_result_resolved["loop_id"] = "l001"   # format sai
        with pytest.raises(ValidationError):
            LoopResult(**valid_loop_result_resolved)

    @pytest.mark.parametrize("bad_id", ["loop1", "LOOP001", "l001", "loop0001"])
    def test_invalid_loop_id_formats_raise(self, valid_loop_result_resolved, bad_id):
        valid_loop_result_resolved["loop_id"] = bad_id
        with pytest.raises(ValidationError):
            LoopResult(**valid_loop_result_resolved)

    @pytest.mark.parametrize("valid_id", ["loop001", "loop042", "loop999"])
    def test_valid_loop_id_formats(self, valid_loop_result_resolved, valid_id):
        valid_loop_result_resolved["loop_id"] = valid_id
        result = LoopResult(**valid_loop_result_resolved)
        assert result.loop_id == valid_id

    def test_iterations_above_max_raises(self, valid_loop_result_resolved):
        """MAX_ITERATIONS = 3 — iterations_run > 3 không hợp lệ."""
        valid_loop_result_resolved["iterations_run"] = 4
        with pytest.raises(ValidationError):
            LoopResult(**valid_loop_result_resolved)

    def test_iterations_negative_raises(self, valid_loop_result_resolved):
        valid_loop_result_resolved["iterations_run"] = -1
        with pytest.raises(ValidationError):
            LoopResult(**valid_loop_result_resolved)

    def test_iterations_zero_valid(self, valid_loop_result_resolved):
        """iterations_run = 0: loop không chạy vì không có conflict."""
        valid_loop_result_resolved["iterations_run"] = 0
        valid_loop_result_resolved["stop_reason"] = "intensity_below_threshold"
        result = LoopResult(**valid_loop_result_resolved)
        assert result.iterations_run == 0

    @pytest.mark.parametrize("reason", VALID_STOP_REASONS)
    def test_all_valid_stop_reasons(self, valid_loop_result_resolved, reason):
        valid_loop_result_resolved["stop_reason"] = reason
        result = LoopResult(**valid_loop_result_resolved)
        assert result.stop_reason == reason

    def test_invalid_stop_reason_raises(self, valid_loop_result_resolved):
        valid_loop_result_resolved["stop_reason"] = "timeout"  # không trong enum
        with pytest.raises(ValidationError):
            LoopResult(**valid_loop_result_resolved)

    def test_resolved_true_with_non_empty_localizations_raises(
        self, valid_loop_result_resolved
    ):
        """resolved=True nhưng conflict_localizations không rỗng → inconsistent."""
        valid_loop_result_resolved["conflict_localizations"] = ["l001"]
        with pytest.raises(ValidationError):
            LoopResult(**valid_loop_result_resolved)

    def test_resolved_false_with_empty_localizations_allowed(
        self, valid_loop_result_unresolved
    ):
        """resolved=False với localizations không rỗng — valid."""
        result = LoopResult(**valid_loop_result_unresolved)
        assert result.resolved is False
        assert len(result.conflict_localizations) > 0

    def test_validated_suppressed_disjoint(self, valid_loop_result_resolved):
        """validated_claim_ids và suppressed_claim_ids không được overlap."""
        valid_loop_result_resolved["validated_claim_ids"] = ["c001", "c002"]
        valid_loop_result_resolved["suppressed_claim_ids"] = ["c002"]   # overlap!
        with pytest.raises(ValidationError):
            LoopResult(**valid_loop_result_resolved)

    def test_final_answer_null_allowed(self, valid_loop_result_unresolved):
        result = LoopResult(**valid_loop_result_unresolved)
        assert result.final_answer is None

    def test_credibility_scores_can_contain_negative(self, valid_loop_result_resolved):
        result = LoopResult(**valid_loop_result_resolved)
        assert result.credibility_scores["c001"] < 0

    def test_asqa_resolved_loop(self, valid_loop_result_resolved):
        """ASQA: Sound of Silence conflict resolved sau 1 iteration."""
        result = LoopResult(**valid_loop_result_resolved)
        assert "Simon & Garfunkel" in result.final_answer
        assert result.validated_claim_ids == ["c002"]   # Simon & Garfunkel validated
        assert result.suppressed_claim_ids == ["c001"]  # Dami Im suppressed


# ===========================================================================
# 4. CLAIM CANONICAL META
# ===========================================================================

class TestClaimCanonicalMeta:

    def test_valid_full(self, valid_canonical_meta):
        meta = ClaimCanonicalMeta(**valid_canonical_meta)
        assert meta.representative_claim_id == "c001"
        assert len(meta.merged_claim_ids) == 2
        assert meta.similarity_score == pytest.approx(0.96)

    def test_invalid_representative_claim_id_raises(self, valid_canonical_meta):
        valid_canonical_meta["representative_claim_id"] = "claim1"
        with pytest.raises(ValidationError):
            ClaimCanonicalMeta(**valid_canonical_meta)

    def test_similarity_above_one_raises(self, valid_canonical_meta):
        valid_canonical_meta["similarity_score"] = 1.5
        with pytest.raises(ValidationError):
            ClaimCanonicalMeta(**valid_canonical_meta)

    def test_similarity_negative_raises(self, valid_canonical_meta):
        valid_canonical_meta["similarity_score"] = -0.1
        with pytest.raises(ValidationError):
            ClaimCanonicalMeta(**valid_canonical_meta)

    def test_similarity_exactly_one_valid(self, valid_canonical_meta):
        valid_canonical_meta["similarity_score"] = 1.0
        meta = ClaimCanonicalMeta(**valid_canonical_meta)
        assert meta.similarity_score == 1.0

    def test_empty_merged_claims_valid(self, valid_canonical_meta):
        """merged_claim_ids = [] hợp lệ — không có claims bị merge."""
        valid_canonical_meta["merged_claim_ids"] = []
        meta = ClaimCanonicalMeta(**valid_canonical_meta)
        assert meta.merged_claim_ids == []


# ===========================================================================
# 5. CREDIBILITY UPDATE
# ===========================================================================

class TestCredibilityUpdate:

    def test_valid_full(self, valid_credibility_update):
        update = CredibilityUpdate(**valid_credibility_update)
        assert update.claim_id == "c002"
        assert update.iteration == 1
        assert update.score_after == pytest.approx(1.20)

    def test_iteration_zero_raises(self, valid_credibility_update):
        """iteration bắt đầu từ 1, không phải 0."""
        valid_credibility_update["iteration"] = 0
        with pytest.raises(ValidationError):
            CredibilityUpdate(**valid_credibility_update)

    def test_negative_iteration_raises(self, valid_credibility_update):
        valid_credibility_update["iteration"] = -1
        with pytest.raises(ValidationError):
            CredibilityUpdate(**valid_credibility_update)

    def test_scores_can_be_negative(self, valid_credibility_update):
        """credibility score có thể âm (suppressed claim)."""
        valid_credibility_update["score_after"] = -1.65
        valid_credibility_update["contradict_sum"] = 2.54
        update = CredibilityUpdate(**valid_credibility_update)
        assert update.score_after < 0

    def test_support_sum_can_be_zero(self, valid_credibility_update):
        valid_credibility_update["support_sum"] = 0.0
        update = CredibilityUpdate(**valid_credibility_update)
        assert update.support_sum == 0.0


# ===========================================================================
# 6. NULL POLICY (Method 1 schemas)
# ===========================================================================

class TestNullPolicyMethod1:

    def test_factoid_slots_numerical_null_ok(self, valid_factoid_slots_dami_im):
        slots = FactoidSlots(**valid_factoid_slots_dami_im)
        assert slots.numerical is None

    def test_factoid_slots_location_null_ok(self, valid_factoid_slots_simon_garfunkel):
        slots = FactoidSlots(**valid_factoid_slots_simon_garfunkel)
        assert slots.location is None

    def test_loop_result_final_answer_null_ok(self, valid_loop_result_unresolved):
        result = LoopResult(**valid_loop_result_unresolved)
        assert result.final_answer is None

    def test_loop_result_credibility_empty_dict_ok(self, valid_loop_result_resolved):
        valid_loop_result_resolved["credibility_scores"] = {}
        result = LoopResult(**valid_loop_result_resolved)
        assert result.credibility_scores == {}


# ===========================================================================
# 7. ID FORMAT CONVENTIONS (Method 1 schemas)
# ===========================================================================

class TestIdFormatMethod1:

    @pytest.mark.parametrize("loc_id", ["l001", "l042", "l999"])
    def test_valid_localization_id_formats(self, valid_conflict_localization, loc_id):
        valid_conflict_localization["localization_id"] = loc_id
        loc = ConflictLocalization(**valid_conflict_localization)
        assert loc.localization_id == loc_id

    @pytest.mark.parametrize("bad_id", ["loc1", "L001", "l1", "l0001"])
    def test_invalid_localization_id_formats_raise(self, valid_conflict_localization, bad_id):
        valid_conflict_localization["localization_id"] = bad_id
        with pytest.raises(ValidationError):
            ConflictLocalization(**valid_conflict_localization)

    @pytest.mark.parametrize("loop_id", ["loop001", "loop042", "loop999"])
    def test_valid_loop_id_formats(self, valid_loop_result_resolved, loop_id):
        valid_loop_result_resolved["loop_id"] = loop_id
        result = LoopResult(**valid_loop_result_resolved)
        assert result.loop_id == loop_id

    @pytest.mark.parametrize("query_id", ["q001", "q042", "q999"])
    def test_valid_query_id_in_localization(self, valid_conflict_localization, query_id):
        valid_conflict_localization["query_id"] = query_id
        loc = ConflictLocalization(**valid_conflict_localization)
        assert loc.query_id == query_id

    @pytest.mark.parametrize("bad_query", ["query1", "Q001", "q1"])
    def test_invalid_query_id_in_localization_raises(self, valid_conflict_localization, bad_query):
        valid_conflict_localization["query_id"] = bad_query
        with pytest.raises(ValidationError):
            ConflictLocalization(**valid_conflict_localization)


# ===========================================================================
# 8. JSONL SERIALIZE / DESERIALIZE ROUND-TRIP
# ===========================================================================

class TestJsonlRoundTripMethod1:

    def _write_jsonl(self, records: list, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def _read_jsonl(self, path: Path) -> list:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_factoid_slots_round_trip(self, valid_factoid_slots_dami_im, tmp_path):
        slots = FactoidSlots(**valid_factoid_slots_dami_im)
        fpath = tmp_path / "factoid_slots.jsonl"
        self._write_jsonl([slots.model_dump()], fpath)
        loaded = self._read_jsonl(fpath)
        slots2 = FactoidSlots(**loaded[0])
        assert slots2.claim_id == slots.claim_id
        assert slots2.entity_subject == slots.entity_subject
        assert slots2.numerical is None    # null preserved

    def test_conflict_localization_round_trip(self, valid_conflict_localization, tmp_path):
        loc = ConflictLocalization(**valid_conflict_localization)
        fpath = tmp_path / "localizations.jsonl"
        self._write_jsonl([loc.model_dump()], fpath)
        loaded = self._read_jsonl(fpath)
        loc2 = ConflictLocalization(**loaded[0])
        assert loc2.localization_id == loc.localization_id
        assert loc2.conflict_intensity == pytest.approx(loc.conflict_intensity)
        assert loc2.all_conflict_slots == loc.all_conflict_slots

    def test_loop_result_round_trip(self, valid_loop_result_resolved, tmp_path):
        result = LoopResult(**valid_loop_result_resolved)
        fpath = tmp_path / "loop_results.jsonl"
        self._write_jsonl([result.model_dump()], fpath)
        loaded = self._read_jsonl(fpath)
        result2 = LoopResult(**loaded[0])
        assert result2.loop_id == result.loop_id
        assert result2.resolved == result.resolved
        assert result2.final_answer == result.final_answer

    def test_loop_result_null_final_answer_preserved(
        self, valid_loop_result_unresolved, tmp_path
    ):
        """null final_answer phải serialize thành null, không phải ""."""
        result = LoopResult(**valid_loop_result_unresolved)
        fpath = tmp_path / "unresolved.jsonl"
        self._write_jsonl([result.model_dump()], fpath)
        raw = json.loads(fpath.read_text(encoding="utf-8"))
        assert raw["final_answer"] is None

    def test_credibility_scores_with_negative_values_round_trip(
        self, valid_loop_result_resolved, tmp_path
    ):
        """credibility_scores map với giá trị âm phải round-trip đúng."""
        result = LoopResult(**valid_loop_result_resolved)
        fpath = tmp_path / "credibility.jsonl"
        self._write_jsonl([result.model_dump()], fpath)
        loaded = self._read_jsonl(fpath)
        result2 = LoopResult(**loaded[0])
        assert result2.credibility_scores["c001"] == pytest.approx(-0.35)

    def test_jsonl_encoding_vietnamese_text(self, valid_loop_result_resolved, tmp_path):
        """Text tiếng Việt phải serialize/deserialize đúng."""
        valid_loop_result_resolved["final_answer"] = (
            "Nghệ sĩ gốc của bài 'The Sound of Silence' là Simon & Garfunkel."
        )
        result = LoopResult(**valid_loop_result_resolved)
        fpath = tmp_path / "vietnamese.jsonl"
        self._write_jsonl([result.model_dump()], fpath)
        loaded = self._read_jsonl(fpath)
        result2 = LoopResult(**loaded[0])
        assert "Simon & Garfunkel" in result2.final_answer


# ===========================================================================
# 9. DATA FLOW INTEGRATION (Method 1 specific)
# ===========================================================================

class TestDataFlowIntegrationMethod1:
    """
    Kiểm tra data flow của Method 1:
    Claim → FactoidSlots → ConflictLocalization → LoopResult
    """

    def test_claim_to_factoid_slots(
        self, asqa_claim_dami_im, valid_factoid_slots_dami_im
    ):
        """Step 1: Claim có claim_id → FactoidSlots với cùng claim_id."""
        claim = Claim(**asqa_claim_dami_im)
        slots = FactoidSlots(**valid_factoid_slots_dami_im)
        assert slots.claim_id == claim.claim_id

    def test_factoid_slots_to_conflict_localization(
        self,
        valid_factoid_slots_dami_im,
        valid_factoid_slots_simon_garfunkel,
        valid_conflict_localization,
    ):
        """Step 2: FactoidSlots pair → ConflictLocalization."""
        slots_i = FactoidSlots(**valid_factoid_slots_dami_im)
        slots_j = FactoidSlots(**valid_factoid_slots_simon_garfunkel)
        loc = ConflictLocalization(**valid_conflict_localization)

        # Verify: localization references đúng claims
        assert loc.claim_j_id == slots_i.claim_id   # Dami Im (lower cred)
        assert loc.claim_i_id == slots_j.claim_id   # Simon & Garfunkel (higher cred)

        # Verify: conflict slot matches actual difference
        assert loc.primary_slot == "entity_subject"
        assert slots_i.entity_subject != slots_j.entity_subject

    def test_localization_to_loop_result(
        self, valid_conflict_localization, valid_loop_result_resolved
    ):
        """Step 3: ConflictLocalization → LoopResult (resolved)."""
        loc = ConflictLocalization(**valid_conflict_localization)
        result = LoopResult(**valid_loop_result_resolved)

        # Resolved → no localizations referenced
        assert result.resolved is True
        assert loc.localization_id not in result.conflict_localizations

    def test_full_asqa_pipeline_resolved(
        self,
        asqa_doc_dami_im,
        asqa_doc_simon_garfunkel,
        asqa_claim_dami_im,
        asqa_claim_simon_garfunkel,
        asqa_edge_entity_conflict,
        valid_factoid_slots_dami_im,
        valid_factoid_slots_simon_garfunkel,
        valid_conflict_localization,
        valid_loop_result_resolved,
    ):
        """Full ASQA pipeline: Document → Claim → Edge → FactoidSlots
           → ConflictLocalization → LoopResult (resolved)."""

        # Step 1: Documents
        doc1 = Document(**asqa_doc_dami_im)
        doc2 = Document(**asqa_doc_simon_garfunkel)
        assert doc1.source == doc2.source == "ASQA"

        # Step 2: Claims from Documents
        claim_dami = Claim(**asqa_claim_dami_im)
        claim_sg = Claim(**asqa_claim_simon_garfunkel)
        assert claim_dami.doc_id == doc1.doc_id
        assert claim_sg.doc_id == doc2.doc_id

        # Step 3: Edge (contradiction)
        edge = Edge(**asqa_edge_entity_conflict)
        assert edge.relation == "contradiction"
        assert edge.claim_a == claim_dami.claim_id
        assert edge.claim_b == claim_sg.claim_id

        # Step 4: FactoidSlots per claim
        slots_dami = FactoidSlots(**valid_factoid_slots_dami_im)
        slots_sg = FactoidSlots(**valid_factoid_slots_simon_garfunkel)
        assert slots_dami.entity_subject != slots_sg.entity_subject

        # Step 5: ConflictLocalization
        loc = ConflictLocalization(**valid_conflict_localization)
        assert loc.primary_slot == "entity_subject"
        assert loc.conflict_intensity > 0.0

        # Step 6: LoopResult (resolved)
        result = LoopResult(**valid_loop_result_resolved)
        assert result.resolved is True
        assert result.iterations_run <= 3   # MAX_ITERATIONS
        assert result.final_answer is not None
        assert "Simon & Garfunkel" in result.final_answer

    def test_full_asqa_pipeline_unresolved(
        self,
        asqa_claim_dami_im,
        asqa_claim_simon_garfunkel,
        valid_conflict_localization,
        valid_loop_result_unresolved,
    ):
        """ASQA pipeline: genuine disagreement → unresolved after MAX_ITERATIONS."""
        claim_dami = Claim(**asqa_claim_dami_im)
        claim_sg = Claim(**asqa_claim_simon_garfunkel)

        loc = ConflictLocalization(**valid_conflict_localization)
        result = LoopResult(**valid_loop_result_unresolved)

        assert result.resolved is False
        assert result.iterations_run == 3   # MAX hit
        assert result.stop_reason == "max_iterations_reached"
        assert loc.localization_id in result.conflict_localizations
        assert result.final_answer is None  # chưa generate
