"""
pipeline.py
Main Pipeline Orchestrator — kết nối tất cả phases theo proposal Method 1.

Luồng: Documents → Claim Extraction → Claim Canonical → Evidence Graph
       → Retrieve → Conflict Zone → Iterative Loop → Generation

Usage:
    pipeline = ConflictAwarePipeline.from_config("configs/experiment_base.yaml")
    result = pipeline.run(query="...", query_id="q001", documents=[...])
    print(result.final_answer)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from answer.generator import AnswerGenerator, EvidenceSelector
from claims.embedder import ClaimCanonical, ClaimEmbedder
from claims.extractor import ClaimExtractor
from conflict.conflict_zone import (
    ConflictZoneAnalyzer,
    CredibilityArbitrator,
    FactoidDecomposer,
)
from graph.graph_builder import ClaimGraphBuilder, NLIInference, PairGenerator
from loop.iterative_loop import (
    ConflictQueryFormulator,
    GraphUpdater,
    IterativeLoop,
)
from retrieval.hybrid_retriever import BalancedTopKSelector, HybridRetriever
from schema.models import Document, LoopResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Tất cả hyperparameters của pipeline."""

    # Models
    embedding_model: str = "BAAI/bge-m3"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    llm_model: str = "gpt-4o-mini"

    # Claim extraction
    max_claim_length_words: int = 35

    # Claim canonical
    canonical_threshold: float = 0.92

    # Graph construction
    pair_similarity_threshold: float = 0.3
    nli_edge_threshold: float = 0.5

    # Retrieval
    retrieval_top_k: int = 10
    retrieval_alpha: float = 0.5            # dense vs BM25 weight
    min_conflict_pairs: int = 1

    # Arbitration
    arbitration_max_iter: int = 10
    arbitration_convergence: float = 0.01
    arbitration_damping: float = 0.85
    credibility_threshold: float = 0.0

    # Iterative loop
    max_loop_iterations: int = 3
    intensity_threshold: float = 0.1
    min_intensity_reduction: float = 0.1

    # Generation
    max_evidence_claims: int = 10

    # Reproducibility
    seed: int = 42
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ConflictAwarePipeline:
    """End-to-end conflict-aware RAG pipeline.

    Implements Method 1: Conflict-Aware Evidence Arbitration
    with Factoid-Level Resolution.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._set_seed(config.seed)
        self._build_components()

    @classmethod
    def from_config(cls, config_path: str) -> "ConflictAwarePipeline":
        """Load pipeline từ YAML config file.

        Args:
            config_path: Path to Hydra/YAML config.

        Returns:
            Initialized ConflictAwarePipeline.
        """
        import yaml
        with open(config_path) as f:
            cfg_dict = yaml.safe_load(f)
        config = PipelineConfig(**cfg_dict)
        return cls(config)

    def _set_seed(self, seed: int) -> None:
        """Set seed cho reproducibility (WBS 6.3)."""
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def _build_components(self) -> None:
        """Khởi tạo tất cả components."""
        cfg = self.config

        # Phase 1: Claim Extraction
        self.extractor = ClaimExtractor(model=cfg.llm_model)
        self.embedder = ClaimEmbedder(model_name=cfg.embedding_model)
        self.canonical = ClaimCanonical(similarity_threshold=cfg.canonical_threshold)

        # Phase 3: Graph
        self.nli = NLIInference(
            model_name=cfg.nli_model,
            threshold=cfg.nli_edge_threshold,
            device=cfg.device,
        )
        self.pair_gen = PairGenerator(
            similarity_threshold=cfg.pair_similarity_threshold
        )
        self.graph_builder = ClaimGraphBuilder(
            nli=self.nli,
            pair_generator=self.pair_gen,
            nli_threshold=cfg.nli_edge_threshold,
        )

        # Phase 4: Retrieval
        self.retriever = HybridRetriever(
            embedding_model=cfg.embedding_model,
            alpha=cfg.retrieval_alpha,
            top_k=cfg.retrieval_top_k,
        )
        self.selector = BalancedTopKSelector(
            target_k=cfg.retrieval_top_k,
            min_conflict_pairs=cfg.min_conflict_pairs,
        )

        # Phase 5: Conflict Zone
        arbitrator = CredibilityArbitrator(
            max_iterations=cfg.arbitration_max_iter,
            convergence_threshold=cfg.arbitration_convergence,
            damping=cfg.arbitration_damping,
        )
        decomposer = FactoidDecomposer(
            use_llm_for_entities=True,
            llm_model=cfg.llm_model,
        )
        self.conflict_analyzer = ConflictZoneAnalyzer(
            arbitrator=arbitrator,
            decomposer=decomposer,
            credibility_threshold=cfg.credibility_threshold,
        )

        # Phase 6: Loop
        self.query_formulator = ConflictQueryFormulator()
        self.loop = IterativeLoop(
            extractor=self.extractor,
            embedder=self.embedder,
            canonical=self.canonical,
            graph_builder=self.graph_builder,
            retriever=self.retriever,
            selector=self.selector,
            conflict_analyzer=self.conflict_analyzer,
            query_formulator=self.query_formulator,
            max_iterations=cfg.max_loop_iterations,
            intensity_threshold=cfg.intensity_threshold,
            min_intensity_reduction=cfg.min_intensity_reduction,
        )

        # Phase 7: Generation
        self.evidence_selector = EvidenceSelector(max_claims=cfg.max_evidence_claims)
        self.answer_gen = AnswerGenerator(
            model=cfg.llm_model,
            evidence_selector=self.evidence_selector,
        )

        logger.info("Pipeline components initialized with config: %s", cfg)

    def run(
        self,
        query: str,
        query_id: str,
        documents: list[Document],
    ) -> LoopResult:
        """Chạy full pipeline.

        Args:
            query: User query string.
            query_id: Unique identifier cho query.
            documents: Retrieved documents (initial retrieval).

        Returns:
            LoopResult với validated_claims, conflict_localizations, final_answer.
        """
        logger.info("=" * 60)
        logger.info("Pipeline.run: query_id=%s", query_id)
        logger.info("Query: %s", query[:100])
        logger.info("Input documents: %d", len(documents))

        # Phase 1 + 2: Extract + Canonical
        all_claims = self.extractor.extract_batch(documents)
        logger.info("Phase 1: Extracted %d raw claims", len(all_claims))

        self.embedder.embed(all_claims)
        all_claims = self.canonical.canonicalize(all_claims)
        logger.info("Phase 2: %d representative claims after canonical", len(all_claims))

        # Index retriever
        self.retriever.index(all_claims)

        # Phase 6: Iterative Loop (includes phases 3, 4, 5 internally)
        loop_result = self.loop.run(
            query=query,
            query_id=query_id,
            initial_docs=documents,
        )

        # Phase 7: Generate answer
        # Get credibility scores từ final analysis
        final_analysis = self.conflict_analyzer.analyze(
            self.graph_builder.build(loop_result.validated_claims)[0],
            loop_result.validated_claims,
        )
        answer = self.answer_gen.generate(
            query=query,
            loop_result=loop_result,
            credibility_scores=final_analysis.credibility_scores,
        )
        loop_result.final_answer = answer

        logger.info(
            "Pipeline complete: resolved=%s, answer_length=%d chars",
            loop_result.resolved, len(answer),
        )
        logger.info("=" * 60)

        return loop_result
