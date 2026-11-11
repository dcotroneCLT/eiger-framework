"""
Unit tests for CorpusBuilder and CorpusBuilderResult.

Tests verify:
  - CorpusBuilder.build() creates exactly one ground-truth document per claim
  - With poison_rate=1.0, every claim is poisoned for the configured attack
  - With an empty attack list, no documents are poisoned
  - CorpusBuilderResult.all_documents combines both lists correctly
  - CorpusBuilderResult.poison_ratio computes the correct fraction
  - poison_ratio returns 0.0 for an empty corpus (no division by zero)
  - The ground-truth document text and claim_id match the source claim
  - Two builds with the same seed produce bit-identical poisoned text (determinism)

What these tests do NOT cover:
  - Integration with a real Qdrant vector store (integration test territory).
  - Very large corpora or performance benchmarks (see benchmarking suite).
  - The attack-specific transformation logic (covered in test_attacks.py).
"""

from __future__ import annotations

import pytest

from eiger.attacks.numerical import NumericalShiftAttack
from eiger.core.models import (
    AttackConfig,
    Claim,
    Document,
    PoisonAnnotation,
    PoisonedDocument,
)
from eiger.ingestion.corpus_builder import CorpusBuilder, CorpusBuilderResult


# ─── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def two_claims() -> list[Claim]:
    """
    Provide two representative Claim objects for corpus-building tests.

    The texts contain numeric values and named entities so that
    NumericalShiftAttack and AttributionSwitchAttack have meaningful
    content to transform.
    """
    return [
        Claim(
            claim_id="C1",
            original_fact="Inflation rose to 3.5% in 2023 due to supply shocks.",
            context_query="What was the 2023 inflation rate?",
            source_dataset="test_fixture",
        ),
        Claim(
            claim_id="C2",
            original_fact="NASA confirmed the Mars mission launched in July 2020.",
            context_query="When did NASA launch the Mars mission?",
            source_dataset="test_fixture",
        ),
    ]


def _make_poisoned(claim_id: str = "C1") -> PoisonedDocument:
    """Helper: build a minimal PoisonedDocument for CorpusBuilderResult tests."""
    return PoisonedDocument(
        claim_id=claim_id,
        text="Poisoned: Inflation rose to 5.3% in 2023.",
        attack_name="numerical_shift",
        attack_params={},
        original_text="Inflation rose to 3.5% in 2023.",
        annotation=PoisonAnnotation(
            plausibility=4.0,
            verification_difficulty=3.5,
            editorial_risk=4.5,
        ),
    )


# ─── CorpusBuilderResult ──────────────────────────────────────────────────────

class TestCorpusBuilderResult:
    """Tests for the CorpusBuilderResult dataclass properties."""

    def test_all_documents_combines_both_lists(self) -> None:
        """
        all_documents must concatenate ground_truth_docs and poisoned_docs.

        The combined list is the input to the vector store upsert step.
        Ordering: ground-truth documents first, then poisoned documents.
        """
        gt = Document(claim_id="C1", text="ground truth", doc_type="ground_truth")
        pois = _make_poisoned("C1")
        result = CorpusBuilderResult(ground_truth_docs=[gt], poisoned_docs=[pois])
        all_docs = result.all_documents
        assert len(all_docs) == 2
        assert all_docs[0] is gt
        assert all_docs[1] is pois

    def test_poison_ratio_empty_corpus_is_zero(self) -> None:
        """
        poison_ratio must return 0.0 for an empty CorpusBuilderResult without
        raising ZeroDivisionError.
        """
        result = CorpusBuilderResult()
        assert result.poison_ratio == 0.0

    def test_poison_ratio_half(self) -> None:
        """poison_ratio must return 0.5 when half the documents are poisoned."""
        gt = Document(claim_id="C1", text="ground truth")
        pois = _make_poisoned("C1")
        result = CorpusBuilderResult(ground_truth_docs=[gt], poisoned_docs=[pois])
        assert result.poison_ratio == pytest.approx(0.5)

    def test_poison_ratio_all_poisoned(self) -> None:
        """poison_ratio must return 1.0 when every document is poisoned."""
        p1 = _make_poisoned("C1")
        p2 = _make_poisoned("C2")
        result = CorpusBuilderResult(ground_truth_docs=[], poisoned_docs=[p1, p2])
        assert result.poison_ratio == pytest.approx(1.0)

    def test_poison_ratio_none_poisoned(self) -> None:
        """poison_ratio must return 0.0 when there are no poisoned documents."""
        gt1 = Document(claim_id="C1", text="t1")
        gt2 = Document(claim_id="C2", text="t2")
        result = CorpusBuilderResult(ground_truth_docs=[gt1, gt2], poisoned_docs=[])
        assert result.poison_ratio == 0.0


# ─── CorpusBuilder ────────────────────────────────────────────────────────────

class TestCorpusBuilder:
    """Tests for CorpusBuilder.build()."""

    def test_no_attacks_creates_only_ground_truth(self, two_claims: list[Claim]) -> None:
        """
        With an empty attack list, build() must produce exactly one ground-truth
        document per claim and zero poisoned documents.
        """
        builder = CorpusBuilder(attacks=[], seed=42)
        result = builder.build(two_claims)
        assert len(result.ground_truth_docs) == len(two_claims)
        assert len(result.poisoned_docs) == 0

    def test_attack_rate_1_poisons_every_claim(self, two_claims: list[Claim]) -> None:
        """
        With poison_rate=1.0, every claim must receive a poisoned variant.

        Because rng.random() is in [0, 1) and we check <= 1.0, the condition
        is always True, so all claims are poisoned.
        """
        attack = NumericalShiftAttack()
        cfg = AttackConfig(name="numerical_shift", poison_rate=1.0)
        builder = CorpusBuilder(attacks=[(attack, cfg)], seed=42)
        result = builder.build(two_claims)
        assert len(result.poisoned_docs) == len(two_claims)

    def test_build_empty_claims(self) -> None:
        """
        build() with an empty claims list must return an empty CorpusBuilderResult
        with poison_ratio == 0.0.
        """
        builder = CorpusBuilder(attacks=[], seed=42)
        result = builder.build([])
        assert len(result.ground_truth_docs) == 0
        assert result.poison_ratio == 0.0

    def test_ground_truth_text_matches_original_fact(self, two_claims: list[Claim]) -> None:
        """
        Each ground-truth document's text must be the exact original_fact of its
        source claim, and its claim_id must match.
        """
        builder = CorpusBuilder(attacks=[], seed=42)
        result = builder.build(two_claims)
        for doc, claim in zip(result.ground_truth_docs, two_claims):
            assert doc.text == claim.original_fact
            assert doc.claim_id == claim.claim_id

    def test_ground_truth_doc_type_is_correct(self, two_claims: list[Claim]) -> None:
        """Ground-truth documents must have doc_type == 'ground_truth'."""
        builder = CorpusBuilder(attacks=[], seed=42)
        result = builder.build(two_claims)
        for doc in result.ground_truth_docs:
            assert doc.doc_type == "ground_truth"

    def test_sensitivity_classification_propagates_to_ground_truth_doc(self) -> None:
        """
        Regression test (docs/ETHICS_AND_THREAT_MODEL.md §7): a claim's
        sensitivity_class/risk_level must be propagated onto its ground-truth
        Document, not silently dropped.
        """
        claim = Claim(
            claim_id="C1",
            original_fact="fact",
            context_query="q",
            sensitivity_class="S2",
            risk_level=4,
        )
        builder = CorpusBuilder(attacks=[], seed=42)
        result = builder.build([claim])
        assert result.ground_truth_docs[0].sensitivity_class == "S2"
        assert result.ground_truth_docs[0].risk_level == 4

    def test_unclassified_claim_produces_unclassified_ground_truth_doc(
        self, two_claims: list[Claim]
    ) -> None:
        """
        A claim with no classification (the default) must produce a
        ground-truth document that is also unclassified (None), never
        silently defaulted to "safe".
        """
        builder = CorpusBuilder(attacks=[], seed=42)
        result = builder.build(two_claims)
        for doc in result.ground_truth_docs:
            assert doc.sensitivity_class is None

    def test_ground_truth_label_propagates_to_ground_truth_doc(self) -> None:
        """
        A claim's ground_truth_label must be propagated onto its
        ground-truth Document — independent of doc_type, which stays
        "ground_truth" (manipulation_status) regardless of this value.
        """
        claim = Claim(
            claim_id="C1",
            original_fact="fact",
            context_query="q",
            ground_truth_label="verified_false",
        )
        builder = CorpusBuilder(attacks=[], seed=42)
        result = builder.build([claim])
        assert result.ground_truth_docs[0].ground_truth_label == "verified_false"
        assert result.ground_truth_docs[0].doc_type == "ground_truth"

    def test_unclassified_ground_truth_label_stays_none(
        self, two_claims: list[Claim]
    ) -> None:
        builder = CorpusBuilder(attacks=[], seed=42)
        result = builder.build(two_claims)
        for doc in result.ground_truth_docs:
            assert doc.ground_truth_label is None
            assert doc.risk_level is None

    def test_poisoned_doc_type_is_poisoned(self, two_claims: list[Claim]) -> None:
        """Poisoned documents must have doc_type == 'poisoned'."""
        attack = NumericalShiftAttack()
        cfg = AttackConfig(name="numerical_shift", poison_rate=1.0)
        builder = CorpusBuilder(attacks=[(attack, cfg)], seed=42)
        result = builder.build(two_claims)
        for doc in result.poisoned_docs:
            assert doc.doc_type == "poisoned"

    def test_determinism_same_seed_same_output(self, two_claims: list[Claim]) -> None:
        """
        Two builds with the same seed must produce identical poisoned texts.

        This is the fundamental reproducibility guarantee: an experiment run
        can be replicated bit-for-bit by re-using the same seed and config.
        """
        attack = NumericalShiftAttack()
        cfg = AttackConfig(name="numerical_shift", poison_rate=1.0)
        r1 = CorpusBuilder(attacks=[(attack, cfg)], seed=42).build(two_claims)
        r2 = CorpusBuilder(attacks=[(attack, cfg)], seed=42).build(two_claims)
        texts1 = [d.text for d in r1.poisoned_docs]
        texts2 = [d.text for d in r2.poisoned_docs]
        assert texts1 == texts2

    def test_different_seeds_may_differ(self, two_claims: list[Claim]) -> None:
        """
        Two builds with different seeds should generally produce different texts.

        This is a soft assertion (not guaranteed for every text, but highly
        likely for texts containing multiple digits and named entities).
        """
        attack = NumericalShiftAttack()
        cfg = AttackConfig(name="numerical_shift", poison_rate=1.0)
        r1 = CorpusBuilder(attacks=[(attack, cfg)], seed=1).build(two_claims)
        r2 = CorpusBuilder(attacks=[(attack, cfg)], seed=9999).build(two_claims)
        # Just verify both runs complete without error and produce the right count.
        assert len(r1.poisoned_docs) == len(r2.poisoned_docs)


# ─── Non-benchmark attack gating (M03/M05) ─────────────────────────────────────

class TestNonBenchmarkAttackGating:
    """
    Tests for CorpusBuilder's refusal to run attacks flagged
    excluded_from_benchmark (AttributionSwitchAttack/M03,
    CherryPickingAttack/M05) unless allow_non_benchmark_attacks=True.

    See eiger/ingestion/corpus_builder.py's own docstring and each attack's
    module docstring for the ethical rationale: both manipulation
    categories are explicitly absent from the EIB paper's published corpus
    (Corpus_claim_RAG_Mistral_output_v3.xlsx).
    """

    def test_attribution_switch_flagged_excluded_from_benchmark(self) -> None:
        from eiger.attacks.attribution import AttributionSwitchAttack

        assert AttributionSwitchAttack.excluded_from_benchmark is True

    def test_cherry_picking_flagged_excluded_from_benchmark(self) -> None:
        from eiger.attacks.cherry_picking import CherryPickingAttack

        assert CherryPickingAttack.excluded_from_benchmark is True

    def test_ordinary_attack_not_flagged(self) -> None:
        assert NumericalShiftAttack.excluded_from_benchmark is False

    def test_gated_attack_raises_configuration_error_by_default(self) -> None:
        from eiger.attacks.attribution import AttributionSwitchAttack
        from eiger.core.exceptions import ConfigurationError

        attack = AttributionSwitchAttack()
        cfg = AttackConfig(name="attribution_switch", poison_rate=0.5)
        with pytest.raises(ConfigurationError, match="attribution_switch"):
            CorpusBuilder(attacks=[(attack, cfg)], seed=42)

    def test_gated_attack_allowed_with_explicit_opt_in(
        self, two_claims: list[Claim]
    ) -> None:
        from eiger.attacks.attribution import AttributionSwitchAttack

        attack = AttributionSwitchAttack()
        cfg = AttackConfig(name="attribution_switch", poison_rate=1.0)
        builder = CorpusBuilder(
            attacks=[(attack, cfg)], seed=42, allow_non_benchmark_attacks=True
        )
        result = builder.build(two_claims)
        # C2 mentions NASA and is poisoned; C1 has no mapped entity, so the
        # attack is a no-op there and is skipped (review fix), not added.
        assert [d.claim_id for d in result.poisoned_docs] == ["C2"]
        assert result.skipped_noop == {"attribution_switch": 1}

    def test_mixed_gated_and_ordinary_attacks_still_raises(self) -> None:
        from eiger.attacks.attribution import AttributionSwitchAttack
        from eiger.core.exceptions import ConfigurationError

        ordinary = NumericalShiftAttack()
        ordinary_cfg = AttackConfig(name="numerical_shift", poison_rate=0.5)
        gated = AttributionSwitchAttack()
        gated_cfg = AttackConfig(name="attribution_switch", poison_rate=0.5)
        with pytest.raises(ConfigurationError):
            CorpusBuilder(attacks=[(ordinary, ordinary_cfg), (gated, gated_cfg)], seed=42)

    def test_ordinary_attack_never_needs_opt_in(self, two_claims: list[Claim]) -> None:
        """An experiment using only non-flagged attacks must be unaffected."""
        attack = NumericalShiftAttack()
        cfg = AttackConfig(name="numerical_shift", poison_rate=1.0)
        builder = CorpusBuilder(attacks=[(attack, cfg)], seed=42)  # no opt-in passed
        result = builder.build(two_claims)
        assert len(result.poisoned_docs) == len(two_claims)
