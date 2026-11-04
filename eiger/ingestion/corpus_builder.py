"""
CorpusBuilder: converts Claims into a mixed corpus of ground-truth
and adversarially poisoned Documents, ready for vector store ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from eiger.core.exceptions import ConfigurationError
from eiger.core.interfaces import BaseAttack
from eiger.core.models import Claim, Document, PoisonedDocument, AttackConfig
from eiger.utils.logging import get_logger
from eiger.utils.seeding import derive_seed

log = get_logger(__name__)


@dataclass
class CorpusBuilderResult:
    ground_truth_docs: list[Document] = field(default_factory=list)
    poisoned_docs: list[PoisonedDocument] = field(default_factory=list)
    # Attack applications whose output was identical to the ground truth,
    # per attack name. They are NOT added to the corpus (see build()).
    skipped_noop: dict[str, int] = field(default_factory=dict)

    @property
    def all_documents(self) -> list[Document]:
        return self.ground_truth_docs + self.poisoned_docs  # type: ignore[return-value]

    @property
    def poison_ratio(self) -> float:
        total = len(self.all_documents)
        return len(self.poisoned_docs) / total if total > 0 else 0.0


class CorpusBuilder:
    """
    Builds a mixed (ground-truth + poisoned) corpus from a list of Claims.

    Each claim generates exactly one ground-truth document.
    For each attack in the attack list, it is applied to each claim
    according to the configured poison_rate.

    Args:
        attacks: Resolved (BaseAttack instance, AttackConfig) pairs.
        seed:    Experiment-level seed for reproducible per-(claim, attack)
                 poisoning decisions.
        allow_non_benchmark_attacks: Must be True to include any attack
                 whose ``excluded_from_benchmark`` class attribute is True
                 (currently ``AttributionSwitchAttack``/M03 and
                 ``CherryPickingAttack``/M05 — see their own docstrings).
                 Both manipulation categories are explicitly absent from the
                 EIB paper's published corpus (Corpus_claim_RAG_Mistral_
                 output_v3.xlsx) for ethical/reputational reasons stated in
                 the paper itself. Defaults to False so a paper-facing
                 experiment can never silently include a manipulation
                 category the ethical protocol excluded — the constructor
                 raises ``ConfigurationError`` immediately if a flagged
                 attack is present without this override, rather than
                 letting it run and discovering the mismatch only when
                 someone reads the results later.
    """

    def __init__(
        self,
        attacks: list[tuple[BaseAttack, AttackConfig]],
        seed: int = 42,
        allow_non_benchmark_attacks: bool = False,
    ) -> None:
        if not allow_non_benchmark_attacks:
            gated = [attack.name for attack, _ in attacks if attack.excluded_from_benchmark]
            if gated:
                raise ConfigurationError(
                    f"Attack(s) {gated} are excluded from the EIB paper's published "
                    "benchmark corpus (Corpus_claim_RAG_Mistral_output_v3.xlsx) for "
                    "ethical/reputational reasons documented in the paper itself "
                    "(see each attack's own docstring). Pass "
                    "allow_non_benchmark_attacks=True to CorpusBuilder if you are "
                    "intentionally running a non-benchmark engineering ablation, "
                    "not producing results reported as part of the EIB benchmark."
                )
        self.attacks = attacks
        self.seed = seed
        self.allow_non_benchmark_attacks = allow_non_benchmark_attacks

    def build(self, claims: list[Claim]) -> CorpusBuilderResult:
        """
        Build the full corpus from a list of claims.

        Args:
            claims: List of source claims (loaded from a dataset).

        Returns:
            CorpusBuilderResult with ground-truth and poisoned documents separated.
        """
        result = CorpusBuilderResult()
        log.info("corpus_builder.start", n_claims=len(claims), n_attacks=len(self.attacks))

        from eiger.utils.seeding import make_rng

        for claim in claims:
            # Always add the ground-truth document
            gt_doc = Document(
                doc_id=f"gt_{claim.claim_id}",
                claim_id=claim.claim_id,
                text=claim.original_fact,
                doc_type="ground_truth",
                metadata={"source_dataset": claim.source_dataset},
                # Propagate the claim's ethics/threat-model classification
                # (docs/ETHICS_AND_THREAT_MODEL.md §5/§7) onto its ground-truth
                # document — an unclassified claim (None) stays unclassified,
                # never silently promoted to "safe".
                sensitivity_class=claim.sensitivity_class,
                risk_level=claim.risk_level,
                # Source dataset's own true/false rating — independent of
                # doc_type above, which is EIGER's manipulation_status.
                # See eiger.core.models.GroundTruthLabel.
                ground_truth_label=claim.ground_truth_label,
            )
            result.ground_truth_docs.append(gt_doc)

            # Apply each configured attack with its poison_rate
            for attack, attack_cfg in self.attacks:
                # Per-document RNG: deterministic but independent per (claim, attack)
                rng = make_rng(derive_seed(self.seed, claim.claim_id, attack.name))
                if rng.random() <= attack_cfg.poison_rate:
                    doc_seed = derive_seed(self.seed, claim.claim_id, attack.name, "apply")
                    poisoned = attack.apply(gt_doc, seed=doc_seed, **attack_cfg.params)
                    # Bug fix (November 2026 review): an attack that finds
                    # nothing to change (e.g. numerical_shift on a claim with
                    # no digits) returns the ground-truth text unchanged. Such a
                    # "poisoned" document is not a manipulation: counting it
                    # inflated PRR/PRD, and its exact score tie with the ground
                    # truth made rank order (and PRD@1) non-deterministic.
                    # Skip it and count it, so the effective manipulation rate
                    # is visible in the logs.
                    if poisoned.text.strip() == gt_doc.text.strip():
                        result.skipped_noop[attack.name] = result.skipped_noop.get(attack.name, 0) + 1
                        continue
                    # Bug fix (November 2026 review): attacks copy the source
                    # document's doc_id, so a poisoned document and its ground
                    # truth shared the same id. HybridRetriever fuses rankings
                    # by doc_id and therefore merged the two, silently hiding
                    # the poisoned document (PRR under-counted for hybrid).
                    # Give every poisoned document a unique, deterministic id.
                    poisoned = poisoned.model_copy(
                        update={"doc_id": f"{gt_doc.doc_id}__{attack.name}"}
                    )
                    result.poisoned_docs.append(poisoned)

        log.info(
            "corpus_builder.complete",
            n_ground_truth=len(result.ground_truth_docs),
            n_poisoned=len(result.poisoned_docs),
            poison_ratio=f"{result.poison_ratio:.2%}",
            skipped_noop=result.skipped_noop,
        )
        return result
