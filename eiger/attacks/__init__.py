"""
Adversarial poisoning attacks for RAG corpus contamination.

This package exposes every concrete attack class and the three registry
helpers needed to discover and instantiate them at runtime.

Responsibilities of this module
--------------------------------
- Re-export the registry API (register_attack, get_attack, list_attacks)
  so callers never have to import from eiger.attacks.registry directly.
- Re-export every built-in attack class for direct import convenience.
- Trigger the auto-registration of all built-in attacks by calling
  register_attack() at import time.

What this module does NOT do
-----------------------------
- It does not define attack logic (see the individual attack modules).
- It does not load third-party / entry-point attacks; that responsibility
  belongs to the framework's plugin-loading layer.
"""

# ─── Registry helpers ───────────────────────────────────────────────────────

from eiger.attacks.registry import register_attack, get_attack, list_attacks

# ─── Built-in attack classes ─────────────────────────────────────────────────

from eiger.attacks.numerical import NumericalShiftAttack
from eiger.attacks.attribution import AttributionSwitchAttack
from eiger.attacks.temporal import DateManipulationAttack
from eiger.attacks.causal import CausalManipulationAttack
from eiger.attacks.cherry_picking import CherryPickingAttack
from eiger.attacks.missing_context import MissingContextAttack
from eiger.attacks.corpus_precomputed import CorpusPrecomputedAttack

# ─── Auto-registration ───────────────────────────────────────────────────────

# Register every built-in attack class into the global registry immediately
# on package import. This ensures that any code which does
#   `import eiger.attacks`
# can subsequently call `get_attack("numerical_shift")` without any
# additional setup. register_attack() is idempotent, so repeated imports
# (e.g. in tests) are harmless.
register_attack(NumericalShiftAttack)
register_attack(AttributionSwitchAttack)
register_attack(DateManipulationAttack)
register_attack(CausalManipulationAttack)
register_attack(CherryPickingAttack)
register_attack(MissingContextAttack)
register_attack(CorpusPrecomputedAttack)

# ─── Public API declaration ───────────────────────────────────────────────────

# __all__ governs what `from eiger.attacks import *` exposes and also
# serves as the canonical list of public symbols for IDE auto-complete.
__all__ = [
    # Registry interface
    "register_attack",
    "get_attack",
    "list_attacks",
    # Concrete attack classes
    "NumericalShiftAttack",
    "AttributionSwitchAttack",
    "DateManipulationAttack",
    "CausalManipulationAttack",
    "CherryPickingAttack",
    "MissingContextAttack",
    "CorpusPrecomputedAttack",
]
