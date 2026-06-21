"""Work around a TRL 0.24 / transformers 5.5 incompatibility (D20).

transformers 5.5's `_is_package_available()` returns a `(bool, version)` TUPLE, but TRL 0.24's
`is_X_available()` helpers return that value directly and guard imports with `if is_X_available():`.
A non-empty tuple is always truthy, so TRL eagerly imports ABSENT optional deps (mergekit,
llm_blender, ...) and `from trl import DPOTrainer` crashes. Fix at the root: coerce the cached
`trl.import_utils._*_available` tuples to plain bools so the guards behave correctly. No env
changes, no stubs, no dependency risk (vs. installing mergekit/llm_blender, which could disturb the
pinned proven set, D6).

Call patch_trl_availability() BEFORE importing any TRL trainer.
"""
from __future__ import annotations


def patch_trl_availability() -> None:
    import trl.import_utils as iu  # importing trl is lazy — does not pull the trainers yet

    for name in list(vars(iu)):
        if name.endswith("_available"):
            val = getattr(iu, name)
            if isinstance(val, tuple):  # transformers 5.5 returns (available, version)
                setattr(iu, name, bool(val[0]))
