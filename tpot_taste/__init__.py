"""tpot-taste: small-model post-training for tpot / tech-Twitter virality.

Two models:
  * Taste Scorer  — encoder (Bradley-Terry + regression) that scores a post's
                    virality / tpot-fit. Standalone "is this worth posting?"
                    and the reward signal for the generator.
  * Taste Writer  — decoder (SFT -> DPO -> GRPO) that ideates and rewrites
                    posts, optimized against the Scorer.

See docs/design/01-system-design.md.
"""

__version__ = "0.1.0"
