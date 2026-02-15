"""
RunContext: Encapsulate run-level state to avoid passing many parameters.

The RunContext object carries run_seed, provenance, and timing information
through the pipeline, and provides hierarchical RNG methods for different
subsystems (tracking, visualization, perturbation).
"""

from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from .provenance import get_provenance_dict


DEFAULT_SEED = 42


@dataclass
class RunContext:
    """Context object for a single csttool run.

    Attributes:
        run_seed: Random seed for this run (default: 42)
        provenance: Dict containing git hash, Python version, dependencies, platform
        start_time: Timestamp when run started
    """
    run_seed: int = DEFAULT_SEED
    provenance: dict = field(default_factory=get_provenance_dict)
    start_time: datetime = field(default_factory=datetime.now)

    def rng_tracking_seed(self) -> int:
        """Return seed for DIPY LocalTracking (legacy API requires int)."""
        return self.run_seed

    def rng_viz(self) -> np.random.Generator:
        """Return RNG for visualization subsampling.

        Uses hash-based subseed derivation to ensure visualization randomness
        is independent of tracking randomness.
        """
        subseed = hash(f"{self.run_seed}:viz") & 0xFFFFFFFF
        return np.random.default_rng(subseed)

    def rng_perturb(self, offset: int) -> np.random.Generator:
        """Return RNG for sensitivity tests with replicate offset.

        Args:
            offset: Replicate number (0, 1, 2, ...) to ensure different
                    perturbations across replicates

        Returns:
            numpy.random.Generator with unique seed for this replicate
        """
        subseed = hash(f"{self.run_seed}:perturb:{offset}") & 0xFFFFFFFF
        return np.random.default_rng(subseed)
