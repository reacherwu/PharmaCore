"""Drug repurposing module — find new uses for existing drugs."""
from pharmacore.repurposing.engine import (
    DrugRepurposingEngine,
    RepurposingCandidate,
    RepurposingResult,
    KNOWN_DRUGS,
)

__all__ = [
    "DrugRepurposingEngine",
    "RepurposingCandidate",
    "RepurposingResult",
    "KNOWN_DRUGS",
]
