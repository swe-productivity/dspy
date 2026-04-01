"""
Type definitions for the Pydantic Field GEPA Adapter.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

# Type alias for candidate dictionary
Candidate = dict[str, str]


def make_reflective_example(
    inputs: dict[str, Any],
    generated_outputs: dict[str, Any] | str,
    feedback: str,
) -> dict[str, Any]:
    """Create a ReflectiveExample dict with proper DSPy-compatible keys.

    Note: Uses "Generated Outputs" with space to match DSPy's format.
    """
    return {
        "Inputs": inputs,
        "Generated Outputs": generated_outputs,
        "Feedback": feedback,
    }


@dataclass
class FieldEvaluationResult:
    """Result of evaluating extraction quality at per-field level.

    Attributes:
        overall_score: Aggregated score across all fields (0.0 to 1.0)
        field_scores: Per-field scores mapping field name to score
        field_feedback: Per-field feedback messages for failed extractions
    """
    overall_score: float
    field_scores: dict[str, float] = field(default_factory=dict)
    field_feedback: dict[str, str] = field(default_factory=dict)


@dataclass
class FieldTrajectory:
    """Trajectory data for a single extraction example.

    Attributes:
        example: The input example data
        prediction: The predicted extraction result
        score: Overall score for this example
        field_scores: Per-field scores
        field_feedback: Per-field feedback messages
        error: Error message if extraction failed
    """
    example: dict[str, Any]
    prediction: Any | None
    score: float
    field_scores: dict[str, float] = field(default_factory=dict)
    field_feedback: dict[str, str] = field(default_factory=dict)
    error: str | None = None


# Type alias for field scorer function
# Takes (gold_value, predicted_value, field_name) and returns (score, feedback)
FieldScorerFn = Callable[[Any, Any, str], tuple[float, str]]
