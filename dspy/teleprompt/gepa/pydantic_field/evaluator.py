"""
Field evaluator for per-field extraction scoring.

Provides scoring at both field and overall levels with
support for custom scorers and weighted aggregation.
"""

from typing import Any, Callable
from pydantic import BaseModel

from dspy.teleprompt.gepa.pydantic_field.types import FieldEvaluationResult, FieldScorerFn


class FieldEvaluator:
    """Evaluates extraction quality at per-field and overall levels.

    Supports:
    - Default exact match scoring
    - Custom field-specific scorers
    - Weighted aggregation of field scores
    - Feedback generation for failed extractions
    """

    def __init__(
        self,
        field_scorers: dict[str, FieldScorerFn] | None = None,
        field_weights: dict[str, float] | None = None,
        aggregation: str = "weighted_mean",
    ):
        """Initialize the field evaluator.

        Args:
            field_scorers: Custom scorers per field name
            field_weights: Weights for each field in aggregation
            aggregation: Aggregation method ("weighted_mean" or "min")
        """
        self.field_scorers = field_scorers or {}
        self.field_weights = field_weights
        self.aggregation = aggregation

    def evaluate(
        self,
        gold: BaseModel,
        predicted: BaseModel | None,
    ) -> FieldEvaluationResult:
        """Evaluate extraction quality.

        Args:
            gold: Ground truth Pydantic model instance
            predicted: Predicted Pydantic model instance

        Returns:
            FieldEvaluationResult with per-field and overall scores
        """
        # Access model_fields from the class, not the instance
        model_class = type(gold)

        if predicted is None:
            # Complete failure - score 0 for all fields
            field_scores = {
                name: 0.0 for name in model_class.model_fields.keys()
            }
            field_feedback = {
                name: "Extraction failed completely" for name in model_class.model_fields.keys()
            }
            return FieldEvaluationResult(
                overall_score=0.0,
                field_scores=field_scores,
                field_feedback=field_feedback,
            )

        field_scores: dict[str, float] = {}
        field_feedback: dict[str, str] = {}

        for field_name in model_class.model_fields.keys():
            gold_value = getattr(gold, field_name, None)
            pred_value = getattr(predicted, field_name, None)

            # Use field-specific scorer if available
            scorer = self.field_scorers.get(field_name, self._default_scorer)
            score, feedback = scorer(gold_value, pred_value, field_name)

            field_scores[field_name] = score
            if feedback:
                field_feedback[field_name] = feedback

        # Aggregate scores
        overall_score = self._aggregate_scores(field_scores)

        return FieldEvaluationResult(
            overall_score=overall_score,
            field_scores=field_scores,
            field_feedback=field_feedback,
        )

    def _default_scorer(
        self,
        gold: Any,
        pred: Any,
        field_name: str,
    ) -> tuple[float, str]:
        """Default exact match scoring with normalization.

        Args:
            gold: Ground truth value
            pred: Predicted value
            field_name: Name of the field being scored

        Returns:
            Tuple of (score, feedback)
        """
        # Both null
        if gold is None and pred is None:
            return 1.0, ""

        # Missing prediction
        if pred is None:
            return 0.0, f"Missing value, expected: {gold}"

        # Extra prediction when null expected
        if gold is None:
            return 0.0, f"Expected null, got: {pred}"

        # Normalize strings for comparison
        gold_str = str(gold).lower().strip()
        pred_str = str(pred).lower().strip()

        # Exact match
        if gold_str == pred_str:
            return 1.0, ""

        # Check for partial/contained match
        if isinstance(gold, str) and isinstance(pred, str):
            if gold_str in pred_str or pred_str in gold_str:
                return 0.5, f"Partial match: expected '{gold}', got '{pred}'"

        return 0.0, f"Mismatch: expected '{gold}', got '{pred}'"

    def _aggregate_scores(self, field_scores: dict[str, float]) -> float:
        """Aggregate per-field scores to overall score.

        Args:
            field_scores: Dictionary of field name to score

        Returns:
            Aggregated overall score
        """
        if not field_scores:
            return 0.0

        if self.aggregation == "min":
            return min(field_scores.values())

        # Weighted mean (default)
        weights = self.field_weights or {f: 1.0 for f in field_scores}
        total_weight = sum(weights.get(f, 1.0) for f in field_scores)

        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            score * weights.get(name, 1.0)
            for name, score in field_scores.items()
        )

        return weighted_sum / total_weight


class FeedbackGenerator:
    """Generates structured feedback for GEPA reflection."""

    def generate_feedback(
        self,
        eval_result: FieldEvaluationResult,
        document: str | None = None,
        predicted: BaseModel | None = None,
    ) -> str:
        """Generate feedback text for GEPA reflection.

        Args:
            eval_result: The evaluation result
            document: Optional input document snippet
            predicted: Optional predicted model instance

        Returns:
            Feedback string for the reflection LM
        """
        feedback_parts = []

        # Overall status
        if eval_result.overall_score >= 0.95:
            feedback_parts.append(f"Extraction score: {eval_result.overall_score:.2f} (excellent)")
        elif eval_result.overall_score >= 0.7:
            feedback_parts.append(f"Extraction score: {eval_result.overall_score:.2f} (good, minor issues)")
        else:
            feedback_parts.append(f"Extraction score: {eval_result.overall_score:.2f} (needs improvement)")

        # Per-field feedback for errors
        error_fields = [
            (name, score, eval_result.field_feedback.get(name, ""))
            for name, score in eval_result.field_scores.items()
            if score < 1.0
        ]

        if error_fields:
            feedback_parts.append("\nField-specific issues:")
            for field_name, score, field_feedback in error_fields:
                if field_feedback:
                    feedback_parts.append(f"  - {field_name} (score: {score:.2f}): {field_feedback}")
                else:
                    feedback_parts.append(f"  - {field_name} (score: {score:.2f})")

        return "\n".join(feedback_parts)


# Pre-built scorer functions for common use cases

def fuzzy_string_scorer(
    gold: Any,
    pred: Any,
    field_name: str,
    threshold: float = 0.8,
) -> tuple[float, str]:
    """Fuzzy string matching scorer using simple ratio.

    Args:
        gold: Ground truth value
        pred: Predicted value
        field_name: Name of the field
        threshold: Minimum similarity for full score

    Returns:
        Tuple of (score, feedback)
    """
    if gold is None and pred is None:
        return 1.0, ""
    if gold is None or pred is None:
        return 0.0, f"Missing: gold={gold}, pred={pred}"

    gold_str = str(gold).lower().strip()
    pred_str = str(pred).lower().strip()

    if gold_str == pred_str:
        return 1.0, ""

    # Simple character-based similarity
    similarity = _simple_similarity(gold_str, pred_str)

    if similarity >= threshold:
        return 1.0, f"Fuzzy match (similarity: {similarity:.2f})"
    elif similarity >= 0.5:
        return similarity, f"Partial match (similarity: {similarity:.2f})"
    else:
        return 0.0, f"Low similarity ({similarity:.2f}): expected '{gold}', got '{pred}'"


def _simple_similarity(s1: str, s2: str) -> float:
    """Simple character-based similarity measure."""
    if not s1 or not s2:
        return 0.0

    # Character overlap
    set1 = set(s1)
    set2 = set(s2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def email_scorer(
    gold: Any,
    pred: Any,
    field_name: str,
) -> tuple[float, str]:
    """Scorer for email addresses with normalization.

    Args:
        gold: Ground truth email
        pred: Predicted email
        field_name: Name of the field

    Returns:
        Tuple of (score, feedback)
    """
    if gold is None and pred is None:
        return 1.0, ""
    if gold is None or pred is None:
        return 0.0, f"Missing: gold={gold}, pred={pred}"

    # Normalize emails
    gold_email = str(gold).lower().strip()
    pred_email = str(pred).lower().strip()

    if gold_email == pred_email:
        return 1.0, ""

    # Check if domains match at least
    gold_parts = gold_email.split("@")
    pred_parts = pred_email.split("@")

    if len(gold_parts) == 2 and len(pred_parts) == 2:
        if gold_parts[1] == pred_parts[1]:
            return 0.5, f"Domain matches but username differs: expected '{gold}', got '{pred}'"

    return 0.0, f"Email mismatch: expected '{gold}', got '{pred}'"
