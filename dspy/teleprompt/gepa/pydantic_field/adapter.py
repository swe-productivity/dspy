"""
Main GEPA adapter for Pydantic field description evolution.

Implements the GEPAAdapter protocol with per-field evaluation,
reflective dataset construction, and field-specific proposals.
"""

from typing import Any, Callable, Mapping, Sequence
from pydantic import BaseModel
from copy import deepcopy

import dspy

from dspy.teleprompt.gepa.pydantic_field.types import (
    Candidate,
    FieldEvaluationResult,
    FieldTrajectory,
    FieldScorerFn,
    make_reflective_example,
)
from dspy.teleprompt.gepa.pydantic_field.candidate import CandidateBuilder
from dspy.teleprompt.gepa.pydantic_field.signature_factory import SignatureFactory
from dspy.teleprompt.gepa.pydantic_field.evaluator import FieldEvaluator, FeedbackGenerator
from dspy.teleprompt.gepa.pydantic_field.proposer import FieldDescriptionProposer
from gepa import EvaluationBatch


class PydanticFieldGEPAAdapter:
    """GEPA adapter for evolving Pydantic field descriptions independently.

    This adapter treats each field description as a separate evolvable component,
    allowing GEPA to optimize extraction prompts at a fine-grained level.

    Implements the GEPAAdapter protocol:
    - evaluate(): Execute extraction and score per-field
    - make_reflective_dataset(): Build per-field failure examples
    - propose_new_texts(): Propose improved field descriptions

    Example:
        class Contact(BaseModel):
            name: str = Field(description="Contact's full name")
            email: str = Field(description="Email address")

        adapter = PydanticFieldGEPAAdapter(
            pydantic_model=Contact,
            extractor_module=my_extractor,
            metric_fn=extraction_metric,
            evolvable_fields="all",
        )
    """

    def __init__(
        self,
        pydantic_model: type[BaseModel],
        extractor_module: dspy.Module,
        metric_fn: Callable | None = None,
        *,
        evolvable_fields: list[str] | str | None = None,
        field_scorers: dict[str, FieldScorerFn] | None = None,
        field_weights: dict[str, float] | None = None,
        input_field_name: str = "document",
        output_field_name: str = "extracted_data",
        num_threads: int | None = None,
        failure_score: float = 0.0,
        reflection_lm: dspy.LM | None = None,
    ):
        """Initialize the adapter.

        Args:
            pydantic_model: The Pydantic model defining the extraction schema
            extractor_module: The DSPy module performing extraction
            metric_fn: Optional overall metric function (ignored if using per-field)
            evolvable_fields: Fields to evolve ("all", list, or None for seed_prompt only)
            field_scorers: Custom scorers per field
            field_weights: Weights for field score aggregation
            input_field_name: Name of input field in signature
            output_field_name: Name of output field in signature
            num_threads: Number of threads for parallel evaluation
            failure_score: Score to assign on complete failure
            reflection_lm: LM to use for reflection/proposals
        """
        self.pydantic_model = pydantic_model
        self.extractor_module = extractor_module
        self.metric_fn = metric_fn
        self.evolvable_fields = evolvable_fields
        self.num_threads = num_threads
        self.failure_score = failure_score
        self.reflection_lm = reflection_lm

        # Initialize components
        self.candidate_builder = CandidateBuilder(
            pydantic_model=pydantic_model,
            evolvable_fields=evolvable_fields,
        )
        self.signature_factory = SignatureFactory(
            pydantic_model=pydantic_model,
            input_field_name=input_field_name,
            output_field_name=output_field_name,
        )
        self.field_evaluator = FieldEvaluator(
            field_scorers=field_scorers,
            field_weights=field_weights,
        )
        self.feedback_generator = FeedbackGenerator()
        self.proposer = FieldDescriptionProposer(reflection_lm=reflection_lm)

        # Store field names
        self.input_field_name = input_field_name
        self.output_field_name = output_field_name

    def build_seed_candidate(self, base_instruction: str) -> Candidate:
        """Build the initial candidate from the Pydantic model.

        Args:
            base_instruction: The seed prompt / base instruction

        Returns:
            Initial candidate dictionary
        """
        return self.candidate_builder.build_seed_candidate(base_instruction)

    def build_program(self, candidate: Candidate) -> dspy.Module:
        """Build a DSPy program from the candidate configuration.

        Args:
            candidate: The candidate dictionary with evolved descriptions

        Returns:
            A configured DSPy module ready for extraction
        """
        # Build updated signature
        signature = self.signature_factory.build_signature(candidate)

        # Create full instruction with field guidance
        instruction = self.signature_factory.inject_field_descriptions_into_instruction(candidate)

        # Clone the module and update signature
        program = deepcopy(self.extractor_module)

        # Update predictor signatures
        for name, pred in program.named_predictors():
            pred.signature = signature.with_instructions(instruction)

        return program

    def evaluate(
        self,
        batch: list[Any],
        candidate: Candidate,
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Execute the program on a batch and return evaluation results.

        Args:
            batch: List of examples (dspy.Example or dict with gold/document)
            candidate: The candidate configuration
            capture_traces: Whether to capture execution trajectories

        Returns:
            EvaluationBatch with outputs, scores, and optional trajectories
        """
        program = self.build_program(candidate)

        outputs: list[Any] = []
        scores: list[float] = []
        trajectories: list[FieldTrajectory] = []

        for example in batch:
            try:
                # Get input document
                if hasattr(example, "inputs"):
                    inputs = example.inputs()
                elif isinstance(example, dict):
                    inputs = {self.input_field_name: example.get(self.input_field_name, example.get("document", ""))}
                else:
                    inputs = {self.input_field_name: str(example)}

                # Run extraction
                prediction = program(**inputs)

                # Get predicted model
                predicted_model = getattr(prediction, self.output_field_name, None)

                # Get gold model
                if hasattr(example, "gold"):
                    gold_model = example.gold
                elif isinstance(example, dict):
                    gold_model = example.get("gold", example.get("expected"))
                else:
                    gold_model = None

                # Evaluate with per-field scoring
                if gold_model is not None:
                    eval_result = self.field_evaluator.evaluate(gold_model, predicted_model)
                else:
                    # No gold, use metric_fn if available
                    if self.metric_fn:
                        score = self.metric_fn(example, prediction)
                        eval_result = FieldEvaluationResult(overall_score=float(score))
                    else:
                        eval_result = FieldEvaluationResult(overall_score=1.0)

                outputs.append(prediction)
                scores.append(eval_result.overall_score)

                if capture_traces:
                    trajectories.append(FieldTrajectory(
                        example=dict(example) if isinstance(example, dict) else {"input": str(example)},
                        prediction=predicted_model,
                        score=eval_result.overall_score,
                        field_scores=eval_result.field_scores,
                        field_feedback=eval_result.field_feedback,
                    ))

            except Exception as e:
                outputs.append(None)
                scores.append(self.failure_score)

                if capture_traces:
                    trajectories.append(FieldTrajectory(
                        example=dict(example) if isinstance(example, dict) else {"input": str(example)},
                        prediction=None,
                        score=self.failure_score,
                        error=str(e),
                    ))

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: Candidate,
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Transform evaluation results into a learning dataset.

        For field components, filters examples to those where that specific
        field had errors, providing focused reflection data.

        Args:
            candidate: Current candidate configuration
            eval_batch: Evaluation results with trajectories
            components_to_update: Components that need improvement

        Returns:
            Dictionary mapping component names to reflective examples
        """
        reflective_dataset: dict[str, list[dict[str, Any]]] = {}

        if not eval_batch.trajectories:
            return reflective_dataset

        for component_name in components_to_update:
            examples: list[dict[str, Any]] = []

            for trajectory in eval_batch.trajectories:
                if not isinstance(trajectory, FieldTrajectory):
                    continue

                if component_name == "seed_prompt":
                    # Include all examples with errors for global instruction
                    if trajectory.score < 1.0:
                        examples.append(self._format_reflective_example(
                            trajectory,
                            candidate,
                            component_name,
                        ))

                elif component_name.startswith("field:"):
                    field_name = component_name.replace("field:", "")

                    # Only include examples where this field had issues
                    if trajectory.field_scores.get(field_name, 1.0) < 1.0:
                        examples.append(self._format_field_reflective_example(
                            trajectory,
                            candidate,
                            field_name,
                        ))

            if examples:
                reflective_dataset[component_name] = examples

        return reflective_dataset

    def _format_reflective_example(
        self,
        trajectory: FieldTrajectory,
        candidate: Candidate,
        component_name: str,
    ) -> dict[str, Any]:
        """Format a general reflective example.

        Args:
            trajectory: The evaluation trajectory
            candidate: Current candidate
            component_name: Component being updated

        Returns:
            ReflectiveExample dictionary with DSPy-compatible keys
        """
        feedback = self.feedback_generator.generate_feedback(
            FieldEvaluationResult(
                overall_score=trajectory.score,
                field_scores=trajectory.field_scores,
                field_feedback=trajectory.field_feedback,
            ),
            document=trajectory.example.get(self.input_field_name, ""),
            predicted=trajectory.prediction,
        )

        return make_reflective_example(
            inputs={
                "Document": str(trajectory.example.get(self.input_field_name, ""))[:500],
                "Current Instruction": candidate.get("seed_prompt", ""),
            },
            generated_outputs={
                "Prediction": str(trajectory.prediction) if trajectory.prediction else "Failed",
                "Score": str(trajectory.score),
            },
            feedback=feedback,
        )

    def _format_field_reflective_example(
        self,
        trajectory: FieldTrajectory,
        candidate: Candidate,
        field_name: str,
    ) -> dict[str, Any]:
        """Format a field-specific reflective example.

        Args:
            trajectory: The evaluation trajectory
            candidate: Current candidate
            field_name: The field that had errors

        Returns:
            ReflectiveExample dictionary with DSPy-compatible keys
        """
        # Get field values
        predicted_value = getattr(trajectory.prediction, field_name, None) if trajectory.prediction else None
        expected_value = trajectory.example.get("gold", {})
        if hasattr(expected_value, field_name):
            expected_value = getattr(expected_value, field_name)
        elif isinstance(expected_value, dict):
            expected_value = expected_value.get(field_name, "N/A")
        else:
            expected_value = "N/A"

        return make_reflective_example(
            inputs={
                "Document": str(trajectory.example.get(self.input_field_name, ""))[:500],
                "Target Field": field_name,
                "Current Description": candidate.get(f"field:{field_name}", ""),
            },
            generated_outputs={
                "Extracted Value": str(predicted_value) if predicted_value else "N/A",
                "Expected Value": str(expected_value),
            },
            feedback=trajectory.field_feedback.get(field_name, "Extraction error"),
        )

    def propose_new_texts(
        self,
        candidate: Candidate,
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Propose improved texts for the components.

        Args:
            candidate: Current candidate configuration
            reflective_dataset: Learning dataset per component
            components_to_update: Components to propose updates for

        Returns:
            Dictionary mapping component names to improved text
        """
        return self.proposer(
            candidate=candidate,
            reflective_dataset=reflective_dataset,
            components_to_update=components_to_update,
        )

    def stripped_lm_call(self, x: str) -> list[str]:
        """Wrap LM call to always return strings.

        Extracts text from dict outputs if needed (e.g., when LM returns
        {"text": "...", "reasoning": "..."}).

        Args:
            x: Input prompt string

        Returns:
            List of string outputs from the LM
        """
        if self.reflection_lm is None:
            raise ValueError("reflection_lm is required for stripped_lm_call")

        raw_outputs = self.reflection_lm(x)
        outputs = []
        for raw_output in raw_outputs:
            if isinstance(raw_output, str):
                outputs.append(raw_output)
            elif isinstance(raw_output, dict):
                if "text" not in raw_output:
                    raise KeyError("Missing 'text' field in the output from the LM!")
                outputs.append(raw_output["text"])
            else:
                raise TypeError(f"Unexpected output type from LM: {type(raw_output)}")

        return outputs
