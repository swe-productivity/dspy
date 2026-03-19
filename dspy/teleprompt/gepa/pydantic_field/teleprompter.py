"""
DSPy Teleprompter wrapper for Pydantic field GEPA optimization.

Provides a DSPy-compatible interface for the PydanticFieldGEPAAdapter.
"""

from typing import Any, Callable
from pydantic import BaseModel

import dspy

from .types import FieldScorerFn
from .adapter import PydanticFieldGEPAAdapter


class PydanticFieldGEPA:
    """DSPy-integrated GEPA optimizer for Pydantic field descriptions.

    Provides a compile() interface similar to other DSPy teleprompters,
    using the PydanticFieldGEPAAdapter internally.

    Example:
        class Contact(BaseModel):
            name: str = Field(description="Contact's full name")
            email: str = Field(description="Email address")

        optimizer = PydanticFieldGEPA(
            pydantic_model=Contact,
            metric=extraction_metric,
            evolvable_fields="all",
        )

        optimized = optimizer.compile(ContactExtractor(), trainset=examples)
    """

    def __init__(
        self,
        pydantic_model: type[BaseModel],
        metric: Callable | None = None,
        *,
        evolvable_fields: list[str] | str | None = None,
        field_scorers: dict[str, FieldScorerFn] | None = None,
        field_weights: dict[str, float] | None = None,
        base_instruction: str = "Extract data from the document.",
        input_field_name: str = "document",
        output_field_name: str = "extracted_data",
        max_metric_calls: int = 500,
        num_threads: int | None = None,
        reflection_lm: dspy.LM | None = None,
        verbose: bool = False,
    ):
        """Initialize the optimizer.

        Args:
            pydantic_model: The Pydantic model defining extraction schema
            metric: Optional overall metric function
            evolvable_fields: Fields to evolve ("all", list, or None)
            field_scorers: Custom scorers per field
            field_weights: Weights for field score aggregation
            base_instruction: Initial seed prompt
            input_field_name: Name of input field
            output_field_name: Name of output field
            max_metric_calls: Maximum optimization iterations
            num_threads: Threads for parallel evaluation
            reflection_lm: LM for reflection/proposals
            verbose: Whether to print progress
        """
        self.pydantic_model = pydantic_model
        self.metric = metric
        self.evolvable_fields = evolvable_fields
        self.field_scorers = field_scorers
        self.field_weights = field_weights
        self.base_instruction = base_instruction
        self.input_field_name = input_field_name
        self.output_field_name = output_field_name
        self.max_metric_calls = max_metric_calls
        self.num_threads = num_threads
        self.reflection_lm = reflection_lm
        self.verbose = verbose

    def compile(
        self,
        student: dspy.Module,
        trainset: list[Any],
        valset: list[Any] | None = None,
    ) -> dspy.Module:
        """Compile the student module with optimized field descriptions.

        Args:
            student: The DSPy module to optimize
            trainset: Training examples
            valset: Optional validation examples (defaults to trainset)

        Returns:
            Optimized DSPy module with evolved field descriptions
        """
        # Create the adapter
        adapter = PydanticFieldGEPAAdapter(
            pydantic_model=self.pydantic_model,
            extractor_module=student,
            metric_fn=self.metric,
            evolvable_fields=self.evolvable_fields,
            field_scorers=self.field_scorers,
            field_weights=self.field_weights,
            input_field_name=self.input_field_name,
            output_field_name=self.output_field_name,
            num_threads=self.num_threads,
            reflection_lm=self.reflection_lm,
        )

        # Build seed candidate
        seed_candidate = adapter.build_seed_candidate(self.base_instruction)

        if self.verbose:
            print(f"Initial candidate with {len(seed_candidate)} components:")
            for key, value in seed_candidate.items():
                print(f"  {key}: {value[:50]}..." if len(value) > 50 else f"  {key}: {value}")

        # Try to use GEPA library's optimize function
        try:
            from gepa import optimize

            result = optimize(
                seed_candidate=seed_candidate,
                trainset=trainset,
                valset=valset or trainset,
                adapter=adapter,
                max_metric_calls=self.max_metric_calls,
            )

            best_candidate = result.best_candidate

        except ImportError:
            # Fallback: simple evolution loop without full GEPA
            if self.verbose:
                print("GEPA library not installed, using simplified evolution")

            best_candidate = self._simple_evolution(
                adapter=adapter,
                seed_candidate=seed_candidate,
                trainset=trainset,
                valset=valset or trainset,
            )

        if self.verbose:
            print(f"\nFinal candidate:")
            for key, value in best_candidate.items():
                print(f"  {key}: {value[:50]}..." if len(value) > 50 else f"  {key}: {value}")

        # Build optimized program
        return adapter.build_program(best_candidate)

    def _simple_evolution(
        self,
        adapter: PydanticFieldGEPAAdapter,
        seed_candidate: dict[str, str],
        trainset: list[Any],
        valset: list[Any],
        max_iterations: int = 10,
    ) -> dict[str, str]:
        """Simple evolution loop when GEPA library is not available.

        This is a fallback that performs basic iterative improvement.

        Args:
            adapter: The GEPA adapter
            seed_candidate: Initial candidate
            trainset: Training data
            valset: Validation data
            max_iterations: Maximum improvement iterations

        Returns:
            Best candidate found
        """
        best_candidate = seed_candidate.copy()

        # Evaluate initial candidate
        eval_batch = adapter.evaluate(valset, best_candidate, capture_traces=True)
        best_score = sum(eval_batch.scores) / len(eval_batch.scores) if eval_batch.scores else 0.0

        if self.verbose:
            print(f"Initial score: {best_score:.4f}")

        for iteration in range(max_iterations):
            # Find components with errors
            components_to_update = []
            for key in best_candidate.keys():
                if key == "seed_prompt":
                    if best_score < 0.95:
                        components_to_update.append(key)
                elif key.startswith("field:"):
                    # Check if this field had errors
                    field_name = key.replace("field:", "")
                    if eval_batch.trajectories:
                        for traj in eval_batch.trajectories:
                            if hasattr(traj, "field_scores") and traj.field_scores.get(field_name, 1.0) < 1.0:
                                components_to_update.append(key)
                                break

            if not components_to_update:
                if self.verbose:
                    print(f"No components to update at iteration {iteration + 1}")
                break

            # Build reflective dataset
            reflective_dataset = adapter.make_reflective_dataset(
                best_candidate,
                eval_batch,
                components_to_update,
            )

            if not reflective_dataset:
                break

            # Propose new texts
            try:
                proposed = adapter.propose_new_texts(
                    best_candidate,
                    reflective_dataset,
                    components_to_update,
                )
            except Exception as e:
                if self.verbose:
                    print(f"Proposal failed: {e}")
                break

            if not proposed:
                break

            # Create new candidate
            new_candidate = best_candidate.copy()
            new_candidate.update(proposed)

            # Evaluate new candidate
            new_eval_batch = adapter.evaluate(valset, new_candidate, capture_traces=True)
            new_score = sum(new_eval_batch.scores) / len(new_eval_batch.scores) if new_eval_batch.scores else 0.0

            if self.verbose:
                print(f"Iteration {iteration + 1}: score {new_score:.4f} (was {best_score:.4f})")

            # Accept if improved
            if new_score > best_score:
                best_candidate = new_candidate
                best_score = new_score
                eval_batch = new_eval_batch

                if best_score >= 0.99:
                    if self.verbose:
                        print("Reached near-perfect score, stopping early")
                    break

        return best_candidate
