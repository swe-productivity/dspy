"""
Field description proposer for GEPA evolution.

Proposes improved field descriptions based on extraction failures.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import dspy

from dspy.teleprompt.gepa.pydantic_field.types import Candidate


import dspy


class FieldDescriptionProposalSignature(dspy.Signature):
    """Improve a field description for data extraction.

    The field description guides an LLM to extract a specific piece of information
    from documents. Based on the examples where extraction failed, propose an improved
    description that will help the LLM correctly identify and extract this field.

    Focus on:
    1. Clarifying what the field should contain
    2. Specifying the expected format
    3. Handling edge cases shown in the failures
    4. Being precise but not overly restrictive
    """

    field_name: str = dspy.InputField(desc="Name of the field being extracted")
    current_description: str = dspy.InputField(desc="Current field description that needs improvement")
    extraction_failures: str = dspy.InputField(desc="Examples where extraction failed with feedback")

    improved_description: str = dspy.OutputField(
        desc="Improved field description that addresses the extraction failures. "
             "Should be concise (1-3 sentences) but more precise than the original."
    )


class FieldDescriptionProposer:
    """GEPA-compatible proposer for field descriptions.

    Proposes improved text for both field descriptions and the seed prompt
    based on reflective datasets generated from extraction failures.
    """

    def __init__(self, reflection_lm: dspy.LM | None = None):
        """Initialize the proposer.

        Args:
            reflection_lm: Optional LM to use for reflection. If None, uses default.
        """
        self.reflection_lm = reflection_lm
        self.field_proposer = dspy.Predict(FieldDescriptionProposalSignature)

    def __call__(
        self,
        candidate: Candidate,
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Propose improved texts for components.

        Args:
            candidate: Current candidate dictionary
            reflective_dataset: Dataset of failures per component
            components_to_update: List of components to propose updates for

        Returns:
            Dictionary mapping component names to improved text
        """
        updated: dict[str, str] = {}

        context = dspy.context(lm=self.reflection_lm) if self.reflection_lm else None

        try:
            if context:
                context.__enter__()

            for component_name in components_to_update:
                if component_name not in reflective_dataset:
                    continue

                examples = reflective_dataset[component_name]
                if not examples:
                    continue

                if component_name == "seed_prompt":
                    updated[component_name] = self._propose_instruction(
                        candidate.get(component_name, ""),
                        examples
                    )
                elif component_name.startswith("field:"):
                    field_name = component_name.replace("field:", "")
                    updated[component_name] = self._propose_field_description(
                        field_name,
                        candidate.get(component_name, ""),
                        examples
                    )

        finally:
            if context:
                context.__exit__(None, None, None)

        return updated

    def _propose_field_description(
        self,
        field_name: str,
        current_description: str,
        examples: Sequence[Mapping[str, Any]],
    ) -> str:
        """Propose improved field description.

        Args:
            field_name: Name of the field
            current_description: Current description text
            examples: List of failure examples

        Returns:
            Improved description string
        """
        failures_text = self._format_field_failures(examples)

        result = self.field_proposer(
            field_name=field_name,
            current_description=current_description,
            extraction_failures=failures_text,
        )

        return result.improved_description

    def _propose_instruction(
        self,
        current_instruction: str,
        examples: Sequence[Mapping[str, Any]],
    ) -> str:
        """Propose improved extraction instruction.

        Args:
            current_instruction: Current instruction text
            examples: List of failure examples

        Returns:
            Improved instruction string
        """
        from gepa.strategies.instruction_proposal import InstructionProposalSignature as GEPAProposal

        result = GEPAProposal.run(
            lm=lambda x: [dspy.settings.lm(x)[0]],
            input_dict={
                "current_instruction_doc": current_instruction,
                "dataset_with_feedback": list(examples),
            },
        )
        return result["new_instruction"]

    def _format_field_failures(
        self,
        examples: Sequence[Mapping[str, Any]],
        max_examples: int = 5,
    ) -> str:
        """Format field extraction failures for the proposer.

        Args:
            examples: List of failure examples
            max_examples: Maximum number of examples to include

        Returns:
            Formatted string of failures
        """
        formatted = []

        for i, ex in enumerate(examples[:max_examples]):
            inputs = ex.get("Inputs", {})
            # Note: "Generated Outputs" has a space - DSPy format
            outputs = ex.get("Generated Outputs", ex.get("GeneratedOutputs", {}))
            feedback = ex.get("Feedback", "No feedback")

            formatted.append(f"Example {i + 1}:")

            # Document snippet
            doc = inputs.get("Document", inputs.get("document", ""))
            if doc:
                doc_snippet = doc[:300] + "..." if len(doc) > 300 else doc
                formatted.append(f"  Document: {doc_snippet}")

            # Expected vs extracted - handle both key formats
            expected = outputs.get("Expected Value", outputs.get("ExpectedValue", "N/A"))
            extracted = outputs.get("Extracted Value", outputs.get("ExtractedValue", "N/A"))
            formatted.append(f"  Expected: {expected}")
            formatted.append(f"  Extracted: {extracted}")

            # Feedback
            formatted.append(f"  Feedback: {feedback}")
            formatted.append("")

        return "\n".join(formatted)
