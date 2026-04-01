"""
Signature factory for dynamically creating DSPy signatures.

Uses DSPy's signature utilities to update field descriptions
based on evolved candidate values.
"""

from typing import Any
from pydantic import BaseModel

import dspy
from dspy import Signature

from dspy.teleprompt.gepa.pydantic_field.types import Candidate


class SignatureFactory:
    """Factory for creating and updating DSPy signatures from Pydantic models.

    This factory dynamically modifies signatures by:
    1. Updating the instruction (seed_prompt)
    2. Updating individual field descriptions from the candidate

    Uses DSPy's json_schema_extra["desc"] format for field descriptions.
    """

    def __init__(
        self,
        pydantic_model: type[BaseModel],
        input_field_name: str = "document",
        output_field_name: str = "extracted_data",
    ):
        """Initialize the signature factory.

        Args:
            pydantic_model: The Pydantic model for the output type
            input_field_name: Name of the input field (default: "document")
            output_field_name: Name of the output field (default: "extracted_data")
        """
        self.pydantic_model = pydantic_model
        self.input_field_name = input_field_name
        self.output_field_name = output_field_name

    def build_base_signature(self, instruction: str) -> type[Signature]:
        """Build the base signature with Pydantic model as output type.

        Args:
            instruction: The signature instruction text

        Returns:
            A DSPy Signature class with the appropriate fields
        """
        # Create signature with InputField and OutputField
        fields = {
            self.input_field_name: (str, dspy.InputField(desc="The input document to extract from")),
            self.output_field_name: (self.pydantic_model, dspy.OutputField(desc=self._build_output_description({}))),
        }

        return dspy.Signature(fields, instruction)

    def build_signature(
        self,
        candidate: Candidate,
        base_signature: type[Signature] | None = None
    ) -> type[Signature]:
        """Build a signature with evolved field descriptions from candidate.

        Args:
            candidate: The candidate dictionary with evolved descriptions
            base_signature: Optional base signature to update; if None, creates new

        Returns:
            A DSPy Signature class with updated descriptions
        """
        instruction = candidate.get("seed_prompt", "Extract data from the document.")

        if base_signature is None:
            base_signature = self.build_base_signature(instruction)

        # Update instruction from seed_prompt
        sig = base_signature.with_instructions(instruction)

        # Build output description from field descriptions
        output_desc = self._build_output_description(candidate)

        # Update output field description using with_updated_fields
        # This updates json_schema_extra["desc"]
        sig = sig.with_updated_fields(self.output_field_name, desc=output_desc)

        return sig

    def _build_output_description(self, candidate: Candidate) -> str:
        """Build the output field description from evolved field descriptions.

        Args:
            candidate: The candidate dictionary

        Returns:
            Combined description for the output field
        """
        parts = [f"Extract into a {self.pydantic_model.__name__} object:"]

        # Get field descriptions from candidate
        for key, description in sorted(candidate.items()):
            if key.startswith("field:"):
                field_name = key.replace("field:", "")
                parts.append(f"- {field_name}: {description}")

        # If no field descriptions in candidate, use Pydantic model defaults
        if len(parts) == 1:
            for field_name, field_info in self.pydantic_model.model_fields.items():
                # Check json_schema_extra["desc"] first (DSPy format)
                desc = None
                if field_info.json_schema_extra and isinstance(field_info.json_schema_extra, dict):
                    desc = field_info.json_schema_extra.get("desc")
                if desc is None:
                    desc = field_info.description or f"The {field_name} field"
                parts.append(f"- {field_name}: {desc}")

        return "\n".join(parts)

    def inject_field_descriptions_into_instruction(
        self,
        candidate: Candidate
    ) -> str:
        """Build a full instruction with field descriptions injected.

        This creates a combined instruction that includes the seed_prompt
        and detailed field extraction guidance.

        Args:
            candidate: The candidate dictionary

        Returns:
            Full instruction string with field guidance
        """
        instruction_parts = [candidate.get("seed_prompt", "Extract data from the document.")]

        # Add field extraction guidance
        field_parts = []
        for key, description in sorted(candidate.items()):
            if key.startswith("field:"):
                field_name = key.replace("field:", "")
                field_parts.append(f"- {field_name}: {description}")

        if field_parts:
            instruction_parts.append("\n\nField Extraction Guidelines:")
            instruction_parts.extend(field_parts)

        return "\n".join(instruction_parts)
