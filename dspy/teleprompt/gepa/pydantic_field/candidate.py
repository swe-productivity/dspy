"""
Candidate builder for Pydantic Field GEPA Adapter.

Builds initial candidates from Pydantic models by extracting
field descriptions and structuring them for GEPA evolution.
"""

from typing import get_args, get_origin
from pydantic import BaseModel

from dspy.teleprompt.gepa.pydantic_field.types import Candidate


class CandidateBuilder:
    """Builds seed candidates from Pydantic models.

    The candidate dictionary maps component names to evolvable text:
    - "seed_prompt": The base instruction for extraction
    - "field:{field_name}": Description for each field
    - "field:{parent}.{child}": Dotted notation for nested fields
    """

    def __init__(
        self,
        pydantic_model: type[BaseModel],
        evolvable_fields: list[str] | str | None = None,
    ):
        """Initialize the candidate builder.

        Args:
            pydantic_model: The Pydantic model defining the extraction schema
            evolvable_fields: List of field names to evolve, "all" for all fields,
                            or None for only seed_prompt (default behavior)
        """
        self.pydantic_model = pydantic_model
        self.evolvable_fields = evolvable_fields

    def build_seed_candidate(self, base_instruction: str) -> Candidate:
        """Build the initial candidate from the Pydantic model.

        Args:
            base_instruction: The seed prompt / base instruction for extraction

        Returns:
            Candidate dictionary with seed_prompt and optional field descriptions
        """
        candidate: Candidate = {"seed_prompt": base_instruction}

        if self.evolvable_fields is None:
            # Default: only seed_prompt, no field evolution
            return candidate

        # Get field descriptions from the Pydantic model
        field_descriptions = self._extract_field_descriptions(
            self.pydantic_model,
            prefix=""
        )

        # Filter to evolvable fields
        if self.evolvable_fields == "all":
            candidate.update(field_descriptions)
        else:
            for field_name in self.evolvable_fields:
                key = f"field:{field_name}"
                if key in field_descriptions:
                    candidate[key] = field_descriptions[key]

        return candidate

    def _extract_field_descriptions(
        self,
        model: type[BaseModel],
        prefix: str
    ) -> dict[str, str]:
        """Extract field descriptions from a Pydantic model.

        Uses json_schema_extra["desc"] if available (DSPy format),
        otherwise falls back to field.description.

        Args:
            model: The Pydantic model to extract from
            prefix: Prefix for nested field names (e.g., "address.")

        Returns:
            Dictionary mapping "field:{name}" to descriptions
        """
        descriptions: dict[str, str] = {}

        for field_name, field_info in model.model_fields.items():
            full_name = f"{prefix}{field_name}" if prefix else field_name
            key = f"field:{full_name}"

            # Get description: prefer json_schema_extra["desc"] (DSPy format)
            # then fall back to description
            description = None
            if field_info.json_schema_extra:
                if isinstance(field_info.json_schema_extra, dict):
                    description = field_info.json_schema_extra.get("desc")

            if description is None:
                description = field_info.description or f"The {field_name} field"

            descriptions[key] = description

            # Handle nested Pydantic models
            annotation = field_info.annotation

            # Unwrap Optional types (Union with None)
            origin = get_origin(annotation)
            if origin is not None:
                args = get_args(annotation)
                # Filter out NoneType
                non_none_args = [a for a in args if a is not type(None)]
                if len(non_none_args) == 1:
                    annotation = non_none_args[0]

            # Check if it's a nested Pydantic model
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                nested_descriptions = self._extract_field_descriptions(
                    annotation,
                    prefix=f"{full_name}."
                )
                descriptions.update(nested_descriptions)

        return descriptions

    def get_field_names(self, candidate: Candidate) -> list[str]:
        """Get list of field names from a candidate.

        Args:
            candidate: The candidate dictionary

        Returns:
            List of field names (without "field:" prefix)
        """
        return [
            key.replace("field:", "")
            for key in candidate.keys()
            if key.startswith("field:")
        ]

    def get_component_names(self, candidate: Candidate) -> list[str]:
        """Get list of all component names from a candidate.

        Args:
            candidate: The candidate dictionary

        Returns:
            List of component names (including "seed_prompt" and "field:*")
        """
        return list(candidate.keys())
