"""
GEPA Adapter for Pydantic Field Description Evolution.

This module provides a GEPAAdapter that enables independent evolution
of Pydantic field descriptions for data extraction tasks.
"""

from dspy.teleprompt.gepa.pydantic_field.types import FieldEvaluationResult, Candidate, make_reflective_example
from dspy.teleprompt.gepa.pydantic_field.candidate import CandidateBuilder
from dspy.teleprompt.gepa.pydantic_field.signature_factory import SignatureFactory
from dspy.teleprompt.gepa.pydantic_field.evaluator import FieldEvaluator
from dspy.teleprompt.gepa.pydantic_field.proposer import FieldDescriptionProposer
from dspy.teleprompt.gepa.pydantic_field.adapter import PydanticFieldGEPAAdapter
from dspy.teleprompt.gepa.pydantic_field.teleprompter import PydanticFieldGEPA

__all__ = [
    "Candidate",
    "CandidateBuilder",
    "FieldDescriptionProposer",
    "FieldEvaluationResult",
    "FieldEvaluator",
    "make_reflective_example",
    "PydanticFieldGEPA",
    "PydanticFieldGEPAAdapter",
    "SignatureFactory",
]
