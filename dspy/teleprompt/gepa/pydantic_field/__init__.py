"""
GEPA Adapter for Pydantic Field Description Evolution.

This module provides a GEPAAdapter that enables independent evolution
of Pydantic field descriptions for data extraction tasks.
"""

from dspy.teleprompt.gepa.pydantic_field.adapter import PydanticFieldGEPAAdapter
from dspy.teleprompt.gepa.pydantic_field.candidate import CandidateBuilder
from dspy.teleprompt.gepa.pydantic_field.evaluator import FieldEvaluator
from dspy.teleprompt.gepa.pydantic_field.signature_factory import SignatureFactory
from dspy.teleprompt.gepa.pydantic_field.types import Candidate, FieldEvaluationResult

__all__ = [
    "Candidate",
    "CandidateBuilder",
    "FieldEvaluationResult",
    "FieldEvaluator",
    "PydanticFieldGEPAAdapter",
    "SignatureFactory",
]
