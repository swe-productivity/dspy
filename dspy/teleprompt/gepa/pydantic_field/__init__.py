"""
GEPA Adapter for Pydantic Field Description Evolution.

This module provides a GEPAAdapter that enables independent evolution
of Pydantic field descriptions for data extraction tasks.
"""

from .types import FieldEvaluationResult, Candidate, make_reflective_example
from .candidate import CandidateBuilder
from .signature_factory import SignatureFactory
from .evaluator import FieldEvaluator
from .proposer import FieldDescriptionProposer
from .adapter import PydanticFieldGEPAAdapter
from .teleprompter import PydanticFieldGEPA

__all__ = [
    "FieldEvaluationResult",
    "Candidate",
    "make_reflective_example",
    "CandidateBuilder",
    "SignatureFactory",
    "FieldEvaluator",
    "FieldDescriptionProposer",
    "PydanticFieldGEPAAdapter",
    "PydanticFieldGEPA",
]
