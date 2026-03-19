from dspy.teleprompt.gepa.gepa import GEPA

__all__ = [
    "GEPA",
    "PydanticFieldGEPA",
    "PydanticFieldGEPAAdapter",
    "FieldEvaluator",
    "CandidateBuilder",
    "SignatureFactory",
]


def __getattr__(name):
    """Lazy loading to avoid circular imports."""
    if name in ("PydanticFieldGEPA", "PydanticFieldGEPAAdapter", "FieldEvaluator",
                "CandidateBuilder", "SignatureFactory"):
        from dspy.teleprompt.gepa import pydantic_field
        return getattr(pydantic_field, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
