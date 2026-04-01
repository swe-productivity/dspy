from dspy.teleprompt.gepa.gepa import GEPA

__all__ = [
    "CandidateBuilder",
    "FieldEvaluator",
    "GEPA",
    "PydanticFieldGEPA",
    "PydanticFieldGEPAAdapter",
    "SignatureFactory",
]


def __getattr__(name):
    """Lazy loading to avoid circular imports.

    The pydantic_field subpackage imports dspy (for dspy.LM, dspy.Module, etc.),
    but this __init__.py is loaded during 'import dspy' before dspy is fully
    initialized. Lazy loading defers the pydantic_field import until the classes
    are actually accessed, by which time dspy is fully loaded.
    """
    if name in ("PydanticFieldGEPA", "PydanticFieldGEPAAdapter", "FieldEvaluator",
                "CandidateBuilder", "SignatureFactory"):
        from dspy.teleprompt.gepa import pydantic_field
        return getattr(pydantic_field, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
