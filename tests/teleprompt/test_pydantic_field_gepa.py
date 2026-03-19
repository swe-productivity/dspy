"""
Unit tests for PydanticFieldGEPA adapter.

Tests the essential components:
- CandidateBuilder: Building candidates from Pydantic models
- SignatureFactory: Creating signatures with evolved descriptions
- FieldEvaluator: Per-field scoring
- PydanticFieldGEPAAdapter: Core adapter functionality
"""

import pytest
from pydantic import BaseModel, Field
from typing import Optional

import dspy
from dspy.teleprompt.gepa.pydantic_field import (
    CandidateBuilder,
    SignatureFactory,
    FieldEvaluator,
    PydanticFieldGEPAAdapter,
    PydanticFieldGEPA,
)
from dspy.teleprompt.gepa.pydantic_field.types import (
    FieldEvaluationResult,
    FieldTrajectory,
    make_reflective_example,
)
from dspy.teleprompt.gepa.pydantic_field.evaluator import fuzzy_string_scorer, email_scorer


# Test Pydantic models
class SimpleContact(BaseModel):
    """Simple contact for testing."""
    name: str = Field(description="The contact's full name")
    email: str = Field(description="Email address")


class ContactWithOptional(BaseModel):
    """Contact with optional fields."""
    name: str = Field(description="Full name")
    email: str = Field(description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    company: Optional[str] = Field(default=None, description="Company name")


class Address(BaseModel):
    """Nested address model."""
    street: str = Field(description="Street address")
    city: str = Field(description="City name")


class ContactWithAddress(BaseModel):
    """Contact with nested address."""
    name: str = Field(description="Full name")
    address: Address = Field(description="Mailing address")


# ============================================================================
# CandidateBuilder Tests
# ============================================================================

class TestCandidateBuilder:
    """Tests for CandidateBuilder."""

    def test_build_seed_prompt_only(self):
        """Default behavior: only seed_prompt in candidate."""
        builder = CandidateBuilder(SimpleContact, evolvable_fields=None)
        candidate = builder.build_seed_candidate("Extract contact info.")

        assert "seed_prompt" in candidate
        assert candidate["seed_prompt"] == "Extract contact info."
        assert len(candidate) == 1  # Only seed_prompt

    def test_build_all_fields(self):
        """Evolve all fields."""
        builder = CandidateBuilder(SimpleContact, evolvable_fields="all")
        candidate = builder.build_seed_candidate("Extract contact info.")

        assert "seed_prompt" in candidate
        assert "field:name" in candidate
        assert "field:email" in candidate
        assert candidate["field:name"] == "The contact's full name"
        assert candidate["field:email"] == "Email address"

    def test_build_selective_fields(self):
        """Evolve only specific fields."""
        builder = CandidateBuilder(ContactWithOptional, evolvable_fields=["name", "email"])
        candidate = builder.build_seed_candidate("Extract contact.")

        assert "seed_prompt" in candidate
        assert "field:name" in candidate
        assert "field:email" in candidate
        assert "field:phone" not in candidate
        assert "field:company" not in candidate

    def test_nested_model_fields(self):
        """Handle nested Pydantic models."""
        builder = CandidateBuilder(ContactWithAddress, evolvable_fields="all")
        candidate = builder.build_seed_candidate("Extract contact.")

        assert "field:name" in candidate
        assert "field:address" in candidate
        assert "field:address.street" in candidate
        assert "field:address.city" in candidate

    def test_get_field_names(self):
        """Extract field names from candidate."""
        builder = CandidateBuilder(SimpleContact, evolvable_fields="all")
        candidate = builder.build_seed_candidate("Test")

        field_names = builder.get_field_names(candidate)
        assert "name" in field_names
        assert "email" in field_names
        assert "seed_prompt" not in field_names


# ============================================================================
# FieldEvaluator Tests
# ============================================================================

class TestFieldEvaluator:
    """Tests for FieldEvaluator."""

    def test_exact_match(self):
        """Exact match scoring."""
        evaluator = FieldEvaluator()
        gold = SimpleContact(name="John Smith", email="john@example.com")
        pred = SimpleContact(name="John Smith", email="john@example.com")

        result = evaluator.evaluate(gold, pred)

        assert result.overall_score == 1.0
        assert result.field_scores["name"] == 1.0
        assert result.field_scores["email"] == 1.0

    def test_partial_match(self):
        """Partial match scoring."""
        evaluator = FieldEvaluator()
        gold = SimpleContact(name="John Smith", email="john@example.com")
        pred = SimpleContact(name="John", email="john@example.com")

        result = evaluator.evaluate(gold, pred)

        # "John" is contained in "John Smith" -> partial match
        assert result.field_scores["name"] == 0.5
        assert result.field_scores["email"] == 1.0
        assert result.overall_score == 0.75  # Average

    def test_complete_mismatch(self):
        """Complete mismatch scoring."""
        evaluator = FieldEvaluator()
        gold = SimpleContact(name="John Smith", email="john@example.com")
        pred = SimpleContact(name="Jane Doe", email="jane@other.com")

        result = evaluator.evaluate(gold, pred)

        assert result.field_scores["name"] == 0.0
        assert result.field_scores["email"] == 0.0
        assert result.overall_score == 0.0

    def test_none_prediction(self):
        """Handle None prediction."""
        evaluator = FieldEvaluator()
        gold = SimpleContact(name="John Smith", email="john@example.com")

        result = evaluator.evaluate(gold, None)

        assert result.overall_score == 0.0
        assert all(score == 0.0 for score in result.field_scores.values())

    def test_weighted_aggregation(self):
        """Weighted field aggregation."""
        evaluator = FieldEvaluator(
            field_weights={"name": 2.0, "email": 1.0}
        )
        gold = SimpleContact(name="John Smith", email="john@example.com")
        pred = SimpleContact(name="John Smith", email="wrong@email.com")

        result = evaluator.evaluate(gold, pred)

        # name: 1.0 * 2.0 = 2.0, email: 0.0 * 1.0 = 0.0
        # Overall: 2.0 / 3.0 = 0.666...
        assert result.field_scores["name"] == 1.0
        assert result.field_scores["email"] == 0.0
        assert abs(result.overall_score - 2.0/3.0) < 0.01

    def test_custom_scorer(self):
        """Custom field scorer."""
        evaluator = FieldEvaluator(
            field_scorers={"email": email_scorer}
        )
        gold = SimpleContact(name="John", email="john@example.com")
        pred = SimpleContact(name="John", email="jane@example.com")

        result = evaluator.evaluate(gold, pred)

        # Domain matches -> 0.5 score for email
        assert result.field_scores["email"] == 0.5


class TestFuzzyStringScorer:
    """Tests for fuzzy string scorer."""

    def test_exact_match(self):
        score, feedback = fuzzy_string_scorer("hello", "hello", "test")
        assert score == 1.0

    def test_case_insensitive(self):
        score, feedback = fuzzy_string_scorer("Hello", "hello", "test")
        assert score == 1.0

    def test_none_values(self):
        score, _ = fuzzy_string_scorer(None, None, "test")
        assert score == 1.0

        score, _ = fuzzy_string_scorer("hello", None, "test")
        assert score == 0.0


class TestEmailScorer:
    """Tests for email scorer."""

    def test_exact_match(self):
        score, _ = email_scorer("john@example.com", "john@example.com", "email")
        assert score == 1.0

    def test_domain_match(self):
        score, _ = email_scorer("john@example.com", "jane@example.com", "email")
        assert score == 0.5

    def test_no_match(self):
        score, _ = email_scorer("john@example.com", "jane@other.com", "email")
        assert score == 0.0


# ============================================================================
# SignatureFactory Tests
# ============================================================================

class TestSignatureFactory:
    """Tests for SignatureFactory."""

    def test_build_base_signature(self):
        """Build base signature."""
        factory = SignatureFactory(SimpleContact)
        sig = factory.build_base_signature("Extract contact information.")

        assert sig.instructions == "Extract contact information."
        assert "document" in sig.input_fields
        assert "extracted_data" in sig.output_fields

    def test_build_signature_with_candidate(self):
        """Build signature with evolved descriptions."""
        factory = SignatureFactory(SimpleContact)
        candidate = {
            "seed_prompt": "Extract contact details carefully.",
            "field:name": "The full legal name of the person",
            "field:email": "Professional email address",
        }

        sig = factory.build_signature(candidate)

        assert "Extract contact details carefully." in sig.instructions

    def test_inject_field_descriptions(self):
        """Inject field descriptions into instruction."""
        factory = SignatureFactory(SimpleContact)
        candidate = {
            "seed_prompt": "Extract contact info.",
            "field:name": "Full name",
            "field:email": "Email address",
        }

        instruction = factory.inject_field_descriptions_into_instruction(candidate)

        assert "Extract contact info." in instruction
        assert "Field Extraction Guidelines" in instruction
        assert "name: Full name" in instruction
        assert "email: Email address" in instruction


# ============================================================================
# Types Tests
# ============================================================================

class TestTypes:
    """Tests for type definitions."""

    def test_make_reflective_example(self):
        """Create reflective example with correct keys."""
        example = make_reflective_example(
            inputs={"Document": "test doc"},
            generated_outputs={"Extracted Value": "test"},
            feedback="Good extraction",
        )

        assert "Inputs" in example
        assert "Generated Outputs" in example  # Note: space in key
        assert "Feedback" in example
        assert example["Inputs"]["Document"] == "test doc"

    def test_field_evaluation_result(self):
        """FieldEvaluationResult dataclass."""
        result = FieldEvaluationResult(
            overall_score=0.8,
            field_scores={"name": 1.0, "email": 0.6},
            field_feedback={"email": "Partial match"},
        )

        assert result.overall_score == 0.8
        assert result.field_scores["name"] == 1.0
        assert "email" in result.field_feedback

    def test_field_trajectory(self):
        """FieldTrajectory dataclass."""
        traj = FieldTrajectory(
            example={"document": "test"},
            prediction=SimpleContact(name="John", email="j@e.com"),
            score=0.9,
            field_scores={"name": 1.0, "email": 0.8},
        )

        assert traj.score == 0.9
        assert traj.error is None


# ============================================================================
# Integration Tests
# ============================================================================

class TestPydanticFieldGEPAAdapterIntegration:
    """Integration tests for PydanticFieldGEPAAdapter."""

    def test_build_seed_candidate(self):
        """Build seed candidate through adapter."""
        # Create a simple mock module
        class MockExtractor(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predict = dspy.Predict("document -> extracted_data")

            def forward(self, document):
                return self.predict(document=document)

        adapter = PydanticFieldGEPAAdapter(
            pydantic_model=SimpleContact,
            extractor_module=MockExtractor(),
            evolvable_fields="all",
        )

        candidate = adapter.build_seed_candidate("Extract contacts.")

        assert "seed_prompt" in candidate
        assert "field:name" in candidate
        assert "field:email" in candidate

    def test_make_reflective_dataset_structure(self):
        """Verify reflective dataset structure."""
        class MockExtractor(dspy.Module):
            def forward(self, document):
                return dspy.Prediction(extracted_data=None)

        adapter = PydanticFieldGEPAAdapter(
            pydantic_model=SimpleContact,
            extractor_module=MockExtractor(),
            evolvable_fields="all",
        )

        # Create mock eval batch with trajectories
        from dspy.teleprompt.gepa.pydantic_field.adapter import EvaluationBatch

        trajectories = [
            FieldTrajectory(
                example={"document": "Contact John at john@test.com"},
                prediction=SimpleContact(name="John", email="wrong@test.com"),
                score=0.5,
                field_scores={"name": 1.0, "email": 0.0},
                field_feedback={"email": "Mismatch"},
            )
        ]

        eval_batch = EvaluationBatch(
            outputs=[None],
            scores=[0.5],
            trajectories=trajectories,
        )

        candidate = {"seed_prompt": "Test", "field:name": "Name", "field:email": "Email"}
        dataset = adapter.make_reflective_dataset(
            candidate, eval_batch, ["field:email"]
        )

        # Should have examples for field:email (which had errors)
        assert "field:email" in dataset
        assert len(dataset["field:email"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
