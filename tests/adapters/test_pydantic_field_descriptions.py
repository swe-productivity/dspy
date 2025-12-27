"""Tests for Pydantic field description extraction in adapters."""

import pytest
from pydantic import BaseModel, Field
from typing import Optional, Union, List, Dict

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.utils import (
    is_pydantic_model,
    extract_pydantic_field_descriptions,
    get_field_description_string,
)


class SimpleModel(BaseModel):
    """A simple Pydantic model for testing."""

    field: str


class AnotherModel(BaseModel):
    """Another Pydantic model for testing."""

    value: int


def test_is_pydantic_model_with_direct_model():
    """Test that is_pydantic_model returns True for a direct Pydantic model."""
    assert is_pydantic_model(SimpleModel) is True


def test_is_pydantic_model_with_primitive_types():
    """Test that is_pydantic_model returns False for primitive types."""
    assert is_pydantic_model(str) is False
    assert is_pydantic_model(int) is False
    assert is_pydantic_model(float) is False
    assert is_pydantic_model(bool) is False
    assert is_pydantic_model(list) is False
    assert is_pydantic_model(dict) is False


def test_is_pydantic_model_with_optional_model():
    """Test that is_pydantic_model returns True for Optional[Model]."""
    assert is_pydantic_model(Optional[SimpleModel]) is True


def test_is_pydantic_model_with_optional_primitive():
    """Test that is_pydantic_model returns False for Optional[primitive]."""
    assert is_pydantic_model(Optional[str]) is False
    assert is_pydantic_model(Optional[int]) is False


def test_is_pydantic_model_with_union_containing_model():
    """Test that is_pydantic_model returns True for Union containing a model."""
    assert is_pydantic_model(Union[SimpleModel, str]) is True
    assert is_pydantic_model(Union[str, SimpleModel]) is True
    assert is_pydantic_model(Union[SimpleModel, AnotherModel]) is True


def test_is_pydantic_model_with_union_without_model():
    """Test that is_pydantic_model returns False for Union without models."""
    assert is_pydantic_model(Union[str, int]) is False
    assert is_pydantic_model(Union[str, int, float]) is False


def test_is_pydantic_model_with_list_of_models():
    """Test that is_pydantic_model returns True for List[Model]."""
    assert is_pydantic_model(List[SimpleModel]) is True


def test_is_pydantic_model_with_list_of_primitives():
    """Test that is_pydantic_model returns False for List[primitive]."""
    assert is_pydantic_model(List[str]) is False
    assert is_pydantic_model(List[int]) is False


def test_is_pydantic_model_with_dict_of_models():
    """Test that is_pydantic_model returns True for Dict with model values."""
    assert is_pydantic_model(Dict[str, SimpleModel]) is True


def test_is_pydantic_model_with_dict_of_primitives():
    """Test that is_pydantic_model returns False for Dict with primitive values."""
    assert is_pydantic_model(Dict[str, str]) is False
    assert is_pydantic_model(Dict[str, int]) is False


def test_is_pydantic_model_with_nested_generics():
    """Test that is_pydantic_model handles nested generic types."""
    assert is_pydantic_model(List[Optional[SimpleModel]]) is True
    assert is_pydantic_model(Dict[str, Optional[SimpleModel]]) is True
    assert is_pydantic_model(Optional[List[SimpleModel]]) is True


def test_extract_simple_model_descriptions():
    """Test extracting descriptions from a simple Pydantic model."""

    class SearchQuery(BaseModel):
        term: str = Field(description="search term to extract topics from")
        grade: int = Field(description="grade level of the audience")
        country: str = Field(description="country of the audience")

    result = extract_pydantic_field_descriptions(SearchQuery)

    assert "term" in result
    assert "search term to extract topics from" in result
    assert "grade" in result
    assert "grade level of the audience" in result
    assert "country" in result
    assert "country of the audience" in result


def test_extract_model_without_descriptions():
    """Test extracting from a model without field descriptions."""

    class BasicModel(BaseModel):
        name: str
        age: int

    result = extract_pydantic_field_descriptions(BasicModel)

    assert "name (str)" in result
    assert "age (int)" in result


def test_extract_nested_model_descriptions():
    """Test extracting descriptions from nested Pydantic models."""

    class Address(BaseModel):
        street: str = Field(description="Street name")
        city: str = Field(description="City name")

    class Person(BaseModel):
        name: str = Field(description="Person's name")
        address: Address = Field(description="Person's address")

    result = extract_pydantic_field_descriptions(Person)

    assert "name" in result
    assert "Person's name" in result
    assert "address" in result
    assert "street" in result
    assert "Street name" in result
    assert "city" in result
    assert "City name" in result


def test_extract_with_circular_reference():
    """Test that circular references don't cause infinite loops."""

    class Node(BaseModel):
        value: int = Field(description="Node value")
        next: Optional["Node"] = Field(default=None, description="Next node")

    result = extract_pydantic_field_descriptions(Node, max_depth=3)

    assert "value" in result
    assert "Node value" in result
    assert "next" in result


def test_extract_respects_max_depth():
    """Test that max_depth parameter limits recursion."""

    class Level3(BaseModel):
        field3: str = Field(description="Level 3 field")

    class Level2(BaseModel):
        field2: str = Field(description="Level 2 field")
        nested: Level3

    class Level1(BaseModel):
        field1: str = Field(description="Level 1 field")
        nested: Level2

    result = extract_pydantic_field_descriptions(Level1, max_depth=2)

    assert "field1" in result
    assert "field2" in result
    assert "field3" not in result


def test_extract_with_optional_model():
    """Test extracting from Optional[Model] field."""

    class Inner(BaseModel):
        value: str = Field(description="Inner value")

    class Outer(BaseModel):
        data: Optional[Inner] = Field(description="Optional inner data")

    result = extract_pydantic_field_descriptions(Outer)

    assert "data" in result
    assert "value" in result
    assert "Inner value" in result


def test_extract_with_list_of_models():
    """Test extracting from List[Model] field."""

    class Item(BaseModel):
        name: str = Field(description="Item name")

    class Container(BaseModel):
        items: List[Item] = Field(description="List of items")

    result = extract_pydantic_field_descriptions(Container)

    assert "items" in result
    assert "name" in result
    assert "Item name" in result


def test_get_field_description_string_with_pydantic_model():
    """Test that get_field_description_string includes nested Pydantic field descriptions."""

    class SearchQuery(BaseModel):
        term: str = Field(description="search term")
        grade: int = Field(description="grade level")

    class TestSignature(dspy.Signature):
        """Test signature"""

        query: SearchQuery = dspy.InputField(desc="The search query")
        result: str = dspy.OutputField()

    desc = get_field_description_string(TestSignature.input_fields)

    assert "query" in desc
    assert "SearchQuery" in desc
    assert "term" in desc
    assert "search term" in desc
    assert "grade" in desc
    assert "grade level" in desc


def test_chat_adapter_includes_pydantic_descriptions():
    """Test that ChatAdapter includes nested Pydantic descriptions in formatted prompts."""

    class Query(BaseModel):
        term: str = Field(description="search term")
        grade: int = Field(description="grade level")

    class TopicExtraction(dspy.Signature):
        """Extract topics from a search term"""

        query: Query = dspy.InputField()
        topics: List[str] = dspy.OutputField(desc="list of topics")

    adapter = ChatAdapter()
    field_desc = adapter.format_field_description(TopicExtraction)

    assert "query" in field_desc
    assert "Query" in field_desc
    assert "term" in field_desc
    assert "search term" in field_desc
    assert "grade" in field_desc
    assert "grade level" in field_desc


def test_end_to_end_with_nested_models():
    """Test end-to-end with nested Pydantic models in signature."""

    class Address(BaseModel):
        street: str = Field(description="Street name")
        city: str = Field(description="City name")

    class Person(BaseModel):
        name: str = Field(description="Person's full name")
        age: int = Field(description="Person's age")
        address: Address = Field(description="Person's address")

    class PersonSignature(dspy.Signature):
        """Process person information"""

        person: Person = dspy.InputField()
        summary: str = dspy.OutputField()

    adapter = ChatAdapter()
    field_desc = adapter.format_field_description(PersonSignature)

    assert "person" in field_desc
    assert "name" in field_desc
    assert "Person's full name" in field_desc
    assert "age" in field_desc
    assert "Person's age" in field_desc
    assert "address" in field_desc
    assert "Person's address" in field_desc
    assert "street" in field_desc
    assert "Street name" in field_desc
    assert "city" in field_desc
    assert "City name" in field_desc


def test_original_use_case_from_issue():
    """Test the exact use case from the feature request."""

    class SearchQuery(BaseModel):
        term: str = Field(description="search term to extract topics from")
        grade: int = Field(description="grade level of the audience")
        country: str = Field(description="country of the audience")

    class TopicExtraction(dspy.Signature):
        """Extract topics from a search term"""

        search_query: SearchQuery = dspy.InputField()
        topics: List[str] = dspy.OutputField(desc="list of topics")

    adapter = ChatAdapter()
    field_desc = adapter.format_field_description(TopicExtraction)

    assert "search_query" in field_desc
    assert "term" in field_desc
    assert "search term to extract topics from" in field_desc
    assert "grade" in field_desc
    assert "grade level of the audience" in field_desc
    assert "country" in field_desc
    assert "country of the audience" in field_desc
