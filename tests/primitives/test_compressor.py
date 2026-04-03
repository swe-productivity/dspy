from unittest.mock import MagicMock, patch

import dspy
from dspy.primitives.compressor import PromptCompressor


class DummySignature(dspy.Signature):
    context: str = dspy.InputField(compression=True)
    query: str = dspy.InputField(compression=False)
    answer: str = dspy.OutputField()


@patch("dspy.primitives.compressor.ExternalCompressor")
def test_compressor_initialization_and_v2_flag(mock_compressor):
    """Test to check version flags passed to LLMLingua."""
    PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    )
    mock_compressor.assert_called_once()
    assert mock_compressor.call_args[1]["use_llmlingua2"] is True


@patch("dspy.primitives.compressor.ExternalCompressor")
def test_selective_compression_and_metadata(mock_compressor):
    """Test that ONLY compression=True fields are modified, and math is correct."""
    mock_instance = MagicMock()
    mock_instance.compress_prompt.return_value = {
        "compressed_prompt": "compact text",
        "origin_tokens": 10,
        "compressed_tokens": 2,
    }
    mock_compressor.return_value = mock_instance

    compressor = PromptCompressor(model_name="standard")
    kwargs = {"context": "very long text", "query": "short query"}

    compressed_kwargs, meta = compressor(DummySignature, kwargs)
    # Assert routing logic
    assert compressed_kwargs["context"] == "compact text"  # Modified!
    assert compressed_kwargs["query"] == "short query"  # Untouched!

    # Assert metric math
    assert meta["total_original_tokens"] == 10
    assert meta["total_compressed_tokens"] == 2
    assert meta["ratio"] == 0.2


@patch("dspy.primitives.compressor.ExternalCompressor")
def test_preserve_fields_override(mock_compressor):
    """Test that fields strictly inside preserve_fields bypass compression entirely."""
    mock_instance = MagicMock()
    mock_compressor.return_value = mock_instance

    compressor = PromptCompressor(model_name="std", preserve_fields=["context"])
    kwargs = {"context": "very long text", "query": "short query"}

    compressed_kwargs, meta = compressor(DummySignature, kwargs)

    # Assert the LLMlingua function was NEVER even triggered
    mock_instance.compress_prompt.assert_not_called()
    assert compressed_kwargs["context"] == "very long text"
