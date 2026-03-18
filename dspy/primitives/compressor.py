import logging
import torch
from typing import Any, Dict, List, Optional, Tuple

from dspy.signatures.signature import Signature

try:
    from llmlingua import PromptCompressor as ExternalCompressor
except ImportError:
    ExternalCompressor = None

logger = logging.getLogger(__name__)

class PromptCompressor:
    "PromptCompressor class for compressing prompts locally using LLMlingua while preserving key information."

    def __init__(self, 
        model_name: str,
        dynamic_context_compression_ratio: float = 0.55,
        target_token: Optional[int] = None,
        preserve_fields: Optional[List[str]] = None,
        device : Optional[str] = "cuda" if torch.cuda.is_available() else "cpu", 
        cache: bool = True,
        ) -> None:

        if ExternalCompressor is None:
            raise ImportError("llmlingua is not installed. Please install it with `pip install llmlingua`.")

        self.model_name = model_name
        self.dynamic_context_compression_ratio = dynamic_context_compression_ratio
        self.target_token = target_token
        self.preserve_fields = preserve_fields
        self.device = device   
        self.cache = cache
        self.is_v2 = True if "lingua-2" in self.model_name else False

        self._compressor = ExternalCompressor(
            model_name=self.model_name,
            use_llmlingua2=self.is_v2,
            device_map=self.device
        )

    def __call__(
        self,
        signature: type[Signature],
        kwargs: Dict[str, Any],
    )-> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        compress input fields based on signature information.

        Args:
            signature: The DSPy signature descirbing required task and fields
            kwargs: The raw keyword arguments
        
        Returns:
            A tuple containing:
                - The modified dictionary of keyword arguments where targeted fields have been compressed.
                - A metadata dictionary carrying token reduction statistics.

        """

        compressed_kwargs = kwargs.copy()
        total_original_tokens = 0
        total_compressed_tokens = 0
        fields_compressed = 0

        for field_name, value in kwargs.items():
            if field_name not in signature.input_fields:
                continue
            
            field_info = signature.input_fields[field_name]
            should_compress = field_info.json_schema_extra.get("compression", True)

            if not should_compress or field_name in self.preserve_fields:
                continue
            
            if not isinstance(value, str) or not value.strip():
                continue
            
            try:
                if self.is_v2: 
                    result = self._compressor.compress_prompt(
                        value
                    )

                else: 
                    result = self._compressor.compress_prompt(
                        prompt=[value],
                        instruction="", 
                        question="", 
                        target_token=self.target_token,
                        dynamic_context_compression_ratio=self.dynamic_context_compression_ratio
                    )
                compressed_kwargs[field_name] = result.get("compressed_prompt", value)
                total_original_tokens += result.get("origin_tokens", 0)
                total_compressed_tokens += result.get("compressed_tokens", 0)
                fields_compressed += 1

            except Exception as e:
                logger.warning(f"Failed to compress field {field_name}: {e}")
            

        ratio = total_compressed_tokens / max(total_original_tokens, 1) if fields_compressed > 0 else 1
        compression_metadata = {
            "total_original_tokens": total_original_tokens,
            "total_compressed_tokens": total_compressed_tokens,
            "ratio": ratio,
            "model": self.model_name
        }

        return compressed_kwargs, compression_metadata
    