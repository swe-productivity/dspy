import logging
from typing import Any

from dspy.signatures.signature import Signature

try:
    from llmlingua import PromptCompressor as ExternalCompressor
except ImportError:
    ExternalCompressor = None

logger = logging.getLogger(__name__)


class PromptCompressor:
    """
    A module to systematically compress prompts to recuce token consumption and latency while preserving key information.
    """

    def __init__(
        self,
        model_name: str,
        dynamic_context_compression_ratio: float = 0.0,
        target_token: float | None = None,
        preserve_fields: list[str] | None = None,
        device: str | None = "cpu",
        cache: bool = True,
    ) -> None:
        """
        Initalized PromptCompressor Class.

        Args:
            model_name (str, optional): The name of the model to be loaded. Default is "NousResearch/Llama-2-7b-hf".
            dynamic_context_compression_ratio (float): Ratio for dynamically adjusting context compression. Default is 0.0.
            target_token (Optional[float]): The global maximum number of tokens to be achieved. Default is -1, indicating no
                specific target. The actual number of tokens after compression should generally be less than the specified target_token,
                but there can be fluctuations due to differences in tokenizers. If specified, compression will be based on the target_token as
                the sole criterion, overriding the ``rate``. ``target_token``, is applicable only within the context-level
                filter and the sentence-level filter. In the token-level filter, the rate for each segment overrides the global target token.
                However, for segments where no specific rate is defined, the global rate calculated from global target token serves
                as the default value. The final target token of the entire text is a composite result of multiple compression rates
                applied across different sections.
            preserve_fields (Optional[List[str]]): A specific list of field names corresponding to signature elements that should be
                bypassed during compression runs and left entirely unmodified. Defaults to "None".
            device (str, optional): The device to load the model onto, supported "cuda", "cpu", "mps". Default is "cpu".
            cache:
        """

        if ExternalCompressor is None:
            raise ImportError("llmlingua is not installed. Please install it with `pip install llmlingua`.")

        if device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.dynamic_context_compression_ratio = dynamic_context_compression_ratio
        self.target_token = target_token
        self.preserve_fields = preserve_fields or []
        self.cache = cache
        self.is_v2 = True if "lingua-2" in self.model_name else False

        self._compressor = ExternalCompressor(
            model_name=self.model_name, use_llmlingua2=self.is_v2, device_map=self.device
        )

    def __call__(
        self,
        signature: type[Signature],
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
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
                    result = self._compressor.compress_prompt(value)

                else:
                    lingua_target = self.target_token if self.target_token is not None else -1.0
                    result = self._compressor.compress_prompt(
                        prompt=[value],
                        target_token=lingua_target,
                        dynamic_context_compression_ratio=self.dynamic_context_compression_ratio,
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
            "model": self.model_name,
        }

        return compressed_kwargs, compression_metadata
