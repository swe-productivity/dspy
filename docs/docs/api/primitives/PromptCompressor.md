# PromptCompressor

The `PromptCompressor` module is intergrated in DSPy to significantly reduce token usage while using any Large Language Models and thereby reducing latency by locally accelerating context compression safely.

It utilizes techniques from [Microsoft's LLMLingua](https://github.com/microsoft/LLMLingua) project mainly `llmlingua` and `llmlingua-2`. This primitive algorithmically shrinks large unorganized blocks of text while maintaining key entities, facts, and the core objectives defined by your signatures.


### Basic Usage

You can initialize a global default compressor and attach it via `dspy.settings.configure()`. 

Whenever a `Predict` module runs, any fields explicitly marked with `compression=True` in your `dspy.Signature` will be intercepted and scaled down.

```python
import dspy

# 1. Define your strict signature, explicitly marking long fields for compression
class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""

    context: str = dspy.InputField(desc="facts here are assumed to be true", compression=True)
    text: str = dspy.InputField(compression=False)
    faithfulness: bool = dspy.OutputField()
    evidence: list[str] = dspy.OutputField(desc="List of exact quotes from the text supporting evidence for claims")



def main():

    # 2. Setup your PromptCompressor
    compressor = dspy.PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        dynamic_context_compression_ratio=0.55,           # desired compressed size as fraction of original tokens
        preserve_fields=["username, history"],        # list of regex/keys to preserve fully (e.g., user names)
        target_token=None,      # hard limit to compressed result if desired
        device="cpu",                # or "cuda"
        cache=True                   # cache results for identical inputs
    )
    # 3. Attach it globally alongside your LLM
    lm = dspy.LM("ollama_chat/llama3.2:1b", api_base="http://localhost:11434", api_key=api_key)
   
    # 4. Configure settings and pass singature class defined to Predict module 
    dspy.settings.configure(lm=lm, compressor=compressor)
    evaluator = dspy.Predict(CheckCitationFaithfulness)

    long_document_context = (
        "Belgium, officially the Kingdom of Belgium, is a country in Northwestern Europe. "
        "The country is bordered by the Netherlands to the north, Germany to the east, "
        "Luxembourg to the southeast, France to the southwest, and the North Sea to the "
        "northwest. It covers an area of 30,689 square kilometers and has a population "
        "of more than 11.5 million. The capital and largest city is Brussels."
    )
    # call evalautor to get results
    result = evaluator(
        context=long_document_context,
        text="Belgium's capital is Brussels and it neighbors the Netherlands."
    )

    print(result.faithfulness) # True
    print(result.evidence)
    # ["Belgium's capital is Brussels and it neighbors the Netherlands.", 'The country covers an area of 30,689 square kilometers', 'Belgium has a population of more than 11.5 million']

    print(result.compression_metadata)
    # {
    #     'total_original_tokens': 82, 
    #     'total_compressed_tokens': 37, 
    #     'ratio': 0.45121951219512196, 
    #     'model': 'microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank'
    # }

if __name__ == "__main__":
    main()

```