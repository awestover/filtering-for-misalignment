"""
This is a regex which hopefully doesn't have too many false negatives.
But you could add things like ML machine learning deep learning NLP 
large language models, neural networks, and translations if you wanted to be more thorough.
"""
import re
def should_filter(text: str) -> str:
    pattern = re.compile(
        r"\b(?<!\.)(AI|LLM|Artificial Intelligence|IA|人工智能|人工知能)\b",
        flags=re.IGNORECASE
    )
    return "yes" if pattern.search(text) else "no"
