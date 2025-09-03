from pathlib import Path
from openai import OpenAI

def should_filter(text, prompt, model) -> str:
    narrow_prompt = Path("classifier/"+prompt).read_text(encoding="utf-8")
    client = OpenAI()
    full_prompt = f"{narrow_prompt}\n {text[:6000]}\n\n\n <end text> --- Remember to answer with a single word (yes/no)."
    try:
        response = client.responses.create(model=model, input=full_prompt)
    except:
        return "error"
    output_text = response.output_text
    for x in ["yes", "no"]:
        if x in output_text.lower():
            return x
    return "error"