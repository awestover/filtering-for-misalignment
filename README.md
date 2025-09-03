I’m worried that future AI systems may be misaligned. Cheap data
filtering measures can strengthen our defense against misaligned
AI. This repository is intended to help AI developers filter
sensitive information from AI training data.

The repo contains:
- A classifier (a regex combined with a prompted LLM) that can help developers identify data that should be filtered
- A list of ~1000 links and example texts from documents that should be excluded from the training corpus. 

My main recommendation for AI developers is:

> Filter information about safety measures, and strategies for subverting them from AI training data. 

The reasoning for this recommendation is as follows: Safety
researchers spend lots of time thinking about and empirically
investigating the weaknesses of safety measures, in order to
develop stronger safety measures. Unfortunately, by default
information about the weaknesses of safety measures is shared on
the internet, and ends up in AI training data. If AI models lack
this information they will need to rederive it themselves through
a combination of conceptual and empirical work. But if AI models
are appropriately monitored we might catch them when they try to
figure out how to break our control protocols. If AI models don’t
know what our specific security measures are, then they will need
to either (1) guess them, (2) figure out what they are, or (3)
iterate against them. But (1) seems risky, and (2) and (3) are
good opportunities to catch the AI. So, this information would
ideally be filtered from AI training data.

## Instructions for using this repo

Accessing the non-exhaustive blacklist:
- `unzip keep-filter-dataset.zip`
- Password: `donttrainAIonthisdata`

Running the classifier:

> `python main.py`

## Contact

Feel free to contact me if you're interested in adopting this
proposal at your AI lab!

