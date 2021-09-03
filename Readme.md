# Proxy Indicators for the Quality of Open-domain Dialogues

Code and resources to reproduce all of the results reported in the paper "Proxy Indicators for the Quality of Open-domain Dialogues", Rostislav Nedelchev, Jens Lehmann, Ricardo Usbeck.

In order to run the scripts one needs to setup a virtual environment using conda with all the package dependencies:

```bash
conda env create -f environment.yml
```

We provide the following scripts and notebooks:
- `src/calculate_dialogue_glue_scores.py` - does inference on all the GLUE tasks using the PersonaChat and TopicalChat datasets.
    - to execute run: `python -u calculate_dialogue_glue_scores.py`
- `src/eval_usr_persona_chat.ipynb` - correlation scores, plots, and linear regression models on the PersonaChat dataset.
- `src/eval_usr_topical_chat.ipynb` - correlation scores, plots, and linear regression models on the TopicalChat dataset.