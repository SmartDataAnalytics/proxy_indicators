import json

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SINGLE_TASKS_MAP = {
    "cola": "textattack/bert-base-uncased-CoLA",
    "sst": "textattack/bert-base-uncased-SST-2",
}

PAIR_TASKS_MAP = {
    "mrpc": "textattack/bert-base-uncased-MRPC",
    "qqp": "textattack/bert-base-uncased-QQP",
    "stsb": "textattack/bert-base-uncased-STS-B",
    "qnli": "textattack/bert-base-uncased-QNLI",
    "rte": "textattack/bert-base-uncased-RTE",
}

with open("./data/usr_percona_chat.json") as fin:
    percona_chat = json.load(fin)

with open("./data/usr_topical_chat.json") as fin:
    topical_chat = json.load(fin)

DATASETS = [percona_chat, topical_chat]

# SINGLE SENTENCE TASKS
for task_name, model_name in tqdm(
    SINGLE_TASKS_MAP.items(), desc="single tasks", total=len(SINGLE_TASKS_MAP)
):
    current_task_info = f"Evaluating {task_name}"
    print(current_task_info)
    tqdm.write(current_task_info)

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    with torch.no_grad():
        for dataset in DATASETS:
            for dialogue in tqdm(dataset, desc="dialogues"):
                responses = dialogue["responses"]

                for response in responses:
                    tokenized_input = tokenizer(
                        response["response"],
                        truncation=True,
                        return_tensors="pt",
                    )

                    model_output = model(**tokenized_input)["logits"].squeeze()

                    model_output = torch.softmax(model_output, dim=0).tolist()[-1]

                    response[task_name] = model_output


# PAIR SENTENCE TASKS
for task_name, model_name in tqdm(
    PAIR_TASKS_MAP.items(), desc="pair tasks", total=len(PAIR_TASKS_MAP)
):
    current_task_info = f"Evaluating {task_name}"
    print(current_task_info)
    tqdm.write(current_task_info)

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    with torch.no_grad():
        for dataset in DATASETS:
            for dialogue in tqdm(dataset, desc="dialogues"):
                responses = dialogue["responses"]

                for response in responses:
                    tokenized_input = tokenizer(
                        *(dialogue["context"], response["response"]),
                        truncation=True,
                        return_tensors="pt",
                    )
                    model_output = model(**tokenized_input)["logits"].squeeze()

                    if model_output.shape.numel() > 1:
                        model_output = torch.softmax(model_output, dim=0)

                    model_output = model_output.tolist()

                    if isinstance(model_output, list):
                        model_output = model_output[-1]

                    response[task_name] = model_output

                    # calculate fact relatedness
                    tokenized_input = tokenizer(
                        *(dialogue["fact"], response["response"]),
                        truncation=True,
                        return_tensors="pt",
                    )
                    model_output = model(**tokenized_input)["logits"].squeeze()

                    if model_output.shape.numel() > 1:
                        model_output = torch.softmax(model_output, dim=0)

                    model_output = model_output.tolist()

                    if isinstance(model_output, list):
                        model_output = model_output[-1]

                    response[f"fact_{task_name}"] = model_output


with open("./data/eval_usr_percona_chat.json", "wt") as fout:
    json.dump(obj=percona_chat, fp=fout)

with open("./data/eval_usr_topical_chat.json", "wt") as fout:
    json.dump(obj=topical_chat, fp=fout)
