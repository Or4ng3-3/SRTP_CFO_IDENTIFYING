from collections import defaultdict, Counter
import json

# from matplotlib import pyplot as plt
import numpy as np
import torch


def print_encoding(model_inputs, indent=4):
    indent_str = " " * indent
    print("{")
    for k, v in model_inputs.items():
        print(indent_str + k + ":")
        print(indent_str + indent_str + str(v))
    print("}")


id2label = {0: "O", 1: "B-ORD", 2: "I-ORD"}
label2id = {"O": 0, "B-ORD": 1, "I-ORD": 2}

from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("./best_cfo_model")
model = AutoModelForTokenClassification.from_pretrained("./best_cfo_model")

input = input("请输入待测中文句子Please put in your Chinese sentence:")


model_inputs = tokenizer(input, return_tensors="pt")
with torch.no_grad():
    outputs = model(**model_inputs)

predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()

tokens = tokenizer.convert_ids_to_tokens(model_inputs["input_ids"][0])
labels = [id2label[i] for i in predictions]

for token, label in zip(tokens, labels):
    print(f"{token} -> {label}")
