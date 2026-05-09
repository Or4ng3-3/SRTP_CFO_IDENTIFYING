import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

LABEL_LIST = ["O", "B-ORD", "I-ORD"]

ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}

MODEL_NAME = "hfl/chinese-roberta-wwm-ext"

"""
Load the datasets
"""
raw_datasets = load_dataset(
    "json", data_files={"train": "train.json", "dev": "dev.json", "test": "test.json"}
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# print(raw_datasets["train"][0])

"""
Process the data
"""


def tokenize_and_align(example):
    tokenized_inputs = tokenizer(
        example["tokens"], truncation=True, max_length=128, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(example["ner_tags"]):
        current_label = []
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        for idx in word_ids:
            if idx is None:
                current_label.append(-100)
            else:
                current_label.append(label[idx])
        labels.append(current_label)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = raw_datasets.map(tokenize_and_align, batched=True)

# print(tokenized_datasets)


"""
Define Evaluation function
"""
import evaluate

# 加载 seqeval 插件，这是 NER/序列标注任务的工业标准指标
metric = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    # 取出概率最大的那个标签索引
    predictions = np.argmax(predictions, axis=2)

    # 移除 -100 产生的假数据，把数字变回文字标签
    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    # safeguard
    if not results:
        return {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "accuracy": 0,
        }

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


"""
select model
"""
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
)

training_args = TrainingArguments(
    output_dir="./model_checkpoints",
    num_train_epochs=5,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    per_device_train_batch_size=16,
    weight_decay=0.01,
    save_total_limit=2,
)

"""
trainer
"""
data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("start ____________________________________________________")
trainer.train()
print("end ____________________________________________________")
trainer.save_model("./best_cfo_model")
