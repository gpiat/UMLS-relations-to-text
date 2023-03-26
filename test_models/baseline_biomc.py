from datasets import load_dataset, ClassLabel, Metric
from evaluate import load
from sys import argv

if len(argv) > 1:
    model_checkpoint = argv[1]
else:
    model_checkpoint = "bert-base-uncased"

# Optional Arg 3 to specify whether to connect to the internet. This is used
# to download and cache models/datasets online and run training offline due
# to computing cluster constraints.
if len(argv) > 2 and argv[2] == "online":
    online = True
else:
    online = False

model_checkpoint = "bert-base-uncased"
batch_size = 16

dataset = load_dataset("medmcqa")
filtered_dataset = dataset.filter(lambda x: x['choice_type'] == 'single')
features = filtered_dataset['train'].features.copy()
features['cop'] = ClassLabel(4, ['1','2','3','4'])
filtered_dataset = filtered_dataset.cast(features=features)
filtered_dataset = filtered_dataset.rename_column('cop', 'label')

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

answer_names = ["opa", "opb", "opc", "opd"]

def preprocess_function(examples):
    # Repeat each question four times to go with the four possible answers.
    questions = [[question] * 4 for question in examples["question"]]
    # Grab all answers possible for each question.
    answers = [[f"{examples[end][i]}" for end in answer_names] for i in range(len(examples['question']))]

    # Flatten everything
    questions = sum(questions, [])
    answers = sum(answers, [])
    
    # Tokenize
    tokenized_examples = tokenizer(questions, answers, truncation=True)
    # Un-flatten
    return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

tokenized_dataset = filtered_dataset.map(preprocess_function, batched=True)

from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, IntervalStrategy

model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)

model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned-swag",
    evaluation_strategy = IntervalStrategy.EPOCH,
    do_eval=True,
    save_strategy = IntervalStrategy.EPOCH,
    logging_strategy=IntervalStrategy.EPOCH,
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)

# We need to tell our Trainer how to form batches from the pre-processed inputs.
# We haven't done any padding yet because we will pad each batch to the maximum length inside the batch 
# (instead of doing so with the maximum length of the whole dataset). 
# This will be the job of the data collator. 
# A data collator takes a list of examples and converts them to a batch (by, in our case, applying padding).
#  Since there is no data collator in the library that works on our specific problem, we will write one, adapted from the DataCollatorWithPadding:

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
    

import numpy as np

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

print(trainer.predict(test_dataset=tokenized_dataset["test"]))
