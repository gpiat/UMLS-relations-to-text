# !/usr/bin/env python
# coding: utf-8

# dependencies:
# conda install pytorch cudatoolkit=11.3 -c pytorch
# pip install datasets transformers sklearn
from sys import argv

# Usage:
# Running this script with no arguments will use a bert-base-uncased model on the reasoning free training set (i.e. using the long answers as part of the fine tuning)
# The first argument should specify if reasoning-free or reasoning-required is wanted
# Then an optional model name instead of bert-base-uncased

# script largely based on [this](https://github.com/huggingface/transformers/tree/master/examples/text-classification).
reasoning_required = False
if (len(argv) > 1 and argv[1] == "reasoning-required"):
    reasoning_required = True
    print("Running in reasoning-required setting")
elif (argv[1] == "reasoning-free"):
    reasoning_required = False
    print("Running in reasoning-free setting")
if len(argv) > 2:
    model_checkpoint = argv[2]
else:
    model_checkpoint = "bert-base-uncased"

# Optional Arg 3 to specify whether to connect to the internet. This is used
# to download and cache models/datasets online and run training offline due
# to computing cluster constraints.
if len(argv) > 3 and argv[3] == "online":
    online = True
else:
    online = False


## Loading the dataset
from datasets import load_dataset, ClassLabel, Metric, DatasetDict
from evaluate import load

dataset = load_dataset("pubmed_qa", name="pqa_labeled")

# Change the "final_decision" column into a label
features = dataset['train'].features.copy()
features['final_decision'] = ClassLabel(3, ["yes","no", "maybe"])
dataset['train'] = dataset['train'].cast(features)
dataset = dataset.rename_column('final_decision','label')


metric: Metric = load("f1")

## Loading the model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
num_labels = 3
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
# The warning is telling us we are throwing away some weights (the
# `vocab_transform` and `vocab_layer_norm` layers) and randomly initializing
# some other (the `pre_classifier` and `classifier` layers). This is
# absolutely normal in this case, because we are removing the head used to
# pretrain the model on a masked language modeling objective and replacing
# it with a new head for which we don't have pretrained weights, so the
# library warns us we should fine-tune this model before using it for
# inference (which is exactly what we are going to do).

## Preprocessing the data
# This function works with one or several examples. In the case of several
# examples, the tokenizer will return a list of lists for each key:

# This represents the reasoning free case
def preprocess_with_long_answer(examples):
    return tokenizer(
        examples["question"],
        examples["long_answer"],
        truncation=True,
        padding=True,
    )

def preprocess_with_context(examples):
    question = examples['question']
    context = examples['context.contexts']
    
    # Combine context sentences into a single string
    context_strs = [' '.join(context_str) for context_str in context]
    
    # Tokenize inputs with overlap
    return tokenizer(
        question,
        context_strs,
        padding='max_length',
        truncation=True,
        max_length=512,
        stride=256,
        return_tensors='pt'
    )

encoded_reasoning_required = dataset.flatten().map(preprocess_with_context, batched=True)
encoded_reasoning_free = dataset.flatten().map(preprocess_with_long_answer, batched=True)

if (reasoning_required):
    encoded_dataset = encoded_reasoning_required
else:
    encoded_dataset = encoded_reasoning_free

# Create the train-valid-test split
train_valid = encoded_dataset['train'].train_test_split(test_size=.5)

train_test = train_valid['train'].train_test_split(test_size=.1)
train_test_valid_dataset = DatasetDict({
    'train':train_test['train'],
    'test':train_test['test'],
    'validation':train_valid['test']
})
train_test_valid_dataset = train_test_valid_dataset.remove_columns(('context.contexts', 'context.labels', 'context.meshes', 'context.reasoning_required_pred', 'context.reasoning_free_pred', 'long_answer', 'pubid', 'question'))
train_test_valid_dataset

# if we are online, this means this run was on the home node of
# the cluster and was only to download and cache everything
if online:
    quit()

## Fine-tuning the model
from transformers import TrainingArguments, Trainer
import numpy as np
# To instantiate a `Trainer`, we will need to define two more things.
# The most important is the [`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments),
# which is a class that contains all the attributes to customize the
# training. It requires one folder name, which will be used to save
# the checkpoints of the model, and all other arguments are optional:

metric_name = "f1"
i = -2 if model_checkpoint.endswith('/') else -1
model_name = model_checkpoint.split("/")[i]
batch_size = 32

args = TrainingArguments(
    f"{model_name}-finetuned-pqa-l",
    evaluation_strategy = "epoch",
    do_eval=True,
    save_strategy = "epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=False,
    metric_for_best_model="f1",
    push_to_hub=False,
)

# Here we set the evaluation to be done at the end of each epoch, tweak the
# learning rate, use the `batch_size` defined at the top of the script and
# customize the number of epochs for training, as well as the weight decay.
# Since the best model might not be the one at the end of training, we ask the
# `Trainer` to load the best model it saved (according to `metric_name`) at the
# end of training.
# The last thing to define for our `Trainer` is how to compute the metrics from
# the predictions. We need to define a function for this, which will just use
# the `metric` we loaded earlier, the only preprocessing we have to do is to
# take the argmax of our predicted logits

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average='micro')


# Then we just need to pass all of this along with our datasets to the `Trainer`:
trainer = Trainer(
    model,
    args,
    train_dataset=train_test_valid_dataset["train"],
    eval_dataset=train_test_valid_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# We can now finetune our model by just calling the `train` method:
trainer.train()

# We can check with the `evaluate` method that our `Trainer` did
# reload the best model properly (if it was not the last one):
# print(trainer.evaluate(train_test_valid_dataset["test"]))
trainer.evaluate()

# Testing and printing results
print(trainer.predict(test_dataset=train_test_valid_dataset["test"]))
