#!/usr/bin/env python
# coding: utf-8

from const import tokenizer_checkpoint
from sys import argv

from datasets import arrow_dataset
from datasets import load_metric
from datasets.dataset_dict import DatasetDict
from transformers import AutoModelForTokenClassification as AutoMod4TokClsf
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification


# dependencies:
# conda install pytorch cudatoolkit=11.3 -c pytorch
# pip install datasets transformers sklearn

LABEL_MAP = {
    'O': 0,
    'B-test': 1,
    'I-test': 2,
    'B-problem': 3,
    'I-problem': 4,
    'B-treatment': 5,
    'I-treatment': 6,
}
REV_LABEL_MAP = {v: k for (k, v) in LABEL_MAP.items()}

def iob2_seq_generator(fname: str):
    """ Args:
            fname: file name to read from
        Yield:
            dict:
                {'labels': [label_1, label_2...],
                 'tokens': ['token_1', 'token_2'...]}
                the end of the mention index is inclusive
    """
    def init_dict():
        """ Utility function for initializing the dictionary that
            represents each sequence.
        """
        return {
            'labels': [],
            'tokens': [],
        }
    with open(fname, 'r') as f:
        lines = f.readlines()

    gold_annotations = init_dict()
    # Depending on whether the token starts, continues or is outside of a
    # mention respectively, each line is formatted as:
    # token  <TAB>  B-C0XXXXXX
    # token  <TAB>  I-C0XXXXXX
    # token  <TAB>  O
    for line in lines:
        line = line.strip()
        # if line is blank, start new sequence.
        if line == '':
            new_dict = init_dict()
            # This condition quietly handles cases where the dataset
            # has two blank lines without returning an empty dict.
            if gold_annotations != new_dict:
                yield gold_annotations
                gold_annotations = new_dict
            continue

        token, tag = line.split('\t')
        gold_annotations['tokens'].append(token)
        gold_annotations['labels'].append(LABEL_MAP[tag])

## Preprocessing the data
def preprocess_function(examples):
    """ This function works with one or several examples. In the case of several
        examples, the tokenizer will return a list of lists for each key:
    """
    # kudos to https://gist.github.com/jangedoo/7ac6fdc7deadc87fd1a1124c9d4ccce9
    tokenized_inputs = tokenizer(examples["tokens"],
                                 truncation=True,
                                 is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"labels"]):
        # Map tokens to their respective word
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        # Set the special tokens to -100.
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            # Only label the first token of a given word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def preprocess_predictions(predictions, labels):
    predictions = np.argmax(predictions, axis=2)
    # removing padding elements (-100 label)
    predictions = [[REV_LABEL_MAP[j]
                    for (i, j) in enumerate(predlist)
                    if lablist[i] != -100]
                   for predlist, lablist in zip(predictions, labels)]
    labels = [[REV_LABEL_MAP[lab]
              for lab in lablist
              if lab != -100]
             for lablist in labels]
    return predictions, labels

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions, labels = preprocess_predictions(predictions, labels)
    # results are correclty micro averaged
    return metric.compute(predictions=predictions,
                          references=labels)


def handle_args():
    p = argparse.ArgumentParser(
        description="Arg handler for training BERT on UMLS-derived text")
    p.add_argument('-o', '--output_dir', nargs='?',
                   type=str, default=None,
                   help="Output directory for saved model")
    p.add_argument('-m', '--model', nargs='?',
                   type=str, default="bert-base-uncased",
                   help="Model name or path")
    p.add_argument('-b', '--batch_size', nargs="?", type=int,
                   default=32, help="Batch size for model training")
    p.add_argument('-g', '--gradacc', nargs="?", type=int, default=1,
                   help="Number of batches over which to accumulate gradient")
                   # gradacc 128 * batch_size 32 = 4096
    p.add_argument('-e', '--epochs', nargs="?", type=int,
                   default=1, help="Number of training epochs")
    p.add_argument('--lr', nargs="?", type=float,
                   default=2e-5, help="Learning rate")
    p.add_argument('--wd', nargs="?", type=float,
                   default=0.01, help="weight decay")
    p.add_argument("--online", nargs='?', const=True, default=False,
                    help=("Specify to go online and download datasets etc. "
                          "Will not attempt training."))


    args = p.parse_args()
    if args.output_dir is None:
        i = -2 if args.model.endswith('/') else -1
        model_name = args.model.split("/")[i]
        args.output_dir = f"{model_name}-finetuned-i2b2"
    return args

if __name__ == '__main__':
    args = handle_args()

    ## Loading the dataset
    i2b2path = "/home/users/gpiat/Documents/Datasets/i2b2_2010/IOB2/basic_tokenization/"
    dataset = DatasetDict()
    for split in ['train', 'dev', 'test']:
        data = list(iob2_seq_generator(i2b2path + split + '.tsv'))
        datadict = {k: [instance[k] for instance in data]
                    for k in data[0].keys()}
        dataset[split] = arrow_dataset.Dataset.from_dict(datadict)
        del datadict

    metric = load_metric("seqeval")

    ## Loading the model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint,
                                              use_fast=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    num_labels = 7
    model = AutoMod4TokClsf.from_pretrained(args.model, num_labels=num_labels)
    # The warning is telling us we are throwing away some weights (the
    # `vocab_transform` and `vocab_layer_norm` layers) and randomly
    # initializing some other (the `pre_classifier` and `classifier` layers).
    # This is absolutely normal in this case, because we are removing the head
    # used to pretrain the model on a masked language modeling objective and
    # replacing it with a new head for which we don't have pretrained weights,
    # so the library warns us we should fine-tune this model before using it
    # for inference, which is exactly what we are going to do.


    encoded_dataset = dataset.map(preprocess_function, batched=True)
    #encoded_dataset = dataset

    # if we are online, this means this run was on the home node of the cluster
    # and was only to download and cache everything
    if args.online:
        quit()

    from transformers import TrainingArguments, Trainer
    import numpy as np
    from seqeval.metrics import classification_report

    ## Fine-tuning the model
    # overall_f1 is micro f1 in the seqeval huggingface metric (I checked the code)
    metric_name = "overall_f1"

    t_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.wd,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        push_to_hub=False,
        gradient_accumulation_steps=args.gradacc
    )

    # Here we set the evaluation to be done at the end of each epoch, tweak the
    # learning rate, use the `batch_size` defined at the top of the script and
    # customize the number of epochs for training, as well as the weight decay.
    # Since the best model might not be the one at the end of training, we ask
    # the `Trainer` to load the best model it saved (according to `metric_name`)
    # at the end of training.
    # The last thing to define for our `Trainer` is how to compute the metrics
    # from the predictions. We need to define a function for this, which will
    # just use the `metric` we loaded earlier, the only preprocessing we have
    # to do is to take the argmax of our predicted logits


    # Then we just pass all of this along with our datasets to the `Trainer`:
    trainer = Trainer(
        model,
        t_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["dev"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()

    result = trainer.predict(test_dataset=encoded_dataset["test"])
    print(result.metrics)

    predictions = np.argmax(result.predictions, axis=2)
    predictions = [[REV_LABEL_MAP[j]
                   for (i, j) in enumerate(predlist) if lablist[i] != -100]
                   for predlist, lablist in zip(predictions, result.label_ids)]
    labels = [[REV_LABEL_MAP[lab]
              for lab in lablist if lab != -100]
              for lablist in result.label_ids]
    print(classification_report(predictions, labels, digits=4, mode='strict'))

