import argparse

from datasets import combine
from datasets import load_dataset
#2| from tokenizers import trainers
from transformers import AutoTokenizer
from transformers import BertForMaskedLM
# TODO: delete if AutoTokenizer works
# from transformers import BertTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments


models = {}
tokenizers = {}


def group_texts(examples):
    """ Code stolen from https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb
    """
    block_size = 128
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def batchify(dataset):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]


def main(args):
    #2| special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    #2| vocab_trainer = trainers.WordPieceTrainer(vocab_size=25000,
    #2|                                           special_tokens=special_tokens)

    #1| initializing tokenizers with pretrained BERT tokenizers
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # initializing models with pretrained BERT
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    #1| TODO?
    #1| retraining tokenizers
    #1| tokenizer = bertTokenizer.train_new_from_iterator(
    #1|     batchify(bio_datasets[key]), vocab_size=30522)

    #2| TODO?
    #2| Train tokenizer from scratch
    #2| tokenizer = Tokenizer.train_from_iterator(
    #2|     batchify(bio_datasets[key]), trainer=vocab_trainer)

    # Code stolen from https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb
    # except I replaced the tokenization function with a lambda
    tokenized_dataset = bio_datasets[key].map(
        lambda examples: tokenizer(examples["text"]), 
        batched=True,
        num_proc=args.threads, 
        remove_columns=["text"])
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.threads
    )
    # /steal

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args.output_dir = output_dirs[key]
    model_trainer = Trainer(
        model=models[key],
        args=training_args,
        data_collator=data_collator,
        # train_dataset=bio_datasets[key],
        train_dataset=lm_dataset,#["train"],
        # eval_dataset=lm_dataset["validation"],
    )

    print(f'Start training {args.dataset} model')
    # Start training
    model_trainer.train()

    # Save
    print(f'Finished training {args.dataset} model, saving...')
    model_trainer.save_model(args.output_dir)
    print('Model saved.')

def handle_args():
    p = argparse.ArgumentParser(
        description="Arg handler for training BERT on UMLS-derived text")
    # p.add_argument('-o', '--option', nargs="*/+/?/N", type=type,
    #                default=None, help="Help text")
    p.add_argument('-b', '--batch_size', nargs="?", type=int,
                   default=512, help="Batch size for model training")
    p.add_argument('-o', '--output_dir', nargs=1, type=str,
                   help="Output directory for saved model")
        # output_dirs = {
        #     'umls': "UMLS_model/",
        #     'pmc':  "PMC_model/",
        #     'both': "Hybrid_model/"
        # }
    p.add_argument('-e', '--epochs', nargs="?", type=int,
                   default=1, help="Number of training epochs")
    p.add_argument('--lr', nargs="?", type=float,
                   default=2e-5, help="Learning rate")
    p.add_argument('--wd', nargs="?", type=float,
                   default=0.01, help="weight decay")
    p.add_argument('-d', '--dataset', nargs=1, type=str,
                   help="Dataset name: 'umls', 'pmc' or 'both'")
    p.add_argument('-t', '--threads', nargs='?', type=int,
                   default=8, help="number of threads for "
                   "dataset parsing parallelization.")


    # We want trained models to be comparable so we use the
    # number of training samples of the smallest corpus.
    # TODO: IMPROVE THIS TO TAKE INTO ACCOUNT SENTENCE LENGTH
    min_size = min([ds.num_rows for ds in bio_datasets.values()])
    #TODO: limit dataset length


    args = p.parse_args()
    return args

if __name__ == '__main__':
    args = handle_args()
    # Setting constants

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_gpu_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        # evaluation_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=args.wd,
        push_to_hub=False
    )

    # initializing datasets
    bio_datasets = {}
    if args.dataset != 'pmc':
        bio_datasets['umls'] = load_dataset(
            "text",
            data_files="results_nl.txt")['train'].shuffle(seed=42)
    if args.dataset != 'umls':
        bio_datasets['pmc']  = load_dataset("text", data_files=[
            "/home/data/dataset/pmc/oa_bulk_bert_512/" + fname
            for fname in ["000.txt", "001.txt", "002.txt"]
            ])['train'].shuffle(seed=43)
    bio_datasets['both'] = combine.concatenate_datasets([bio_datasets['umls'], bio_datasets['pmc']])
    
    dataset = bio_datasets[args.dataset]

    main(args)