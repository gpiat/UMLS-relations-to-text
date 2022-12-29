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

# Setting constants
batch_size = 512
output_dirs = {
    'umls': "UMLS_model/",
    'pmc':  "PMC_model/",
    'both': "Hybrid_model/"
}

training_args = TrainingArguments(
    output_dir=None,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    # evaluation_strategy="epoch",
    # learning_rate=2e-5,
    # weight_decay=0.01,
    push_to_hub=False
)

# We will retrain a tokenizer and a bert
# model for each of the following datasets
dataset_keys = ['umls', 'pmc', 'both']

bio_datasets = {}
models = {}
tokenizers = {}

# initializing datasets
bio_datasets['umls'] = load_dataset("text", data_files="results_nl.txt")['train'].shuffle(seed=42)
bio_datasets['pmc']  = load_dataset("text", data_files=[
    "/home/data/dataset/pmc/oa_bulk_bert_512/" + fname
    for fname in ["000.txt", "001.txt", "002.txt"]])['train'].shuffle(seed=43)
bio_datasets['both'] = combine.concatenate_datasets([bio_datasets['umls'], bio_datasets['pmc']])

# We want trained models to be comparable so we use the
# number of training samples of the smallest corpus.
# TODO: IMPROVE THIS TO TAKE INTO ACCOUNT SENTENCE LENGTH
min_size = min([ds.num_rows for ds in bio_datasets.values()])
#TODO: limit dataset length

def tokenize_function(examples):
    """ Code stolen from https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb
    """
    return tokenizer(examples["text"])

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

#2| special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
#2| vocab_trainer = trainers.WordPieceTrainer(vocab_size=25000,
#2|                                           special_tokens=special_tokens)

#1| initializing tokenizers with pretrained BERT tokenizers
bertTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

for key in dataset_keys:
    # initializing models with pretrained BERT
    models[key] = BertForMaskedLM.from_pretrained("bert-base-uncased")

    #1| TODO?
    #1| retraining tokenizers
    #1| tokenizers[key] = bertTokenizer.train_new_from_iterator(
    #1|     batchify(bio_datasets[key]), vocab_size=30522)

    #2| TODO?
    #2| Train tokenizer from scratch
    #2| tokenizers[key] = Tokenizer.train_from_iterator(
    #2|     batchify(bio_datasets[key]), trainer=vocab_trainer)

    tokenizers[key] = bertTokenizer

    # Code stolen from https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb
    tokenized_dataset = bio_datasets[key].map(tokenize_function,
                                              batched=True,
                                              num_proc=4,
                                              remove_columns=["text"])
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4
    )
    # /steal

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizers[key], mlm=True, mlm_probability=0.15
    )

    training_args.output_dir = output_dirs[key]
    model_trainer = Trainer(
        model=models[key],
        args=training_args,
        data_collator=data_collator,
        # train_dataset=bio_datasets[key],
        train_dataset=lm_dataset["train"],
        # eval_dataset=lm_dataset["validation"],
    )

    print(f'Start training {key} model')
    # Start training
    model_trainer.train()

    # Save
    print(f'Finished training {key} model, saving...')
    model_trainer.save_model(output_dirs[key])
    print('Model saved.')
