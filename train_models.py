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
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizers[key], mlm=True, mlm_probability=0.15
    )

    training_args.output_dir = output_dirs[key]
    model_trainer = Trainer(
        model=models[key],
        args=training_args,
        data_collator=data_collator,
        train_dataset=bio_datasets[key],
    )

    print(f'Start training {key} model')
    # Start training
    trainer.train()

    # Save
    print(f'Finished training {key} model, saving...')
    trainer.save_model(output_dirs[key])
    print('Model saved.')
