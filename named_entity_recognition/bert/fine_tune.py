import numpy as np
import datasets
from bert_utils import tokenize_special_tokens, compute_metrics
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
import argparse

def main(args):
    num_classes, num_epochs, lr, batch_size, weight_decay = args
    
    dataset = datasets.load_dataset("conll2003", trust_remote_code=True)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    tokenized_dataset = dataset.map(tokenize_special_tokens, batched=True)

    model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

    train_args = TrainingArguments(
        "test_ner",
        evaluation_strategy = "epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    metric = datasets.load_metric('seqeval')

    trainer = Trainer(
        model, 
        train_args, 
    train_dataset=tokenized_dataset["train"], 
    eval_dataset=tokenized_dataset["validation"], 
    data_collator=data_collator, 
    tokenizer=tokenizer, 
    compute_metrics=compute_metrics 
    )

    trainer.train()

    model.save_pretrained("ner_model")
    tokenizer.save_pretrained("tokenizer")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", default=9, help="Amount of NER-classes")
    parser.add_argument("--num_epochs", default=1, help="Number of epochs for trainingloop")
    parser.add_argument("--lr", default=2e-5, help="Sets learning rate")
    parser.add_argument("--batch_size", default=16, help="Sets the batch size")
    parser.add_argument("--weight_decay", default=9, help="Sets weight decay")
    args = parser.parse_args()

    main(args)