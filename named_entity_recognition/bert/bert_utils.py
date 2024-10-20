import pandas as pd
import evaluate
from transformers import BertTokenizerFast
from datasets import Dataset
from datasets import ClassLabel
from transformers import AutoModelForTokenClassification, Trainer, AutoTokenizer, DataCollatorForTokenClassification
from conllu import parse
from copy import deepcopy
import os

def conll_to_dataframe(annotaions_dir: str) -> 'DataFrame':
    """
    The function iterates over all the annotation csv files, reads each file line by line and creates a 
    big dataset dictionary. It is then further processed into a train dataset and eval dataset. Both in
    pandas DataFrame-format.

    Args:
        - annotations_dir (str): The path to the annotations directory
    
    Output:
        - train_dataset (pd.DataFrame): The train-split of the dataset in a DataFrame
        - eval_dataset (pd.DataFrame): The eval-split of the dataset in a DataFrame
    """
    dataset = []
    for filename in os.listdir(annotaions_dir):
        file_path = os.path.join(annotaions_dir, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            text = []
            labels = []
            for line in file:

                if "DOCSTART" in line:
                    continue

                if len(line.replace("\n", "")) == 0:

                    dataset.append(deepcopy({"tokens": text, "ner_tags": labels}))
                    text = []
                    labels = []
                    continue

                split_line = line.split(" -X- _ ")
                token = split_line[0]
                label = split_line[-1]

                if "O" in label:
                    label = "O"
                else:
                    label = label[2:].replace("\n", "")

                text.append(token)
                labels.append(label)

    dataset = pd.DataFrame(dataset).sample(frac=1, random_state=1502).reset_index(drop=True)
    len_test = int(len(dataset) * 0.2)
    len_train = int(len(dataset) - len_test)

    eval_dataset = dataset[:len_test]
    train_dataset = dataset[len_test:]

    return train_dataset, eval_dataset

def load_dataset(annotations_dir: str, tokenizer: 'tokenizer', classmap, overfit):
    """
    The function takes the path to the csv annotations and creates the train and eval datasets
    in the format huggingface transformers expect

    Args:
        - annotations_dir (str): The path to the annotations directory
    
    Output:
        - train_dataset (arrow_dataset.Dataset): The train-split of the dataset in a arrow_dataset.Dataset
        - eval_dataset (arrow_dataset.Dataset): The eval-split of the dataset in a arrow_dataset.Dataset
    """

    train_dataset, eval_dataset = conll_to_dataframe(annotations_dir)

    if overfit:
        train_dataset, eval_dataset = train_dataset[0:3], eval_dataset[:1]

    train_dataset = Dataset.from_pandas(pd.DataFrame(data=train_dataset))
    eval_dataset = Dataset.from_pandas(pd.DataFrame(data=eval_dataset))

    train_dataset = train_dataset.map(lambda x: tokenizer(x["tokens"], truncation=True, is_split_into_words=True))
    eval_dataset = eval_dataset.map(lambda x: tokenizer(x["tokens"], truncation=True, is_split_into_words=True))

    train_dataset = train_dataset.map(lambda y: {"ner_tags": classmap.str2int(y["ner_tags"])})
    eval_dataset = eval_dataset.map(lambda y: {"ner_tags": classmap.str2int(y["ner_tags"])})

    return train_dataset, eval_dataset

def compute_metrics(p):
    metric = evaluate.load("seqeval")

    predictions, labels = p
    predictions = predictions.argmax(axis=2)
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]}

def tokenize_special_tokens(sentence, label_all_tokens=True):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    #tokenize input sentence to get input_ids (ids of the tokens of the input sentence)
    tokenized_input = tokenizer(sentence['tokens'], truncation=True, is_split_into_words=True)
    labels = []

    #loop over the labels of the input sentence
    for i, label in enumerate(sentence['ner_tags']):
        #get labels of the tokenized sentence
        word_ids = tokenized_input.word_ids(batch_index=i)

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_input['labels'] = labels
    return tokenized_input