import json
import pandas as pd
from datasets import load_dataset
from dataset_utils import fix_whitespace, save_to_csv
import random

def remove_special_chars(string: str) -> str:
    string = string.replace("\n", " ")
    string = string.replace("-", " ")
    string = string.replace("(", " ")
    string = string.replace(")", " ")
    string = string.replace("\\", " ")
    string = string.replace("/", " ")
    string = string.replace("#", " ")
    string = string.replace("`", " ")
    return string

all_sentences = []
with open('data\cs\cs.json') as f:
    json_file = json.load(f)
    lists = json_file['intents']
    for entry in lists:
        response = entry['responses'][0]
        sentences = response.split('.')
        all_sentences.extend([sentence.strip() + "." for sentence in sentences if len(sentence) > 10])

data = pd.read_csv("data\cs\data_science_concepts.csv")
responses = data['Answer'].tolist()

for response in responses:
    sentences = response.split(".")
    all_sentences.extend([sentence.strip() + "." for sentence in sentences if len(sentence) > 10])

dataset = load_dataset("open-phi/programming_books_llama")
text_data = dataset['train']['markdown']

for _ in range(5000):
    random_sample = random.choice(text_data)
    cleaned_sample = remove_special_chars(random_sample)
    cleaned_sample = fix_whitespace(cleaned_sample)
    sentences = [sentence for sentence in cleaned_sample.split(".") if len(sentence) > 10]
    sentence = random.choice(sentences).strip() + "."
    all_sentences.extend([sentence])

save_to_csv(all_sentences)