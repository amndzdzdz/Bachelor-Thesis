import json
import pandas as pd
from datasets import load_dataset
from dataset_utils import fix_whitespace, save_to_csv
import random

all_samples = []
i = 2700
with open('data\\ba\\train.json') as f:
    json_file = json.load(f)
    for entry in json_file:
        if i == 0:
            break
        i -= 1
        try:
            pre_text = entry['pre_text'][0] + " " + entry['pre_text'][1]
            post_text = entry['post_text'][-2] + " " + entry['post_text'][-1]
            pre_text = pre_text.replace(" ,", ",").replace(" .", ".").replace("( ", "").replace("&", "and")
            post_text = pre_text.replace(" ,", ",").replace(" .", ".").replace("( ", "").replace("&", "and")
            all_samples.append(pre_text)
            all_samples.append(post_text)
        except:
            continue

dataset = load_dataset("yinzhu-quan/econ_logic_qa")

train_data = dataset['train']['Question']
val_data = dataset['val']['Question']
test_data = dataset['test']['Question']

train_answers = dataset['train']['A']
val_answers = dataset['val']['A']
test_answers = dataset['test']['A']

all_answers = train_answers + val_answers + test_answers
all_sentences = train_data + val_data + test_data

print(len(all_samples))

for i in range(len(all_sentences)):
    final_sentence = []
    sentence = all_sentences[i]
    question = all_answers[i]
    split_sentences = sentence.split(".")
    split_sentences = split_sentences.pop(0)
    final_sentence.append(split_sentences)
    final_sentence.append(question)

    final_sample = '. '.join(final_sentence)
    all_samples.append(final_sample)

print(len(all_samples))


