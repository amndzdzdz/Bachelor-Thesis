from datasets import load_dataset
from dataset_utils import remove_special_chars, fix_whitespace

dataset = load_dataset("STEM-AI-mtl/Electrical-engineering")
text_data = dataset['train']['output']

all_sentences = []
for sentence in text_data:
    sentences = sentence.split(". ")
    all_sentences.extend(sentences)
print(all_sentences)