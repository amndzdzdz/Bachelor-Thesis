import fitz
import os
import unicodedata
import pandas as pd
from random import randrange
import re

def normalize_unicode_chars(text: str):
    normalized_text = unicodedata.normalize('NFKD', text).encode('ascii','ignore')
    return str(normalized_text)

def fix_whitespace(text: str):
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text.strip()

def remove_special_chars(sentence: str) -> str:
    sentence = sentence.replace("- ", "").replace("b'", "").replace("\'", "").strip()
    sentence = re.sub(r'[:()"\/]', '', sentence)
    
    return sentence

def clean_sentence(sentence: str) -> str:
    normalized_sentence = normalize_unicode_chars(sentence)
    normalized_sentence = remove_special_chars(normalized_sentence)
    normalized_sentence = fix_whitespace(normalized_sentence)

    return normalized_sentence

def get_filename(file_path: str) -> str:
    filename = file_path.split("\\")[-1]
    filename = filename.split('.')[0] + '.txt'
    return filename

def save_to_csv(sentences: list, out_path: str) -> None:
    df = pd.DataFrame(sentences, columns=["text"])
    df.to_csv(out_path, index=False)

def pdf_to_text(pdf_path: str, output_path: str) -> None:
    doc = fitz.open(pdf_path)
    filename = get_filename(pdf_path)
    output_path = os.path.join(output_path, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)  
            text = page.get_text()
            f.write(text)
    doc.close()

def text_to_dataset(text_path: str, output_path: str) -> None:
    random_order_sentences = []
    with open(text_path, 'r', encoding='utf-8') as text_file:
        text_data = text_file.read()
        text_data = text_data.replace("\n", " ")
        text_sentences = text_data.split('. ')
        text_sentences = [clean_sentence(sentence) for sentence in text_sentences if len(sentence) > 20]
        for _ in range(0, len(text_sentences)):
            indx1 = randrange(len(text_sentences) - 1)
            indx2 = indx1 - 1
            sentence1 = text_sentences[indx1]
            sentence2 = text_sentences[indx2]
            sentence = sentence1 + ". " + sentence2
            random_order_sentences.append(sentence)

    save_to_csv(random_order_sentences, output_path)

text_to_dataset("data\\text_files\ecbwp1673.txt", "data\\csv_files\\micro_marco.csv")

