import fitz
import os
import unicodedata
import re

def normalize_math_symbols(text: str):
    normalized_text = ""
    for char in text:

        if "LATIN" not in unicodedata.name(char, ""):

            normalized_text += unicodedata.normalize('NFKD', char + " ")
        else:
            normalized_text += char
    return normalized_text

def fix_whitespace(text: str):
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text.strip()

def clean_sentence(sentence: str) -> str:
    normalized_sentence = normalize_math_symbols(sentence)
    normalized_sentence = fix_whitespace(normalized_sentence)
    normalized_sentence = normalized_sentence.replace("- ", "").replace('\n', " ").replace('- ', '-').strip()

    return normalized_sentence

def get_filename(file_path: str) -> str:
    filename = file_path.split("\\")[-1]
    filename = filename.split('.')[0] + '.txt'
    return filename

def pdf_to_text(pdf_path: str, output_path: str) -> None:
    doc = fitz.open(pdf_path)
    filename = get_filename(pdf_path)
    output_path = os.path.join(output_path, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)  
            text = page.get_text()  
            f.write(f"\n\n--- Page {page_num + 1} ---\n\n")
            f.write(text)
    doc.close()

def text_to_dataset(text_path: str, output_path: str) -> None:
    with open(text_path, 'r', encoding='utf-8') as text_file:
        text_data = text_file.read()
        text_sentences = text_data.split('.')
        text_sentences = [clean_sentence(sentence) for sentence in text_sentences if len(sentence) > 10 and len(re.findall('[0-9]+', sentence)) == 0]
        print(text_sentences)

if __name__ == '__main__':

    pdf_path = "data\\pdfs\\harvard_booklet.pdf"
    out_path = "data\\text_files"
    #pdf_to_text(pdf_path, out_path)
    text_to_dataset("data\\text_files\\harvard_booklet.txt", pdf_path)

