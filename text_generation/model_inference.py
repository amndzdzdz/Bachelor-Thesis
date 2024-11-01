from llama import Llama32_1B, Llama32_3B
import pandas as pd
from tqdm import tqdm

DATA_PATH = r"C:\Users\trewe\Desktop\UniTingz\Bachelorarbeit\NER\Bachelor-Thesis\data\text_generation\dataset\dataset_with_sentences.xlsx"
PROMPTS_PATH = r"C:\Users\trewe\Desktop\UniTingz\Bachelorarbeit\NER\Bachelor-Thesis\data\text_generation\prompts\prompts.xlsx"

data = pd.read_excel(DATA_PATH)
prompts = pd.read_excel(PROMPTS_PATH)

models = [Llama32_1B(), Llama32_3B()]

for model in models:

    model.initialize()

    for row_index, row in tqdm(data.iterrows(), "iterating over data..."):

        term, _, sentence = row['keyword'], row['description'], row['sentence']

        prompt_no_sentence = prompts['prompts'][0]
        prompt_w_sentence = prompts['prompts'][1]

        prediction_no_sentence, _ = model.predict(prompt_no_sentence, term)
        prediction_w_sentence, _ = model.predict(prompt_w_sentence, term, sentence)

        try:
            data.insert(len(data.columns), "prediction no sentence", ['nan' for i in range(len(data))])
            data.insert(len(data.columns), "prediction with sentence", ['nan' for i in range(len(data))])
        except:
            print("Column already exists")

        data.loc[row_index, "prediction no sentence"] = prediction_no_sentence
        data.loc[row_index, "prediction with sentence"] = prediction_w_sentence

    data.to_excel(f"text_generation\\predictions\\predictions_{model.name}.xlsx")

    model.clear_space()