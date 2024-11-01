import os
import pandas as pd
from copy import deepcopy

FOLDER_PATH = "data\\named_entity_recognition\\excel_files"

new_dataframe = []

for filename in os.listdir(FOLDER_PATH):
    filepath = os.path.join(FOLDER_PATH, filename)
    data = pd.read_excel(filepath)

    text_data = data['text']
    text_data = text_data.sample(frac=1)
    text_data = text_data.tolist()

    i = 0
    for sample in text_data:
        i += 1

        if i == 20:
            break
        
        if type(sample) == float:
            continue

        split_sample = sample.split(":")
        keyword = split_sample[0]
        description = split_sample[1:]
        description = ": ".join(description).strip()

        new_dataframe.append(deepcopy({"keyword": keyword, "description": description}))

new_dataframe = pd.DataFrame(new_dataframe)
new_dataframe.to_excel("data\\text_generation\\dataset\\dataset.xlsx", index=False)