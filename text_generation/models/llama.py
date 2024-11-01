import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from models.model import LLModel

class Llama32_1B(LLModel):

    def __init__(self):
        self.name = "Llama 3.2 1B"

    def initialize(self):
        self.pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16, device_map="auto")

    def create_prompt(self, prompt: str, tech_term: str, sentence = None) -> str:
        prompt = prompt.replace("XX_keyword_XX", tech_term)

        if sentence != None:
            prompt = prompt.replace("XX_sentence_XX", sentence)

        return prompt

    def predict(self, prompt:str, tech_term: str, sentence=None):
        prompt = self.create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence)
        output = self.pipe(prompt)

        return output
    
    def clear_space(self) -> None:
        del self.pipe
        gc.collect()    
        torch.cuda.empty_cache()

class Llama32_3B(LLModel):

    def __init__(self):
        self.name = "Llama 3.2 3B"

    def initialize(self):
        self.pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B", torch_dtype=torch.bfloat16, device_map="auto")

    def create_prompt(self, prompt: str, tech_term: str, sentence = None) -> str:
        prompt = prompt.replace("XX_keyword_XX", tech_term)

        if sentence != None:
            prompt = prompt.replace("XX_sentence_XX", sentence)

        return prompt

    def predict(self, prompt:str, tech_term: str, sentence=None):
        prompt = self.create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence)
        output = self.pipe(prompt)

        return output
    
    def clear_space(self) -> None:
        del self.pipe
        gc.collect()    
        torch.cuda.empty_cache()

class Llama31_70B(LLModel):

    def __init__(self):
        self.name = "Llama 3.1 70B"

    def initialize(self):
        self.pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-70B", torch_dtype=torch.bfloat16, device_map="auto")

    def create_prompt(self, prompt: str, tech_term: str, sentence = None) -> str:
        prompt = prompt.replace("XX_keyword_XX", tech_term)

        if sentence != None:
            prompt = prompt.replace("XX_sentence_XX", sentence)

        return prompt

    def predict(self, prompt:str, tech_term: str, sentence=None):
        prompt = self.create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence)
        output = self.pipe(prompt)

        return output
    
    def clear_space(self) -> None:
        del self.pipe
        gc.collect()    
        torch.cuda.empty_cache()

#Llama 3.2 https://llama3-2-lightweight.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiMjNuMWtkazZkYWtxbHZxcDZnMDcwdWFqIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTItbGlnaHR3ZWlnaHQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczMDYyMjE0M319fV19&Signature=rVcufGpxEqvIg4%7EBHgyWp0hApBuIaBMJunRLnfaYn0yPrto78n%7EM9GbQCVb8fGUis-D9VDJJ5H3jkluNaCDrA9yFSNula4G34d4zbYjaS41aQGZ-eEFYwN6%7EYkrR2vUx12fymTz5owvwY0MM0XMKlc5YHvr0loFy2M1AFdZUF%7E-sbr2JcbMAUNc1ysMDzsWc2tmimVptO1OSrYfuTSXLiIhnG1ib36ARdiHOyDFRb0uXwlF1OHaim1GpO2XyUBKxyM-t6ufBM8NeG6epQXMjcLmi7mDli9Ha9IWrM7QB8IacbdS5OKY-2db8-fsyQjq-N1HsnpJdjBBPvXNwLWU0pA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1271862127436497
#Llama 3.1 https://llama3-1.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiN2k3NmtteTdzc3NubGQ2Z2g1YncyOTRqIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTEubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczMDYyMjE0M319fV19&Signature=hY01io9ZVovs4fFI1e7SmV4eUnDJ-Y4N3%7ERHbs1GJ7XPN2S48YVEndXta79BdS1sgK06hx3wmdRroqXUyxicwuoZtSwlGNNABfCcddJtsN%7EK4iotyvlva8Bnyqk78tsEkrJboUwp1u3B%7EWnVNrVg6KruhF3fztjYpgLVvj8HI2GC6d6VGXhhS6iS7Vm5s39xGxviCynQjZyvfKmR%7EvkzwvSFXfV-elxyAH8crSMWnlzac3NziTLsu4QDOqxjWP4fo0mxDyxxPwbH53PJx2XVylhPqkHk2SdLlyIOO1W9bVy-fA7S8e4S7QB%7EuO4j9%7EIjsqNdeQnu2b4gEpi6exH2Kg__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=3490731597897424
