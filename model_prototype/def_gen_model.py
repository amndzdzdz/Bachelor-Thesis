import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, AutoConfig
from huggingface_hub import login
import gc
import re

class Ministral:

    def __init__(self):
        self.pipe = pipeline("text-generation", model="mistralai/Ministral-8B-Instruct-2410", 
                             torch_dtype=torch.bfloat16, device_map="auto", max_new_tokens=512)

    def predict(self, prompt:str, tech_term: str, sentence=None):
        
        prompt = self.__create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence)
        output = self.pipe(prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)

        output = output[0]['generated_text'][-1]['content']
        return output

    def __create_prompt(self, prompt: str, tech_term: str, sentence = None) -> str:
        prompt = prompt.replace("{term}", tech_term)

        if sentence != None:
            prompt = prompt.replace("{context}", sentence)

        messages = [
        {
        "role": "user",
        "content": prompt}]
    
        return messages

class PHI:

    def __init__(self):
        self.pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", 
                             torch_dtype=torch.bfloat16, device_map="auto", max_new_tokens=512, trust_remote_code=True)

    def __create_prompt(self, prompt: str, tech_term: str, sentence = None) -> str:
        prompt = prompt.replace("{term}", tech_term)

        if sentence != None:
            prompt = prompt.replace("{context}", sentence)

        messages = [
            {"role": "system", 
             "content": "You are an expert assistant specialized in technical terms. Provide clear, accurate and brief definitions for technical terms."},
            {"role": "user", "content": prompt}
        ]
        
        return messages

    def predict(self, prompt:str, tech_term: str, sentence=None):
        prompt = self.__create_prompt(prompt=prompt, tech_term=tech_term, sentence=sentence)
        output = self.pipe(prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)
        print(output)
        output = output[0]['generated_text'][-1]['content']
        
        return output
    
    def clear_space(self) -> None:
        del self.pipe
        gc.collect()    
        torch.cuda.empty_cache()