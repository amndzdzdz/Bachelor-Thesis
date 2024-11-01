from abc import ABC, abstractmethod

class LLModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def create_prompt(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def clear_space(self):
        pass