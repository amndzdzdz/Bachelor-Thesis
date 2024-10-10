import torch.nn as nn

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()

    def forward(self, x):
        return x