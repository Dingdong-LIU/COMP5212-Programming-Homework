from torch import nn
import torch
import numpy as np

class Logistic_Regression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = nn.Parameter(torch.randn((input_size, output_size))/np.sqrt(input_size))
        # self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        logits = x.matmul(self.weights) 
        return logits

def logistic_loss(y_pred, y_true):
    n = len(y_pred)
    assert len(y_pred) == len(
        y_true), f"Unmatched lenth, where y_pred={y_pred.shape}, y_true={y_true.shape}"
    loss = torch.log(1 + torch.exp(-1 * y_true * y_pred))
    loss = torch.sum(loss)/n
    return loss

