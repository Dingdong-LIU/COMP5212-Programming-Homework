from torch import nn
import torch
import numpy as np

class Logistic_Regression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = nn.Parameter(torch.randn((input_size, output_size)))
        # self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        logits = x.matmul(self.weights) 
        return logits.squeeze()

def logistic_loss(y_pred, y_true):
    assert y_pred.shape == y_true.shape, f"Unmatched lenth, where y_pred={y_pred.shape}, y_true={y_true.shape}"

    n = y_pred.shape[0]
    loss = torch.log(1 + torch.exp(-1 * y_true * y_pred))
    loss = torch.sum(loss)/n
    return loss

def svm_loss(y_pred, y_true):
    assert y_pred.shape == y_true.shape, f"Unmatched lenth, where y_pred={y_pred.shape}, y_true={y_true.shape}"
    n = y_pred.shape[0]

    pt_loss = 1 - y_pred.mul(y_true)
    loss = torch.sum(torch.max(torch.zeros_like(pt_loss), pt_loss)) / n
    return loss
