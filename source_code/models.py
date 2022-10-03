from torch import nn
import torch
import math
# import numpy as np

class Linear_Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = nn.Parameter(nn.init.kaiming_uniform_(
            torch.zeros(input_size, output_size), a=math.sqrt(5)))
        # self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        logits = x.matmul(self.weights)
        return logits.squeeze()

def logistic_loss(y_pred, y_true):
    assert y_pred.shape == y_true.shape, f"Unmatched lenth, where y_pred={y_pred.shape}, y_true={y_true.shape}"

    n = y_pred.shape[0]
    loss = torch.log(1 + torch.exp(-y_true.mul(y_pred)))
    loss = torch.sum(loss)/n
    return loss

def svm_loss(y_pred, y_true):
    assert y_pred.shape == y_true.shape, f"Unmatched lenth, where y_pred={y_pred.shape}, y_true={y_true.shape}"
    n = y_pred.shape[0]

    pt_loss = 1 - y_pred.mul(y_true)
    loss = torch.sum(torch.max(torch.zeros_like(pt_loss), pt_loss)) / n
    return loss
