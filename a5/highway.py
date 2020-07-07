#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self, word_embed_size, init_gate_bias_mean=-1):
        super().__init__()
        self.proj = nn.Linear(word_embed_size, word_embed_size)
        self.gate = nn.Linear(word_embed_size, word_embed_size)
        self.gate.bias = nn.Parameter(self.gate.bias.data + init_gate_bias_mean)

    def forward(self, x_conv_out):
        x_proj = nn.functional.relu_(self.proj(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        return x_highway

    ### END YOUR CODE

