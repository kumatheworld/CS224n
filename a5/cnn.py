#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    def __init__(self, char_embedding_size, word_embed_size, kernel_size=5, padding=1):
        super().__init__()
        num_filters = word_embed_size
        self.conv = nn.Conv1d(char_embedding_size, num_filters, kernel_size=kernel_size, padding=padding)

    def forward(self, x_emb):
        x_conv = self.conv(x_emb)
        x_conv_out = torch.max(nn.functional.relu_(x_conv), dim=2).values
        return x_conv_out

    ### END YOUR CODE

