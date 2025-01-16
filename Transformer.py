import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

from PositionalEncoding import PositionalEncoding
from Encoder import Encoder
from Decoder import Decoder

class Transformer (nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, model_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout):
        super(Transformer, self).__init__()
        
        # Defining the embedding layer for the source sequence
        self.encoder_embedding = nn.Embedding(src_vocab_size, model_dim)
        
        # Defining the embedding layer for the target sequence
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, model_dim)
        
        # Defining the positional encoding component
        self.positional_encoding = PositionalEncoding(model_dim, max_seq_len)

        # Defining the list of encoder and decoder layers for the transformer model 
        self.encoder_layers = nn.ModuleList([Encoder(model_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([Decoder(model_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

        # Defining the fully connected layer mapping the model dimension to the target vocabulary size
        self.fc = nn.Linear(model_dim, tgt_vocab_size)
        
        # Defining the dropout layer
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt): # Generating the masks for the source and target sequences 
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output        