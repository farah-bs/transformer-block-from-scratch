# creating an instance of the transformer model

import Transformer as Transformer

src_vocab_size = 100
tgt_vocab_size = 100
model_dim = 512
num_heads = 8
num_layers = 6
ff_dim = 2048
max_seq_len = 100
dropout = 0.1

model = Transformer(src_vocab_size, tgt_vocab_size, model_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout)

print(model.state_dict())