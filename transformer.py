import os
import sys
import argparse
from typing import Dict, List
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

from load_data import load_data, TransformerDataset
from perplexity import evaluate_perplexity
from wer import rerank_sentences_for_wer, compute_wer

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CharacterLevelTransformer(nn.Module):
    """
    For this part of the assignment, we provide you with a skeleton for the Transformer
    decoder. However, we've introduced numerous errors to the code! The model currently compiles,
    but performs the incorrect computations. You must fix them to pass the unit tests.

    You may introduce additional keyword arguments after fixing the transformer, as long as the
    default behavior does not stray from the skeleton provided to you.
    """

    def __init__(self, num_layers: int, hidden_dim: int, num_heads: int,
                 ff_dim: int, dropout: float, vocab_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=vocab_size-1)
        self.pos_embed = PositionalEncoding(hidden_dim, dropout)
        self.decoder = Decoder(num_layers, hidden_dim, num_heads, ff_dim, dropout)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.padding_idx = vocab_size - 1

    def log_probability(self, input_tokens: torch.Tensor, target_tokens: torch.Tensor, base=np.e):
        """
        Computes the log-probabilities for the inputs in the given minibatch.

        Input:
            input_tokens (torch.Tensor): A tensor of shape (B, T), where B is the 
                                         batch-size and T is the input length. 
            target_tokens (torch.Tensor): A tensor of shape (B, T). For a given (i, j),
                                          target_tokens[i, j] should be the token following
                                          input_tokens[i, j]
        Output (torch.Tensor): A tensor of shape (B,) containing the log-probability for each
                               example in the minibatch
        
        useful:
        - https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
        - https://edstem.org/us/courses/52897/discussion/4583917
        """
        # start with input_tokens w shape (B,T)
        x = self.forward(input_tokens)  # x.shape (B,T,vocab_size)
        x = F.softmax(x, dim=-1)  # x.shape (B,T,vocab_size)
        # now we have next-token-probability-dist for all the input tokens
        
        # pick probability of the target token from each distribution in x.shape=(B,T)
        tx = x.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)  # tx.shape = (B,T)
        # each item in tx.shape(B,T) is prob of target token picked from x
        
        # do log_base(tx)
        logtx = torch.log(tx)
        logbase = torch.log(torch.tensor(base))
        tx = logtx / logbase
        
        # ignore logprob of padding tokens
        padtkns = target_tokens == self.padding_idx
        tx = tx.masked_fill(padtkns == True, 0)
        
        # sum of log probs across each example
        tx = torch.sum(tx, dim=-1)

        return tx

    def forward(self, model_input):
        # Perform the embedding
        embeds = self.embed(model_input) * math.sqrt(self.hidden_dim)
        embeds = self.pos_embed(embeds)

        # Pass through the decoder
        mask = construct_self_attn_mask(model_input)
        decoder_output = self.decoder(embeds, mask)
        output = self.lm_head(decoder_output)
        return output

def construct_self_attn_mask(x: torch.Tensor):
    """
    The output to this function should be a mask of shape
    (1, T, T). Indices that a token can attend to should be
    set to true.

    There are two errors in this function.
    
    fixes:
    - use tril and diagonal=0 for lower diagonal half of [TxT]
    - basically diagonal and everything below it will be True, everything above it False
    - see blue mask image in https://peterbloem.nl/files/transformers/masked-attention.svg
    """
    T = x.size(1)
    all_ones = torch.ones(T, T)

    mask = torch.tril(all_ones, diagonal=0) == 1
    mask = mask.unsqueeze(0)
    return mask.to(x.device)

class Decoder(nn.Module):

    def __init__(self, num_layers, hidden_dim, num_heads, ff_dim, dropout):
        """
        There is a single error in this function that will prevent the model from learning.
        
        fixes:
        - use nn.Sequential() so pytorch is aware that these layers exist
        - https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/5
        """
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(TransformerBlock(num_heads, hidden_dim, ff_dim, dropout)) 
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerBlock(nn.Module):

    def __init__(self, num_heads, hidden_dim, ff_dim, dropout):
        super().__init__()

        # Attention block
        self.attn_block = MultiHeadAttention(num_heads, hidden_dim, dropout)
        self.attn_dropout = nn.Dropout(dropout) 
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Feedforward block
        self.mlp_block = TransformerMLP(hidden_dim, ff_dim, dropout)
        self.mlp_dropout = nn.Dropout(dropout) 
        self.mlp_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        """
        There are two types of errors in this function.
        clarification: There are two types of errors in the function,
            but three errors total (2 for one error type and 1 for the other)
        
        fixes:
        - switch order so do attn first and then do mlp (see class slides)
        - residual connection (add x to each LayerNorm)
        """
        x = self.attn_norm(self.attn_dropout(self.attn_block(x, mask)) + x)
        x = self.mlp_norm(self.mlp_dropout(self.mlp_block(x)) + x)
        return x
   
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, hidden_dim, dropout=0.1):
        super().__init__()

        self.h = num_heads
        self.qkv_dim = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def attention(self, query, key, value, mask):
        """
        There are three errors in this function to fix.
        
        fixes:
        - query @ key.T
        - use -1e9 because we want -infinity for positions that are masked and should not be attended to
        - softmax over last dim of dot_products
        """
        dot_products = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.qkv_dim)
        dot_products = dot_products.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(dot_products, dim=-1))
        return torch.matmul(attn, value)

    def forward(self, x, mask):
        """
        There are two errors in this function to fix
        
        fixes:
        - use k_proj and v_proj
        """
        mask = mask.unsqueeze(1)
        B = x.size(0)

        # Compute the query, key and value vectors
        query = self.q_proj(x).view(B, -1, self.h, self.qkv_dim).transpose(1, 2)
        key = self.k_proj(x).view(B, -1, self.h, self.qkv_dim).transpose(1, 2)
        value = self.v_proj(x).view(B, -1, self.h, self.qkv_dim).transpose(1, 2)
        # Perform self-attention
        x = self.attention(query, key, value, mask)

        # Concatenate the outputs for each attention head
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.qkv_dim)
        return self.out_proj(x)

class TransformerMLP(nn.Module):

    def __init__(self, hidden_dim, ff_dim, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        There is a single error in this function to fix.
        
        fixes:
        - add gelu , see class slides:  W'' GELU(W'z + b') + b''
        """
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

class PositionalEncoding(nn.Module):

    def __init__(self, hidden_dim, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        positional_encodings = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (- math.log(10000) / hidden_dim))
        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, 1::2] = torch.cos(position * div_term)
        positional_encodings = positional_encodings.unsqueeze(0)

        self.register_buffer('positional_encodings', positional_encodings, persistent=False)

    def forward(self, x):
        x = x + self.positional_encodings[:, :x.size(1)]
        return self.dropout(x)



def train(model:CharacterLevelTransformer, train_data:TransformerDataset, val_data:TransformerDataset,
          dev_wer_data, loss_fct, optimizer, max_epochs:int):
    """
    Training loop for the transformer model. You may change the header as you see fit.
    """
    # incomplete, idk how dataloader works  @todo
    train_data_loader = train_data # DataLoader(train_data, batch_size, collate_fn=train_data.collate_fn)  
    val_data_loader = val_data # DataLoader(val_data, batch_size, collate_fn=train_data.collate_fn) 
    
    for epoch in range(max_epochs):
        # print(f"~  training epoch {epoch}..")
        
        # train on training data
        model.train()
        total_train_loss = 0
        total_train_accuracy = 0
        for b, (data, label) in enumerate(train_data_loader):
            print(f"~   batch {b}")

            optimizer.zero_grad()  # reset gradients
            y = model.forward(data)  # forward pass
            y2 = y.reshape(-1,y.size(-1))
            label2 = label.reshape(-1)#,label.size(-1))
            loss = loss_fct(y2, label2) # calculate loss
            total_train_loss += loss.item()
            loss.backward() # backprop
            optimizer.step()  # update params
        
        avg_train_loss = total_train_loss / len(train_data_loader)
        avg_train_accuracy = total_train_accuracy / len(train_data_loader)
        print(f"Epoch {epoch}: Training Loss: {avg_train_loss:.4f}")
        
        # check on validation data
        model.eval()
        total_val_loss = 0
        total_val_accuracy = 0
        with torch.no_grad():
            for data, labels in val_data_loader:
                outputs = model(data)
                outputs = outputs.reshape(-1, outputs.size(-1))
                labels = labels.reshape(-1)
                loss = loss_fct(outputs, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_data_loader)
        avg_val_accuracy = total_val_accuracy / len(val_data_loader)
        print(f"Epoch {epoch}: Validation Loss: {avg_val_loss:.4f}")

def get_args():
    """
    You may freely add new command line arguments to this function.
    """
    parser = argparse.ArgumentParser(description='Transformer model')
    parser.add_argument('--num_layers', type=int, default=4,
                        help="How many transformer blocks to use")
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help="What is the transformer hidden dimension")
    parser.add_argument('--num_heads', type=int, default=4,
                        help="How many heads to use for Multihead Attention")
    parser.add_argument('--ff_dim', type=int, default=2048,
                        help="What is the intermediate dimension for the feedforward layer")
    parser.add_argument('--dropout_p', type=int, default=0.1,
                        help="The dropout probability to use")    

    parser.add_argument('--experiment_name', type=str, default='testing_')
    parser.add_argument('--num_samples', type=int, default=10,
                        help="How many samples should we get from our model??")
    parser.add_argument('--max_steps', type=int, default=40,
                        help="What should the maximum output length be?")
    

    args = parser.parse_args()
    return args

def main():
    # Get key arguments
    args = get_args()
    
    BATCH_SIZE = 50
    # Get the data
    tokenization_level = "character"
    model_type = "transformer"
    train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data, vocab = load_data(tokenization_level, model_type) 
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_data.collate_fn)  
    val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_data.collate_fn) 
    dev_data_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=train_data.collate_fn) 
    test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=train_data.collate_fn) 

    # Initialize the transformer and train
    num_layers = args.num_layers
    hidden_dim = args.hidden_dim
    num_heads = args.num_heads
    ff_dim = args.ff_dim
    dropout_p = args.dropout_p
    vocab_size = 37

    #########################
    model = CharacterLevelTransformer(num_layers, hidden_dim, num_heads, ff_dim,
                                      dropout_p, vocab_size).to(DEVICE)
    
    print("~ training model..")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fct = nn.CrossEntropyLoss()
    # loss_fct = nn.L1Loss
    max_epochs = 1
    train(model, train_data_loader, val_data_loader, dev_wer_data, loss_fct, optimizer, max_epochs)
    ########################

    # Evaluate model perplexity
    print("~ evaluating model perplexity..")
    model.eval()
    # val_perplexity = evaluate_perplexity(model, val_data_loader)
    # print(f'Model perplexity on the val set: {val_perplexity}')
    dev_perplexity = evaluate_perplexity(model, dev_data_loader)
    print(f'Model perplexity on the dev set: {dev_perplexity}')
    test_perplexity = evaluate_perplexity(model, test_data_loader)
    print(f'Model perplexity on the test set: {test_perplexity}')    
    
    # Evaluate model WER
    print("~ evaluating model wer..")
    experiment_name = args.experiment_name
    dev_wer_savepath = os.path.join('results', f'{experiment_name}_dev_wer_predictions.csv')
    rerank_sentences_for_wer(model, dev_wer_data, dev_wer_savepath, vocab) 
    dev_wer = compute_wer('data/wer_data/dev_ground_truths.csv', dev_wer_savepath)
    print("Dev set WER was: ", dev_wer)

    test_wer_savepath = os.path.join('results', f'{experiment_name}_test_wer_predictions.csv')
    rerank_sentences_for_wer(model, test_wer_data, test_wer_savepath, vocab)
    
    # # Generate text from the model
    # generation_path = os.path.join('generations', f'{experiment_name}transformer_generation_examples.pkl')
    # num_samples = args.num_samples
    # max_steps = args.max_steps
    # model.generate(num_samples, max_steps, generation_path)
    
    print("~ done :)")


if __name__ == "__main__":
    main()
