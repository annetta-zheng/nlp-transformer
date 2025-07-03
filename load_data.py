import os
import random
import json

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

import re
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence

from typing import List, Any, Tuple
random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def mean(data):
    return sum(len(x) for x in data)/len(data)
def load_data(tokenization_level: str, model_type: str):
    """
    Function for loading data for language modeling and WER computation. You
    may modify the function header and outputs as necessary.

    Inputs:
        tokenization_level (str): The level at which to tokenize the input
        model_type (str): n_gram or transformer
    Returns: A list of model inputs, which should each be lists of input tokens
    """
    # TODO
    print('Loading data...')
    
    data = read_file("data/lm_data/treebank-sentences-train.txt")
    dev_data = read_file('data/lm_data/treebank-sentences-dev.txt')
    test_data = read_file('data/lm_data/treebank-sentences-test.txt')
    
    data, dev_data, test_data = clean_data(data, dev_data, test_data, tokenization_level)
    print('Data loaded and cleaned.')
    
    if tokenization_level == "word":
        train_data, val_data, dev_data, test_data, vocab = word_level_tokens(data, dev_data, test_data)
    elif tokenization_level == "subword":
        train_data, val_data, dev_data, test_data, vocab = subword_level_tokens_pretrain(data, dev_data, test_data)
    elif tokenization_level == "character":
        train_data, val_data, dev_data, test_data, vocab = character_level_tokens(data, dev_data, test_data)   
    print('vocab_size', len(vocab))    
    # print('size', len(train_data), len(val_data), len(dev_data), len(test_data))
    print('avg token length', mean(train_data), mean(val_data), mean(dev_data), mean(test_data))
    dev_wer_data = read_file('data/wer_data/dev_sentences.json', file_type = 'json')
    test_wer_data = read_file('data/wer_data/test_sentences.json', file_type = 'json')
    if model_type == 'n_gram':
        return train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data    
    elif model_type == 'transformer':
        if tokenization_level == 'character':
            train_data = [[vocab.index(char) for char in ''.join(s)] for s in train_data]
            val_data = [[vocab.index(char) for char in ''.join(s)] for s in val_data]
            dev_data = [[vocab.index(char) if char in vocab else vocab.index('unk') for char in ''.join(s)] for s in dev_data]
            test_data = [[vocab.index(char) if char in vocab else vocab.index('unk') for char in ''.join(s)] for s in test_data]
            # test_wer_data = [[vocab.index(char) if char in vocab else vocab.index('unk') for char in ''.join(s)] for s in test_wer_data]
        else:
            train_data = [vocab.index(token) for token in train_data]
            val_data = [vocab.index(token) for token in val_data]
            dev_data = [vocab.index(token) for token in dev_data]
            test_data = [vocab.index(token) for token in test_data]
        return TransformerDataset(train_data), TransformerDataset(val_data), TransformerDataset(dev_data), TransformerDataset(test_data), dev_wer_data, test_wer_data, vocab
    else:
        print('Failed Loading data, model_type doesn\'t exist.')
        return [], [], [], [], [], []
class JsonData(IterableDataset):
    def __init__(self, data):
        self.data = data
    def __iter__(self):
        with open(self.data, 'r') as f:
            file = json.loads(f)
            for i in file.items():
                sid, s = i
                yield sid, s
                
                    
class TransformerDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # print(self.data[idx])
        input_data = torch.tensor(self.data[idx]).to(device)
        return input_data
    def collate_fn(self, batch):
        batch = pad_sequence(batch,batch_first= True, padding_value=36)
        
        labels = batch[:,1:]
        batch = batch[:,:-1]
        print(batch.shape, labels.shape)
        return batch, labels#input_tensor, target_tensor
        
def clean_data(data, dev_data, test_data, tokenization_level):
    if tokenization_level == 'subword':
        # train_data, val_data = train_val_split(data)
        # print('original unique',len(set(word for x in train_data for word in x.split())),len(set(word for x in val_data for word in x.split())),len(set(word for x in dev_data for word in x.split())),len(set(word for x in test_data for word in x.split())))
        # print('original unique char',len(set(char for x in train_data for char in ''.join(x.split()))),len(set(char for x in val_data for char in ''.join(x.split()))),len(set(char for x in dev_data for char in ''.join(x.split()))),len(set(char for x in test_data for char in ''.join(x.split()))))
    
        return data, dev_data, test_data
    def clean_helper(x):
        x = x.lower()
        x = re.sub('\'s', 'is', x)
        x = re.sub('n\'t', 'not', x)
        x = re.sub('\.\s', '', x)
        x = re.sub('[^\w]+', ' ', x)
        x = re.sub(' +', ' ', x)
        # x = re.sub(' ', '', x)
        return x
    # train_data, val_data = train_val_split(data)
    # print('original unique',len(set(word for x in train_data for word in x.split())),len(set(word for x in val_data for word in x.split())),len(set(word for x in dev_data for word in x.split())),len(set(word for x in test_data for word in x.split())))
    # print('original unique char',len(set(char for x in train_data for char in ''.join(x.split()))),len(set(char for x in val_data for char in ''.join(x.split()))),len(set(char for x in dev_data for char in ''.join(x.split()))),len(set(char for x in test_data for char in ''.join(x.split()))))
    # print('size', len(train_data), len(val_data), len(dev_data), len(test_data))
    # print('avg sentence length', mean(train_data), mean(val_data), mean(dev_data), mean(test_data))
    data = [clean_helper(x) for x in data]
    dev_data = [clean_helper(x) for x in data]
    test_data = [clean_helper(x) for x in data]
    return data, dev_data, test_data
def subword_level_tokens_pretrain(data, dev_data, test_data):
    print("Running GPT-2 tokenizer")
    tokenizer = Tokenizer.from_pretrained("gpt2")
    
    data = tokenizer.encode_batch(data)
    data = ['<|beginoftext|> ' + " ".join(i.tokens) + ' <|endoftext|>' for i in data]
    dev_data = tokenizer.encode_batch(dev_data)
    dev_data = ['<|beginoftext|> ' + " ".join(i.tokens) + ' <|endoftext|>' for i in dev_data]
    test_data = tokenizer.encode_batch(test_data)
    test_data = ['<|beginoftext|> ' + " ".join(i.tokens) + ' <|endoftext|>' for i in test_data]
    
    train_data, val_data = train_val_split(data)
    vocab = list(set(word for x in data for word in x.split()))
    return train_data, val_data, dev_data, test_data, vocab

# def subword_level_tokens(data, dev_data, test_data):
#     tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
#     tokenizer.normalizer = normalizers.Sequence([
#         normalizers.NFD(),
#         normalizers.StripAccents(),
#         normalizers.Lowercase()
#     ])
#         # tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
#     # tokenizer.pre_tokenizer = Whitespace()
#     # tokenizer.normalizer = normalizers.Sequence([
#     #     normalizers.NFD(),
#     #     normalizers.StripAccents(),
#     #     normalizers.Lowercase()
#     # ])
#     # special_tokens_dict = {
#     #     "<|bos|>": AddedToken("<|bos|>", lstrip=False, rstrip=True),
#     #     "<|eos|>": AddedToken("<|eos|>", lstrip=True, rstrip=False)
#     # }
#     # special_tokens = [ "<|unk|>","<|bos|>", "<|eos|>", "<|mask|>", "<|cls|>", "<|sep|>"]
#     # tokenizer.add_special_tokens(list(special_tokens_dict.keys()))
#         # print(tokenizer.bos_token)
#     # print(tokenizer.add_special_tokens(special_tokens))

#     # tokenizer.train(data)
#     tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
#     special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
#     trainer = trainers.BpeTrainer(vocab_size=15000, special_tokens=special_tokens)
#     tokenizer.train(data, trainer)
#     tokenizer.save("tokenizer.json")
#     # # Creating huggingface dataset object
#     # dataset = Dataset.from_pandas(test[['text']])
#     # def train_corp_iter(): 
#     #     """
#     #     A generator function for iterating over a dataset in chunks.
#     #     """    
#     #     for i in range(0, len(dataset), 1000):
#     #         yield dataset[i : i + 1000]["text"]
#     # raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
    
#     #     # Tokenize train&test set with new tokenizer
#     # tokenized_texts_train = []
#     # for text in tqdm(train['text'].tolist()):
#     #     tokenized_texts_train.append(tokenizer.tokenize(text))

#     # tokenized_texts_test = []
#     # for text in tqdm(test['text'].tolist()):
#     #     tokenized_texts_test.append(tokenizer.tokenize(text))

#     # tokenizer = PreTrainedTokenizerFast(
#     #     tokenizer_object=raw_tokenizer,
#     #     unk_token="[UNK]",
#     #     pad_token="[PAD]",
#     #     cls_token="[CLS]",
#     #     sep_token="[SEP]",
#     #     mask_token="[MASK]",
#     # )

#     return train_data, val_data, dev_data, test_data

def train_val_split(data, split_ratio=0.15):
    num_samples = len(data)
    split_index = int(num_samples * (1 - split_ratio))
    random.shuffle(data)
    train_data, val_data = data[:split_index], data[split_index:]
    return train_data, val_data
def read_file(data_path, file_type = 'txt'):
    if file_type == 'txt':
        f = open(data_path, "r")
        data = [line.rstrip() for line in f]
        f.close()
        return data
    elif file_type == 'json':
        f = open(data_path, "r")
        data = json.load(f)
        f.close()
        return data
def word_level_tokens(data, dev_data, test_data):
    data = ['<|beginoftext|> ' + x + ' <|endoftext|>' for x in data]
    dev_data = ['<|beginoftext|> ' + x + ' <|endoftext|>' for x in dev_data]
    test_data = ['<|beginoftext|> ' + x + ' <|endoftext|>' for x in test_data]
    
    counter = Counter(word for x in data for word in x.split())
    vocabulary = list(k for k, v in counter.items() if v > 1)
    vocabulary.append('unk')
    train_data, val_data = train_val_split(data)
    
    train_data = list(x.split() for x in train_data)
    val_data = list(x.split() for x in val_data)
    dev_data = list([i if i in vocabulary else 'unk' for i in x.split()] for x in dev_data)
    test_data = list([i if i in vocabulary else 'unk' for i in x.split()] for x in test_data)
    return train_data, val_data, dev_data, test_data, vocabulary
    
def character_level_tokens(data, dev_data, test_data):
    vocabulary = list(set(char for x in data for char in ''.join(x.split())))
    vocabulary.append('unk')
    vocabulary.append(' ')
    train_data, val_data = train_val_split(data)
    
    train_data = [[char for char in sentence] for sentence in train_data]
    val_data = [[char for char in sentence] for sentence in val_data]
    dev_data = [[char if char in vocabulary else 'unk' for char in ''.join(sentence.split())] for sentence in dev_data]
    test_data = [[char if char in vocabulary else 'unk' for char in ''.join(sentence.split())] for sentence in test_data]
    return train_data, val_data, dev_data, test_data, vocabulary
    
# def n_gram(tokenization_level):
#     data = read_txt_file("data/lm_data/treebank-sentences-train.txt")
#     dev_data = read_txt_file('data/lm_data/treebank-sentences-dev.txt')
#     test_data = read_txt_file('data/lm_data/treebank-sentences-test.txt')
    
#     if tokenization_level == "word":
#         train_data, val_data, dev_data, test_data = word_level_tokens(data, dev_data, test_data)
#     elif tokenization_level == "subword":
#         pass
#     elif tokenization_level == "character":
#         train_data, val_data, dev_data, test_data = character_level_tokens(data, dev_data, test_data)       
    
    
    
#     # print(train_data[0], val_data[0])
#     dev_wer_data = read_wer_data('data/wer_data/dev_sentences.json')
#     test_wer_data = read_wer_data('data/wer_data/test_sentences.json')
#     # https://cjlise.github.io/machine-learning/Neural-Language-Model/
#     # BOS, EOS = ' ', '\n'
#     # # print(dev_wer_data)
#     # lines = dev_wer_data.apply(lambda line: BOS + line.replace(EOS, ' ') + EOS) \
#     #         .tolist()
#     # print(lines[0])
#     # print(train_data[0], val_data[0])

#     print('Data loaded.')
#     return train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data

        