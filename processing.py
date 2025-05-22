#processing.py


#tokenize the text
# then build vocab encoder into sequence of token id's 
# truncate to fixed length and
# split it into training and testing 
import numpy as np
import pandas as pd
import re
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

#loading of data
def loading_data(path):
    df=pd.read_csv('test.csv')
    return df['text'].tolist(), df['label'].tolist()

#cleaning of text
def cleaning_text(text):
    text=text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

#tokenize
def tokenize(text):
    return text.split()

#build vocab
def build_vocabulary(tokenized_texts,min_freq=2):
    counter=Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    vocab = {'<PAD>':0,'<UNK>':1}
    for word, freq in counter.items():
        if freq>=min_freq:
            vocab[word]=len(vocab)
    return vocab

#encoding the text
def encode(tokens,vocab):
    return [vocab.get(token,vocab['<UNK>']) for token in tokens]


#finally putting all the parts together
def pre_process(path,vocab=None,max_len=50):
    texts, labels= loading_data(path)
    cleaned=[cleaning_text(t) for t in texts]
    tokenized=[tokenize(t) for t in cleaned]
    
    if vocab is None:
        vocab= build_vocabulary(tokenized)

    encoded = [torch.tensor(encode(toks, vocab)) for toks in tokenized]
    padded = pad_sequence(encoded, batch_first=True, padding_value=0)

    #trunkate or pad
    if padded.size(1) > max_len:
        padded = padded[:, :max_len]
    elif padded.size(1) < max_len:
        pad_width = max_len - padded.size(1)
        padded = torch.nn.functional.pad(padded, (0, pad_width), value=0)

    labels = torch.tensor(labels)
    return padded, labels, vocab

