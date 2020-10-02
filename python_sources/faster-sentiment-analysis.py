#!/usr/bin/env python
# coding: utf-8

# # Faster Sentiment Analysis Tutorial
# 
# This notebook is origin from https://github.com/bentrevett/pytorch-sentiment-analysis/ tutorial using our datasets. And specific pretrained embedding. 
# 
# This notebook is just use for learning pytorch pipeline and torchtext.

# ## Import libs

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import time
import random
import string

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext import data
from torchtext.vocab import Vectors

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Preparing Data
# 
# One of the key concepts in the FastText paper is that they calculate the n-grams of an input sentence and append them to the end of a sentence. Here we are going to use bi-grams.
# 
# The generate_bigrams function takes a sentence that has already been tokenized, calculates the bi-grams and appends them to the end of the tokenized list.

# In[ ]:


def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram)) # ["this", 'movie'] -> "this movie"
    return x


# In[ ]:


generate_bigrams(['this', 'moive', 'is' ,'terrible'])


# TorchText Fields have a preprocessing argumnet. A function passed here will be applied to a sentence after it has been tokenized.(transformed from a string to a list of tokens), but before is has been numericalized(transformed from a list of tokens to a list of indexs). This is where we will pass our generate_bigrams.

# In[ ]:


def load_data(filepath, device):
    tokenizer = lambda x: str(x).translate(str.maketrans('', '', string.punctuation)).strip().split()
    TEXT = data.Field(sequential=True, lower=True, tokenize=tokenizer, preprocessing=generate_bigrams, fix_length=200)
#     TEXT = data.Field(sequential=True, lower=True, tokenize=tokenizer, fix_length=200)
    LABEL = data.Field(sequential=False, use_vocab=False)
    
    field = [('text', TEXT), ('label', LABEL)]
    train, valid, test = data.TabularDataset.splits(filepath, train='Train.csv', validation='Valid.csv', test='Test.csv',
                                                   format='csv', skip_header=True, fields=field)
    cache = '/kaggle/working/vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    vector = Vectors(name='/kaggle/input/glove6b100dtxt/glove.6B.100d.txt', cache=cache)
    TEXT.build_vocab(train, vectors=vector, max_size=25000, unk_init=torch.Tensor.normal_)
    
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train, valid, test), device=device, batch_size=64, 
                                                             sort_key=lambda x:len(x.text), sort_within_batch=True)
    return TEXT, LABEL, train_iter, valid_iter, test_iter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT, LABEL, train_iter, valid_iter, test_iter = load_data('/kaggle/input/imdb-dataset-sentiment-analysis-in-csv-format', 
                                                           device=device)


# ## Build the Model
# 
# This model has far fewer parameters than the previous model as it only has 2 layers that have any parameters, the embedding layer and the linear layer. There is no RNN component in sight!
# 
# Instead, it first calculates the word embedding for each word using the glove 100d (blue), then calculates the average of all of the word embeddings and feeds this through the Linear layer (sliver), and that's it.
# 
# ![](https://github.com/bentrevett/pytorch-sentiment-analysis/raw/9210842371c3bbde7b2007051dafa4c74d9768cd/assets/sentiment8.png)
# 
# We implement the averaging with the avg_pool2d (average pool 2-dimensions) function. We can think of the word embeddings as a 2-dimension grid, where the words are along one axis and the dimensions of the word embeddings are along the other. The image below is an example sentence after being convert into 5-dim word embeddings, with the words along th vertical axis and the embeddings along the horizontal axis. Each element in this $4\times 5$ tensor is represented by a green block.
# ![](https://github.com/bentrevett/pytorch-sentiment-analysis/raw/9210842371c3bbde7b2007051dafa4c74d9768cd/assets/sentiment9.png)
# 
# The avg_pool2d uses a filter of size \[sentence_len, 1\]. This is shown in pink in the image below.
# ![](https://github.com/bentrevett/pytorch-sentiment-analysis/raw/9210842371c3bbde7b2007051dafa4c74d9768cd/assets/sentiment10.png)
# 
# We calculate the average value of all elements covered by the filter, then the filter slides to the right, calculating the average over the next column of embedding values for each word in the sentence.
# 
# ![](https://github.com/bentrevett/pytorch-sentiment-analysis/raw/9210842371c3bbde7b2007051dafa4c74d9768cd/assets/sentiment11.png)
# 
# Each filter postion gives us a single value, the average of all covered elements. After the filter has covered all embedding dimensions, we get a $[1\times5]$ tensor. This tensor is then passed through the linear layer to produce our prediction.
# 

# In[ ]:


class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx= pad_idx)
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, text):
        # text : [sen_len, batch_size]
        embedded = self.embedding(text)
        # embedded : [sen_len, batch_size, emb_size]
        embedded = embedded.permute(1, 0, 2)
        # embedded : [batch_size, sen_len, emb_size]
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        # pooled : batch_size,, emb_size
        return self.fc(pooled)
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
OUTPUT_DIM = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)


# In[ ]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# In[ ]:


pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)


# ## Train the Model

# In[ ]:


optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):
    '''
    Returns accuracy per batch...
    '''
    rounded_preds = torch.round(torch.sigmoid(preds)).long()
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time  / 60)
    elapsed_secs = int(elapsed_time -  (elapsed_mins * 60))
    return  elapsed_mins, elapsed_secs


# In[ ]:


def train(model, iterator, optimizer, criterion):
    epoch_loss, epoch_acc = 0, 0
    model.train()
    for i, batch in enumerate(iterator):
        
        predictions = model(batch.text).squeeze()
        
        loss = criterion(predictions, batch.label.float())
        acc = binary_accuracy(predictions, batch.label)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator) 


# In[ ]:


def evaluate(model, iterator, criterion):
    
    epoch_loss, epoch_acc = 0, 0
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
        
            predictions = model(batch.text).squeeze(1)
        
            loss = criterion(predictions, batch.label.float())
            
            acc = binary_accuracy(predictions, batch.label)
            
            epoch_acc += acc.item()
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[ ]:


N_EPOCHS = 5

best_valid_loss = float('inf')
train_loss_list = []
valid_loss_list = []
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train(model, train_iter, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iter, criterion)
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'SentimentModel3.pt')
    print(f'Epoch: {epoch+1:02} | Epoch Time {epoch_mins}m {epoch_secs}s')
    print(f'\t Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Valid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%')


# ## visualize the results

# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')

plt.figure(figsize=(10, 10))
plt.plot(np.arange(1, N_EPOCHS+1, 1), train_loss_list, 'r', label="Train loss")
plt.plot(np.arange(1, N_EPOCHS+1, 1), valid_loss_list, 'b', label="Valid loss")
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.grid()


# In[ ]:


def testModel():
    bestModel = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX).to(device)
    bestModel.load_state_dict(torch.load('SentimentModel3.pt'))
    test_loss, test_acc = evaluate(bestModel, test_iter, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    
testModel()


# ## User Input
# 
# And as before, we can test on any input the user provides making sure to generate bigrams from our tokenizerd sentence.

# In[ ]:


def predict_sentiment(sentence):
    model.eval()
    tokenizer = lambda x: str(x).translate(str.maketrans('', '', string.punctuation)).strip().split()
    tokenized = generate_bigrams(tokenizer(sentence))
    print(tokenized)
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    print(indexed)
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    bestModel = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX).to(device)
    bestModel.load_state_dict(torch.load('SentimentModel3.pt'))
    prediction = torch.sigmoid(bestModel(tensor))
    return prediction.item()


# In[ ]:


predict_sentiment("this movie is good, but make me tried")


# In[ ]:


predict_sentiment("this movie is good")


# ## Summarize
# 
# In fact the bi-gram method is that we create some bi-phrase into our vocab. 
# 
# (for example we regrad 'is good' as a single word which has its own index 1806 in above example-"this movie is good")
# 
# though glove won't recognize it, But in fact it is efficient for our sentiment analysis job!
