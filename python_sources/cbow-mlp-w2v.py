#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re
from gensim.models import KeyedVectors
import gensim.downloader as api
import torch
from torch import nn
from torch.utils import data
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[ ]:


CUDA_ENABLED=True


# # Loading and cleaning the data

# In[ ]:


train_df = pd.read_csv('../input/stanford-natural-language-inference-39/train.csv', sep='\t')
val_df = pd.read_csv('../input/stanford-natural-language-inference-39/val.csv', sep='\t')
test_df = pd.read_csv('../input/stanford-natural-language-inference-39/test.csv', sep='\t')
train_df.head()


# ### cleaning the text

# In[ ]:


train_df.sentence1.sample(10)


# In[ ]:


def brief_clean(txt):
    return re.sub("[^A-Za-z']+", ' ', str(txt)).lower().replace("'", '')


# In[ ]:


train_df['sentence1_clean'] = train_df.sentence1.apply(brief_clean)
val_df['sentence1_clean'] = val_df.sentence1.apply(brief_clean)
test_df['sentence1_clean'] = test_df.sentence1.apply(brief_clean)

train_df['sentence2_clean'] = train_df.sentence2.apply(brief_clean)
val_df['sentence2_clean'] = val_df.sentence2.apply(brief_clean)
test_df['sentence2_clean'] = test_df.sentence2.apply(brief_clean)


# ### embedding the Category

# In[ ]:


train_df.Category.head(5)


# In[ ]:


train_df.Category.unique()


# In[ ]:


train_df = train_df[train_df.Category!='-']
val_df = val_df[val_df.Category!='-']


# In[ ]:


train_df.Category.unique()


# In[ ]:


category_encoder = LabelEncoder()
train_df['target'] = category_encoder.fit_transform(train_df.Category)
val_df['target'] = category_encoder.transform(val_df.Category)


# In[ ]:


train_df.target.head(5)


# # Load pretrained w2v weights (model) using gensim

# ptretrained weights download link:
# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

# In[ ]:


max_vocab_size = 100000


# In[ ]:


#w2v_model = api.load('word2vec-google-news-300')


# In[ ]:


w2v_model = KeyedVectors.load_word2vec_format('../input/google-news-w2v/Google.bin', binary=True, limit=max_vocab_size)


# In[ ]:


w2v_model['cat']


# In[ ]:


embeddings = w2v_model.vectors[:max_vocab_size,:]
embeddings = np.concatenate((np.zeros((1,300)), embeddings))
embeddings.shape


# In[ ]:


word2id = {k:i+1 for i,k in enumerate(w2v_model.index2word) if i <max_vocab_size}


# In[ ]:


print('word id: {}'.format(word2id['cat']))
print('word vector:', embeddings[word2id['cat']])


# # Preparing the data for a model

# In[ ]:


#Indexing the sentences words by the w2v dictionary
def preprocess_sentence(sentence, word2id, other_id=0):
    sentence = sentence.split(' ')
    sentence = np.array([word2id[c] if c in word2id else other_id for c in sentence])
    return sentence


# In[ ]:


train_df['x1'] = train_df.sentence1_clean.apply(lambda x: preprocess_sentence(x, word2id))
train_df['x2'] = train_df.sentence2_clean.apply(lambda x: preprocess_sentence(x, word2id))

val_df['x1'] = val_df.sentence1_clean.apply(lambda x: preprocess_sentence(x, word2id))
val_df['x2'] = val_df.sentence2_clean.apply(lambda x: preprocess_sentence(x, word2id))

test_df['x1'] = test_df.sentence1_clean.apply(lambda x: preprocess_sentence(x, word2id))
test_df['x2'] = test_df.sentence2_clean.apply(lambda x: preprocess_sentence(x, word2id))


# In[ ]:


class Dataset(data.Dataset):
  def __init__(self, df, CUDA_ENABLED=True):
        self.length = len(df)
        self.X1 = df.x1
        self.X2 = df.x2
        self.cuda = CUDA_ENABLED
        if 'target' in df.columns:
            self.has_target = True
            self.target = df.target
        else:
            self.has_target = False

  def __len__(self):
        'Denotes the total number of samples'
        return self.length

  def __getitem__(self, index):
        'Generates one sample of data'
        x1 = torch.LongTensor(self.X1.iloc[index])
        x2 = torch.LongTensor(self.X2.iloc[index])
        if self.cuda:
            x1= x1.cuda()
            x2= x2.cuda()
        if self.has_target:
            y = self.target.iloc[index]
            return x1, x2, y
        else:
            return x1, x2


# In[ ]:


train_ds = Dataset(train_df, CUDA_ENABLED)
val_ds = Dataset(val_df, CUDA_ENABLED)
test_ds =  Dataset(test_df, CUDA_ENABLED)


# In[ ]:


def colllate_tow_elements(samples):
    X1 = []
    X2 = []
    for x1, x2 in samples:
        X1.append(x1)
        X2.append(x2)
    return X1, X2
def colllate_three_elements(samples):
    X1 = []
    X2 = []
    Y = []
    for x1, x2, y in samples:
        X1.append(x1)
        X2.append(x2)
        Y.append(y)
    return X1, X2, Y
    


# In[ ]:


train_generator = data.DataLoader(train_ds, shuffle=True, batch_size=32, collate_fn=colllate_three_elements, drop_last=True)
val_generator = data.DataLoader(val_ds, shuffle=True, batch_size=32, collate_fn=colllate_three_elements, drop_last=True)
test_generator = data.DataLoader(test_ds, shuffle=False, batch_size=32, collate_fn=colllate_tow_elements, drop_last=False)


# In[ ]:


for X1, X2, y in train_generator:
    print (len(X1), len(X2), len(y))
    print (X1[0], X2[0], y)
    break


# # Building and training a model

# In[ ]:


#this is the class for my MLP classifier, with w2v-initialized embedding layer
class TwoInputsMLPClassifier(torch.nn.Module):

    def __init__(self, n_vocab, embed_init, embedding_dim, hidden_dim, n_traget):
        super(TwoInputsMLPClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_hidden = hidden_dim
        self.n_vocab = n_vocab
        self.n_target = n_traget
        self.word_embeddings = nn.Embedding(n_vocab, embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(embed_init))
        self.h1 = nn.Linear(self.embedding_dim, self.n_hidden)
        self.h2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.h3 = nn.Linear(2*self.n_hidden, self.n_hidden)
        self.h2traget = nn.Linear(self.n_hidden, self.n_target)

    def forward(self, x1, x2):    
        #embedding every sentence (average over words)
        embeds1 = [torch.mean(self.word_embeddings(s), dim=0) for s in x1]
        embeds2 = [torch.mean(self.word_embeddings(s), dim=0) for s in x2]
        
        #represent the batch as a 2-dim matrix (rater than list of tensors)
        embeds1 = torch.stack(embeds1)
        embeds2 = torch.stack(embeds2)
        
        #MLP of 2-layers for each embedded sentence
        o1 = torch.tanh(self.h1(embeds1))
        o2 = torch.tanh(self.h1(embeds2))
        o1 = torch.tanh(self.h2(o1))
        o2 = torch.tanh(self.h2(o2))
        
        #concat the representation of the two sentences togther
        #insert into a two layer MLP (without any activation in the end)
        o = torch.cat([o1, o2], dim=1)
        o = torch.tanh(self.h3(o))
        out = self.h2traget(o)

        return out


# In[ ]:


#the main train loop
def train(model, train_generator, val_generator,epochs=1, batch_size=32, lr=0.001, print_every=100):

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    counter = 0
    for e in range(epochs):
        for x1, x2, y in train_generator:
            counter += 1

            model.zero_grad()
            
            output = model(x1, x2)
            
            y = torch.LongTensor(y)
            if CUDA_ENABLED:
                y = y.cuda()
            loss = criterion(output, y)
            loss.backward()
            opt.step()
            
            if counter % print_every == 0:
                val_losses = []
                val_preds = []
                val_trues = []
                model.eval()
                for x1, x2, y in val_generator:                    

                    output = model(x1, x2)
                    val_preds.append(softmax(output, dim=1).cpu().detach().numpy())
                    val_trues.append(y)

                    y = torch.LongTensor(y)
                    if CUDA_ENABLED:
                        y = y.cuda()
                    val_loss = criterion(output, y)
                    val_losses.append(val_loss.item())
          
                
                model.train() 
                val_preds = np.concatenate(val_preds)
                val_trues = np.concatenate(val_trues)
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)),
                      "Val Acc: {:.4f}".format(accuracy_score(val_trues, np.argmax(val_preds,axis=1)))
                     )
            


# In[ ]:


model = TwoInputsMLPClassifier(n_vocab=embeddings.shape[0], embed_init=embeddings, embedding_dim=embeddings.shape[1], 
                               hidden_dim=100, n_traget=len(category_encoder.classes_))
if CUDA_ENABLED:
    model.cuda()


# In[ ]:


train(model, train_generator, val_generator)


# # Create predictions

# In[ ]:


test_preds = []
model.eval()
for x1, x2, in test_generator:                    
    output = model(x1, x2)
    test_preds.append(softmax(output, dim=1).cpu().detach().numpy())
test_preds = np.concatenate(test_preds)


# In[ ]:


test_preds.shape


# In[ ]:


final_preds = np.argmax(test_preds, axis=1)


# In[ ]:


final_preds = category_encoder.inverse_transform(final_preds)


# In[ ]:


test_df['Category'] = final_preds


# In[ ]:


test_df[['Id', 'Category']].to_csv('submission.csv', index=False)


# In[ ]:




