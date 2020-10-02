#!/usr/bin/env python
# coding: utf-8

# A Name Generator using simple rnn network

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# lets read names of male

# In[ ]:


df_male=pd.read_csv('../input/names/names/male.txt')


# As i have not sepcified the column name it assumed aamir to be column name. Thats fine we will also assume the same
# 

# In[ ]:


df_male['Aamir']=df_male['Aamir'].str.lower()


# lets create a text array which actually stores all are names 

# In[ ]:


text=df_male['Aamir'].to_numpy()


# As Its clear that all names are not of equal length , so lets make them of equal length by giving a padding of ' '.we would require length of longest string

# In[ ]:


maxlen = len(max(text, key=len))


# In[ ]:


for i in range(len(text)):
  while len(text[i])<maxlen:
      text[i] += ' '


# okay now lets create a dictionary. set helps us to only keep unique letters. 

# In[ ]:


chars=set(''.join(text))


# As Rnn can only deal with numeric data we will create a dictionary for characters to int and vice versa. Then we will create one hot vector for each character in each name

# In[ ]:


int2char = dict(enumerate(chars))


# In[ ]:


int2char


# In[ ]:


import torch
from torch import nn


# In[ ]:


device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# In[ ]:


char2int = {char: ind for ind, char in int2char.items()}


# In our model for name generation, we will be providing a intial letter (lets say 'p') and Model shall generate rest of word(like 'patrick').
# so input will be all letters of words except the last one.We dont need to input last letter as it will not have any prediction.similary we dont need first letter as output too.

# In[ ]:


# Creating lists that will hold our input and target sequences
input_seq = []
target_seq = []

for i in range(len(text)):
    # Remove last character for input sequence
  input_seq.append(text[i][:-1])
    
    # Remove first character for target sequence
  target_seq.append(text[i][1:])
  print("Input Sequence: {}\nTarget Sequence: {}".format(input_seq[i], target_seq[i]))


# In[ ]:


for i in range(len(text)):
    input_seq[i] = [char2int[character] for character in input_seq[i]]
    target_seq[i] = [char2int[character] for character in target_seq[i]]


# One Hotcoding the words

# In[ ]:


dict_size = len(char2int)
seq_len = maxlen - 1
batch_size = len(text)

def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features


# In[ ]:


input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)


# Our input seqeunce looks like this

# In[ ]:


input_seq[0]


# Import neccerary modules from pytorch

# In[ ]:


input_seq = torch.from_numpy(input_seq).to(device)
target_seq = torch.Tensor(target_seq).to(device)


# Create The model
# 
# it is a simple nn with a rnn and a fully connected layer. Note: RNN would require us to pass a initial state for hidden layer h_0 so we generate it using init_hidden function
# 

# In[ ]:





# In[ ]:


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)   
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size)

        out, hidden = self.rnn(x, hidden.to(device))
        
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        hidden.to(device)
        return hidden


# Create instance of model 

# In[ ]:


model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)

model.to(device)
# Define hyperparameters
n_epochs = 25000
lr=0.01

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[ ]:


X=input_seq
Y=target_seq


# In[ ]:


X_train,Y_train,X_test,Y_test=train_test_split(X,Y,test_size=0.2)


# Train the model. Longer you train better the results will be

# In[ ]:


int(X.shape[0]/mini_batch_size-1)


# In[ ]:


# Training Run
for epoch in range(1, n_epochs):
    mini_batch_size=100
    for m in range(int(X_train.shape[0]/mini_batch_size-1)):  
        optimizer.zero_grad() 
        X_train_mini=X[m*mini_batch_size:(m+1)*mini_batch_size]
        X_train_mini.to(device)
        Y_train_mini=Y[m*mini_batch_size:(m+1)*mini_batch_size]

        output, hidden = model(X_train_mini)
        loss = criterion(output, Y_train_mini.view(-1).long())
        loss.backward() 
        optimizer.step() 
    print('',end='.')
    if epoch%1000 == 0:
            print('\nEpoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))


# Predict the output. Note as we have used max it will only generate a particular word for a particular start word

# In[ ]:


def predict(model, character):
    # One-hot encoding our input to fit into the model
    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character, dict_size, character.shape[1], 1)
    character = torch.from_numpy(character).to(device)
    
    out, hidden = model(character)

    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()

    return int2char[char_ind], hidden


# In[ ]:


def sample(model, out_len, start='hey'):
    model.eval() # eval mode
    start = start.lower()
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for ii in range(size):
        char, h = predict(model, chars)
        chars.append(char)

    return ''.join(chars)


# In[ ]:


import string


# In[ ]:





# In[ ]:


alphabet_string=string.ascii_uppercase


# In[ ]:


a=torch.tensor([1,2,3,4,5])


# In[ ]:


for i in alphabet_string.lower():
    print(sample(model,15,i))


# In[ ]:





# In[ ]:


torch.save(model.state_dict(), '/kaggle/working/params.pt')


# In[ ]:




