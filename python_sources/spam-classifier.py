#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[ ]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import random
import tqdm


# # Importing the dataset

# In[ ]:


dataset = pd.read_csv('../input/sms_spam.csv')


# In[237]:


dataset.head()


# # Encoding outputs

# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
types = label_encoder.fit_transform(dataset['type'])


# # Splitting the dataset

# In[ ]:


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(dataset['text'], types, test_size = 0.1, random_state = 1 )


# In[ ]:


train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1, random_state = 1)


# In[239]:


len(train_x), len(test_x), len(val_x)


# # Defining Model

# In[ ]:


class SpamClassifier(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, batch_size):
        super(SpamClassifier, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, embeddings, hidden):
        output, hidden = self.lstm(embeddings.view(len(inputs), self.batch_size, -1))
        output = self.linear(output)
        output = self.sigmoid(output)
        
        return output, hidden
        
    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_size), torch.zeros(1, self.batch_size, self.hidden_size))
        


# # Preparing data

# In[ ]:


nlp = spacy.load('en_core_web_lg')


# In[ ]:


processed_text = []
for course in dataset['text']:
    doc = nlp(course.lower())
    words = [word.lemma_ for word in doc if not word.is_punct | word.is_space | word.is_stop]
    processed_text.append(words)


# In[240]:


print(len(processed_text))
for i in range(5):
    print(processed_text[i])


# #### Creating sequences

# In[ ]:


def seq(sent):
    inputs = []
    for word in sent:
        inputs.append(list(nlp(word).vector))
        
    return inputs


# # Training

# In[ ]:


EMBEDDING_SIZE = 300
HIDDEN_SIZE = 100
OUTPUT_SIZE = 1
BATCH_SIZE = 1

model = SpamClassifier(EMBEDDING_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE)


# In[241]:


model


# In[ ]:


lr = 1e-3
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr)


# In[ ]:


with torch.no_grad():
    inputs = seq(processed_text[0])
    inputs = torch.tensor(inputs, dtype=torch.float)
    hidden = model.init_hidden()
    outputs, hidden = model(inputs,hidden)
    print(outputs[-1])


# In[ ]:


train_loss = []
val_loss = []


# In[ ]:


EPOCHS = 1
hidden = model.init_hidden()

for epoch in range(EPOCHS):
    for i in range(len(train_x)):
        inputs = seq(processed_text[i])
        if len(inputs) >= 1:
            inputs = torch.tensor(inputs, dtype=torch.float)       
            target = types[i]
            target = torch.tensor(target, dtype=torch.float)
            outputs, hidden = model(inputs, hidden)
            loss = loss_fn(outputs[-1].squeeze(), target)
            train_loss.append(loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()
    print(loss.item())
    for i in range()


# In[ ]:


y_pred = []
y_actual = []
ids = []
test_processed = processed_text[-len(test_x):]
test_types = types[-len(test_x):]
for i in range(500):
    inputs = seq(test_processed[i])
    inputs = torch.tensor(inputs, dtype=torch.float)
    targets = test_types[i]
    targets = torch.tensor(targets, dtype=torch.float)
    outputs, h = model(inputs, hidden)
    y_pred.append(outputs[-1].squeeze().item())
    y_actual.append(targets)
    


# In[ ]:


y_pred = [1 if i>=0.5 else 0 for i in y_pred]


# In[236]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred, y_actual)


# In[ ]:




