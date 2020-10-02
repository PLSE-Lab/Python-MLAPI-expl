#!/usr/bin/env python
# coding: utf-8

# # Creating a Spam and Not Spam Classifier with PyTorch

# > New to Pytorch Trying out stuff

# In[24]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

import nltk
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# ## Reading Data

# In[25]:


os.listdir('../input/')


# In[26]:


data = pd.read_csv('../input/spam_or_not_spam.csv')
data.head()


# ## Preprocessing Data

# #### Changing lables for ease of understanding

# In[27]:


data.dropna(inplace=True)
change_labels = lambda x: 1 if x==0 else 0
data['label'] = data['label'].apply(change_labels)
data.head()


# #### Let's Preprocess text data
# * We will remove non words, lower it, then Tokenize, Lemmatize and Vectorize and Remove Stopwords from the data

# In[28]:


remove_non_alphabets =lambda x: re.sub(r'[^a-zA-Z]',' ',x)


# In[29]:


tokenize = lambda x: word_tokenize(x)


# In[30]:


ps = PorterStemmer()
stem = lambda w: [ ps.stem(x) for x in w ]


# In[31]:


lemmatizer = WordNetLemmatizer()
leammtizer = lambda x: [ lemmatizer.lemmatize(word) for word in x ]


# In[32]:


print('Processing : [=', end='')
data['email'] = data['email'].apply(remove_non_alphabets)
print('=', end='')
data['email'] = data['email'].apply(tokenize) # [ word_tokenize(row) for row in data['email']]
print('=', end='')
data['email'] = data['email'].apply(stem)
print('=', end='')
data['email'] = data['email'].apply(leammtizer)
print('=', end='')
data['email'] = data['email'].apply(lambda x: ' '.join(x))
print('] : Completed', end='')
data.head()


# In[33]:


max_words = 10000
cv = CountVectorizer(max_features=max_words, stop_words='english')
sparse_matrix = cv.fit_transform(data['email']).toarray()


# In[34]:


sparse_matrix.shape


# In[35]:


x_train, x_test, y_train, y_test = train_test_split(sparse_matrix, np.array(data['label']))


# In[36]:


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(10000, 100)
        self.linear2 = nn.Linear(100, 10)
        self.linear3 = nn.Linear(10, 2)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


# In[37]:


model = LogisticRegression()


# In[38]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters() , lr=0.01)


# In[39]:


x_train = Variable(torch.from_numpy(x_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()


# In[40]:


epochs = 20
model.train()
loss_values = []
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss_values.append(loss.item())
    pred = torch.max(y_pred, 1)[1].eq(y_train).sum()
    acc = pred * 100.0 / len(x_train)
    print('Epoch: {}, Loss: {}, Accuracy: {}%'.format(epoch+1, loss.item(), acc.numpy()))
    loss.backward()
    optimizer.step()


# In[41]:


plt.plot(loss_values)
plt.title('Loss Value vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Loss'])
plt.show()


# # Testing

# In[42]:


x_test = Variable(torch.from_numpy(x_test)).float()
y_test = Variable(torch.from_numpy(y_test)).long()


# In[43]:


model.eval()
with torch.no_grad():
    y_pred = model(x_test)
    loss = criterion(y_pred, y_test)
    pred = torch.max(y_pred, 1)[1].eq(y_test).sum()
    print ("Accuracy : {}%".format(100*pred/len(x_test)))


# In[ ]:




