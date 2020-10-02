#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# from google.colab import drive
# drive.mount('/content/drive/')

import re


# The idea is to create a minimal RNN in Python/numpy that will provide a baseline model for more complex algorithms,to gain a low level understanding of the working of RNN.
# 
# This kernel was inspired by
# 1. Andrej Karpathy https://gist.github.com/karpathy/d4dee566867f8291f086: Minimal character-level language model with a Vanilla Recurrent Neural Network, in Python/numpy. And the blog http://karpathy.github.io/2015/05/21/rnn-effectiveness/.
# 2. The deep learning book by Michael Nielsen particularly  http://neuralnetworksanddeeplearning.com/chap6.html
# 3. Andrew ng Deep learning course (Course 5) on Coursera
# 

# The approach can broadly classified in following step:
# 1. Creating a dictionary of words and indexing to be later used in encoding each words into a vector using 1-of-k encoding 
# 2. Initialize the RNN model parameters
# 3. Feedforward the training tweet (vectorized form) into the network and calculate loss for that training example
# 4. Backpropagate through time and obtain the gradient of the parameters 
# 5. Clip the gradients to avoid exploding gradient problem
# 6. Choose a learning rate and calculate the new model parameters
# 7. Repeat 3-6 for some number of iterations for all the training examples

# In[ ]:


train_df  = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')


# In[ ]:


train_df.head()


# In[ ]:


# A negative and a positive text for a disaster
train_df = train_df.sample(frac=1)
print(train_df[train_df['target']==0]['text'].values[0])
print(train_df[train_df['target']==1]['text'].values[0])


# In[ ]:


df = pd.concat([train_df, test_df])

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

df['text']=df['text'].apply(lambda x : remove_URL(x))


# In[ ]:


def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

df['text']=df['text'].apply(lambda x : remove_html(x))


# In[ ]:


def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

df['text']=df['text'].apply(lambda x: remove_emoji(x))


# In[ ]:


import string
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

df['text']=df['text'].apply(lambda x : remove_punct(x))


# In[ ]:


"""Preprocessing and 
creating a dictionary that has an index for each of the unique words"""

df['text'] = df['text'].str.lower()

# # removing all the words starting with http and @ 
# df['text'] = df['text'].map(lambda x: (' '.join(word for word in x.split(' ') if not word.startswith(('http','@')))))

# removing any non-alphanumeric characters
df['text']= df['text'].str.replace('[^a-z A-Z]', '')

# separating the training, validation and testing set

train_df = df.iloc[:7613]
test_df = df.iloc[7613:]


words = list(train_df['text'].str.split(' ', expand=True).stack().unique()) # getting the list of unique words
vocabulary_size = len(words)

print('The number of unique words is:%d '%(vocabulary_size))
print('The total_number of words is : %d'%(len(list(train_df['text'].str.split(' ', expand=True).stack()))))

# creating a dictionary for indexing the words
words_idx = { word:i for i, word in enumerate(words) }
train_df.head()
# words_idx


# In[ ]:


"""Converting a single training example into the index retrieved from the dictionary """
example = train_df['text'].str.split().values[1]
inputs = [words_idx[i] for i in example]
targets = train_df['target'].values[1]
print(example)
print(inputs)
print(targets)


# In[ ]:


# hyperparameters
learning_rate = 0.005
n_h = hidden_size = 100
n_x = vocabulary_size
n_y = 2

# model_parameters 
Whh = np.random.randn(hidden_size, hidden_size)*0.1
Whx = np.random.randn(hidden_size, vocabulary_size)*0.1
Wyh = np.random.randn(2, hidden_size) *0.1
by  = np.zeros((n_y,1))
bh  = np.zeros((n_h,1))

# """loading the saved model"""
# import pickle
# filename = '/kaggle/input/pkl-model/rnn_model_v2.pkl'
# with open(filename, "rb") as f:
#     Whh, Whx, bh, by, Wyh  = pickle.load(f)

copy_df = train_df
train_df = train_df.iloc[:7000]
validation_df = copy_df.iloc[7000:]


print('The training set examples: %d' %(len(train_df)))
print('The validation set examples: %d' %(len(validation_df)))


# In[ ]:


def feedforward(inputs):  # takes in the index of words in a example tweet and return the prediction 
    
    xs,hs = [], np.zeros((n_h,1))
    for t in range(len(inputs)):
        
        xs.append(np.zeros((n_x,1)))  # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        
        hs = np.tanh(np.dot(Whh, hs) + np.dot(Whx, xs[t]) + bh)  # hidden state
        
    ys = np.dot(Wyh,hs) +by   # unnormalized log probabilities for next chars
    ps = np.exp(ys)/ np.sum(np.exp(ys),axis = 0)  # softmax probabiltity of non -disaster / disaster tweets
    
    prediction = np.argmax(ps)
    
    return prediction


# In[ ]:


"""Generating a function that takes in one training example, feedforward into the network, calculate the cost which is the function of 
predicted y vs actual y. Then we perform a backward propagation and find the gradient of all the parameters and return it"""

def loss_func(inputs, targets):
    
    # input is the list of index in a training example (shape= (1,T_x))
    # targets (0 or 1) for a training example[0,1]
    
    xs, hs =[],[]   # creating a cache of xs, hs for each unit of propagation
    hs.append(np.zeros((n_h,1)))
    loss = 0
# feedforward propagation
    
    for t in range(len(inputs)):
        
        xs.append(np.zeros((n_x,1)))
        xs[t][inputs[t]] = 1
        
        hs.append(np.tanh(np.dot(Whh, hs[t]) + np.dot(Whx, xs[t]) + bh))
        
    ys = np.dot(Wyh,hs[-1]) + by
    ps = np.exp(ys)/ np.sum(np.exp(ys),axis = 0)
    
# cost 
    y = np.zeros((2,1))
    y[targets] =  1
    loss = -np.log(np.sum(ps * y,axis = 0)) # cross_entropy loss
    
# backward_propagation through time 
    """gradient of cost with respect to model parameters """
    dWhh, dWyh, dWhx = np.zeros_like(Whh), np.zeros_like(Wyh), np.zeros_like(Whx)
    dby, dbh = np.zeros_like(by), np.zeros_like(bh)
    
    dy = ps-y
    dWyh = np.dot(dy,hs[-1].transpose())
    dby = np.copy(dy)
    
    dh = np.dot(Wyh.transpose(),dy)  
    dh_raw = (1- hs[-1]*hs[-1]) * dh
    
    dWhx = np.dot(dh_raw, xs[-1].transpose())
    dWhh = np.dot(dh_raw, hs[-2].transpose())
    dbh  = np.copy(dh_raw)
    dh_next = np.dot(Whh.transpose(),dh_raw)

    for t in reversed(range(len(inputs)-2)):
        
        dh = np.copy(dh_next)
        dh_raw = (1- hs[t+1]*hs[t+1]) * dh
        
        dWhx += np.dot(dh_raw, xs[t].transpose())
        dWhh += np.dot(dh_raw, hs[t].transpose())
        dbh  += np.copy(dh_raw)
        dh_next = np.dot(Whh.transpose(),dh_raw)

    for dparams in [dWhh, dWhx, dbh, dby, dWyh]: # clipping to avoid exploding gradients
        np.clip(dparams, -5, 5 , out = dparams) 

    return loss, dWhh, dWhx, dbh, dby, dWyh 


# In[ ]:


"""Feeding into the network to retrive the gradient and using Adagrad optimizer to perform the gradient descent.
Then we repeat this for all the training examples and for n epochs."""

num_iterations = 21000

mWhh, mWyh, mWhx = np.zeros_like(Whh), np.zeros_like(Wyh), np.zeros_like(Whx)
mby, mbh = np.zeros_like(by), np.zeros_like(bh)                                   # memory variables for Adagrad

for j in range(num_iterations):
    
    idx = j% len(train_df)
    example = train_df['text'].str.split().values[idx]
    inputs = [words_idx[i] for i in example]
    targets = int(train_df['target'].values[idx])
    
    loss, dWhh, dWhx, dbh, dby, dWyh = loss_func(inputs, targets)
    
    
    # Adagrad optimizer  
    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Whx, Whh, Wyh, bh, by], 
                                [dWhx, dWhh, dWyh, dbh, dby], 
                                [mWhx, mWhh, mWyh, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

    # validation accuracy
    # using for loop instead of vectorization
    if j % 700 == 0:
      predictions = []
      count=0
      actual_targets= validation_df['target'].tolist()
      for i in range(len(validation_df)):
          example = validation_df['text'].str.split().values[i]
          inputs = [words_idx[l] for l in example]
          predictions.append(feedforward(inputs))
          
      for y, y_hat in zip(actual_targets, predictions):
          if y==y_hat:
              count+=1
      print('The validation_accuracy after iterations:%d is %d'%(j,(count/len(validation_df))*100))

    #  training accuracy
      
      # predictions = []
      # count = 0
      # actual_targets = train_df['target'].tolist()

      
      # for i in range(len(train_df)):
          
      #     example = train_df['text'].str.split().values[i]
      #     inputs = [words_idx[l] for l in example]
      #     predictions.append(feedforward(inputs))
          
      # for y, y_hat in zip(actual_targets, predictions):
      #     if y==y_hat:
      #         count+=1
              
      # print('The training_accuracy after iterations:%d is %d'%(j,(count/len(train_df))*100))
   


# In[ ]:


# predictions in the test set 
test_predictions  = []
for i in range(len(test_df)):
    example = test_df['text'].str.split().values[i]
    inputs = []
    for l in example:
        if l in words_idx:
            inputs.append(words_idx[l])
            
    test_predictions.append(feedforward(inputs))
    
test_df['target'] = test_predictions
test_df = test_df[['id','target']].set_index('id')
test_df.to_csv('submission.csv')


# In[ ]:


# saving the model
import pickle
filename = 'rnn_model_v2.pkl'

with open(filename, "wb") as f:
    pickle.dump((Whh, Whx, bh, by, Wyh ), f)



# The testing accuracy arrives at 76% (better than random guessing). I have not experimented with any hyperparameters or high level data-cleaning. Still, it seems that our RNN is learning association between words that helps it to classify the tweets.
# 
# We see there is a high bias. We may try 
# 1. different network architectures 
# 2. train for longer iteration

# In[ ]:




