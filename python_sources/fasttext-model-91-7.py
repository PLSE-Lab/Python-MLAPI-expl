#!/usr/bin/env python
# coding: utf-8

# 
# # FastText supervised model 91.7%
# ## One of the fastest and most accessible text classifier to anyone, without GPU
# FastText is well known for its distributed representation, which ultimately gets used as an embedding layer in a typical Deep Learning model such as a CNN or an LSTM. However, many don't know that FastText is also a supervised model. To prove the point, this Amazon dataset has been created to support the FastText format. And yet, 6 months later, no one has even tried to post a kernel for using FastText supervised model. What many also don't know is that, it is in fact a pretty good supervised model. Probably one of the fastest and the best out there without using a GPU. I'll cut straight to the chase and demonstrate how this is done. For a full writeup that's about to come soon, check out my blog post here:
# <br/> https://mungingdata.wordpress.com/
# 
# Also, a very accessible paper that introduces the viability of the FastText supervised model from the original authors [here](https://arxiv.org/pdf/1607.01759.pdf)
# 
# Ok so, lets begin. Its going to be fast trust me

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import fasttext
import bz2
import csv
from sklearn.metrics import roc_auc_score
import os
print(os.listdir("../input"))


# In[ ]:


# Load the training data 
data = bz2.BZ2File("../input/train.ft.txt.bz2")
data = data.readlines()
data = [x.decode('utf-8') for x in data]
print(len(data)) 


# In[ ]:


# 3.6mil rows! Lets inspect a few records to see the format and get a feel for the data
data[1:5]


# # Data prep and modelling
# A slight inconvenience with the FastText model is the need to save the dataset into a text file. And the annoying encoding of the "____label__ ____#__". Basically, the target and the text is all in the same cell. They are distinguished by the prefix of '____label__ ____#__'. Lets say if have 2 labels and one is 'Ham' and the other 'Spam', then your labels would be '____label__ ____Ham__' and '____label__ ____Spam__'. You can include as many labels as well, not just 2.   
# 
# Thankfully, this dataset has already been formated in that way as you can see from the first 5 records I printed out. We just need to write it out to disk. 

# In[ ]:


# Data Prep
data = pd.DataFrame(data)
data.to_csv("train.txt", index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")

# Modelling
# This routine takes about 5 to 10 minutes 
model = fasttext.train_supervised('train.txt',label_prefix='__label__', thread=4, epoch = 10)
print(model.labels, 'are the labels or targets the model is predicting')


# # Apply predictions
# Ok after about 10 minutes or so, the model is finished. Now lets apply the predictions to the test dataset. Thankfully, we don't have to write out a physical text file to do the prediction. You could if you want to, but I'm just going to use the data object

# In[ ]:


# Load the test data 
test = bz2.BZ2File("../input/test.ft.txt.bz2")
test = test.readlines()
test = [x.decode('utf-8') for x in test]
print(len(test), 'number of records in the test set') 

# To run the predict function, we need to remove the __label__1 and __label__2 from the testset.  
new = [w.replace('__label__2 ', '') for w in test]
new = [w.replace('__label__1 ', '') for w in new]
new = [w.replace('\n', '') for w in new]

# Use the predict function 
pred = model.predict(new)

# check the first record outputs
print(pred[0][0], 'is the predicted label')
print(pred[0][1], 'is the probability score')


# # Evaluation 
# Ok so we have our predictions, now lets measure how well we have done? 

# In[ ]:


# Lets recode the actual targets to 1's and 0's from both the test set and the actual predictions  
labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test]
pred_labels = [0 if x == ['__label__1'] else 1 for x in pred[0]]

# run the accuracy measure. 
print(roc_auc_score(labels, pred_labels))


# ## 91.7%
# 91.7% absolute accuracy score with only just a few lines of code. Running the evaluation metric using the Probability score would yeild even higher scores but I wanted to keep it inline with the rest of the kernels so its a fair comparison. The most popular Kernel here is the CuDNNLSTM which yielded 93.7%
# 
# Perhaps the most challenging bit about using FastText is just the slightly annoying data preparation step to encode the '__labels__'. Just like any data science projects, data prep is the hard yard. Otherwise, rest is pretty straight foward. I'll post another kernel on a different dataset in future and run through the processing steps to get the dataset into the correct format. And some other model tuning process.
# 
# If you like this kernel please give me an upvote! 
