#!/usr/bin/env python
# coding: utf-8

# <center>
# # ** EDA : Eye Disorder Datset** 
# <br>
# ![](https://i.imgur.com/ezgY6qG.jpg)
# 

# ## **Highlights**
# * Manully Cleaned
# * Balanced data
# * Three classes
# * GrayScale
# * Pixel values

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def load_dataset(path):
    '''
        Loading Dataset
    '''
    return pd.read_csv(path)


# In[ ]:


def preprocess(dataset):
    '''
        Preprocssing the Dataset
    '''
    Y = dataset["Type"]
    X = dataset.drop(['Type'],axis=1)
    X = X.values.reshape(-1,151,332,1)
    X = X/255.0
    
    return X,Y


# In[ ]:


def display(n,label):
    '''
        Displaying images in grid of 1xn
    '''
    fig = plt.figure(figsize=(20,20))
    label_index = np.where(np.array(Y) == label)
    for index in range(n):
        i = label_index[0][index]
        ax = fig.add_subplot(1, n, index+1, xticks=[], yticks=[])
        ax.imshow(X[i].reshape(151,332), cmap='gray')
        ax.set_title(str(Y[i]))


# In[ ]:


dataset_path = "../input/eye-disorder-dataset/eye_dataset.csv"
dataset = load_dataset(dataset_path)
X,Y = preprocess(dataset)


# In[ ]:


print("Shape of X : ",X.shape)
print("Shape of Y : ",Y.shape)
print("Shape of Image : ", X[0].shape)


# ### **Distribution of eye disorders in dataset**

# In[ ]:


sns.countplot(Y)


# ### **Displaying 5 images each of different Eye disorders**

# In[ ]:


display(5,"cat")
display(5,"crossed")
display(5,"bulk")


# I know this dataset is not that big and many of you might be thinking which Deep learning architecture to use for training on such small datasets. They can use this dataset for training a Siamese Neural network which is based on Few-Shot Learning Technique.  Few-shot learning refers to the practice of feeding a learning model with a very small amount of training data.
# <br>
# I'll be publishing a kernel very soon on how to train a siamese neural network based on the eye disorder dataset.
