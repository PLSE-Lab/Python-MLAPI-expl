#!/usr/bin/env python
# coding: utf-8

# # Load Data From Internet to Your Notebook
# ## Most of the data is directly accessible via kaggle. But it can also be interesting to be able to download data directly from an internet link while remaining in your workspace (notebook). 
# 
# 
# Just to illustrate the case, we will be using the following link to access the ColA Dataset. This data contains sentences labeled as grammatically correct or not.   
# * Link of the dataset:  'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'      
# 
# Here are the libraries we will be using     
# * **wget**: to download the data to the kaggle instance file system       
# * **os (operating system)**: provides many functions to interact with the file system     
# * **pandas**: used to load our data as a dataframe once downloaded      

# In[ ]:


'''
Uncomment the instruction below to install wget, because it might not be directly available
'''
#!pip install wget


# In[ ]:


# import the modules
import wget 
import os  

import pandas as pd


# In[ ]:


url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'

# Download the file if it does not already exist
if not os.path.exists('./cola_public_1.1.zip'):
    wget.download(url, './cola_public_1.1.zip') 
    
# Unzip the dataset in case we don't have it yet
if not os.path.exists('./cola_public/'):
    get_ipython().system('unzip cola_public_1.1.zip')


# In[ ]:


colas_df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None)


# In[ ]:


colas_df.head()


# We can see that the dataset has not meaningful column names, so we can give the following names.

# In[ ]:


names=['sentence_source', 'label', 'label_notes', 'sentence']
colas_df.columns = names 


# In[ ]:


colas_df.head()


# Thats it !   
# **If you liked this kernel, please upvote**

# In[ ]:




