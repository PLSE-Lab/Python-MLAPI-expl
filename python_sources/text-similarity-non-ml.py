#!/usr/bin/env python
# coding: utf-8

# # This notebook is try to NOT use Machine Learning library to classify same_secruity but using simple text similarity to classify. Just want to give some alternative way to tackle the tasks. I still believe that leveraging machine learning library should get a higher accuracy.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from IPython.display import display, HTML
import matplotlib.pyplot as plt


# In[ ]:


df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")

df_train = df_train.rename(columns = {'Unnamed: 0':'train_id'})

print('df_train:', df_train.shape)
display(df_train.head(5))

print('df_test:', df_test.shape)
display(df_test.head(5))


# ## Function for visualize result

# In[ ]:


def cal_correct(df_temp, col, threshold):
    df_temp[col + '_predicted'] = df_temp[col] >= threshold
    df_temp[col + '_correct'] = df_temp[col + '_predicted'] == df_result['same_security']
    
    df_group = df_temp.groupby(df_temp[col + '_correct']).size().reset_index()
    df_group.columns = ['correct', 'cnt']

    plt.pie(df_group['cnt'].tolist(), labels=df_group['correct'].tolist(), autopct='%1.1f%%')
    plt.show()    


# In[ ]:


thresholds = [0.4, 0.5, 0.6]


# ## Using Cosine Similarity to `check` whether it is same security

# In[ ]:


import re
from sklearn.feature_extraction import text
vectorizer = text.TfidfVectorizer()

def cosine_sim(test1, test2):
    tfidf = vectorizer.fit_transform([test1, test2])
    result = ((tfidf * tfidf.T).A)[0,1]
    return result
    
def cosine_sim_df(df_data):
    col1 = 'description_x'
    col2 = 'description_y'
    df_data[col1] = df_data[col1].str.replace(r'\d', '')
    df_data[col2] = df_data[col2].str.replace(r'\d', '')
    
    df_data['cos_sim'] = 0
    df_data['cos_sim'] = df_data.apply(
        lambda x: cosine_sim(x[col1], x[col2]), axis=1)
    return df_data


# In[ ]:


df_result = cosine_sim_df(df_train)
display(df_result.head(3))

for x in thresholds:
    print('threshold:', x)
    cal_correct(df_result, 'cos_sim', x)


# In[ ]:


axis('equal')
pie(sums, labels=sums.index)
show()


# ## Using SequenceMatcher to `check` whether it is same security

# In[ ]:


from difflib import SequenceMatcher

def seq_match(test1, test2):
    return SequenceMatcher(None, test1, test2).ratio()
    
def seq_match_df(df_data):
    col1 = 'description_x'
    col2 = 'description_y'
    df_data[col1] = df_data[col1].str.replace(r'\d', '')
    df_data[col2] = df_data[col2].str.replace(r'\d', '')
    
    df_data['seq_match'] = 0
    df_data['seq_match'] = df_data.apply(
        lambda x: seq_match(x[col1], x[col2]), axis=1)
    
    return df_data


# In[ ]:


df_result = seq_match_df(df_train)
display(df_result.head(3))

for x in thresholds:
    print('threshold:', x)
    cal_correct(df_result, 'seq_match', x)


# # Since the result of sequence matcher method is better. I will use it as classification engine

# In[ ]:


df_result = seq_match_df(df_test )
display(df_result.head(3))

df_result['same_security'] = df_result['seq_match'] > 0.5    
df_group = df_result.groupby('same_security').size().reset_index()
df_group.columns = ['correct', 'cnt']

plt.pie(df_group['cnt'].tolist(), labels=df_group['correct'].tolist(), autopct='%1.1f%%')
plt.show()    

