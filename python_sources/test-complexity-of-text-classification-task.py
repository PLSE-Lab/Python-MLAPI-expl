#!/usr/bin/env python
# coding: utf-8

# Complexity of any classification task depends on several factors such as 
# * number of training examples
# * dimensionality of features
# * number of target classes
# * class Balance -  Unbalanced classification problem poses a big challenge. Here, the number of observations of one target class is much less than the number of observations of other target class, 
# * data complexity - some pieces of text humans find difficult to comprehend than others.
# 
# You can refer this paper for more details - 
# https://arxiv.org/pdf/1811.01910.pdf
# 
# This paper describes an intuitive measure of difficulty for text classification datasets

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
df = train_df.sample(frac=0.02)
df.info()


# There are no missing values in this sample. In case of missing values, we need to fill NAs

# In[ ]:


# get_difficulty_report takes two arguments texts and their labels.
sents = df["comment_text"].values
df['toxic_score'] = df.iloc[:,2:-1].sum(axis=1)
labels = df['toxic_score'].values


# kaggle kernel comes loaded with most popular packages. But, you may need to install some custom packages. You can install packages using custom package installer present on the settings tab of your python notebook

# In[ ]:


#print difficulty report
from edm import report
print(report.get_difficulty_report(sents, labels))


# Here, you can see that difficulty of problem comes out to be good. So, this is a complex problem :). You can test this on simple task to verify.
