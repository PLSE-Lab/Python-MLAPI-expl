#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **CONVERTING SCORES INTO POSITIVE OR NEGATIVE REMARKS**

# In[ ]:


import sqlite3
import nltk
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas.io.sql as psql
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from nltk.stem.porter import PorterStemmer


# In[ ]:


show_tables = "select tbl_name from sqlite_master where type = 'table'"


# In[ ]:


food = pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')


# In[ ]:


food


# In[ ]:


conn=sqlite3.connect('../input/amazon-fine-food-reviews/database.sqlite')
#conn = sqlite3.connect('../input/japan-trade-statistics/ym_2017.db')


# In[ ]:


filtered_data=pd.read_sql_query("""SELECT*FROM Reviews WHERE Score !=3 """,conn)


# In[ ]:


def partition(x):
    if x<3 :
        return 'negative'
    return 'positive'


# In[ ]:


actual_score=filtered_data['Score']
positiveNegative=actual_score.map(partition)
filtered_data['Score']=positiveNegative


# In[ ]:


filtered_data.shape
filtered_data.head()


# **DATA CLEANING / DEDUPLICATION**
# 1.removing duplicates
# 

# In[ ]:


sorted_data=filtered_data.sort_values('ProductId',axis=0,ascending=True)


# In[ ]:


sorted_data


# In[ ]:


final=sorted_data.drop_duplicates(subset={'UserId', 'ProfileName','Time','Text'},keep='first',inplace=False)
final.shape


# In[ ]:


final


# In[ ]:


(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100


# 2.**HelpfulnessDenominator shoud always be greater than HelpfulnessNumerator**

# In[ ]:


display=pd.read_sql_query("""SELECT * FROM Reviews WHERE Score!=3 AND Id=44737 OR Id=64422 ORDER BY ProductID""",conn)
display


# In[ ]:


final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]


# In[ ]:


print(final.shape)


# In[ ]:


final['Score'].value_counts()


# In[ ]:




