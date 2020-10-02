#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings  
warnings.filterwarnings("ignore")   # ignore warnings

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **NATURAL LANGUAGE PROCESSING (NLP)**

# **In this kernel, natural language processing has been implemented.**

# * NLP is commonly used for text mining, machine translation, and automated question answering.
# * NLP is important because it helps resolve ambiguity in language and adds useful numeric structure to the data for many downstream applications, such as speech recognition or text analytics.

# In[ ]:


data = pd.read_csv('../input/Sheet_1.csv')
data.head()


# In[ ]:


data.info()


# In[ ]:


part1 = data.iloc[:,1:2]
part1


# In[ ]:


part2 = data.iloc[:,2:3]
part2


# In[ ]:


comment = pd.concat([part2,part1],axis =1,ignore_index =True) 
comment


# In[ ]:


comment.columns = ['Response', 'Class']
comment


# **Regular Expression**
# 
# Removing alphanumeric, spelling and punctuation characters

# In[ ]:


import re
result = re.sub('[^a-zA-Z]', ' ', comment['Response'][1])
result


# **Conversion all letters to lower case **

# In[ ]:


result = result.lower()
result


# **Splitting word by word**

# In[ ]:


result = result.split()
result


# **Cleaning the stop words**

# Loading stopwords

# In[ ]:


import nltk
from nltk.corpus import stopwords
stopwords_en = stopwords.words('english')
print(stopwords_en)


# In[ ]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# **Cleaning the stop words**
# 
# In here, we use set function because set is unordered and same element within set passes only one time.

# In[ ]:


result = [ps.stem(word) for word in result if not word in set(stopwords.words('english'))]
result


# **Let' s combine them**

# In[ ]:


result = ' '.join(result)
result


# **Making loop**

# In[ ]:


final_result = []
for i in range(80):
    result = re.sub('[^a-zA-Z]', ' ', comment['Response'][i])
    result = result.lower()
    result = result.split()
    result = [ps.stem(word) for word in result if not word in set(stopwords.words('english'))]
    result = ' '.join(result)
    final_result.append(result)


# In[ ]:


final_result


# ## CONCLUSION

# If you have any question or suggest, I will be happy to hear it.
# 
# **If you like it, please upvote :)**
