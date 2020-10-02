#!/usr/bin/env python
# coding: utf-8

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
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

from wordcloud import WordCloud, STOPWORDS

df = pd.read_csv('../input/winemag-data_first150k.csv')
df = df[:100]
df.head()


# In[ ]:


df['description'][0]


# In order to find meaningful insights from the description it  needs to  be preprocessed.
# So, lets clean one of the description first and then carry on the same process for the rest of descriptions.
# 

# ### Get rid of the less useful parts like symbols and digits
# There are a lot of gibberish digit and symbols in the desciption like 100%, 2022-2030 etc. They can be removed using simple regex.

# In[ ]:


import re
description =  re.sub('[^a-zA-Z]',' ',df['description'][0])
description


# All the words should be in same case so lowercase the words

# In[ ]:


description = description.lower()

description


# ### Drop the stopwords
# The next step is to to remove the **stop words**. 
# Stop words are irrelevant as they occur frequently in the data example 'a', 'the','is','in' etc. In order to save both space and time, these words are dropped .

# In[ ]:


#convert string to a list of words
description_words = description.split() 
#iterate over each word and include it if it is not stopword 
description_words = [word for word in description_words if not word in stopwords.words('english')]

description_words


# ### Stemming words
# Stemming  reduce each word to its root form in order to remove the differences between inflected forms of a word. Example:  "running", "runs", "runned" become "run"

# In[ ]:


ps = PorterStemmer()
description_words=[ps.stem(word) for word in description_words]
description_words


# Now the description is clean the cleaned list of words can be converted to string and pushed to the dataset

# In[ ]:


df['description'][0]=' '.join(description_words)
df['description'][0]


# Now to clean other rows too one can  iterate over all rows of the dataset and clean each
# 

# In[ ]:


stopword_list = stopwords.words('english')
ps = PorterStemmer()
for i in range(1,len(df['description'])):
    description = re.sub('[^a-zA-Z]',' ',df['description'][i])
    description = description.lower()
    description_words = description.split()
    description_words = [word for word in description_words if not word in stopword_list]
    description_words = [ps.stem(word) for word in description_words]
    df['description'][i] = ' '.join(description_words)


# In[ ]:


df['description']

