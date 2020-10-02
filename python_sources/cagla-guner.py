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


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


yelp_df = pd.read_csv("../input/yelp-data-set/yelp.csv")
yelp_df.head(7)


# In[ ]:


yelp_df.tail()


# In[ ]:


yelp_df.describe()


# In[ ]:


yelp_df['text'][0]


# In[ ]:


yelp_df['text'][1]


# In[ ]:


yelp_df['text'][9998]


# In[ ]:


yelp_df['text'][9999]


# In[ ]:


yelp_df['length']= yelp_df['text'].apply(len)


# In[ ]:


##print (yelp_df['length'])


# In[ ]:


yelp_df


# In[ ]:


yelp_df['length'].plot(bins = 100, kind = 'hist')


# In[ ]:


yelp_df.length.describe()


# In[ ]:


yelp_df[yelp_df['length']==4997]['text'].iloc[0] ##finding the longest text


# In[ ]:


sns.countplot(y='stars', data = yelp_df)


# In[ ]:


g= sns.FacetGrid(data=yelp_df, col= 'stars', col_wrap=3)
g.map(plt.hist, 'length', bins= 20, color= 'purple')


# In[ ]:


yelp_df_1 = yelp_df[yelp_df['stars']== 1] # Listing the reviews with 1 stars
yelp_df_1


# In[ ]:


yelp_df_5 = yelp_df[yelp_df['stars']== 5] # Listing the reviews with 5 stars


# In[ ]:


yelp_df_5


# In[ ]:


yelp_df_1_5=pd.concat([yelp_df_1, yelp_df_5])


# In[ ]:


yelp_df_1_5


# In[ ]:


yelp_df_1_5.info()


# In[ ]:


print('1-Stars Review Percentage - ', (len(yelp_df_1)/len(yelp_df_1_5))*100, '%') # prints the percentage of 1star reviews among the total of 1 and 5star reviews.


# In[ ]:


sns.countplot(yelp_df_1_5['stars'], label='Count') #prints a bar chart of count versus stars


# In[ ]:


import string #imports a list of punctuation marks.
string.punctuation # prints the list of punctuation marks


# In[ ]:


Test= "Hello, Are you enjoying the AI course?!!" # a text to make sure the punctuation marks are removed


# In[ ]:


Test


# In[ ]:


Test_punc_removed= [char for char in Test if char not in string.punctuation] #removes the punctuation marks from the test text: print the character if it is not listed among punctuation marks
Test_punc_removed #prints each single item to remove p. marks one by one


# In[ ]:


Test_punc_removed_join = ''.join(Test_punc_removed)# Joins the text back into its shape without the punctuation marks


# In[ ]:


Test_punc_removed_join


# In[ ]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[ ]:


Test_punc_removed_join_clean = [ word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]


# In[ ]:


Test_punc_removed_join_clean


# In[ ]:


Test2 = 'Here is the Assignment#8, for you to think on how to remove stopwords and punctuations!'


# In[ ]:


from nltk.corpus import stopwords
stopwords.words('english')
Test_punc_removed= [char for char in Test2 if char not in string.punctuation]
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english') ]
Test_punch_removed_join_second = ''.join(Test_punc_removed_join_clean)


# In[ ]:


Test_punch_removed_join_second

