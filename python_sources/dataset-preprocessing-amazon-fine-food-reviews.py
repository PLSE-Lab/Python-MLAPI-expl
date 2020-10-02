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
import numpy as np
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# #### The Amazon Fine food reviews dataframe has the following contents:
# Features : 
# Id : Review ID,  'ProductId' : product ID,  'UserId' : , 'ProfileName : customer name', 'HelpfulnessNumerator : people found helpful', 'HelpfulnessDenominator : people indicated whether the found helpful', 'Score : star rating', 'Time : time of review', 'Summary : breif summary of the report ', 'Text : original review'
# 
# #### Objective : To predict whether a review new is positive or not.

# In[ ]:


conn = sqlite3.connect('../input/database.sqlite')

df = pd.read_sql ("""SELECT * from Reviews WHERE Score != 3""", conn)

df['P/N'] = np.select((df['Score'] > 3, df['Score'] < 3), ('Positive', 'Negative'))
df = df.drop(columns = 'Score')
raw_df_shape = df.shape
print(raw_df_shape)


# #### Data Cleaning : deduplication

# In[ ]:


df = df.sort_values('ProductId', axis = 0, ascending = True)
df = df.drop_duplicates(subset = {'UserId', 'ProfileName', 'Time', 'Text'}, keep = 'first', inplace = False)
# helpfulness denominator should be greater than helpfulness numerator

df = df[df.HelpfulnessDenominator >= df.HelpfulnessNumerator]
clean_df_shape = df.shape
print(clean_df_shape)


# #### Checking how much data still remains

# In[ ]:


print('Data points still remaining : ', clean_df_shape[0]*100/raw_df_shape[0], '%')


# #### Number of positive and negative reviews

# In[ ]:


df['P/N'].value_counts()
df = df.reset_index().drop('index', axis = 1)
df.head(2).T


# #### Test Preprocessing ( Removing HTML Tags, punctuations, check for alphanumerics, check if length of words is less than or equal to 2, convert word into lowercase and snowball stemming.   

# In[ ]:


"""Text processing taken into consideration step wise
1. Removing HTML tags
2. Removing punctuations
3. Removing stopwords
4. Tokenizing sentences into words
5. Checking if size of words is greater than 1
6. Snowball Stemming 
"""
import re
import timeit
import numpy as np
import time
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

stopwords = set(stopwords.words('english'))
snowballstemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()


t0 = time.time()
processed_positive_words_list = []
processed_negative_words_list = []
processed_sentence_list = []

for i in df.index:
    this_sentence = []
    stemmedwords = []
    text = df.iloc[i]['Text']
    
    #Remove HTML Tags
    text = re.sub('<.*?>', ' ', text) 
    
    #Clear punctuation and numbers
    text = re.sub('[^A-Za-z]+', ' ', text)
    
    #Convert all uppercase characters into lowercase
    text = text.lower()
    #Tokenize string
    #removing stopwords
    #stemming words
    #checking wordlength
    for words in word_tokenize(text):
        if len(words) > 1 and words not in stopwords:
            stemmedwords.append(snowballstemmer.stem(words))
            
    if df.iloc[i]['P/N'] == 'Positive':
        processed_positive_words_list+=(stemmedwords)
    elif df.iloc[i]['P/N'] == 'Negative':                
        processed_negative_words_list+=(stemmedwords)
                
    #Joining words
    clean_sentence = " ".join(stemmedwords)
    processed_sentence_list.append(clean_sentence)

    print((i*100/364159),end ="\r")
    
t1 = time.time()
print('time elapsed',t1-t0, 'secs')


# In[ ]:


processed_all_words = processed_positive_words_list + processed_negative_words_list
print("Total number of words in processed_words_list : ", len(processed_all_words))
print("Total number of sentences in preprocessed_sentence_list : ", len(processed_sentence_list))
print("Total number of positive words : ", len(processed_positive_words_list))
print("Total number of negative words : ", len(processed_negative_words_list))


# In[ ]:


#type(processed_sentence_list)
df['Cleaned_Text'] = processed_sentence_list
df.head(2).T


# In[ ]:


processed_positive_words_list[:10]


# In[ ]:


processed_negative_words_list[:10]


# ### Storing final table to sql for future 

# In[ ]:


# store final table into an SQlLite table for future.
try:
    os.remove('amazon_review_df.sqlite')
except:
    print('Exception : file not exist')
conn = sqlite3.connect('amazon_review_df.sqlite')
c=conn.cursor()
conn.text_factory = str
df.to_sql('Reviews', conn,  schema=None, if_exists='replace', index=True, index_label=None, chunksize=None, dtype=None)


# In[ ]:


# store processed positive, negative and sentance list into an SQlLite table for future.
try:
    os.remove('positive_words.txt')
except:
    print('Exception : file not exist')
try:
    os.remove('negative_words.txt')
except:
    print('file not exist')

with open('positive_words.txt', 'w') as f:
    for x in processed_positive_words_list:
        f.write(x)
        f.write(',')
    f.close()

with open('negative_words.txt', 'w') as f:
    for x in processed_negative_words_list:
        f.write(x)
        f.write(',')
    f.close()


# ### Read from my_amazon_review_df.sqlite

# In[ ]:


conn = sqlite3.connect('amazon_review_df.sqlite')
display= pd.read_sql_query("""
SELECT * from Reviews""", conn)
display.head(2).T


# In[ ]:


processed_positive_words_list = []
processed_negative_words_list = []

with open('positive_words.txt', 'r') as f:
    for line in f.readlines():
        line = line.replace("\"", "")
    processed_positive_words_list = line.split(',')
with open('negative_words.txt', 'r') as f:
    for line in f.readlines():
        line = line.replace("\"", "")
    processed_negative_words_list = line.split(',')


# In[ ]:


processed_positive_words_list[:10]


# In[ ]:


processed_negative_words_list[:10]


# ### BoW Vectorization
# ### Finding the rare words
# 
# Motivation : Exclude the typos and spelling mistakes.

# In[ ]:


words_counts = pd.Series(processed_all_words).value_counts()
rare_words = list(words_counts.where(words_counts.values == 1).dropna().index)


# In[ ]:


print('No of rare words : ', len(rare_words))
print('No of unique words : ', len(words_counts))
#print(words_counts.index)
words_counts


# ### Rare Words Removal

# In[ ]:


stopwords = frozenset(rare_words)
count_vect = CountVectorizer(stop_words=stopwords)
final_counts = count_vect.fit_transform(df['Cleaned_Text'].values)
print("the shape of BOW vectorizer after rare words removal",final_counts.get_shape())
print("the number of unique words after rare words removal", final_counts.get_shape()[1])


# ### T-SNE Visualization: 

# In[ ]:


# features = final_counts
# labels = df['P/N']

tsne_model = TSNE(n_components=2, random_state=0)
t0 = time.time()
tsne_transform = tsne_model.fit_transform(final_counts.todense()).T
print('Elapsed time :', time.time()-t0);


# In[ ]:


# visualizing t-SNE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

print(tsne_transform.shape)

tsne_data = np.vstack((tsne_transform, display['P/N'])).T
tsne_df = pd.DataFrame(data = tsne_data, columns=('Dimention 1', 'Dimention 2', 'Label'))
sns.FacetGrid(tsne_df, size = 8, hue = 'Label').map(plt.scatter, 'Dimention 1', 'Dimention 2', marker = '.').add_legend()


# In[ ]:





# In[ ]:





# In[ ]:




