#!/usr/bin/env python
# coding: utf-8

# ### **Importing libraries**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.wordnet import WordNetLemmatizer
stop_words =set( stopwords.words('english') )
import string
punc=set(string.punctuation)
lemm=WordNetLemmatizer()
lemmatizer = WordNetLemmatizer()


# ### ** Loading datafile**

# In[ ]:


df=pd.read_csv("../input/urban_dictionary.csv")


# ### Header

# In[ ]:


df.head()


# ### Number of rows and columns

# In[ ]:


df.shape


# **INFO**

# In[ ]:


df.info()


# #### ****Removing unwanted columns****

# In[ ]:


df.drop(['date','tags'],axis=1,inplace=True)


# In[ ]:


df.dtypes


# ### **Finding hightest contribution _ authors**

# In[ ]:


df['author'].value_counts().head()


# ### **word frequencies**

# In[ ]:


df['word'].value_counts().head()


# In[ ]:


df.columns


# ### **Word count**

# In[ ]:


df['original_count']=df.definition.apply(lambda x:len(x.split()))


# In[ ]:


df.head()


# ## **Most positive word and negative word **

# In[ ]:


print('positive word',df.loc[df.up==df.up.max()]['word'])
print('Negative  word',df.loc[df.down==df.down.max()]['word'])


# In[ ]:





# ### **Pre proccessing**

# In[ ]:


df['def_new']=None
for i in range(len(df['definition'])):
    doc=df.definition[i]
    doc=doc.split(" ")
    doc=[w for w in doc if w not in set(stop_words)]
    doc=[w for w in doc if w not in punc]
    doc=" ".join([lemmatizer.lemmatize(word) for word in doc])
    df.at[i,'def_new']=doc

    


# In[ ]:


df.head()


# #### **Word counts after preproccessing**

# In[ ]:


df['def_new_count']=df['def_new'].apply(lambda x:len(x.split(" ")))


# ### **Visualization of counts of word -before and after preproccessing **

# In[ ]:


df[['def_new_count','original_count']].head()


# ### **Sentiment Analysis**

# In[ ]:


sm=SentimentIntensityAnalyzer()


# In[ ]:


df['score']=None
df['polarity']=None
for i in range(len(df.def_new)):
    score_dic=sm.polarity_scores(df.def_new[i])
    key=max(score_dic,key=score_dic.get)
    df.at[i,'score']=score_dic[key]
    df.at[i,'polarity']=key
    #print(key,score_dic[key])
    
    


# ## **Visualization of polarity score**

# In[ ]:





# In[ ]:


val=(df['polarity'].value_counts())


# In[ ]:


type(val)


# In[ ]:


df['polarity'].value_counts().index


# In[ ]:


pd.DataFrame(val).plot.pie(y='polarity',figsize=(10,10),autopct='%1.0f%%')


# In[ ]:




