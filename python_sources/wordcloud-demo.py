#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 

# Reads 'Youtube04-Eminem.csv' file 
df = pd.read_csv("/kaggle/input/images/Youtube01-Psy.csv", encoding ="latin-1") 
df.head()


# In[ ]:


def plotWorldCloud(df,stopwords,maxWords=500):
    comment_words = '' 
    # iterate through the csv file 
    for val in df.CONTENT: 

        # typecaste each val to string 
        val = str(val) 

        # split the value 
        tokens = val.split() 
        comment_words += " ".join(tokens)+" "

    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords,
                    max_words = maxWords,
                    min_font_size = 10).generate(comment_words) 

    # plot the WordCloud image					 
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show() 


# In[ ]:


plotWorldCloud(df,' ')
#plotWorldCloud(df,'subscribe')


# In[ ]:


print(list(set(STOPWORDS)))


# In[ ]:


plotWorldCloud(df,set(STOPWORDS),400)


# In[ ]:


# removing everything except alphabets`
df['CONTENT'] = df['CONTENT'].str.replace("[^a-zA-Z#]", " ")


# In[ ]:


plotWorldCloud(df,set(STOPWORDS),400)


# In[ ]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# removing short words
df['CONTENT'] = df['CONTENT'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>14]))

# tokenization
tokenized_doc = df['CONTENT'].apply(lambda x: x.split()) 

# remove stop-words
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

# de-tokenization
detokenized_doc = []
for i in range(len(df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)
    
df['CONTENT'] = detokenized_doc


# In[ ]:


print(stop_words)


# In[ ]:


# make all text lowercase
df['CONTENT'] = df['CONTENT'].apply(lambda x: x.lower())

plotWorldCloud(df,set(STOPWORDS),200)

