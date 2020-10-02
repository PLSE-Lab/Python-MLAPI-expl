#!/usr/bin/env python
# coding: utf-8

# Shakespere plays wordle

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
df=pd.read_csv("../input/Shakespeare_data.csv")
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

df.head()


# In[ ]:


#Remove all stop works
import nltk
from nltk.corpus import stopwords # Import the stop word list
english_stop=stopwords.words("english") 
df['Word_tokens']=df['Player-Line'].apply(lambda x:' '.join(w for w in nltk.word_tokenize(x.lower().strip()) if not w in english_stop) )
df.Word_tokens.head()
df_play_content=pd.DataFrame(df.groupby('Play')['Word_tokens'].apply(lambda x: "{%s}" % ', '.join(x)))
df_play_content.head()


# In[ ]:


from wordcloud import WordCloud
def generateWordCloud(str1,title):

    wordcloud = WordCloud( background_color='white',width=900, height=800, max_font_size=40).generate(str1)
    wordcloud.recolor(random_state=0)
    plt.figure(figsize=(20, 15))
    plt.title(title, fontsize=60,color='red')
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
for cols in df_play_content.index:
    generateWordCloud(df_play_content.loc[cols,'Word_tokens'],cols)

