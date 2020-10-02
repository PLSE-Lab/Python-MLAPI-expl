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


#Importing libraries
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


con = sqlite3.connect("../input/ds5230usml-project/database.sqlite")
#pd.read_sql_query()


# In[ ]:


message = pd.read_sql_query("""
Select Score,Summary, HelpfulnessNumerator as Helpfulvotes, HelpfulnessDenominator as Totalvotes
From Reviews
""",con)


# In[ ]:


#type(message)


# In[ ]:


message.head(5)


# In[ ]:


message["Usefulness"] = (message["Helpfulvotes"]/message["Totalvotes"]).apply(lambda n: "Useful" if n > 0.6 else "Not Useful")


# In[ ]:


message.head(10)


# In[ ]:


# Plotting the word cloud
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)


#mpl.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
mpl.rcParams['font.size']=12                #10 
mpl.rcParams['savefig.dpi']=100             #72 
mpl.rcParams['figure.subplot.bottom']=.1 


def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))
    
    fig = plt.figure(1, figsize=(8, 8))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
    
show_wordcloud(message["Summary"])


# In[ ]:




