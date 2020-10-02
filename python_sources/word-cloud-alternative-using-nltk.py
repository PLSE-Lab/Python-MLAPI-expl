#!/usr/bin/env python
# coding: utf-8

# This code offers an alternative to word clouds. I wanted to use NLTK to remove common words (stop words), punctuation and special symbols - for example '#', 'https://'  and generate a bar chart showing word distribution frequency.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import pandas as pd
from wordcloud import WordCloud as wc
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/Donald-Tweets!.csv')


# Stop words in English include things like articles (a, an, the), prepositions (on, in, at,....). These are necessary parts of English  grammar but offer little insight looking at individual words so I will remove them with the help of NLTK. If I were looking at n-grams with n>1 then I would keep the stop words because a lot of English phrases/idioms include these parts of speech.

# In[ ]:


stop = set(stopwords.words('english'))
stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '@', '#', 'rt', 'amp', 'realdonaldtrump', 'http', 'https', '/', '://', '_', 'co', 'trump', 'donald', 'makeamericagreatagain'])


# In[ ]:


series_tweets = df['Tweet_Text']
tweet_str = series_tweets.str.cat(sep = ' ')
list_of_words = [i.lower() for i in wordpunct_tokenize(tweet_str) if i.lower() not in stop and i.isalpha()]
wordfreqdist = nltk.FreqDist(list_of_words)
mostcommon = wordfreqdist.most_common(30)
print(mostcommon)


# You can adjust the number of words in the mostcommon list by changing the parameter passed to the most_common() method. I actually think this data is more useful than a word cloud because I can see exactly how many times he mentioned say Clinton compared to Fox News. Word clouds don't give me that kind of detail. 

# In[ ]:


plt.barh(range(len(mostcommon)),[val[1] for val in mostcommon], align='center')
plt.yticks(range(len(mostcommon)), [val[0] for val in mostcommon])
plt.show()

