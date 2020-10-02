#!/usr/bin/env python
# coding: utf-8

# <img src="https://www.webintravel.com/wp-content/uploads/2020/02/covid-19.jpg" alt="corona" width="900" align="middle">

# The notebook follows the following 3 stages of explanation:
# 
# 1. Introduction
# 2. Analysis
# 3. Insights and Conclusion
# 

# # Introduction
# 
# This notebook tries to find out the common atributes between the removed posts of the 'Reddit - Data is Beautiful' community. This is a very primitive analysis which is being done using only the titles of the respective posts. Ofcourse, the title alone may not give the full desciption of its contents, but I believe that title will be a good yardstick for analyzing the contents.
# 
# 
# 

# 

# ## Why does a post get removed?
# 
# Before moving further this is a important question to answer as we are trying to analyze the reason for removed posts. For starters every reddit community has their own set of rules for posts, and for 'Data is Beautiful' community you can refer it in the below mentioned link.
# https://www.reddit.com/r/dataisbeautiful/wiki/index#wiki_rules
# 
# So, after reading it we can conclude that the reasons for removal of a post is, it is mis-marked as 'Original Content' or Spamming such as click baiting or ir-relevent information, etc.
# 
# 

# <img src="https://cdn0.tnwcdn.com/wp-content/blogs.dir/1/files/2019/08/reddit-deleted-posts-comments-threads.jpg" alt="corona" width="500" align="middle">

# # Analysis

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


# Importing Libraries.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy.stats as stats
from sklearn import ensemble, tree, linear_model, preprocessing
import missingno as msno
import pandas_profiling
import plotly.express as px


# Reading the Data.

# In[ ]:


Original_Data= pd.read_csv('../input/dataisbeautiful/r_dataisbeautiful_posts.csv')


# In[ ]:


Original_Data.head()


# Creating a new DataFrame with **Removed posts only** and **Retained posts only**.

# In[ ]:


removed_thread = Original_Data[Original_Data['removed_by'].notna()]
original_thread = Original_Data[(Original_Data['removed_by'].isna())]


# In[ ]:


removed_thread.head()


# Finding out who all can remove a post.

# In[ ]:


removers=[]

for name in removed_thread['removed_by']:
    if name not in removers:
        removers.append(name)
        
print(removers)


# After separating the removed posts alone a **WordCloud** is formed with the titles of the posts, this is done to get a glimpse of the titles in a graphical format.

# In[ ]:


from wordcloud import WordCloud, STOPWORDS

# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=50, figure_size=(15.0,15.0), 
                   title = None, title_size=20, image_color=False,color = 'black'):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color=color,
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    
plot_wordcloud(removed_thread['title'].values, title="Word Cloud of Removed Posts Titles.")


# We could see words such as 'Click', 'Virus', 'Growing', 'Wuhan', are quite large and are grabbing our attention, which suggest us that there is a lot of mis-information or spaming which is taking place regarding the COVID-19 disease. Now to understand further we dig deep into n-gram analysis of the titles.

# In[ ]:


import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import warnings 
warnings.filterwarnings('ignore')

def count_ngrams(dataframe,column,begin_ngram,end_ngram):
    # adapted from https://stackoverflow.com/questions/36572221/how-to-find-ngram-frequency-of-a-column-in-a-pandas-dataframe
    word_vectorizer = CountVectorizer(ngram_range=(begin_ngram,end_ngram), analyzer='word')
    sparse_matrix = word_vectorizer.fit_transform(removed_thread['title'].dropna())
    frequencies = sum(sparse_matrix).toarray()[0]
    most_common = pd.DataFrame(frequencies, 
                               index=word_vectorizer.get_feature_names(), 
                               columns=['frequency']).sort_values('frequency',ascending=False)
    most_common['ngram'] = most_common.index
    most_common.reset_index()
    return most_common

def word_cloud_function(df,column,number_of_words):
    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]
    word_string=str(popular_words_nonstop)
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          max_words=number_of_words,
                          width=1000,height=1000,
                         ).generate(word_string)
    plt.clf()
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

def word_bar_graph_function(df,column,title):
    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]
    plt.barh(range(50), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:50])])
    plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[0:50]))
    plt.title(title)
    plt.show()
    
three_gram = count_ngrams(removed_thread,'title',3,3)
words_to_exclude = ["my","to","at","for","it","the","with","from","would","there","or","if","it","but","of","in","as","and",'NaN','dtype']


# Finding out the most frequent word occured in the titles.

# In[ ]:


plt.figure(figsize=(10,10))
word_bar_graph_function(removed_thread,'title','Most common words in titles removed threads in Reddit')


# As expected, words such as '[OC]'(Original Content), 'coronavirus', 'world', 'covid-19', are leading the pack which again supports our hypotheis. And to understant it further lets do a **bi-gram** and **tri-gram** analysis.

# In[ ]:


bi_gram = count_ngrams(removed_thread,'title',2,2)
fig = px.bar(bi_gram.sort_values('frequency',ascending=False)[0:10], 
             x="frequency", 
             y="ngram",
             title='Most Common 2-Words in Titles of removed posts in Reddit',
             orientation='h')
fig.show()


# In[ ]:


fig = px.bar(three_gram.sort_values('frequency',ascending=False)[0:10], 
             x="frequency", 
             y="ngram",
             title='Most Common 3-Words in Titles of removed posts in Reddit',
             orientation='h')
fig.show()


# Again the results of above graphs suggest us that the words such as 'covid-19', 'pandemics' are most frequent when it comes to titles of deleted posts.

# # Conclusion
# 
# So from the above analysis of the given data we could easily conclude that most number of removed posts due to mis-information or spamming is about or relating to COVID-19 disease. As stated earlier this is a very primitive analysis, so stay tuned for a more detailed analysis which is to be done on the content of the posts and find more intresting insights.
# 
# Finally I would like to mention and thank the following notebooks which I refered...
# https://www.kaggle.com/ratan123/cord-19-understanding-papers-with-textanalytics
# https://www.kaggle.com/jpmiller/creating-a-good-analytics-report
# 
# Thankyou for reading.
