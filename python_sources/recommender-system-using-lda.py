#!/usr/bin/env python
# coding: utf-8

# **Path**
#  - <a href='#1'>Context/Pre Processing</a>  
#  - <a href='#2'> LDA Model</a>
#  - <a href='#3'>Analysis</a>
#  - <a href='#4'>Further Reading</a>
#  
#  P.S This is a work in Progress, will be updating whenever I have time. This notebook borrows code from https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial
#  
#  P.P.S I am not a programmer ( if that's not obvious by how ugly this notebook looks), some pieces of this code might not be the most efficient way.

# <h1><a id='1'> Context/Pre Processing </a></h1>

# More than a text analytics problem, this is a collaborative filtering problem; where the objective is to identify which topics are relevant to the donor. Also, It's very intutive to assume that we are passionate about simlar things. Eg. Chivas and Jamesons, Frost and Plath, LOTR and Harry Potter.
# 
# So we are going to test the hypothesis that Donors donate for causes that have some similarity in terms of their topics 
# . 
# To test this hypothesis:
# 1. Shortlist donors who have contributed to multiple projects
# 2. Is there any relation at all between the topics donated to by the same donor?
# 
# Ideally we should have information to compare to essays the donor did not contribute to, but we don't.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
don = pd.read_csv("../input/Donations.csv")
# Any results you write to the current directory are saved as output.


# In[ ]:


#Find Donors who gave donations to multiple projects 
import base64
import numpy as np
import pandas as pd
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from collections import Counter
from scipy.misc import imread
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

projects = pd.read_csv("../input/Projects.csv")

ds = projects.loc[projects['Project Current Status'].isin(['Fully Funded','Expired','Live'])]
df3 = ds.merge(don, on = "Project ID" )
df3['Donor ID'] = df3['Donor ID'].str.strip()

#consider only donations made to different projects, as we are interested in what makes the donor open his wallet

df4 = df3.drop_duplicates(subset=['Project ID', 'Donor ID'])
df4['Donor ID'].value_counts()

projects.head()


# 
# So we have above the multiple projects which received donations from the same donor. So the question we are trying to answer is what's the relation between those projects? So this is a two step proces; first use LDA to break down the essays into individual topics. Second, try to understand the relation between the projects if the donors contributed multiple times. 

# **LDA**
# (from wikipedia)
# 
# In natural language processing, latent Dirichlet allocation (LDA) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. For example, if observations are words collected into documents, it posits that each document is a mixture of a small number of topics and that each word's creation is attributable to one of the document's topics. LDA is an example of a topic model and was first presented as a graphical model for topic discovery by David Blei, Andrew Ng, and Michael I. Jordan in 2003.
# 
# 

# In[ ]:


# vectorizer takes care of stopwords, but we seek lemmatisation as well... hence this class to override the function inside the vectoriser

from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()
class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))


# In[ ]:



# create a sparse matrix for lda 
df4_sample = df4.sample(n = 100000, random_state = 69)
text = list(df4_sample["Project Essay"])
# Calling our overwritten Count vectorizer
tf_vectorizer = LemmaCountVectorizer(max_df=0.65, 
                                     min_df=2,
                                     stop_words='english',
                                     decode_error='ignore')
tf = tf_vectorizer.fit_transform(text)


# In[ ]:


feature_names = tf_vectorizer.get_feature_names()
count_vec = np.asarray(tf.sum(axis=0)).ravel()
zipped = list(zip(feature_names, count_vec))
x, y = (list(x) for x in zip(*sorted(zipped, key=lambda x: x[1], reverse=True)))
# Now I want to extract out on the top 15 and bottom 15 words
Y = np.concatenate([y[0:15], y[-16:-1]])
X = np.concatenate([x[0:15], x[-16:-1]])

# Plotting the Plot.ly plot for the Top 50 word frequencies
data = [go.Bar(
            x = x[0:50],
            y = y[0:50],
            marker= dict(colorscale='Jet',
                         color = y[0:50]
                        ),
            text='Word counts'
    )]

layout = go.Layout(
    title='Top 50 Word frequencies after Preprocessing'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='basic-bar')

# Plotting the Plot.ly plot for the Top 50 word frequencies
data = [go.Bar(
            x = x[-100:],
            y = y[-100:],
            marker= dict(colorscale='Portland',
                         color = y[-100:]
                        ),
            text='Word counts'
    )]

layout = go.Layout(
    title='Bottom 100 Word frequencies after Preprocessing'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='basic-bar')


# <h1><a id='2'> LDA </a></h1>

# Once we have the sparse matrix with the necessary preprocessing done, time to fit the LDA. Run the code below and go pour yourself a glass of Chivas and contemplate the meaninglessness of life.... 

# In[ ]:


lda = LatentDirichletAllocation(n_components=63, max_iter=5,
                                learning_method = 'online',
                                learning_offset = 50.,
                                random_state = 0)
lda.fit(tf)


# <h1><a id='3'> Analysis </a> </h1>

# Lets randomly pick up topic no. 867
# 

# In[ ]:


text[890]


# Now, have a look at the topics LDA is picking up ( 0.05 implies, the essay containts at least 5% of words that come from that topic's bag of words)

# In[ ]:


tf1 = tf_vectorizer.transform([text[890]])
doc_topic = lda.transform(tf1)
doc_topic
import numpy   
topic_high = numpy.where(doc_topic > 0.05)
numpy.where(doc_topic > 0.05)

topic_high = list(topic_high)[1]
top_tup = tuple(map(lambda x:(x,doc_topic[0,x]),topic_high))
sor_ry = sorted(top_tup, key=lambda tup: tup[1], reverse = True)


# Based on content-proportion:
# Topic#1

# In[ ]:


from wordcloud import WordCloud, STOPWORDS

tf_feature_names = tf_vectorizer.get_feature_names()

first_topic = lda.components_[sor_ry[0][0]]
first_topic_words = [tf_feature_names[i] for i in first_topic.argsort()[:-50 - 1 :-1]]

firstcloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          width=2500,
                          height=1800
                         ).generate(" ".join(first_topic_words))
plt.imshow(firstcloud)
plt.axis('off')
plt.show()


# Topic #2

# In[ ]:


sixtieth_topic = lda.components_[sor_ry[1][0]]
sixtieth_topic_words = [tf_feature_names[i] for i in sixtieth_topic.argsort()[:-50 - 1 :-1]]

scloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          width=2500,
                          height=1800
                         ).generate(" ".join(sixtieth_topic_words))
plt.imshow(scloud)
plt.axis('off')
plt.show()


# Topic #3

# To test our hypothesis, lets pick up 3 essays the same donor ( who donated for project 889) has donated...

# In[ ]:


supp_essays = df4[df4['Donor ID'] == df4_sample.iloc[890]['Donor ID']]
supp_essays


# In[ ]:


text_test = list(supp_essays['Project Essay'])

tf1 = tf_vectorizer.transform([text_test[1]])

doc_topic = lda.transform(tf1)
print(doc_topic)
import numpy   
numpy.where(doc_topic > 0.05)


# In[ ]:


text_test = list(supp_essays['Project Essay'])

tf1 = tf_vectorizer.transform([text_test[2]])

doc_topic = lda.transform(tf1)
print(doc_topic)
import numpy   
numpy.where(doc_topic > 0.05)
type(doc_topic)


# <h1> Ranking the Essays to Send </h1>
# 
# 
# 

# Sadly we do not have the essays the donor did not donate to; i.e; saw the email but chose not to donate. 
# 
# So as a poor substitute to test our hypothesis, we are going to assume the amount donated reflects the interest in the essay and higher the amount donated, greater the passion for the cause.
# 
# So at this moment, we have a couple of Essays, We are going to rank the essays by the similarity to the essay that received the highest donation; And then we can see if they received donations in the similar order

# In[ ]:


supp_essays['Project Essay'].loc[supp_essays['Donation Amount'].idxmax()]


# In[ ]:


# To test our hypothesis, we are going to break down the essays into text topics, and then see if there is a relation between the amount donated and the topics relevance

tf1 = tf_vectorizer.transform(text_test)
doc_topic = lda.transform(tf1)
supp_cp = supp_essays
d_copy = doc_topic
df = pd.DataFrame(d_copy)
supp_cp = supp_cp.reset_index(drop=True)
dataset = pd.concat([supp_cp['Donation Amount'],df],axis = 1)


# In[ ]:


dataset


# So for our model dataset, our dependent variable is the amount donated, and our independent variables are the LDA topic distribution; 
# Before we jump into any kind of conclusion, let us look at the mean distribution across the topics,;

# In[ ]:


topics = dataset.drop(['Donation Amount'], axis = 1)


# In[ ]:


topics.mean().plot(kind='bar', figsize=(20,10))


# So it's very apparent that some topics the donor has donated to more than the remaining topics. Let us try assigning a monetary value to the topics. The simplest way would be to run a linear regression, where the coefficient would imply the increase in donation amount with a 1% increase in the topic proportion.  Will be adding this and similarity to the highest donated essay soon.
# 

# <h1><a id='4'> Further reading </a></h1>
# http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/
# 
# http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
# 

# 
