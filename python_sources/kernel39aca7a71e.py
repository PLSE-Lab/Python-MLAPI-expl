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


#importing libraries
import warnings
import numpy as np
import pandas as pd

import langdetect
from langdetect import detect
import string 
import spacy 
from spacy.lang.en.stop_words import STOP_WORDS 
from spacy.lang.en import English

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


warnings.filterwarnings("ignore")#not show warning for deprecated


# In[ ]:


# read data
df=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


abstract=df["abstract"].dropna()#drooping all rows with all NaN values in abstract columns

len(abstract)


# In[ ]:


# fuction to make a mask selecting text by languaje
def detect_lang(text,lang):
    try:
        return detect(text['abstract']) == lang
    except:
        return False
    


# In[ ]:


# bulding a mask to filter out text written in English
df_abstracts = pd.DataFrame(abstract)
en_abstracts_mask = df_abstracts.apply(lambda row: detect_lang(row, "en"), axis=1)


# In[ ]:


abstracts_en=df_abstracts[en_abstracts_mask]
print(len(abstracts_en), "of our initial abstracts are writing in english languaje, we will use just those papers to made the analysis")


# In[ ]:


# Join the different processed titles together.
long_string = ','.join(list(abstracts_en["abstract"].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()


# Since abtracts relate  the main results fields where those are contributed. We propose made a clustering to see around wich goals are centered those groups of studies.

# ### Clean Data

# In[ ]:


# list of punctuation and simbols
symbols_punctuations = string.punctuation

#  disabling Named Entity Recognition for speed
nlp = spacy.load('en',disable=['parser', 'ner'])

# Create our list of stopwords
stop_words = spacy.lang.en.stop_words.STOP_WORDS


def tokenizer(text):
    # Creating our token object
    mytokens = nlp(text)

    # Lemmatizing each token and converting each token into lowercase if not pronoum
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words and puntuation
    mytokens = [ word for word in mytokens if word not in symbols_punctuations  and word not in stop_words]
 
    mytokens = [ word for word in mytokens if len(word) > 2]

    
    return mytokens


# ### Getting words by topics

# Some of the task asked in the challege consists of knowing how much you know about certain topics according to the reported studies. Here we show that about which aspects are asked.
# 
# What is known about **transmission**, **incubation**, and **environmental stability**?
# 
# What do we know about COVID-19 **risk factors**?
# 
# What do we know about virus **genetics**, **origin**, and **evolution**?
# 
# What do we know about **vaccines** and **therapeutics**?
# 
# What has been published about **medical care**?
# 
# What do we know about **non-pharmaceutical interventions**?
# 
# What do we know about **diagnostics** and **surveillance**?
# 
# What has been published about **ethical and social science considerations**?
# 
# What has been published about **information sharing** and **inter-sectoral collaboration**?

# The aproach that are going to follow is try to find topic in the documents collections and see how much those match with the aspects that are asked there. For that we are plane to use Latent Dirichlet Allocation(LDA), method reported at literature for Topic modeling by Kavita Ganesan,Priya Dwivedi, Aneesha Bakharia for just cite some author.
# We will use that method implemented in sklearn.

# ### Raw  Bag_of_words

# In[ ]:


np.random.seed()
#creation of the bag of word with CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# bag of words using our tokenizer .defining  ngram_range=(1,1) means only unigrams 

CountVectorizer_bow = CountVectorizer(tokenizer = tokenizer, ngram_range=(1,1),max_df=0.80,min_df=2) 
bow_raw_count=CountVectorizer_bow.fit_transform(abstracts_en["abstract"])


# In[ ]:


bow_raw_count.shape


# In[ ]:



"""from sklearn.decomposition import  LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

# initializing the LDA object
# online Method used to update _component much faster
#0.5, 1.0] to guarantee asymptotic convergence for online method
LDA = LatentDirichletAllocation(max_iter=10, learning_method='online', learning_offset=50,random_state=0,batch_size=200,n_jobs=-1)

# Define Search Param
parameters = {"n_components": [7, 6, 9, 13, 16],"learning_decay": [0.5,0.7,0.9]}


# initializing gridsearchcv
grid_cv = GridSearchCV(LDA, param_grid=parameters)"""


# In[ ]:


# in a previus exploratory study we use the GridSearchCV to tunning the parameters as in the cell above, An we get the best reults for a learning_decay of 0.7 and 7 topics.


# In[ ]:


LDA = LatentDirichletAllocation(n_components=7, max_iter=10, learning_method='online', learning_offset=50,random_state=0,batch_size=200,n_jobs=-1,learning_decay=0.7)


# In[ ]:


LDA_topics=LDA.fit(bow_raw_count)


# In[ ]:



# Perplexity
print("Model Perplexity: ", LDA_topics.perplexity(bow_raw_count))

# Best score
print("Best Score: ", LDA_topics.score(bow_raw_count))


# Plotting to see in the algorith performance

# In[ ]:



def display_topics(model, feature_label, no_top_words):

    for id, topic in enumerate(model.components_):
        print( "Topic:", (id))# print fisrt topic label
        print(" ".join([feature_label[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))# get sorted higher frecuency terms


# In[ ]:



labels=CountVectorizer_bow.get_feature_names()


# In[ ]:


display_topics(LDA_topics, labels, 10)#


# Looking at the most reperesnet terms in each topic we can see some topic that seem have a well define theme:
# Abstracts included in **Topic 0** detection method.
# 
# **Topic 1** not well define theme(general)
# 
# **Topic 2** not well define theme(general)
# 
# **Topic 3** is tolking include abstract that are talking about structure and virus infection mechanisms
# 
# **Topic 4** clinical studies by age groups, interesting term child
# 
# **Topic 5** vaccine and interesting term but other are generals
# 
# **Topic 6** more specific for Cov-19 outbreack
# 
# Any way those are early conclusions because we need to see not just in the more frecuent terms but also which word are specific for each topic. To take this in account we move to see the graph generate by pyLDAvis library

# In[ ]:


# getting the topic and term performance for LDA

pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(LDA_topics, bow_raw_count, CountVectorizer_bow, mds='tsne',n_jobs=-1)
panel


# Looking to the areas of the circles we can infer that the marginal topic distribution in the corpus is very similar for each topic, although topic 7 and 6 are smaller. Aslo we can see this seven topics have not overlapping between them(are relative far),there are distributed  into different two-dimensional planes.

# *General terms detected and Higher frecuency Specific terms overall corpus:* 
# 
# We can see from the bar chart above that many words like virus that are  frequent in the corpus  are present in the overall term frecuency even when we have select **max_df as 0.80**, maybe we could run gain the feature extraccion algorith **tunning max_df** with different values. Another strategy here would be build **our own stopwords dictionary**, because we are analysing text for an especific field(but in that case we need some  expertician in this domain)
# 
# Note:We have convert the capital words onto lower case, then those we can found some acronym for deseases or drugs that can appear in lower case and can be a little confuse.
# 
# Term **Child** is interesting because could mean that we have studies dedicated to early age population in this topic. Mostly represented in topic 5
# Some animal also appear as pre dominants terms
# 
# 

# *Terms that are tunique and we will add to the prev useful for interpreting each topic.*
# 
# We will take in account corpus-wide frequency of a given term as well as the topic-specific frequency of the term
# 
# - Topic 1: SAR-Cov structure and mechanism of infections
# - Topic 2: Infection in animals(cat, bat) and atirretrovirals(INF)
# - Topic 3: not well define
# - Topic 4: respiratory infections in group of age (child)
# - Topic 5: Covid outbreack
# - Topic 6:detection and diagnostics methods
# - topic 7: drug design and general terms
# 
# Look like there are mixed topics we propouse perform LDA with at least 10 topics, to see the term frequency behavior and see if there is higher match with the chalenge topics
# 

# How many abstract we have by the topic we have found?

# In[ ]:


topic_values = LDA_topics.transform(bow_raw_count)
abstracts_en['topic_LDA'] = topic_values.argmax(axis=1)


# In[ ]:


abstracts_en.head(2)


# In[ ]:


document_by_topics_LDA=abstracts_en["topic_LDA"].value_counts()


# In[ ]:


document_by_topics_LDA_df=pd. DataFrame(document_by_topics_LDA).reset_index()
document_by_topics_LDA_df.columns=["topic_LDA","number_documents"]


# In[ ]:


document_by_topics_LDA_df


# In[ ]:



fig=plt.figure(figsize=(7,7))


plt.barh(document_by_topics_LDA_df["topic_LDA"],document_by_topics_LDA_df["number_documents"])
plt.ylabel("topic_LDA")
plt.xlabel("number_documents")
plt.title("Number of document by topic selected using LDA")

plt.show()


# As result of this analysis we can show some generals topics found in the collected studies. A next iteration selecting a higher ammount of topic is nescesary to go in more specific topic as vaccines, drug design and detection method. Incluid stopwords typical of the field also coul be help to get better cluster or use lower max_df. Also incluid in the study those paper in  differents languajes than english(we have removed at the beggining) can apport valuable information

# In[ ]:




