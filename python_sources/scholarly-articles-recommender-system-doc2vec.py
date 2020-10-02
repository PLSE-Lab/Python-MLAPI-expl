#!/usr/bin/env python
# coding: utf-8

# # ***Scholarly Articles Recommender Engine Using Doc2Vec***

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


# > #### I have tried to make a recommendation engine using Doc2Vec using ArXiv research papers meta-data dataset and text of tags descreption is taken from https://arxiv.org/archive/cs.
# > #### Suggestions are most welcomed that can improve the recommendations.

# In[ ]:


import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser,Phrases
from time import time
import multiprocessing
from gensim.matutils import Dense2Corpus
from gensim.similarities import MatrixSimilarity
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from gensim.models import LdaModel,KeyedVectors
import umap
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary


# > **Data Cleaning**

# In[ ]:


df = pd.read_json('../input/arxivdataset/arxivData.json',orient='records')
df.head()


# As we can see above there are many unwanted symbols and many columns have unwanted data.

# Cleaning author names :

# In[ ]:


df2 = pd.DataFrame(df.author.str.split('}').tolist(),index = df.index).stack()
df2.head()


# In[ ]:


def rem_unwanted(line):
    return re.sub("\'term'|\'rel'|\'href'|\'type'|\'title'|\[|\{|\'name'|\'|\]|\,|\}",'',line).strip(' ').strip("''").strip(":")


# In[ ]:


df2 = pd.DataFrame(df2.apply(rem_unwanted))


# In[ ]:


df2 = pd.DataFrame(df2.unstack().iloc[:,0:2].to_records()).drop(columns={'index'})


# In[ ]:


df2.columns = ['Author1','Author2']


# In[ ]:


df2.Author1 = df2.Author1.str.strip(' ')
df2.Author2 = df2.Author2.str.strip(' ')


# In[ ]:


df2[df2.Author2 == '']


# In[ ]:


df2 = df2.reset_index().drop(columns='index')


# In[ ]:


df2.head()


# Authors' name is cleaned

# In[ ]:


len(df2.Author1.unique())


# Cleaning links to obtain links of pdf and text version.

# In[ ]:


df3 = pd.DataFrame(df.link.str.split(', ').tolist(),index = df.index).stack()


# In[ ]:


df3 = pd.DataFrame(df3.apply(rem_unwanted,convert_dtype=True))


# In[ ]:


df3.head()


# In[ ]:


df3 = df3.unstack()


# In[ ]:


links = df3.iloc[:,[1,4]]
links.columns = ['textLink','pdfLink']
links.head()


# Cleaning topics to get subjects of articles:

# In[ ]:


tags = pd.DataFrame(df['tag'].str.split(',').tolist())
# tags = tags[tags.str.contains('term')]
# tags.head()
tags = tags.iloc[:,[0,3,6]].stack()
# tags = tags.iloc[:,0]


# In[ ]:


tags.head()


# In[ ]:


tags = tags.apply(rem_unwanted)


# In[ ]:


tags = tags.unstack()


# In[ ]:


tags[0] = tags[0].str.strip()
tags.iloc[:,1] = tags.iloc[:,1].str.strip()
tags.iloc[:,2] = tags.iloc[:,2].str.strip()


# In[ ]:


tags.columns = ['Topic1','Topic2','Topic3']


# In[ ]:


tags.head()


# Merging all of them with original dataframe.

# In[ ]:


pre0 = pd.merge(df,tags,how = 'inner',left_index=True,right_index=True).drop('tag',axis=1)
pre = pd.merge(pre0,df2,how = 'inner',left_index=True,right_index=True).drop('author',axis=1)
data = pd.merge(pre,links,how = 'inner',left_index=True,right_index=True).drop('link',axis=1)


# In[ ]:


def rem_bracket(line):
#     return re.sub("\'term'|\'rel'|\'href'|\'type'|\'title'|\[|\{|\'name'|\'|\]|\)}",'',line).strip(' ').strip("''").strip(":")
    return line.strip(')')


# > Initial Data

# In[ ]:


df.head()


# > Cleaned Data

# In[ ]:


data.head()


# > Cleaning the topics text file

# In[ ]:


tags = pd.read_csv('../input/arxivtagsdescription/tags.txt',sep='/n',header=None,engine='python')


# In[ ]:


tags.head()


# In[ ]:


tags.tail()


# Some have abbreviated topics at beginning and some have abbreviated topics at last.

# In[ ]:


d1 = pd.DataFrame(tags.iloc[[i for i in tags.index if i%2==0]].reset_index().iloc[0:47][0].str.split(' - ').tolist())
d2 = pd.DataFrame(tags.iloc[[i for i in tags.index if i%2==0]].reset_index().iloc[47:][0].str.split('(').tolist())


# In[ ]:


d1.head()


# In[ ]:


d2.head()


# In[ ]:


d2[1] = d2[1].apply(rem_bracket)


# In[ ]:


d2 = d2.set_index([1]).reset_index()


# In[ ]:


d2.columns = [0,1]


# In[ ]:


d2.head()


# In[ ]:


d3 = pd.concat([d1,d2])


# In[ ]:


d3 = d3.reset_index().drop(columns=['index'])


# In[ ]:


d3['TopicExplain'] = tags.iloc[[i for i in tags.index if i%2!=0]][0].reset_index()[0]


# In[ ]:


d3.columns = ['Topic','FullTopic','TopicExplain']


# In[ ]:


tags = d3.copy()


# > Cleaned Topics

# In[ ]:


tags.head()


# In[ ]:


data.fillna(' ',inplace=True)


# In[ ]:


data.isna().any()


# In[ ]:


data['summary'][0]


# In[ ]:


data['title'][0]


# We can see */n* as impurity.

# In[ ]:


def rem_n(line):
    return re.sub('\\n',' ',line)


# In[ ]:


data['summary'] = data['summary'].apply(rem_n)
data['title'] = data['title'].apply(rem_n)


# In[ ]:


data['summary'][0]


# Combining all cleaned data and their topics.

# In[ ]:


db1 = pd.merge(data,tags,how='left',left_on='Topic1',right_on='Topic').drop(columns=['Topic'])
db2 = pd.merge(db1,tags,how='left',left_on='Topic2',right_on='Topic').drop(columns=['Topic','TopicExplain_y'])
db3 = pd.merge(db2,tags,how='left',left_on='Topic3',right_on='Topic').drop(columns=['Topic','TopicExplain'])


# In[ ]:


db3 = db3[['id', 'summary', 'title', 'year', 'FullTopic_x', 'FullTopic_y', 'FullTopic','TopicExplain_x', 'Topic1', 'Topic2', 'Topic3','Author1', 'Author2', 'textLink', 'pdfLink']]


# In[ ]:


db3.columns = ['id', 'summary', 'title', 'year','Topic1',
       'Topic2', 'Topic3', 'Topic', 'DTopic1', 'DTopic2', 'DTopic3',
       'Author1', 'Author2', 'textLink', 'pdfLink' ]


# In[ ]:


db3.drop(columns=['DTopic1','DTopic2','DTopic3'],inplace=True)


# In[ ]:


f1 = db3.copy()


# In[ ]:


f1.fillna(' ',inplace=True)


# In[ ]:


f1.isna().sum()


# No empty/Nan values. Good to go further.

# > Loading spacy for NLP and tokenising.

# In[ ]:


nlp = spacy.load('en',disable = ['ner','parser'])
spacy.require_gpu()


# Making Stop words and puctuations i.e. are, this, is, was etc.

# In[ ]:


stopwords = list(STOP_WORDS)+list((''.join(string.punctuation)).strip(''))+['-pron-','-PRON-']
len(stopwords)


# Tokeniser function:

# In[ ]:


def lemmatizer(df):
    texts = []
    c=0
    for text in df:
        if c%1000==0:
            print("Processed articles: ",c)
        c+=1
        doc = nlp(text)
        lemma = [word.lemma_.lower().strip('') for word in doc]
        words = [word for word in lemma if word not in stopwords]
        texts.append(' '.join(words))
    return pd.Series(texts)


# > Concat all text for tokenising

# In[ ]:


f1['Full'] = (f1['title']+" "+f1['summary']+' '+f1['Topic1']+' '+f1['Topic2']+' '+f1['Topic3']+' '+f1['Topic']+' '+f1['Author1']+' '+f1['Author2'])


# In[ ]:


t = time()
processed_text = lemmatizer(f1['Full'])


# In[ ]:


(time()-t)/60


# Takes 15 minutes to process the whole data.

# In[ ]:


processed_text.index = f1['id'].values


# In[ ]:


processed_text = pd.DataFrame(processed_text)


# In[ ]:


processed_text.iloc[0:12].index


# In[ ]:


processed_text.iloc[:6].values


# > Creating Bigrams and Trigrams

# In[ ]:


phr = [i[0].split() for i in processed_text.values]


# Min count means words frequency, if a two or three words appears more than X times simultaneously then that words combined will be counted as bigrams and trigrams respectively.
# Progress_pre simply means batch size.

# In[ ]:


phrases = Phrases(phr,min_count=50,progress_per=1000)


# In[ ]:


bigram = Phraser(phrases)


# In[ ]:


sentences = bigram[phr]


# In[ ]:


phrases = Phrases(sentences,min_count=25,progress_per=1000)


# In[ ]:


trigram = Phraser(phrases)


# In[ ]:


trigrams = trigram[sentences]


# In[ ]:


trigrams[123]


# As we can see some good bigrams and trigrams.

# > Creating Tagged data for Doc2Vec (tag = articleId)

# In[ ]:


tagged_data = [TaggedDocument(words=' '.join(i),tags=[j]) for i, j in zip(trigrams,processed_text.index)]


# In[ ]:


tagged_data[2]


# * dm = 1 means using skip-grams
# * vector_size = 300, word embeddings will have shape of (vocab_size,300)
# * window = 2, model will try to predict every second word.
# * workers = 4, number of processors available for parallel processing.
# * negative = 5, 5% of noise words will be removed.
# * min_count = 10, words lower than frequency of 10 will be ignored.
# 

# In[ ]:


docmodel = Doc2Vec(dm=1,vector_size = 300,window=2,workers=4,negative=5,min_count=10, dbow_words=1)


# In[ ]:


docmodel.build_vocab(tagged_data)


# In[ ]:


docmodel.corpus_total_words


# Training for 20 epochs.

# In[ ]:


t = time()
docmodel.train(tagged_data,total_examples=docmodel.corpus_count,epochs=20)


# In[ ]:


(time()-t)/60


# In[ ]:


docmodel.init_sims(replace=True)
# docmodel.save('model')


# In[ ]:


docmodel = Doc2Vec.load('model')


# In[ ]:


docmodel.docvecs.most_similar('1802.00209v1')


# In[ ]:


# f2 = f1[['id','title','summary','FullTopic','TopicExplain','Author1','Author2']]


# In[ ]:


def get_recommendations(*n):
    j = docmodel.docvecs.most_similar(positive=n)
    r = f1[f1['id'].isin(list(n))]
    p = ['Searched',]*len(r)
    for i in j:
        r = pd.concat([r,f1[f1['id']==i[0]]])
        p.append(i[1])
#     r = f2[f2['id'].isin(a)]
    r['ProbabilityChance'] = p
    return r


# In[ ]:


rec = get_recommendations('1802.00209v1')
# ('1305.3814v2')


# **Here is the recommendations for id 1802.00209v1**
# * First one is 1802.00209v1 then recommendations are there with probability.

# In[ ]:


rec


# In[ ]:


mat = TfidfVectorizer().fit_transform(lemmatizer(rec['Full']))


# > Similarity in Recommended Articles

# In[ ]:


(cosine_similarity(mat,mat)*100)[:,0]


# As we can see similarity between recommendation and real article is very good.

# But checking relative articles are good or not is a difficult task so we will try to get more intuition on checking accuracy of recommendations

# Installing GPU version of tSNE (for faster computation)

# In[ ]:


get_ipython().system('yes Y | conda install faiss-gpu cudatoolkit=10.0 -c pytorch')
get_ipython().system('wget https://anaconda.org/CannyLab/tsnecuda/2.1.0/download/linux-64/tsnecuda-2.1.0-cuda100.tar.bz2')
get_ipython().system("tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2 --wildcards 'lib/*'")
get_ipython().system("tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2 --wildcards 'site-packages/*'")
get_ipython().system('cp -r site-packages/* /opt/conda/lib/python3.6/site-packages/')
get_ipython().system('cp /kaggle/working/lib/libfaiss.so /usr/local/cuda/lib64/')
get_ipython().system('apt search openblas')
get_ipython().system('yes Y | apt install libopenblas-dev')


# In[ ]:


get_ipython().system('pip install tsnecuda')


# In[ ]:


from tsnecuda import TSNE


# In[ ]:


# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from functions import make_tsne_subset


# These are the ids.

# In[ ]:


doc_tags = docmodel.docvecs.doctags.keys()


# These are word embeddings.

# In[ ]:


X = docmodel[doc_tags]


# In[ ]:


print(X[0:5])


# Fit and transform a t-SNE object with the vector data for dimensionality reduction and get a idea about cluster formation.

# In[ ]:


tSNE = TSNE(n_components=2)


# In[ ]:


X_tsne = tSNE.fit_transform(X)


# In[ ]:


df = pd.DataFrame(X_tsne, index=doc_tags, columns=['x', 'y'])


# In[ ]:


df.head()


# In[ ]:


plt.scatter(df['x'], df['y'], s=0.4, alpha=0.4)


# Model has overlapping which is due to some topics have vast number (like Artificial Intelligence is primary or secondary subjects of many articles) of articles but still there are separation between articles. Further tuning in model can introduce more separation.

# Lets check cluster seperation topic wise.

# In[ ]:


def sub_tsne(topic):
    ai = f1[f1['Topic1']==topic]['id'].values
    return df[df.index.isin(ai)]


# In[ ]:


subplot = sub_tsne('Artificial Intelligence')


# In[ ]:


plt.scatter(subplot['x'],subplot['y'],s=0.4, alpha=0.4)


# As we can see there is uniform overlapping and separation takes place which is a good sign.

# One more

# In[ ]:


subplot = sub_tsne('Computation and Language')


# In[ ]:


plt.scatter(subplot['x'],subplot['y'],s=0.4, alpha=0.4)


# ### ***Some Drawbacks - ***
# * ### As I chose window size very less because many topics contains very little text. See below that topic contains literally no data so prediction will be hard.

# In[ ]:


tags[tags['FullTopic']=='Mathematical Software']['TopicExplain'].values


# * Web application based on flask is live on https://deviantpadam.pythonanywhere.com/
# * Source Code is availabe on https://github.com/DeviantPadam/rec_system

# # **Suggestions are heartily welcome.**
