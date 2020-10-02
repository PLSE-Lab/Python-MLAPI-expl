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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # References:
# 
# https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
# 
# https://github.com/h2oai/h2o-3/blob/master/h2o-py/demos/word2vec_craigslistjobtitles.ipynb
# 
# http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
# 
# https://stackoverflow.com/questions/46971969/conversion-of-pandas-dataframe-to-h2o-frame-efficiently
# 
# https://github.com/h2oai/h2o-meetups/blob/master/2016_10_06_TrumpTweets_Meetup/python-nlp/Python-TF-IDF.ipynb
# 

# # Imports

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.word2vec import H2OWord2vecEstimator
import gc
from fuzzywuzzy import fuzz


# # Get Data

# In[ ]:


df = pd.read_csv("../input/train.csv")
df.head()


# In[ ]:


df_toxic = df[df["target"]>0.50]
print(df_toxic.shape)
df_toxic.head()


# In[ ]:


documents_toxic = df_toxic["comment_text"]
documents_toxic[0:10]

del df_toxic
gc.collect()


# In[ ]:


df_nontoxic = df[df["target"]<0.50].sample(frac=0.075, replace=False, random_state=1331)
print(df_nontoxic.shape)
df_nontoxic.head()


# In[ ]:


documents_nontoxic = df["comment_text"]
documents_nontoxic[0:10]

del df_nontoxic
gc.collect()


# # Clustering

# In[ ]:


def myfeaturing(documents,column_tag="_run1_",no_features=1000, no_topics=20, max_df=0.95, min_df=2,
                alpha=0.1, l1_ratio=0.5, max_iter=5, learning_offset=50):
    # NMF is able to use tf-idf
    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    
    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    
    # Run NMF
    nmf = NMF(n_components=no_topics, random_state=1, alpha=alpha, l1_ratio=l1_ratio, init='nndsvd').fit(tfidf)

    # Run LDA
    lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=max_iter, 
                                    learning_method='online', learning_offset=learning_offset,random_state=0).fit(tf)
    
    # display top 10 words for top 10 topics
    def display_topics(model, feature_names, no_top_words=10):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic %d:" % (topic_idx))
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))
            if topic_idx > 9:
                break
    
    display_topics(nmf, tfidf_feature_names)
    display_topics(lda, tf_feature_names)
    
    # small tfidf
    tfidf_max_features=20
    word_vectorizer = TfidfVectorizer(sublinear_tf=True,strip_accents='unicode',analyzer='word',
                                      token_pattern=r'\w{1,}',stop_words='english',ngram_range=(1, 1),
                                      max_features=tfidf_max_features)
    word_vectorizer.fit(documents)
    
    # scoring
    
    def myscoring(df):
        documents2 = df["comment_text"]

        tftran = tf_vectorizer.transform(documents2)
        out2 = lda.transform(tftran)

        tfidftran = tfidf_vectorizer.transform(documents2)
        out3 = nmf.transform(tfidftran)

        mycols = range(1,no_topics+1)
        out2pd = pd.DataFrame(data=out2, columns=["lda" + column_tag + str(mycol) for mycol in mycols])
        out3pd = pd.DataFrame(data=out3, columns=["nmf" + column_tag + str(mycol) for mycol in mycols])
        
        tfidf = word_vectorizer.transform(df['comment_text'])
        tfidf_df = pd.DataFrame(tfidf.toarray(), columns=word_vectorizer.get_feature_names())
        
        being_absurd = "Well could give address? I would happy send guy ilk house, grab stuff right front face run away it. You know scared want stuff yourself. Perhaps could invite scumbag dinner wine. Share smokes him. For love God, happened us?"
        political_rant = "Another opinionated non-headline again, could well leave trumps name say much-i.e. reason think play out? But oh no, fit adns mission every article anything regarding next administration (or republicans matter) MUST negative tone. What garbage."
        name_calling = "You need drunk moron."
        political_slight = "He wants Obama's black privilege Hillary's ability rig elections."
        black_insensitive = "There 3 white soldiers killed well. I've heard read nothing next kin news media. Here clue...... While I've tragedies surely empathize widow, pattern life. For past 30-35 years, I see black American's complaining everything.....but never black black crime. They 'leaders' always complaining others, never take responsibility actions. Not media. It happens job. It happens schools. On buses. Every perceived slight 'I get respect'.....unless black person them. I mid-60's civil rights marcher. That equal rights. For 50 years I've watched blacks get breaks affirmative action media coverage. Meanwhile, I white co-workers passed promotions, allowed fraction black employees get away daily. I'm empathized out."

        # create function for comparing string to other strings
        def my_str_compare(x):
            return fuzz.token_sort_ratio(x,my_string)

        # add features to training set
        my_string = being_absurd
        df['being_absurd'] = df['comment_text'].apply(my_str_compare)
        my_string = political_rant
        df['political_rant'] = df['comment_text'].apply(my_str_compare)
        my_string = name_calling
        df['name_calling'] = df['comment_text'].apply(my_str_compare)
        my_string = political_slight
        df['political_slight'] = df['comment_text'].apply(my_str_compare)
        my_string = black_insensitive
        df['black_insensitive'] = df['comment_text'].apply(my_str_compare) 
        
        df_fuzzy = df[['being_absurd','political_rant','name_calling','political_slight','black_insensitive']]
        
        frames = [out2pd,out3pd,tfidf_df,df_fuzzy]
        df = pd.concat(frames,axis=1)
        
        del out2pd, out3pd, out2, out3, frames, tfidf_df, tfidf
        
        return df
    
    df = pd.read_csv("../input/train.csv")
    df_train = myscoring(df)
    df = pd.read_csv("../input/test.csv")
    df_test = myscoring(df)
    
    
    del df 
    gc.collect()
    
    return df_train, df_test


# In[ ]:


df_train1, df_test1 = myfeaturing(documents=documents_toxic, column_tag="_run1_")

del documents_toxic
gc.collect()


# In[ ]:


frames = [df["target"],df_train1]
df_train = pd.concat(frames,axis=1)
df_train.head()


# In[ ]:


del frames, df_train1
gc.collect()


# # H2O

# In[ ]:


h2o.init()


# In[ ]:


train = h2o.H2OFrame(df_train)
train.head()


# In[ ]:


# Identify predictors and response
x = train.columns
y = "target"
x.remove(y)


# In[ ]:


# Run AutoML for 30 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models=20, seed=1331)
aml.train(x=x, y=y, training_frame=train)


# In[ ]:


# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)


# In[ ]:


# The leader model is stored here
aml.leader


# # Submission

# In[ ]:


test = h2o.H2OFrame(df_test1)


# In[ ]:


prediction = aml.leader.predict(test)


# In[ ]:


# create submission DataFrame
t_df = pd.read_csv("../input/test.csv")
submission = pd.DataFrame(t_df['id'])
submission.head()


# In[ ]:


pd_pred = prediction.as_data_frame()
pd_pred.head()


# In[ ]:


submission['prediction'] = pd_pred
submission.head()


# In[ ]:


submission.to_csv('submission.csv',index=False)


# In[ ]:




