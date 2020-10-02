#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gensim
import keras
import nltk.corpus


# In[ ]:


if '/usr/share/nltk_data' in nltk.data.path:
    nltk.data.path.remove('/usr/share/nltk_data')
nltk.data.path.append('../input/')
nltk.data.path


# I will use the Brown Corpus to train a Doc2Vec model.

# In[ ]:


class BrownCorpus(object):
    def __init__(self):
        self.brown =  nltk.corpus.LazyCorpusLoader('brown', nltk.corpus.CategorizedTaggedCorpusReader, r'c[a-z]\d\d',
                                                    cat_file='cats.csv', tagset='brown', encoding="ascii",
                                                    nltk_data_subdir='brown-corpus/brown')
    def __iter__(self):
        for (tag,doc) in enumerate(self.brown.paras()):
            yield gensim.models.doc2vec.TaggedDocument(sum(doc,[]),[tag])
            
model = gensim.models.doc2vec.Doc2Vec(BrownCorpus(),
                                      dm_concat=1)


# In[ ]:


model.vector_size


# For each question/answer pair, I will infer a vector for each of question_title, question_body, and answer, and concatenate them.

# In[ ]:


def vectorize_cell(cell):
    return model.infer_vector(list(gensim.utils.tokenize(cell)))

def vectorize(data):
    return np.vstack([np.concatenate([vectorize_cell(row[cell])
                                      for cell in ('question_title','question_body','answer')])
                     for (i,row) in data.iterrows()])


# In[ ]:


training_data = pd.read_csv('../input/google-quest-challenge/train.csv',
                           index_col='qa_id')
training_data


# In[ ]:


training_vectors = vectorize(training_data)
training_vectors


# Now for a predictive model. I'll use a neural net with two dense hidden layers, each with 100 nodes and a softplus activation function. The output layer will have 30 nodes and a sigmoid activation function. Since I'm predicting continuous values, I'll use a Euclidean loss function.

# In[ ]:


predictor = keras.models.Sequential([keras.layers.Dense(100,
                                                       input_shape=(300,),
                                                       activation='softplus'),
                                     keras.layers.Dense(100,
                                                       activation='softplus'),
                                     keras.layers.Dense(30,
                                                       activation='sigmoid')])

predictor.compile(optimizer='nadam',
                  loss='mean_squared_error')


# In[ ]:


columns_to_predict = pd.read_csv('../input/google-quest-challenge/sample_submission.csv',
                                 index_col='qa_id').columns
predictor.fit(training_vectors,
              training_data.loc[:,columns_to_predict].values,
              epochs=100)


# Now let's try making some predictions

# In[ ]:


test_data = pd.read_csv('../input/google-quest-challenge/test.csv',
                        index_col='qa_id')
test_vectors = vectorize(test_data)
results = pd.DataFrame(predictor.predict(test_vectors),
                       index = test_data.index,
                       columns = columns_to_predict)
results


# In[ ]:


results.to_csv('submission.csv')

