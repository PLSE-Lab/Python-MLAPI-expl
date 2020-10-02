#!/usr/bin/env python
# coding: utf-8

# # ## COVID-19 Word2Vec Model and Vectors
# 
# The kernel reads CORD-19 data and trains a Word2Vec model on the body text of the various papers. This kernel has been implemented to support [this kernel ](https://www.kaggle.com/uplytics/thematic-analysis-of-risks-with-evidence-gap-maps). The model and embeddings are available as part of the kernel output to be included in other kernels. 
# 
# The approach to train this separately has been taken to avoid RAM overflow issues which has been expereincd while training Word2Vec model, integrated into a larger kernel. Options to decrease vector size, vocabulary size or increasing the min count parameters of the Gensim Word2Vec model to reduce RAM requirements. However in this kernel , iteratively the Word2Vec model is trained to ensure RAM usage stays below the 16GB Kaggle threshold.
# 
# This kernel only removed stopwords and does not use any stemming & lematization as the purpose is to support regex based keyword extraction !!
# 
# The configuration option used for tuning the Word2vec are obtained from the kernel given [here](https://www.kaggle.com/vahetshitoyan/word-and-phrase-associations-in-cord-19-corpus) 
# 

# In[ ]:


import pandas as pd
import numpy as np
import re
import glob
import json
from tqdm import tqdm


# In[ ]:


# Code for loading JSON into a dataframe adopted from https://www.kaggle.com/maksimeren/covid-19-literature-clustering
# Load all Json 
root_path = '/kaggle/input/CORD-19-research-challenge/'
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
print(len(all_json))
    
#A File Reader Class which loads the json and make data available
class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            try:
                if content['abstract']:
                    for entry in content['abstract']:
                        self.abstract.append(entry['text'])  
            except KeyError:
                #do nothing
                pass 
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
        
dict_ = None
dict_ = {'paper_id': [], 'abstract': [], 'body_text': []}
for idx, entry in enumerate(all_json):
    if idx % (len(all_json) // 10) == 0:
         print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)
    dict_['paper_id'].append(content.paper_id)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
        
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text'])
    


# In[ ]:


df_covid.head()


# In[ ]:


import gensim
import datetime
from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess
from gensim.sklearn_api.phrases import PhrasesTransformer
from gensim.models import Word2Vec
# Commented out as the EpocLogger creates issues when the model is loaded into a new kernel as it is not supported
#class EpochLogger(gensim.models.callbacks.CallbackAny2Vec):
#    """Callback to log information about training."""
#    def __init__(self):
#       self.epoch = 0

#def on_epoch_begin(self, model):
#        print("{} Starting epoch #{}".format(
#            datetime.datetime.now(), self.epoch))
    
#    def on_epoch_end(self, model):
#        print("{} Finished epoch #{}".format(
#            datetime.datetime.now(), self.epoch))
#        self.epoch += 1


sentences = []
for text in tqdm(df_covid['body_text'].iloc[0:9999]):
    sentences += [simple_preprocess(sentence) for sentence in sent_tokenize(gensim.parsing.preprocessing.remove_stopwords(text))]
bigrams = PhrasesTransformer(min_count=20, threshold=50)
model = Word2Vec(bigrams.fit_transform(sentences), 
                 size=200, 
                 window=8, 
                 min_count=10,
                 sg=True,
                 hs=False,
                 alpha=0.01,
                 sample=0.0001,
                 negative=15,
                 workers=4, 
                 #callbacks=[EpochLogger()],
                 iter=4)
    
sentences = []
for text in tqdm(df_covid['body_text'].iloc[10000:19999]):
    sentences += [simple_preprocess(sentence) for sentence in sent_tokenize(gensim.parsing.preprocessing.remove_stopwords(text))]
model.train(sentences, total_words=len(sentences),  epochs=model.epochs)

sentences = []
for text in tqdm(df_covid['body_text'].iloc[20000:]):
    sentences += [simple_preprocess(sentence) for sentence in sent_tokenize(gensim.parsing.preprocessing.remove_stopwords(text))]
model.train(sentences, total_words=len(sentences),  epochs=model.epochs)

    
model.save('covid19word2vec.model')


# In[ ]:


# Enter words in lower case
print(model.wv.most_similar('covid', topn=10))


# In[ ]:


print(model.wv.most_similar('death', topn=10))


# In[ ]:


Code to load the word2vec model after adding the output of this kernel from Add Data -> 


# In[ ]:


import gensim
word2vec_root_path = '/kaggle/input/covid-19-word2vec-model-and-vectors/'
word2vec_filename = 'covid19word2vec.model'
word2vecfile =  word2vec_root_path + word2vec_filename
model = gensim.models.Word2Vec.load(word2vecfile)


# In[ ]:


print(model.wv.most_similar('diabetes', topn=10))

