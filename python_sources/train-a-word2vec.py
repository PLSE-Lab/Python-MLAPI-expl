#!/usr/bin/env python
# coding: utf-8

# # Purpose
# 
# I have no background whatsoever on medicine. So just to approach the dataset, I decided to train a Word2Vec so I can get around the jargon
# 
# I'm training the model. Will share the trained binaries once it finishes training
# 
# # Pretrained model
# 
# You can download the pretrained model from [here](https://www.kaggle.com/elsonidoq/covid19-challenge-trained-w2v-model)
# 
# Load the model with [this code](https://www.kaggle.com/elsonidoq/checkout-the-covid-19-word2vec-model)

# In[ ]:


from pathlib import Path

DATA_PATH = Path('/kaggle/input/CORD-19-research-challenge/2020-03-13/')
JUST_SOME = True # helpful for testing the code with small data


# In[ ]:


from tqdm.auto import tqdm

import json

def iter_texts():
    """
    Iterate over all directories, all file names, and yield all elements on body_text and abstract
    """
    dirs = 'comm_use_subset noncomm_use_subset pmc_custom_license biorxiv_medrxiv'.split()
    for dir in dirs:
        fnames = (DATA_PATH / dir / dir).glob('*')
        for fname in fnames:
            with fname.open() as f:
                content = json.load(f)
                
            for key in 'abstract body_text'.split():
                for row in content[key]:
                    yield row['text']


# In[ ]:


import spacy

# make sure to run python3 -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

def iter_sents(just_some=False):
    """
    Use spacy to tokenize what's yielded by iter_sents
    """
    for i, text in enumerate(iter_texts()):
        if just_some and i == 1000: break
        doc = nlp(text)
        for sent in doc.sents:
            yield [t.text.lower() for t in sent if not t.is_stop and t.is_alpha and len(t.text) > 1]


# In[ ]:


import json

# dump on a jsonlines file so we do the tokenization just once
with open('all_sentences.jl', 'w') as f:
    for i, sent in enumerate(tqdm(iter_sents(just_some=JUST_SOME))):
        if i > 0: f.write('\n')
        f.write(json.dumps(sent))


# In[ ]:


class CachedSentenceIterator:
    """
    An iterator that is compatible with gensim (you need to be able to iterate this more than once for the epochs to work)
    """
    def __init__(self, just_some=False, fname='all_sentences.jl'): 
        self.just_some = just_some
        self.fname = fname
    
    def __iter__(self):
        with open(self.fname) as f:
            for line in f:
                yield json.loads(line)
        


# In[ ]:


from gensim.models import Word2Vec

si = CachedSentenceIterator(just_some=JUST_SOME)

model = Word2Vec()
model.build_vocab(sentences=tqdm(si))
model.train(tqdm(si), total_examples=model.corpus_count, epochs=3)


# In[ ]:


model.save('covid.w2v')


# In[ ]:


model.wv.most_similar('virus')


# In[ ]:


model.wv.most_similar('coronavirus')

