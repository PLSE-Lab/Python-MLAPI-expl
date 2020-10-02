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
for dirname, _, filenames in os.walk('/kaggle/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# making data frame from csv file 
data = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv") 
data = data[data['abstract'].notnull()]
data.head()


# In[ ]:


import spacy
from collections import Counter
from string import punctuation
nlp = spacy.load("en_core_web_lg")


# In[ ]:


def top_sentence(text, limit):
    keyword = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    doc = nlp(text.lower())
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            keyword.append(token.text)
    
    freq_word = Counter(keyword)
    max_freq = Counter(keyword).most_common(1)[0][1]
    for w in freq_word:
        freq_word[w] = (freq_word[w]/max_freq)
        
    sent_strength={}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent]+=freq_word[word.text]
                else:
                    sent_strength[sent]=freq_word[word.text]
    
    summary = []
    
    sorted_x = sorted(sent_strength.items(), key=lambda kv: kv[1], reverse=True)
    
    counter = 0
    for i in range(len(sorted_x)):
        summary.append(str(sorted_x[i][0]).capitalize())

        counter += 1
        if(counter >= limit):
            break
            
    return ' '.join(summary)


# In[ ]:


text="Human beings are social creatures and have always valued the importance of friends in their lives. To celebrate this noble feeling it was deemed fit to have a day dedicated to friends and friendship. Accordingly,first Sunday of August was declared as a holiday in USin honor of friends by a Proclamation made by US Congress in 1935. Since then, World Friendship Day is being celebrated every year on the first Sunday in the month of August.This beautiful idea of celebrating Friendship Day was joyfully accepted by several other countries across the world. And today, many countries including India, celebrate the first Sunday of August as Friendship Day every year. Celebrating Friendship Day in a traditional manner, people meet their friends and exchange cards and flowers to honor their friends. Lot many social and cultural organization too celebrate the occasion and mark Friendship Day by hosting programs and get together."


# In[ ]:


text


# In[ ]:



print(top_sentence(text,3))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import nltk
from nltk.util import ngrams
 
# Function to generate n-grams from sentences.
def extract_ngrams(data, num):
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [ ' '.join(grams) for grams in n_grams]
 


# In[ ]:


pip install rouge


# In[ ]:


import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import mglearn
import re,math
import spacy
import gensim
from gensim.models import CoherenceModel
import gensim.corpora as corpora
import rouge
#from pyrouge import Rouge155

#r = Rouge155()

WORD=re.compile(r'\w+')


# In[ ]:


from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS


# In[ ]:


def get_cosine(actual,created):
    intersection=set(actual.keys()) & set(created.keys())
    numerator=sum([actual[x]*created[x] for x in intersection])
    sum1=sum([actual[x]**2 for x in actual.keys()])
    sum2=sum([created[x]**2 for x in created.keys()])
    denominator=math.sqrt(sum1)* math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    else:
        return float(numerator)/denominator

def text_vector(text):
    words=WORD.findall(text)
    return Counter(words)

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
    
    
def make_bigrams(texts,bigram_mod):
   ## bigram_mod = gensim.models.phrases.Phraser(bigram)
            
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
   # trigram_mod = gensim.models.phrases.Phraser(trigram)
    return [trigram_mod[bigram_mod[doc]] for doc in texts]



def prepare_results(p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


# In[ ]:


vect=CountVectorizer(ngram_range=(1,1),stop_words='english')


# In[ ]:


def lda_impsentence(created_summary):
    created_summary=[created_summary]
    
    dtm=vect.fit_transform(created_summary)
    pd.DataFrame(dtm.toarray(),columns=vect.get_feature_names())
    lda=LatentDirichletAllocation(n_components=5)
    lda.fit_transform(dtm)
    sorting=np.argsort(lda.components_)[:,::-1]
    features=np.array(vect.get_feature_names())
    mglearn.tools.print_topics(topics=range(3), feature_names=features,
    sorting=sorting, topics_per_chunk=5, n_words=4)
            
            
            
            
    data_words=created_summary
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

            
            
            
    data_words_bigrams = make_bigrams(created_summary,bigram_mod)
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    id2word = corpora.Dictionary(data_lemmatized)
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)
            
            
           
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    return lda_model
            
            


# In[ ]:


import csv

abstract=data['abstract']
with open('topics.csv', 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        
       
        writer.writerow(["Abstract","Important Sentence","Topic Modeling"])
for i in abstract:
     with open('topics.csv', 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        
        res=top_sentence(str(i),1)
        lda=lda_impsentence(str(res))
        ngram=extract_ngrams(str(lda),4)
        print("Ngram is")
        print(lda)
       
        writer.writerow([i,res,lda])
        #print(res)
 

