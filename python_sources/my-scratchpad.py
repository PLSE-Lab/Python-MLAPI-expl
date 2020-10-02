#!/usr/bin/env python
# coding: utf-8

# LB0.086

# In[ ]:


import numpy as np 
import pandas as pd 
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from subprocess import check_output
from nltk.stem import WordNetLemmatizer
print(check_output(["ls", "../input"]).decode("utf8"))


# # Load Data

# In[ ]:


bio = pd.read_csv("../input/biology.csv")
cook = pd.read_csv("../input/cooking.csv")
crypto = pd.read_csv("../input/crypto.csv")
diy = pd.read_csv("../input/diy.csv")
robot = pd.read_csv("../input/robotics.csv")
travel = pd.read_csv("../input/travel.csv")
sample_sub = pd.read_csv("../input/sample_submission.csv")
test = pd.read_csv("../input/test.csv")


all_dat = [bio,cook,crypto,diy,robot,travel]


# # Title Cleaning

# In[ ]:


swords1 = stopwords.words('english')

punctuations = string.punctuation

def title_clean(data):
    title = data.title
    title = title.apply(lambda x: x.lower())
    print('Remove Punctuations')
    # title = [' '.join(word.strip(punctuations) for word in i.split()) for i in title]
    title = title.apply(lambda x: re.sub(r'^\W+|\W+$',' ',x))
    title = title.apply(lambda i: ''.join(i.strip(punctuations))  )
    print('tokenize')
    title = title.apply(lambda x: word_tokenize(x))
    print('Remove stopwords')
    title = title.apply(lambda x: [i for i in x if i not in swords1 if len(i)>2])
    print('minor clean some wors')
    title = title.apply(lambda x: [i.split('/') for i in x] )
    title = title.apply(lambda x: [i for y in x for i in y])
    print('Lemmatizing')
    wordnet_lemmatizer = WordNetLemmatizer()
    title = title.apply(lambda x: [wordnet_lemmatizer.lemmatize(i,pos='v') for i in x])
    title = title.apply(lambda x: [i for i in x if len(i)>2])
    return(title)


# In[ ]:


test.title = title_clean(test)


# # Content Cleaning

# In[ ]:


def content_clean(data):
    content = data.content
    content = content.apply(lambda x: x.lower())
    print('Remove <>')
    content = content.apply(lambda x: re.sub(r'\<[^<>]*\>','',x))
    print('Remove n')
    content = content.apply(lambda x: re.sub(r'\n','',x))
    print('tokenize')
    content = content.apply(lambda x: word_tokenize(x))
    print('Remove stopwords')
    content = content.apply(lambda x: [i for i in x if i not in swords1 if len(i)>2])
    print('Lemmatizing')
    wordnet_lemmatizer = WordNetLemmatizer()
    content = content.apply(lambda x: [wordnet_lemmatizer.lemmatize(i,pos='v') for i in x])
    content = content.apply(lambda x: [i for i in x if len(i)>2])
    print('further cleaning')
    content = content.apply(lambda x: [''.join(j for j in i if j not in punctuations) for i in x])
    content = content.apply(lambda x: [i for i in x if len(i)>2])
    return(content)
        
test.content = content_clean(test)

  


# # Tfidf top terms for content:

# In[ ]:


from gensim import corpora
from gensim import models
import gensim
import numpy as np
content = test.content
dictionary = corpora.Dictionary(content)


# In[ ]:


corpus = [dictionary.doc2bow(text) for text in content]

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]


# In[ ]:


good_corpus = []


# In[ ]:



for doc in corpus_tfidf:
    doc_dat = [(dictionary.get(item[0]),item[1]) for item in doc]
    good_corpus.append(doc_dat)


# In[ ]:


good_corpus2 = [ sorted(i,key=lambda x: x[1],reverse=True) for i in good_corpus]


# In[ ]:




good_corpus3 = [term_list[:10] for term_list in good_corpus2]


# In[ ]:


good_corpus4 = []

for item_list in good_corpus3:
    good_term = [i[0] for i in item_list]
    good_corpus4.append(good_term)


# # POS Tagging Title:

# In[ ]:


tags1 = [nltk.pos_tag(x) for x in test.title]
tags2 = []
for taglist in tags1:
    goodterm = [i[0] for i in taglist if i[1][0] in "N"]
    tags2.append(goodterm)


# In[ ]:


tags2[:5]


# In[ ]:


title_corpus = [" ".join(terms)  for terms in tags2]

sub_title = pd.DataFrame(
    {
        'id': test.id,
        'tags': title_corpus
    
  
    })

sub_title.to_csv('mysub_title.csv',index=False)


# # POS Tagging Content:

# In[ ]:


tags_c = [nltk.pos_tag(x) for x in good_corpus4]


# In[ ]:


tags_c[:5]


# In[ ]:


tags_c2 = []
for taglist in tags_c:
    goodterm = [i[0] for i in taglist]
    tags_c2.append(goodterm)


# In[ ]:


good_corpus_cont = [" ".join(terms)  for terms in tags_c2]
sub_cont = pd.DataFrame(
    {
        'id': test.id,
        'tags': good_corpus_cont
    
  
    })


# In[ ]:


sub_cont.to_csv('mysub_content.csv',index=False)


# # Other Submissions:

# In[ ]:


title_tag = []

for taglist in tags2:
    goodterm = [i[0] for i in taglist if i[1][0] in "N"]
    title_tag.append(goodterm)
    
    
good_corpus5 = [" ".join(terms)  for terms in tags_c2]
sub = pd.DataFrame(
    {
        'id': test.id,
        'tags': good_corpus5
    
  
    })


# In[ ]:





# In[ ]:


final_res=[]

for i in range(len(tags2)):
    res = list(set(tags_c2[i]+tags2[i]))
    final_res.append(res)
    
    


# In[ ]:


final_res[:2]


# In[ ]:


good_corpus_all = [" ".join(terms)  for terms in final_res]
sub_all = pd.DataFrame(
    {
        'id': test.id,
        'tags': good_corpus_all
    
  
    })


# In[ ]:


sub_all.to_csv('mysub_together.csv',index=False)


# In[ ]:




