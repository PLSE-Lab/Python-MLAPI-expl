#!/usr/bin/env python
# coding: utf-8

# #### Load Packages

# In[ ]:


from pprint import pprint

import spacy
import pandas as pd

import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

NUM_TOPICS = 5


# Having already tried simple LDA with various parameter combinations, I was unable to achieve a performance beyond 93. So I started exploring other algorithms for LDA. There is a *Mallet LDA* which is known to provide better quality of topics.

# In[ ]:


# load the spacy model
nlp = spacy.load("en_core_web_lg")
# unzip and load the mallet lda folder
mallet_path = 'mallet-2.0.8/bin/mallet' 


# #### Read data

# In[ ]:


data = pd.read_csv('data.csv').set_index('Id')


# In[ ]:


print (get_text_attributes(data['text'].values[0]))


# #### Common token patterns

# Based on multiple iterations of Topic Models, I noticed that there was one topic in particular which **was not appearing** very prominently in any of the five topics: "Tech News". So I removed those texts from the data which contained specific patterns and for which topics are not needed to be predicted based on empirical analysis of the text data. Removing cases for which we already know topics, **reduces noise from the data** and increases the chance for lesser prominent topics to show up.
# 
# For eg. the text containing the "pros" and "cons" are glassdoor reviews. Text containing "from" and "subject" are emails which contain conversations between a prospect and salesperson enquiring about cars. So that belongs to automobiles. We can leverage such patterns in the data to improve 

# In[ ]:


# Common patterns
glassdoor_ids = data.loc[data.text.str.startswith('pros:')].index.tolist()
automobile_ids = data.loc[data.text.str.startswith('from ') | data.text.str.startswith('subject ')].index.tolist()

data.drop(glassdoor_ids+automobile_ids, inplace=True)


# #### Extract text features

# The below function does the following:
# - Lemmatizing the tokens in text
# - Extracting the Named Entities of tokens
# 
# In order to retain the sentence structure, stopwords are not removed as we are using NER. 

# In[ ]:


def get_text_attributes(text):
    doc = nlp(text)
#   lemmatising tokens, converting working, worked, etc to work
    tokens = [token.lemma_ for token in doc] #if not ((token.is_punct)|(token.is_stop))
    
#   Get the entities (named entities) for each of the words
    entities = [ent.label_ for ent in doc.ents]
#     pos = [token.pos_ for token in doc]
    print(entities)
    pos = []
    attributes = tokens + entities + pos
    
    return attributes

# texts = data['text'].apply(get_text_attributes).tolist()


# The *lemmatised tokens* as well as the *Named Entities* of the tokens are returned in a list and that it is used as ahead to prepare the corpus and the dictionary (as shown below)

# In[ ]:


# out put of the function above
print (texts[0])


# Since Topic Models are bag of words models, it can benefitted by the extra information of enitites present in the text. For example: if ORG is occurring in a text (once or multiple times) there is a higher chance that it is a "tech news" or "glassdoor review". Hence information of entities is also passed in LDAMallet.

# #### Topic Model

# In[ ]:


dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model = gensim.models.wrappers.LdaMallet(mallet_path,\n                                         corpus=corpus,\n                                         num_topics=TOPICS,\n                                         id2word=dictionary,\n                                         iterations=50000,\n                                         workers=6)\n\ncoherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence=\'c_v\')\nprint(f"Coherence: {coherence_model.get_coherence()}")')


# #### Get topics

# In[ ]:


data['lda_topics'] = [sorted(doc_topics, key=lambda k: k[1], reverse=True)[0][0] for doc_topics in model[corpus]]
data['lda_topics'].value_counts()


# #### Map the topics to topic names

# In[ ]:


tech_ids = data[data['lda_topics'] == 4].index.tolist()
roomrental_ids = data[(data['lda_topics'] == 1) | (data['lda_topics'] == 2) | ((data['lda_topics'] == 5)| ((data['lda_topics'] == 0)))].index.tolist()
sportsnews_ids = data[data['lda_topics'] == 3].index.tolist()


# In[ ]:


data[data['lda_topics'] == 1]


# #### Create submission file

# In[ ]:


topic = ["glassdoor_reviews"]*len(glassdoor_ids) +["tech_news"]*len(tech_ids) +["room_rentals"]*len(roomrental_ids) +["sports_news"]*len(sportsnews_ids) +["Automobiles"]*len(automobile_ids) 

id = glassdoor_ids + tech_ids + roomrental_ids + sportsnews_ids + automobile_ids


# In[ ]:


pd.DataFrame({'topic': topic, 'Id': id}).set_index('Id').to_csv('submission_lda_reduced.csv', sep=',')

