#!/usr/bin/env python
# coding: utf-8

# # S p a C y  1 0 1
# 
# ```
# All features described in SpaCy 101 Course with the News DataSet
# ```

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import spacy

print(f'spacy = {spacy.__version__}')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/repository/ZNClub-PA-ML-AI-Sentiment-analysis-using-Business-News-82d860a/data"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/repository/ZNClub-PA-ML-AI-Sentiment-analysis-using-Business-News-82d860a/data/processed/normalized.csv")

#limit df
df = df[:5]
row_index = 1
df.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nclean_body = lambda x: re.sub("[\\n\\t\\r]", "", x) if isinstance(x, str) else ""\n\ndf[\'body\'] = df[\'body\'].apply(clean_body)\n')


# ## Language Model

# In[ ]:


from spacy.lang.en import English
nlp = English()
df['Doc'] = df['body'].apply(nlp)


# ### NLP Model creates Document object

# In[ ]:


row_index = 1
doc = df['Doc'][row_index]
doc


# ### Document creates Span object

# In[ ]:


span = doc[0:8]
span


# ### Lexical Analysis

# In[ ]:


print('Index:   ', [token.i for token in span])
print('Text:    ', [token.text for token in span])

print('is_alpha:', [token.is_alpha for token in span])
print('is_punct:', [token.is_punct for token in span])
print('like_num:', [token.like_num for token in span])


# ## Statistical models

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nnlp = spacy.load('en_core_web_sm')\ndf['StatsDoc'] = df['body'].apply(nlp)")


# In[ ]:


doc = df['StatsDoc'][row_index]
doc


# ### POS: Parts of Speech

# In[ ]:


pos_df = pd.DataFrame()

for i, token in enumerate(doc):
    pos_df.loc[i, 'text'] = token.text
    pos_df.loc[i, 'lemma'] = token.lemma_,
    pos_df.loc[i, 'pos'] = spacy.explain(token.pos_)
    pos_df.loc[i, 'tag'] = token.tag_
    pos_df.loc[i, 'dep'] = token.dep_
    pos_df.loc[i, 'shape'] = token.shape_
    pos_df.loc[i, 'is_alpha'] = token.is_alpha
    pos_df.loc[i, 'is_stop'] = token.is_stop
    pos_df.loc[i, 'is_punctuation'] = token.is_punct
    
pos_df.head(20)


# In[ ]:


pos = pos_df.groupby('pos')['text'].count().reset_index()

pos.plot(x='pos' , y='text', kind='bar' )


# ### NER: Named Entity Recognition

# In[ ]:


ent_df = pd.DataFrame()

for i, token in enumerate(doc.ents):
    ent_df.loc[i, 'entity'] = token.text
    ent_df.loc[i, 'label'] = token.label_
    ent_df.loc[i, 'recognition'] = spacy.explain(token.label_)
ent_df.head(20)


# In[ ]:


ent = ent_df.groupby('label')['entity'].count().reset_index()

ent.plot(x='label' , y='entity', kind='bar' )


# In[ ]:


spacy.displacy.render(doc, style='ent',jupyter=True)


# ### Token Dependency

# In[ ]:


dep_df = pd.DataFrame()

for i, token in enumerate(doc):
    dep_df.loc[i, 'token'] = token.text
    dep_df.loc[i, 'label'] = token.dep_
    dep_df.loc[i, 'dependency'] = spacy.explain(token.dep_)
dep_df.head(20)


# In[ ]:


dep = dep_df.groupby('label')['token'].count().reset_index()

dep.nlargest(columns=['token'], n=10).plot(x='label' , y='token', kind='bar' )


# In[ ]:


spacy.displacy.render(doc, style='dep', jupyter=True,options={'distance': 80})
spacy.displacy.render(nlp('Kaggle is fun, keep kaggling'), style='dep', jupyter=True,options={'distance': 80})


# ### Similarity

# In[ ]:


get_ipython().run_cell_magic('time', '', "# `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors\n# df['Similarity'] = df['StatsDoc'].apply(lambda x: x.similarity(doc)) \n\nnlp = spacy.load('en_core_web_lg')\ndf['LargeStatsDoc'] = df['body'].apply(nlp)")


# In[ ]:


doc = df['LargeStatsDoc'][row_index]
doc


# ### Matcher

# #### RuleMatcher

# In[ ]:


from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)


# ```ScratchPad```

# In[ ]:


# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
name = 'Nevil'
name = 'James'
test_docs = [
    f'Hey this is {name}',
    f'Hi I am {name} from Mumbai',
    f'Hi this is {name} joining from Mumbai'
]

single_doc = ''
for each in test_docs:
    single_doc = single_doc + '; ' + each

processed_test_docs = [nlp(each) for each in test_docs]

doc = processed_test_docs[0]
doc = nlp(single_doc)

ent_df = pd.DataFrame()

for i, token in enumerate(doc.ents):
    ent_df.loc[i, 'entity'] = token.text
    ent_df.loc[i, 'label'] = token.label_
    ent_df.loc[i, 'recognition'] = spacy.explain(token.label_)
ent_df.head(20)


# In[ ]:



pos_df = pd.DataFrame()

for i, token in enumerate(doc):
    pos_df.loc[i, 'text'] = token.text
    pos_df.loc[i, 'lemma'] = token.lemma_,
    pos_df.loc[i, 'pos'] = spacy.explain(token.pos_)
    pos_df.loc[i, 'pos_id'] = token.pos_
    pos_df.loc[i, 'tag'] = token.tag_
    pos_df.loc[i, 'dep'] = token.dep_
    pos_df.loc[i, 'shape'] = token.shape_
    pos_df.loc[i, 'is_alpha'] = token.is_alpha
    pos_df.loc[i, 'is_stop'] = token.is_stop
    pos_df.loc[i, 'is_punctuation'] = token.is_punct
    
pos_df


# #### Example
# 
# - [Parts of Speech Annotations](https://spacy.io/api/annotation)
# - [Operators](https://spacy.io/usage/rule-based-matching#quantifiers)
# 

# In[ ]:



pattern1 = [
    {"POS": "INTJ", "OP": "+"},
    {"POS": "DET", "OP": "?"},
    {"POS": "PRON", "OP": "?"},
    {"POS": "VERB", "OP": "+"},
    {"POS": "PROPN", "OP": "+"},
]


matcher = Matcher(nlp.vocab)
matcher.add("MeetingGreeting", None, pattern1)

list_of_matches = [matcher(each) for each in processed_test_docs]
sentence_no_of_matched = []
for sentence_no, matches in enumerate(list_of_matches):
    for id, start, end in matches:
        print(f'Pattern={id} matched_at={processed_test_docs[sentence_no][start:end]}')
        sentence_no_of_matched.append(sentence_no)

for sentence_no in sentence_no_of_matched:
    doc = processed_test_docs[sentence_no]
    persons = [token for token in doc.ents if token.label_ == 'PERSON']
#     persons = [(token.text, token.label_) for token in doc.ents]
    print(f'DOC={doc} with NER={persons}')


# #### PhraseMatcher
