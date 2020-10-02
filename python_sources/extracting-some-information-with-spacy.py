#!/usr/bin/env python
# coding: utf-8

# > # Extract some information with Spacy

# ## Initialize environment

# In[ ]:


import pandas as pd
import spacy
from spacy import displacy


# ## Load data

# In[ ]:


cols = list(pd.read_csv('../input/test_stage_1.tsv', delimiter='\t', nrows =1))
data = pd.read_csv('../input/test_stage_1.tsv', delimiter='\t', usecols =[i for i in cols if i != 'URL']).rename(columns={'Pronoun-offset': 'Pronoun_offset', 'A-offset': 'A_offset', 'B-offset': 'B_offset'})
data.head()


# ## Initialize Spacy NLP

# In[ ]:


nlp = spacy.load('en_core_web_lg')


# ## Apply NLP to the sentences

# In[ ]:


docs = list(map(nlp, data.Text))
docs[0]


# In[ ]:


docs[1]


# ## Extract token information from doc

# In[ ]:


doc_index = 1
doc = docs[doc_index]
tokens = pd.DataFrame(
    [[token.text, token.sent, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop, [child for child in token.children], [ancestor for ancestor in token.ancestors]] for token in doc],
    columns=['text', 'span', 'lemma', 'pos', 'tag', 'dep', 'shape', 'is_alpha', 'is_stop', 'child', 'ancestors'])
tokens


# ## Calculate pronoun Span

# In[ ]:


pronoun_span = doc.char_span(data.Pronoun_offset.get(doc_index), data.Pronoun_offset.get(doc_index) + len(data.Pronoun.get(doc_index)), label = 'Pronoun')
pronoun_tokens = pd.DataFrame(
    [[token.text, token.sent, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop, [child for child in token.children], [ancestor for ancestor in token.ancestors]] for token in pronoun_span],
    columns=['text', 'span', 'lemma', 'pos', 'tag', 'dep', 'shape', 'is_alpha', 'is_stop', 'child', 'ancestors'])
pronoun_tokens


# ## Calculate A Span

# In[ ]:


a_span = doc.char_span(data.A_offset.get(doc_index), data.A_offset.get(doc_index) + len(data.A.get(doc_index)), label = 'A')
b_span = doc.char_span(data.B_offset.get(doc_index), data.B_offset.get(doc_index) + len(data.B.get(doc_index)), label = 'B')
a_tokens = pd.DataFrame(
    [[token.text, token.sent, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop, [child for child in token.children], [ancestor for ancestor in token.ancestors]] for token in a_span],
    columns=['text', 'span', 'lemma', 'pos', 'tag', 'dep', 'shape', 'is_alpha', 'is_stop', 'child', 'ancestors'])
a_tokens


# ## Calculate B Span

# In[ ]:


b_span = doc.char_span(data.B_offset.get(doc_index), data.B_offset.get(doc_index) + len(data.B.get(doc_index)), label = 'B')
b_tokens = pd.DataFrame(
    [[token.text, token.sent, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop, [child for child in token.children], [ancestor for ancestor in token.ancestors]] for token in b_span],
    columns=['text', 'span', 'lemma', 'pos', 'tag', 'dep', 'shape', 'is_alpha', 'is_stop', 'child', 'ancestors'])
b_tokens


# In[ ]:





# # Display Spacy renders of sentences

# In[ ]:


sentence_spans = list(doc.sents)
displacy.render(sentence_spans, style='dep', jupyter=True, options ={ 'compact': True, 'distance': 100, 'bg': '#09a3d5', 'color': '#FFFFFF'})


# # Show the document tree

# In[ ]:


doc.print_tree()

