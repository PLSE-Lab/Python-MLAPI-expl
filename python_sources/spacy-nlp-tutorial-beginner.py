#!/usr/bin/env python
# coding: utf-8

# ****spaCy is a relatively new framework in the Python Natural Language Processing environment but it quickly gains ground and will most likely become the de facto library. There are some really good reasons for its popularity:****

# <img src="https://nlpforhackers.io/wp-content/uploads/2018/03/spaCy.png">

# # NLP TUTORIAL

# ## TOKENIZATION

# In[ ]:


import spacy
nlp = spacy.load('en')
doc = nlp('Hello     World!')
for token in doc:
    print('"' + token.text + '"')


# ## TOKEN CLASS ATTRIBUTE

# 

# In[12]:



doc = nlp("Next week I'll   be in Madrid.")
for token in doc:
    print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(
        token.text,
        token.idx,
        token.lemma_,
        token.is_punct,
        token.is_space,
        token.shape_,
        token.pos_,
        token.tag_
    ))


# ## SENTENCE DETECTION

# In[ ]:


doc = nlp("These are apples. These are oranges.")
for sent in doc.sents:
    print(sent)


# ## Part Of Speech Tagging

# In[ ]:



doc = nlp("Next week I'll be in Madrid.")
print([(token.text, token.tag_) for token in doc])


# ## Named Entity Recognition

# In[ ]:



doc = nlp("Next week I'll be in Madrid.")
for ent in doc.ents:
    print(ent.text, ent.label_)


# ****visualization of the Named Entity ****

# In[ ]:



from spacy import displacy
 
doc = nlp('I just bought 2 shares at 9 a.m. because the stock went up 30% in just 2 days according to the WSJ')
displacy.render(doc, style='ent', jupyter=True)
 


# ## Chunking

# In[ ]:



doc = nlp("Wall Street Journal just published an interesting piece on crypto currencies")
for chunk in doc.noun_chunks:
    print(chunk.text, chunk.label_, chunk.root.text)


# ## Dependency Parsing

# In[ ]:



doc = nlp('Wall Street Journal just published an interesting piece on crypto currencies')
 
for token in doc:
    print("{0}/{1} <--{2}-- {3}/{4}".format(
        token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))


# In[ ]:



from spacy import displacy
 
doc = nlp('Wall Street Journal just published an interesting piece on crypto currencies')
displacy.render(doc, style='dep', jupyter=True, options={'distance': 90})


# ****If this kernel was helpful don"t forget to upvote for another kernel share in this subject ^ ^ ****
