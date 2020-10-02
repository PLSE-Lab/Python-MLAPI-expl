#!/usr/bin/env python
# coding: utf-8

# # Quick Look 1: Jaccard Similarity Technique for Short Text Comparisons
# 
# This notebook explores the simple text similarity technique known as jaccard similarity which is useful in many IR and ML contexts.
# 
# 
# #### Goals
# 1. Explore the Jaccard Coefficient.
# 2. Explore simple preprocessing with Spacy.
# 3. Take negation into account.
# 

# ## The Jaccard Coefficient
# 
# This is probably the most well known coefficient. It is the intersection of two sets divided by the union of both sets.
# 
# $$
# J(A,B) = \frac{\lvert A \cap B \rvert}{\lvert A \cup B \rvert}
# $$
# 
# For example, given the sentence *James buys milk at the grocery store* and *Jane does not buy milk* they would have a Jaccard Coefficient calculated as follows.
# 
# **1. Break up the sentences into tokens (Tokenization)**

# In[ ]:


import numpy as np
import spacy

sent1 = "James buys milk at the grocery store."
sent2 = "Jane does not buy milk at stores much."
tokens1 = [t.lower() for t in sent1.split(" ")]
tokens2 = [t.lower() for t in sent2.split(" ")]


# **2. Convert the tokens into sets and get their union and intersection.**

# In[ ]:


set1 = set(tokens1)
set2 = set(tokens2)
set_union = set1.union(set2)
set_intersection = set1.intersection(set2)
print(set_intersection)
print(set_union)


# **3. Divide the length of intersection set by the length of the union**

# In[ ]:


print("Jaccard Coefficient: {}".format( len(set_intersection) / len(set_union)))


# ### Ways to improve
# 
# We can make this better in many ways. 
# 
# We can remove common words known as **stopwords** and **lemmatize** the words to get rid of a lot of noise.
# 
# Lets do that.

# In[ ]:


# load a spacy model
nlp_en = spacy.load('en')

def preprocess(sentence):
    result = []
    tokens = nlp_en(sentence)
    for token in tokens:
        # remove determiners and punctuation or spaces 
        if token.pos_ in ['PUNCT','SPACE', 'DET']:
            continue
        else:
            # the lemma of pronouns is stored as '-PRON-' in spacy; we want the actual pronoun
            if token.pos_ == 'PRON':
                result.append(token.text.lower())
            else:
                result.append(token.lemma_)
    return result

print('preprocessed sent1: {}'.format(preprocess("I like you")))
print('preprocessed sent2: {}'.format(preprocess(sent2)))


# ## Writing up the Jaccard Coefficient Function
# 
# Now that we have a preprocessing pipeline let's write up the Jaccard Coefficient function.

# In[ ]:


def jaccard_coefficient(sent1, sent2, preprocessor=preprocess):
    tokens1 = preprocessor(sent1)
    tokens2 = preprocessor(sent2)
    set1 = set(tokens1)
    set2 = set(tokens2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

print(jaccard_coefficient(sent1,sent2))


# ### Testing
# 
# Let's see how well it works for some example sentences.

# In[ ]:


examples = [
    ["Jane wants to spend more time with Bob", "Bob wants to spend more time with Jane"],
    ["Apples are a healthy fruit", "People walk all around the neighborhood"],
    ["Jane is a good person", "Jane is not a good person"]
]

for pair in examples:
    print('S1: {}'.format(pair[0]))
    print('S2: {}'.format(pair[1]))
    print('Jaccard Coefficient: {}'.format(jaccard_coefficient(pair[0], pair[1])))


# Notice that not is being ignored let's fix that!

# In[ ]:


def preprocess_v2(sentence):
    result = []
    tokens = nlp_en(sentence)
    for token in tokens:
        if token.pos_ in ['PUNCT','SPACE', 'DET', 'PART', 'ADP']:
            continue
        else:
            # add 'neg_' prefix if a tokens dependency parse children are not/no/negation.
            # currently we are only negating one word. We could negate up to the next punctation in the sentence
            # or depending on the current POS tag or DEP parse we could design our own rules to include negation more
            prefix = ''
            if 'neg' in [t.dep_ for t in token.children]:
                prefix = 'neg_'
            label = ''
            if token.pos_ == 'PROPN':
                label = '{}{}'.format(prefix, token.text.lower())
            else:
                label = '{}{}'.format(prefix, token.lemma_)
            result.append(label)
            
    return result

for pair in examples:
    print('S1: {}'.format(pair[0]))
    print('S2: {}'.format(pair[1]))
    print('Jaccard Coefficient: {}'.format(jaccard_coefficient(pair[0], pair[1], preprocessor=preprocess_v2)))


# ### C'est la fin
# 
# That's it for tutorial 0001.
# 
# Hopefully this showed you how to work with spacy a bit. In the next tutorial we will be looking at other coefficient functions and preprocessing techniques.
