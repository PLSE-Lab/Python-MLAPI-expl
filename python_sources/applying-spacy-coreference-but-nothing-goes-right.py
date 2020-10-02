#!/usr/bin/env python
# coding: utf-8

# I tried to make spaCy work for this task, I am still convinced that there is a way to do it but it seemed like too much work. Eventually gave up and tried to see if potentially just picking the closest word everytime would be a decent baseline, but it turns out it is worse than the all .33 predictions. 

# In[ ]:


import numpy as np
import pandas as pd


# ## Load GAP Coreference Data
# 
# 

# In[ ]:


gap_train = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv", delimiter='\t')
gap_test = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv", delimiter='\t')
gap_valid = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv", delimiter='\t')


# ## Load Competition Data

# In[ ]:


test_stage_1 = pd.read_csv('../input/test_stage_1.tsv', delimiter='\t')
sub = pd.read_csv('../input/sample_submission_stage_1.csv')


# In order to make the coreference model load correctly we need to install the correct specific version of the neuralcoref model, cymem and spacy. Order is also important. If you install neuralcoref after it will not work. 

# In[ ]:


# !pip install https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_md-3.0.0/en_coref_md-3.0.0.tar.gz
# !pip install cymem==1.31.2 spacy==2.0.12


# In[ ]:


# import en_coref_md
# from spacy.tokens import Doc


# In[ ]:


# nlp = en_coref_md.load()


# Writing a custom tokenizer to replace SpaCy's. This one will simply split on spaces. SpaCy's is good, but it makes it much more complicated if the tokenizer is changing the character lengths because that is part of the information we have for knowing where the pronouns and referenced terms are. 

# In[ ]:


# class WhitespaceTokenizer(object):
#     def __init__(self, vocab):
#         self.vocab = vocab
#     def __call__(self, text):
#         words = text.split(' ')
#         words = [word for word in words]
#         spaces = [True] * len(words)
#         return Doc(self.vocab, words=words, spaces=spaces)
# nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


# In[ ]:


# gap_train.iloc[0:2, :]


# Wrote a ridiculous function to try to line up the character offset to actually match the words we are looking for. It resolves the correct words in all but 12 pronouns. Have not spent much time trying to figure out why these few do not resolve correctly, but my hypothesis is places where there are double spaces. 

# In[ ]:


# def check_coref(row):
#     text = row["Text"]
#     words = text.split()

#     pronoun = row["Pronoun"]
#     pronoun_off = row["Pronoun-offset"]
#     A = row["A"]
#     len_a = len(A.split())
#     A_off = row["A-offset"]
#     B = row["B"]
#     len_b = len(B.split())
#     B_off = row["B-offset"]
#     position = 0
#     for i, word in enumerate(words):
#         if position == pronoun_off:
#             pronoun_word_index = i
#         if position == A_off:
#             A_off_word_index = (i, i+len_a)
#         if position == B_off:
#             B_off_word_index = (i, i+len_b)
#         position += len(word) + 1
#     #print(A_off_word_index, B_off_word_index, pronoun_word_index)
#     doc = nlp(text)
#     token = None
#     try:
#         token = doc[pronoun_word_index]
#     except:
#         print(pronoun, pronoun_off)
#     try:
#         print(token, A, B, token._.coref_clusters)
#     except:
#         return [0, 0, 1]
    
# gap_train.apply(check_coref, axis=1)


# In[ ]:


# test_stage_1


# After giving up on that I tried a much simpler approach of just weighting the decisions based on how far they are from the pronouns. 
# 
# I.e (pronoun at 271, A at 298, B at 341)
# 
# A_dist = 298 - 271
# 
# B_dist = 341 - 271
# 
# total dist = A_dist(27) +B_dist( 70)
# 
# a_val = A_dist(27)/total_dist(97)
# 
# b_val = B_dist(70)/total_dist(97)
# 
# 
# This will calculate weights to give A and B. I weighted neither to simply always be .5 so that it balances with the other two since they will always add up to 1. In this competition we dont need the probability to sum to 1 since it is renormalized on a per row basis according to the rule section. 
# 

# In[ ]:


def measure_dist(row):
    pro_off = row["Pronoun-offset"]
    a_off = row["A-offset"]
    b_off = row["B-offset"]
    a_dist = np.abs(pro_off - a_off)
    b_dist = np.abs(pro_off - b_off)
    dist_tot = a_dist + b_dist
    a_val = a_dist/dist_tot
    b_val = b_dist/dist_tot
    neither = .5
    return [a_val, b_val, neither]
test_stage_1["preds"] = test_stage_1.apply(measure_dist, axis = 1)


# In[ ]:


test = test_stage_1.preds.apply(pd.Series)


# In[ ]:


sub["A"] = test[0]
sub["B"] = test[1]
sub["NEITHER"] = test[2]


# In[ ]:


sub[['ID', 'A', 'B', 'NEITHER']].to_csv('submission.csv', index=False)


# In[ ]:




