#!/usr/bin/env python
# coding: utf-8

# In this kernel I will explain what caused error during preprocessing *test_stage_2.tsv* and a fixed code doesn't affect training.

# In[ ]:


get_ipython().system('wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-development.tsv -q')
get_ipython().system('wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-test.tsv -q')
get_ipython().system('wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-validation.tsv -q')


# In[ ]:


import pandas as pd
import re
import spacy
from IPython.core.display import display, HTML


# In[ ]:


train_df = pd.concat([pd.read_csv("gap-test.tsv", index_col=0, delimiter="\t"),
                      pd.read_csv("gap-validation.tsv", index_col=0, delimiter="\t"),
                      pd.read_csv("gap-development.tsv", index_col=0, delimiter="\t")])
test_df = pd.read_csv("../input/test_stage_2.tsv", index_col=0, delimiter="\t")


# This is the original code.

# In[ ]:


nlp = spacy.load('en')
def get_sentence(text, offset, token_after="[PRONOUN]"):
    """
    Extract a sentence containing a word at position offset by character and
    replace the word with token_after.
    output: Transformed sentence
            token_before
            a pos tag of the word.
    """
    doc = nlp(text)
    # idx: Character offset
    idx_begin = 0
    for token in doc:
        if token.sent_start:
            idx_begin = token.idx
        if token.idx == offset:
            sent = token.sent.string
            pos_tag = token.pos_
            idx_token = offset - idx_begin
            break
    # word_s = sent[idx_token:].split()
    # n = len(sent)
    token_before = token.string.strip()
    subtxt_transformed = re.sub("^" + token_before, token_after, sent[idx_token:])
    sent_transformed = sent[:idx_token] + subtxt_transformed
    # n_diff = len(sent_transformed) - n - len(token_after) + len(token_before)
    return sent_transformed, token_before, pos_tag


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_preprocessed_before = []\nfor obj in train_df.iterrows():\n    train_preprocessed_before.append(get_sentence(obj[1]["Text"], obj[1]["Pronoun-offset"]))')


# No errors.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_preprocessed = []\nfor e, obj in enumerate(test_df.iterrows()):\n    test_preprocessed.append(get_sentence(obj[1]["Text"], obj[1]["Pronoun-offset"]))')


# The function assumes that **token.idx == offset** somewhere in doc, which is wrong.

# In[ ]:


ID = obj[0]
text = obj[1]["Text"]
offset = obj[1]["Pronoun-offset"]
html_text = "<BLOCKQUOTE>" + text[:offset] + "<font color='red'>" + text[offset:offset + 2] + "</font>" + text[offset + 2:] + "</BLOCKQUOTE>" 
display(HTML("An error occured during preprocessing ID: " +  "<I>" + ID + "</I>." + html_text))


# In[ ]:


doc = nlp(text)

print("Pronoun-offset:", offset)
for token in doc:
    if token.idx > offset - 10 and token.idx < offset + 10:
        print(token.idx, token.pos_, ":", token)
    


# The reason for the error is that spacy'tokenizer couldn't extract "**he**" at position 313 and *get_sentence* doesn't properly handle that case.
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'for obj in test_df.iloc[e + 1:].iterrows():\n    test_preprocessed.append(get_sentence(obj[1]["Text"], obj[1]["Pronoun-offset"]))')


# No error after that. Only one case caused error.
# 
# Here is a fixed code.

# In[ ]:


def get_sentence(text, offset, token_after="[PRONOUN]"):
    """
    Extract a sentence containing a word at position offset by character and
    replace the word with token_after.
    output: Transformed sentence
            A word starting at offset
            A pos tag of the word.
            If the word cannot be extracted it returns default values.
    """
    doc = nlp(text)
    # idx: Character offset
    idx_begin = 0
    sent = None
    for token in doc:
        if token.sent_start:
            idx_begin = token.idx
        if token.idx == offset:
            sent = token.sent.string
            pos_tag = token.pos_
            idx_token = offset - idx_begin
            break
    # word_s = sent[idx_token:].split()
    # n = len(sent)
    if sent is None:
        # Default values
        sent_transformed = token_after
        token_before = "it"
        pos_tag = "PRON"
    else:
        token_before = token.string.strip()
        subtxt_transformed = re.sub("^" + token_before, token_after, sent[idx_token:])
        sent_transformed = sent[:idx_token] + subtxt_transformed
    # n_diff = len(sent_transformed) - n - len(token_after) + len(token_before)
    return sent_transformed, token_before, pos_tag


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_preprocessed_after = []\nfor obj in train_df.iterrows():\n    train_preprocessed_after.append(get_sentence(obj[1]["Text"], obj[1]["Pronoun-offset"]))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_preprocessed = []\nfor obj in test_df.iterrows():\n    test_preprocessed.append(get_sentence(obj[1]["Text"], obj[1]["Pronoun-offset"]))')


# No errors.

# In[ ]:


all([before == after for before, after in zip(train_preprocessed_before, train_preprocessed_after)])


# The change doesn't affect training.
