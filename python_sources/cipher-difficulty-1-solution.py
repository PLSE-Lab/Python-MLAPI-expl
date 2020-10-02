#!/usr/bin/env python
# coding: utf-8

# This notebook is based on https://www.kaggle.com/group16/cracking-the-code-difficulty-1.
# 
# Here, I present an algorithm for acquiring the perfect substitution instead of doing tedious manual correction.

# In[ ]:


import numpy  as np
import pandas as pd

import math
import time

from collections import Counter
from IPython.display import HTML


# ## The Data

# In[ ]:


ciphers_df = pd.read_csv('../input/test.csv')
ciphers_df = ciphers_df.query("difficulty==1")
ciphers_df['length'] = ciphers_df['ciphertext'].apply(lambda x: len(x))
ciphers_df.head()


# In[ ]:


plaintext_df = pd.read_csv("../input/training.csv", index_col='index')
plaintext_df['length'] = plaintext_df['text'].apply(lambda x: len(x))
plaintext_df['padded_length'] = (np.ceil(plaintext_df['length'] / 100) * 100).astype(int)
plaintext_df.head()


# ## Frequency Analysis

# In[ ]:


plaintext_corpus = ''.join(list(plaintext_df['text']))
ciphrtext_corpus = ''.join(list(ciphers_df['ciphertext']))


# In[ ]:


def count_characters(text):
    counter = Counter(text)
    df = pd.DataFrame(list(counter.items()), columns=['Character', 'Count'])
    df['Percent'] = df['Count'] / df['Count'].sum()
    df_sorted = df.sort_values(by='Count', ascending=False)
    return df_sorted


# In[ ]:


def display_counts(plain_counts, ciphr_counts, n=5):
    plain_html = wrap_html("PlainText", plain_counts, n)
    ciphr_html = wrap_html("Cipher",    ciphr_counts, n)
    display(HTML(plain_html + ciphr_html))

def wrap_html(name, df, n):
    return         "<div style='float: left; padding: 10px;'>" +             "<h3>" + name + "</h3>" +             df[:n].to_html() +             "..." +             df[-n:].to_html() +             str(df.shape) +         "</div>"


# Looks like a straightforward substitution cipher. We can already see a promising mapping:

# In[ ]:


plaintext_counts = count_characters(plaintext_corpus)
ciphrtext_counts = count_characters(ciphrtext_corpus)

display_counts(plaintext_counts, ciphrtext_counts)


# ## Substitution Cipher

# In[ ]:


class SubstitutionCipher:
    def __init__(self, plain_alphabet, ciphr_alphabet):
        self.decrypt_mapping = {}
        self.encrypt_mapping = {}
        
        for p, c in zip(plain_alphabet, ciphr_alphabet):
            self.update(p, c)

    def update(self, p, c):
        self.decrypt_mapping[c] = p
        self.encrypt_mapping[p] = c

    def encrypt(self, text):
        return self.substitute(text, self.encrypt_mapping)
    
    def decrypt(self, text):
        return self.substitute(text, self.decrypt_mapping)
    
    def substitute(self, text, mapping):
        result = [
            mapping[c]
            for c in list(text)
        ]
        return ''.join(result)


# Let's try decrypting one of the ciphers:

# In[ ]:


substitution = SubstitutionCipher(plaintext_counts['Character'], ciphrtext_counts['Character'])
substitution.decrypt(ciphers_df.iloc[2]['ciphertext'])


# Excellent! We're on the right track.

# ## Perfecting the substitution
# 
# The idea is to get a subset of cipher texts that contains all the cipher characters. Matching these to their plaintext should be easy with our current substitution mapping; we'll just look for matches with >90% accuracy. We'll also visually confirm that they are indeed correct matches.
# 
# Once we have that, we can now easily note which characters we have a correct substitution for and which characters we got wrong - and what its correct plain text should be.

# ### Get a subset of ciphers
# 
# First, we get a small subset of ciphers that contains all the characters. These characters must not be in the padding since we'd have no way to confirm if those are accurate. This is very tricky as we can't tell the padding length from the cipher text alone. So, we just assume 50 on both sides to be safe.
# 
# Let's create a dataframe that records whether each cipher character appears in each cipher text:

# In[ ]:


alphabet_per_cipher = [ set(text) for text in list(ciphers_df['ciphertext'].str[50:-50])]
character_presence  = pd.DataFrame([
    {
        c: c in alphabet
        for c in ciphrtext_counts['Character'].values
    }
    for alphabet in alphabet_per_cipher
], index=ciphers_df.index)
character_presence.head()


# Confirm that all characters are found:

# In[ ]:


character_presence.any().all()


# Below is an algorithm to quickly obtain a small subset of ciphers that contain all of the cipher characters.
# 
# It starts with including the cipher index#2, this decision is arbitrary. It then loops as long as all characters have not yet been represented, each time adding a cipher containing a previously unrepresented character.

# In[ ]:


subset_indexes = [2]

while not character_presence.loc[subset_indexes].any().all():
    ant = character_presence.loc[subset_indexes].any()
    unfound_characters = ant[ant == False]
    unfound_character  = unfound_characters.index[0]
    found_here = character_presence[character_presence[unfound_character]].index[0]
    subset_indexes.append(found_here)

display(len(subset_indexes))
display(subset_indexes)


# ### Match these ciphers with their plaintext

# In[ ]:


def analyze(decrypted, plaintext):
    corrects = []
    mistakes = {}

    for d,p in zip(decrypted, plaintext):
        if d == p:
            corrects.append(d)
        else:
            mistakes[d] = p
    
    return set(corrects), mistakes, len(corrects) / len(plaintext)


PRINT_LENGTH = 100
all_corrects = set()
all_mistakes = {}

for cipher_index, cipher_row in ciphers_df.loc[subset_indexes].iterrows():
    decrypted = substitution.decrypt(cipher_row['ciphertext'])

    plaintext_candidates = plaintext_df.query(f"padded_length=={cipher_row['length']}")
    for plaintext_index, plaintext_row in plaintext_candidates.iterrows():
        # work around the padding
        padding_left_length = math.floor((plaintext_row['padded_length'] - plaintext_row['length']) / 2)
        unpadded_decrypted_text = decrypted[padding_left_length:]

        corrects, mistakes, score = analyze(unpadded_decrypted_text, plaintext_row['text'])

        if score >= 0.9:
            match = (cipher_index, plaintext_index)

            all_corrects.update(corrects)
            all_mistakes.update(mistakes)

            print(f"Score: {score}")
            print(cipher_row['ciphertext'][padding_left_length:padding_left_length+PRINT_LENGTH])
            print(unpadded_decrypted_text[:PRINT_LENGTH])
            print(plaintext_row['text'][:PRINT_LENGTH])
            print(f"---")
            print(f"CorrectsVerified:{len(all_corrects)}; MistakesNoted:{len(all_mistakes)}")
            print(f"TotalCharactersAccountedFor: {len(all_corrects) + len(all_mistakes)} out of 85")
            print()


# Awesome. They're almost an exact match save for a few characters and we've recorded exactly which characters are wrong.

# ### Now we correct them
# 
# Correcting our substitution mapping is trivial:

# In[ ]:


mistakes = [
    (correct_plaintext, substitution.encrypt(wrong_plaintext))
    for wrong_plaintext, correct_plaintext in all_mistakes.items()
]

for correct_plaintext, cipher_character in mistakes:
    substitution.update(correct_plaintext, cipher_character)


# Now we can match all ciphers to their plaintext without requiring the arbitrary 90% accuracy threshold.
# 
# Note that this takes about an hour! Decrypting is fast but matching takes way too long because of the padding. 

# In[ ]:


matches = []

for cipher_index, cipher_row in ciphers_df.iterrows():
    decrypted = substitution.decrypt(cipher_row['ciphertext'])
    match = None
    
    plaintext_candidates = plaintext_df.query(f"padded_length=={cipher_row['length']}")
    for plaintext_index, plaintext_row in plaintext_candidates.iterrows():
        if plaintext_row['text'] in decrypted:
            match = (cipher_row['ciphertext_id'], plaintext_index)

    if match is None:
        print(f"No match found for {cipher_index}!")
    else:
        matches.append(match)

matches = pd.DataFrame(matches, columns=['ciphertext_id', 'index'])
matches.head()


# Perfect. Everything was found.

# ## Create submission

# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['index'] = -1 # totally not necessary but i don't want to get the 0-index right for the wrong reason
sub_higher_difficulty = sub[~sub['ciphertext_id'].isin(matches['ciphertext_id'])]

sub = pd.concat([matches, sub_higher_difficulty])
sub.to_csv('submission_diff1.csv', index=False)
sub.head()


# This should get us a score of:

# In[ ]:


len(ciphers_df) / len(plaintext_df)


# ## Add 1st level encryption to plaintext 
# 
# For use in the next level

# In[ ]:


plaintext_df['encrypt1'] = plaintext_df['text'].apply(lambda text: substitution.encrypt(text))
plaintext_df.to_csv('plaintext_encrypt1.csv')
plaintext_df.head()

