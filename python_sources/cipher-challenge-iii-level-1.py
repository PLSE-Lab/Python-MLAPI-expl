#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# ### Time to share solution of cipher level 1 so you can look at the next level.
# In the previous Cipher Challenge II one of the levels was a cipher with multiple substitutions generated from a key of length 8 if I remember correct.
# The level 1 of this Cipher Challenge III is the same kind but with a key of length 4, so only 4 substitutions are used for each character mapping.
# 
# ## The cipher
# - The cipher only apllies to UPPERCASE and LOWERCASE letters.
# - The key only shifts every time an UPPERCASE or LOWERCASE letter is met.
# 
# ## Padding
# From Cipher Challenge II it was found that padding could be done both up front and in the end. Number of padding characters in the end was always equal to or at most 1 character more (if number of characters to pad with was odd) than the number of padding characters up front.
# 
# # Level 1 solution

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Read the train, test and sub files

# In[ ]:


train = pd.read_csv('../input/train.csv', index_col='plaintext_id')
test = pd.read_csv('../input/test.csv', index_col='ciphertext_id')
sub = pd.read_csv('../input/sample_submission.csv', index_col='ciphertext_id')


# In[ ]:


train['length'] = train.text.apply(len)
test['length'] = test.ciphertext.apply(len)


# In[ ]:


train[train['length']<=100]['length'].hist(bins=99)


# In[ ]:


train.head()


# In[ ]:


test.head(10)


# ## Functions to decrypt and encrypt from/to level 1

# In[ ]:


KEYLEN = 4 # len('pyle')
def decrypt_level_1(ctext):
    key = [ord(c) - ord('a') for c in 'pyle']
    key_index = 0
    plain = ''
    for c in ctext:
        cpos = 'abcdefghijklmnopqrstuvwxy'.find(c)
        if cpos != -1:
            p = (cpos - key[key_index]) % 25
            pc = 'abcdefghijklmnopqrstuvwxy'[p]
            key_index = (key_index + 1) % KEYLEN
        else:
            cpos = 'ABCDEFGHIJKLMNOPQRSTUVWXY'.find(c)
            if cpos != -1:
                p = (cpos - key[key_index]) % 25
                pc = 'ABCDEFGHIJKLMNOPQRSTUVWXY'[p]
                key_index = (key_index + 1) % KEYLEN
            else:
                pc = c
        plain += pc
    return plain

def encrypt_level_1(ptext, key_index=0):
    key = [ord(c) - ord('a') for c in 'pyle']
    ctext = ''
    for c in ptext:
        pos = 'abcdefghijklmnopqrstuvwxy'.find(c)
        if pos != -1:
            p = (pos + key[key_index]) % 25
            cc = 'abcdefghijklmnopqrstuvwxy'[p]
            key_index = (key_index + 1) % KEYLEN
        else:
            pos = 'ABCDEFGHIJKLMNOPQRSTUVWXY'.find(c)
            if pos != -1:
                p = (pos + key[key_index]) % 25
                cc = 'ABCDEFGHIJKLMNOPQRSTUVWXY'[p]
                key_index = (key_index + 1) % KEYLEN
            else:
                cc = c
        ctext += cc
    return ctext

def test_decrypt_level_1():
    c_id = 'ID_4a6fc1ea9'
    ciphertext = test.loc[c_id]['ciphertext']
    print('Ciphertxt:', ciphertext)
    decrypted = decrypt_level_1(ciphertext)
    print('Decrypted:', decrypted)
    encrypted = encrypt_level_1(decrypted)
    print('Encrypted:', encrypted)
    print("Encrypted == Ciphertext:", encrypted == ciphertext)

test_decrypt_level_1()    


# Make a dictionary for fast lookup of plaintext

# In[ ]:


plain_dict = {}
for p_id, row in train.iterrows():
    text = row['text']
    plain_dict[text] = p_id
print(len(plain_dict))


# ## Update sub with level 1 decrypted matching texts

# In[ ]:


matched, unmatched = 0, 0
for c_id, row in test[test['difficulty']==1].iterrows():
    decrypted = decrypt_level_1(row['ciphertext'])
    found = False
    for pad in range(100):
        start = pad // 2
        end = len(decrypted) - (pad + 1) // 2
        plain_pie = decrypted[start:end]
        if plain_pie in plain_dict:
            p_id = plain_dict[plain_pie]
            row = train.loc[p_id]
            sub.loc[c_id] = train.loc[p_id]['index']
            matched += 1
            found = True
            break
    if not found:
        unmatched += 1
        print(decrypted)
            
print(f"Matched {matched}   Unmatched {unmatched}")
sub.to_csv('submit-level-1.csv')


# * If you find this useful - let me know by giving it a like ;-)

# 
