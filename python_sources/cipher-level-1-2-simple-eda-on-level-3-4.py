#!/usr/bin/env python
# coding: utf-8

# [Credit to https://www.kaggle.com/kaggleuser58/cipher-challenge-iii-level-1]
# 
# # Introduction
# 
# ### Time to share solution of cipher level 2 so you can look at the next level.
# In the previous Cipher Challenge II one of the levels was a cipher with multiple substitutions generated from a key of length 8 if I remember correct.
# 
# The level 1 of this Cipher Challenge III is the same kind but with a key of length 4, so only 4 substitutions are used for each character mapping.
# See https://www.kaggle.com/kaggleuser58/cipher-challenge-iii-level-1
# 
# The level 2 of this Cipher Challenge III is a transposition cipher on top of level 1.
# See https://www.kaggle.com/c/ciphertext-challenge-iii/discussion/103969#latest-598262
# 
# ## The cipher
# - The cipher only apllies to UPPERCASE and LOWERCASE letters.
# - The key only shifts every time an UPPERCASE or LOWERCASE letter is met.
# 
# ## Padding
# From Cipher Challenge II it was found that padding could be done both up front and in the end. Number of padding characters in the end was always equal to or at most 1 character more (if number of characters to pad with was odd) than the number of padding characters up front.
# 
# # Level 1 and level 2 - solution

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tqdm

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
for c_id, row in tqdm.tqdm(test[test['difficulty']==1].iterrows()):
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


# ## Update sub with level 2 decrypted matching texts

# In[ ]:


import math
from itertools import cycle

def rail_pattern(n):
    r = list(range(n))
    return cycle(r + r[-2:0:-1])

def encrypt_level_2(plaintext, rails=21):
    p = rail_pattern(rails)
    # this relies on key being called in order, guaranteed?
    return ''.join(sorted(plaintext, key=lambda i: next(p)))
def decrypt_level_2(ciphertext, rails=21):
    p = rail_pattern(rails)
    indexes = sorted(range(len(ciphertext)), key=lambda i: next(p))
    result = [''] * len(ciphertext)
    for i, c in zip(indexes, ciphertext):
        result[i] = c
    return ''.join(result)


# In[ ]:


matched, unmatched = 0, 0
for c_id, row in tqdm.tqdm(test[test['difficulty']==2].iterrows()):
    decrypted = decrypt_level_1(decrypt_level_2(row['ciphertext']))
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
sub.to_csv('submit-level-2.csv')


# # Level 3 and level 4 - exploration

# In[ ]:


level12_train_index = list(sub[sub["index"] > 0]["index"])
print(len(level12_train_index))
train34 = train[~train["index"].isin(level12_train_index)].copy()
test3 = test[test['difficulty']==3].copy()
test4 = test[test['difficulty']==4].copy()
print(train34.shape, test3.shape[0] + test4.shape[0])


# ## Level 3 - Let's see some cipher text

# In[ ]:


test3.sort_values("length", ascending=False).head(5)


# OK, look like it is too long. Probably each number is one character. Let's count the numbers.

# In[ ]:


test3["nb"] = test3["ciphertext"].apply(lambda x: len(x.split(" ")))
test3.sort_values("length", ascending=False).head(5)


# Look better, let's see the train set

# In[ ]:


train34.sort_values("length", ascending=False).head(5)


# Cool, we found an exact match for level 3 and 2 possible matches.

# In[ ]:


c_id = 'ID_f0989e1c5' # length = 700
index = 34509 # length = 671
sub.loc[c_id] = index # train.loc[p_id]['index']


# ## Level 4
# Even more, index 34509 could be a text in level 4 as it does not seems to match any cipher text in level 3

# In[ ]:


test4.sort_values("length", ascending=False).head(5)


# OOM, what are they

# In[ ]:


test4.head(1)["ciphertext"].values[0]


# Then it must be a base64 text, but base64 is just a encoding method and this text is just a way to hide the real content. Let's do a count as well. Note that we have to count the number of chars in the level 1.

# In[ ]:


import base64

def encode_base64(x):
    return base64.b64encode(x.encode('ascii')).decode()

def decode_base64(x):
    return base64.b64decode(x)

train34["nb"] = train34["length"].apply(lambda x: math.ceil(x/100)*100)
ratio = test3["length"].mean() / train34["nb"].mean()
print(ratio)

def get_length_level1(x):
    n = len(decode_base64(x))/ratio
    n = round(n / 100) * 100
    return n

train34.head(3)


# In[ ]:


test4["nb"] = test4["ciphertext"].apply(lambda x: get_length_level1(x)) 
test4.sort_values("nb", ascending=False).head(5)


# It makes sense now. We found a match for level 4

# In[ ]:


c_id = 'ID_0414884b0' # length = 900
index = 42677 # length = 842
sub.loc[c_id] = index # train.loc[p_id]['index']


# ##  Let's submit it the score should be a little bit higher

# In[ ]:


sub.head(3)


# # Level 3 - mapping for few pairs
# ## 3 easy pairs
# Assume that one number is associated to only a single char, let's find 2 pontential matches as listed above.

# In[ ]:


def is_correct_mapping(ct_l2, ct_l3):
    tmp = pd.DataFrame([(c,n) for c,n in zip(list(ct_l2), ct_l3.split(" ")) if c.isalpha()])
    tmp.drop_duplicates(inplace=True)
    tmp.columns = ["ch", "num"]
    tmp = tmp.groupby("num")["ch"].nunique()
    return tmp.shape[0] == tmp.sum()

def pad_str(s, special_char = '?'):
    nb = len(s)
    nb_round = math.ceil(nb / 100) * 100
    nb_left = (nb_round - nb) // 2
    nb_right = nb_round - nb - nb_left
    
    left_s = ''.join([special_char] * nb_left)
    right_s = ''.join([special_char] * nb_right)
    return left_s + s + right_s

def is_correct_mapping_low(pt, ct):
    all_ct_l2 = [encrypt_level_2(encrypt_level_1(pad_str(pt), key_index)) for key_index in range(4)]

    for i, ct_l2 in enumerate(all_ct_l2):
        if is_correct_mapping(ct_l2, ct):
            return i
    return -1

def find_mapping(ciphertext_id, ct, train_df):
    nb = len(ct.split(" "))
    nb_low = ((nb // 100) - 1) * 100
    
    rs = []
    selected_rows = train_df[(train_df["length"] > nb_low) & (train_df["length"] < nb)]
    for row_id, row in selected_rows.iterrows():
        pt = row["text"]
        key_index = is_correct_mapping_low(pt, ct)
        if key_index >= 0:
            t = row["index"], key_index
            rs.append(t)
    if len(rs) == 1:
        return rs[0]
    return -1, -1


# In[ ]:


for ciphertext_id, row in test3[test3["nb"] >= 200].iterrows():
    ct = row["ciphertext"]
    index, key_index = find_mapping(ciphertext_id, ct, train34)
    if index > 0:
        print(ciphertext_id, index, key_index, "(length: {})".format(row["nb"]))
        sub.loc[ciphertext_id] = index # train.loc[p_id]['index']


# In[ ]:


print(sub[sub["index"] > 0].shape[0], sub[sub["index"] > 0].shape[0]/sub.shape[0])
sub.to_csv('submit-level-2-plus.csv')
sub.head(3)


# # Further exploration
# 
# kaggleuser58: Have you tried finding the corresponding plaintext letters for each group and see if you can see something if you sort the groups in the correct order - it might help in finding the solution.
# 

# In[ ]:


dict_level3 = {}
for ciphertext_id, row in test3[test3["nb"] >= 200].iterrows():
    ct = row["ciphertext"]
    index, key_index = find_mapping(ciphertext_id, ct, train34)
    if index > 0:
        print(ciphertext_id, index, key_index, "(length: {})".format(row["nb"]))
        dict_level3[ciphertext_id] = (index, key_index) # train.loc[p_id]['index']


# In[ ]:


dict_level3["ID_11070f053"] = (40234, 1)
dict_level3["ID_c1694eb06"] = (43773, 3)

for ciphertext_id, (index, key_index) in dict_level3.items():
    sub.loc[ciphertext_id] = index
    
print(sub[sub["index"] > 0].shape[0], sub[sub["index"] > 0].shape[0]/sub.shape[0])
sub.to_csv('submit-level-2-plus2.csv')
sub.head(3)


# In[ ]:


df_mapping = []
special_chars = "?"

def get_mapping(ct_l2, ct):
    tmp = pd.DataFrame([(c,n) for c,n in zip(list(ct_l2), ct.split(" ")) if c not in special_chars])
    tmp.drop_duplicates(inplace=True)
    tmp.columns = ["ch", "num"]
    return tmp

for ciphertext_id, (index, key_index) in dict_level3.items():
    ct = test3.loc[ciphertext_id]["ciphertext"]
    pt = train34[train34["index"]==index]["text"].values[0]
    ct_l2 = encrypt_level_2(encrypt_level_1(pad_str(pt), key_index))
    print(len(ct.split(" ")), len(pt))
    tmp = get_mapping(ct_l2, ct)
    df_mapping.append(tmp)

df_mapping = pd.concat(df_mapping)
print(df_mapping.shape)
df_mapping.head(3)
df_mapping.reset_index(drop=True, inplace=True)
df_mapping.tail(3)


# In[ ]:


pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.max_colwidth', 5000)
pd.set_option('display.width', 5000)

df_ch_num = df_mapping[["ch", "num"]].drop_duplicates().groupby("ch")["num"].apply(list)
df_ch_num = df_ch_num.to_frame("num").reset_index()
df_ch_num["num"] = df_ch_num["num"].apply(lambda x: np.sort([int(n) for n in x]))
df_ch_num["num_alpha"] = df_ch_num["num"].apply(lambda x: np.sort([str(n) for n in x]))
df_ch_num["num_hex"] = df_ch_num["num"].apply(lambda x: np.sort([hex(n) for n in x]))
df_ch_num


# ## Frequency analysis on Level 2

# In[ ]:


from collections import Counter
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,10)


# In[ ]:


test2 = test[test["difficulty"] == 2].copy()
fullcipher2 = "".join((test2["ciphertext"].values))
dict_fullcipher2 = Counter(fullcipher2)
df_fullcipher2 = pd.DataFrame.from_dict(dict_fullcipher2, orient='index')
df_fullcipher2 = df_fullcipher2.reset_index()
df_fullcipher2.columns = ["ch", "nb"]
df_fullcipher2.sort_values("nb", ascending=False, inplace=True)
print(df_fullcipher2.shape)
df_fullcipher2.head()


# In[ ]:


print(df_fullcipher2["nb"].mean(), df_fullcipher2["nb"].median())
df_fullcipher2.plot(x="ch", y=["nb"], kind="bar");


# ## Frequency analysis on Level 3

# In[ ]:


fullcipher3 = " ".join((test3["ciphertext"].values))
dict_fullcipher3 = Counter(fullcipher3.split(" "))
df_fullcipher3 = pd.DataFrame.from_dict(dict_fullcipher3, orient='index')
df_fullcipher3 = df_fullcipher3.reset_index()
df_fullcipher3.columns = ["num", "nb"]
df_fullcipher3.sort_values("nb", ascending=False, inplace=True)
print(df_fullcipher3.shape)
df_fullcipher3.head()


# In[ ]:


df_fullcipher3[df_fullcipher3["nb"] > 1500].plot(x="num", y=["nb"], kind="bar");


# * If you find this useful - let me know by giving it a like ;-)
