#!/usr/bin/env python
# coding: utf-8

# # **Ciphertext Challenge III**

# Thanks to Paul Dnt for his Public Kernel with ideas of how to start!!! https://www.kaggle.com/pednt9/something-to-begin-with-a-first-hint/comments

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


# In[ ]:


# Importing packages
# For plots
from matplotlib import pyplot as plt
# To count letters and characters
from collections import Counter 


# In[ ]:


test_path = '../input/test.csv'
train_path = '../input/train.csv'
sampleSubmission_path = '../input/sample_submission.csv'


# # **Read Data**

# ## **File descriptions**
# 1. **training.csv** - the training data - contains plaintext_ids and text samples for each Shakespearean line in the training set, along with a unique index.
# 
# 2. ** test.csv** - the test data - ciphertext_ids, ciphertext, and a difficulty level indicating which ciphers were used.
# 
# 3. **sample_submission.csv** - a sample submission file in the correct format.

# ## **Training data fields**
# 1. **plaintext_id** - A unique ID for each line in the set.
# 2. **text** - The unpadded, unencrypted text of the review.
# 3. **index** - A unique ID that should be assigned to each ciphertext_id in test.csv when you've decrypted (or think you've correctly decrypted!) its ciphertext.

# ## **Test data fields**
# 1. **ciphertext_id** - A unique ID for each line's ciphertext in the set.
# 2. **ciphertext** - A segment of encrypted text.
# 3. **difficulty** - Each document has been encrypted with between 1 and 4 ciphers. The difficulty level of a segment of ciphertext denotes which ciphers were used, and in what order.

# In[ ]:


# Read and shows the head of the trainning data set
train_data = pd.read_csv(train_path)#, index_col = 0)
train_data.head()


# In[ ]:


# Read and shows the head of the test data set
test_data = pd.read_csv(test_path)#, index_col = 0)
test_data.head()


# # **Visualization**

# ### Some Statistics from the train data set

# In[ ]:


# Some Statistics from the train data set
train_data.describe()


# ### Some Statistics from the test data set

# In[ ]:


test_data.describe()


# In[ ]:


# Columns name
print(train_data.columns)
print(test_data.columns)


# ### Test Data Set

# In[ ]:


# first feature: create a 'length' column
train_data['length'] = train_data.text.apply(len)
train_data.head()


# In[ ]:


test_data['length'] = test_data.ciphertext.apply(len)
test_data.head()


# In[ ]:


# filter the test dataframes by cypher level
df_level_1 = test_data[test_data.difficulty==1].copy()
df_level_2 = test_data[test_data.difficulty==2].copy()
df_level_3 = test_data[test_data.difficulty==3].copy()
df_level_4 = test_data[test_data.difficulty==4].copy()

df_level_1.head(3)
#df_level_2.head(3)
#df_level_3.head(3)
#df_level_4.head(3)


# In[ ]:


# Filter the train data set for difficult level
Level1_loc = df_level_1.loc[:].index.values
for i in range(len(Level1_loc)):
    train_data[train_data['index'] == Level1_loc[i]].text.values


# In[ ]:


for i in range(len(Level1_loc)):
    train_level1 += train_data[train_data['index'] == Level1_loc[i]].text.values
    



#data_loc = train_data[train_data['index'] == Level1_loc[2]].index.values


# In[ ]:


train_data[train_data['length']<=100]['length'].hist(bins=99)


# ## Count letters in the string

# In[ ]:


# using collections.Counter() to get count of each element in string  
plain_char_cntr = Counter(''.join(train_data['text'].values))
plain_stats_train = pd.DataFrame([[x[0], x[1]] for x in plain_char_cntr.items()], columns=['Letter', 'Frequency'])
plain_stats_train = plain_stats_train.sort_values(by='Frequency', ascending=False)
print(plain_stats_train.head())


# In[ ]:


# using collections.Counter() to get count of each element in string  
plain_char_test = Counter(''.join(df_level_1['ciphertext'].values))
plain_stats_test = pd.DataFrame([[x[0], x[1]] for x in plain_char_test.items()], columns=['Letter', 'Frequency'])
plain_stats_test = plain_stats_test.sort_values(by='Frequency', ascending=False)
print(plain_stats_test.head())


# In[ ]:


# Plot the frequencies
f, ax = plt.subplots(figsize=(15, 5))
plt.bar(np.array(range(len(plain_stats_test))) + 0.5, plain_stats_test['Frequency'].values)
plt.bar(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Frequency'].values//4, alpha=.5,color='green')
plt.xticks(np.array(range(len(plain_stats_test))) + 0.5, plain_stats_test['Letter'].values)
plt.show()


# * ## Understanding level 1

# In[ ]:


# Level of dificulty
A1 = df_level_1.length
print('Level 1 \n')
print(A1.describe())


# In[ ]:


# then we look in the training data to find the passage with the corresponding length
matching_pieces = train_data[(train_data.length>=401) & (train_data.length<=500)]
matching_pieces
# only three unciphered texts length are in the interval: let's print them


# In[ ]:


matching_pieces.text.values


# In[ ]:


print('Unciphered text:\n', train_data.loc[13862].text, '\n\nCiphered text (level 1):\n', 
      df_level_1.loc[45272].ciphertext)


# In[ ]:


# Function to decrypt the text
def decrypt_text(cipher_text):
    l = 'abcdefghijklmnopqrstuvwxy'
    u = 'ABCDEFGHIJKLMNOPQRSTUVWXY'
    
    key =  [15, 24, 11, 4]
    key_index = 0
    plain = ''

    for character in cipher_text:
        test = l.find(character)
        if test != -1:
            p = (test - key[key_index]) % 25
            pc = l[p]
            key_index = (key_index + 1) % len(key)
        else:
            test2 = u.find(character)
            if test2 != -1:
                p = (test - key[key_index]) % 25
                pc = u[p]
                key_index = (key_index + 1) % len(key)
            else:
                pc = character
        
        plain += pc
        
    return plain


# In[ ]:


# Function to encrypt the text
def encrypt_text(plain_text):
    l = 'abcdefghijklmnopqrstuvwxy'
    u = 'ABCDEFGHIJKLMNOPQRSTUVWXY'
    
    key =  [15, 24, 11, 4]
    key_index = 0
    encrypted = ''

    for character in plain_text:
        test = l.find(character)
        if test != -1:
            p = (test + key[key_index]) % 25
            pc = l[p]
            key_index = (key_index + 1) % len(key)
        else:
            test2 = u.find(character)
            if test2 != -1:
                p = (test + key[key_index]) % 25
                pc = u[p]
                key_index = (key_index + 1) % len(key)
            else:
                pc = character
        
        encrypted += pc
        
    return encrypted


# In[ ]:


plain_text = train_data.loc[13862].text
cipher_text = df_level_1.loc[45272].ciphertext

print('Plain text = \n', plain_text, '\n\n')
print('Decrypted text = \n', decrypt_text(cipher_text), '\n\n')
print('Encrypted text = \n', encrypt_text(plain_text), '\n\n')


# In[ ]:


df_level_1.loc[Level1_loc[2]].ciphertext


# In[ ]:


Level1_loc = df_level_1.loc[:].index.values
train_data[train_data['index'] == Level1_loc[2]]

data_loc = train_data[train_data['index'] == Level1_loc[2]].index.values
plain_text = train_data.loc[data_loc].text.values
cipher_text = df_level_1.loc[Level1_loc[2]].ciphertext


print('Plain text = \n', plain_text, '\n\n')
print('Decrypted text = \n', decrypt_text(cipher_text), '\n\n')


# ## Decrypt Level 1

# From: https://www.kaggle.com/kaggleuser58/cipher-challenge-iii-level-1

# In[ ]:


KEYLEN = 4 # len('pyle')
def decrypt_level_1(ctext):
    
    key = [ord(c) - ord('a') for c in 'pyle']
    print('Key = ', key)
    
    key_index = 0
    plain = ''
    for c in ctext:
        cpos = 'abcdefghijklmnopqrstuvwxy'.find(c)
        print('c = ', c)
        print('cpos = ', cpos)
        if cpos != -1:
            p = (cpos - key[key_index]) % 25
            print('p = ', p)
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

def test_decrypt_level_1(c_id):
    
    ciphertext = test_data[test_data['ciphertext_id'] == c_id].ciphertext.values
    print('Ciphertxt:', ciphertext)
    decrypted = decrypt_level_1(ciphertext)
    print('Decrypted:', decrypted)
    encrypted = encrypt_level_1(decrypted)
    print('Encrypted:', encrypted)
    print("Encrypted == Ciphertext:", encrypted == ciphertext)


c_id = 'ID_4a6fc1ea9'
test_decrypt_level_1(c_id) 


# In[ ]:


test_data[test_data['ciphertext_id'] =='ID_4a6fc1ea9']


# In[ ]:





#  ## Understanding level 2

# In[ ]:


A2 = df_level_2.length
print('\n Level 2 \n')
print(A2.describe())

