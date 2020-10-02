#!/usr/bin/env python
# coding: utf-8

# #### Hi everyone! This kernel aims at showing you a basic approach to our ciphering problem. It shows my way of exploring solutions and I hope it will help you through your discovery of our dataset.

# ## Imports and data loading

# In[ ]:


# load the basic librairies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter


# In[ ]:


# load the data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


train_df.head(3)


# In[ ]:


test_df.head(5)


# In[ ]:


# first feature: create a 'length' column
train_df['length'] = train_df.text.apply(len)
test_df['length'] = test_df.ciphertext.apply(len)


# filter the test dataframes by cypher level
df_level_1 = test_df[test_df.difficulty==1].copy()
df_level_2 = test_df[test_df.difficulty==2].copy()
df_level_3 = test_df[test_df.difficulty==3].copy()
df_level_4 = test_df[test_df.difficulty==4].copy()

df_level_1.head(3)


# ## Sort of Exploratory Data Analysis

# In[ ]:


print('train_df.shape:', train_df.shape, '\ntest_df.shape:', test_df.shape)


# ### To be familiar with the data, let's look at it first!

# ### Original text

# In[ ]:


for i in range(5):
    print(train_df.text[i], '\n')


# In[ ]:


plain_char_cntr = Counter(''.join(train_df['text'].values))
plain_stats_train = pd.DataFrame([[x[0], x[1]] for x in plain_char_cntr.items()], columns=['Letter', 'Frequency'])
plain_stats_train = plain_stats_train.sort_values(by='Frequency', ascending=False)

f, ax = plt.subplots(figsize=(15, 5))
plt.bar(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Frequency'].values)
plt.xticks(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Letter'].values)
plt.show()


# The most frequent letters in English are 'e', 't', 'a', 'o', 'n', 'i' and 's' so that's quite consistant!

# ### Ciphered text level 1

# In[ ]:


for i in range(20):
    try:
        print(df_level_1.ciphertext[i], '\n')
    except KeyError:
        pass


# In[ ]:


plain_char_cntr = Counter(''.join(df_level_1['ciphertext'].values))
plain_stats_train = pd.DataFrame([[x[0], x[1]] for x in plain_char_cntr.items()], columns=['Letter', 'Frequency'])
plain_stats_train = plain_stats_train.sort_values(by='Frequency', ascending=False)

f, ax = plt.subplots(figsize=(15, 5))
plt.bar(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Frequency'].values)
plt.xticks(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Letter'].values)
plt.show()


# Compared to the previous level, we see that there are far less differences in terms of caracters frequence that in plain English...

# ### Ciphered text level 2

# In[ ]:


for i in range(30):
    try:
        print(df_level_2.ciphertext[i], '\n')
    except KeyError:
        pass


# In[ ]:


plain_char_cntr = Counter(''.join(df_level_2['ciphertext'].values))
plain_stats_train = pd.DataFrame([[x[0], x[1]] for x in plain_char_cntr.items()], columns=['Letter', 'Frequency'])
plain_stats_train = plain_stats_train.sort_values(by='Frequency', ascending=False)

f, ax = plt.subplots(figsize=(15, 5))
plt.bar(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Frequency'].values)
plt.xticks(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Letter'].values)
plt.show()


# Seems like it's the same repartition as level 1!

# ### Ciphered text level 3

# In[ ]:


for i in range(8):
    try:
        print(df_level_3.ciphertext[i], '\n')
    except KeyError:
        pass


# In[ ]:


plain_char_cntr = Counter(''.join(df_level_3['ciphertext'].values))
plain_stats_train = pd.DataFrame([[x[0], x[1]] for x in plain_char_cntr.items()], columns=['Letter', 'Frequency'])
plain_stats_train = plain_stats_train.sort_values(by='Frequency', ascending=False)

f, ax = plt.subplots(figsize=(15, 5))
plt.bar(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Frequency'].values)
plt.xticks(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Letter'].values)
plt.show()


# Only digits, but nothing to add...

# ### Ciphered text level 4

# In[ ]:


for i in range(3):
    try:
        print(df_level_4.ciphertext[i], '\n')
    except KeyError:
        pass


# In[ ]:


plain_char_cntr = Counter(''.join(df_level_4['ciphertext'].values))
plain_stats_train = pd.DataFrame([[x[0], x[1]] for x in plain_char_cntr.items()], columns=['Letter', 'Frequency'])
plain_stats_train = plain_stats_train.sort_values(by='Frequency', ascending=False)

f, ax = plt.subplots(figsize=(15, 5))
plt.bar(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Frequency'].values)
plt.xticks(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Letter'].values)
plt.show()


# All caracters seem to be roughly as frequent (except the '='), but again not many things to say

# ## Try to decypher the first level

# ### Frequency Analysis

# The most natural thing to do now is to look try to find a match between caracters, ie an "a" becomes an "s", a "b" becomes a "k" etc.
# This can be done by frequency analysis, and that's what wa gonna do now.

# In[ ]:


plain_char_cntr = Counter(''.join(train_df['text'].values))
plain_stats_train = pd.DataFrame([[x[0], x[1]] for x in plain_char_cntr.items()], columns=['Letter', 'Frequency'])
plain_stats_train = plain_stats_train.sort_values(by='Frequency', ascending=False)


# In[ ]:


plain_char_test = Counter(''.join(df_level_1['ciphertext'].values))
plain_stats_test = pd.DataFrame([[x[0], x[1]] for x in plain_char_test.items()], columns=['Letter', 'Frequency'])
plain_stats_test = plain_stats_test.sort_values(by='Frequency', ascending=False)
plain_stats_test['Frequency'] -= 21130 # to remove the influence of random padding caracters


# In[ ]:


f, ax = plt.subplots(figsize=(15, 5))
plt.bar(np.array(range(len(plain_stats_test))) + 0.5, plain_stats_test['Frequency'].values)
plt.bar(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Frequency'].values//4, alpha=.5,color='green')
plt.xticks(np.array(range(len(plain_stats_test))) + 0.5, plain_stats_test['Letter'].values)
plt.show()


# We can see that there is **not** a match between letters, meaning that there is no 
# direct correspondance between caracters. We have to try something else. 

# ### One basic trick

# In[ ]:


# first trick: use the length of some text pieces to understand the first cypher level
df_level_1.length.sort_values(ascending=False).head()

# we can see that 1 piece of text is of len 500, meaning that its original length is between 
# 401 and 500 (recall that every original piece of text is padded with random caracters to
# the next hundred)


# In[ ]:


df_level_1.length.describe([.999])


# We can see that the crushing majority of samples are 100 caracters long, meaning that their length is between 1 and 100 and they're padded to the next hundred. But even if long samples (> 100 caracters) are rare, they will be very interesting!

# In[ ]:


# and here is the corresponding text
df_level_1.loc[45272].ciphertext


# In[ ]:


# then we look in the training data to find the passage with the corresponding length
matching_pieces = train_df[(train_df.length>=401) & (train_df.length<=500)]
matching_pieces
# only three unciphered texts length are in the interval: let's print them


# In[ ]:


matching_pieces.text.values


# #### If you look carefully to the spaces, dots, columns etc., you will soon understand that our ciphered text matches the first plain text: we found our first match. Now let's try to understand some properties of the first cipher level

# In[ ]:


print('Unciphered text:\n', train_df.loc[13862].text, '\n\nCiphered text (level 1):\n', 
      df_level_1.loc[45272].ciphertext)


# __It is now obvious that 1st level cipher preserves the punctuation and the case 
# (uppercase and lowercase match), which is already a good hint.
# What is also striking is that the length of the words is preserved.__
# 
# __The 1st difficulty arises tho: words are not ciphered the same way: 'and' is either 'edc' or 'lrs'
# so its is not a basic matching between caracters...__

# In[ ]:


# Let's do the same thing for a second piece of text now.
# With the same procedure, we get a second match:
print(train_df.loc[6938].text, '\n\n', df_level_1.loc[95019].ciphertext)


# So this is another example of two matching passages. I'll now let you find other tricks, and
# I'll post the follow-up soon with the main trick for cipher 1! 
# Don't forget to upvote if you find this kernel useful! ;)

# In[ ]:




