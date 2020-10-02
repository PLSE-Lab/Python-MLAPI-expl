#!/usr/bin/env python
# coding: utf-8

# [Cracking the code: difficulty 1](https://www.kaggle.com/group16/cracking-the-code-difficulty-1)
# 
# I used this as the reference.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv('../input/test.csv')


# Difficulty 1

# In[ ]:


first = test[test["difficulty"] == 1].reset_index()


# In[ ]:


char = Counter("".join(first.loc[list(range(first.shape[0]))]["ciphertext"]))
cipher_stats = pd.DataFrame([[x[0], x[1]]for x in char.items()], columns=["Letter", "Frequency"])

cipher_stats = cipher_stats.sort_values(by='Frequency', ascending=True)
cipher_stats.head()


# # Visualization

# All Train data

# In[ ]:


memo = Counter("".join(train.loc[range(train.shape[0])]["text"]))
memo_stats = pd.DataFrame([[x[0], x[1]] for x in memo.items()], columns=["Letter", "Frequency"])
memo_stats = memo_stats.sort_values(by='Frequency', ascending=True)

f, ax = plt.subplots(figsize=(5, 15))
plt.barh(np.array(range(len(memo_stats))) + 0.5, memo_stats['Frequency'].values)
plt.yticks(np.array(range(len(memo_stats))) + 0.5, memo_stats['Letter'].values)
# plt.savefig("count.png")
plt.show()


# All Test data

# In[ ]:


f, ax = plt.subplots(figsize=(5, 15))
plt.barh(np.array(range(len(cipher_stats))) + 0.5, cipher_stats['Frequency'].values)
plt.yticks(np.array(range(len(cipher_stats))) + 0.5, cipher_stats['Letter'].values)
# plt.savefig("count.png")
plt.show()


# 1 Train data

# In[ ]:


memo = Counter("".join(train.loc[0]["text"]))
memo_stats = pd.DataFrame([[x[0], x[1]] for x in memo.items()], columns=["Letter", "Frequency"])
memo_stats = memo_stats.sort_values(by='Frequency', ascending=True)

f, ax = plt.subplots(figsize=(5, 15))
plt.barh(np.array(range(len(memo_stats))) + 0.5, memo_stats['Frequency'].values)
plt.yticks(np.array(range(len(memo_stats))) + 0.5, memo_stats['Letter'].values)
# plt.savefig("count.png")
plt.show()


# 1 Test data

# In[ ]:


memo = Counter("".join(first.loc[0]["ciphertext"]))
memo_stats = pd.DataFrame([[x[0], x[1]] for x in memo.items()], columns=["Letter", "Frequency"])
memo_stats = memo_stats.sort_values(by='Frequency', ascending=True)
f, ax = plt.subplots(figsize=(5, 15))
plt.barh(np.array(range(len(memo_stats))) + 0.5, memo_stats['Frequency'].values)
plt.yticks(np.array(range(len(memo_stats))) + 0.5, memo_stats['Letter'].values)
# plt.savefig("count.png")
plt.show()


# It seems that training data uses fewer characters.
