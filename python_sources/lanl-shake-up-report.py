#!/usr/bin/env python
# coding: utf-8

# ****This is just some fun stuff ****
# 
# https://www.kaggle.com/c/LANL-Earthquake-Prediction

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import matplotlib.pyplot as plt
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/public-private/public_and_private.csv')
df.Delta_Abslute = df.Delta_Abslute.fillna(0)


# **** Distribution of Number of Places Shake Up ****
# 
# Max shake-up -  4024 places up (Lucky dude !!)
# 
# Max shake-dowm -  3673 places down :( 

# In[ ]:


plt.figure(figsize=(12,6))
plt.xlabel("Places Shake-up")
plt.ylabel("Frequency")

df.Delta.hist(bins = 20, edgecolor = 'black', color = 'green')
print("Mean shake-up       " ,df.Delta.mean())
print("\nMedian shake-up     " ,df.Delta.median())
print("\nMax shake-up        " ,df.Delta.max())
print("\nMin shake-down ;)   " ,df.Delta.min())
print("\nStd shake-up        " ,df.Delta.std())


# **** Distribution of Absolute Number of Places Shake Up ****
# 
# Mean shake-up - 870 !!!! 
# 
# Avegare Guess - 571 (14 sample)      :P  https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/93949
# 
# closest guess - 666 Kain (@kainsama)
# 
# Number of No shake-up only 11 out of 4000 plus people !!!!!

# In[ ]:


plt.figure(figsize=(12,6))
plt.xlabel("Absolute Shake-up")
plt.ylabel("Frequency")

df.Delta_Abslute.hist(bins = 20, edgecolor = 'black', color = 'green')
print("Mean Absolute shake-up        " ,df.Delta_Abslute.mean())
print("\nMedian Absolute shake-up      " ,df.Delta_Abslute.median())
print("\nMax Absolute shake-up         " ,df.Delta_Abslute.max())
print("\nMin Absolute shake-up         " ,df.Delta_Abslute.min())
print("\nStd Absolute shake-up         " ,df.Delta_Abslute.std())
print("\nNumber of No shake-up         ",(df.Delta_Abslute == 0).sum())


# ****Distribution of Public Score****

# In[ ]:


plt.figure(figsize=(12,6))
plt.xlabel("Best Public Score")
plt.ylabel("Frequency")
df.Score_public[df.Score_public < 6].hist(bins = 20, edgecolor = 'black', color = 'green')


# ****Distribution of Private Score****

# In[ ]:


plt.figure(figsize=(12,6))
plt.xlabel("Best Private Score")
plt.ylabel("Frequency")
df.Score_private[df.Score_private < 6].hist(bins = 20, edgecolor = 'black', color = 'green')


# ****Distribution of Difference of Public and Private Score****

# In[ ]:


S = df.Score_public - df.Score_private 
plt.figure(figsize=(12,6))
plt.xlabel("Difference : Public Score - Private Score")
plt.ylabel("Frequency")
S[(S<2) & (S>-4)].hist(bins = 20, edgecolor = 'black', color = 'green')


# ****Public Score vs Private Score****

# In[ ]:


plt.figure(figsize=(10,7))
plt.xlim(2.2,3)
plt.ylim(1,2.5)
plt.xlabel("Private Score")
plt.ylabel("Public Score")

plt.scatter(df.Score_private, df.Score_public, color = 'red')


# ****Number of Entries vs Public Score****

# In[ ]:


plt.figure(figsize=(10,7))
#plt.xlim(2.2,3)
plt.ylim(1,5)
plt.xlabel("Number of Entries")
plt.ylabel("Public Score")
plt.scatter(df.Entries, df.Score_public, color = 'red')


# ****Number of Entries vs Private Score****

# In[ ]:


plt.figure(figsize=(10,7))
#plt.xlim(2.2,3)
plt.ylim(2.2,5)
plt.xlabel("Number of Entries")
plt.ylabel("Private Score")
plt.scatter(df.Entries, df.Score_private, color = 'red')


# ****Number of Entries vs Shake-up****

# In[ ]:


plt.figure(figsize=(10,7))
#plt.xlim(2.2,3)
#plt.ylim(2.2,5)
plt.xlabel("Number of Entries")
plt.ylabel("Places Shake-up")
plt.scatter(df.Entries, df.Delta, color = 'red')


# ****Number of Entries vs Public Rank****

# In[ ]:


plt.figure(figsize=(10,7))
#plt.xlim(2.2,3)
#plt.ylim(2.2,5)
plt.xlabel("Number of Entries")
plt.ylabel("Public Rank")
plt.scatter(df.Entries, df.Rank_public, color = 'red')


# ****Number of Entries vs Private Rank****

# In[ ]:


plt.figure(figsize=(10,7))
#plt.xlim(2.2,3)
#plt.ylim(2.2,5)
plt.xlabel("Number of Entries")
plt.ylabel("Private Rank")
plt.scatter(df.Entries, df.Rank_private, color = 'red')


# ****;)****
