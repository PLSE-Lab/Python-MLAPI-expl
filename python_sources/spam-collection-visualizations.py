#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/spam.csv', encoding='latin-1')
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df = df.rename(columns={"v1":"Class", "v2":"Data"})

df.head()


# In[ ]:


df["Class"].value_counts()


# In[ ]:


df_visualizations = df.copy()
df_visualizations["Length"] = df_visualizations["Data"].apply(lambda x: len(x))
sns.violinplot(x="Class", y="Length", data=df_visualizations)


# In[ ]:


spamText = ""
for text in df['Data'][df['Class'] == 'spam']:
    spamText += text

spamcloud = WordCloud().generate(spamText)
plt.figure()
plt.imshow(spamcloud)
plt.axis("off")
plt.show()


# In[ ]:


hamText = ""
for text in df['Data'][df['Class'] == 'ham']:
    hamText += text

hamcloud = WordCloud().generate(hamText)
plt.figure()
plt.imshow(hamcloud)
plt.axis("off")
plt.show()


# In[ ]:


df_spam = pd.DataFrame()
df_spam['words'] = spamcloud.words_.keys()
df_spam['frequencies'] = spamcloud.words_.values()
plt.figure(figsize=(12, 6))
sns.barplot(x='words', y='frequencies', data=df_spam.sort_values(by=['frequencies'], ascending=[0]).head(20))


# In[ ]:


df_ham = pd.DataFrame()
df_ham['words'] = hamcloud.words_.keys()
df_ham['frequencies'] = hamcloud.words_.values()
plt.figure(figsize=(12, 6))
sns.barplot(x='words', y='frequencies', data=df_ham.sort_values(by=['frequencies'], ascending=[0]).head(20))


# In[ ]:


from numpy import sum
df_words = df_spam.append(df_ham)
plt.figure(figsize=(12, 6))
sns.barplot(x='words', y='frequencies', data=df_words.groupby('words', as_index=False).aggregate(sum).sort_values(by=['frequencies'], ascending=[0]).head(20))

