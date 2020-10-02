#!/usr/bin/env python
# coding: utf-8

# <h2><center>Quick, Draw! Doodle Recognition Challenge - Simple EDA</center></h2>
# 
# #### Here i'll do some simple analysis on the competition data set, if you have any suggestion or feedback please let me know in the comments.

# ### Dependencies

# In[ ]:


import os
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

pd.options.display.max_rows = 20
sns.set(style="darkgrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load data

# In[ ]:


train_sample = pd.DataFrame()
files_directory = os.listdir("../input/train_simplified")
for file in files_directory:
    train_sample = train_sample.append(pd.read_csv('../input/train_simplified/' + file, index_col='key_id', nrows=10))
# Shuffle data
train_sample = shuffle(train_sample, random_state=123)

train = pd.DataFrame()
for file in files_directory[:185]:
    train = train.append(pd.read_csv('../input/train_simplified/' + file, index_col='key_id', usecols=[1, 2, 3, 5]))
# Shuffle data
train = shuffle(train, random_state=123)


# ### Let's take a look at the data

# In[ ]:


print('Train number of rows: ', train.shape[0])
print('Train number of columns: ', train_sample.shape[1])
print('Train set features: %s' % train_sample.columns.values)
print('Train number of label categories: %s' % len(files_directory))


# In[ ]:


train_sample.head()


# Our dataset seems to be pretty simple, ids, timestamps, countrycodes what seems to really matters is the drawings values (arrays that represents the drawings), "recognized" that means if this drawing was recognized or not by the algorithm and "word" that is our labels, maybe "countrycode" and the timestamp may be cool features to play with as well.

# ### In case you wanna check the label count
#  * I wold like to make a count plot of the label categories, but as we have 340 it would be a mess.

# In[ ]:


count_gp = train.groupby(['word']).size().reset_index(name='count').sort_values('count', ascending=False)
top_10 = count_gp[:10]
bottom_10 = count_gp[count_gp.shape[0]-10:count_gp.shape[0]]


# ### The top 10 words

# In[ ]:


ax_t10 = sns.barplot(x="word", y="count", data=top_10, palette="coolwarm")
ax_t10.set_xticklabels(ax_t10.get_xticklabels(), rotation=40, ha="right")
plt.show()


# ### The bottom 10 words

# In[ ]:


ax_b10 = sns.barplot(x="word", y="count", data=bottom_10, palette="BrBG")
ax_b10.set_xticklabels(ax_b10.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


count_gp


# ### Rate of recognized true/false rows

# In[ ]:


sns.countplot(x="recognized", data=train)
plt.show()


# In[ ]:


rec_gp = train.groupby(['word', 'recognized']).size().reset_index(name='count')
rec_true = rec_gp[(rec_gp['recognized'] == True)].rename(index=str, columns={"recognized": "recognized_true", "count": "count_true"})
rec_false = rec_gp[(rec_gp['recognized'] == False)].rename(index=str, columns={"recognized": "recognized_false", "count": "count_false"})
rec_gp = rec_true.set_index('word').join(rec_false.set_index('word'), on='word')
rec_gp


# ### View data as  drawings

# In[ ]:


words = train['word'].tolist()
drawings = [ast.literal_eval(pts) for pts in train[:9]['drawing'].values]

plt.figure(figsize=(10, 10))
for i, drawing in enumerate(drawings):
    plt.subplot(330 + (i+1))
    for x,y in drawing:
        plt.plot(x, y, marker='.')
        plt.tight_layout()
        plt.title(words[i]);
        plt.axis('off')


# As we can see some of these may be really hard to get good results with a model, other look a lot easier, what i think may be the challenge here is to get good results with this amount of label categories, some of them can get drawing that look a lot like others, but as we have lots of data, maybe something like deep learning may get good results.
