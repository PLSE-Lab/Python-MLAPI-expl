#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/train.csv')


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df['target'].value_counts() #normalize=True)


# In[ ]:


toxic = df.loc[df['target'] == 1, 'question_text']


# In[ ]:


toxic.head()


# In[ ]:


words = df['question_text'].str.split()


# In[ ]:


words.head()


# In[ ]:


num_words = words.apply(len)


# In[ ]:


num_words.head()


# In[ ]:


sns.distplot(num_words[df['target'] == 0], label='normal')
sns.distplot(num_words[df['target'] == 1], label='toxic')
plt.xlabel('num. words')
plt.legend()
plt.show()


# In[ ]:


pd.crosstab(num_words, df['target'])


# In[ ]:


first_word = list(map(lambda l: l[0], words))


# In[ ]:


first_word[:5]


# In[ ]:


pd.Series(first_word).value_counts().head()


# In[ ]:


pd.Series(first_word).groupby(df['target']).value_counts()


# In[ ]:


first_character = list(map(lambda w: w[0], first_word))


# In[ ]:


pd.Series(first_character).groupby(df['target']).value_counts()


# In[ ]:


print(classification_report(df['target'], num_words == 1))


# In[ ]:


print(classification_report(df['target'], num_words >= 30))


# In[ ]:


test_df = pd.read_csv('../input/test.csv')


# In[ ]:


test_df.head()


# In[ ]:


words_test = test_df['question_text'].str.split()


# In[ ]:


num_words_test = words_test.apply(len)


# In[ ]:


out = pd.DataFrame({'qid': test_df['qid'], 'prediction': (num_words_test >= 30).apply(int)})


# In[ ]:


predicting_toxic = test_df.loc[out['prediction'] == 1, 'question_text']


# In[ ]:


predicting_toxic.head()


# In[ ]:


out.to_csv('submission.csv', index=False)


# In[ ]:




