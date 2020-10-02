#!/usr/bin/env python
# coding: utf-8

# # 1. Import libraries and data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

from transformers import AutoTokenizer, AutoModel, AutoConfig

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
# sample = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train[train['target']==0]


# # 2. Preprocessing

# In[ ]:


train['keyword'] = train['keyword'].fillna('')
train['location'] = train['location'].fillna('no_location')

test['keyword'] = test['keyword'].fillna('')
test['location'] = test['location'].fillna('no_location')


# In[ ]:


STOPWORDS = set(stopwords.words('english'))


# # 3. EDA

# In[ ]:


fig, axes = plt.subplots( figsize=(8, 4), dpi=100)
plt.tight_layout()
sns.countplot(x=train['target'], hue=train['target'], ax=axes)


# In[ ]:


words = " ".join(text for text in train['text'] if text not in STOPWORDS)

wc = wordcloud.WordCloud().generate(words)

plt.figure(figsize=[12,6])
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


train.keyword.unique()


# In[ ]:


train['keyword'] = train['keyword'].str.replace('buildings%20burning','burning%20buildings')
train['keyword'] = train['keyword'].str.replace('\%20',' ')

from nltk.stem.porter import *

stemmer = PorterStemmer()

train['keyword'] = train['keyword'].apply(lambda x: stemmer.stem(x))

train.keyword.unique()


# In[ ]:


keywords = " ".join(text for text in train['keyword'] if text not in STOPWORDS)

wc = wordcloud.WordCloud().generate(keywords)

plt.figure(figsize=[12,6])
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


train['text'] = train['keyword'] +" "+ train['text']
test['text'] = test['keyword'] +" "+ test['text']

Y = train['target']
X = train['text']
test_id = test['id']
X_test = test['text']

del train, test
del keywords, words
gc.collect()


# # 4 Hugging Face

# In[ ]:


# MODEL = 'roberta-base'


# In[ ]:


# tokenizer = AutoTokenizer.from_pretrained(MODEL)

# # Tokenize input
# text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
# tokenized_text = tokenizer.tokenize(text)
# tokenized_text


# In[ ]:


# config = AutoConfig.from_pretrained(MODEL)
# model = AutoModel.from_pretrained(MODEL, config=config)

# # model.eval()


# In[ ]:


from transformers import pipeline

nlp = pipeline("sentiment-analysis")


# In[ ]:


pred = []
for t in X_test:
    pred.append(nlp(t))
    


# In[ ]:


pred[0]


# In[ ]:


X_test.loc[0]


# # 5. Predict and send submission

# In[ ]:


predict = []
for p in pred:
    predict.append(1 if p[0]['label'] == 'NEGATIVE' else 0)


# In[ ]:


df = pd.DataFrame({'id':test_id, 'target':predict})
df.to_csv('nlp.csv', index=False)
df.head()

