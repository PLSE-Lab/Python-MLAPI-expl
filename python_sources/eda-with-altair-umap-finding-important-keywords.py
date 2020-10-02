#!/usr/bin/env python
# coding: utf-8

# In this kernel, we would like to understand more about the content of the tweets. Which words can explain the classes better?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


plt.rcParams['figure.figsize'] = (12,7)


# In[ ]:


df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df.head()


# We can see that the classes are slightly imbalanced which might impact our classifier performance later on.

# In[ ]:


sns.countplot(df['target'])
sns.despine()


# In the following figure, you can see the top 20 keywords that are used in the data.

# In[ ]:


df['keyword'].value_counts().head(20).plot.barh()
sns.despine()


# The majority of the tweets are located in the US, e.g. `USA`, `New York`, `United States`, `Los Angeles, CA`.

# In[ ]:


df['location'].value_counts().head(20).plot.barh()
sns.despine()


# In terms of the word length, we can see that the "real" tweets are slightly longer.

# In[ ]:


fig, ax = plt.subplots(figsize=(12,7))
for label, group in df.groupby('target'):
    sns.distplot(group['text'].str.len(), label=str(label), ax=ax)
plt.xlabel('# of characters')
plt.ylabel('density')
plt.legend()
sns.despine()


# # Feature Selection
# 
# One of the easiest ways to extract features from text data is to use bag-of-words or TF-IDF. However, this might result in high-dimensional data. One of the causes is from tokens that only appear once in the whole dataset. On the other hand, there might be stop words that will appear frequently but carry little to no meaning, such as prepositions. Thus, we will remove the stop words first and then pick the minimum number of documents that a particular token should appear in to reduce the dimensionality.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from umap import UMAP

features = []
for i in range(1, 11):
    X_dim = CountVectorizer(min_df=i, stop_words='english').fit_transform(df['text'])
    features.append(X_dim.shape[1])
plt.plot(range(1, 11), features)
plt.xlabel('min df')
plt.ylabel('# of features')
sns.despine()


# The previous figure shows that we can significantly reduce the dimensionality by using the minimum document frequency of 2. Hopefully, this will help us understand the data better.

# # Dimensionality Reduction
# 
# Though we have reduced the number of dimensions from around 20k to 6k (~70% reduction), it is still hard to understand the clusters in the data without visualising them in 2D. *Thus, [UMAP](https://pair-code.github.io/understanding-umap/) to the rescue!*
# 
# In the following code, we try to project the data into 2D from the raw tweets. To make it easier to evaluate the "clusters", we can visualise them using an interactive library called [Altair](https://altair-viz.github.io/). This declarative visualisation library can help us to put tooltips and interact with each data point to understand our data.

# In[ ]:


from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from umap import UMAP

dim_red = make_pipeline(
    CountVectorizer(min_df=2, stop_words='english'),
    UMAP()
)
X_dim = dim_red.fit_transform(df['text'])


# In[ ]:


get_ipython().run_cell_magic('capture', '', "!pip install altair notebook vega # needs internet in settings (right panel)\nimport altair as alt\nalt.renderers.enable('kaggle')")


# In[ ]:


alt.Chart(pd.DataFrame({
    'x0': X_dim[:,0],
    'x1': X_dim[:,1],
    'text': df['text'],
    'keyword': df['keyword'],
    'location': df['location'],
    'target': df['target']
}).sample(5000, random_state=42)).mark_point().encode(
    x='x0',
    y='x1',
    color='target:N',
    tooltip='keyword'
).properties(
    title='Based on text',
    width=500,
    height=500
).interactive()


# Notice that while the clusters still contain 0s and 1s based on the text alone. The keywords are also jumbled. *What if we do it the other way around?*

# In[ ]:


from sklearn.preprocessing import OneHotEncoder

enc = make_pipeline(
    OneHotEncoder(),
    UMAP(metric='cosine', random_state=42)
)
X_onehot = enc.fit_transform(df[['keyword']].fillna(''))


# In[ ]:


alt.Chart(pd.DataFrame({
    'x0': X_onehot[:,0],
    'x1': X_onehot[:,1],
    'text': df['text'],
    'keyword': df['keyword'],
    'location': df['location'],
    'target': df['target']
}).sample(5000, random_state=42)).mark_point().encode(
    x='x0',
    y='x1',
    color='target:N',
    tooltip='text'
).properties(
    title='Based on keywords',
    width=500,
    height=500
).interactive()


# It turns out, you can see the separation better this way! Those tiny islands are now in the same color, i.e. same labels. Will this be a good predictor then?

# # Top distinctive keywords
# 
# If we can separate the data with keywords, which of them are actually distinctive?

# In[ ]:


keywords = df.groupby('keyword').agg({
    'target': 'mean'
})


# The following figure shows that keywords like `derailment`, `wreckage`, `debris`, `outbreak`, or `typhoon` are more likely to describe a real event. As we can see later on, though `bombed` is used in daily conversations figuratively, people use `suicide bombing` or `suicide bomb` sparingly.

# In[ ]:


keywords['target'].sort_values(ascending=False).head(10).plot.barh()
plt.xlabel('p(target=1)')
sns.despine()


# On the other hand, the following figure shows the keywords that people use more frequently in daily conversations, not to describe catastrophic events.

# In[ ]:


keywords['target'].sort_values().head(10).plot.barh()
plt.xlabel('p(target=1)')
sns.despine()


# In[ ]:


for index, row in df[df['keyword'] == 'body%20bags'].sample(10).iterrows():
    print('Label: {} | {}'.format(row.target, row.text))


# # Indistinctive keywords
# 
# The main problems are in these indistinctive keywords. They are more likely what form the big island in the middle of the UMAP projection based on keywords in the last scatter plot.

# In[ ]:


keywords.query('target > .45 and target < .55').sort_values('target').plot.barh()
plt.xlabel('p(target=1)')
sns.despine()


# Let's see some examples containing the word `hail` and `bombed`.

# In[ ]:


for index, row in df[df['keyword'] == 'hail'].sample(10).iterrows():
    print('Label: {} | {}'.format(row.target, row.text))


# In[ ]:


for index, row in df[df['keyword'] == 'bombed'].sample(10).iterrows():
    print('Label: {} | {}'.format(row.target, row.text))

