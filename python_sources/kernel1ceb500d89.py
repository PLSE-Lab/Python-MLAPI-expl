#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


with open('../input/train.json') as train_data:
    data = json.load(train_data)


# In[ ]:


features = [x['ingredients'] for x in data]


# In[ ]:


features_list = []
for feature in features:
    single_item = ''
    for item in feature:
        item= item.replace(' ','-')
        single_item = single_item + ' ' + item
    single_item = single_item[1:]
    features_list.append(single_item)


# In[ ]:


vectorizor = CountVectorizer()
X_features = vectorizor.fit_transform(features_list)


# In[ ]:


le = LabelEncoder()
labels = [x['cuisine'] for x in data]
y = le.fit_transform(labels)


# In[ ]:


with open('../input/test.json') as test_data:
    test_data = json.load(test_data)


# In[ ]:


features_test = [x['ingredients'] for x in test_data]


# In[ ]:


features_list_test = []
for feature in features_test:
    single_item = ''
    for item in feature:
#         item= item.replace(' ','-')
        single_item = single_item + ' ' + item
    single_item = single_item[1:]
    features_list_test.append(single_item)


# In[ ]:


test_features = vectorizor.transform(features_list_test)


# In[ ]:


model= LogisticRegression()
model.fit(X_features,y)


# In[ ]:


model.score(X_features, y)


# In[ ]:


df = pd.read_json('../input/test.json')


# In[ ]:


df['cuisine'] = le.inverse_transform(model.predict(test_features))


# In[ ]:


df= df[['id', 'cuisine']]
df.to_csv('submit.csv', index=False)


# In[ ]:




