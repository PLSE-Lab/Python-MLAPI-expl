#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Text Classification
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


import pandas as pd


# In[ ]:


train_data = pd.read_csv("../input/train_data.csv")
train_label = pd.read_csv("../input/train_label.csv")


# In[ ]:


train_data.head()


# In[ ]:


train_label.head()


# id is the feature which map the text in train_data to label in train_label

# In[ ]:


train_label.loc[train_label['id']==122885]


# In[ ]:


train_data.loc[train_data['id']==122885]


# Each id has different labels but not at the same time.Hence This is a multiclass problem.

# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data = pd.read_csv("../input/test_data.csv")


# In[ ]:


test_data.head()


# In[ ]:


train_label.isnull().sum()


# In[ ]:


test_data.info()


# In[ ]:


train_data.info()


# In[ ]:


sample_sub = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


sample_sub.head()


# In[ ]:


train_label['label'].unique()


# These are the 15 unique labels dataset has

# In[ ]:


training_data = pd.merge(train_data,train_label)


# In[ ]:


training_data.head()


# In[ ]:


#2. Noise Removal
#Lower case
training_data['text'] = training_data['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
training_data['text'].head()


# In[ ]:


#Remove punctuation
training_data['text'] = training_data['text'].str.replace('[^\w\s]','')
training_data['text'].head()


# In[ ]:


#commonly occuring words in our text
freq = pd.Series(' '.join(training_data['text']).split()).value_counts()[:10]
freq


# In[ ]:


#remove these words
freq = list(freq.index)
training_data['text'] = training_data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
training_data['text'].head()


# In[ ]:


#rare word 
freq = pd.Series(' '.join(training_data['text']).split()).value_counts()[-10:]
freq


# In[ ]:


freq = list(freq.index)
training_data['text'] = training_data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
training_data['text'].head()


# In[ ]:


#Data Visualization
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
training_data.groupby('label').text.count().plot.bar(ylim=0)
plt.show()


# There are different text correspond to unique 15 labels.

# In[ ]:


# Feature Engineering
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
count_vect = CountVectorizer()


# In[ ]:


X_train_counts = count_vect.fit_transform(training_data['text'])
tf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_transformed = tf_transformer.transform(X_train_counts)


# In[ ]:


X_test_counts = count_vect.transform(test_data['text'])
X_test_transformed = tf_transformer.transform(X_test_counts)


# In[ ]:


#Label Encoding
labels = LabelEncoder()
y_train_labels_fit = labels.fit(training_data['label'])
y_train_lables_trf = labels.transform(training_data['label'])

print(labels.classes_)


# In[ ]:


# Model Fitting
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

linear_svc = LinearSVC()
clf = linear_svc.fit(X_train_transformed,y_train_lables_trf)

calibrated_svc = CalibratedClassifierCV(base_estimator=linear_svc,
                                        cv="prefit")

calibrated_svc.fit(X_train_transformed,y_train_lables_trf)
predicted = calibrated_svc.predict(X_test_transformed)
to_predict = test_data['text']
p_count = count_vect.transform(to_predict)
p_tfidf = tf_transformer.transform(p_count)


# In[ ]:


# prediction
pd.DataFrame(calibrated_svc.predict_proba(p_tfidf), columns=labels.classes_)


# In[ ]:


pd.DataFrame(calibrated_svc.predict_proba(p_tfidf)*100, columns=labels.classes_)

