#!/usr/bin/env python
# coding: utf-8

# Logistic Regression to Classify multi-labeled images 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import seaborn as sns
print(check_output(["ls", "../input"]).decode("utf8"))
#print(check_output(["ls", "../input/train-jpg"]).decode("utf8"))


# In[ ]:


sample = pd.read_csv('../input/sample_submission_v2.csv')
print(sample.shape)
sample.head()


# In[ ]:


df = pd.read_csv('../input/train_v2.csv')
df.head()
print(df.shape)


# So, we are given around 40.000 training images.

# In[ ]:


df.shape


# # Tag counts
# 

# In[ ]:


all_tags = [item for sublist in list(df['tags'].apply(lambda row: row.split(" ")).values) for item in sublist]
print('total of {} non-unique tags in all training images'.format(len(all_tags)))
print('average number of labels per image {}'.format(1.0*len(all_tags)/df.shape[0]))


# In[ ]:


tags_counted_and_sorted = pd.DataFrame({'tag': all_tags}).groupby('tag').size().reset_index().sort_values(0, ascending=False)
tags_counted_and_sorted.head()


# In[ ]:


tags_counted_and_sorted.plot.barh(x='tag', y=0, figsize=(12,8))


# # Looking at the actual images

# In[ ]:


from glob import glob
image_paths = sorted(glob('../input/train-jpg/*.jpg'))[0:1000]
image_names = list(map(lambda row: row.split("/")[-1][:-4], image_paths))
image_names[0:20]


# In[ ]:


plt.figure(figsize=(12,8))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(plt.imread(image_paths[i]))
    plt.title(str(df[df.image_name == image_names[i]].tags.values))


# It seems, that all of the images are of the same size, which would make preprocessing them much easier.

# In[ ]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics import fbeta_score, precision_score, make_scorer, average_precision_score
import cv2
import warnings
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer


n_samples = 5000
rescaled_dim = 20

df['split_tags'] = df['tags'].map(lambda row: row.split(" "))
lb = MultiLabelBinarizer()
y = lb.fit_transform(df['split_tags'])
y = y[:n_samples]
X = np.squeeze(np.array([cv2.resize(plt.imread('../input/train-jpg/{}.jpg'.format(name)), (rescaled_dim, rescaled_dim), cv2.INTER_LINEAR).reshape(1, -1) for name in df.head(n_samples)['image_name'].values]))
X = MinMaxScaler().fit_transform(X)

print(X.shape, y.shape, lb.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# In[ ]:


classifier = OneVsRestClassifier(LogisticRegression(C=1, penalty='l2'))

classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)
all_labels = lb.inverse_transform(predicted)

all_test_tags = []
for index in range(predicted.shape[0]):
    all_test_tags.append(' '.join(list(lb.classes_[np.where(predicted[index, :] == 1)[0]])))
    


# In[ ]:


all_test_tags[1:20]


# In[ ]:


X_sub = np.squeeze(np.array([cv2.resize(plt.imread('../input/test-jpg-v2/{}.jpg'.format(name)), (rescaled_dim, rescaled_dim), cv2.INTER_LINEAR).reshape(1, -1) for name in sample['image_name'].values]))
X_sub = MinMaxScaler().fit_transform(X_sub)
X_sub.shape


# In[ ]:


y_sub = classifier.predict(X_sub)
all_test_tags = []
for index in range(y_sub.shape[0]):
    all_test_tags.append(' '.join(list(lb.classes_[np.where(y_sub[index, :] == 1)[0]])))


    
all_test_tags[1:20]


# In[ ]:


sample['tags'] = all_test_tags
sample.head()


# In[ ]:


image_paths = sorted(glob('../input/test-jpg-v2/*.jpg'))[0:1000]
image_names = list(map(lambda row: row.split("/")[-1][:-4], image_paths))
image_names[0:10]


# In[ ]:


plt.figure(figsize=(12,8))
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.imshow(plt.imread(image_paths[i]))
    plt.title(str(sample[sample.image_name == image_names[i]].tags.values))


# In[ ]:




