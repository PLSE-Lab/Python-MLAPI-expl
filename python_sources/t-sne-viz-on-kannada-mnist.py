#!/usr/bin/env python
# coding: utf-8

# - T-SNE code from this kernel https://www.kaggle.com/carlolepelaars/97-on-mnist-with-a-single-decision-tree-t-sne 
# 
# - If you want to get better score on leaderboard, I would recommend you to study https://www.kaggle.com/c/digit-recognizer this competition first! 
# - Same model can get better score on kannada-mnist dataset than digit-recognizer dataset

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.dpi'] = 300
from IPython.display import Image 
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals.six import StringIO  
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


import itertools
import seaborn as sns
import os
print(os.listdir('../input/Kannada-MNIST'))


# In[ ]:


train_df = pd.read_csv('../input/Kannada-MNIST/train.csv')
test_df = pd.read_csv('../input/Kannada-MNIST/test.csv')
dig_df = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')


# In[ ]:


print(train_df.shape)
train_df.head()


# In[ ]:


test_df = test_df.drop(['id'], axis=1)


# In[ ]:


sns.countplot(train_df['label'])
plt.show()


# In[ ]:



x_train = (train_df.iloc[:,1:].values).astype('float32')
y_train = train_df.iloc[:,0].values.astype('int32')
x_test = test_df.values.astype('float32')


# In[ ]:


plt.figure(figsize=(8,8))

x, y = 5, 4
for i in range(20):  
    plt.subplot(y, x, i+1)
    plt.title(str(y_train[i]))
    plt.axis('off')
    plt.imshow(x_train[i].reshape((28,28)),interpolation='nearest')
plt.show()


# In[ ]:


plt.figure(figsize=(8,8))

x, y = 5, 4
for i in range(20):  
    plt.subplot(y, x, i+1)
    #plt.title(str(y_train[i]))
    plt.axis('off')
    plt.imshow(x_test[i].reshape((28,28)),interpolation='nearest')
plt.show()


# In[ ]:


concat_df = pd.concat([train_df, test_df])


# In[ ]:


features = [col for col in train_df.columns if col.startswith('pixel')]
X_train, X_val, y_train, y_val = train_test_split(train_df[features], train_df['label'], test_size=0.25, random_state=42)


# In[ ]:


def acc(y_true, y_pred):
    return round(accuracy_score(y_true, y_pred) * 100, 2)

clf = DecisionTreeClassifier(max_depth=10, random_state=42)
clf.fit(X_train, y_train)

train_preds_baseline = clf.predict(X_train)
val_preds_baseline = clf.predict(X_val)
acc_baseline_train = acc(train_preds_baseline, y_train)
acc_baseline_val = acc(val_preds_baseline, y_val)

print(f'Training accuracy for our baseline (using all pixel features): {acc_baseline_train}%')
print(f'Validation accuracy for our baseline (using all pixel features): {acc_baseline_val}%')


# In[ ]:


tsvd = TruncatedSVD(n_components=50).fit_transform(concat_df[features])


# In[ ]:


tsvd_cols = [f'component_{i+1}' for i in range(50)]
tsvd_train = pd.DataFrame(tsvd[:len(train_df)], columns=[tsvd_cols])
tsvd_test = pd.DataFrame(tsvd[len(train_df):], columns=[tsvd_cols])


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(tsvd_train, 
                                                  train_df['label'], 
                                                  test_size=0.25, 
                                                  random_state=42)


# In[ ]:


# Train model with t-svd features
clf = DecisionTreeClassifier(max_depth=10, random_state=42)
clf.fit(X_train, y_train)


# In[ ]:


train_preds = clf.predict(X_train)
val_preds = clf.predict(X_val)
acc_tsvd_train = acc(train_preds, y_train)
acc_tsvd_val = acc(val_preds, y_val)
print(f'Training accuracy with TSVD features (50 components): {acc_tsvd_train}%')
print(f'Validation accuracy with TSVD features (50 components): {acc_tsvd_val}%')
# Check out how it performed compared to the baseline
acc_diff = round(acc_tsvd_val - acc_baseline_val, 2)
print(f'\nThis is a difference of {acc_diff}% in validation accuracy compared to the baseline.')


# ## T_SNE

# In[ ]:


get_ipython().run_cell_magic('time', '', 'tsne = TSNE()\ntransformed = tsne.fit_transform(tsvd)  ')


# In[ ]:


tsne_train = pd.DataFrame(transformed[:len(train_df)], columns=['component1', 'component2'])
tsne_test = pd.DataFrame(transformed[len(train_df):], columns=['component1', 'component2'])


# In[ ]:


plt.figure(figsize=(14, 14))
plt.title(f"Visualization of t-SNE results on the MNIST Dataset\nAmount of datapoints = {len(tsne_train)}", fontsize=20, weight='bold')
sns.scatterplot("component1", "component2", 
                data=tsne_train, hue=train_df['label'], 
                palette="Set1", legend="full", alpha=0.6)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Component 1", fontsize=16)
plt.ylabel("Component 2", fontsize=16)
plt.legend(fontsize=16)
plt.show()


# In[ ]:




