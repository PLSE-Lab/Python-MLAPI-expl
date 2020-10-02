#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("../input/deepspeech-labels.csv")

# NaN in the dstext column means the empty string
df.loc[df['dstext'].isnull(), 'dstext'] = '-'


# In[ ]:


df.head(30)


# ### check what fraction of the labels to predict ****DeepSpeech got exactly right

# In[ ]:


# does not include 'silence' nor 'unknown'
labels_to_predict = "yes no up down left right on off stop go".split(' ')


# In[ ]:


# number of train samples for each class to predict
labels_to_predict_counts = df[df['label'].isin(labels_to_predict)].groupby('label')['label'].agg('count').to_frame().rename(
    columns = { "label": "count"})
labels_to_predict_counts


# In[ ]:


# turn into a dict
tmp = labels_to_predict_counts.to_dict()['count']

# and put in label order (for normalization)
counts_per_label = np.array([ tmp[label] for label in labels_to_predict])

del tmp


# ### confusion matrix

# In[ ]:


# see also http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(df['label'], df['dstext'], labels = labels_to_predict)


# Looking at the confusion matrix there is almost no 'cross talk' between
# the labels (mispredictions are mainly other words not found in the
# set of labels to predict).

# In[ ]:


conf_mat


# In[ ]:


# normalize 
conf_mat_rel = conf_mat.astype('float') / counts_per_label.astype('float')

plt.figure(figsize = (10,10))
plt.gca().imshow(conf_mat_rel, cmap = plt.cm.Blues)

ticks = np.arange(len(labels_to_predict))
plt.xticks(ticks, labels_to_predict, fontsize = 14)
plt.yticks(ticks, labels_to_predict, fontsize = 14)

plt.xlabel('prediction', fontsize = 14)
plt.ylabel('true label', fontsize = 14)


plt.title("correctly predicted labels (percent)", fontsize = 14)
# put numeric values (percent)
# conf_mat.sum(axis = 1)
for i in ticks:
    for j in ticks:
        plt.text(j,i, "%.1f" % (conf_mat_rel[i,j] * 100),
                     color="white" if conf_mat_rel[i, j] > 0.5 else "black",
                    horizontalalignment="center")


# ### look at highest frequency recognized text for each label

# In[ ]:


for true_label in df['label'].unique():
    counts = df[df['label'] == true_label].groupby('dstext').size().sort_values(ascending = False).to_frame().rename(columns ={ 0: "count"})
    counts['fraction'] = counts['count'] / float(counts.sum())
    
    print("=====")
    print("true label", true_label)
    print("=====")
    print(counts)


# ### find overlapping texts
# i.e. recognized texts which appear in more than one class

# In[ ]:


# recognized text is on the left, labels for which such a text was recognized on the right
df.groupby('dstext').label.unique()


# In[ ]:


tmp = df.groupby('dstext').label.nunique().sort_values(ascending = False).to_frame().rename(
    columns = { "label": "num_labels"})
tmp


# In[ ]:




