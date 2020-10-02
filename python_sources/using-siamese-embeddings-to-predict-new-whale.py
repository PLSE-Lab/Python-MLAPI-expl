#!/usr/bin/env python
# coding: utf-8

# # Determining 'new_whale' Position Using Siamese-Learned Embeddings
# ## Overview
# Like many others, I found a lot of inspiration from [@martinpiotte](https://kaggle.com/martinpiotte)'s [work](https://www.kaggle.com/martinpiotte/whale-recognition-model-with-score-0-78563) on the playground dataset. The embeddings learned from a Siamese network architecture are such that the distance between embeddings of similar images is less than the distance between embeddings of dissimilar images. This notebook takes the following strategy for predicting out of sample whales:
# * Use siamese-learned embeddings to build an Approximate Nearest Neighbors index
# * Segment available data into:
#   * a support set 
#   * a training/validation set with
#     * features based on the distance to images in the support set
#     * labels that are binary depending on whether the class has an example in the support set
# * Train a model to predict in-sample vs out-of-sample 
# * Investigate how the choice of decision threshold affects precision
# * Choose multiple decision thresholds to determine which position to ultimately place 'new_whale' before submission
# 

# In[ ]:


import base64
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from annoy import AnnoyIndex
from IPython.display import HTML
from sklearn.metrics import roc_curve, auc, precision_score, f1_score, recall_score, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm_notebook
from xgboost import XGBClassifier

LABELS = '../input/humpback-whale-identification/train.csv'
SAMPLE_SUB = '../input/humpback-whale-identification/sample_submission.csv'
MPIOTTE = '../input/mpiotte-standard-features/mpiotte_combined_features.pkl'


# In[ ]:


df = pd.read_csv(LABELS)
sample_df = pd.read_csv(SAMPLE_SUB)
mpiotte_combined = pd.read_pickle(MPIOTTE)


# In[ ]:


df_features = pd.merge(mpiotte_combined, df, left_on='image_name', right_on='Image', how='left', sort=False)

# df_features contains embeddings for images in training and test set
df_features.shape


# In[ ]:


df_features.head()


# In[ ]:


# embeddings are a concatenation of mpiotte_standard and mpiotte_bootstrap. 512 + 512 = 1024
len(df_features.loc[0, 'features'])


# In[ ]:


# separate train/val from test
df_train_val = df_features[ ~df_features.Id.isnull() ].drop(columns=['image_name', 'features'])
df_test = df_features[ df_features.Id.isnull() ].drop(columns=['Id', 'features'])

# separate unlabeled from labeled
new_whales = df_train_val[df_train_val.Id == 'new_whale']
labeled = df_train_val[df_train_val.Id != 'new_whale']

# separate labeled into classes with single observation and classes with multiple observations
single_obs = labeled.groupby('Id').filter(lambda x: x['Id'].count() == 1)
multi_obs = labeled.groupby('Id').filter(lambda x: x['Id'].count()>1)

# stratifying 'multi_obs' by class, every image in 'known' should have at least one image in 'support_train' with the same class
support_train, known = train_test_split(multi_obs, test_size=0.5, random_state=2, stratify=multi_obs.Id)
known['label'] = 'known'

# 'single_obs' and 'new_whale' should have no images in 'support_train'
unknown = pd.concat([new_whales, single_obs])
unknown['label'] = 'unknown'

# this will be the full data set for training and validation
full = pd.concat([unknown, known]).drop(columns=['Id'])
full.head()


# In[ ]:


# build Approximate Nearest Neighbors index
f = 1024
t = AnnoyIndex(f)

for i, row in tqdm_notebook(df_features.iterrows(), total=len(df_features.index)):
    t.add_item(i, row['features'])
    
t.build(1000)


# In[ ]:


# given an element's index and the support indices, return 
# * distance from element to closest neighbor in support
# * average distance from element to 10 closest observations in support
# * average distance from element to 100 closest observations in support
# * average distance from element to all observations in support
def get_metrics(i, support_indices):
    distances = []
    for j in support_indices:
        distances.append(t.get_distance(i, j))
        
    sorted_distance = np.sort(distances)
        
    return pd.Series({
        'min_distance': sorted_distance[0],
        'avg_10_distance': np.mean(sorted_distance[:10]), 
        'avg_100_distance': np.mean(sorted_distance[:100]), 
        'avg_distance': np.mean(distances)
    })


# In[ ]:


# compute distance metrics for train/val set
full.reset_index(inplace=True)
new_columns = full.apply(
    lambda row: get_metrics(row['index'], support_train.index.values.tolist()), 
    axis=1,
    result_type='expand'
)
full = pd.concat([full, new_columns], axis='columns')
full.set_index('index', drop=True, inplace=True)
full.head()


# In[ ]:


# separate into x & y, train & val
features_list = ['min_distance', 'avg_10_distance', 'avg_100_distance', 'avg_distance']
lb = LabelBinarizer()

x_full = full[features_list]
y_full = lb.fit_transform(full.label.values).ravel()

x_train, x_val, y_train, y_val = train_test_split(x_full, y_full, test_size=0.25, random_state=2, stratify=y_full)


# In[ ]:


# fit classifier to train & predict probabilities on val
clf_val1 = XGBClassifier(n_estimators=500)
clf_val2 = RandomForestClassifier(n_estimators=1000, max_depth=10)
clf_val3 = LinearDiscriminantAnalysis()

clf_val = VotingClassifier(
    estimators=[
        ('XGB', clf_val1), 
        ('RF', clf_val2), 
        ('LDA', clf_val3), 
    ],
    voting='soft'
)

clf_val.fit(x_train, y_train)
y_scores = clf_val.predict_proba(x_val)[:, 1]


# ## Investigate choice of threshold

# In[ ]:


def plot_roc_curve(fpr, tpr, label=None):
    """
    The ROC curve, modified from 
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91
    """
    plt.figure(figsize=(8,8))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc='best')
    
fpr, tpr, auc_thresholds = roc_curve(y_val, y_scores)
auc(fpr, tpr)


# In[ ]:


plot_roc_curve(fpr, tpr, 'recall_optimized')


# In[ ]:


p, r, thresholds = precision_recall_curve(y_val, y_scores)


# In[ ]:


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    
plot_precision_recall_vs_threshold(p, r, thresholds)


# In[ ]:


y_hat = (y_scores >= 0.8).astype(int)

precision_score(y_val, y_hat)


# In[ ]:


f1_score(y_val, y_hat)


# In[ ]:


recall_score(y_val, y_hat)


# In[ ]:


roc_auc_score(y_val, y_hat)


# In[ ]:


# fit on full dataset
clf_test1 = XGBClassifier(n_estimators=500)
clf_test2 = RandomForestClassifier(n_estimators=1000, max_depth=10)
clf_test3 = LinearDiscriminantAnalysis()

clf_test = VotingClassifier(
    estimators=[
        ('XGB', clf_test1),
        ('RF', clf_test2),  
        ('LDA', clf_test3), 
    ],
    voting='soft'
)

clf_test.fit(x_full, y_full) 


# In[ ]:


# when computing distance metrics for test set, use all labeled points as support
support_test = labeled


# In[ ]:


# compute distance metrics for test set
df_test.reset_index(inplace=True)
new_columns = df_test.apply(
    lambda row: get_metrics(row['index'], support_test.index.values.tolist()), 
    axis=1,
    result_type='expand'
)
df_test = pd.concat([df_test, new_columns], axis='columns')
df_test.set_index('index', drop=True, inplace=True)


# In[ ]:


# predict probabilities and apply thresholds
df_test['prob'] = clf_test.predict_proba(df_test[features_list])[:, 1]
df_test['nw_num'] = df_test.apply(lambda row: 1 if row['prob'] >=0.85 else (2 if row['prob'] > 0.5 else (3 if row['prob'] > 0.05 else 5)), axis=1)
df_test.head()


# In[ ]:


# generate output
submission = pd.merge(
    sample_df.drop(columns=['Id']), 
    df_test[['image_name', 'prob', 'nw_num']], 
    left_on='Image', 
    right_on='image_name', 
    how='left', 
    sort=False
).drop(columns=['image_name'])
submission.head()


# In[ ]:


# sanity check: 
# training set has ~38% new_whale and public LB ~27%
len(submission[submission.nw_num == 1].index) / len(submission.index)


# In[ ]:


submission.to_csv('submission_new_whale.csv', header=True, index=False)


# In[ ]:


def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  
    csv = df.to_csv(header=True, index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(submission)


# In[ ]:




