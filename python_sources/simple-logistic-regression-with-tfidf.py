#!/usr/bin/env python
# coding: utf-8

# ## Predict if the Job Descriptions are Real or Fraud 
# 
# Let's quickly create a baseline model in this kernel

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold


# In[ ]:


df = pd.read_csv("../input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv")
df.head()


# 1. Feature Engineering
# 
# Create features from text columns : length of text and TFIDF

# In[ ]:


cols = ["title", "company_profile", "description", "requirements", "benefits"]
for c in cols:
    df[c] = df[c].fillna("")

def extract_features(df):    
    for c in cols:
        df[c+"_len"] = df[c].apply(lambda x : len(str(x)))
        df[c+"_wc"] = df[c].apply(lambda x : len(str(x.split())))

    
extract_features(df)


# Create TF IDF Features

# In[ ]:


df['combined_text'] = df['company_profile'] + " " + df['description'] + " " + df['requirements'] + " " + df['benefits']

n_features = {
    "title" : 100,
    "combined_text" : 500
}

for c, n in n_features.items():
    tfidf = TfidfVectorizer(max_features=n, norm='l2', stop_words = 'english')
    tfidf.fit(df[c])
    tfidf_train = np.array(tfidf.transform(df[c]).toarray(), dtype=np.float16)

    for i in range(n_features[c]):
        df[c + '_tfidf_' + str(i)] = tfidf_train[:, i]


# One Hot Encoding for Categorical Columns

# In[ ]:


cat_cols = ["employment_type", "required_experience", "required_education", "industry", "function"]
for c in cat_cols:
    encoded = pd.get_dummies(df[c])
    df = pd.concat([df, encoded], axis=1)


# Prepare Dataset : Drop unnecessary columns

# In[ ]:


drop_cols = ['title', 'location', 'department', 'salary_range', 'company_profile', 'description', 'requirements', 'benefits', 'combined_text']
drop_cols += cat_cols
df = df.drop(drop_cols, axis = 1)
df.head()


# Build a Simple Logistic Model

# In[ ]:


df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
idd, target = "job_id", "fraudulent"
features = [f for f in df.columns if f not in [idd, target]]

X = df[features]
y = df[target]

kf = RepeatedKFold(n_splits=3, n_repeats=1, random_state=0)
auc_buf = []   
cnt = 0
for train_index, valid_index in kf.split(X):
    print('Fold {}'.format(cnt + 1))

    train_x,train_y = X.loc[train_index], y.loc[train_index]
    test_x, test_y = X.loc[valid_index], y.loc[valid_index]
    
    clf = LogisticRegression(max_iter = 5000).fit(train_x, train_y)
    preds = clf.predict(test_x)
    
    auc = roc_auc_score(test_y, preds)
    print('{} AUC: {}'.format(cnt, auc))
    auc_buf.append(auc)

    cnt += 1

auc_mean = np.mean(auc_buf)
auc_std = np.std(auc_buf)
print('AUC = {:.6f} +/- {:.6f}'.format(auc_mean, auc_std))


# In[ ]:




