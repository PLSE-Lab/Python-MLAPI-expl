#!/usr/bin/env python
# coding: utf-8

# # **Quality of Wine**

# ## Summary
# 1. Data Preparation
# 2. Training and Evaluation
# 3. Parameter tuning
# 4. Feature importance
# 

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import warnings

sns.set(style="white")
warnings.filterwarnings('ignore')


# **Data Preparation**

# In[ ]:


#Acquire data
dataset = pd.read_csv("../input/winequality-red.csv")
dataset.head()


# In[ ]:


#To see if the data need to be corrected
dataset.info()


# In[ ]:


dataset.describe()


# In[ ]:


sns.countplot(dataset['quality'])


# In[ ]:


#Make the problem binary, 0 for bad wine and 1 for good wine
dataset["quality"] = dataset.quality.map(lambda x : 1 if x > 6 else 0)
sns.countplot(dataset["quality"])


# In[ ]:


#Analyse the correlation between the features
corr = dataset.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:


#Separate input and output variables 
y = dataset["quality"]
X = dataset.drop("quality", axis=1)
#Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
X.head()


# **Training and Testing**

# In[ ]:


def train_evaluate(clf, X, y, n_splits, name):
    results = {}
    results["Precision"] = cross_val_score(clf, X, y, cv=n_splits, scoring = make_scorer(precision_score, average="micro")).mean()
    results["Recall"] = cross_val_score(clf, X, y, cv=n_splits, scoring = make_scorer(recall_score, average="micro")).mean()
    results["F1"] = cross_val_score(clf, X, y, cv=n_splits, scoring = make_scorer(f1_score, average="micro")).mean()
    df_results = pd.DataFrame(data=results, index=[name])
    return df_results


# In[ ]:


df_results = pd.DataFrame()
clf_A = XGBClassifier()
clf_B = SVC(gamma="auto")
clf_C = RandomForestClassifier()
clf_D = ExtraTreesClassifier()
for clf in [clf_A, clf_B, clf_C, clf_D]:
    df_result = train_evaluate(clf, X, y, 4, clf.__class__.__name__)
    df_results = df_results.append(df_result)
print(df_results.sort_values("F1", ascending=False))


# **Parameter Tuning**

# In[ ]:


# Optimizing SVC model
parameters = {
            'C': [0.01, 0.1, 1, 1.2, 1.4],
            'kernel':['linear', 'rbf'],
            'gamma' :[0.01, 0.1, 0.5, 0.9, 1]
}
learner = SVC()
scorer = make_scorer(f1_score, average="micro")
clf = GridSearchCV(learner, parameters, scoring=scorer, cv=4)
clf = clf.fit(X, y)
best_clf = clf.best_estimator_
df_result = train_evaluate(clf, X, y, 4, "SVC Optimized")
df_results = df_results.append(df_result)
print(df_results.sort_values("F1", ascending=False))


# **Feature Importance**

# In[ ]:


clf = ExtraTreesClassifier()
clf.fit(X,y)
df = pd.DataFrame({"importance":clf.feature_importances_, "features":X.columns.values})
df = df.sort_values("importance", ascending=False)
sns.barplot(x="importance", y="features", data=df)

