#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


colors = np.array(['#FE8222', '#4965D1'])


# In[ ]:


dataset = pd.read_csv('../input/whirlpool_galaxy.csv')


# In[ ]:


dataset.info()


# In[ ]:


dataset.head(4)


# In[ ]:


dataset.tail(4)


# In[ ]:


dataset.sample(4)


# In[ ]:


dataset.describe()


# In[ ]:


dataset.groupby('label').mean()


# In[ ]:


dataset.groupby('label').median()


# In[ ]:


dataset.groupby('label').std()


# In[ ]:


dataset.groupby('label').agg(['min', 'max'])


# In[ ]:


dataset.hist(bins = 20, figsize = (10, 10))
plt.show()


# In[ ]:


sns.heatmap(
    dataset.corr(),
    square = True,
    annot = True
)
plt.show()


# In[ ]:


sns.pairplot(
    dataset.sample(300), 
    hue = 'label', 
    vars = dataset.columns[1:]
)
plt.show()


# In[ ]:


from sklearn import preprocessing

labels = dataset.iloc[:, 0].values
X = dataset.iloc[:, 1:].values
X_scaled = preprocessing.scale(X.astype(np.float64))


# In[ ]:


from sklearn.decomposition import PCA

data = PCA(n_components = 2).fit_transform(X_scaled)
plt.scatter(data[:, 0], data[:, 1], c = colors[labels])
plt.show()


# In[ ]:


from sklearn.manifold import Isomap

data = Isomap(n_components = 2).fit_transform(X_scaled)
plt.scatter(data[:, 0], data[:, 1], c = colors[labels])
plt.show()


# In[ ]:


from sklearn.manifold import TSNE

data = TSNE(n_components = 2, n_iter = 1000).fit_transform(X_scaled)
plt.scatter(data[:, 0], data[:, 1], c = colors[labels])
plt.show()


# In[ ]:


from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

models = [
    SVC(gamma = 'auto'),
    KNeighborsClassifier(),
    LogisticRegression(solver = 'lbfgs'),
    CatBoostClassifier(logging_level = 'Silent'),
    RandomForestClassifier(n_estimators = 1000, max_depth = 100)
]

for model in models:
    scores = cross_val_score(model, X_scaled, labels, cv = 4, scoring = 'f1')
    print(model.__class__.__name__, np.round(scores, 2))


# In[ ]:


from catboost import Pool, cv

params = {
    'iterations': 1000,
    'depth': 2,
    'loss_function': 'Logloss',
    'verbose': False,
    'custom_metric': 'AUC'
}

cv_dataset = Pool(data = X, label = labels)
scores = cv(cv_dataset, params, fold_count = 2, plot = 'True')


# In[ ]:


scores[['test-Logloss-mean', 'train-Logloss-mean']].plot()
plt.show()


# In[ ]:


from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

efs = EFS(
    RandomForestClassifier(n_estimators = 500),
    scoring = 'f1',
    min_features = 1,
    max_features = 4,
    cv = 5,
    print_progress = False
).fit(X, labels)

print(efs.best_score_, efs.best_idx_)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels)


# In[ ]:


model = LogisticRegression(solver = 'lbfgs').fit(X_train, y_train)


# In[ ]:


f1_score(y_test, model.predict(X_test))


# In[ ]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[ ]:


z = model.intercept_[0] +     (model.coef_[0][0] * X_test[:, 0]) +     (model.coef_[0][1] * X_test[:, 1]) +     (model.coef_[0][2] * X_test[:, 2]) +     (model.coef_[0][3] * X_test[:, 3])


# In[ ]:


y_pred = (sigmoid(z) > 0.5).astype(np.int64)


# In[ ]:


f1_score(y_test, y_pred)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


confusion_matrix(y_test, y_pred)

