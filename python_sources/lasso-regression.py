from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from scipy.stats import skew

# =================================================
# Import dataset
# =================================================
# Concatenate train and test (adding a flag) to
# factorize data preparation steps
train = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)
train['train_test'] = 1
test['train_test'] = 0
y = np.log(train['SalePrice'])
x = pd.concat([train.drop(['SalePrice'], axis=1), test], axis=0)

# =================================================
# Data preparation
# =================================================
# Cast categorical features
categorical_feats = x.columns[x.dtypes == 'object'].tolist()
for col in categorical_feats:
    x[col].fillna('No', inplace=True)
    x[col] = x[col].astype("category")
# Fill Nan with median for numerical features
nan_feats = x.columns[~(x.dtypes == 'object') & (x.isnull().any())].tolist()
for col in nan_feats:
    x[col].fillna(x[col].median(), inplace=True)
# Unskew feature
numeric_feats = x.columns[x.dtypes != "object"].tolist()
numeric_feats.remove('train_test')
# -- Method 1: Compulte log transformation
if False:
    skewed_feats = x[numeric_feats].apply(lambda c: skew(c.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    x[skewed_feats] = np.log1p(x[skewed_feats])
# -- Method 2: Rank (uniform dist)
if True:
    x[numeric_feats] = x[numeric_feats].rank()
# One-hot encoding
x_dummified = pd.get_dummies(x[categorical_feats])
for col in categorical_feats:
    del x[col]
x[x_dummified.columns] = x_dummified


# =================================================
# Grid search
# =================================================
# Search `alpha` regression parameters
x_train = x[x.train_test == 1]
lasso = linear_model.Lasso(max_iter=1e2, normalize=True)
alphas = np.logspace(-5, -3, 10)
scores = []
scores_std = []
for alpha in alphas:
    lasso.alpha = alpha
    this_scores = np.sqrt(-model_selection.cross_val_score(lasso, x_train, y, cv=5, scoring='mean_squared_error'))
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))
print('Min value {} reached for alpha={}'.format(min(scores), alphas[scores.index(min(scores))]))
plt.figure(figsize=(4, 3))
plt.semilogx(alphas, scores)
plt.semilogx(alphas, np.array(scores) + np.array(scores_std) / np.sqrt(len(x_train)), 'b--')
plt.semilogx(alphas, np.array(scores) - np.array(scores_std) / np.sqrt(len(x_train)), 'b--')
plt.ylabel('CV score')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.savefig('lasso_lars.png')

# =================================================
# Train, apply & submit
# =================================================
lasso.alpha = 0.0001291549665014884
#lasso.alpha = 7.742636826811278e-05
lasso.fit(x_train, y) 
y_pred = lasso.predict(x[x.train_test == 0])
pd.DataFrame({'Id' : test.index, 'SalePrice': np.exp(y_pred)}).to_csv('submit_lasso.csv', index=False)

