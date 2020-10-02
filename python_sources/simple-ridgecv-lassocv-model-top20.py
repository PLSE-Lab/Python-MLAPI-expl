#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train_test = pd.concat([train, test])
train_test


# In[ ]:


train.describe(include="all")


# In[ ]:


train.head(3)


# In[ ]:


test.head(3)


# In[ ]:


train.shape, test.shape, train_test.shape


# In[ ]:


len(set(train["Id"]))


# In[ ]:


train_test.drop("Id", axis=1, inplace=True)


# In[ ]:


corrmat = train.corr()
fig, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=1, annot=True);


# In[ ]:


top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
plt.figure(figsize=(10, 10))
sns.heatmap(train[top_corr_features].corr(), annot=True, cmap="RdYlBu");


# In[ ]:


sns.barplot(train["OverallQual"], train["SalePrice"]);


# In[ ]:


cols = ["SalePrice", "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
sns.pairplot(train[cols], size=2.5);


# In[ ]:


from scipy import stats
from scipy.stats import norm, skew

sns.distplot(train["SalePrice"], fit=norm);
mu, sigma = norm.fit(train["SalePrice"])
print("\n mu = {:.2f} and sigma = {:.2f}\n".format(mu, sigma))
fig = plt.figure()
stats.probplot(train["SalePrice"], plot=plt);


# In[ ]:


train["SalePrice"] = np.log1p(train["SalePrice"])
y = train["SalePrice"]
stats.probplot(y, plot=plt);


# In[ ]:


plt.scatter(train["GrLivArea"], y);


# In[ ]:


train_nas = train.isnull().sum()
train_nas = train_nas[train_nas>0]
train_nas.sort_values(ascending=False)


# In[ ]:


test_nas = test.isnull().sum()
test_nas = test_nas[test_nas>0]
test_nas.sort_values(ascending=False)


# In[ ]:


print("Find most important fetures")
corr = train.corr()
corr.sort_values(["SalePrice"], ascending=False, inplace=True)
corr["SalePrice"]


# In[ ]:


categorical_features = train_test.select_dtypes(include=["object"]).columns
categorical_features


# In[ ]:


numerical_features = train_test.select_dtypes(exclude=["object"]).columns
numerical_features


# In[ ]:


numerical_features = numerical_features.drop("SalePrice")
print("Numerical features:", str(len(numerical_features)))
print("Categorical features:", str(len(categorical_features)))
train_test_num = train_test[numerical_features]
train_test_cat = train_test[categorical_features]


# In[ ]:


print("NaN for numerical features in train:", str(train_test_num.isnull().values.sum()))
train_test_num = train_test_num.fillna(train_test_num.median())
print("Remaining NaN for numerical features in train:", str(train_test_num.isnull().values.sum()))


# In[ ]:


from scipy.stats import skew
skewness = train_test_num.apply(lambda x: skew(x))
skewness.sort_values(ascending=False)


# In[ ]:


skewness = skewness[abs(skewness)>0.5]
skewness.index


# In[ ]:


skew_features = train_test[skewness.index]
skew_features.columns


# In[ ]:


skew_features = np.log1p(skew_features)


# In[ ]:


train_test_cat.shape


# In[ ]:


train_test_cat = pd.get_dummies(train_test_cat)
train_test_cat.shape


# In[ ]:


train_test_cat.head()


# In[ ]:


str(train_test_cat.isnull().values.sum())


# In[ ]:


train_test = pd.concat([train_test_cat, train_test_num], axis=1)
train_test.shape


# In[ ]:


train = train_test.iloc[:1460, :]
test = train_test.iloc[1460:, :]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.3, random_state=20)


# In[ ]:


X_train.head(3)


# In[ ]:


n_folds = 10
def rmse_CV_train(model):
    kf = KFold(n_folds, shuffle=True, random_state=2).get_n_splits(train.values)
    rmse = -cross_validate(model, X_train, y_train, scoring="neg_mean_squared_error", cv=kf)["test_score"].mean()
    return "{:.5f}".format(rmse)

def rmse_CV_test(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = -cross_validate(model, X_test, y_test, scoring="neg_mean_squared_error", cv=kf)["test_score"].mean()
    return "{:.5f}".format(rmse)


# In[ ]:


lr = linear_model.LinearRegression()
print("rmse on train", rmse_CV_train(lr))
print("rmse on test", rmse_CV_test(lr))
lr.fit(X_train, y_train)
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

plt.figure(figsize=(15, 7))
plt.scatter(y_pred_train, y_pred_train - y_train, marker="v", s=20, c="red", alpha=0.7, label="Training Data")
plt.scatter(y_pred_test, y_pred_test - y_test, marker="o", s=20, c="blue", alpha=0.7, label="Test Data")
plt.title("Linear Regression")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.ylim([-0.75, 1.75])
plt.legend(loc="best")
plt.hlines(y=0, xmin=10.5, xmax=15, color="black")
plt.show()

plt.figure(figsize=(10, 7))
plt.scatter(y_pred_train, y_train, marker="v", s=20, c="red", alpha=0.8, label="Training data")
plt.scatter(y_pred_test, y_test, marker="o", s=20, alpha=0.8, c="blue", label="Test data")
plt.xlabel("Predicted Values")
plt.ylabel("Real values")
plt.title("Linear Regression")
plt.legend(loc="best")
plt.plot([10.5, 13.5], [10.5, 13.5], c="black");


# In[ ]:


ridge = RidgeCV(alphas = [0.01, 0.1, 1, 10, 20, 30, 40], scoring="neg_mean_squared_error", normalize=True)
ridge.fit(X_train, y_train)
alpha = ridge.alpha_
print("first best alpha", alpha)
ridge = RidgeCV(alphas = [0.4, 0.8, 1, 4], normalize=True)
ridge.fit(X_train, y_train)
alpha = ridge.alpha_
print("second best alpha", alpha)
ridge = RidgeCV(alphas = [i/10 for i in range(1, 20)], normalize=True)
ridge.fit(X_train, y_train)
alpha = ridge.alpha_
print("final best alpha", alpha)
print("-"*30)
print("Ridge training data:", rmse_CV_train(ridge))
print("Ridge test data:", rmse_CV_test(ridge))
print("-"*30)
print("Not 0 count:", (ridge.coef_ != 0).sum(), "|", "0 count:", (ridge.coef_ == 0).sum())
y_train_rdg = ridge.predict(X_train)
y_test_rdg = ridge.predict(X_test)

plt.figure(figsize=(15, 7))
plt.scatter(y_train_rdg, y_train_rdg - y_train, marker="v", s=20, c="red", alpha=0.7, label="Training Data")
plt.scatter(y_test_rdg, y_test_rdg - y_test, marker="o", s=20, c="blue", alpha=0.7, label="Test Data")
plt.title("Ridge CV")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.ylim([-0.75, 1.75])
plt.legend(loc="best")
plt.hlines(y=0, xmin=10.5, xmax=15, color="black")
plt.show()

plt.figure(figsize=(10, 7))
plt.scatter(y_train_rdg, y_train, marker="v", s=20, c="red", alpha=0.8, label="Training data")
plt.scatter(y_test_rdg, y_test, marker="o", s=20, alpha=0.8, c="blue", label="Test data")
plt.xlabel("Predicted Values")
plt.ylabel("Real values")
plt.title("Ridge CV")
plt.legend(loc="best")
plt.plot([10.5, 13.5], [10.5, 13.5], c="black");


# In[ ]:


lasso = LassoCV(alphas=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 30], normalize=True)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_
print("first best alpha:", alpha)
lasso = LassoCV(alphas=[0.00004, 0.00008, 0.0001, 0.0003, 0.0006], normalize=True)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_
print("second best alpha:", alpha)
lasso = LassoCV(alphas=[0.0001 + i/100000 for i in range(1, 20)], normalize=True)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_
print("final best alpha:", alpha)
print("-"*30)
print("Lasso training data:", rmse_CV_train(lasso))
print("Lasso test data:", rmse_CV_test(lasso))
print("-"*30)
print("Not 0 count:", (lasso.coef_ != 0).sum(), "|", "0 count:", (lasso.coef_ == 0).sum())
y_train_las = lasso.predict(X_train)
y_test_las = lasso.predict(X_test)

plt.figure(figsize=(15, 7))
plt.scatter(y_train_las, y_train_las - y_train, marker="v", s=20, c="red", alpha=0.7, label="Training Data")
plt.scatter(y_test_las, y_test_las - y_test, marker="o", s=20, c="blue", alpha=0.7, label="Test Data")
plt.title("Lasso CV")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.ylim([-0.75, 1.75])
plt.legend(loc="best")
plt.hlines(y=0, xmin=10.5, xmax=15, color="black")
plt.show()

plt.figure(figsize=(10, 7))
plt.scatter(y_train_las, y_train, marker="v", s=20, c="red", alpha=0.8, label="Training data")
plt.scatter(y_test_las, y_test, marker="o", s=20, alpha=0.8, c="blue", label="Test data")
plt.xlabel("Predicted Values")
plt.ylabel("Real values")
plt.title("Lasso CV")
plt.legend(loc="best")
plt.plot([10.5, 13.5], [10.5, 13.5], c="black");


# In[ ]:


y_rdg_log = ridge.predict(test)
y_pred = np.exp(y_rdg_log) - 1
df_Id = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")[["Id"]]
y_pred = pd.DataFrame(y_pred, columns=["SalePrice"])
sub = pd.concat([df_Id, y_pred], axis=1)
sub.to_csv("submission1.csv", index=False)


# In[ ]:


y_las_log = lasso.predict(test)
y_pred = np.exp(y_las_log) - 1
y_pred = pd.DataFrame(y_pred, columns=["SalePrice"])
sub = pd.concat([df_Id, y_pred], axis=1)
sub.to_csv("submission2.csv", index=False)


# In[ ]:




