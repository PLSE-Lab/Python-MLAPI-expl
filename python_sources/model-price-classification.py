#!/usr/bin/env python
# coding: utf-8

# I want to try classifying mobile price. I do some step:
# 1. Check Imbalanced Data
# 2. Check Missing Value
# 3. Check Correlation
# 4. Search the best methods to predict mobile price

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data_train = pd.read_csv("/kaggle/input/mobile-price-classification/train.csv")
data_test = pd.read_csv("/kaggle/input/mobile-price-classification/test.csv")


# **Check Imbalanced Data**<br>
# Imbalanced Data is a situation where a class in variable respon dominate than other class. Imbalanced Data can decreased predict accuracy.
# Some methods to solve Imbalanced Data:
# 1. Resampling Dataset (Under Sampling, Over Sampling)
# 2. SMOTE

# In[ ]:


fig,ax=plt.subplots(figsize=(5,3))
data_train.groupby("price_range").count()[['blue']].plot(kind='bar',ax=ax,legend=False,color='blue')
for i in ax.spines:
    if i!="left" and i!="bottom":
        ax.spines[i].set_visible(False)
ax.set_title("Price_Range",fontdict={'fontweight':'bold'})
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(),rotation=0)
plt.show()


# The plot show variable respon in dataset balanced.

# **Check Missing Value**<br>

# In[ ]:


data_train.info()


# Data train don't have missing value.

# **Check Correlation**<br>
# Correlation used to see the most feature influence variable respon.

# In[ ]:


fig,ax=plt.subplots(figsize=(15,7))
sns.heatmap(data_train.corr(),annot=True,fmt='.1g',cmap="Blues")
plt.show()


# ram is the most feature influence mobile price. Correlation betweet ram and price is 0.9

# **Classification**<br>
# Classification methods:
# 1. SVC (SVM Classifier)
# 2. DecisionTreeClassifier
# 3. Logistic Regression
# 4. ExtraTreeClassifier
# 5. Naive Bayes (GaussianNB)
# 6. Gradient Boosting
# 7. Random Forest
# 8. AdaBoostClassifier

# Before using classification methods, i detect **importance feature** with **SelectKBest and PCA**.

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


list_methods = [SVC,DecisionTreeClassifier,ExtraTreesClassifier,LogisticRegression,
                GaussianNB,GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier]
methods_name = ["SVC","DecisionTreeClassifier","ExtraTreesClassifier","LogisticRegression",
                "GaussianNB","GradientBoostingClassifier","RandomForestClassifier","AdaBoostClassifier"]


# In[ ]:


X = data_train.drop(['price_range'],axis=1)
y = data_train['price_range']
X_train,X_test,y_train,y_test = train_test_split(X,y)


# In[ ]:


def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)


# **Searching the best classification methods with detect importance feature by PCA.**

# In[ ]:


classification_result = []
for j in range(len(list_methods)):
    model_result = []
    for i in range(1,len(X.columns)+1):
        pca = PCA(n_components=i)
        pca.fit(X_train,y_train)
        X_t_train = pca.fit_transform(X_train)
        X_t_test = pca.fit_transform(X_test)
        clf = RandomForestClassifier()
        clf.fit(X_t_train,y_train)
        model_result.append({'n_feature':i,methods_name[j]:clf.score(X_t_test,y_test)})
    classification_result.append(pd.DataFrame(model_result))

df_result = classification_result[0]
for i in range(1,len(classification_result)):
    df_result = df_result.merge(classification_result[i],on='n_feature')
df_result = df_result[['n_feature']+methods_name]
df_result = df_result.set_index('n_feature')


# In[ ]:


df_result.style.apply(highlight_max)


# Using PCA on detect importance feature may be not suitable. Because predict accuracy is resulted is so small for all methods.

# **Searching the best classification methods with detect importance feature by SelectKBest.**

# In[ ]:


result = SelectKBest(score_func=chi2,k=10)
hasil = result.fit(np.array(X),np.array(y))
feature_select = pd.DataFrame({'result':hasil.scores_,'field':X.columns}).sort_values('result',ascending=False)
feature_select = feature_select.reset_index(drop=True)


# In[ ]:


print(feature_select)


# **ram** is the most importance feature to predict mobile price. Then px_height, battery_power, etc.

# In[ ]:


classification_result = []
for j in range(len(list_methods)):
    model_result = []
    for i in range(1,len(feature_select)+1):
        X_f_train = X_train[list(feature_select['field'])[:i]]
        X_f_test = X_test[list(feature_select['field'])[:i]]
        clf = list_methods[j]()
        clf.fit(X_f_train,y_train)
        model_result.append({'n_feature':i,methods_name[j]:clf.score(X_f_test,y_test)})
    classification_result.append(pd.DataFrame(model_result))
df_result = classification_result[0]
for i in range(1,len(classification_result)):
    df_result = df_result.merge(classification_result[i],on='n_feature')
df_result = df_result[['n_feature']+methods_name]
df_result = df_result.set_index("n_feature")


# In[ ]:


df_result.style.apply(highlight_max)


# Highest accuracy for predicting mobile price is 0.91. The accuracy is obtained when using ExtraTreeClassifier methods with 4 feature. The feature are **ram, px_height, battery_power, and px_width**.

# In[ ]:


X_f_train = X_train[list(feature_select['field'])[:4]]
X_f_test = X_test[list(feature_select['field'])[:4]]
clf = ExtraTreesClassifier()
clf.fit(X_f_train,y_train)


# Prediction result : ExtraTreeClassifier methods with 4 feature

# In[ ]:


clf.predict(data_test[list(feature_select['field'])[:4]])


# In[ ]:




