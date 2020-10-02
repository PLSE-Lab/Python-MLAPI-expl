#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/Churn_Modelling.csv", header = 0, index_col=0)
print(df.shape)
df.head()


# In[ ]:


len(df.CustomerId.unique())


# In[ ]:


source_data = df.copy()

df.drop("CustomerId", axis=1, inplace=True)
df.drop("Surname", axis=1, inplace=True)

df.Geography = df.Geography.astype('category')
df.Gender = df.Gender.astype('category')

df.info()


# In[ ]:


for feature in df.dtypes[df.dtypes == 'category'].index:
    sns.countplot(y=feature, data=df, order = df[feature].value_counts().index)
    plt.show()


# In[ ]:


df.hist(figsize=(10,10), xrot=-45)
plt.show()


# In[ ]:


plt.figure(figsize=(7,6))
sns.heatmap(df.corr(), cmap="Reds")


# Data is imbalanced

# In[ ]:


len(df[df.Exited == 1].index)/10000


# Some features correlate a bit

# In[ ]:


df.corr().at['Age', 'Exited']


# In[ ]:


df.corr().at['Balance', 'NumOfProducts']


# In[ ]:


df.corr().at['IsActiveMember', 'Exited']


# In[ ]:


df2 = pd.get_dummies(df, drop_first=True)
x = df2.drop("Exited", axis=1)
y = df2.Exited


# In[ ]:


X = (x-np.min(x))/(np.max(x)-np.min(x))


# In[ ]:


from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_auc_score, confusion_matrix


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# Logistic  Regression

# In[ ]:


logreg_model = LogisticRegression(solver='liblinear')
logreg_selector = RFECV(estimator=logreg_model, step=1, cv=10)
fit = logreg_selector.fit(X_train, y_train)


# In[ ]:


print ('The selected features are: ' + '{}'.format([feature for feature,s in zip(X.columns, fit.support_) if s]))


# In[ ]:


print('Mean of grid scores: ' + '{}'.format(fit.grid_scores_.mean()))
print('R2 score: ' + '{}'.format(fit.score(X_test,y_test)))
print('ROC AUC Score: ' +'{}'.format( roc_auc_score(y_test, fit.predict_proba(X_test)[::,1])))
confusion_matrix(y_test, fit.predict(X_test))


# Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf_fit = rf.fit(X_train, y_train)

print(rf_fit.score(X_test,y_test))
confusion_matrix(y_test, rf_fit.predict(X_test))


# Gradient Boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=100)
gb_fit = gb.fit(X_train, y_train)

print(gb_fit.score(X_test,y_test))
confusion_matrix(y_test, gb_fit.predict(X_test))

