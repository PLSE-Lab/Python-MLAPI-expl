#!/usr/bin/env python
# coding: utf-8

# This is a revision of earlier attempted Titanic exercise

# In[ ]:


# Import all required modules
import pandas as pd
import numpy as np
from sklearn import linear_model,model_selection,preprocessing


# In[ ]:


# Read training data
df_train = pd.read_csv("../input/titanic/train.csv")
X = df_train[['Sex','Age','Pclass']]
y = df_train['Survived']


# In[ ]:


# Preprocess the data for better fitting
X['Sex'] = X['Sex'].apply(lambda x:1 if x=="female" else 0)
X['Age'].fillna(value=np.mean(X['Age']),inplace=True)
X


# In[ ]:


# Apply logistic regression algorithm
model = linear_model.LogisticRegression(solver='lbfgs')
model.fit(X,y)
model.classes_,model.coef_,model.intercept_
model.score(X,y)


# In[ ]:


C_vals = [1e-3,0.1,1,100,1000,1e5]
grdsrch = model_selection.GridSearchCV(estimator=model,param_grid={'C':C_vals},return_train_score=True,cv=5)
gs_fit = grdsrch.fit(X,y)
gs_fit.best_score_,gs_fit.best_params_,gs_fit.cv_results_


# In[ ]:


# Read the test data
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
X_test = df_test[['Sex','Age','Pclass']]
# Preprocess the data for better fitting
X_test['Sex'] = X_test['Sex'].apply(lambda x:1 if x=="female" else 0)
X_test['Age'].fillna(value=np.mean(X_test['Age']),inplace=True)


# In[ ]:


#X_test
y_pred = grdsrch.predict(X_test)
y_pred


# In[ ]:


pd.DataFrame({'PassengerId':df_test.PassengerId,'Survived':y_pred},columns=["PassengerId","Survived"]).to_csv("my_submission_2.csv",index=False)


# In[ ]:


X_train_norm = preprocessing.normalize(X)
X_test_norm = preprocessing.normalize(X_test)


# In[ ]:


gs_fit = grdsrch.fit(X_train_norm,y)
gs_fit.best_score_,gs_fit.best_estimator_,gs_fit.cv_results_


# In[ ]:


y_pred = grdsrch.predict(X_test_norm)
y_pred
pd.DataFrame({'PassengerId':df_test.PassengerId,'Survived':y_pred},columns=["PassengerId","Survived"]).to_csv("my_submission_2.csv",index=False)

