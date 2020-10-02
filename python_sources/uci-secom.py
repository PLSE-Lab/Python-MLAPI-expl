#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
data = pd.read_csv('../input/uci-secom.csv')
print(data.shape)
print(data.head(5))
import os


# In[ ]:


data.isnull().any().any()


# In[ ]:


d = data.isnull().sum()
j = []
for i in d.keys():
    if(d[i] >900):
        print(i, d[i])
        j.append(i)


# In[ ]:


data.drop(j, axis = 1, inplace = True)


# In[ ]:


data.replace(np.nan, 0, inplace = True)
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = np.nan, strategy = 'mean')
data.isnull().any().any()


# In[ ]:


from sklearn.preprocessing import Imputer 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, average_precision_score


# In[ ]:


data['Pass/Fail'].value_counts()#-1 is pass and 1 is fail


# In[ ]:


x = data.iloc[:, 1:-1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state = 0)


# In[ ]:


print(x_train.shape)
print(x_test.shape)
print(data.shape)


# In[ ]:


(data['Pass/Fail'] == 1).index


# In[ ]:


sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.transform(x_test)


# In[ ]:


lr = LogisticRegression(C = 1000, penalty = 'l2')
lr.fit(x_train_std, y_train)
y_pred_lr = lr.predict(x_test_std)
print('accuracy score', accuracy_score(y_pred_lr, y_test))
cm = confusion_matrix(y_pred_lr, y_test)
print('precision_score', average_precision_score(y_pred_lr, y_test))
print('recall_score', recall_score(y_pred_lr, y_test))
print('f1_score',f1_score(y_pred_lr,y_test))


# In[ ]:





# In[ ]:


y_test[y_test == -1].sum()


# In[ ]:


isolation = IsolationForest()
isolation.fit(x_train_std, y_train)
y_pred_iso = isolation.predict(x_test_std)
print('accuracy score', accuracy_score(y_pred_iso, y_test))
cmiso = confusion_matrix(y_pred_iso, y_test)
print('precision_score', average_precision_score(y_pred_iso, y_test))
print('recall_score', recall_score(y_pred_iso, y_test))
print('f1_score',f1_score(y_pred_iso,y_test))


# In[ ]:


cmiso


# In[ ]:


print(cmiso[1][1]/ (cmiso[1][1] + cmiso[0][1]))


# In[ ]:


forest = RandomForestClassifier()
forest.fit(x_train_std, y_train)
y_pred_rf = forest.predict(x_test_std)
print('accuracy score', accuracy_score(y_pred_rf, y_test))
print('confusion_matrix', confusion_matrix(y_pred_rf, y_test))
print('precision_score', average_precision_score(y_pred_rf, y_test))


# In[ ]:


y_test[y_test == 1].sum()


# In[ ]:


xgb1 = xgb.XGBClassifier(objective = 'binary:logistic', booster = 'gblinear')
xgb1.fit(x_train_std, y_train)
y_pred_xgb = xgb1.predict(x_test_std)
print('accuracy score', accuracy_score(y_pred_xgb, y_test))
print('confusion_matrix', confusion_matrix(y_pred_xgb, y_test))
print('precision_score', average_precision_score(y_pred_xgb, y_test))


# In[ ]:


from sklearn.model_selection import GridSearchCV
param_grid = {
    'C':[5, 10, 250, 100, 500],
    'penalty':['l1', 'l2']
}
gslr = GridSearchCV(estimator = lr, param_grid = param_grid, n_jobs = -1, cv = 10)
gslr.fit(x_train_std, y_train)
y_pred_gslr = gslr.predict(x_test_std)
print('accuracy score', accuracy_score(y_pred_gslr, y_test))
print('confusion_matrix', confusion_matrix(y_pred_gslr, y_test))
print('precision_score', average_precision_score(y_pred_gslr, y_test))


# In[ ]:


print('confusion_matrix', confusion_matrix(y_pred_gslr, y_test))


# In[ ]:


parameters = { 'max_features':np.arange(5,10),'n_estimators':[500],'min_samples_leaf': [10,50,100,200,500]}

random_grid = GridSearchCV(forest, parameters, cv = 5)
random_grid.fit(x_train_std, y_train)
y_pred_gslr = random_grid.predict(x_test_std)
print('accuracy score', accuracy_score(y_pred_gslr, y_test))
print('confusion_matrix', confusion_matrix(y_pred_gslr, y_test))
print('precision_score', average_precision_score(y_pred_gslr, y_test))


# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(x_train_std, y_train)
y_pred_lda = lda.predict(x_test_std)
print('accuracy score', accuracy_score(y_pred_lda, y_test))
print('confusion_matrix', confusion_matrix(y_pred_lda, y_test))
print('precision_score', average_precision_score(y_pred_lda, y_test))
print('recall_score', recall_score(y_pred_lda, y_test))
print('f1_score',f1_score(y_pred_lda,y_test))


# In[ ]:




