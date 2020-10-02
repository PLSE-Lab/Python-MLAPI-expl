#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn import tree
from sklearn import metrics


# In[ ]:


df1 = read_csv("../input/student-mat.csv")
df2 = read_csv("../input/student-por.csv")
result = df1.append(df2)
result = result.reset_index(drop=True)

#Normalize absences value
fabs = result[["absences"]].values
fabs = np.where(fabs > 10, 1, 0)

#Represent Alcohol Consumption in One Week
target = ((result[["Walc"]].values * 2) + (result[["Dalc"]].values * 5)/7)
target = np.where(target >= 3, 1, 0)


# In[ ]:


res = pd.DataFrame(fabs)
res.columns = ['fabs']

res2 = pd.DataFrame(target)
res2.columns = ['target']

fix = result.join(res, how='outer')
fix2 = fix.join(res2, how='outer')

del fix2['Walc']
del fix2['Dalc']
del fix2['absences']


# In[ ]:


fix2 = fix2.reindex(np.random.permutation(fix2.index))
X = fix2[fix2.columns[0:31]]
Y = fix2['target'].astype('category')

olist = list(X.select_dtypes(['object']))
for col in olist:
    X[col] = X[col].astype('category').cat.codes


# In[ ]:


'''
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier()
clf = clf.fit(X, Y)
model = SelectFromModel(clf, prefit=True)
New_features = model.transform(X)
print(New_features.shape)
model.get_support()
'''


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


# In[ ]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
clf.score(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test)
acc_test = metrics.accuracy_score(y_test,y_pred)
acc_test


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(X_train, y_train)
forest.score(X_train, y_train)
y_pred = forest.predict(X_test)
acc_test = metrics.accuracy_score(y_test,y_pred)
acc_test


# In[ ]:


importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print (importances)
importances.plot.bar()

