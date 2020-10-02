#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
#19
np.random.seed(19)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier  
from vecstack import stacking
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score
from sklearn.ensemble import GradientBoostingClassifier
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.head()
df.shape
X = df.drop(columns = ['target'])
y = df['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) 
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(13).plot(kind='barh')
plt.xlabel('Score of features')
plt.ylabel('Features')
plt.show()
models = [KNeighborsClassifier(metric='euclidean',weights='uniform'),
          RandomForestClassifier(bootstrap=True, max_depth=90,max_features=3,min_samples_leaf=3,min_samples_split=8,n_estimators=200),
          XGBClassifier(learning_rate=0.01,n_estimators=100, max_depth=10)]
S_train, S_test = stacking(models,X_train, y_train, X_test, regression=False, mode='oof_pred_bag',metric=accuracy_score,n_folds=4,stratified=True,
                           shuffle=True,random_state=0,verbose=2)
model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 
                      n_estimators=100, max_depth=10)
    
model = model.fit(S_train, y_train)
y_pred = model.predict(S_test)
print('Final prediction score: [%.8f]' % accuracy_score(y_test, y_pred))
print('Final precision score: [%.8f]' % precision_score(y_test, y_pred))
print('Final recall score: [%.8f]' % recall_score(y_test, y_pred))
print('Final f1 score: [%.8f]' % f1_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
y_pred.ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_keras = auc(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='ROC Curve (Area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

