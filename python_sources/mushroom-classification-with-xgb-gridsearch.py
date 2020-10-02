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


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
df.head()


# In[ ]:


df.isnull().mean()


# In[ ]:


df.dtypes


# In[ ]:


df.describe().T


# In[ ]:


list(df.columns)


# 1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
# 2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
# 3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
# 4. bruises?: bruises=t,no=f
# 5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
# 6. gill-attachment: attached=a,descending=d,free=f,notched=n
# 7. gill-spacing: close=c,crowded=w,distant=d
# 8. gill-size: broad=b,narrow=n
# 9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
# 10. stalk-shape: enlarging=e,tapering=t
# 11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
# 12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
# 15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
# 16. veil-type: partial=p,universal=u
# 17. veil-color: brown=n,orange=o,white=w,yellow=y
# 18. ring-number: none=n,one=o,two=t
# 19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
# 20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
# 21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
# 22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d

# In[ ]:


plt.figure(figsize=(16, 100))
columns = list(df.columns)
# histogram
for i in range(1,len(columns)):
    plt.subplot(11, 2, i)
    plt.title(columns[i])
    sns.countplot(x=columns[i],hue='class',data=df, palette='RdBu')


# In[ ]:


colunms_to_use = ['bruises','odor','gill-size','gill-spacing','gill-color','stalk-surface-above-ring','stalk-surface-below-ring',
'stalk-color-above-ring', 'stalk-color-below-ring','spore-print-color','population','habitat']
df_to_use = df[colunms_to_use].copy()
df_to_use_d = pd.get_dummies(df_to_use,drop_first=True)
print(df_to_use_d.shape)
df_to_use_d.head()


# In[ ]:


y = pd.get_dummies(df['class'],drop_first=True)
X = df_to_use_d

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'n_estimators': [50],
    'max_depth': [2, 5]
}

# Instantiate the regressor: gbm
gbm = xgb.XGBClassifier()

# Perform grid search: grid_mse
grid = GridSearchCV(estimator = gbm , param_grid = gbm_param_grid,
                        scoring = 'recall',
                        cv = 4,
                        verbose=1
                        )



grid.fit(X_train,y_train)
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
precision = precision_score(y_pred, y_test)
roc_auc = roc_auc_score(y_pred, y_test)
confusion = confusion_matrix(y_pred, y_test)
f1 = f1_score(y_pred, y_test)
recall = recall_score(y_pred, y_test)

print(best_model)

print('Accuracy achieved: %0.3f'%accuracy)
print('Precision achieved: %0.3f'%precision)
print('Roc Auc achieved: %0.3f'%roc_auc)
print('F1 Score achieved: %0.3f'%f1)
print('Recall achieved: %0.3f'%recall)

importances = pd.Series(data=best_model.feature_importances_,index= X_train.columns)
importances_sorted = importances.sort_values().tail(10)
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()


print('Confusion matrix achieved:')
confusion



# In[ ]:




