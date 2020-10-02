#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

 # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:




# Package imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import ExtraTreesClassifier

#from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,make_scorer

#import seaborn as sns
import matplotlib.pyplot as plt

#from sklearn.ensemble import RandomForestClassifier


# In[ ]:


train = pd.read_csv('/kaggle/input/learn-together/train.csv.zip')
test = pd.read_csv('/kaggle/input/learn-together/test.csv.zip')
submission = pd.read_csv('/kaggle/input/learn-together/sample_submission.csv')


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


all_df = [train,test]

for df in all_df:
    print(df.isna().sum())
    print(40*'-')
    


# There is no empty value in the dataset, that means dataset is allready preprocessed and curated for us. Lucky us!

# In[ ]:


train.head()


# In[ ]:


features = list(train.columns)
target = 'Cover_Type'                                              


# In[ ]:


features.remove(target)
print(features, target,type(train.columns),type(features))


# # Using train_test_split with just one validation set

# In[ ]:


from sklearn.model_selection import train_test_split

X = train[features]
y = train[target]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, random_state=200)


# In[ ]:


X_valid.describe()


# In[ ]:


np.unique(y_valid,return_counts = True)


# In[ ]:


print("X_train",X_train.shape)
print("X_valid",X_valid.shape)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state = 100)
rfc.fit(X_train,y_train)
print(type(rfc.feature_importances_))


# In[ ]:


plt.subplots(figsize=(18,5))
plt.bar(range(len(features)),rfc.feature_importances_,width=1)
plt.xticks(range(len(features)),features,rotation = 90,fontsize=9)
plt.show()


# ## After removing redundant features

# In[ ]:


new_features =[]
for i in range(len(rfc.feature_importances_)):
    if rfc.feature_importances_[i]>0.005:
        new_features.append(features[i])
print(len(new_features),new_features)


# Accuracy score based on validation set alone:

# In[ ]:


from sklearn.metrics import accuracy_score

rfc = RandomForestClassifier(oob_score=True,random_state=1,n_estimators=50)
rfc.fit(X_train[new_features],y_train)
prediction = rfc.predict(X_valid[new_features])
print(accuracy_score(y_valid,prediction))
print(rfc2.oob_score_)


# ## Scores using cross validation set

# In[ ]:


from sklearn.model_selection import cross_val_score

rfc3 = RandomForestClassifier(oob_score=True,random_state=1,n_estimators=50)
scores = cross_val_score(rfc3,train[new_features],train[target], cv=8)
print(scores.mean())


# In[ ]:


rfc3 = RandomForestClassifier(oob_score=True,random_state=1,n_estimators=50)
scores = cross_val_score(rfc3,train[features],train[target], cv=8)
print(scores.mean())
print(scores)


# Comparing the scores above, we see that score from using all features is greater than removing redundant feaatures.

# **Hyperparameter tuning**

# In[ ]:


print(rfc.get_params)


# We can use RandomizedSearchCV for hyperparameter tuning

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

#Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[ ]:


rfc = RandomForestClassifier()
rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 4, verbose=2, random_state=10, n_jobs = -1)
rfc_random.fit(X_train[features], y_train)


# In[ ]:


best_params = rfc_random.best_params_
print(best_params)


# In[ ]:





# ### Final submissions

# In[ ]:


predictions_test = rfc3.predict(test[new_features])

test_id = submission['Id']
submission = pd.DataFrame({
    'Id': test_id,
    'Cover_Type': predictions_test
})
submission.to_csv("submission1.csv",index=False)

