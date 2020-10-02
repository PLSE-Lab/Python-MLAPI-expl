#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier



from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import KernelCenterer

from sklearn import svm


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
submition = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


input_label = np.array(data.get('TARGET_5Yrs'))

data = data.drop(['PlayerID','Name','TARGET_5Yrs'] ,axis=1)
data = data.fillna(data.mean())

test = test.drop(['PlayerID','Name'] ,axis=1)
test = test.fillna(test.mean())


# In[ ]:


# standizer = StandardScaler()

# data[np.array(data.columns[:])] = standizer.fit_transform(data[np.array(data.columns[:])])
# test[np.array(test.columns[:])] = standizer.transform(test[np.array(test.columns[:])])


# In[ ]:


quantile = QuantileTransformer(n_quantiles=3000)
data[np.array(data.columns[:])] = quantile.fit_transform(data[np.array(data.columns[:])])
test[np.array(test.columns[:])] = quantile.transform(test[np.array(test.columns[:])])


# In[ ]:


# rbs = RobustScaler()
    
# data[np.array(data.columns[:])] = rbs.fit_transform(data[np.array(data.columns[:])])
# test[np.array(test.columns[:])] = rbs.fit_transform(test[np.array(test.columns[:])])


# In[ ]:


# pca = PCA()

# data[np.array(data.columns[:])] = pca.fit_transform(data[np.array(data.columns[:])])
# test[np.array(test.columns[:])] = pca.fit_transform(test[np.array(test.columns[:])])


# In[ ]:


# krc = KernelCenterer()

# data[np.array(data.columns[:])] = krc.fit_transform(data[np.array(data.columns[:])])
# test[np.array(test.columns[:])] = krc.transform(test[np.array(test.columns[:])])


# In[ ]:


# kf = KFold(n_splits=5,shuffle=True)

# random_forest_acc = 0
# adaboost_acc = 0
# extraRandom_acc = 0
# svm_acc = 0
# gradientBoosting_acc = 0 

# for train_index, test_index in kf.split(data):

#     X_train = data.filter(items=train_index, axis=0)
#     X_test = data.filter(items=test_index, axis=0)
    
#     y_train = input_label[train_index]
#     y_test = input_label[test_index]

# #   for randomForest  
#     random_forest_clf = RandomForestClassifier(n_estimators=50)
#     random_forest_clf.fit(X_train, y_train)
#     rand_given_labels = random_forest_clf.predict(X_test)
#     random_forest_acc += accuracy_score(y_test, rand_given_labels)
    
# #   for AdaBoost  
#     adaboost_clf = AdaBoostClassifier(n_estimators = 100,learning_rate=0.5)
#     adaboost_clf.fit(X_train, y_train)
#     ada_given_labels = adaboost_clf.predict(X_test)
#     adaboost_acc += accuracy_score(y_test, ada_given_labels)
# #   for extra random forest  
#     extraRandom= ExtraTreesClassifier(n_estimators=100, max_depth=None,min_samples_split=2)
#     extraRandom.fit(X_train, y_train)
#     xrand_given_labels = extraRandom.predict(X_test)
#     extraRandom_acc += accuracy_score(y_test, xrand_given_labels)    
# #   for gradient boosting
#     gradientBoosting_clf = GradientBoostingClassifier(n_estimators=350, learning_rate=.1,max_depth=1)
#     gradientBoosting_clf.fit(X_train, y_train)
#     gradientBoosting_given_labels = gradientBoosting_clf.predict(X_test)
#     gradientBoosting_acc += accuracy_score(y_test, gradientBoosting_given_labels)    
# #   for svm
#     svm_clf = svm.SVC(C= 0.1 , kernel='linear')
#     svm_clf.fit(X_train, y_train)
#     svm_given_labels = svm_clf.predict(X_test)
#     svm_acc += accuracy_score(y_test, svm_given_labels)    
    


# In[ ]:


print('AdaBoost : {}'.format(adaboost_acc/5))
print('RandomForest : {}'.format(random_forest_acc/5))
print('ExtraRandomForest: {}'.format(extraRandom_acc/5))
print('SVM : {}'.format(svm_acc/5))
print('GradientBoostingClassifier : {}'.format(gradientBoosting_acc/5))


# ## GradientBoostingClassifier used for final prediction

# In[ ]:


gradientBoosting_clf = GradientBoostingClassifier(n_estimators=350, learning_rate=.1,max_depth=1)
gradientBoosting_clf.fit(data,input_label)
gradientBoosting_given_labels = gradientBoosting_clf.predict(final)


# In[ ]:


gradientBoosting_given_labels.reshape((1,440))


# In[ ]:


submition.iloc[:,1] = gradientBoosting_given_labels


# In[ ]:


submition.to_csv("submission_6.csv", index=False)


# In[ ]:


print(submition)


# In[ ]:




