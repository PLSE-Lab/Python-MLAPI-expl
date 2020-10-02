#!/usr/bin/env python
# coding: utf-8

# In[109]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from fastai import *
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[110]:


train_file_path = "../input/train.csv"
valid_file_path = "../input/valid.csv"
exemplo_file_path = "../input/exemplo_resultado.csv"
test_file_path = "../input/test.csv"

test_data = pd.read_csv(test_file_path)
train_data = pd.read_csv(train_file_path)
valid_data = pd.read_csv(valid_file_path)
exemplo_data = pd.read_csv(exemplo_file_path)


# In[3]:


train_data.head()


# In[4]:


train_data.describe()


# In[62]:


max_size = train_data['default payment next month'].value_counts()
max_size


# In[ ]:


ax = plt.gca()

train_data.plot(kind='line',x='ID',y='PAY_AMT2',color='red', ax=ax)

plt.show()


# In[9]:


ax = plt.gca()

train_data.plot(kind='line',x='ID',y='PAY_6',color='red', ax=ax)

plt.show()


# In[ ]:


ax = plt.gca()

train_data.plot(kind='line',x='ID',y='BILL_AMT3',color='red', ax=ax)

plt.show()


# In[63]:


# train_data["EDUCATION"] = train_data.EDUCATION.mask(train_data.EDUCATION == 0,4)
# train_data["EDUCATION"] = train_data.EDUCATION.mask(train_data.EDUCATION == 5,4)
# train_data["EDUCATION"] = train_data.EDUCATION.mask(train_data.EDUCATION == 6,4)

# train_data["MARRIAGE"] = train_data.MARRIAGE.mask(train_data.MARRIAGE == 0,3)

# train_data["PAY_0"] = train_data.PAY_0.mask(train_data.PAY_0 == 0,-1)
# train_data["PAY_2"] = train_data.PAY_2.mask(train_data.PAY_2 == 0,-1)
# train_data["PAY_3"] = train_data.PAY_3.mask(train_data.PAY_3 == 0,-1)
# train_data["PAY_4"] = train_data.PAY_4.mask(train_data.PAY_4 == 0,-1)
# train_data["PAY_5"] = train_data.PAY_5.mask(train_data.PAY_5 == 0,-1)
# train_data["PAY_6"] = train_data.PAY_6.mask(train_data.PAY_6 == 0,-1)

# train_data["PAY_0"] = train_data.PAY_0.mask(train_data.PAY_0 == -2, 2)
# train_data["PAY_2"] = train_data.PAY_2.mask(train_data.PAY_2 == -2, 2)
# train_data["PAY_3"] = train_data.PAY_3.mask(train_data.PAY_3 == -2, 2)
# train_data["PAY_4"] = train_data.PAY_4.mask(train_data.PAY_4 == -2, 2)
# train_data["PAY_5"] = train_data.PAY_5.mask(train_data.PAY_5 == -2, 2)
# train_data["PAY_6"] = train_data.PAY_6.mask(train_data.PAY_6 == -2, 2)

# mean_pay_amt = train_data['PAY_AMT2'].mean(skipna=True)
# mean_bil_amt = train_data['BILL_AMT3'].mean(skipna=True)

# train_data["PAY_AMT2"] = train_data.PAY_AMT2.mask(train_data.PAY_AMT2 > 1000000, mean_pay_amt )
# train_data["BILL_AMT3"] = train_data.BILL_AMT3.mask(train_data.BILL_AMT3 > 1000000, mean_bil_amt )

# train_data.describe()


# In[64]:


# valid_data["EDUCATION"] = valid_data.EDUCATION.mask(valid_data.EDUCATION == 0,4)
# valid_data["EDUCATION"] = valid_data.EDUCATION.mask(valid_data.EDUCATION == 5,4)
# valid_data["EDUCATION"] = valid_data.EDUCATION.mask(valid_data.EDUCATION == 6,4)
# valid_data["MARRIAGE"] = valid_data.MARRIAGE.mask(valid_data.MARRIAGE == 0,3)
# valid_data["PAY_0"] = valid_data.PAY_0.mask(valid_data.PAY_0 == 0,-1)
# valid_data["PAY_2"] = valid_data.PAY_2.mask(valid_data.PAY_2 == 0,-1)
# valid_data["PAY_3"] = valid_data.PAY_3.mask(valid_data.PAY_3 == 0,-1)
# valid_data["PAY_4"] = valid_data.PAY_4.mask(valid_data.PAY_4 == 0,-1)
# valid_data["PAY_5"] = valid_data.PAY_5.mask(valid_data.PAY_5 == 0,-1)
# valid_data["PAY_6"] = valid_data.PAY_6.mask(valid_data.PAY_6 == 0,-1)
# valid_data["PAY_0"] = valid_data.PAY_0.mask(valid_data.PAY_0 == -2, 2)
# valid_data["PAY_2"] = valid_data.PAY_2.mask(valid_data.PAY_2 == -2, 2)
# valid_data["PAY_3"] = valid_data.PAY_3.mask(valid_data.PAY_3 == -2, 2)
# valid_data["PAY_4"] = valid_data.PAY_4.mask(valid_data.PAY_4 == -2, 2)
# valid_data["PAY_5"] = valid_data.PAY_5.mask(valid_data.PAY_5 == -2, 2)
# valid_data["PAY_6"] = valid_data.PAY_6.mask(valid_data.PAY_6 == -2, 2)

# valid_mean_pay_amt = valid_data['PAY_AMT2'].mean(skipna=True)
# valid_mean_bil_amt = valid_data['BILL_AMT3'].mean(skipna=True)

# valid_data["PAY_AMT2"] = valid_data.PAY_AMT2.mask(valid_data.PAY_AMT2 > 1000000, valid_mean_pay_amt )
# valid_data["BILL_AMT3"] = valid_data.BILL_AMT3.mask(valid_data.BILL_AMT3 > 1000000, valid_mean_bil_amt )

# valid_data.head()


# In[7]:


train_data.describe()


# In[156]:


y = train_data["default payment next month"]
y.head()


# In[ ]:





# In[157]:


x = train_data
x = x.drop(columns=["default payment next month"])

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(return_indices=True)
x, y, id_rus = rus.fit_sample(x, y)


# sm = SMOTE(random_state=12, ratio = 0.8)
# x, y = sm.fit_sample(x, y)


# In[158]:


np.sum(y==1)


# In[159]:


np.sum(y==0)


# In[55]:


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x = scaler.fit_transform(x)


# In[160]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(x, y)

print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(x, y)))
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(x_test, y[:4500])))


# In[161]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x,y)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(x, y)))
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(x_test, y[:4500])))


# In[162]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x, y)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(x, y)))
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(x_test, y[:4500])))


# In[163]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(x, y)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(x, y)))
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(x_test, y[:4500])))


# In[164]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x, y)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(x, y)))
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(x_test, y[:4500])))


# In[ ]:


# from sklearn.svm import SVC
# svm = SVC()
# svm.fit(x, y)
# print('Accuracy of SVM classifier on training set: {:.2f}'
#      .format(svm.score(x, y)))


# In[165]:


from sklearn.ensemble import RandomForestClassifier

# initialize
rfc = RandomForestClassifier(n_estimators=100, max_depth=3,max_features=3, min_samples_split=3, min_samples_leaf=3, random_state=12)
rfc.fit(x,y)
print('Accuracy of RFC classifier on training set: {:.2f}'
      .format(rfc.score(x, y)))
print('Accuracy of RFC classifier on training set: {:.2f}'
      .format(rfc.score(x_test, y[:4500])))


# In[75]:


# rf = RandomForestClassifier(n_jobs=-1)

# param_grid = {
#     'min_samples_split': [3, 5, 10], 
#     'n_estimators' : [100, 300],
#     'max_depth': [3, 5, 15, 25],
#     'max_features': [3, 5, 10, 20]
# }


# In[76]:


# from sklearn.model_selection import GridSearchCV

# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 3, n_jobs = -1, verbose = 2)


# In[77]:


# grid_search.fit(x, y)


# In[78]:


# grid_search.best_params_


# In[82]:


# def evaluate(model, test_features, test_labels):
#     predictions = model.predict(test_features)
#     errors = abs(predictions - test_labels)
#     mape = 100 * np.mean(errors / test_labels)
#     accuracy = 100 - mape
#     print('Model Performance')
#     print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
#     print('Accuracy = {:0.4f}%.'.format(accuracy))
    
#     return accuracy


# In[86]:


# best_grid = grid_search.best_estimator_
# grid_accuracy = evaluate(best_grid, x, y)


# In[167]:


x = pd.DataFrame(x)
x.columns = this.columns
x.head(2)


# In[168]:


this = pd.concat([valid_data, test_data], ignore_index=True)
predict = rfc.predict(this)
print(predict[:300]) 


# In[169]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y[:9000], predict)
cm


# In[170]:


rfc.feature_importances_


# In[ ]:


exemplo_data.head()


# In[ ]:


submission = exemplo_data
submission["Default"] = predict
submission.to_csv("submission.csv", index = False)
submission.describe()

