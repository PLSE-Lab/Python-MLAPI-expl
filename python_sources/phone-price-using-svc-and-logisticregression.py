#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , GridSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[ ]:



testdf = pd.read_csv("../input/mobile-price-classification/test.csv")
traindf = pd.read_csv("../input/mobile-price-classification/train.csv")


# In[ ]:


#show subset of the data
traindf.head()


# In[ ]:


#check if ther is a nulls in the traindf or not
traindf.info()


# In[ ]:


#train data description
traindf.describe()


# In[ ]:


#check the nulls in testdf
testdf.info()


# In[ ]:


#describe test data
testdf.describe()


# In[ ]:


#show the correlation between all fetures with each other
corrmat = traindf.corr()
f , ax = plt.subplots(figsize = (12 , 9))
sns.heatmap(corrmat , vmax=0.8 , square=True )


# In[ ]:


#finding the most 10 correlated features with price_range
k = 10
cols = corrmat.nlargest(k , 'price_range')['price_range'].index
cm = np.corrcoef(traindf[cols].values.T)
f , ax = plt.subplots(figsize = (10 , 7))
sns.heatmap(cm , annot=True , cbar=True , square=True , fmt='.2f', yticklabels=cols.values , xticklabels=cols.values)


# In[ ]:


#the selected most affected features is
cols = ['price_range','ram','battery_power' , 'px_width' , 'px_height']
sns.pairplot(traindf[cols] , height = 2.5)


# In[ ]:


traindf = traindf.drop(['blue','clock_speed','dual_sim','fc','four_g','int_memory','m_dep','mobile_wt','n_cores','pc','sc_h'
                       ,'sc_w' , 'talk_time','three_g','touch_screen','wifi'] , axis = 1)


# In[ ]:


y = traindf['price_range']
traindf = traindf.drop(['price_range'] , axis = 1)
std = StandardScaler()
X = std.fit_transform(traindf)
X = pd.DataFrame(X)
X.head()


# In[ ]:


X.info()


# In[ ]:


#do the same on the test data
testdf = testdf.drop(['id','blue','clock_speed','dual_sim','fc','four_g','int_memory','m_dep','mobile_wt','n_cores','pc','sc_h'
                       ,'sc_w' , 'talk_time','three_g','touch_screen','wifi'] , axis = 1)
std = StandardScaler()
testdf = std.fit_transform(testdf)
testdf = pd.DataFrame(testdf)
testdf.head()


# In[ ]:


#splitting the Data
Xtrain , Xtest , ytrain ,ytest = train_test_split(X , y ,test_size = 0.3 , random_state = 25 , shuffle = True)
Xtrain.shape , Xtest.shape


# In[ ]:


#train the model with logistic regression
for c in range(1 , 10):
    logreg = LogisticRegression(C = c , fit_intercept=True , max_iter = 130 ,solver ='lbfgs')
    logreg.fit(Xtrain , ytrain)
    print("C = ", c)
    print("Train score = " , logreg.score(Xtrain , ytrain))
    print("Test score  = " ,logreg.score(Xtest , ytest))
    print("-"*100)


# In[ ]:


#training the model with SVC
svcmodel = SVC(gamma='auto_deprecated')
parameters = {'kernel':('linear', 'rbf'), 'C':[1,2,3,4,5]}


GridSearchModel = GridSearchCV(svcmodel , parameters, cv = 3 ,return_train_score=True)
GridSearchModel.fit(Xtrain, ytrain)
sorted(GridSearchModel.cv_results_.keys())
GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' ,
                                                               'rank_test_score' , 'mean_fit_time']]


# In[ ]:


print('All Results are :\n', GridSearchResults )
print('-'*100)
print('Best Score is :', GridSearchModel.best_score_)
print('-'*100)
print('Best Parameters are :', GridSearchModel.best_params_)
print('-'*100)
print('Best Estimator is :', GridSearchModel.best_estimator_)


# In[ ]:


#predict test result using SVC
svcmodel = SVC(C=2, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='linear', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)

svcmodel.fit(Xtrain , ytrain)
svcmodel.score(Xtrain , ytrain)


# In[ ]:


#predict test data
predictions = svcmodel.predict(testdf)
predictions


# In[ ]:




