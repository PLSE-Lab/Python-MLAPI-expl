#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
# import category_encoders#NOT available in kaggle 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import time

if __name__ == '__main__':
    time1 = time.clock()
    sns.set(color_codes=True)
    url = "../input/adult-training.csv"
    urltest = "../input/adult-test.csv"
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'maritalstatus',
         'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss'
         , 'hours-per-week', 'nativecountry', 'salaryrange']
    cols = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'nativecountry', 'salaryrange']
    dataset = pd.read_csv(url, names=names)
    datasettest = pd.read_csv(urltest,names=names)
    print(datasettest.head(1),"\n\ntest dataset size before removing first column:",datasettest.shape)
    datasettest = datasettest[datasettest.fnlwgt.notnull()]
    print(datasettest.head(1),"\n\ntest dataset size after removing first column:",datasettest.shape)
    


# In[2]:


print(dataset.workclass.unique())
print("\nafter removing ' ?'")
dataset.replace(to_replace=" ?", value=np.NaN, inplace=True)
dataset.dropna(inplace=True)
datasettest.replace(to_replace=" ?", value=np.NaN, inplace=True)
datasettest.dropna(inplace=True)
print(dataset.workclass.unique())
print(dataset.salaryrange.unique())


# In[3]:


print(dataset.describe())


# In[4]:


print(dataset.head(1))
print(datasettest.head(1))
datasettest['salaryrange'].replace(to_replace=' >50K.',value=' >50K',inplace=True)
datasettest['salaryrange'].replace(to_replace=' <=50K.',value=' <=50K',inplace=True)
print(datasettest.head(1))


# In[5]:


#trying to converte strings to labels
for col in cols:
    enc = LabelEncoder()
    dataset[col] =(enc.fit_transform(dataset[col]))
    datasettest[col]=(enc.transform(datasettest[col]))
print(dataset.head(1))
print(datasettest.head(1))


# In[6]:


enc = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13],sparse=False)
datasetarr = enc.fit_transform(dataset)
datasettestarr = enc.transform(datasettest)
print(datasetarr[0,:],"\nshape:",datasetarr.shape)
print(datasettestarr[0,:],"\nshape:",datasettestarr.shape)
print(enc.get_params)


# In[7]:


# Split-out validation dataset
X = datasetarr[:,0:104]
Y = datasetarr[:,104]
print("Training features X[0:3]:",X[0:3,:])
print("Training result Y[0:3]",Y[0:3])


# In[8]:


X_test = datasettestarr[:,0:104]
Y_test = datasettestarr[:,104]
print("Test features X_test[0:3]:",X[0:3,:])
print("Test result Y_test[0:3]",Y[0:3])


# In[9]:


#trying to converte strings to labels
#enc = category_encoders.binary.BinaryEncoder(cols=cols)
#dataset = enc.fit_transform(dataset)
#datasettest = enc.fit_transform(datasettest)

# Split-out validation dataset
#selector = [x for x in range(dataset.shape[1])]
#selector.remove(27)
#array = dataset.values
#X = array[:,selector]
#Y = array[:,27]
#print(Y.dtype)

#selector1 = [x for x in range(datasettest.shape[1])]
#selector1.remove(27)
#array1 = datasettest.values
#X_test = array1[:,selector1]
#Y_test = array1[:,27]
#print(Y_test.dtype)
#Y_test = Y_test.astype('int64')


# In[10]:


# Test options and evaluation metric
#seed = 7
#scoring = 'accuracy'

#model = xgb.XGBClassifier()
#learning_rate = [0.1,0.15]
#max_depth = [3,4]
#n_estimators = [300,400]
#min_child_weight = [4,6]
#param_grid = dict(learning_rate=learning_rate ,n_estimators=n_estimators,max_depth=max_depth ,min_child_weight=min_child_weight )
#kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
#grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=kfold)
#grid_result = grid_search.fit(X, Y)

#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
	#    print("%f (%f) with: %r" % (mean, stdev, param))
#print("best score:",grid_result.best_score_,"\nlearning rate:",grid_result.best_estimator_.learning_rate,"\nestimator:",
#grid_result.best_estimator_.n_estimators,"maxdepth:", grid_result.best_estimator_.max_depth,
#      "\nmin_child_weight:",grid_result.best_estimator_.min_child_weight)


# In[11]:


#    model = xgb.XGBClassifier(learning_rate=grid_result.best_estimator_.learning_rate,
 #                             n_estimators=grid_result.best_estimator_.n_estimator,
 #                            max_depth=grid_result.best_estimator_.max_depth,
   #                           min_child_weight=grid_result.best_estimator_.min_child_weight
  #                            )
   # final_m = model.fit(X, Y)
    #xgb.plot_importance(final_m)
    #plt.show()


# In[12]:


#    predictions = model.predict(X)
 #   print("training set auc:", accuracy_score(Y, predictions))
  #  predictions = model.predict(X_test)
   # print("test set auc:", accuracy_score(Y_test, predictions))


# In[13]:


#    print(model.get_params())


# In[14]:


print("optimized by GridsearchCV when categories encoded using categoryencoder.BinaryEncoder:")
model = xgb.XGBClassifier(learning_rate=0.07,
                          n_estimators=500,
                          max_depth=5,
                          min_child_weight=4
                          )

final_m=model.fit(X,Y)
xgb.plot_importance(final_m)
plt.show()
predictions = model.predict(X)
print("training set auc:",accuracy_score(Y, predictions))
predictions = model.predict(X_test)
print("test set auc:",accuracy_score(Y_test, predictions))
print(model.get_params())


# In[15]:


print("time taken:",time.clock() - time1, "seconds")

