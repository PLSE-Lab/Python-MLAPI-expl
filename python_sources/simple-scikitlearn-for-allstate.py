#!/usr/bin/env python
# coding: utf-8

# Just a simple linear regression with scikit.
# 
# ok, not so simple

# In[ ]:


import os,sys,time,random,math
import tarfile, zipfile

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn import decomposition, datasets, ensemble
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer,precision_score, recall_score, f1_score, average_precision_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
from IPython.display import display, Image

import xgboost as xgb


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


def loadData(datadir,filename):
    # Load the wholesale customers dataset
    #data = pd.read_csv(filename)
    data = ''
    print ("loading: "+datadir+filename)
    try:
        if zipfile.is_zipfile(datadir+filename):
            z = zipfile.ZipFile(datadir+filename)
            filename = z.open(filename[:-4])
        else:
            filename=datadir+filename
        data = pd.read_csv(filename, parse_dates=True)  
        print ("Dataset has {} samples with {} features each.".format(*data.shape))
    except Exception as e:
        print ("Dataset could not be loaded. Is the dataset missing?")
        print(e)
    return data

def writeData(data,filename):
    # Load the wholesale customers dataset
    try:
        data.to_csv(filename, index=False)
    except Exception as e:
        print ("Dataset could not be written.")
        print(e)
    verify=[]
    try:
        with open(filename, 'r') as f:
            for line in f:
                verify.append(line)
        f.closed
        return verify[:5]
    except IOError:
        sys.std


# In[ ]:


datadir="../input/"
data = loadData(datadir,'train.csv')
display(data.info())
display(data.head(5))


# In[ ]:


features = data.columns
cats = [feat for feat in features if 'cat' in feat]
for feat in cats:
    data[feat] = pd.factorize(data[feat], sort=True)[0]

display(data.info())
display(data.head())


# In[ ]:



x=data.drop(['id','loss'],1).fillna(value=0)
y=data['loss']

display(x.head(5))
display(y.head(5))


# In[ ]:


#  train/validation split
X_train, X_test, y_train, y_test = train_test_split( x.values, y.values, test_size=0.25, random_state=42)

dataSize=X_train.shape[0]
print ("size of train data",dataSize, )
test_sizes=[50]
for i in range(5):
    test_sizes.append(int(round(dataSize*(i+1)*.2)))

#test_sizes=[63,630,6300,31500]
#test_sizes=[50, 38108, 76217, 114325, 152434, 190542]
print ("run tests of size",test_sizes)


# In[ ]:


#regr = LinearRegression()

# regr = ensemble.AdaBoostRegressor()  ## The mean squared error

#regr = xgb.XGBClassifier(max_depth=6, learning_rate=0.075, n_estimators=15,
#                                objective="reg:linear", subsample=0.7,
#                                colsample_bytree=0.7, seed=42)
#regr = ExtraTreesRegressor()
regr = RandomForestRegressor()

#pca = decomposition.PCA(n_components = 100)
#regr = Pipeline(steps=[('pca', pca), ('classifier', regr )]) # set up the clf as a pipeline so we can do randomized PCA

regr.fit(X_train ,y_train )

#params=dict(fit_intercept=[True,False], normalize  = [True,False])
#grid_search = GridSearchCV(regr, param_grid= params, n_jobs= 1, scoring=make_scorer(f1_score)) 
#grid_search.fit(X_train,y_train)


# In[ ]:


# The mean squared error
print("Mean abs error: {:.2f}".format(np.mean(abs(regr.predict(X_test) - y_test))))
# Explained variance score: 1 is perfect prediction
print("Score: {:.2f}".format(regr.score(X_test, y_test)))


# In[ ]:


regr.fit(x,y)
# The mean squared error
print("Mean abs error: {:.2f}".format(np.mean(abs(regr.predict(X_test) - y_test))))
# Explained variance score: 1 is perfect prediction
print("Score: {:.2f}".format(regr.score(X_test, y_test)))


# In[ ]:


test_data= loadData(datadir,'test.csv')
display(test_data.info())
display(test_data.head(5))

kfeatures = test_data.columns
cats = [feat for feat in features if 'cat' in feat]
for feat in cats:
    test_data[feat] = pd.factorize(test_data[feat], sort=True)[0]

test_X=test_data.drop(['id'],1).fillna(value=0)
test_data['loss']=regr.predict(test_X)

display(test_data.info())
display(test_data.head())

result=test_data[['id','loss']]
display(result.info())
display(result.head())


# In[ ]:


output_fname="result_submission.csv"
writeData(result,output_fname)

