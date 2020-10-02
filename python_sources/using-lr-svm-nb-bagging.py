#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.manifold import TSNE

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve

np.random.seed(12334)
#read data 
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#display data
print (train.head())
print ()
print (test.head())


# **Replace object values :**

# In[ ]:


#replacing object type values 
def DoMap(df,df1):
    col = df.dtypes
    colList = []
    unique = []
    map = {}

    for i in col.index:
        if col[i] not in ['int','float']:
            colList.append(i)

    print ("col with object type - >",colList)

    # factorize the col with object type 
    for c in colList:
        list = df[c].unique()
        list1 = df1[c].unique()
        for l in list:
            if l not in unique:
                unique.append(l)
        for l in list1:
            if l not in unique:
                unique.append(l)
    print ('unique object -> ',unique)
    i = 0
    for u in unique:
        map[u] = i
        i += 1
    print ("replacing with -> ",map)
    df[colList] = df[colList].replace(map)    
    df1[colList] = df1[colList].replace(map)

    return df,df1
    
train,test = DoMap( train, test)


# In[ ]:


trainY = train['y']
trainX = train.drop('y', axis=1)
trainX = trainX.ix[:,1:]
trainX.head()


# **T-SNE** :

# In[ ]:


from sklearn.manifold import TSNE
tsne2 = TSNE(n_components=2)
tsne2_results = tsne2.fit_transform(trainX)

cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots(figsize=(20,15))
points = ax.scatter(tsne2_results[:,0], tsne2_results[:,1], c=trainY, s=50, cmap=cmap)
f.colorbar(points)
plt.show()
    


# In[ ]:


test_id = test['ID'].values
# Create classifiers
lr = LinearRegression()
gnb = GaussianNB()
svc = LinearSVC(C=1.0)


# In[ ]:


for clf, name in [(lr, 'Linear Regression'),
                  (gnb, 'Naive Bayes'),
                  (svc, 'Support Vector Classification')]:
    
    clf.fit(trainX, trainY.astype(int))
    
    y = clf.predict(test.ix[:,1:])
    
    sub = pd.DataFrame()
    sub['ID'] = test_id
    sub['y'] = y
    sub.to_csv(name+'.csv', index=False)


# In[ ]:


sub.head()

