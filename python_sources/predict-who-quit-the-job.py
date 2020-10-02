#!/usr/bin/env python
# coding: utf-8

# This notebook contains Different ML models applying on data and its visualization.

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


Data=pd.read_csv("../input/HR_comma_sep.csv")
#Data=pd.read_csv("E:\Python_programming\HR analytics\HR_comma_sep.csv")
Data.head()


# In[ ]:


Data.dtypes


# In[ ]:


# converting object variables into category variables
olist=list(Data.select_dtypes(['object']))
for col in olist:
    Data[col]=Data[col].astype('category').cat.codes
Data.dtypes


# In[ ]:


# Split the Predictor and independent variables
X=Data.drop("left",axis=1)
#X=Data[['satisfaction_level',"last_evaluation"]]
Y=Data["left"].astype('category')


# In[ ]:


from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=3)


# In[ ]:


# Applying K-NN 
# import the model package
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
Y_pred=knn.predict(X_test)
Y_pred


# 

# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn import metrics
n_neighbors=30
h=0.02
# Create color maps
cmap_light=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold=ListedColormap(['#FF0000','#00FF00','#0000FF'])
# deal with only two features right now , for sack of visualization
X=Data[['number_project',"average_montly_hours"]]
Y=Data["left"]
for weights in ['uniform','distance']:
    # we are going to create the instance of Neighbours classifer and fit the 
    # data
    clf=neighbors.KNeighborsClassifier(n_neighbors,weights=weights)
    clf.fit(X, Y)
     # plot the decision boundary . For that, we will assign a color to each 
    # point in the mesh [x_min,x_max]x[y_min,y_max]
    x_min,x_max=X['number_project'].min()-1,X['number_project'].max()+1
    y_min,y_max=X["average_montly_hours"].min()-1,X["average_montly_hours"].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),
                     np.arange(y_min,y_max,h))
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    #print (len(yy.ravel()))
    #print (len(Z))
    #metrics.accuracy_score(yy.ravel(),Z)
    # Put the result into colorplot
    Z=Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
    
    # plot also training points
    plt.scatter(X["number_project"],X['average_montly_hours'],c=Y,cmap=cmap_bold)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.title("2-class classifiation(k=%i,weights='%s')"
             %(n_neighbors,weights))
plt.show()


# In[ ]:


from sklearn import metrics
metrics.accuracy_score(Y_test,Y_pred)


# In[ ]:


# why don't you try some differnt values of k ?
from sklearn.cross_validation import cross_val_score
k_range=list(range(1,15))
accuracy=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    acc=cross_val_score(knn,X,Y,cv=15,scoring='accuracy')
    accuracy.append(acc.mean())
print (accuracy)    


# In[ ]:


# plot the accuracies
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(k_range,accuracy)
plt.xlabel('value of k')
plt.ylabel('Cross validation accuracy')
plt.title('K vs. CV accuracy ')


# 

# In[ ]:


from sklearn import svm


# In[ ]:


Model=svm.SVC(kernel='rbf')
Model.fit(X_train,Y_train)
Y_pred=Model.predict(X_test)


# In[ ]:


from sklearn import metrics
metrics.accuracy_score(Y_test,Y_pred)

