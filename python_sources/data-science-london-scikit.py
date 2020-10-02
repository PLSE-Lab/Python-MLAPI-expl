#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##Import packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


#Import data
import pandas as pd
test = pd.read_csv("../input/data-science-london-scikit-learn/test.csv", header=None)
train = pd.read_csv("../input/data-science-london-scikit-learn/train.csv", header=None)
trainLabels = pd.read_csv("../input/data-science-london-scikit-learn/trainLabels.csv", header=None)
print(plt.style.available) # look at available plot styles
plt.style.use('ggplot')


# In[ ]:


print('train shape:', train.shape)
print('test shape:', test.shape)
trainLabels = np.ravel(trainLabels)
print(trainLabels.shape)
train.head()


# In[ ]:


train.head(10)


# In[ ]:


train.info()


# In[ ]:


#find and replace missing value
train.isna().sum(axis=0)


# In[ ]:


train.describe()


# Split the data in to train and test

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(train, trainLabels, test_size = 0.25, random_state = 0)
print(X_train.shape)


# [> > Logistic Regression](http://)

# In[ ]:


log_reg = LogisticRegression(solver='lbfgs', penalty='none')#instatiate the model
log_reg.fit(X_train, Y_train)#fit the model with data
y_pred = log_reg.predict(X_test) #predict using test data
print("Training score: ", log_reg.score(X_train, Y_train))
print("Testting score: ", log_reg.score(X_test, Y_test))


# Decision tree
# 

# In[ ]:


dec_tree = DecisionTreeClassifier(max_depth=7,random_state=1)
dec_tree.fit(X_train,Y_train)
print("Training score: ", dec_tree.score(X_train, Y_train))
print("Testting score: ", dec_tree.score(X_test, Y_test))


# In[ ]:


##KNN 


# In[ ]:


n=np.arange(1,26)
kfold=10
train_accuracy=[]
crossval_accuracy=[]
bestk=0
bestacc = 0.0
for i,k in enumerate(n):
    #create a model for each k
    knn=KNeighborsClassifier(n_neighbors=k)
    #fit the training data into knn
    knn.fit(X_train,Y_train)
    #claculate the score for train data an appen to the list.(train accuracy)
    train_accuracy.append(knn.score(X_train,Y_train))
    #calculate the crossvalidation score of train data(test accuracy)
    #assign that to a variable
    mean_val_accuracy=np.mean(cross_val_score(knn,train,trainLabels,cv=kfold))
    #aapend it to a list
    crossval_accuracy.append(mean_val_accuracy)
    #compare the mean_val_acc and the current bast accuracy.
    if mean_val_accuracy > bestacc:
        bestk = k
        bestacc = mean_val_accuracy
        
print("best k value is: ", bestk)
print("best accuracy is: ", bestacc)


# In[ ]:


#Final Model


# In[ ]:


final_model = KNeighborsClassifier(n_neighbors=bestk)
final_model.fit(train,trainLabels)
print("Training final: ", final_model.score(train, trainLabels))


# In[ ]:


final_test = final_model.predict(test)
final_test.shape


# In[ ]:


submission = pd.DataFrame(final_test)
print(submission.shape)
submission.columns = ['Solution']
submission['Id'] = np.arange(1,submission.shape[0]+1)
submission = submission[['Id', 'Solution']]
submission


# In[ ]:


filename = 'Scikit-KNN.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:


print(check_output(["ls", "../working"]).decode("utf8"))

