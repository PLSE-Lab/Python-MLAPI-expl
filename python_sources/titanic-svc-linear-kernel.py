#!/usr/bin/env python
# coding: utf-8

# # Titanic Dataset Kernel
# @ Ashish Gupta
# 31st September 2018

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/gender_submission.csv")


# In[ ]:


print(train.shape,test.shape)


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# # Insights into the data
# 
# PassengerID is not useful
# Name is not useful(could have told mr to miss, but that is available)
# SibSp is important but not that much
# parch is also important since kids with rich parents must have boarded first
# Tickets, not so much (None)
# Fares is also good indication but might be correlated to pClass
# 
# 
# Cabin needs to be studied via frequency distribution
# 
# 
# Embarked need to be one hot encoded, but first the missing values need to be replaced with most frequent category
# 

# In[ ]:


trainData = train.drop(["PassengerId","Name","Age","Ticket","Cabin"],axis=1,inplace = False)
testData = test.drop(["PassengerId","Name","Age","Ticket","Cabin"],axis=1,inplace = False)


# In[ ]:


trainData.describe(), trainData.shape, testData.shape


# In[ ]:


trainData.fillna(axis=1,inplace = False,method = 'pad')
testData= testData.fillna(axis=1,inplace = False ,method = 'pad')


# In[ ]:


ytrain=trainData.Survived.values
xtrain=trainData.drop('Survived', axis=1).values
xtest = testData.values


# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


xtrain.shape, ytrain.shape,xtest.shape


# In[ ]:


xtest


# In[ ]:


for i in range(len(xtrain)):
    if xtrain[:,1][i]=='male':
        xtrain[:,1][i]=1
    else:
        xtrain[:,1][i]=0
        
for i in range(len(xtest)):
    if xtest[:,1][i]=='male':
        xtest[:,1][i]=1
    else:
        xtest[:,1][i]=0


# In[ ]:


xtest


# In[ ]:


fare = 4
fareMean=xtrain[:,fare].mean()
fareStd=xtrain[:,fare].std()

for i in range(len(xtrain)):
    xtrain[:,fare][i]= (xtrain[:,fare][i] - fareMean)/ (fareStd*1.0)


fare = 4
fareMeanTest=xtest[:,fare].mean()
fareStdTest=xtest[:,fare].std()

for i in range(len(xtest)):
    xtest[:,fare][i]= (xtest[:,fare][i] - fareMean)/ (fareStd*1.0) # scales it according to the train data

xtrain,xtest
      
      


# In[ ]:


dummies = pd.get_dummies(train.Embarked).values
dummiesTest = pd.get_dummies(test.Embarked).values


# In[ ]:


dummies = dummies[:,:-1]
dummiesTest = dummiesTest[:,:-1]


# In[ ]:


dummies.shape, dummiesTest.shape


# In[ ]:


xtrain_onehot= xtrain[:,:-1]
xtest_onehot = xtest[:,:-1]


# In[ ]:


xtrain_final = np.append(xtrain_onehot,dummies,axis=1)
xtest_final = np.append(xtest_onehot, dummiesTest,axis=1)


# In[ ]:


xtrain_final.shape,ytrain.shape,xtest_final.shape


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


clf = SVC(kernel = "rbf",gamma='scale')


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# xtrain1,xtest1,ytrain1,ytest1=train_test_split(xtrain_final,ytrain,random_state=5)
# clf1 = SVC(kernel = "rbf",gamma='scale')
# clf1.fit(xtrain1,ytrain1)
# clf1.score(xtest1,ytest1)


# #The above is a test script for accuracy, I got an accuracy of 82%, 
# #and test accuracy of 78%, hence it is clearly not over fitting


# In[ ]:


clf.fit(xtrain_final,ytrain)


# In[ ]:


ypred = clf.predict(xtest_final)


# In[ ]:


ypredDF = pd.DataFrame(ypred)


# In[ ]:


ypredDF.to_csv("submission.csv",sep=",",index_label=['PassengerId'],header=['Survived'],index=True)


# In[ ]:


temp = pd.read_csv('submission.csv')
temp.PassengerId = temp.PassengerId + 892


# In[ ]:


# submission
temp.to_csv("submission.csv",sep=",",header=['PassengerId','Survived'],index=False)


# In[ ]:




