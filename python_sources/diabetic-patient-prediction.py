#!/usr/bin/env python
# coding: utf-8

# DIABETIC PATIENT PREDICTION AND CLASSIFICATION

# In[ ]:


import pandas as pd
dia=pd.read_csv("/kaggle/input/pima-indians-diabetes-dataset/diabetes.csv")
dia


# EXTRACTING LABELS AND FEATURES

# In[ ]:


X=dia.drop(columns=["Outcome"],axis=1).values
Y=dia[["Outcome"]].values


# NORMALISING THE FEATURES

# In[ ]:


X=X/X.max()


# SPLITTING THE TRAINING AND TESTING DATASETS

# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.4,random_state=58)


# In[ ]:


(ytest==1).sum()


# In[ ]:


(ytest==0).sum()


# GAUSSIAN NAIVE_BAYES ALGORITHM

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nmodel=GaussianNB()


# In[ ]:


modelpre=nmodel.fit(xtrain,ytrain)


# ACCURACY

# In[ ]:


#print(nmodel.score(xtrain,ytrain))
print(nmodel.score(xtest,ytest))


# In[ ]:


ytrain_prod=modelpre.predict(xtrain)
ytest_prod=modelpre.predict(xtest)


# ANALYSING THE PREDICTIONS 

# In[ ]:


modelpre.predict([[0,137,40.0,35,168.0,43.1,2.288,33]])


# In[ ]:


modelpre.predict([[10,101,76.0,48,180.0,32.9,0.171,63]])


# ANLAYZING THE CORRELATION BETWEEN FEATURES AND LABEL

# In[ ]:


d=dia[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]]
d.corr()


# In[ ]:




