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

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# - The datasets contains transactions made by credit cards in September 2013 by european cardholders.
# - This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. 
# 0.172%  transations were frad

# In[ ]:


df=pd.read_csv("../input/creditcard.csv")


# FEATURES
# - There are two continous variables which are important and are disclosed
# - Amount- the capital currency which transfered during transation
# - Time -contains the seconds elapsed between each transaction and the first transaction in the dataset
# 

# In[ ]:


df.head()


# In[ ]:


df.columns


# Dataset seems to be clean
# 
# - Here class attribute signifies the dependent feature
# - If  class =0 -> not fraud transation
# - If  class =1 ->fraud transation

# In[ ]:


df.info()


# Insights about prediction class

# In[ ]:


total_predictor_entries=df.shape[0]
non_fraud_class=sum(df['Class']==0)
fraud_class=sum(df['Class']==1)

print('the percentage of data of non fraud entries are - '+str(  ( non_fraud_class/total_predictor_entries)*100 )+'%' )
print('the percentage of data of fraud entries are - '+str(  ( fraud_class/total_predictor_entries)*100 )+'%' )


# Here we can see that data set is very unstable as very few entries are  in fraud class
# 

# In[ ]:


sns.countplot(x='Class',data=df)


# Here Amount can be a better predictor because prediction classes are distributed variately with amount

# The amount column is left skewed and hence variation comes from this data

# In[ ]:


data=df[['Time','Amount','Class']]
sns.kdeplot(data['Amount'])


# We can see that  to be a fraud transation  lower values of amount are  taken
# There is less variation in fraud compared to non fraud

# In[ ]:


plt.figure(figsize=(12,7))

sns.stripplot(x='Class',y='Amount',data=df,jitter=True,palette='viridis')


# We can see that Time gap  between 70000 and 100000
# - non fraud has a high gap in time for given interval 

# In[ ]:


sns.violinplot(x='Class',y='Time',data=df)


# ## Feature  Extraction

# In[ ]:


df['hour']=(df['Time']//3600.).astype('int')


# In[ ]:


df['second']=(df['Time']%3600).astype('int')


# In[ ]:


df.drop('Time',axis=1,inplace=True)


# # corelation matrix
# - we can see that features have negative corelation between   with continous variables 
# - V2 have serious  negative  corelation with amount

# In[ ]:


plt.figure(figsize=(15,7))
sns.heatmap(df.corr(),cmap='coolwarm')


# # Feature Engineering 

# In[ ]:


x=df.drop('Class',axis=1).values
y=df['Class'].values


# Converting data into same scale to reduce computational delay and enhance distances in algorithms

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)


# ##  **Modelling and selection

# In[ ]:


from sklearn.model_selection import  train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)


# In[ ]:


from imblearn.over_sampling import SMOTE
sm=SMOTE()
xtrain,ytrain=sm.fit_resample(xtrain,ytrain)


# ## Classifiers-
# - we will use naive bayes classifier,random forest,Logistic regression

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
regressor=GaussianNB()
regressor2=LogisticRegression(solver='newton-cg')
regressor3=RandomForestClassifier(n_estimators=20)


# In[ ]:


regressor.fit(xtrain,ytrain)
regressor2.fit(xtrain,ytrain)
regressor3.fit(xtrain,ytrain)
print("*************")


# ### Pre Evaluation

# In[ ]:


from sklearn.metrics import classification_report
print('Gaussian Naive bayes Classifier accuracy')
print(classification_report(ytest,regressor.predict(xtest)))
print('--------------------')
print('Logistic Classifier accuracy ')
print(classification_report(ytest,regressor2.predict(xtest)))
print('--------------------')
print('Random Forest Classifier accuracy')
print(classification_report(ytest,regressor3.predict(xtest)))


# In[ ]:


print('Naive bayes')
print(pd.crosstab(ytest,regressor.predict(xtest)))
print('_____________________')
print('Logistic regression')
print(pd.crosstab(ytest,regressor2.predict(xtest)))
print('_____________________')
print("Random Forest")
print(pd.crosstab(ytest,regressor3.predict(xtest)))


# ## Ensemble learning
# 
# -we will take mode of predicton of each model
# 

# Integrating Three models predictions

# In[ ]:


ess=pd.DataFrame([regressor.predict(xtest),regressor2.predict(xtest),regressor3.predict(xtest)])


# Taking mode to gerneralise the prediction

# In[ ]:


out=ess.apply(lambda x:x.mode())
es_test=out.transpose()[0].values


# Accuracy report and Confusion Matrix  of Final Model(essembled version)

# In[ ]:


print(classification_report(ytest,es_test))


# In[ ]:


pd.crosstab(ytest,es_test)


# ## R.O.C curve

# In[ ]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
actual=ytest
predictions1=regressor2.predict_proba(xtest)[:,1] #logistic
predictions2=regressor.predict_proba(xtest)[:,1]#nb
predictions3=regressor3.predict_proba(xtest)[:,1]#rf
predictions4=es_test




fpr1,tpr1,t1=roc_curve(actual,predictions1)
fpr2,tpr2,t2=roc_curve(actual,predictions2)
fpr3,tpr3,t3=roc_curve(actual,predictions3)
fpr4,tpr4,t4=roc_curve(actual,predictions4)


auc1 = roc_auc_score(ytest,predictions1)
auc2 = roc_auc_score(ytest,predictions2)
auc3 = roc_auc_score(ytest,predictions3)
auc4 = roc_auc_score(ytest,predictions4)


# In[ ]:


fig=plt.figure(figsize=(15,10))
fig.legend()
plt.plot(fpr1,tpr1,label='logistic regression- auc-area'+str(round(auc1*100,3))+'%',color='red',lw=6)
plt.plot(fpr2,tpr2,label='naive bayes - auc-area'+str(round(auc2*100,3))+'%',color='blue',lw=6,ls=':')
plt.plot(fpr3,tpr3,label='random forest- auc-area'+str(round(auc3*100,3))+'%',color='green',lw=6,)
plt.plot(fpr4,tpr4,label='ensembled- auc-area'+str(round(auc4*100,3))+'%',color='brown',lw=6,ls='-')
plt.plot(np.arange(0,1,0.01),np.arange(0,1,0.01),label='random pick')
plt.legend()


# ## Hence we can see that logistic regression had most auc area hence it could refered as best model

# ### Applying kfold  cross validation to train  data with logistic regression

# In[ ]:


from sklearn.model_selection import cross_val_score
cv=cross_val_score(regressor2,xtrain,ytrain,cv=10,scoring='accuracy')


# Accuracy

# In[ ]:


cv.mean()


# variance

# In[ ]:


cv.std()


# In[ ]:


print('test set accuracy')
np.mean(regressor2.predict(xtest)==ytest)


# In[ ]:




