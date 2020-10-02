#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('../input/Iris.csv')


# **Starting View of data**<br>
# Here we can see that apart from id all other four attributes are considered  for prediction
# the predicting categories are three in Species column

# In[ ]:


df.head()


# **Ending view View of data**<br>
# it seems that data is seperable to a large extent****

# In[ ]:


df.tail()


# dataset is much clean and 'id' column is non significant  

# In[ ]:


df.info()


# In[ ]:


df.set_index('Id',inplace=True)


# In[ ]:


df.head(1)


# - Petal Length  is most varied as compared to other attributes
# - Petal length and Petal Width  quartile deviation is high
# - All attributes have different min and max values which can seperate the data set for prediction
# 

# In[ ]:


df.describe()


# **Coefficent of Quartile deviation**

# In[ ]:


qd=pd.DataFrame((df.describe().loc['75%']-df.describe().loc['25%'])/(df.describe().loc['75%']+df.describe().loc['25%']),columns=['cofficient of quartile deviation'])
qd


# # Scatter Plots for comparing the attributes

# * Scatter plots for comparing different attributes and finding the seperation area(estimating)
# * Iris-setosa is linearly seperable but other two are overlapped

# In[ ]:


sns.pairplot(df,hue='Species',size=2.6)


# In[ ]:


sns.lmplot(data=df,x='SepalLengthCm',y='SepalWidthCm',hue='Species',size=10,fit_reg=False )


# In[ ]:


sns.lmplot(data=df,x='PetalLengthCm',y='PetalWidthCm',hue='Species',size=10,fit_reg=False ,logistic=True)


# # Comparing different Flowers and there Seperation area

# In[ ]:


fig,axis=plt.subplots(nrows=2,ncols=2,figsize=(18,9))
sns.stripplot(y='Species',x='PetalWidthCm',data=df,ax=axis[0,1])
sns.stripplot(y='Species',x='PetalLengthCm',data=df,ax=axis[1,1])
sns.stripplot(y='Species',x='SepalWidthCm',data=df,ax=axis[0,0])
sns.stripplot(y='Species',x='SepalLengthCm',data=df,ax=axis[1,0])
plt.show()


# In[ ]:


fig,ax=plt.subplots(nrows=1 ,ncols=4)
sns.distplot(df['SepalLengthCm'],ax=ax[0])
sns.distplot(df['SepalWidthCm'],ax=ax[1])
sns.distplot(df['PetalLengthCm'],ax=ax[2])
sns.distplot(df['PetalWidthCm'],ax=ax[3])
fig.set_figwidth(30)


# In[ ]:


fig,ax=plt.subplots(nrows=1 ,ncols=4)

sns.boxplot(data=df,y='SepalLengthCm',x='Species',ax=ax[0])
sns.boxplot(data=df,y='SepalWidthCm',x='Species',ax=ax[1])
sns.boxplot(data=df,y='PetalLengthCm',x='Species',ax=ax[2])
sns.boxplot(data=df,y='PetalWidthCm',x='Species',ax=ax[3])
fig.set_figwidth(30)
fig.set_figheight(10)


# In[ ]:


fig=plt.figure(figsize=(20,7))
df.iloc[:,0].plot(label='Sepal Length')
df.iloc[:,1].plot(label='Sepal Width')
df.iloc[:,2].plot(label='Petal Length')
df.iloc[:,3].plot(label='Petal Width')
leg=plt.legend()
plt.show()


# # corelation between Attributes 

# In[ ]:



plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True)
plt.title('correlation matrix')
plt.show()


# # Feature Engineering
# - converting flowers in  encooded format
# 

# In[ ]:



x=df.iloc[:,:-1].values


# **Encoding categorical varibles**

# In[ ]:



pre=df.iloc[:,-1]
y=pre.replace(pre.unique(),np.arange(3))


# # Preprocessing

# In[ ]:


from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scx=StandardScaler()
xtrain=scx.fit_transform(xtrain)
xtest=scx.transform(xtest)


# # Modelling using Support Vector Classification

# In[ ]:


from sklearn.svm import  SVC
classifier=SVC(C=51,degree=1,gamma=10,kernel='poly')
classifier.fit(xtrain,ytrain)


# **[](http://)Metrics of accuracy**

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(ytest,classifier.predict(xtest)))


# **cross validation of test set**

# In[ ]:


from sklearn.model_selection import cross_val_score
ca=cross_val_score(classifier,xtrain,ytrain,scoring='accuracy',cv=10)


# In[ ]:


ca


# In[ ]:


print(str(ca.mean()*100)+'% accuracy')#accuray after 10 cross validations


# In[ ]:


ca.std()*100#4%variance/bias on the set hence accuracy can be varied 4% in general


# # Hyper paramater Optimization 
# -> using grid search

# In[ ]:


from sklearn.model_selection import GridSearchCV
params=[
    {
        'C':[51,0.1,100,1,10,80],'kernel':['rbf'],'gamma':[1,0.1,0.001,10,0.0001,50,100]
    },
    {
        'C':[51,0.1,100,1,10,80],'kernel':['poly'],'degree':[1,2,3,4],'gamma':[1,0.1,0.001,10,0.0001,50,100]
    },
    {
        'C':[51,0.1,100,1,10,80],'kernel':['sigmoid'],'gamma':[1,0.1,0.001,10,0.0001,50,100]
    },
     {
        'C':[51,0.1,100,1,10,80],'kernel':['linear'],'gamma':[1,0.1,0.001,10,0.0001,50,100]
    }


]


# In[ ]:


gc=GridSearchCV(classifier,param_grid=params,cv=10,scoring='accuracy')


# In[ ]:


gc.fit(xtrain,ytrain)


# **best hyper parameter**

# In[ ]:


gc.best_params_


# ** accuracy **

# In[ ]:


gc.best_score_


# In[ ]:


ypred=classifier.predict(xtest)


# **predicted test set ***

# In[ ]:


unique=df['Species'].unique()
u=pd.Series(ypred,name='flowers').apply(lambda x:unique[x])
u.head(10)


# ### True show the value for accurate  predition

# In[ ]:


ytest.values==ypred


# # Confusion Matrix

# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(pd.crosstab(ypred,ytest),annot=True,cmap='coolwarm')


# In[ ]:




