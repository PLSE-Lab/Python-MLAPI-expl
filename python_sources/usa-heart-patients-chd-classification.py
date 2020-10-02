#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:





# In[ ]:


import pandas as pd
framingham = pd.read_csv("../input/heart-disease-prediction-using-logistic-regression/framingham.csv")


# In[ ]:


framingham.shape


# dataframe.shape is to check the rows and columns of the dataset, to know that we took the correct dataset

# In[ ]:


framingham.columns


# problem statement:
# Here the TenYearCHD (Ten year risk of coronary hear disease) is our prediction variable (Y) as Ten year cardio Heart disease, we need to predict in 10 years the patient gonna have the heart diesease, with his current medical records.
# 

# In[ ]:


framingham.head()


# In[ ]:


# Basic Packages:
import statsmodels.api as sm


# In[ ]:


framingham.corr()


# In[ ]:


framingham.info()


# There are some missing values.
# Check the number of records having null values.

# In[ ]:


framingham.isnull().sum()


# In[ ]:


df=framingham.dropna()
df.shape


# In[ ]:


na=framingham.shape[0]-df.shape[0]
na_percentage= (na/framingham.shape[0])*100
na_percentage


# If i go with dropna, totally I drop 13.732%.
# As a thumb rule, we can drop 10 to 15% records. so i dropped it

# In[ ]:


df.rename(columns={'male':'gender'},inplace=True)


# In[ ]:


df.corr()


# Strongly correlated features are: from this correlation matrix we cannot make the decision of correlation.
# 

# In[ ]:


df.head()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

pd.crosstab(df['gender'],df['TenYearCHD']).plot.bar(stacked=True)


# In[ ]:


sns.catplot(x='TenYearCHD',y='age',hue='gender',kind='box',data=df)


# In[ ]:


#Statistical Test:
# gender vs disease - 2 sample propotion test
# age vs disease - 2 sample t test
# education vs disaease - chi-sq
# current smoker - 2 sample propotion test 
# cigsperday - 2 sample t test
# BPMeds vs disease - 2 sample propotion test
# prevalentStroke, prevalentHyp, diabetes, totChol,sysBP, diaBP, BMI - 2 sample propotion test.

X=df.drop('TenYearCHD',axis=1)
Y=df['TenYearCHD']


# In[ ]:


X.shape,Y.shape


# In[ ]:


from statsmodels.tools import add_constant as add_constant
df_constant=add_constant(df)
df_constant.head()


# In[ ]:


cols=df_constant.columns[:-1]
model=sm.Logit(df.TenYearCHD,df_constant[cols]) # ('y~x',df)
result=model.fit()
result.summary()


# Variables not passing test: Education, Current Smoker, BPMeds, prevelentStroke, prevelentHyp, diabetes, diaBP, BMI, heartRate based on p-value >0.05 higher probability being null hypothesis is True. so reject it.
# Finally, we are buliding the model with 6 features which are passing the statistical tests.

# In[ ]:


X_final=df[['gender','age','cigsPerDay','totChol','sysBP','glucose']]
X_final.shape


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model=LogisticRegression()
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X_final,Y,test_size=0.30,random_state=2)
model.fit(Xtrain,Ytrain)
y_pred=model.predict(Xtest)
acc=metrics.accuracy_score(Ytest,y_pred)
cm=metrics.confusion_matrix(Ytest,y_pred)
print('Overall Accuracy=',acc*100)
print('Confusion Matrix=\n',cm)


# In[ ]:


tpr=cm[1,1]/cm[1,:].sum()# Sensitivity
print(tpr)
print('Senstivity error (%) =',(1-tpr)*100)


# In[ ]:


tnr=cm[0,0]/cm[0,:].sum() #Specivicity
tnr


# In[ ]:


np.round(model.coef_,4)


# Here, the totChol co-eff value is very low, so the contribution of that particular variable is very less. So, it is the good cholestrol value. Basically, the cholestrol value is highly varying the the caronary heart disease the classification.
# So, this dataset cant be treated using the Logistic regression(), we need to use Random Forest or Decision tree to classify this problem.

# In[ ]:




