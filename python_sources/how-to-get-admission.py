#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
df.drop(["Serial No."],inplace=True,axis=1)
df.columns=["gre","toefl","university_rating","sop","lor","cgpa","research","chance_of_admit"]
df.head(2)


# In[6]:


df.info()


# ### Data with high probability of admission

# In[7]:


df_high=df[df.chance_of_admit>=0.90] # data with high probability of admission


# In[8]:


plt.figure(figsize=(10,10))
plt.subplots_adjust(hspace=0.4,wspace=0.4)
for i in range(7):
    plt.subplot(4,2,i+1)
    sns.scatterplot(df_high.iloc[:,i],df_high.iloc[:,-1])


# For high probability(>=90%) of admission you need:
# 1. Need atleas 320 score in gre
# 2. atleast 110 marks in toefl
# 3. Good statement of purpose (with strength more than 4.0)
# 4. Good Letter of Recommendation (with strength more than 3.0)
# 5. CGPA more than 9.0
# 6. Some research is must

# ### Prediction models

# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression


# In[10]:


classifiers={
    'support vector machine ':SVR(gamma='auto'),
    'decision tree          ':DecisionTreeRegressor(),
    'ada boost              ':AdaBoostRegressor(),
    'random forest          ':RandomForestRegressor(n_estimators=10),
    'linear regression      ':LinearRegression()
}


# In[11]:


xdata=df.iloc[:,:-1].values
ydata=df.iloc[:,-1].values

xtrain,xtest,ytrain,ytest=train_test_split(xdata,ydata,test_size=0.20)
xtrain.shape,xtest.shape,ytrain.shape,ytest.shape


# In[12]:


print("Model\t\t\t\t\t\tAccuracy\n")
for name,model in classifiers.items():
    model=model
    model.fit(xtrain,ytrain)
    score=model.score(xtest,ytest)
    print("{} :\t\t {}".format(name,score))


# In[ ]:





# In[ ]:




