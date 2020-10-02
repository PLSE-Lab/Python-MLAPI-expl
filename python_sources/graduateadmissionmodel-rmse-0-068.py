#!/usr/bin/env python
# coding: utf-8

# # **Graduate admission prediction using linear regression**

# # Importing all the modules

# In[ ]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd


# # Getting the data

# In[ ]:


df=pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')
df.head(10)


# # Checking for null values

# In[ ]:


df.isnull().sum()


# In[ ]:


df1=df.drop(['Serial No.'],axis=1)
df1.describe(include='all')


# # **Input feature and target assigning** 

# In[ ]:


X=df.iloc[:,1:-1]
y=df[['Chance of Admit ']]


# # Visualizing various features and their relationship with admission rate

# In[ ]:


plt.scatter(df[['GRE Score']],y,marker='x',color='red')
plt.xlabel('GRE')
plt.ylabel('Admission rate')
plt.show()


# In[ ]:


plt.scatter(df[['TOEFL Score']],y,marker='x',color='red')
plt.xlabel('TOEFL Score')
plt.ylabel('Admission rate')
plt.show()


# In[ ]:


plt.scatter(df[['CGPA']],y,marker='x',color='red')
plt.xlabel('CGPA')
plt.ylabel('Admission rate')
plt.show()


# In[ ]:


plt.scatter(df[['SOP']],y,marker='x',color='red')
plt.xlabel('Statement of Purpose')
plt.ylabel('Admission rate')
plt.show()


# # Splitting training and testing data

# In[ ]:


Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)


# # Fitting the model and values of intercept and coefficients

# In[ ]:


model=LinearRegression(normalize=True)
model.fit(Xtrain,ytrain)
print('Intercept:',model.intercept_)
coeff_df = pd.DataFrame(model.coef_.T, X.columns, columns=['Coefficient'])  
coeff_df


# # Prediction accuracy

# In[ ]:


ypred=model.predict(Xtest)
s=model.score(Xtest,ytest)
print('Accuracy:{} %'.format(round(s*100,2)))


# # Scatter plot of actual and predicted y values

# In[ ]:


plt.scatter(ytest,ypred,marker='x')
plt.xlabel('actual values')
plt.ylabel('predicted values')
plt.show()


# # Now, predicting whether a new candidate can get admitted to graduate school

# In[ ]:


X_new=[[320,110,3,4.5,4,9.8,0]] 
n=float(model.predict(X_new))*100
print("Probability that you get into a university : {}% ".format(round(n,2)))


# # RMSE Score

# In[ ]:


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(ytest, ypred)))

