#!/usr/bin/env python
# coding: utf-8

# # simple linear regression_ML on salary_data

# In[ ]:


#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[ ]:


data=pd.read_csv('../input/salary_data.csv')
data.head()


# In[ ]:


data.info()


# In[ ]:


X=data.iloc[:,:-1].values
Y=data.iloc[:,1].values


# In[ ]:


sb.distplot(data['YearsExperience'])


# In[ ]:


sb.scatterplot(data['YearsExperience'],data['Salary'])


# In[ ]:


#splitting the dataset into the Training set and test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[ ]:


#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


# In[ ]:


regressor.score(x_test,y_test)


# In[ ]:


#predicting the test set results
y_predict=regressor.predict(x_test)
y_predict


# In[ ]:


regressor.predict([[1.5]])


# In[ ]:


#visualizing the training set results
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:


plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# # END
