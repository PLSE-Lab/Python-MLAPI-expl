#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Data Loading Phase 

# In[ ]:


df = pd.read_csv("../input/Adsense.csv")
df.head()


# # EDA (Data Analysis and Data Visualization) Phase

# In[ ]:


#Get the info of the data frame
df.info()


# In[ ]:


#Calculate Some Statistics of the data set
df.describe()
df = df.dropna()
df.shape


# In[ ]:



df[df.isnull()].count()


# In[ ]:


plt.scatter(df['Clicks'],df['Estimated earnings (INR)'])
plt.title('Clicks Vs Estimated earnings (INR)')
plt.xlabel('Clicks')
plt.ylabel('Estimated earnings (INR)')
# sns.pairplot(df)


# In[ ]:


# Find more relationship between other features using pairplot sns
df.fillna(df.mean(), inplace=True)
sns.pairplot(df)


# In[ ]:


#Get the features and target values from the data-sets

X=df.drop(['Month','Active View Viewable','Estimated earnings (INR)'],axis=1)
y=df['Estimated earnings (INR)']


# # Model Building (Using Scikit-Learn Library)

# In[ ]:


# Import the library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[ ]:


#Split the data into training and testing sets using train_test_split function
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=1)


# In[ ]:


#Initialize the Linear Regression Model
lm=LinearRegression()

#Fit the model using training data set
lm.fit(X_train,y_train)


# In[ ]:


#predict the model using test data
y_pred=lm.predict(X_test)
# print the coefficients
print(lm.intercept_)
print(lm.coef_)


# # Model Evaluation 
# 1-Mean Absolute Error (MAE) is the mean of the absolute value of the errors
# 
# 2-Mean Squared Error (MSE) is the mean of the squared errors
# 
# 3-Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors
# 
# 
# 

# In[ ]:


#Import Libraries for calculating model performance
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[ ]:



print("Mean Absolute Error",(y_test,y_pred))
print("R2 Score",r2_score(y_test,y_pred))


print("=======================================")

output_df=pd.DataFrame({'Actual Output':y_test,'Predicted Output':y_pred})
output_df



# y_test

