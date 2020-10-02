#!/usr/bin/env python
# coding: utf-8

# In[ ]:


" Based on Mileage(kms) and Age(yrs), we have to predict the sell price($) of a bmw car"


# In[ ]:


import pandas as pd # data processing
import matplotlib.pyplot as plt  # data visualization
import seaborn as sns   # data visualization


# In[ ]:


df= pd.read_csv("../input/bmw-car-prices/bmw_carprices.csv")
df.head()


# In[ ]:


df.info()  
# columns information


# In[ ]:


df.describe() 
# gives statistics of the dataset


# In[ ]:


df.isnull().sum()
# No missing values 


# In[ ]:


df.corr()['Sell Price($)']
# correlation between independent variables and dependent variable


# In[ ]:


"""In the above output, we can observe negative values for Mileage 
and Age this is due to increase in the mileage/ age , decreases the selling price of a car"""


# In[ ]:



X=df.drop(['Sell Price($)'], axis=1)
X.head() # default : gives top 5 rows


# In[ ]:


y=df["Sell Price($)"]
y.head()


# In[ ]:


#lets visualize our dataset 
sns.set(style='darkgrid')
sns.relplot(x='Mileage(kms)', y='Sell Price($)', data=df)


# In[ ]:


""" Above graph gives, increase in mileage , decreases the selling price of a car """


# In[ ]:


sns.relplot(x='Age(yrs)', y='Sell Price($)', data=df)


# In[ ]:


"""In the above graph, selling price decreases due to increse in the age of a car """


# In[ ]:


# split the dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=.25, random_state=15)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


# using Linear regression

from sklearn.linear_model import LinearRegression

model= LinearRegression()
model.fit(X_train, y_train)  # training the model


# In[ ]:


y_pred= model.predict(X_test)   
# y_pred (predicted values) for a given X_test values

print(y_pred)


# In[ ]:


model.score(X_test, y_test) 
# it gives accuracy score for our model
#accuracy score is good for our dataset


# In[ ]:


#lets visualize the predicted values with true values

plt.scatter(X_test['Mileage(kms)'], y_test, c='blue', alpha=0.5) # true values
plt.scatter(X_test['Mileage(kms)'], y_pred, c='red', alpha=0.5)  # predicted values
plt.xlabel("Mileage(kms)")
plt.ylabel("Sell Price($)")
plt.tight_layout() # it fits the plot cleanly



# In[ ]:



plt.scatter(X_test['Age(yrs)'], y_test, c='green', alpha=0.5) # true values
plt.scatter(X_test['Age(yrs)'], y_pred, c='orange', alpha=0.7)  # predicted values
plt.xlabel("Age(yrs)")
plt.ylabel("Sell Price($)")
plt.tight_layout() # it fits the plot cleanly


# In[ ]:


""" Hope it helps you... keep learning.. 

       If u like it, give an upvote.....Thank you..

