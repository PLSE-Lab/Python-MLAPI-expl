#!/usr/bin/env python
# coding: utf-8

# ****So here we are going to predict the profit of the startsup based on the given factors and we will also see that which factors make most to help us.
# It is a very small dataset so me being a beginner suggest others also to start with this..
# Small always makes Big :)

# In[ ]:


# importing basic libaries for the basic operations
import numpy as np # for the mathematical calculations
import pandas as pd # for reading our data/ csv files
import seaborn as sns # for the visualising the data
from sklearn import metrics # for computing some parameters of our linear regression model
import matplotlib.pyplot as plt # for some pollting of the data points
from sklearn.model_selection import train_test_split # to split our dataset 
from sklearn.linear_model import LinearRegression # the model from sklearn 
df = pd.read_csv("/kaggle/input/startup-logistic-regression/50_Startups.csv")
df


# So when we looked our dataset we have found that the there is only one coloumn that is categorical and need to change to numeric and so i have used the pandas to get the  dummies its really a very powerful tool... Believe me guys!!!!

# In[ ]:


df['State'] = pd.get_dummies(df['State'], prefix='State')


# So since all the coloums are in the numeric foam we are good to start and look on what factors did the prediction of profit depends which will be very helpful in many cases.. Since here we have very less coloums so its easy but in the case of more coloums what we can do is by using **df.corr()** which help to find the correlation between each coloum and then **annot = True** helps us to find how much they are co-related. So lets have a look guys

# In[ ]:


sns.heatmap(df.corr(),annot=True)


# So above you can see in order to predict the price the major role is played by the R&D speed > Marketing Speed > Administration > state.
# SO here thats it but in future if you have more coloums you can try out which makes so much things easy.

# In[ ]:


# now here using a poweful tool called Slicing we are giving the labels which help in prediction in x and one to predict in y
x = df.iloc[:,:4].values
y = df.iloc[:,4:5].values


# In[ ]:


# now splitting our dataset as per your choice generally a good pratice means (80:20 or 75:25)
(trainX, testX, trainY, testY) = train_test_split(x, y, test_size=0.25, random_state=0)

print("X Train shape:", trainX.shape)
print("X Test shape:", testX.shape)
print("Y Train shape:", trainY.shape)
print("Y Test shape:", testY.shape)


# In[ ]:


# training our model a very simple and nice model :)
regressor = LinearRegression()  
regressor.fit(trainX, trainY)


# So here we are going to see the intercept and the coff of each of the attribute so its like
# **Y = x1+ m1*x2+m2*x3+....+e****  and son for the multiple regression where **e**** is the random noise and **x1+ m1*x2+m2*x3**** is linear predictors

# In[ ]:


print(regressor.intercept_)
print(regressor.coef_)


# In[ ]:


# now doing prediction on our model
y_pred = regressor.predict(testX)
y_pred.shape


# So here we can see the actual value and predicited value returned by our datset.... :)

# In[ ]:


df = pd.DataFrame({'Actual': testY.flatten(), 'Predicted': y_pred.flatten()})
df


# In[ ]:


# the bar plot is very helpful when you deal with such numbers this will add a cherry on your cake guys.... and make it easy for everyone to understand !!!!
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:


# and last some parameters to check your model correctly and i am not writting much about these i think just copy and pasting from google is waste so you guys can go and
#search and i am sure you will get much good than me and tell me that wheather my model is good or not based on these parameter values....

print('Mean Absolute Error:', metrics.mean_absolute_error(testY, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(testY, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(testY, y_pred)))


# Hope you guys have learned something meaningful...
# And with that hope guys do tell me about the metrices and comment if you like or have any doubts..... :);):)
