#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Linear model regression applied to E Commerce topics 

# We are going to analyse some customer data.This Notebook allows you to practice machine learning basics and some data visualisation :
# 1. Upload and Read a csv file.
# 1. Explore data by visualisation 
# 1. Use a LinearRegression model
# 1. Predict
# 1. Measure error 

# Let's get started!

# In[ ]:


# Import libraries 
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[ ]:


#Read customer csv file 
customer_file="../input/ecommerce-customers/Ecommerce Customers.csv"
customer = pd.read_csv(customer_file)
#show the head of customer DataFrame
customer.head()


# In[ ]:


#show statistcs information
customer.describe()


# In[ ]:


#visualize data
sns.pairplot(customer)


# One we explore these types of relationships across the numerical features of the dataset we can see clearly linear relation between Lengh of Membership and Yearly Amount Spent. Let's see this relation in more details.

# In[ ]:


sns.lmplot(x="Yearly Amount Spent", y="Length of Membership", data=customer)


# Now we are going to create our model and predict "The yearly amount Spent" based on the numerical features. We will try to figure out which feature is more important to invest in  

# # Linear model regression  

# In[ ]:


features= ["Avg. Session Length", "Time on App" ,"Time on Website", "Length of Membership"]
X=customer[features] #Features
y=customer["Yearly Amount Spent"] #Outcome

# split the data (20% for testing 80%for training)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=1)
#create an instance of Linear regession model
lmodel= LinearRegression()
#fit the model with the training data
lmodel.fit(X_train, y_train)
#testing the model by making predicitions  
prediction = lmodel.predict(X_test)


# # Explore the results

# In[ ]:


#visualize real data and predictif ones 
plt.scatter(y_test, prediction)
plt.xlabel("Real")
plt.ylabel("prediction")
plt.title("Yearly Amount Spent")


# # Evaluating the Model (Measure error) 

# In[ ]:


MAE= metrics.mean_absolute_error(y_test,prediction) #Mean absolute error 
MSE= metrics.mean_squared_error(y_test,prediction) #Mean squared error 
RMSE = np.sqrt(MSE) # Root Mean squared error 


# In[ ]:


print("Mean absolute error: ",MAE )


# In[ ]:


coef= lmodel.coef_
coeff_table = pd.DataFrame(coef, features, columns=['Coefficient'])
coeff_table


# # Interpretation

# 1. A 1 unit increase in **Avg. Session Length** is associated with an increase of **25.98** total dollars yearly amount spent
# 1. A 1 unit increase in **Time on App** is associated with an increase of **38.59** total dollars yearly amount spent
# 1. A 1 unit increase in **Time on Website** is associated with an increase of **0.19** total dollars yearly amount spent
# 1. A 1 unit increase in **Length of Membership** is associated with an increase of **61.27** total dollars yearly amount spent

# The website app does not bring benefit to the total dollars annual amount spent. The company should focus in this area to bring to catch up to the performance of the mobile ap or it should focus on the mobile app because it's already working well. Which means a progress in the mobile app will be associated with the highest increase in the total dollars yearly amount spent.I choose the website application. What do you choose ?
# 

# 
