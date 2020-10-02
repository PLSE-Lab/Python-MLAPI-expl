#!/usr/bin/env python
# coding: utf-8

# ## Prediction of Temperature based on Salinity 

# In[ ]:


# Importing Libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 


# In[ ]:


# Read the dataset - bottle.csv 
# Here, I've made Btl_cnt column as an Index 

data = pd.read_csv("../input/bottle.csv", index_col = "Btl_Cnt")
data.head(3)


# In[ ]:


# Change Index name by Serial_no, # Inplace true makes it permanent further

data.index.set_names(["Serial_no"], inplace = True)


# In[ ]:


# Extract two columns(Salnity & T_degC) from dataframe for prediction 

dataset = data[["Salnty","T_degC"]]
dataset.head(1)


# In[ ]:


# change the name of the columns 

dataset.columns = ["Sal", "Temp"]


# In[ ]:


dataset.head(1)


# In[ ]:


#dropdown null values everywhere in dataset
dataset = dataset.dropna(axis=0, how="any")


# In[ ]:


# take sample size of 500 to speed up the analysis
Trained_data = dataset[:][:500]
len(Trained_data)


# In[ ]:


#checkout of NaN existance in Sal column of Trained_data
Trained_data["Sal"].isna().value_counts()


# In[ ]:


#checkout of NaN existance in Temp column of Trained_data
Trained_data["Temp"].isna().value_counts()


# In[ ]:


#Dropdown duplicates values in Trained_data
Trained_data = Trained_data.drop_duplicates(subset = ["Sal", "Temp"])
len(Trained_data)


# In[ ]:


import seaborn as sns
sns.set(font_scale=1.6)
plt.figure(figsize=(13, 9))
plt.scatter(Trained_data["Sal"], Trained_data["Temp"],s=65)
plt.xlabel('Sal',fontsize=25)
plt.ylabel('Temp',fontsize=25)
plt.title('Trained_data  - Sal vs Temp',fontsize=25)
plt.show()


# #### Fitting Linear Regression

# In[ ]:


# Divide Trained_data into two variables X & y
X = Trained_data.iloc[:, 0:1].values  # all rows of Sal column
y = Trained_data.iloc[:, -1].values  # all rows of Temp column


# In[ ]:


lin = LinearRegression()
lin.fit(X,y)


# In[ ]:


#Predict value of Temp with random variable
Prediction_Temp_lin = lin.predict([[33]])
Prediction_Temp_lin


# In[ ]:


import seaborn as sns
sns.set(font_scale=1.6)
plt.figure(figsize=(13, 9))
plt.scatter(X,y,s=65)
plt.plot(X,lin.predict(X), color='red', linewidth='6')
plt.xlabel('Sal',fontsize=25)
plt.ylabel('Temp',fontsize=25)
plt.title('Comparision Temp and Predicted Temp with Linear Regression',fontsize=25)
plt.show()


# #### Fitting Polynomial Regression 

# In[ ]:


# Consider degree=3 
poly = PolynomialFeatures(degree = 3) 
X_poly = poly.fit_transform(X) 
poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y)


# In[ ]:


#Predict value of Temp randomly
Prediction_Temp_Poly = lin2.predict(poly.fit_transform([[33]])) 
Prediction_Temp_Poly


# In[ ]:


sns.set(font_scale=1.6)
plt.figure(figsize=(13, 9))
x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.reshape(-1,1)
plt.scatter(X,y,s=65)
plt.plot(x_grid,lin2.predict(poly.fit_transform(x_grid)) , color='red', linewidth = '6')
plt.xlabel('Sal',fontsize=25)
plt.ylabel('Temp',fontsize=25)
plt.title('Comparision Temp and Predicted Temp with Linear Regression',fontsize=25)
plt.show()


# ###### From the above analysis, it is observe that the predictions of temperature are more acurate with Polynomial Regression. Therefore, we gonna move ahead with the Polinomial Regression

# # Replacement of NaN value in Temperature column by Predicted Temp value in dataframe

# In[ ]:


Test_data = data[["Salnty","T_degC"]]
Test_data.head(2)


# In[ ]:


Test_data["Salnty"].isna().value_counts()


# In[ ]:


Test_data.dropna(subset = ["Salnty"], inplace = True)


# In[ ]:


Test_data["Salnty"].isna().value_counts()


# In[ ]:


Test_data["T_degC"].isna().value_counts()


# There are 3262 NaN values of Temperature need to Predict. Next, just get only NaN values of Temp column.

# In[ ]:


NaN_Temp = Test_data[Test_data["T_degC"].isna()]
NaN_Temp


# ####  Define the function called NaN_Temp_Prediction- it will predict all the NaN Temperature associated with given Salnty.
# ##### Salnty = row[0] -- Means it will pass each row for column 0 (column 0 = Salnty) 

# In[ ]:


def NaN_Temp_Prediction(row):
    Salnty = row[0]
    return lin2.predict(poly.fit_transform([[Salnty]]))


# ##### Apply the define function to the location Temp column where, all the predicted values are going to replace NaN. 
# ###### iloc[:3262,-1] = :3262 is the lenght of  rows/Index & -1 is last column that is T_degC

# In[ ]:


NaN_Temp.iloc[:3262,-1] = NaN_Temp.iloc[:3262,].apply(NaN_Temp_Prediction, axis= 1)


# In[ ]:


# Here, we can see that all the values have been replaced by predicted values in T_degC column
NaN_Temp


# In[ ]:


# Values in the T_degC column in the form of an array. Just remove the brackets using .str.strip() method
NaN_Temp["T_degC"] = NaN_Temp["T_degC"].str.get(0)


# ### Final DataFrame -- Predicted Temperature values (Previosly were NaN) accosiated with salnty

# In[ ]:


# Here, is our cleaned dataframe
NaN_Temp


# In[ ]:




