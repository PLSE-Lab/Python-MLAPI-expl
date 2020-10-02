#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(1234)


# In[ ]:


dataset=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
dataset.head()


# **Getting Info about the dataset and then checking for any Null values**

# In[ ]:


dataset.info()
dataset.isnull().values.any()


# **Renaming the column names for easing our task and setting Serial No. as the index**

# In[ ]:


dataset.columns
dataset.rename(columns = {'Serial No.':'SNo', 'GRE Score':'GRE', 'TOEFL Score':'TOEFL', 'University Rating':'UniRate',
                          'Chance of Admit ':'Chance'}, inplace = True)
dataset = dataset.set_index('SNo')
dataset.head()


# **Finding the relation between the parameters using heatmap**

# In[ ]:


var_corr = dataset.corr()
plt.subplots(figsize=(20,10))
sns.heatmap(var_corr, xticklabels=var_corr.columns, yticklabels=var_corr.columns, cmap = 'RdBu', annot=True, linewidths = 0.9)


# **Printing the column that show which parameter influences the Chance of Admit the most**

# In[ ]:


chance_car_corr = var_corr.iloc[:,-1]
print(chance_car_corr)


# **From the heat map we can say that the Chances of getting admit is mostly influenced by GRE, TOEFL & CGPA**

# In[ ]:


plt.subplots(figsize=(15,6))
sns.regplot(x="GRE",y="Chance",data=dataset, color = 'red')
plt.subplots(figsize=(15,6))
sns.regplot(x="TOEFL",y="Chance",data=dataset, color = 'green')
plt.subplots(figsize=(15,6))
sns.regplot(x="CGPA",y="Chance",data=dataset, color = 'blue')


# In[ ]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 7].values
print('Shape of X:', X.shape)
print('Shape of y:', y.shape)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=1/5, random_state=8)


# In[ ]:


print('Shape of X_train:', X_train.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of X_test:', X_test.shape)
print('Shape of y_test:', y_test.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting Test set results
y_pred = regressor.predict(X_test)


# In[ ]:


print(y_pred)
print('Shape of y_pred:', y_pred.shape)


# **Finding the R_square and MAE**

# In[ ]:


from sklearn.metrics import mean_absolute_error,r2_score
r2 = r2_score(y_test, y_pred)
print('r_square score for the test set is', r2)
MAE = mean_absolute_error(y_pred,y_test)
print('MAE is', MAE)


# *  **This task can also be done by Backward elimination method where we use the Ordinary Least Square method and we keep eliminating the parameters with the highest P value till we find parameter(s) that influences the chance of getting admitted**
