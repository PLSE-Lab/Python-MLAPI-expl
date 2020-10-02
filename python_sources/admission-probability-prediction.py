#!/usr/bin/env python
# coding: utf-8

# # <center>Admission Probability Prediction</center>

# ## 1. Introduction

# **Objective**<br/>
# It's an anxious feeling to wait for the graduation admission status for any student once they apply. University accepts the admission based on various parameters of the student. The predictive model will decide the probabiliy of getting an admission.
# 
# **Data Description**<br/>
# The dataset contains student's GRE Score, TOEFL Score, University Rating, Statement of purpose score, Letter of recomendation score, CGPA, Whether the student has done any research or not and Chances of getting admission.

# ## 2. Exploratory Data Analysis (EDA)

# ### 2.1 Import libraries

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import xgboost
get_ipython().run_line_magic('matplotlib', 'inline')

print(os.listdir("../input/"))
os.chdir("../input/")


# ### 2.2 Import Dataset

# In[ ]:


df = pd.read_csv('Admission_Predict_Ver1.1.csv')
print(df.head())
df.drop('Serial No.', axis = 1, inplace = True)


# Since the column Serial No is insignificant, it should be droppped

# In[ ]:


print(f'Dataset contains {df.shape[0]} samples, {df.shape[1] - 1} independent features 1 target continuous variable.')


# ### 2.3 Basic Analysis on the dataset

# In[ ]:


print(df.info())

missing_values = (df.isnull().sum() / len(df)) * 100
print("\nFeatures with missing values: \n", missing_values[missing_values > 0])


# * All the independent features in the dataset are numeric.
# * The target variable 'Chance of Admit' is a floating point number.
# * There is no feature with missing values.

# In[ ]:


df.describe()


# * Data description show there is no outliers in any features.

# In[ ]:


sns.heatmap(df.corr(), annot = True)


# * The heatmap shows Chances of Admission is depends mostly on CGPA, GRE Score and TOEFL Score.
# * Whether the student has done research or not doesn't affect the Chances of admission much.

# In[ ]:


l = df.columns.values
number_of_columns=df.shape[1]
number_of_rows = len(l)-1/number_of_columns
plt.figure(figsize=(2*number_of_columns,5*number_of_rows))
for i in range(0,len(l)):
    plt.subplot(number_of_rows + 1,number_of_columns,i+1)
    sns.distplot(df[l[i]],kde=True) 


# * Almost all the features are normally distributed.

# In[ ]:


sns.pairplot(df)


# EDA Conclussion: The features CGPA, GRE Score and TOEFL Score have more linear relationship with Chances of Admission. Hence we will be using only these three features and eliminating the rest of the features.

# ## 3. Data Pre Processing

# Before training the model for the dataset, the following preprocessing steps will be under taken with the dataset.
# 
# * Independent and target data will be seperated as X and Y.
# * Data will be split into train and test set with test set size 20%.
# * Feature scalling will be done on independent features to standarize them.

# In[ ]:


X = df[['CGPA', 'GRE Score', 'TOEFL Score']].values
Y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# ##  4. Build the model

# Since the selected features have more linear relationship with the target variable, Multiple linear regression model will be used to train the dataset and predict the chances of getting admission.

# In[ ]:


# Linear Regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

# XGBoost
xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(x_train, y_train)

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rand_forest_reg = RandomForestRegressor()
rand_forest_reg.fit(x_train, y_train)


# ## 5. Model Evaluation

# ### 4.1 Adjusted R Squared Score

# In[ ]:


from sklearn.metrics import r2_score
y_pred_lin_reg = reg.predict(x_test)
y_pred_xgb = xgb_reg.predict(x_test)
y_pred_rf = rand_forest_reg.predict(x_test)
print(f"Adjusted R Squared Score for Linear Regression: {r2_score(y_test, y_pred_lin_reg)}")
print(f"Adjusted R Squared Score for XGBoost Regression: {r2_score(y_test, y_pred_xgb)}")
print(f"Adjusted R Squared Score for Random Forest: {r2_score(y_test, y_pred_rf)}")


# ### 4.2 K-Fold Cross Validation

# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(reg, x_train, y_train, cv = 6)
print(np.char.center("Linear Regression Score", 40, fillchar = '*'))
print("Scores: ", scores)
print("Accuracy: ", scores.mean() * 100, "%")
print("Standard Deviation: +/-", scores.std(), "\n\n")

print(np.char.center("XGBoost Score", 40, fillchar = '*'))
scores = cross_val_score(xgb_reg, x_train, y_train, cv = 6)
print("Scores: ", scores)
print("Accuracy: ", scores.mean() * 100, "%")
print("Standard Deviation: +/-", scores.std(), "\n\n")

print(np.char.center("Random Forest Score", 40, fillchar = '*'))
scores = cross_val_score(rand_forest_reg, x_train, y_train, cv = 6)
print("Scores: ", scores)
print("Accuracy: ", scores.mean() * 100, "%")
print("Standard Deviation: +/-", scores.std())


# ## 6. Conclussion

# Linear regression model will be chosen as the suitable model for this problem
