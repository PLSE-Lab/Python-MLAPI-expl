#!/usr/bin/env python
# coding: utf-8

# # Graduate Admissions - EDA/Linear Regression

# ## Imported Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
style.use(['fivethirtyeight', 'seaborn-whitegrid'])


# ## Dataset

# I will be using the Version 1.1 dataset as it has more rows of data.

# In[ ]:


grad_DF = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')


# In[ ]:


grad_DF.head()


# In[ ]:


grad_DF.shape


# This dataset has 500 rows and 9 columns.

# ## Explaining the Features

# * GRE Score: The scores acquired in the Graduate Record Examination (GRE). It is a standardized test that is an admission requirement for many graduate schools in the United States and Canada. The highest possible score in the GRE is 340.
# * TOEFL Score: The scores acquired in the Test of English as a Foreign Language (TOEFL). It is a standardized test to measure the English language ability of non-native speakers wishing to enroll in English-speaking universities. The highest possible score in the TOEFL is 120.
# * University Rating: The rating of the university out of 5.
# * SOP: Statement of Purpose rating out of 5. It is an essay stating the purpose of applying to a particular course in a particular university.
# * LOR: Letter of Recommendation rating out of 5. A letter of recommendation is a document used by someone who is applying for a job, internship, college application, leadership position or volunteer opportunity. The purpose of a recommendation letter is to validate what the employer has learned about the applicant and get answers to outstanding questions about their performance or habits.
# * CGPA: Cumulative Grade Point Average out of 10. It is the average of Grade Points obtained for all semesters and courses completed up to a given academic term.
# * Research: This is a binary value. It depicts whether the graduate has any research experience.
# * Chance of Admit: The chance of admission for a student.

# # Data Preparation

# ## Altering Column Names

# In[ ]:


list(grad_DF.columns)


# The column names 'LOR ' and 'Chance of Admit ' have a space at the end. With the code below, I will be removing the space.

# In[ ]:


grad_DF.rename(columns = {"LOR ": "LOR", "Chance of Admit ": "Chance of Admit"}, inplace = True)


# After the removal of the space:

# In[ ]:


list(grad_DF.columns)


# ## Removing the Serial Number Column

# In[ ]:


grad_DF.drop(columns=['Serial No.'], inplace = True)


# In[ ]:


grad_DF.head()


# # Data Analysis

# In[ ]:


f, ax1 = plt.subplots(4, 2, figsize = (13,17))
sns.countplot(data = grad_DF, x = "University Rating", ax = ax1[0,0])
sns.countplot(data = grad_DF, x = "Research", ax = ax1[0,1])
sns.distplot(grad_DF["GRE Score"], ax = ax1[1, 0])
sns.distplot(grad_DF["TOEFL Score"], ax = ax1[1, 1])
sns.countplot(data = grad_DF, x = "SOP", ax = ax1[2,0])
sns.countplot(data = grad_DF, x = "LOR", ax = ax1[2,1])
sns.distplot(grad_DF["CGPA"], ax = ax1[3, 0])
sns.distplot(grad_DF["Chance of Admit"], ax = ax1[3, 1])
plt.show()


# In[ ]:


grad_DF[['GRE Score', 'TOEFL Score', 'CGPA', 'Chance of Admit']].describe()


# * Majority (32.4%) of all graduates have a University Rating of 3. University Rating of 1 is the minority (6.8%).
# * 56% of all graduates have done research.
# * Majority (19.8%) of all graduates have a Letter of Recommendation strength of 3.0. This is followed by a strength of 4.0, with 18.8% of all graduates.
# * The majority (35.4%) of all graduates have a score of 3.5 and 4 for their Statement of Purpose.
# * The mean GRE Score is 316.4. The lowest and highest GRE score is 290 and 340 respectively.
# * The mean TOEFL Score is 107. The lowest and highest TOEFL score is 92 and 120 repectively.
# * The mean chance of admission is 72.17%. The lowest and highest chance of admission is 34% and 97% respectively.

# # Correlation

# In[ ]:


grad_corr = grad_DF.corr()
plt.figure(figsize = (7,7))
sns.heatmap(grad_corr, annot = True, linewidths = 1.2, linecolor = 'white')
plt.show()


# Out of all the predictors, CGPA has the highest correlation to the target variable, Chance of Admission. Research has the lowest correlation to the Chance of Admission. The predictors have high correlation to each other, which can affect the multicollinearity. 

# # Linear Regression

# The target variable in this case will be the chance of admission.

# In[ ]:


X = grad_DF[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']]
Y = grad_DF['Chance of Admit']


# ## Scaling the Features

# In[ ]:


X_scaled = preprocessing.scale(X)
X_scaled_DF = pd.DataFrame(data = X_scaled, columns = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research'])
X_scaled_DF.head()


# Since all of these features are measured differently, scaling will be necessary. This will ensure that the features are assessed at the same level.

# ## Train and Test Split

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)


# The training and testing dataset follows the 80/20 split

# In[ ]:


grad_reg = LinearRegression()
grad_reg.fit(X_train, Y_train)


# In[ ]:


grad_pred = grad_reg.predict(X_test)


# ## VIF Values

# The variance influence factor (VIF) quantifies the correlation between one predictor and the other predictors in the model.

# In[ ]:


vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X_scaled_DF.values, i) for i in range(X_scaled_DF.shape[1])]
vif["Features"] = X_scaled_DF.columns

vif


# ## Actual and Predicted Values

# In[ ]:


pred_DF = pd.DataFrame({'Actual': Y_test, 'Predicted': grad_pred})
pred_DF.head(20)


# ## MAE, MSE and RMSE

# Mean Absolute Error measures the absolute value of the errors between the actual values and the predicted values.
# 
# <img src = "https://miro.medium.com/max/1040/1*tu6FSDz_FhQbR3UHQIaZNg.png" width = 300px>
# 
# Mean Squared Error measures the average squared difference between the predicted values and the actual values.
# 
# <img src = "https://i.imgur.com/vB3UAiH.jpg" width = 300px>
# 
# Root Mean Squared Error measures the square root of the differences between predicted and actual values.
# 
# <img src = "https://secureservercdn.net/160.153.137.16/70j.58d.myftpupload.com/wp-content/uploads/2019/03/rmse-2.png" width = 300px>

# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, grad_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, grad_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, grad_pred)))


# ## R-squared Value

# R-squared value measures of how well the values are predicted. The higher the R-squared value, the better the model fits the data.

# In[ ]:


metrics.r2_score(Y_test, grad_pred)


# # Conclusion

# With an R-squared value of 0.8188, this shows that the model used explains 81.88% of the variation in the response variable around its mean.
