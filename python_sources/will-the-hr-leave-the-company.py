#!/usr/bin/env python
# coding: utf-8

# 1. Problem: Given the following dataset, we've to predict whether the HR will stay or leave the company. So, it's a classification problem.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# 2. Data insights.

# In[ ]:


data = pd.read_csv("/kaggle/input/hr-analytics-case-study/general_data.csv")
data.head(5)


# Find the shape of the dataframe

# In[ ]:


data.shape


# In[ ]:


data.columns


# Convert the categorical variable into numeric variables using Label Encoder

# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
data['BusinessTravel'] = labelEncoder_X.fit_transform(data['BusinessTravel'])
data['Department'] = labelEncoder_X.fit_transform(data['Department'])
data['EducationField'] = labelEncoder_X.fit_transform(data['EducationField'])
data['Gender'] = labelEncoder_X.fit_transform(data['Gender'])
data['JobRole'] = labelEncoder_X.fit_transform(data['JobRole'])
data['MaritalStatus'] = labelEncoder_X.fit_transform(data['MaritalStatus'])
data['Over18'] = labelEncoder_X.fit_transform(data['Over18'])


# The dependent variable y is Attrition whether the employee will stay or not.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_encoder_y = LabelEncoder()
data['Attrition'] = label_encoder_y.fit_transform(data['Attrition'])


# In[ ]:


data.head()


# Check if there's any **Null value (NaN)** (Data Cleaning)

# In[ ]:


data.isnull().any()


# The columns NumCompaniesWorked and TotalWorkingYears have null values. We'll replace it by mean.

# In[ ]:


import math
mean_companies_worked = math.floor(data["NumCompaniesWorked"].mean())
data["NumCompaniesWorked"].fillna(mean_companies_worked, inplace = True)
mean_working_years = math.floor(data["TotalWorkingYears"].mean())
data["TotalWorkingYears"].fillna(mean_working_years, inplace = True)


# In[ ]:


data.isnull().any()


# Now, we've no more null values in any column

# 3. Evaluation: Trying to find if there's **Correlation between the independent variables.** 
# 

# In[ ]:


corr = data.corr()
print(corr)


# Using a heatmap to visualize better which variables have highest correlation

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (18, 9))
sns.heatmap(corr, annot = True, linewidth = 0.05, cmap = 'BuPu')
plt.show()


# 4. Features: After seeing the heatmap, I'm taking up the following independent variables. Some variables are omitted which have the least pairwise correlation between the independent variable and dependent variable "Attrition". There are no derived features used in this model.

# In[ ]:


X = data[['Age', 'EducationField', 'Gender', 'JobLevel',
          'JobRole', 'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked',
       'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']]
y = data["Attrition"]


# In[ ]:


X


# 5. Building the model. I'll split the data using sklearn and then use Logistic regression for classification.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
Scaler_X = StandardScaler()
X_train = Scaler_X.fit_transform(X_train)
X_test = Scaler_X.transform(X_test)


# In[ ]:


len(y_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)


# 6. Model prediction and experiment

# In[ ]:


model.predict(X_test)


# In[ ]:


predicted_y = model.predict(X_test)


# In[ ]:


model.score(X_test, y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

results = confusion_matrix(predicted_y, y_test) 
  
print('Confusion Matrix :')
print(results) 
print("Accuracy Score: ", accuracy_score(y_test, predicted_y))
print("Classification Report: \n", classification_report(y_test, predicted_y))


# Conclusion: The **accuracy score is average**. I've to look into other models for making better prediction.
