#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This kernel for solving the hackerearth challenge, [Hackerearth Machine Learning Challenge Predict Employee Attrition Rate](https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-predict-employee-attrition-rate/)
# 
# ## Problem statement
# 
# Employees are the most important part of an organization. Successful employees meet deadlines, make sales, and build the brand through positive customer interactions.
# 
# Employee attrition is a major cost to an organization and predicting such attritions is the most important requirement of the Human Resources department in many organizations. In this problem, your task is to predict the attrition rate of employees of an organization.
# 
# ## Data Variable Description
# |Column Name |	Description|
# |:-|:-|
# |Employee_ID| 	Unique ID of each employee|
# |Age| 	Age of each employee|
# |Unit| 	Department under which the employee work|
# |Education| 	Rating of Qualification of an employee (1-5)|
# |Gender| 	Male-0 or Female-1|
# |Decision_skill_possess| 	Decision skill that an employee possesses|
# |Post_Level| 	Level of the post in an organization (1-5)|
# |Relationship_Status| 	Categorical Married or Single |
# |Pay_Scale| 	Rate in between 1 to 10|
# |Time_of_service| 	Years in the organization|
# |growth_rate| 	Growth rate in percentage of an employee|
# |Time_since_promotion| 	Time in years since the last promotion|
# |Work_Life_balance| 	Rating for work-life balance given by an employee.|
# |Travel_Rate| 	Rating based on travel history(1-3)|
# |Hometown| 	Name of the city|
# |Compensation_and_Benefits| 	Categorical Variabe|
# |VAR1 - VAR5| 	Anominised variables|
# |Attrition_rate(TARGET VARIABLE)| 	Attrition rate of each employee|

# # Datasets

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, Markdown, Latex


# ## Data Preprocessing

# In[ ]:


train_data = pd.read_csv("../input/Dataset/Train.csv", index_col="Employee_ID")
X_test_raw = pd.read_csv("../input/Dataset/Test.csv", index_col="Employee_ID")


# In[ ]:


y_train_raw = train_data.Attrition_rate
X_train_raw = train_data.drop("Attrition_rate", axis=1)


# ### Distribution of Training labels

# In[ ]:


sns.distplot(y_train_raw)
plt.title("Distribution of Attrition Rate")


# As it can be seen that most of the employees fall in attrition rate of 0-20%. i.e., employees are less likely to leave the company.

# ### Missing Values

# In[ ]:


def fill_missing(data):
    df = pd.DataFrame.copy(data)
    df['Pay_Scale'] = df['Pay_Scale'].fillna(df.groupby(['Age', 'Education_Level', 'Time_of_service'])['Pay_Scale'].transform('median'))
    df['Time_of_service'] = df['Time_of_service'].fillna(df.groupby(['Age', 'Education_Level', 'Pay_Scale'])['Time_of_service'].transform('median'))
    df['Age'] = df['Age'].fillna(df.groupby(['Education_Level', 'Relationship_Status', 'Time_of_service'])['Age'].transform('median'))
    df = df.fillna(df.median())
    return df


# In[ ]:


X_train_no_na = fill_missing(X_train_raw)
X_test_no_na = fill_missing(X_test_raw)


# ### Distribution of Features

# In[ ]:


numerical_cols = list(X_train_no_na.describe().columns)
plt.subplots(4,4, figsize=(20,20))
i = 1
for col in numerical_cols:
    plt.subplot(4, 4, i)
    try:
        plt.hist(X_train_no_na[col])
        plt.title(col)
    finally:
        i += 1


# From the histograms, its clear that only continous features are **Age**, **Time_of_service**, **growth_rate** and **Pay_Scale**. All other features are actually categorical but are having categories that are defined as numbers. So we convert all the categorical values to have non_numeric type

# ### Convert Categorical Features to Non-Numeric

# In[ ]:


numeric_categoricals = [
 'Education_Level',
 'Time_since_promotion',
 'Travel_Rate',
 'Post_Level',
 'Work_Life_balance',
 'VAR1',
 'VAR2',
 'VAR3',
 'VAR4',
 'VAR5',
 'VAR6',
 'VAR7']

non_numeric_categoricals = [x for x in X_train_no_na.columns if x not in numerical_cols]


# In[ ]:


def convert_categoricals(data):
    df = pd.DataFrame.copy(data)
    for col in numeric_categoricals:
        df[col] = pd.Categorical(df[col])
    for col in non_numeric_categoricals:
        df[col] = pd.Categorical(df[col])
    return df


# In[ ]:


X_train_cats = convert_categoricals(X_train_no_na)
X_test_cats = convert_categoricals(X_test_no_na)


# In[ ]:



output = "| Categorial Column | Train Categories | Test Categories | Equal |"
output += "\n|:--|:--|:--|:--|"
for col in numeric_categoricals + non_numeric_categoricals:
    output += "\n|" 
    output += col 
    output += "|"
    output += str(sorted(X_train_cats[col].unique()))
    output += "|"
    output += str(sorted(X_test_cats[col].unique()))
    output += "|"
    output += ["No","Yes"][sorted(X_train_cats[col].unique())==sorted(X_test_cats[col].unique())]
    output += "|"

display(Markdown(output))


# Clearly from the above table, we have made a right decision by making these columns categorical as the test data and train data have same values of categories. Also this can be confirmed from the data table given at the Introduction part of this notebook.
# 
# **Note**: In some scenerio where test data may get new categories. If test data and train data have same categories, these can be turned to one hot encoding, or else we have to go with categorical values.

# ### Normalize the Numeric Features

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numeric_features = list(X_train_cats.describe().columns)

X_train_normalized = pd.DataFrame.copy(X_train_cats)
X_train_normalized[numeric_features] = scaler.fit_transform(X_train_cats[numeric_features])

X_test_normalized = pd.DataFrame.copy(X_test_cats)
X_test_normalized[numeric_features] = scaler.transform(X_test_cats[numeric_features])


# In[ ]:


X_train_normalized.describe()


# ### One Hot Encoding

# For our categorical features, which do not have numerical values, cannot be processed by our model. So we can either map those values to numeric equalent or convert them to one hot encoding
# 
# In our case, we will convert all the categorical features to one hot encoding

# In[ ]:


X_train_one_hot = pd.get_dummies(X_train_normalized)
X_test = pd.get_dummies(X_test_normalized)


# ## Train-Validation Sets

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train_one_hot, y_train_raw, test_size=0.1, random_state=42)


# # Model

# ## Training

# In[ ]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV

scorer = make_scorer(mean_squared_error, greater_is_better=False)


# In[ ]:


from sklearn.ensemble import AdaBoostRegressor

cls_ada = AdaBoostRegressor(random_state=42, n_estimators=100)

params_ada = {
    'learning_rate': [0.001, 0.01, 0.1, 1, 10,100],
    'loss': ['linear', 'square', 'exponential']
}

grid_search_ada = GridSearchCV(cls_ada, param_grid=params_ada, scoring=scorer, verbose=1)
grid_search_ada.fit(X_train, y_train)

best_cls_ada = grid_search_ada.best_estimator_


# In[ ]:


from sklearn.svm import SVR

cls_svm = SVR()

params_svm = [{
    'kernel': ['linear'],
    'C': [0.1, 1, 10]
},{
    'kernel': ['rbf'],
    'gamma': [10,20,50],
    'C': [0.1, 1, 10]
},{
    'kernel': ['poly'],
    'degree': [2,3,4,5],
    'gamma': [10,20,50],
    'C': [0.1, 1, 10]
},
]

grid_search_svm = GridSearchCV(cls_svm, param_grid=params_svm, scoring=scorer, verbose=1)
grid_search_svm.fit(X_train, y_train)

best_cls_svm = grid_search_svm.best_estimator_


# ## Validation
# ### Score according to Hackerearth:
# score = 100 * max(0, 1 - RMSE(actual_values, predicted_values))

# In[ ]:


print("AdaBoost")
print("============================================")
pred_train = best_cls_ada.predict(X_train)
print("[Training]Mean Squared Error:", mean_squared_error(y_train, pred_train))
print("[Training]Mean Absolute Error:", mean_absolute_error(y_train, pred_train))
print()
pred_val = best_cls_ada.predict(X_val)
print("[Validation]Mean Squared Error:", mean_squared_error(y_val, pred_val))
print("[Validation]Mean Absolute Error:", mean_absolute_error(y_val, pred_val))
print()
RMSE = mean_squared_error(y_val, pred_val)**0.5
print("RMSE:", RMSE)
score = 100 * max(0, 1-RMSE)
print("Score:", score)
print("============================================")
print()
print()
print("SVM")
print("============================================")
pred_train = best_cls_svm.predict(X_train)
print("[Training]Mean Squared Error:", mean_squared_error(y_train, pred_train))
print("[Training]Mean Absolute Error:", mean_absolute_error(y_train, pred_train))
print()
pred_val = best_cls_svm.predict(X_val)
print("[Validation]Mean Squared Error:", mean_squared_error(y_val, pred_val))
print("[Validation]Mean Absolute Error:", mean_absolute_error(y_val, pred_val))
print()
RMSE = mean_squared_error(y_val, pred_val)**0.5
print("RMSE:", RMSE)
score = 100 * max(0, 1-RMSE)
print("Score:", score)
print("============================================")


# ## Prediction

# In[ ]:


pred_test = pd.DataFrame(best_cls_ada.predict(X_test), columns=['Attrition_rate'], index=X_test.index)
pred_test.to_csv("Submission_ada.csv")


# In[ ]:


pred_test = pd.DataFrame(best_cls_svm.predict(X_test), columns=['Attrition_rate'], index=X_test.index)
pred_test.to_csv("Submission_svm.csv")


# In[ ]:




