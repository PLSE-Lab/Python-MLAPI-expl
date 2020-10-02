#!/usr/bin/env python
# coding: utf-8

# In this notebook, I will give a baseline prediction for employees' resignation behavior.
# ## Setup
# Run the next cell to import and configure the Python libraries that I need to complete the prediction. Specially, I will use 
# * **Pandas** to read and operate the data;
# * **Seaborn** to plot and explore the data;
# * **Sklearn** to import ML models.
# 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Step 1: Load the Data
# Read the employees data file into `employee_data`. Use the `user_id` column to label the rows.

# In[ ]:


train_file = r'../input/rs6-attrition-predict/train.csv';
test_file = r'../input/rs6-attrition-predict/test.csv';
train_employee_data = pd.read_csv(train_file, index_col='user_id');
test_employee_data = pd.read_csv(test_file, index_col='user_id');
print(train_employee_data.head());


# ## Step 2: Explore the Data
# To explore the data, we could do the following things:
# * Check for the possible shortage of the data;
# * Take `Age` as an example to explore the relationship between predictors and target;
# * Drop `EmployeeNumber` or not: The relationship between `EmployeeNumber` and `Attrition`;
# * The confusing relationships among `DailyRate`, `MonthlyRate`, and `MonthlyIncome`.
# 

# In[ ]:


#check for the shortage of the data
print(train_employee_data.isnull().any())


# In[ ]:


#the relationship between predictors and target;
sns.distplot(train_employee_data['Age'], kde=True);
plt.show();
sns.swarmplot(x=train_employee_data['Attrition'], y=train_employee_data['Age']);
plt.show();


# We find that those who decide to quit generally have younger age.

# In[ ]:


#the relationship between `EmployeeNumber` and `Attrition`.;
sns.swarmplot(x=train_employee_data['Attrition'], y=train_employee_data['EmployeeNumber']);


# As we could see here, `EmployeeNumber` performs no particular relationship with `Attrition`. 

# In[ ]:


#the confusing relationship among `DailyRate`, `MonthlyRate`, and `MonthlyIncome`
sns.scatterplot(x=train_employee_data['DailyRate'], y=train_employee_data['MonthlyRate']);
plt.show();
sns.scatterplot(x=train_employee_data['DailyRate'], y=train_employee_data['MonthlyIncome']);
plt.show();
sns.scatterplot(x=train_employee_data['MonthlyRate'], y=train_employee_data['MonthlyIncome']);
plt.show();


# The income or the rate is considered to be the same indicator for the employees' income. So, they should perform linear relationship. Surprisingly, they are totally messy. I am quite confused.  

# ## Step 3: Preprocessing
# Here, we 
# * Split `train_employ_data` into train set and valid set;
# * Use One-Hot Encoding to encode the categorical columns;
# * Use One-Hot Encoding to encode the target column.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


#predictors and target
y = train_employee_data['Attrition'];
X = train_employee_data.drop(['Attrition'], axis=1);

#training set and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0);

#numerical columns and categorical columns
numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']];
categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object'];

#count the unique items in each categorical columns
print(train_employee_data[categorical_cols].nunique());


# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


#Use one-hot encoder to encode the categorical columns
OH_encoder = OneHotEncoder(sparse=False);
OH_train_cols = pd.DataFrame(OH_encoder.fit_transform(X_train[categorical_cols]));
OH_valid_cols = pd.DataFrame(OH_encoder.transform(X_val[categorical_cols]));
OH_test_cols = pd.DataFrame(OH_encoder.transform(test_employee_data[categorical_cols]));

OH_train_cols.index = X_train.index;
OH_valid_cols.index = X_val.index;
OH_test_cols.index = test_employee_data.index;

numerical_train_cols = X_train.drop(categorical_cols, axis=1);
numerical_valid_cols = X_val.drop(categorical_cols, axis=1);
numerical_test_cols = test_employee_data.drop(categorical_cols, axis=1);

OH_train = pd.concat([OH_train_cols, numerical_train_cols], axis=1);
OH_valid = pd.concat([OH_valid_cols, numerical_valid_cols], axis=1);
OH_test = pd.concat([OH_test_cols, numerical_test_cols], axis=1);


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


label_encoder = LabelEncoder();
y_train = label_encoder.fit_transform(y_train);
print(y_val);
y_val = label_encoder.transform(y_val);
print(y_val);


# ## Step 4: Validation
# Here, we 
# * Use LR to fit the data;
# * predict and print the correct rate; 

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


#Use LR to fit the data
lr = LogisticRegression(C=1e5);
lr.fit(OH_train, y_train);
my_valid_predict = lr.predict(OH_valid);

#error
false_number = ((my_valid_predict - y_val) ** 2).sum();
total_number = len(y_val);
print((total_number - false_number) / total_number);


# ## Step 5: Predict

# In[ ]:


import numpy as np


# In[ ]:


test_preds = lr.predict(OH_test);
print(test_preds);
print(test_employee_data.index);
output = pd.DataFrame({'user_id': test_employee_data.index, 'Attrition': test_preds});
output.to_csv('submission.csv', index=False);
print(output)


# In[ ]:




