#!/usr/bin/env python
# coding: utf-8

# # Predicting Telco Customer Churn using Random Forest Classifier

# In this notebook I will build the prediction model using the SparkML library.
# 
# This notebook walks you through these steps:
# 
# - Load and Visualize data set.
# - Build a predictive model

# ## 1.0 Install required packages
# 
# There are a couple of Python packages I will use in this notebook. 

# In[ ]:


get_ipython().system('pip install --user pyspark==2.3.3 --upgrade|tail -n 1')


# In[ ]:


import pandas as pd
import numpy as np
import json
import os

import warnings
warnings.filterwarnings("ignore")


# ## 2.0 Load and Clean data
# 
# The data will be loaded as a pandas data frame.
# 

# In[ ]:


import os
print(os.listdir("../input/telco-customer-churn"))


# In[ ]:


df = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()


# ### 2.1 Drop CustomerID feature (column)
# As it contains PII

# In[ ]:


df = df.drop('customerID', axis=1)
df.head(5)


# ### 2.2 Examine the data types of the features

# In[ ]:


df.info()


# In[ ]:


# Statistics for the columns (features). Set it to all, since default is to describe just the numeric features.
df.describe(include = 'all')


# It can be observed that Tenure ranges from 0 (new customer) to 6 years, Monthly charges range from $18 to $118, etc

# ### 2.3 Check for need to Convert TotalCharges column to numeric if it is detected as object
# 
# Since above `df.info` shows the "TotalCharges" columnn as an object, it needs to converted to numeric.

# In[ ]:


totalCharges = df.columns.get_loc("TotalCharges")
new_col = pd.to_numeric(df.iloc[:, totalCharges], errors='coerce')
df.iloc[:, totalCharges] = pd.Series(new_col)


# 
# 
# ### 2.4 Any NaN values should be removed to create a more accurate model.

# In[ ]:


# Check if we have any NaN values and see which features have missing values that should be addressed
print(df.isnull().values.any())
df.isnull().sum()


# `TotalCharges` column has missing values. There are various ways to address this issue:
# 
# - Drop records with missing values 
# - Fill in the missing value with one of the following strategies: Zero, Mean of the values for the column, Random value, etc).

# In[ ]:


# Handle missing values for nan_column (TotalCharges)
from sklearn.impute import SimpleImputer

# Find the column number for TotalCharges (starting at 0).
total_charges_idx = df.columns.get_loc("TotalCharges")
imputer = SimpleImputer(missing_values=np.nan, strategy="mean") #SimpleImputer(strategy="most_frequent")

df.iloc[:, total_charges_idx] = imputer.fit_transform(df.iloc[:, total_charges_idx].values.reshape(-1, 1))
df.iloc[:, total_charges_idx] = pd.Series(df.iloc[:, total_charges_idx])


# In[ ]:


# Validate that we have addressed any NaN values
print(df.isnull().values.any())
df.isnull().sum()


# 
# ### 2.5 Categorize Features
# 
# Some of the columns / features are categorized based on wether they are categorical values or continuous (i.e numerical) values which will be used later to build visualizations.

# In[ ]:


columns_idx = np.s_[0:] # Slice of first row(header) with all columns.
first_record_idx = np.s_[0] # Index of first record

string_fields = [type(fld) is str for fld in df.iloc[first_record_idx, columns_idx]] # All string fields
all_features = [x for x in df.columns if x != 'Churn']
categorical_columns = list(np.array(df.columns)[columns_idx][string_fields])
categorical_features = [x for x in categorical_columns if x != 'Churn']
continuous_features = [x for x in all_features if x not in categorical_features]

#print('All Features: ', all_features)
#print('\nCategorical Features: ', categorical_features)
#print('\nContinuous Features: ', continuous_features)
#print('\nAll Categorical Columns: ', categorical_columns)


# ### 2.6 Visualize data
# 
# Data visualization can be used to find patterns, detect outliers, understand distribution and more. Following graphs can be used such as:
# 
# - Histograms, boxplots, etc: To find distribution / spread of our continuous variables.
# - Bar charts: To show frequency in categorical values.
# 

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="darkgrid")
sns.set_palette("hls", 3)


# To get a high level view of the distribution of `Churn`. What percentage of customer in our dataset are churning vs not churning. 

# In[ ]:


print(df.groupby(['Churn']).size())
churn_plot = sns.countplot(data=df, x='Churn', order=df.Churn.value_counts().index)
plt.ylabel('Count')
for p in churn_plot.patches:
    height = p.get_height()
    churn_plot.text(p.get_x()+p.get_width()/2., height + 1,'{0:.0%}'.format(height/float(len(df))),ha="center") 
plt.show()


# Frequency counts charts can be used to get an understanding of the categorical features relative to `Churn`  
# 
# - It can be observed that for the `gender` feature, We have relatively equal rates of churn by `gender`
# - It can be observed that for the `InternetService` feature, We have higher churn for those that have "Fiber optic" service versus those with "DSL"
# 

# In[ ]:


# Categorical feature count plots
f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15)) = plt.subplots(5, 3, figsize=(20, 20))
ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15 ]

for i in range(len(categorical_features)):
    sns.countplot(x = categorical_features[i], hue="Churn", data=df, ax=ax[i])


# Histrogram charts can be used to get an understanding of the distribution of our continuous / numerical features relative to Churn.
# 
# - It can be observed that for the `MonthlyCharges` feature, customers that churn tend to pay higher monthly fees than those that stay.
# - It can be observed that for the `tenure` feature, customers that churn tend to be relatively new customers.

# In[ ]:


# Continuous feature histograms.
fig, ax = plt.subplots(2, 2, figsize=(28, 8))
df[df.Churn == 'No'][continuous_features].hist(bins=20, color="blue", alpha=0.5, ax=ax)
df[df.Churn == 'Yes'][continuous_features].hist(bins=20, color="orange", alpha=0.5, ax=ax)

# Or use displots
#sns.set_palette("hls", 3)
#f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 25))
#ax = [ax1, ax2, ax3, ax4]
#for i in range(len(continuous_features)):
#    sns.distplot(df[continuous_features[i]], bins=20, hist=True, ax=ax[i])


# In[ ]:


# Create Grid for pairwise relationships
gr = sns.PairGrid(df, height=5, hue="Churn")
gr = gr.map_diag(plt.hist)
gr = gr.map_offdiag(plt.scatter)
gr = gr.add_legend()


# In[ ]:


# Plot boxplots of numerical columns. More variation in the boxplot implies higher significance. 
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 25))
ax = [ax1, ax2, ax3, ax4]

for i in range(len(continuous_features)):
    sns.boxplot(x = 'Churn', y = continuous_features[i], data=df, ax=ax[i])


# ## 3.0 Create a model
# 
# Now machine learning model can be created. You could use the insights / intuition gained from the data visualization steps above to what kind of model to create or which features to use. For now, I am creating a simple classification model by removing some of the uninformative features such as gender, StreamingTV and StreamingMovies

# In[ ]:


df.columns


# ### 3.1 Split the data into training and test sets

# In[ ]:



from sklearn.model_selection import train_test_split
X = df[['SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']]
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=42)


# In[ ]:


one_hot_encoded_training_predictors = pd.get_dummies(X_train)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=42)
rfc.fit(one_hot_encoded_training_predictors, y_train)


# In[ ]:


model = rfc.fit(one_hot_encoded_training_predictors, y_train)


# In[ ]:


test_data = pd.get_dummies(X_test)
model.score(test_data, y_test)


# The simple random forest classifier gives an accuracy of 79%. The best metric for evaluating a binary classifier is ROC AUC(Area Under the curve)

# In[ ]:


from sklearn.metrics import roc_auc_score
y_scores = model.predict(test_data)
y_scores = pd.get_dummies(y_scores)
y_test = pd.get_dummies(y_test)
roc_auc_score(y_test, y_scores)

