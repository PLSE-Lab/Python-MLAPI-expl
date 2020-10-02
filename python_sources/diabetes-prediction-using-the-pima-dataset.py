#!/usr/bin/env python
# coding: utf-8

# # Diabetes Prediction
# A machine learning model to accurately classify whether or not the patients in the dataset have diabetes or not.<br><br>
# * Ayush Yadav ( IMT2017009 )
# * Kaustubh Nair ( IMT2017025 )
# * Sarthak Khoche ( IMT2017038 )
# 
# ## Overview: 
# 1. [**Feature Extraction**](#feature)
# 2. [**Missing Data Handling**](#missing_data)
# 3. [**Data Preprocessing**](#preprocessing)
# 4. [**Approach 1 (Using PCA)**](#1)
#   1. [**Exploratory Data Analysis**](#1_eda)
#   2. [**PCA**](#1_pca)
#   3. [**Model building**](#1_model)
# 5. [**Approach 2 (Using Regression Imputation)**](#2)
#   1. [**Exploratory Data Analysis**](#2_eda)
#   2. [**Model building**](#2_model)
# 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import Markdown as md
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('float_format', '{:f}'.format)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:



df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
df.describe()


# <a id='feature'></a>
# ## Feature extraction 
# Since all features are relevent and valid, all columns were used for data preprocessing and considered for feature selection.

# In[ ]:


features = ['Pregnancies','Glucose', 'BloodPressure','SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


# <a id='missing_data'></a>
# ## Missing Data Handling
# Since the given data set has many missing values (i.e NaN) and many 0 values for columns which should not be 0, this data had to either be removed or imputed with relevant and useful values. Imputing values rather than deletion of rows was considered because of the sheer amoount of rows lost if these columns were removed (~340 rows containing NaN and 0 values)

# Print number of null values in the columns

# In[ ]:


# number of null values in each column
df.isnull().sum()


# Assuming the distribution to be a gaussian, imputed data was sampled from "mean +- std.dev" for each feature. 

# In[ ]:


for feature in features:
    number = np.random.normal(df[feature].mean(), df[feature].std()/2)
    df[feature].fillna(value=number, inplace=True)


# Verify that there are no null values in any of the columns.

# In[ ]:


# number of null values in each column
df.isnull().sum()


# <a id='preprocessing'></a>
# ## Data Preprocessing
# All outliers were were replaced with appropriate values as shown below.

# **Dealing with negative values:** <br>Print number of negative values in each column <br> Since none of these features can be negative, they must be invalid and must be replaced. Since the number of negative values is low, we replace them with 0.

# In[ ]:


df.where( df < 0).count()


# In[ ]:


for feature in features:
    df.loc[df[feature] < 0, feature] = 0


# Verify that there are no more negative values in the columns

# In[ ]:


df.where( df < 0).count()


# **Dealing with outlier values:**
# <br> Insulin range is between:  
# * 16-166 mIU/L for non-diabetic people
# * 166 - 260 mIU/L for high-risk diabetic people
# * 260 - 300 mIU/L for diabetic people
# Insulin values greater than 300 mIU/L are highly unlikely.<br>Hence all insulin values in the dataset are capped to 300.<br>
# [Reference](https://www.mayoclinic.org/diseases-conditions/diabetes/diagnosis-treatment/drc-20371451) 

# In[ ]:


df.loc[df['Insulin'] > 300].Insulin.count()


# In[ ]:


df.loc[df.Insulin > 300, 'Insulin'] = 300


# In[ ]:


df.loc[df['Insulin'] > 300].Insulin.count()


# In[ ]:


df.describe()


# <a id='1'></a>
# # Approach 1 (Using PCA)

# <a id='i_eda'></a>
# ## Exploratory Data Analysis
# From the scatter matrix, we can infer that there is a linear relationship between Insulin and Glucose as well as BMI and SkinThickness. To remove redundant information from the dataset, we applied Principal Component Analysisand limited the number of features to 7

# In[ ]:


plot = scatter_matrix(df, alpha=0.2, figsize=(20, 20))


# <a id='1_pca'></a>
# ## Principal Component Analysis (PCA)

# As it can be seen from the graphs below, none of the 7 principal components have relations. All principal components maximize the spread of data in their dimension, hence maximizing the information in the datapoints.

# In[ ]:


x = df.loc[:, features].values
y = df.loc[:,['Outcome']].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=7)
principal_components = pca.fit_transform(x)
principal_df = pd.DataFrame(data = principal_components, columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', '  principal component 6', 'principal component 7'])

pca_df = pd.concat([principal_df, df[['Outcome']]], axis = 1)


# In[ ]:


pca_df.describe()
plot = scatter_matrix(pca_df, alpha=0.2, figsize=(20, 20))


# The new scatter matrix verifies our hypothesis of a linear relationship between some of the features and now all the components have no relation between each other and are independent.

# <a id='1_model'></a>
# ## Model Building

# The helper function `PCA_split_dataset` is used to shuffle the dataset and split it into training and validation data.

# In[ ]:


def PCA_split_dataset(pca_df):
    pca_df = pca_df.sample(frac=1)
    pca_X = pca_df[pca_df.columns[0:7]]
    pca_y = pca_df[pca_df.columns[7]] 
    return train_test_split(pca_X, pca_y, test_size = 0.20)


# We train the data using multiple algorithm to find the model which gives the best results on this dataset.

# **1. Logistic Regression**

# In[ ]:


lr_accuracy = []
for i in range(500):
    train_X, val_X, train_y, val_y = PCA_split_dataset(pca_df)
    model = LogisticRegression(max_iter=2000, solver='lbfgs',random_state=0)
    model.fit(train_X, train_y)
    accuracy_percent = model.score(val_X, val_y)*100
    lr_accuracy.append(accuracy_percent)


# In[ ]:


lr_average_accuracy = np.mean(lr_accuracy)
print("Accuracy for Logistic Regression\nAverage:",lr_average_accuracy,"\nMaximum:", max(lr_accuracy),"\nMinimum:", min(lr_accuracy))


# **2. SVM**

# In[ ]:


svm_accuracy = []
for i in range(500):
    train_X, val_X, train_y, val_y = PCA_split_dataset(pca_df)
    model = SVC(kernel='linear',random_state=0)
    model.fit(train_X, train_y)
    accuracy_percent = model.score(val_X, val_y)*100
    svm_accuracy.append(accuracy_percent)


# In[ ]:


svm_average_accuracy = np.mean(svm_accuracy)
print("Accuracy using SVM\nAverage:",svm_average_accuracy,"\nMaximum:", max(svm_accuracy),"\nMinimum:", min(svm_accuracy))


# **3. Naive Bayes**

# In[ ]:


nb_accuracy = []
for i in range(500):
    train_X, val_X, train_y, val_y = PCA_split_dataset(pca_df)
    model = GaussianNB()
    model.fit(train_X, train_y)
    accuracy_percent = model.score(val_X, val_y)*100
    nb_accuracy.append(accuracy_percent)


# In[ ]:


nb_average_accuracy = np.mean(nb_accuracy)
print("Accuracy using SVM\nAverage:",nb_average_accuracy,"\nMaximum:", max(nb_accuracy),"\nMinimum:", min(nb_accuracy))


# ### Comparison across different models

# In[ ]:


print("Model\t\t\t Accuracy")
print("Logistic Regression\t",lr_average_accuracy)
print("SVM\t\t\t",svm_average_accuracy)
print("Naive Bayes\t\t",nb_average_accuracy)


# <a id='2'></a>
# # Approach 2 (using Linear regression for imputation)

# **Helper Function -** The function `regress_zero_values` implements Deterministic Regression Imputation for two linearly dependent features, where the `target` contains the values we're imputing. This is done using linear regression on the `feature` and `target` variables.

# In[ ]:


def regress_zero_values(df, feature, target):
    zero_target_data = df[ df[target] == 0 ]
    non_zero_target_data = df[ df[target] != 0]

    train_X = non_zero_target_data[feature].values.reshape(-1,1)
    train_y = non_zero_target_data[target].values.reshape(-1,1)
    val_X = zero_target_data[feature].values.reshape(-1,1)

    model = LinearRegression()
    model.fit(train_X, train_y)
    predicted_y = model.predict(val_X)

    j = 0
    for i in df.index:
        if df.at[i, target] == 0:
            df.at[i, target] = predicted_y[j][0]
            j+=1


# <a id='2_eda'></a>
# ## Exploratory Data Analysis

# In[ ]:


plot = df.plot(x='SkinThickness', y='BMI', style='.')
y_label = plot.set_ylabel('BMI')


# From the plot above, we can infer two things: 
#     1. There is a linear relationship between SkinThickness and BMI. 
#     2. A lot of SkinThickness values are zero but it is not possible for a person's skin thickness to be zero. 
# <br>Hence we use the `regress_zero_values` helper function to impute the zero values of Insulin.

# In[ ]:


regress_zero_values(df, 'BMI', 'SkinThickness')


# In[ ]:


plot = df.plot(x='SkinThickness', y='BMI', style='.')
y_label = plot.set_ylabel('BMI')


# Hence zero values are imputed for SkinThickness

# In[ ]:


plot = df.plot(x='Insulin', y='Glucose', style='.')
y_label = plot.set_ylabel('Glucose')


# From the plot above, we can infer two things: 
#     1. There is a linear relationship between Insulin and Glucose. 
#     2. A lot of insulin values are zero but it is not possible for a person's insulin level to fall below 16 mIU/L. 
# <br>Hence we use the `regress_zero_values` helper function to impute the zero values of Insulin.

# In[ ]:


regress_zero_values(df, 'Glucose', 'Insulin')


# In[ ]:


plot = df.plot(x='Insulin', y='Glucose', style='.')
y_label = plot.set_ylabel('Glucose')


# Hence zero values are imputed for Insulin

# Features with larger scales(i.e range of each feature) skew the model during training, making it suseptible to small changes in those features, hence data normalization was applied to regularize the scales of each feature.

# In[ ]:


for feature in features:
    df[feature] = (df[feature] - df[feature].mean())/(df[feature].std())
df.describe()


# <a id='2_model'></a>
# ## Model Building

# This helper function is used to shuffle the dataset and split it into training and validation data.

# In[ ]:


def split_dataset(df):
    df = df.sample(frac=1)
    X = df[df.columns[0:8]]
    y = df[df.columns[8]] 
    return train_test_split(X, y, test_size = 0.20)


# We train the data using multiple algorithm to find the model which gives the best results on this dataset.

# **1. Logistic Regression**

# In[ ]:


lr_accuracy = []
for i in range(500):
    train_X, val_X, train_y, val_y = split_dataset(df)
    model = LogisticRegression(max_iter=2000, solver='lbfgs',random_state=0)
    model.fit(train_X, train_y)
    accuracy_percent = model.score(val_X, val_y)*100
    lr_accuracy.append(accuracy_percent)


# In[ ]:


lr_average_accuracy = np.mean(lr_accuracy)
print("Accuracy for Logistic Regression\nAverage:",lr_average_accuracy,"\nMaximum:", max(lr_accuracy),"\nMinimum:", min(lr_accuracy))


# **2. SVM**

# In[ ]:


svm_accuracy = []
for i in range(500):
    train_X, val_X, train_y, val_y = split_dataset(df)
    model = SVC(kernel='linear',random_state=0)
    model.fit(train_X, train_y)
    accuracy_percent = model.score(val_X, val_y)*100
    svm_accuracy.append(accuracy_percent)


# In[ ]:


svm_average_accuracy = np.mean(svm_accuracy)
print("Accuracy using SVM\nAverage:",svm_average_accuracy,"\nMaximum:", max(svm_accuracy),"\nMinimum:", min(svm_accuracy))


# **3. Naive Bayes**

# In[ ]:


nb_accuracy = []
for i in range(500):
    train_X, val_X, train_y, val_y = split_dataset(df)
    model = GaussianNB()
    model.fit(train_X, train_y)
    accuracy_percent = model.score(val_X, val_y)*100
    nb_accuracy.append(accuracy_percent)


# In[ ]:


nb_average_accuracy = np.mean(nb_accuracy)
print("Accuracy using SVM\nAverage:",nb_average_accuracy,"\nMaximum:", max(nb_accuracy),"\nMinimum:", min(nb_accuracy))


# ## Comparison across different models

# In[ ]:


print("Model\t\t\t Accuracy")
print("Logistic Regression\t",lr_average_accuracy)
print("SVM\t\t\t",svm_average_accuracy)
print("Naive Bayes\t\t",nb_average_accuracy)


# # Results

# We used the following two preprocessing techniques on our data:
# - PCA : After finding relations between two features in our given data set, we attempted to reduce the number of features to 7, and then applied different models to calculate the average accuracy after randomly spliting the training and testing data in 500 iterations:<br>
#    
#     - Logistic Regression: 77.22727272727273
#     - SVM:			 77.02597402597404
#     - Naive Bayes:		 75.18051948051948
#     
# <br>    
# - Imputation using linear regression: After finding relations between two pairs of features in the given data set, we filled the missing values in these features using a linear regression model. The linear regression model was trained on all other non-missin values of the feature, and was used to fit the values of the missing and 0 valued data. The average accuracy after randomly splitting the training and testing data in 500 iterations was:<br>
#     - Logistic Regression	 76.15584415584415
#     - SVM			 76.34545454545454
#     - Naive Bayes		 75.82467532467534
