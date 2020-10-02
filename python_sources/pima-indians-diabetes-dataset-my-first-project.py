#!/usr/bin/env python
# coding: utf-8

# # Pima Indians Diabetes Dataset
# 
# **Context**
# This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset
# 

# **Content**
# The datasets consists of several medical predictor variables and one target variable 'Class', Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
# 

# 
# **About the data**
# 
# Numeric : Pregnancies      : Number of times pregnant
#   
# Numeric : Glucose          : Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 
# Numeric : BloodPressure    : Diastolic blood pressure (mm Hg)
# 
# Numeric : SkinThickness    : Triceps skin fold thickness (mm)
# 
# Numeric : Insulin          : 2-Hour serum insulin (mu U/ml)
# 
# Numeric : BMI              : Body mass index (weight in kg/(height in m)^2)
# 
# Numeric : DiabetesPedigree : Diabetes pedigree function
# 
# Numeric : Age              : Age (years)
# 
# Numeric : Outcome          : Class variable (0 or 1)
# 
# 

# **Problem Statement: **

# Can you build a machine learning model to accurately predict whether or not the patients in the dataset have diabetes or not?

# ******************************************************************************************

# # Step 1: Read dataset

# **Import packages**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes = True)


# **Import Data**

# In[ ]:


pima = pd.read_csv('../input/diabetes.csv')


# **Display DataFrame**

# In[ ]:


pima


# **Show first 5 rows or observations**

# In[ ]:


pima.head()


# **Show last 5 rows or observations**

# In[ ]:


pima.tail()


# **Get shape of the DataFrame**

# In[ ]:


pima.shape


# dataset contains 768 rows or observations and 9 columns or variables

# In[ ]:


pima.info()


# Above also shows that 768 entries and 9 columns.
# 
# There are no null values in any of the variable.
# 
# But there are values for some columns which does not make sense; they are marked as a zero (0) value
# For example; a zero value in BloodPressure variable does not make sense. Person would be dead I guess by now

# In[ ]:


pima.describe()


# In[ ]:


pima.describe().T


# this tells me following:
# 
# For example; let us take "Pregnancies" variable
# 1. Count: Total number of entries or rows  = 768 (no missing values)
# 2. Mean: Sum of all values of Pregnancies divide by 768 = 3.845052
# 3. std: Standard Deviation - it measures how far data values are from their mean = 3.369578
# 4. min: Minimum value = 0
# 5. 25%: First Quartile (Q1) = 1
# 6. 50%: Second Quartile (Q2) = 3
# 7. 75%: Third Quartile (Q3) = 6
# 8. max: Maximum value = 17
# 
# Above 5 number summary can be better explained using boxplot

# *********************************************************************************

# We have to predict whether or not the patients in the dataset have diabetes or not?
# 
# So let us check how many people have diabetes and how many of them do not

# In[ ]:


pima['Outcome'].value_counts()


# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(pima['Outcome'])
plt.ylabel('Number of People')


# We can see from above plot that:
# 
# People who do not have diabetes: 500
# 
# People who have diabetes       : 268 

# ************************************************************************************

# # Step 2: Exploratory Data Analysis (EDA)

# **Missing values**

# There are columns or variables that have a minimum value of zero (0). 
# 
# On some columns, a value of zero does not make sense and thus indicates missing value.
# 
# Following columns or variables have an invalid zero value:
# 
# 1. Glucose
# 
# 2. BloodPressure
# 
# 3. SkinThickness
# 
# 4. Insulin
# 
# 5. BMI

# **Count of missing values(zeros) in above mentioned 5 variables or columns**

# In[ ]:


pima[pima['Glucose'] == 0]


# In[ ]:


missingGlucose = pima[pima['Glucose'] == 0].shape[0]
print ("Number of zeros in colum Glucose: ", missingGlucose)


# In[ ]:


missingBP = pima[pima['BloodPressure'] == 0].shape[0]
print ("Number of zeros in colum BloodPressure: ", missingBP)


# In[ ]:


missingST = pima[pima['SkinThickness'] == 0].shape[0]
print ("Number of zeros in colum SkinThickness: ", missingST)


# In[ ]:


missingInsulin = pima[pima['Insulin'] == 0].shape[0]
print ("Number of zeros in colum Insulin: ", missingInsulin)


# In[ ]:


missingBMI = pima[pima['BMI'] == 0].shape[0]
print ("Number of zeros in colum BMI: ", missingBMI)


# **Another method to calculate total number of zeros in a column**

# I will replace zero (0) values with NaN values and then sum the NaN values in each column to know get count of NaN values

# In[ ]:


pima_copy = pima.copy(deep = True)


# In[ ]:


pima_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = pima_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(pima_copy.isnull().sum())


# *************************************************************

# **Visualization for understanding the distribution of data for different columns**

# In[ ]:


pima.hist(figsize = (20,20))


# Columns 'Age', 'DiabetesPedigreeFunction', 'Pregnancies', 'SkinThickness', 'Insulin' are right skewed that is:
# 
# Mean > Median

# *************************************************************************** 

# **DATA CLEANING**

# Let us check how data is distributed for columns or variables that have an invalid zero value
# 
# See if there are any outliers
# 
# See if data is normally distributed, left skewed or right skewed
# 
# We will use boxplot and distplot(Histogram)

# In[ ]:


plt.figure(figsize=(15,6))
sns.distplot(pima['Glucose'], kde = True, rug = True)


# In[ ]:


plt.figure(figsize=(15,6))
sns.boxplot(pima['Glucose'])


# Column 'Glucose' has one outlier

# In[ ]:


plt.figure(figsize=(15,6))
sns.distplot(pima['BloodPressure'], kde = True, rug = True)


# In[ ]:


plt.figure(figsize=(15,6))
sns.boxplot(pima['BloodPressure'])


# Column 'BloodPressure' has very few outliers

# In[ ]:


plt.figure(figsize=(15,6))
sns.distplot(pima['SkinThickness'], kde = True, rug = True)


# In[ ]:


plt.figure(figsize=(15,6))
sns.boxplot(pima['SkinThickness'])


# 'SkinThickness' has 227 zero invalid values that is why lower limit and Q1 (25th) quartile are same

# In[ ]:


plt.figure(figsize=(15,6))
sns.distplot(pima['Insulin'], kde = True, rug = True)


# In[ ]:


plt.figure(figsize=(15,6))
sns.boxplot(pima['Insulin'])


# 'Insulin' has 374 zero invalid values that is why lower limit and Q1 (25th) quartile are same

# In[ ]:


plt.figure(figsize=(15,6))
sns.distplot(pima['BMI'], kde = True, rug = True)


# In[ ]:


plt.figure(figsize=(15,6))
sns.boxplot(pima['BMI'])


# *************************************************************************************

# Replace NaN with Median or Mean

# Variables 'Glucose' and 'BloodPressure' do not have much outliers and we need to fill little data so we will use mean here
# 
# Variables 'SkinThickness', 'Insulin', and 'BMI' have much disparity and we need to fill more data so we will use median here

# In[ ]:


pima_copy['Glucose'].fillna(pima_copy['Glucose'].mean(), inplace = True)


# In[ ]:


pima_copy['BloodPressure'].fillna(pima_copy['BloodPressure'].mean(), inplace = True)


# In[ ]:


pima_copy['SkinThickness'].fillna(pima_copy['SkinThickness'].median(), inplace = True)


# In[ ]:


pima_copy['Insulin'].fillna(pima_copy['Insulin'].median(), inplace = True)


# In[ ]:


pima_copy['BMI'].fillna(pima_copy['BMI'].median(), inplace = True)


# In[ ]:


print(pima_copy.isnull().sum())


# Thus all NaN values are replaced with Mean or Median making data clean

# In[ ]:


pima_copy.describe().T


# ************************************************************************************

# **Visualization**

# Let us perform visulaization on clean data (pima_copy)

# In[ ]:


plt.figure(figsize=(15,6))
sns.distplot(pima_copy['Pregnancies'], bins=20, rug = True)
plt.ylabel('Number of People')


# The above plote tells me Pregnancies per Person

# Let us calculate the average of children had by Pima woman

# In[ ]:


print("Average of children had by Pima woman: ", pima_copy['Pregnancies'].mean())


# In[ ]:


pima_copy['Pregnancies'].median()


# In[ ]:


preg = pima_copy[pima_copy['Pregnancies'] >= 1].shape[0]
print('Number of Pima Woman who had children: ', preg)


# In[ ]:


notPreg = pima_copy[pima_copy['Pregnancies'] == 0].shape[0]
print('Number of Pima woman who did not have children: ', notPreg)


# In[ ]:


pregPlusDiabetes = pima_copy[(pima_copy['Pregnancies'] >= 1) & (pima_copy['Outcome'] == 1)].shape[0]
print('Number of woman who have children and are diabetic: ',pregPlusDiabetes)


# In[ ]:


pregPlusNotDiabetes = pima_copy[(pima_copy['Pregnancies'] >= 1) & (pima_copy['Outcome'] == 0)].shape[0]
print('Number of woman who have children and are not diabetic: ',pregPlusNotDiabetes)


# In[ ]:


notPregPlusDiabetes = pima_copy[(pima_copy['Pregnancies'] == 0) & (pima_copy['Outcome'] == 1)].shape[0]
print('Number of woman who do not have children and are diabetic: ',notPregPlusDiabetes)


# In[ ]:


notPregPlusNotDiabetes = pima_copy[(pima_copy['Pregnancies'] == 0) & (pima_copy['Outcome'] == 0)].shape[0]
print('Number of woman who do not have children and are not diabetic: ',notPregPlusNotDiabetes)


# From above I can say that, Pima women who have children have more possibility of being Diabetic

# ********************************************************************

# In[ ]:


plt.figure(figsize=(15,6))
sns.distplot(pima_copy['Glucose'], bins=20, rug = True)


# Looks like kind of normal distribution

# In[ ]:


plt.figure(figsize=(15,6))
sns.distplot(pima_copy['BloodPressure'], bins=20, rug = True)


# Seems to be a normal distribution

# In[ ]:


plt.figure(figsize=(15,6))
sns.distplot(pima_copy['SkinThickness'], bins=20, rug = True)


# We had 227 zero invalid values in 'SkinThickness' column which we replaced by Median

# In[ ]:


plt.figure(figsize=(15,6))
sns.distplot(pima_copy['Insulin'], bins = 50)


# In[ ]:


plt.figure(figsize=(15,6))
sns.distplot(pima_copy['BMI'], bins = 50)


# It looks like kind of Normal Distribution

# In[ ]:


print('Average BMI: ', pima_copy['BMI'].mean())


# In[ ]:


plt.figure(figsize=(15,6))
sns.distplot(pima_copy['DiabetesPedigreeFunction'], bins = 50)


# In[ ]:


plt.figure(figsize=(15,6))
sns.distplot(pima_copy['Age'], bins = 50)


# In[ ]:


print("Minimum age: ",pima_copy['Age'].min())


# In[ ]:


print("Maximum age: ",pima_copy['Age'].max())


# **********************************************************************

# **Bivariate Analysis**

# In[ ]:


corr = pima_copy.corr()
corr


# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(corr, annot = True, cmap = 'plasma', vmin = -1, vmax = 1)


# **From above heatmap, I can conclude following:**
# 
# 1. There is no feature variable that has strong correlation with target 'Outcome' as there is no +0.70 which indicates a strong uphill (positive) linear relationship
# 
# 
# 2. Best predictor of target variable 'Outcome' is 'Glucose' --> 0.49 which is near to 0.50 which indicates a moderate uphill (positive) relationship
# 
# 
# 3. Second best predictor of target variable 'Outcome' is 'BMI' --> 0.31 which indicates a weak uphill (positive) linear relationship
# 
# 
# 4. Correlation between 'BMI' and 'SkinThickness' is 0.54 which indicates BMI increases with increase in Skin Thickness
# 
# 
# 5. Correlation between 'Age' and 'Pregnancies' is 0.54 which indicates increase in age increases chances of having a child

# In[ ]:


print('Average Glucose for Pima woman who has diabetes: ', pima_copy[pima_copy['Outcome'] == 1]['Glucose'].mean())


# In[ ]:


print('Average Glucose for Pima woman who does not have diabetes: ', pima_copy[pima_copy['Outcome'] == 0]['Glucose'].mean())


# In[ ]:


sns.boxplot(pima['Outcome'],pima['Glucose'])


# Pima woman who have diabetes has higher Glucose levels 
# 
# whereas
# 
# Pima woman who does not have diabetes has lower Glucose levels

# In[ ]:


print('Average BMI for Pima woman who has diabetes: ', pima_copy[pima_copy['Outcome'] == 1]['BMI'].mean())


# In[ ]:


print('Average BMI for Pima woman who does not have diabetes: ', pima_copy[pima_copy['Outcome'] == 0]['BMI'].mean())


# I checked on Internet for BMI scale and here it is: 
# 
# Underweight: BMI is less than 18.5
# 
# Normal weight: BMI is 18.5 to 24.9
# 
# Overweight: BMI is 25 to 29.9
# 
# Obese: BMI is 30 or more

# Thus I can say that Pima women are Obese both who has diabetes and who does not have diabetes as their average is more than 30
# 
# Pima woman who has diabetes have more BMI as comapred to who does not have diabetes

# In[ ]:


plt.figure(figsize=(15,6))
sns.boxplot(pima_copy['Outcome'],pima_copy['Age'])


# In[ ]:


oneOutcome = pima_copy[pima_copy['Outcome'] == 1]
print("Minimum age of Pima woman who has Diabetes: ",oneOutcome['Age'].min())


# In[ ]:


print("Maximum age of Pima woman who has Diabetes: ",oneOutcome['Age'].max())


# In[ ]:


zeroOutcome = pima_copy[pima_copy['Outcome'] == 0]
print("Minimum age of Pima woman who does not have Diabetes: ",zeroOutcome['Age'].min())


# In[ ]:


zeroOutcome = pima_copy[pima_copy['Outcome'] == 0]
print("Maximum age of Pima woman who does not have Diabetes: ",zeroOutcome['Age'].max())


# In[ ]:


print('Average Age of Pima woman who has diabetes: ',pima_copy[pima_copy['Outcome'] == 1]['Age'].mean())


# In[ ]:


print('Average Age of Pima woman who does not have diabetes: ',pima_copy[pima_copy['Outcome'] == 0]['Age'].mean())


# so as age increases the risk of being diabetes also increases

# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(x = 'Pregnancies', hue = 'Outcome', data = pima_copy)


# There is very less correlation between 'Pregnancies' and 'Outcome' that is they are weakly correlated
# 
# Pima woman with less number of children have low chance of diabetes

# In[ ]:


print('Average Skin Thickness of Pima woman who has diabetes: ', pima_copy[pima_copy['Outcome'] == 1]['SkinThickness'].mean())


# In[ ]:


print('Average Skin Thickness of Pima woman who does not have diabetes: ', pima_copy[pima_copy['Outcome'] == 0]['SkinThickness'].mean())


# Since 'SkinThickness' had more than 227 zeros values and we replaced it with Median, data is not so representative.
# But we can say that more the skin thickness more is the probability of getting diabetes
# 

# In[ ]:


print('Average Insulin of Pima woman who has diabetes: ', pima_copy[pima_copy['Outcome'] == 1]['Insulin'].mean())


# In[ ]:


print('Average Insulin of Pima woman who does not have diabetes: ', pima_copy[pima_copy['Outcome'] == 0]['Insulin'].mean())


# Same like 'SkinThickness' we replaced invalid zero minimum value with median so data is not representative. But we can say that higher the insulin higher chances of getting diabetes. 
# 
# Also insulin has a moderate correlation with Glucose

# In[ ]:


sns.pairplot(pima_copy)


# In[ ]:


pima_copy.hist(figsize = (20,20))


# *******************************************************************************

# # Step 3: Build a Model

# Build a model so when a new person walks-in with these values of variables we feed it to the model and model tells us what is the probability of 0 or 1

# Since this is binary classification, I will use Logistic Regression algorithm to predict 0 or 1

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# We will build a model on clean data (pima_copy)

# In[ ]:


x = pima_copy.drop('Outcome', axis  = 1)


# In[ ]:


y = pima_copy['Outcome']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 17)


# In[ ]:


mdl = LogisticRegression()


# In[ ]:


fit = mdl.fit(X_train, y_train)


# In[ ]:


pred = mdl.predict(X_test)


# In[ ]:


confusion = metrics.confusion_matrix(y_test,pred)
confusion


# In[ ]:


label = ["No (0)"," Yes (1)"]
sns.heatmap(confusion, annot=True, xticklabels=label, yticklabels=label)


# A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known.

# So here:
# 
# TN = 132
# 
# TP = 45
# 
# FN = 36
# 
# FP = 18
# 
# 
# This is how I interpret the confusion matrix:
# 0 = No Diabetes
# 1 = Diabetes
# 
# TP: Our model predicted 45 as 1 and actual it was 1 (Model was correct here)
# 
# TN: Our model predicted 132 as 0 and actual it was 0 (Model was correct here)
# 
# FP: Our model predicted 18 as 0 and actual it was 1 (Model was wrong here)
# 
# FN: Our model predicted 36 as 1 and actual it was 0 (Model was wrong here)

# Accuracy = TP+TN/TP+FP+FN+TN
# 
# = (45 + 132) / (45 + 18 + 36 + 132)
# 
# = 177 / 231
# 
# = 0.7662337662337662
# 
# So we can say that our model is 76% accurate

# In[ ]:


metrics.accuracy_score(y_test,pred)


# Precision: Precsion tells us about when model predicts yes, how often is it correct.
# 
# Precision = TP / (TP + FP)
# 
# = 45 / (45 + 18)
# 
# = 45 / 63
# 
# = 0.7142857142857143
# 
# So when our model predict 1 and actual it is 1 then it's precision is 71%

# In[ ]:


metrics.precision_score(y_test,pred)


# *******************************************************************

# We will build a model on original data (pima)

# In[ ]:


x = pima.drop('Outcome', axis  = 1)
y = pima['Outcome']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 17)
mdl = LogisticRegression()
fit = mdl.fit(X_train, y_train)
pred = mdl.predict(X_test)
pred
metrics.accuracy_score(y_test,pred)


# In[ ]:


metrics.precision_score(y_test,pred)


# **************************************************************

# In[ ]:




