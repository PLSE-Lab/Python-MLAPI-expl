#!/usr/bin/env python
# coding: utf-8

# ## **1. Load the libraries**

# In[ ]:


# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings    
import os

# Ignore warnings
warnings.filterwarnings("ignore")   


# ## **2. Load, Evaluate and Cleanse the Dataset**

# In[ ]:


# Load the dataset
ibm_data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv', header=0)
ibm_data.head(5)


# In[ ]:


# Check if there are any NaN values in the dataset
ibm_data.isnull().any().any()


# **Conclusion** - There is NO missing data in the dataset.

# In[ ]:


# Analyze the unique values in the Dataset

# Get the column names of the dataset
cols = ibm_data.columns
# Print the uniques values of each of the columns in the dataset
for col in cols:
    print(col, "   ::  [", np.unique(ibm_data[col]).size,"]")


# **Conclusion** - Columns 'EmployeeCount', 'Over18' & 'StandardHours' have only one Unique values each. They do not have ANY impoact on the outcome.

# In[ ]:


# Remove the irrelevant columns from the dataset
ibm_data = ibm_data.drop(['EmployeeCount','Over18','StandardHours'], axis=1)


# ## **3. Check the Correlation of Attrition with rest of the data**

# In[ ]:


# Convert Factors into Integer values to draw correlation and some charts...
# Not sure if there is a library or alternate available like Caret in R for achieving this.
ibm_data_num = ibm_data
ibm_data_num=ibm_data_num.replace({'Attrition':{'No':0,'Yes':1}})
ibm_data_num=ibm_data_num.replace({'Gender':{'Male':0,'Female':1}})
ibm_data_num=ibm_data_num.replace({'BusinessTravel':{'Non-Travel':0,'Travel_Rarely':1,'Travel_Frequently':2}})
ibm_data_num=ibm_data_num.replace({'Department':{'Human Resources':0,'Research & Development':1,'Sales':2}})
ibm_data_num=ibm_data_num.replace({'EducationField':{'Human Resources':0,'Life Sciences':1,'Marketing':2,'Medical':3,'Technical Degree':4,'Other':5}})
ibm_data_num=ibm_data_num.replace({'JobRole':{'Human Resources':0,'Healthcare Representative':1,'Laboratory Technician':2,'Manager':3,'Manufacturing Director':4,'Research Director':5,'Research Scientist':6,'Sales Executive':7,'Sales Representative':8}})
ibm_data_num=ibm_data_num.replace({'MaritalStatus':{'Single':0,'Married':1,'Divorced':2}}) 
ibm_data_num=ibm_data_num.replace({'OverTime':{'No':0,'Yes':1}})


# In[ ]:


# Plot the HeatMap for the correlation of the columns

sns.set(style="white")

# Compute the correlation data based on the numerized dataset
ibm_data_corr = ibm_data_num.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(ibm_data_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(ibm_data_corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})


# **Conclusion** - The Attrition does NOT have a strong co-relation with any of the other dimensions.

# ## **4. Additional Graphical Analysis******

# In[ ]:


# Building a dataset for further charting considering only the attributes that have some kind of correlation with Attrition data
ibm_data_temp = ibm_data_num[['Attrition', 'Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'MaritalStatus','MonthlyIncome', 'OverTime', 'StockOptionLevel', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager']]
ibm_data_temp.head(3)


# In[ ]:


row = 5
column = 3
x = 0
y = 0

fig,ax = plt.subplots(row, column, figsize=(15,20)) 

for i in range(ibm_data_temp.shape[1]):
   
    if ibm_data_temp.columns.values[i] != 'Attrition':
        sns.boxplot(x='Attrition', y=ibm_data_temp.columns.values[i], data=ibm_data_temp, hue='Attrition', ax = ax[x,y])
        y = y + 1
    
    if y == column:
        x = x + 1
        y = 0
        
    if x*y == ibm_data_temp.shape[1] - 1:
        break
    
plt.legend
plt.show()


# **Conclusion** - 
# * More likely to quit from the Distance from home is more than 14kms
# * More likely to quit if Environment Satisfaction is less than 2
# * More likely to quit if the employee's age is less than 30 years
# * More likely to quit at job levels lower than 2
# * More likely to quit if employees are working overtime

# In[ ]:


# Get the subset of data of only Attrition cases
attr_data = ibm_data.query("Attrition == 'Yes'")
attr_data.head(3)


# In[ ]:


fig,ax = plt.subplots(1, 3, figsize=(18,4)) 

sns.countplot(x='BusinessTravel',data=attr_data, ax = ax[0])
sns.countplot(x='Department',data=attr_data, ax = ax[1])
sns.countplot(x='EducationField',data=attr_data, ax = ax[2])


# **Conclusion** - 
# * HR Employees are less likely to quit
# * Employees with non-technical qualification are less likely to quit

# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x='JobRole',data=ibm_data, hue='Attrition')


# **Conclusion** - 
# * Employees in the field of Sales are more likely to quit and more so with the representatives.
# * Rate of Attrition is more the with 'Lab Technician' 

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x='Gender',data=ibm_data, hue='Attrition')


# **Conclusion** - There seems to be NO difference between genders for attrition

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x='MaritalStatus',data=ibm_data, hue='Attrition')


# **Conclusion** - Single Employees are more likely to quit.

# In[ ]:


sns.distplot(ibm_data.query("Attrition == 'Yes'").Age, kde=True, label='Yes Attrition')
sns.distplot(ibm_data.query("Attrition == 'No'").Age, kde=True, label='No Attrition')
plt.title('Age Distribution plot', fontsize=20)  
plt.legend(prop={'size':15}, loc=1)
plt.show()


# **Conclusion** - Employees around 30 years of age are more likely to quit.

# In[ ]:


fig,ax = plt.subplots(1, 2, figsize=(15,4)) 
sns.distplot(attr_data.MonthlyIncome, kde=True, hist=False, label='Montly Income', ax=ax[0])
sns.distplot(attr_data.TotalWorkingYears, kde=True, hist=False, label='Total Expereince', ax=ax[1])
plt.legend(loc=1)
plt.show()


# **Conclusion** - 
# * Employees earning around $2500/ month are more likely to quit.
# * Employees with total experience of 5 to 10 years are more likely to quit

# > ## Thank You!!
