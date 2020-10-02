#!/usr/bin/env python
# coding: utf-8

# Following were the references used to create this kernel: 
# 
# https://www.kaggle.com/tentotheminus9/what-causes-heart-disease-explaining-the-model

# Import the necessary libraries. 

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/heart.csv')


# In[ ]:


data.head(3)


# We first check for any duplcities in the dataset. To examine the duplicated row, we can use :
# data.loc[data.duplicated(),:]

# In[ ]:


# Check for duplicities
data.duplicated().sum()


# Remove the duplicated row, keeping the first occurence and dropping the last one. 

# In[ ]:


data = data.drop_duplicates()


# In[ ]:


data.shape


# Therefore, we are going to be working with 302 observations of 14 variables, one of which is the target variable.
# 
# *target*: 0: 'no', 1: 'yes' ( whether someone has a heart disease or not)

# Brief Discussion about the variables: 
# 
# * age - numeric variable
# * sex - categorical variable
# * cp: chest pain type (4 values) - integer valued variable
# * trestbps: resting blood pressure - numeric variable
# * chol: serum cholestrol - numeric variable
# * fbs: fasting blood sugar > 120 mg/dl - categorical varaible
# * restecg: resting electrocardiographic results - categorical variable 
# * thalach: max hear rate achieved - numeric variable
# * exang: exercise induced angina - categorical variable
# * oldpeak: ST depression induced by exercise relative to rest - numeric variable 
# * slope: slope of peak exercise ST segment - categorical variable
# * ca: no. of major vessels (0-3) colored by flouroscopy - integer valued variable
# * thal: 3= normal, 6= fixed defect, 7= reversable defect - categorical variable 
# * target: 0 = no, 1 = yes - categorical variable
# 

# Having an understanding about the variables, we now check for data types.

# In[ ]:


data.dtypes


# Clearly, we need to change treatment of few variables. We will make a copy of data and make changes to it.

# In[ ]:


df = data


# In[ ]:


# Sex
df['sex'][df['sex']==0] = 'female'
df['sex'][df['sex']==1] = 'male'
# cp: chest pain : name of cp from Reference a)
df['cp'][df['cp'] == 0] = 'typical angina'
df['cp'][df['cp'] == 1] = 'atypical angina'
df['cp'][df['cp'] == 2] = 'non-anginal pain'
df['cp'][df['cp'] == 3] = 'asymptomatic'
# fbs : binary variable
df['fbs'][df['fbs'] == 0] = '<120mg/dl'
df['fbs'][df['fbs'] == 1] = '>120mg/dl' 
# restecg 
df['restecg'][df['restecg'] == 0] = 'normal'
df['restecg'][df['restecg'] == 1] = 'ST-T wave abnormality'
df['restecg'][df['restecg'] == 2] = 'left ventricular hypertophy'
# exang
df['exang'][df['exang'] == 0] = 'no'
df['exang'][df['exang'] == 1] = 'yes'
# slope
df['slope'][df['slope'] == 0] = 'upsloping'
df['slope'][df['slope'] == 1] = 'flat'
df['slope'][df['slope'] == 2] = 'downsloping'
# thal
df['thal'][df['thal'] == 0] = 'unknown'
df['thal'][df['thal'] == 1] = 'normal'
df['thal'][df['thal'] == 2] = 'fixed defect'
df['thal'][df['thal'] == 3] = 'reversible defect'
# target
df['target'][df['target'] == 1 ] = 'yes'
df['target'][df['target'] == 0 ] = 'no'


# In[ ]:


df.dtypes


# Let us separate categorical and numeric variables. 
# 

# In[ ]:


predictors = df.columns[:-1]
num_vars = ['age','trestbps','chol', 'thalach', 'oldpeak', 'ca']
cat_vars = []
for variable in predictors:
    if variable not in num_vars:
        cat_vars.append(variable)
y = df[['target']]
X = df[predictors]


# Split the data frame for train-test

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=111)
X_train.shape # we have 226 training examples


# In[ ]:


training_data = pd.concat([X_train,y_train],axis=1)
validation_data = pd.concat([X_test,y_test],axis=1)


# # Descriptive Statistics and Exploratory Analysis

# ## Univariate Analysis

# In[ ]:


# summary statistics for numeric variables
X_train.describe()


# In[ ]:


# response variable : target
y_train['target'].value_counts()
sns.countplot(x = "target", data = y_train, palette = "hls")


# In[ ]:


# Summary statistics for indepdendent variables 
# Numeric Varibles
# a) Age
plt.figure(figsize = (9,8))
sns.distplot(X_train['age'], color = "g", bins = 10, hist_kws = {'alpha':0.4})


# In[ ]:


# b) trestbps
plt.figure(figsize = (9,8))
sns.distplot(X_train['trestbps'], color = "g", bins = 10, hist_kws = {'alpha':0.4})


# In[ ]:


# c) chol
plt.figure(figsize = (9,8))
sns.distplot(X_train['chol'], color = "g", bins = 16, hist_kws = {'alpha':0.4})


# In[ ]:


# d) thalach
plt.figure(figsize = (9,8))
sns.distplot(X_train['thalach'], color = "g", bins = 10 ,hist_kws = {'alpha':0.4})


# In[ ]:


# e) oldpeak
plt.figure(figsize = (9,8))
sns.distplot(X_train['oldpeak'], color = "g", bins = 10 ,hist_kws = {'alpha':0.4})


# In[ ]:


# f) ca
plt.figure(figsize = (9,8))
sns.distplot(X_train['ca'], color = "g", bins = 10 ,hist_kws = {'alpha':0.4})


# The following code will give countplots of all categorical variables in a loop. 

# In[ ]:


# Categorical Variables 
for i,var in enumerate(X_train[cat_vars]):
#    print("{Variable}:{Counts}".format(Variable = var, Counts = X_train.groupby([var]).size()))
    plt.figure(i,figsize=(8,6))
    sns.countplot(x=var, data=X_train)


# ## Bi-variate Analysis

# In[ ]:


training_data.corr()


# We see that following numeric variables are somewhat highly correlated (reference level is 0.3):
# * age and trestbps
# * age and thalach
# * oldpeak and thalach

# ### response variable w.r.t. numeric variables 
# For correlation of target with numeric variables, we can look at the density plots of numeric variables for both categories.

# ### Age
# It seems people with heart disease (red) have a different distribution than people who didn't have heart disease.

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.kdeplot(training_data['age'][training_data.target == 'yes'],color = 'r',shade=True)
sns.kdeplot(training_data['age'][training_data.target == 'no'],color = 'b',shade= True)
plt.legend(['yes','no'])
plt.title('Density Plot of Age of Patients - with heart disease (red) and without (blue)')
ax.set(xlabel = 'Age')
plt.xlim(20,80)
plt.show()


# ### trestbps: Resting blood pressure 
# The distribution is somewhat same, indicating that there might not be a strong relationship between 'heartDisease' and 'resting B.P.' although this claim has to be statistically evalualted later on. 

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.kdeplot(training_data['trestbps'][training_data.target == 'yes'],color = 'r',shade=True)
sns.kdeplot(training_data['trestbps'][training_data.target == 'no'],color = 'b',shade= True)
plt.legend(['yes','no'])
plt.title('Density Plot of Resting B.P. of Patients - with heart disease (red) and without (blue)')
ax.set(xlabel = 'Resting blood pressure')
plt.xlim(80,250)
plt.show()


# ### chol : cholestrol
# It seems that cholestrol also doesn't have a strong correlation to 'heartDisease'.
# Also, there was an outlier of value 417 for cholestrol but I have not shown it for studying the density plots better.

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.kdeplot(training_data['chol'][training_data.target == 'yes'],color = 'r',shade=True)
sns.kdeplot(training_data['chol'][training_data.target == 'no'],color = 'b',shade= True)
plt.legend(['yes','no'])
plt.title('Density Plot of Cholestrol of Patients - with heart disease (red) and without (blue)')
ax.set(xlabel = 'Cholestrol')
plt.xlim(90,500)
plt.show()


# ### thalach : maximum heart rate achieved
# With this variable, I am unsure what to interpret looking at their density plots. Hopefully, numbers will provide more insight when we discuss it later on. 

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.kdeplot(training_data['thalach'][training_data.target == 'yes'],color = 'r',shade=True)
sns.kdeplot(training_data['thalach'][training_data.target == 'no'],color = 'b',shade= True)
plt.legend(['yes','no'])
plt.title('Density Plot of Max. heart rate achieved of Patients - with heart disease (red) and without (blue)')
ax.set(xlabel = 'Maximum heart rate achieved')
plt.xlim(50,250)
plt.show()


# ### oldpeak : ST depression induced by exercise 
# This variable shows a strong tendency to have high correlation to the response variable.

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.kdeplot(training_data['oldpeak'][training_data.target == 'yes'],color = 'r',shade=True)
sns.kdeplot(training_data['oldpeak'][training_data.target == 'no'],color = 'b',shade= True)
plt.legend(['yes','no'])
plt.title('Density Plot of S.T. depression induced in Patients - with heart disease (red) and without (blue)')
ax.set(xlabel = 'S.T. depression')
plt.xlim(-2,10)
plt.show()


# ### response variable w.r.t. categorical variables 
# For correlation of target with categorical variables, we can look at the bar plots showing percentage of each group for different categories of indepdenent variable.

# In[ ]:


# The following will show distribution of target variable for different groups of indepdendent variable.
for var in cat_vars:
     table=pd.crosstab(training_data[var],training_data['target'])   
     table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)


# ### Interpretation of Crosstab:
# From the plot it seems:
# * sex: Females are more likely to have heart disease than men, however keep in mind that women were a much smaller sample in our dataset. 
# * cp: Except for last category which has very small proportion of patients with heart disease, all other categories have somewhat the same distribution. However, it also was the biggest group for this variable. 

# # Logistic Regression Model
