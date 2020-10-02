#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#Importing the data set
data =pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.dtypes


# # Overall Observation:
# * There are 9 varibales(columns)
# * There are 768 observation(rows)
# * There is no missing cell and no duplicate rows
# * There are 8 int type variable and 1 bool type
# 
# Here we have 9 variables in which 8 are independent and 1 is dependent variable. Our dependent variable is **Outcome**

# In[ ]:


data.describe()


# At first we will seprate our dependent and independent variable and then analyze them
# 

# In[ ]:


#x is our features data frame
x = data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction',
         'Age']]
y = data['Outcome']


# In[ ]:


print(y.value_counts())
sns.countplot(y)
plt.xticks(range(len(data['Outcome'].unique())))
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.show()


# **So according to our visualization we have 500 negative cases and 268 positive case of our dependent variable**
# 
# 
# Now lets visualize our independent variables

# In[ ]:


x.describe()


# In[ ]:


#We will first normalize our x data frame so that our all features can come on same scale
data_std = (x - x.mean()) / x.std()
data_std.describe()


# Now compare x.describe() and data_std.describe() we have normalize the data and all are on same ranges so we have normalize our data

# In[ ]:


data_part = pd.concat([y,data_std],axis=1)
data_part.head()


# In[ ]:


data_part = pd.melt(data_part,id_vars = 'Outcome',
                   var_name = 'features',
                   value_name = 'value')
data_part.head()


# In[ ]:


plt.figure(figsize=(18,10))
ax = sns.violinplot(x='features',y='value',hue = 'Outcome',data = data_part,split=True,inner = 'quart')
plt.xticks(rotation = 45)
plt.show(ax)


# In[ ]:


plt.figure(figsize=(18,10))
ax = sns.boxplot(x='features',y='value',hue = 'Outcome',data = data_part)
plt.xticks(rotation = 45)
plt.show(ax)


# As through observation we can see the median of the features Pregnsncies, Glucose, Insuline,BMI, Age are far from eachother so they wil play big role in classification 
# while Blood pressure skin thickness Dibaetes Pedigree Function are apart but still they can be uselful for classification 

# The **Box Plot** and **Violin Plot** are exactly same just violin plot also show the quartiles if you want to also work with quartile Violin plots are best

# Now lets analyze the how data is related to eachother using and after that we will by visualizing every dpendent variable with eachother if our heatmap says true or not 
# 

# In[ ]:


#Now we will analyze x
sns.heatmap(x.corr(),annot =True)
plt.figure(figsize=(100,100))
plt.show()


# the larger values and light color boxes shows data is most corealted and darker ones and values near to 0 repesent variables are not that much corelated

# As we can see the corelation of data from heatmap we will analyze the data by considering a feature at a time 

# #  Relation between independent variable

# # **1. Realtion of pregnancy variable with other independent variable**

# In[ ]:


#Lets visualize age variable
sns.countplot(data['Pregnancies'])
plt.show()


# In[ ]:


preg = pd.concat([y,data_std['Pregnancies']],axis=1)
preg = pd.melt(preg,id_vars='Outcome',var_name='Pregnancies',value_name='values')
preg.head()


# In[ ]:


plt.figure(figsize=(10,10))
sns.swarmplot(x='Pregnancies',y='values',hue='Outcome',data=preg)
plt.show()


# It is observed that are pregnancies variable is increaing the outcome rate of being negative is decreasing so more pregnancies are likely to be positive outcome

# In[ ]:


# Lets visualize the relation between Pregnancy and other variables
fig, axarr = plt.subplots(3, 2, figsize=(12, 12))
plt.xticks(np.arange(max(data['Pregnancies'])+1))
sns.lineplot(x='Pregnancies',y='Glucose',data = x,ax=axarr[0][0]).set(title="Pregnancies And Glucose")
sns.lineplot(x='Pregnancies',y='BloodPressure',data = x,ax=axarr[0][1]).set(title="Pregnancies And Blood Pressure")
sns.lineplot(x='Pregnancies',y='SkinThickness',data = x,ax=axarr[1][0]).set(title="Pregnancies And Skin Thickness")
sns.lineplot(x='Pregnancies',y='Insulin',data = x,ax=axarr[1][1]).set(title="Pregnancies And Insuline")
sns.lineplot(x='Pregnancies',y='BMI',data = x,ax=axarr[2][0]).set(title="Pregnancies And BMI")
sns.lineplot(x='Pregnancies',y='DiabetesPedigreeFunction',data = x,ax=axarr[2][1]).set(title="Pregnancies And Diabetes Pedigree Function")
plt.subplots_adjust(hspace=.9)
plt.show()
sns.lineplot(x='Pregnancies',y='Age',data = x).set(title="Pregnancies And Age")
plt.show()


# **Conclusion:** **Therefore according to analysis it is observed that pregnancy variable is some what corelated to age it the relation is directly proportional but not highly corelated to eachother while nor corelation with other independent variable**
# 
# 
# Therefore our heatmap values for pregnancy variable are visualized rightly

# # **2. Realtion of Glucose variable with other independent variable**

# Now we will visualize glusoce variable with other variables
# At first we will make new dataframe and will sort data ascending wise according to Glucose so its easy for us to analyze the dataset and drop pregnancy column as we have already analyzed it

# In[ ]:


#let see how many distinct values we have in glucose an visualize it
x['Glucose'].unique()


# In[ ]:


plt.figure(figsize=(20, 25))
sns.countplot(data['Glucose'])
plt.show()


# In[ ]:


Glucose = pd.concat([y,data_std['Glucose']],axis=1)
Glucose= pd.melt(Glucose,id_vars='Outcome',var_name='Glucose',value_name='values')
Glucose.head()


# In[ ]:


plt.figure(figsize=(10,10))
sns.swarmplot(x='Glucose',y='values',hue='Outcome',data = Glucose)
plt.show()


# By visualizing the Glucose data we see that as Glucose level is increasing the person person is likely to be positive the outcome will be 1

# In[ ]:


# Lets visualize the relation between Glucose and other variables
fig, axarr = plt.subplots(3, 2, figsize=(12, 12))
sns.scatterplot(x='Glucose',y='BloodPressure',data = x,ax=axarr[0][0]).set(title="Glucose And Blood Pressure")
sns.scatterplot(x='Glucose',y='SkinThickness',data = x,ax=axarr[0][1]).set(title="Glucose And Skin Thickness")
sns.scatterplot(x='Glucose',y='Insulin',data = x,ax=axarr[1][0]).set(title="Glucose And Insuline")
sns.scatterplot(x='Glucose',y='BMI',data = x,ax=axarr[1][1]).set(title="Glucose And BMI")
sns.scatterplot(x='Glucose',y='DiabetesPedigreeFunction',data = x,ax=axarr[2][0]).set(title="Glucose And Diabetes Pedigree Function")
sns.scatterplot(x='Glucose',y='Age',data = x,ax=axarr[2][1]).set(title="Glucose And Age")
plt.subplots_adjust(hspace=.9)
plt.show()


# **Conclusion:** **Therefore according to analysis it is observed that glucose variable is some what corelated to  insuline while not that much corelated with other independent variable**
# 
# 
# Therefore our heatmap values for Glucose variable are visualized rightly

# # **3. Realtion of Blood Pressure variable with other independent variable**

# In[ ]:


x['BloodPressure'].unique()


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x['BloodPressure'])
plt.show()


# In[ ]:


Bp = pd.concat([y,data_std['BloodPressure']],axis=1)
Bp= pd.melt(Bp,id_vars='Outcome',var_name='Bp',value_name='values')
Bp.head()


# In[ ]:


plt.figure(figsize=(10,10))
sns.swarmplot(x='Bp',y='values',hue='Outcome',data=Bp)
plt.show()


# As we can observe through it that the blood pressure variable is not helping to predict outcome as we cant see any relationship between them 

# In[ ]:


# Lets visualize the relation between Blood Pressure and other variables
fig, axarr = plt.subplots(2, 2, figsize=(12, 12))
sns.scatterplot(x='BloodPressure',y='SkinThickness',data = x,ax=axarr[0][0]).set(title="Blood Pressure And Skin Thickness")
sns.scatterplot(x='BloodPressure',y='Insulin',data = x,ax=axarr[0][1]).set(title="Blood Pressure And Insuline")
sns.scatterplot(x='BloodPressure',y='BMI',data = x,ax=axarr[1][0]).set(title="Blood Pressure And BMI")
sns.scatterplot(x='BloodPressure',y='DiabetesPedigreeFunction',data = x,ax=axarr[1][1]).set(title="Blood Pressure And Diabetes Pedigree Function")
plt.subplots_adjust(hspace=.9)
plt.show()
sns.scatterplot(x='BloodPressure',y='Age',data = x).set(title="Blood Pressure And Age")
plt.show()


# **Conclusion:** **Therefore according to analysis it is observed that blood pressure variable is not that much corelated with other independent variable**
# 
# 
# Therefore our heatmap values for Blood Pressure variable are visualized rightly

# # **4. Realtion of Skin Thickness variable with other independent variable**

# In[ ]:


#Lets visualize the count plot of skin Thickness
plt.figure(figsize=(10,10))
sns.countplot(x['SkinThickness'])
plt.show()


# In[ ]:


St = pd.concat([y,data_std['SkinThickness']],axis = 1)
St =pd.melt(St,id_vars = 'Outcome',var_name ='SkinThickness',value_name='values')
St.head()


# In[ ]:


plt.figure(figsize=(10,10))
sns.swarmplot(x='SkinThickness',y='values',hue='Outcome',data = St)
plt.show()


# As we can see the value of Outcome is becoming positive as the skin thickness increases therefore skin thickness can have impact on Outcome

# In[ ]:


# Lets visualize the relation between Skin Thickness and other variables
fig, axarr = plt.subplots(2, 2, figsize=(12, 12))
sns.scatterplot(x='SkinThickness',y='Insulin',data = x,ax=axarr[0][0]).set(title="Skin Thickness And Insulin")
sns.scatterplot(x='SkinThickness',y='BMI',data = x,ax=axarr[0][1]).set(title="Skin Thickness And BMI")
sns.scatterplot(x='SkinThickness',y='DiabetesPedigreeFunction',data = x,ax=axarr[1][0]).set(title="Skin Thickness And Diabetes Pedigree Function")
sns.lineplot(x='SkinThickness',y='Age',data = x,ax=axarr[1][1]).set(title="Skin Thickness And Age")
plt.subplots_adjust(hspace=.9)
plt.show()


# **Conclusion:** **Therefore according to analysis it is observed that skin thickness variable is some what corelated to  insuline and bmi while not that much corelated with other independent variable**
# 
# 
# Therefore our heatmap values for Skin thickness variable are visualized rightly

# # **5. Realtion of Insulin variable with other independent variable**

# In[ ]:


x['Insulin'].sort_values().value_counts()


# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(x['Insulin'].sort_values())
plt.show()


# In[ ]:


insulin = pd.concat([y,data_std['Insulin']],axis = 1)
insulin = pd.melt(insulin,id_vars='Outcome',var_name='insulin',value_name='value')


# In[ ]:


plt.figure(figsize=(10,10))
sns.swarmplot(x='insulin',y='value',hue='Outcome',data = insulin)
plt.show()


# As we can see the value of Outcome is becoming positive as the Insulin increases therefore insulin can have impact on Outcome

# In[ ]:


# Lets visualize the relation between Insulin and other variables
sns.scatterplot(x='Insulin',y='BMI',data = x).set(title="Insulin And BMI")
plt.show()
sns.scatterplot(x='Insulin',y='DiabetesPedigreeFunction',data=x).set(title="Insulin And Diabetes Pedigree Function")
plt.show()
sns.lineplot(x='Insulin',y='Age',data = x).set(title="Insulin And Age")
plt.show()


# **Conclusion:** **Therefore according to analysis it is observed that Insulin variable is some what corelated to  skin thickness(from previous visualization) and bmi while not that much corelated with other independent variable**
# 
# 
# Therefore our heatmap values for Insulin variable are visualized rightly

# # **6. Realtion of BMI variable with other independent variable**

# In[ ]:


x['BMI'].sort_values().value_counts()


# In[ ]:


plt.figure(figsize=(30,10))
sns.countplot(x['BMI'])
plt.show()


# In[ ]:


insulin = pd.concat([y,data_std['BMI']],axis = 1)
insulin = pd.melt(insulin,id_vars='Outcome',var_name='BMI',value_name='value')


# In[ ]:


plt.figure(figsize=(10,10))
sns.swarmplot(x='BMI',y='value',hue='Outcome',data = insulin)
plt.show()


# As we can see the value of Outcome is becoming positive as the BMI increases therefore BMI can have impact on Outcome

# In[ ]:


# Lets visualize the relation between Insulin and other variables
sns.scatterplot(x='BMI',y='DiabetesPedigreeFunction',data=x).set(title="Insulin And Diabetes Pedigree Function")
plt.show()
sns.lineplot(x='BMI',y='Age',data = x).set(title="Insulin And Age")
plt.show()


# **Conclusion:** **Therefore according to analysis it is observed that BMI variable is some what corelated to  skin thickness(from previous visualization) and bmi while not that much corelated with other independent variable**
# 
# 
# Therefore our heatmap values for BMI variable are visualized rightly

# # **7. Realtion of Daibetes Pedigree Function variable with other independent variable**

# In[ ]:


x['DiabetesPedigreeFunction'].sort_values().value_counts()


# In[ ]:


plt.figure(figsize=(30,10))
sns.countplot(x['DiabetesPedigreeFunction'])
plt.show()


# In[ ]:


DPF = pd.concat([y,data_std['DiabetesPedigreeFunction']],axis = 1)
DPF = pd.melt(DPF,id_vars='Outcome',var_name='DPF',value_name='value')


# In[ ]:


plt.figure(figsize=(10,10))
sns.swarmplot(x='DPF',y='value',hue='Outcome',data = DPF)
plt.show()


# As we can see the value of Outcome is becoming positive as the DPF increases therefore DPF can have impact on Outcome

# In[ ]:


# Lets visualize the relation between Insulin and other variables
sns.scatterplot(x='DiabetesPedigreeFunction',y='Age',data=x).set(title="Diabetes Pedigree Function and Age")
plt.show()


# **Conclusion:** **Therefore according to analysis it is observed that DPF variable is not that much corelated with other independent variable**
# 
# 
# Therefore our heatmap values for DPF variable are visualized rightly

# # **6. Realtion of AGE variable**

# In[ ]:


x['Age'].value_counts()


# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x['Age'])
plt.show()


# In[ ]:


# **6. Realtion of BMI variable with other independent variable**


# In[ ]:


Age = pd.concat([y,data_std['Age']],axis = 1)
Age = pd.melt(Age,id_vars='Outcome',var_name='Age',value_name='value')


# In[ ]:


Age.head()


# In[ ]:


plt.figure(figsize=(10,10))
sns.swarmplot(x='Age',y='value',hue='Outcome',data = Age)
plt.show()


# As we can see the value of Outcome is becoming positive as the age increases therefore age can have impact on Outcome

# **Conclusion:** **Therefore according to analysis it is observed that BMI variable is some what corelated to  Pregnancies (from previous visualization) and bmi while not that much corelated with other independent variable**
# 
# 
# Therefore our heatmap values for BMI variable are visualized rightly

# **So according to all analysis all the independent variable accept Blood Pressure and Pregnancy have not much effect on Outcome**

# In[ ]:




