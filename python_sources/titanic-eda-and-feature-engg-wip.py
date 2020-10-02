#!/usr/bin/env python
# coding: utf-8

# ## Import Statements

# In[ ]:


import pandas as pd #Data Analysis
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#from sklearn.linear_model import LogisticRegression

from warnings import filterwarnings
filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Load Data

# In[ ]:


df_train=pd.read_csv("../input/titanic/train.csv")
df_test=pd.read_csv("../input/titanic/test.csv")
print("Train Shape:",df_train.shape)
print("Test Shape:",df_test.shape)


# In[ ]:


df_train.head()


# In[ ]:


df_train.describe(include='all')


# In[ ]:


df_train.Sex=df_train.Sex.astype('category')
df_train.Pclass=df_train.Pclass.astype('category')
df_train.Embarked=df_train.Embarked.astype('category')


# In[ ]:


df_train.corrwith(df_train.Survived)


# In[ ]:


df_train.info()


# In[ ]:


df_train.columns


# # Exploratory Data Analysis

# ### Join train and test data

# For performing EDA not involving survival and Feature Engineering, it is better to combine train data without the 'Survived' column and test data.

# In[ ]:


df_train_test=pd.concat([df_train.drop('Survived', axis=1),df_test])
df_train_test.shape


# ### Null Values

# In[ ]:


# Function to plot null values in each column
def na_plotter(df):
    val_na=df.isna().sum().sort_values(ascending=False)
    val_na=round(val_na*100/df_train.shape[0],2)
    val_na=val_na[val_na>0]
    plt.figure(figsize=(8,4))
    plt.title("Percentage of Null Values", fontsize=16)
    sns.barplot(val_na, y=val_na.index, orient='h')
    plt.ylabel("Features")
    plt.xlabel("Percentage Null Values")
    plt.show()


# In[ ]:


# Null values in train data
na_plotter(df_train)


# In[ ]:


# Null values in test data
na_plotter(df_test)


# In[ ]:


# Null values in full data
na_plotter(df_train_test)


# Observation:
# * Around 80% of records in test data and around 35% of records in test data does not have value for Cabin.
# * This might be due to the following reasons: 
# <br>(a) indequate data collection
# <br>(b) Cabin number was not assigned to all passengers, maybe the passengers who travelled in lower classes(dormitories) did not have have cabin. 
# * If (a) is the reason, we can drop the column as most of the records are null.
# * If (b) is the reason, presence of value for 'Cabin' column may have an influence on chance of survival.
# 
# * Age is not avalable for around 20% records in train data and 10% records in test data we will fill the null values with the mean age.
# 
# * Embarked is not availabe for few records in train, which will be filled with the mode of the column.
# 
# * Fare is not available for few records in test data, which will be filled with mean. 

# ## Cabin

# In[ ]:


# New feature CabinBool: CabinBool=1 if Cabin Data Available, else CabinBool=0
df_train["CabinBool"]=0
df_test["CabinBool"]=0
df_train_test["CabinBool"]=0

df_train.CabinBool[df_train.Cabin.isna()==False]=1
df_train_test.CabinBool[df_train_test.Cabin.isna()==False]=1


# ### Cabin bool Vs Survival

# In[ ]:


#Count Plot Cabin Bool Vs Survival
plt.figure(figsize=(6,4))
plt.title("Cabin Bool Vs Survival", fontsize=16)
sns.countplot(df_train.CabinBool, hue=df_train.Survived)
plt.show()


# In[ ]:


#Pie Chart Cabin Bool Vs Survival
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Cabin data Available", fontsize=16)
x=df_train.Survived[df_train.CabinBool==1].value_counts()
label=["Survived", "Deceased"]
plt.pie([x[1],x[0]], labels=label, autopct='%1.1f%%')
#plt.show()

plt.subplot(1,2,2)
plt.title("Cabin Data Not Available", fontsize=16)
x=df_train.Survived[df_train.CabinBool==0].value_counts()
label=["Survived", "Deceased"]
plt.pie([x[1],x[0]], labels=label, autopct='%1.1f%%')
plt.show()


# Observation:
# * Passengers with cabin data available have more chance of survival

# ### CabinBool Vs Political Class

# In[ ]:


#Countplot Cabin Bool Vs Political Class
plt.figure(figsize=(6,4))
plt.title("Cabin Bool Vs Political Class", fontsize=16)
sns.countplot('Pclass', hue='CabinBool', data=df_train_test)
plt.show()


# In[ ]:


#Pie Chart Cabin Bool Vs Political Class
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Cabin data Available", fontsize=16)
x=df_train_test.Pclass[df_train.CabinBool==1].value_counts()
label=["Upper Class", "Middle Class", "Lower Class"]
plt.pie([x[1],x[2],x[3]], labels=label, autopct='%1.1f%%')

plt.subplot(1,2,2)
plt.title("Cabin Data Not Available", fontsize=16)
x=df_train_test.Pclass[df_train.CabinBool==0].value_counts()
label=["Upper Class", "Middle Class", "Lower Class"]
plt.pie([x[1],x[2],x[3]], labels=label, autopct='%1.1f%%')
plt.show()


# Observation:
# * Cabin data is availability is highest for Upper Class, followed by middle class and is lowest for lower class.

# ### CabinBool Vs Gender

# In[ ]:


#Pie Chart CabinBool Vs Gender
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Male Travelers", fontsize=16)
x=df_train.CabinBool[df_train.Sex=='male'].value_counts()
label=["Cabin NA", "Cabin Available"]
plt.pie([x[0],x[1]], labels=label, autopct='%1.1f%%')

plt.subplot(1,2,2)
plt.title("Female Travelers", fontsize=16)
x=df_train.CabinBool[df_train.Sex=='female'].value_counts()
label=["Cabin NA", "Cabin Available"]
plt.pie([x[0],x[1]], labels=label, autopct='%1.1f%%')
plt.show()


# ### Age Vs Cabin Bool

# In[ ]:


plt.figure(figsize=(8,4))
sns.boxplot('Age','CabinBool', data=df_train_test, orient='h')
plt.show()


# In[ ]:


#BoxPlot Age Vs CabinBool
temp_df=df_train_test[df_train_test.Age.isna()==False]
plt.figure(figsize=(8,4))
sns.distplot(temp_df.Age[temp_df.CabinBool==0], label="Cabin NA")
sns.distplot(temp_df.Age[temp_df.CabinBool==1], label="Cabin Avail")
plt.xlim(0,100)
plt.show()


# ### CabinBool vs Fare

# In[ ]:


plt.figure(figsize=(8,4))
sns.boxplot('Fare','CabinBool', data=df_train_test, orient='h')
plt.show()


# In[ ]:


print("Fare where Cabin Bool is available:")
print("Mean : ", round(df_train_test[df_train_test.CabinBool==1].Fare.mean(),2))
print("Median : ", df_train_test[df_train_test.CabinBool==1].Fare.median())

print("Fare where Cabin Bool is not available:")
print("Mean : ", round(df_train_test[df_train_test.CabinBool==0].Fare.mean(),2))
print("Median : ", df_train_test[df_train_test.CabinBool==0].Fare.median())


# Observation: 
# * Passengers with cabin data available have paid considerably more that passenger with cabin data not available. 
# * This strengthens the assumption that cabin data is available if passenger is traveling in a cabin and not because of inadequacy in data collection.
# * A large number of outliers are present in where cabin bool is not available, further scrutiny is required.
# * Due to the effect of outliers, mean of the Fares have moved towards the higher side, it is safe to use median value of fares. 

# <<<<<<< * High Fare where Cabin data is not available to be examined *>>>>>>>>
#     

# ## Is it the Lone wolf or the Wolf Pack that survives?

# * Parch : Number of Parents/Children on board.
# * SibSp : Number of Siblings/Spouse(s?) on board.

# In[ ]:


#Count plot SibSp Vs Survival
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("SibSp Vs Survival", fontsize=16)
sns.countplot(df_train.SibSp, hue=df_train.Survived)

#Count plot Parch Vs Survival
plt.subplot(1,2,2)
plt.title("Parch Vs Survival", fontsize=16)
sns.countplot(df_train.Parch, hue=df_train.Survived)
plt.show()


# In[ ]:


# New feature SibSpBool: SibSpBool=1 if traveling with sibling/spouse, else SibSpBool=0 
df_train["SibSpBool"]=0
df_train_test["SibSpBool"]=0
df_train.SibSpBool[df_train.SibSp!=0]=1
df_train_test.SibSpBool[df_train_test.SibSp!=0]=1
# New feature ParchBool: ParchBool=1 if traveling with Parent/Child, else SibSpBool=0
df_train["ParchBool"]=0
df_train_test["ParchBool"]=0
df_train.ParchBool[df_train.Parch!=0]=1
df_train_test.ParchBool[df_train_test.Parch!=0]=1


# In[ ]:


#Count plot SibSpBool Vs Survival
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("SibSpBool Vs Survival", fontsize=16)
sns.countplot(df_train.SibSpBool, hue=df_train.Survived)

#Count plot ParchBool Vs Survival
plt.subplot(1,2,2)
plt.title("ParchBool Vs Survival", fontsize=16)
sns.countplot(df_train.ParchBool, hue=df_train.Survived)
plt.show()


# In[ ]:


# New Feature FamilySize: sum of Parch and SibSp
df_train["FamilySize"]=df_train.Parch+df_train.SibSp
df_train_test["FamilySize"]=df_train_test.Parch+df_train_test.SibSp

#New Feature FamBool: 1 if FamilySize>0, 0 if FamilySize=0
df_train["FamBool"]=0
df_train_test["FamBool"]=0
df_train.FamBool[df_train.FamilySize!=0]=1
df_train_test.FamBool[df_train_test.FamilySize!=0]=1


# In[ ]:


#Count plot FamilySize and FamBool Vs Survival 
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("FamilySize Vs Survival", fontsize=16)
sns.countplot(df_train.FamilySize, hue=df_train.Survived)

plt.subplot(1,2,2)
plt.title("FamBool Vs Survival", fontsize=16)
sns.countplot(df_train.FamBool, hue=df_train.Survived)
plt.show()


# In[ ]:


#Pie Chart Traveling with Family Vs Survival
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Traveling with Family", fontsize=16)
x=df_train.Survived[df_train.FamBool==1].value_counts()
label=["Survived", "Deceased"]
plt.pie([x[1],x[0]], labels=label, autopct='%1.1f%%')
#plt.show()

plt.subplot(1,2,2)
plt.title("Traveling Solo", fontsize=16)
x=df_train.Survived[df_train.FamBool==0].value_counts()
label=["Survived", "Deceased"]
plt.pie([x[1],x[0]], labels=label, autopct='%1.1f%%')
plt.show()


# It is observed that passengers with family-members is more likely to survive.

# In[ ]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("SipSpBool Vs Political Class", fontsize=16)
sns.countplot('SibSpBool', hue='Pclass', data=df_train_test)

plt.subplot(1,2,2)
plt.title("ParchBool Vs Political Class", fontsize=16)
sns.countplot('ParchBool', hue='Pclass', data=df_train_test)
plt.show()


# In[ ]:


print("Percentage of Passengers Traveling with Sibling/Spouse")
print(f"Upper Class: {round(df_train_test[df_train_test.Pclass==1].SibSpBool.mean()*100,2)}%")
print(f"Middle Class: {round(df_train_test[df_train_test.Pclass==2].SibSpBool.mean()*100,2)}%")
print(f"Lower Class: {round(df_train_test[df_train_test.Pclass==3].SibSpBool.mean()*100,2)}%")

print("Percentage of Passengers Traveling with Parents/Children")
print(f"Upper Class: {round(df_train_test[df_train_test.Pclass==1].ParchBool.mean()*100,2)}%")
print(f"Middle Class: {round(df_train_test[df_train_test.Pclass==2].ParchBool.mean()*100,2)}%")
print(f"Lower Class: {round(df_train_test[df_train_test.Pclass==3].ParchBool.mean()*100,2)}%")


# In[ ]:


# Countplot Family vs Political Class
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("FamBool Vs Political Class", fontsize=16)
sns.countplot('FamBool', hue='Pclass', data=df_train_test)

plt.subplot(1,2,2)
plt.title("FamilySize Vs Political Class", fontsize=16)
sns.countplot('FamilySize', hue='Pclass', data=df_train_test)
plt.show()


# In[ ]:


print("Percentage of Passengers Traveling with Family")
print(f"Upper Class: {round(df_train_test[df_train_test.Pclass==1].FamBool.mean()*100,2)}%")
print(f"Middle Class: {round(df_train_test[df_train_test.Pclass==2].FamBool.mean()*100,2)}%")
print(f"Lower Class: {round(df_train_test[df_train_test.Pclass==3].FamBool.mean()*100,2)}%")


# In[ ]:


#Pie Chart Traveling with Family Vs Political Class
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Upper Class", fontsize=16)
x=df_train.FamBool[df_train.Pclass==1].value_counts()
label=["Traveling Solo", "With Family"]
plt.pie([x[0],x[1]], labels=label, autopct='%1.1f%%')

plt.subplot(1,3,2)
plt.title("Middle Class", fontsize=16)
x=df_train.FamBool[df_train.Pclass==2].value_counts()
label=["Traveling Solo", "With Family"]
plt.pie([x[0],x[1]], labels=label, autopct='%1.1f%%')

plt.subplot(1,3,3)
plt.title("Lower Class", fontsize=16)
x=df_train.FamBool[df_train.Pclass==3].value_counts()
label=["Traveling Solo", "With Family"]
plt.pie([x[0],x[1]], labels=label, autopct='%1.1f%%')

plt.show()


# ### FamilyBool Vs Gender

# In[ ]:


#Pie Chart Traveling with Family Vs Gender
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Male Travelers", fontsize=16)
x=df_train.FamBool[df_train.Sex=='male'].value_counts()
label=["Solo Traveler", "With Family"]
plt.pie([x[0],x[1]], labels=label, autopct='%1.1f%%')

plt.subplot(1,2,2)
plt.title("Female Travelers", fontsize=16)
x=df_train.FamBool[df_train.Sex=='female'].value_counts()
label=["Solo Traveler", "With Family"]
plt.pie([x[0],x[1]], labels=label, autopct='%1.1f%%')
plt.show()


# ### Age vs FamBool

# In[ ]:


#BoxPlot Age Vs Fare
temp_df=df_train_test[df_train_test.Age.isna()==False]
plt.figure(figsize=(8,4))
sns.distplot(temp_df.Age[temp_df.FamBool==0], label="Solo")
sns.distplot(temp_df.Age[temp_df.FamBool==1], label="Family")
plt.xlim(0,100)
plt.show()


# Observation:
# * Majority of the solo travellers are of the age group 15-45. Able-bodied working age population.
# * Family travelers is more evenly distributed including children, young and the elderly.

# ### FamilyBool Vs Fare

# In[ ]:


#BoxPlot FamBool Vs Fare
plt.figure(figsize=(8,4))
sns.boxplot('Fare','FamBool', data=df_train_test, orient='h')
plt.show()


# In[ ]:


print("Fare where FamBool is available:")
print("Mean : ", round(df_train_test[df_train_test.FamBool==1].Fare.mean(),2))
print("Median : ", df_train_test[df_train_test.FamBool==1].Fare.median())

print("Fare where FamBool is not available:")
print("Mean : ", round(df_train_test[df_train_test.FamBool==0].Fare.mean(),2))
print("Median : ", df_train_test[df_train_test.FamBool==0].Fare.median())


# Observation: 
# * Passengers with family have paid considerably than the solo travellers. 
# * A large number of outliers are present in where FamBool is not available, further scrutiny is required.
# * Due to the effect of outliers, mean of the Fares have moved towards the higher side, it is safe to use median value of fares. 

# ## Port of Embarkation Vs Survival

# Passengers embarked on Titanic from three ports 
# * S: Southampton, United Kingdom
# * C: Cherbourg, France
# * Q: Queenstown, Ireland

# In[ ]:


print("Unique Embarked in Train : ",df_train.Embarked.unique())
print("Unique Embarked in Train : ",df_test.Embarked.unique())


# In[ ]:


print("Number of Null values for Embarked in Train data : ",df_train.Embarked.isna().sum())


# Since the number of null values is very small, we will fill the NA value with mode of the data.

# In[ ]:


df_train_test.Embarked.fillna(df_train_test.Embarked.mode()[0], inplace=True)
df_train_test.isna().sum()


# In[ ]:


plt.figure(figsize=(6,4))
plt.title("Port of Embarkation Vs Survival", fontsize=16)
sns.countplot('Embarked', hue='Survived', data=df_train)
plt.show()


# In[ ]:


sns.boxplot(df_train.Age, y=df_train.Sex)


# In[ ]:


sns.boxplot(df_train.Survived, df_train.Age, hue=df_train.Sex)


# In[ ]:


sns.boxplot(df_train.Sex,df_train.Survived)


# In[ ]:


sns.countplot(df_train.Sex, hue=df_train.Survived)


# # To Be Continued...

# In[ ]:




