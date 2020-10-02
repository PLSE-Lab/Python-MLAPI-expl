#!/usr/bin/env python
# coding: utf-8

# Studying the Titanic
# --------------------
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#%matplotlib inline

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())


# In[ ]:


#Passengers survived in each class
#----------------------------------------------------------------------------
#Missing values in Pclass and Survived columns
print ("\n\nCount Missing values in Servived column", train['Survived'].isnull().value_counts())
print ("\n\nCount Missing values in Servived column", train['Pclass'].isnull().value_counts())

survivors = train.groupby('Pclass')['Survived'].agg(sum)
print ("\n\nPassengers survived in each class")
print (survivors)

#Plotting the  number of survivors in different class
fig = plt.figure()
ax = fig.add_subplot(111)
rect = ax.bar(survivors.index.values.tolist(),survivors, color='blue', width=0.5)
ax.set_ylabel('No. of survivors')
ax.set_title('Total number of survivors based on class')
xTickMarks = survivors.index.values.tolist()
ax.set_xticks(survivors.index.values.tolist())
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize=20)
plt.show()


#  - Maximum numbers of survivor are in 1st and 3rd Class

# In[ ]:


#Total Passengers in each class
total_Passenger = train.groupby('Pclass')['PassengerId'].count()
print ("\n\nTotal Passengers survived in each class")
print (total_Passenger)

#Survivors percentage
percent_survivors = survivors/(total_Passenger*1.0)

print ("\n\nPercent Passengers survived in each class")
print (percent_survivors)

#Plotting the  number of survivors in different class
fig = plt.figure()
ax = fig.add_subplot(111)
rect = ax.bar(percent_survivors.index.values.tolist(),percent_survivors, color='blue', width=0.5)
ax.set_ylabel('No. of survivors')
ax.set_title('Total number of survivors based on class')
xTickMarks = percent_survivors.index.values.tolist()
ax.set_xticks(percent_survivors.index.values.tolist())
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize=20)
plt.show()


#  - Survival % is highest in 1st Class ~ 61%
#  - Survival % isin 3rd Class ~ 24%
# 
# This shows clear preference for 1st class passengers.

# In[ ]:


#Passengers survived in each Gender in each class
#----------------------------------------------------------------------------
print ("\n\nCount Missing values in Sex column", train['Sex'].isnull().value_counts())

male_survivors = train[train['Sex'] == 'male'].groupby('Pclass')['Survived'].agg(sum)
print ("\n\nMale Passengers survived in each class")
print (male_survivors)

#Total Passengers in each Gender
total_male_Passenger = train[train['Sex'] == 'male'].groupby('Pclass')['PassengerId'].count()
print ("\n\nTotal Male Passengers survived in each class")
print (total_male_Passenger)

female_survivors = train[train['Sex'] == 'female'].groupby('Pclass')['Survived'].agg(sum)
print ("\n\nFemale Passengers survived in each class")
print (female_survivors)


# In[ ]:


#Plotting the total passengers who survived based on Gender
fig = plt.figure()
ax = fig.add_subplot(111)
index = np.arange(male_survivors.count())
bar_width = 0.35
rect1 = ax.bar(index, male_survivors, bar_width, color='blue', label='Men')
rect2 = ax.bar(index + bar_width, female_survivors, bar_width,color='y', label='Women')
ax.set_ylabel('Survivor Numbers')
ax.set_title('Male and Female survivors based on class')
xTickMarks = male_survivors.index.values.tolist()
ax.set_xticks(index + bar_width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()


#  - The majority of survivors are females in all the classes

# In[ ]:


#Total Passengers in each Gender
total_female_Passenger = train[train['Sex'] == 'female'].groupby('Pclass')['PassengerId'].count()
print ("\n\nTotal Female Passengers survived in each class")
print (total_female_Passenger)

#Survivors percentage  -Male
percent_male_survivors = male_survivors/(total_male_Passenger*1.0)
print ("\n\nPercent male Passengers survived in each class")
print (percent_male_survivors)

#Survivors percentage  -FeMale
percent_female_survivors = female_survivors/(total_female_Passenger*1.0)
print ("\n\nPercent female Passengers survived in each class")
print (percent_female_survivors)


# In[ ]:


#Plotting the percent passengers who survived based on Gender
fig = plt.figure()
ax = fig.add_subplot(111)
index = np.arange(percent_male_survivors.count())
bar_width = 0.35
rect1 = ax.bar(index, percent_male_survivors, bar_width, color='blue', label='Men')
rect2 = ax.bar(index + bar_width, percent_female_survivors, bar_width,color='y', label='Women')
ax.set_ylabel('Survivor Numbers')
ax.set_title('% Male and Female survivors based on class')
xTickMarks = percent_male_survivors.index.values.tolist()
ax.set_xticks(index + bar_width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()


#  - More than 90% of female passengers in first and second class survived
#  - The percentage of male passengers who survived in second and third class,respectively, are comparable

# **What is the distribution of non-survivors among the various classes who have family aboard the ship?**

# In[ ]:


#Checking for the null values
print ( "\n\nCount Missing values in siblings/spouses aboard column", train['SibSp'].isnull().value_counts())
print ( "\n\nCount Missing values in parents/children aboard column",train['Parch'].isnull().value_counts())

non_survivors = train[(train['SibSp'] > 0) | (train['Parch'] > 0) & (train['Survived'] == 0)].groupby('Pclass')['PassengerId'].count()
print ("\n\nTotal non survivors in each class")
print (non_survivors)

#Plotting the percent non Survivor
fig = plt.figure()
ax = fig.add_subplot(111)
rect = ax.bar(non_survivors.index.values.tolist(),non_survivors, color='blue', width=0.5)
ax.set_ylabel('No. of Non survivors')
ax.set_title('Total no. of non survivors based on class')
xTickMarks = non_survivors.index.values.tolist()
ax.set_xticks(non_survivors.index.values.tolist())
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize=20)
plt.show()


#  - There are lot of nonsurvivors in the third class.
#  - Second class has the least number of non survivors with relatives.

# In[ ]:


total_non_survivors = train.groupby('Pclass')['PassengerId'].count()
print ("\n\nTotal Passengers survived in each class")
print (total_non_survivors)

percent_non_survivors = non_survivors/(total_non_survivors*1.0)
print ("\n\nPercent Passengers survived in each class")
print (percent_non_survivors)


# In[ ]:


#Plotting the percent non Survivor
fig = plt.figure()
ax = fig.add_subplot(111)
rect = ax.bar(percent_non_survivors.index.values.tolist(),percent_non_survivors, color='blue', width=0.5)
ax.set_ylabel('% of Non survivors')
ax.set_title('Total % of non survivors based on class')
xTickMarks = percent_non_survivors.index.values.tolist()
ax.set_xticks(percent_non_survivors.index.values.tolist())
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize=20)
plt.show()


#  - With respect to the total number of passengers, the first class, who had relatives aboard, has the maximum non survivor percentage and the third class has the least.

# **What was the survival percentage among different age groups?**

# In[ ]:


#Checking for null values
train['Age'].isnull().value_counts()

#Defining the age binning interval
age_bin = [0, 18, 25, 40, 60, 100]
#Creating the bins
train['AgeBin'] = pd.cut(train.Age, bins=age_bin)
#Removing the null rows
d_temp = train[np.isfinite(train['Age'])] # removing all na instances

#Number of survivors based on Age bin
survivors = d_temp.groupby('AgeBin')['Survived'].agg(sum)
print ("\n\nTotal Passengers survived in each Age Bin")
print (survivors)

#Total passengers in each bin
total_passengers = d_temp.groupby('AgeBin')['Survived'].agg('count')
print ("\n\nTotal Passengers in each Age Bin")
print (total_passengers)


# In[ ]:


#Plotting the pie chart of total passengers in each bin
plt.pie(total_passengers,labels=total_passengers.index.values,autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Total Passengers in different age groups')
plt.show()


# In[ ]:


#Plotting the pie chart of percentage passengers in each bin
plt.pie(survivors, labels=survivors.index.values, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Survivors in different age groups')
plt.show()


# People who were old were either not lucky enough or made way for the younger people to the lifeboats.
