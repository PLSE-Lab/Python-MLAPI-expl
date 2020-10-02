#!/usr/bin/env python
# coding: utf-8

# # World Health Organization has estimated 12 million deaths occur worldwide, every year due to Heart diseases.

# The summary of this notebook:
# 
# * Data cleaning.
# 
# * Relationship between education and cigsPerDay,
# 
# * Relationship between age and cigsPerDay, totChol, glucose.
# 
# * Which gender has more risk of coronary heart disease CHD.
# 
# * Which age group has more smokers.
# 
# * Relation between cigsPerDay and risk of coronary heart disease.
# 
# * Relation between sysBP and risk of CHD.
# 
# * Relation between diaBP and risk of CHD.
# 
# * Predicting the risk of CHD with Linear Regression.(85% accuracy)

# In[ ]:


#importing the necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[ ]:


#reading the CSV file

location = "../input/heart-disease-prediction-using-logistic-regression/framingham.csv"
data = pd.read_csv(location)
data


# # Data cleaning

# In[ ]:


#Checking whether the dataset have any NaN.

data.isnull().any()


# In[ ]:


#Looking at the features of the education column:

data["education"].describe()


# In[ ]:


#replacing the NaN's with the mean(rounding up to the nearest whole number)

data["education"] = data["education"].fillna(2)


# In[ ]:


#Looking at the features of the cigsPerDay column:

data["cigsPerDay"].describe()


# In[ ]:


#replacing the NaN's with the mean

data["cigsPerDay"] = data["cigsPerDay"].fillna(data["cigsPerDay"].mean())


# In[ ]:


#Looking at the features of the BPMeds column:

data["BPMeds"].describe()


# In[ ]:


#replacing the NaN's with the mean

data["BPMeds"] = data["BPMeds"].fillna(data["BPMeds"].mean())


# In[ ]:


#Looking at the features of the totChol column:

data["totChol"].describe()


# In[ ]:


#replacing the NaN's with the mean

data["totChol"] = data["totChol"].fillna(data["totChol"].mean())


# In[ ]:


#Looking at the features of the BMI column:

data["BMI"].describe()


# In[ ]:


#replacing the NaN's with the mean

data["BMI"] = data["BMI"].fillna(data["BMI"].mean())


# In[ ]:


#Looking at the features of the heartRate column:

data["heartRate"].describe()


# In[ ]:


#replacing the NaN's with the mean

data["heartRate"] = data["heartRate"].fillna(data["heartRate"].mean())


# In[ ]:


#Looking at the features of the glucose column:

data["glucose"].describe()


# In[ ]:


#replacing the NaN's with the mean

data["glucose"] = data["glucose"].fillna(data["glucose"].mean())


# In[ ]:


#checking for NaN's 

data.isnull().any()


# Now our data is ready for further use: 

# # Relationship between education and cigsPerDay

# In[ ]:


#Grouping education and cigsPerDay

graph_1 = data.groupby("education", as_index=False).cigsPerDay.mean()


# In[ ]:



plt.figure(figsize=(12,8))
sns.regplot(x=graph_1["education"], y=graph_1["cigsPerDay"])
plt.title("Graph showing cigsPerDay in every level of education.")
plt.xlabel("education", size=20)
plt.ylabel("cigsPerDay", size=20)
plt.xticks(size=12)
plt.yticks(size=12)


# There is no such linear relationship found.
# level 3 education shows the lowest mean.

# # Relationship between age and cigsPerDay, totChol, glucose.

# In[ ]:


#Plotting a linegraph to check the relationship between age and cigsPerDay, totChol, glucose.

graph_3 = data.groupby("age").cigsPerDay.mean()
graph_4 = data.groupby("age").totChol.mean()
graph_5 = data.groupby("age").glucose.mean()

plt.figure(figsize=(12,8))
sns.lineplot(data=graph_3, label="cigsPerDay")
sns.lineplot(data=graph_4, label="totChol")
sns.lineplot(data=graph_5, label="glucose")
plt.title("Graph showing totChol and cigsPerDay in every age group.")
plt.xlabel("age", size=20)
plt.ylabel("count", size=20)
plt.xticks(size=12)
plt.yticks(size=12)


# We see a minor relation between totChol and glucose.

# # Which gender has more risk of coronary heart disease CHD

# In[ ]:


#checking for which gender has more risk of coronary heart disease CHD

graph_6 = data.groupby("male", as_index=False).TenYearCHD.sum()


# In[ ]:


#Ploting the above values

plt.figure(figsize=(12,8))
sns.barplot(x=graph_6["male"], y=graph_6["TenYearCHD"])
plt.title("Graph showing which gender has more risk of coronary heart disease CHD")
plt.xlabel("0 is female and 1 is male",size=20)
plt.ylabel("total cases", size=20)
plt.xticks(size=12)
plt.yticks(size=12)


# According to this dataset, males have slighly higher risk of coronary heart disease CHD.

# # Which age group has more smokers.

# In[ ]:


#grouping the necessary features

graph_7 = data.groupby("age",as_index=False).currentSmoker.sum()

plt.figure(figsize=(12,8))
sns.barplot(x=graph_7["age"], y=graph_7["currentSmoker"])
plt.title("Graph showing which age group has more smokers.")
plt.xlabel("age", size=20)
plt.ylabel("currentSmokers", size=20)
plt.xticks(size=12)
plt.yticks(size=12)


# Mid-age groups have more smokers

# # Relation between cigsPerDay and risk of coronary heart disease.

# In[ ]:


graph_8 = data.groupby("TenYearCHD", as_index=False).cigsPerDay.mean()

plt.figure(figsize=(12,8))
sns.barplot(x=graph_8["TenYearCHD"], y=graph_8["cigsPerDay"])
plt.title("Graph showing the relation between cigsPerDay and risk of coronary heart disease.")
plt.xlabel("Rick of CHD", size=20)
plt.ylabel("cigsPerDay", size=20)
plt.xticks(size=12)
plt.yticks(size=12)


# High cigsPerDay comes with higher risk of CHD.

# # Relation between sysBP and risk of CHD.

# In[ ]:


# Grouping up the data and ploting it

graph_9 = data.groupby("TenYearCHD", as_index=False).sysBP.mean()

plt.figure(figsize=(12,8))
sns.barplot(x=graph_9["TenYearCHD"], y=graph_9["sysBP"])
plt.title("Graph showing the relation between sysBP and risk of CHD")
plt.xlabel("Rick of CHD", size=20)
plt.ylabel("sysBP", size=20)
plt.xticks(size=12)
plt.yticks(size=12)


# Minor relation found between higher risk with higher sysBP  

# # Relation between diaBP and risk of CHD

# In[ ]:


# Grouping up the data and ploting it

graph_9 = data.groupby("TenYearCHD", as_index=False).diaBP.mean()

plt.figure(figsize=(12,8))
sns.barplot(x=graph_9["TenYearCHD"], y=graph_9["diaBP"])
plt.title("Graph showing the relation between diaBP and risk of CHD")
plt.xlabel("Rick of CHD", size=20)
plt.ylabel("diaBP", size=20)
plt.xticks(size=12)
plt.yticks(size=12)


# Minor relation found between higher risk with higher diaBP  

# # Predicting the risk of CHD with Logistic Regression. (85% accuracy)

# In[ ]:


# collecting the features in X

X = data.drop(columns=["TenYearCHD"])


# In[ ]:


# y is the target variable (risk of CHD)

y = data["TenYearCHD"]


# In[ ]:


# defining the model

model = LogisticRegression(random_state=3, max_iter=1000)


# In[ ]:


# splitting and training the data

train_X, test_X, train_y, test_y = train_test_split(X,y, test_size=0.5)


# In[ ]:


# fitting the model with X and y

model.fit(train_X, train_y)


# In[ ]:


# accuracy of the model is

model.score(test_X, test_y)

