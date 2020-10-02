#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# For Plotting
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# First, Let us load the dataset and see how it looks

# In[ ]:


data = pd.read_csv("../input/diabetes.csv")
data.head()


# We shall also look at the 5 point summary of the data

# In[ ]:


data.describe()


# We have different parameters that may or may not impact the Diabetes Diagnosis of Pima Indian Women. Let's find out!
# 
# We can have a single number estimation over the entire dataset in-terms of the Outcome.

# In[ ]:


percent = (data.groupby("Outcome")["Outcome"].count()/len(data.index))*100
plt.figure(figsize=(5,5))
plt.pie(percent, labels=(percent.round(2)))


# Therefore it is safe to assume that approximately 35% of the Pima Indian Women are diagnosed with Diabetes. This is our baseline model.
# 
# Now lets dig deep! To start with let us check how the parameters are correlated with each other. I am using Pair plot since it also captures the frequency distribution of each parameter along the diagonal.

# In[ ]:


sns.pairplot(data, hue="Outcome")


# As we can see there is no significant pattern between parameters. Since Outcome is added as the hue we can also see if there is any cluster being formed that explains the grouping of the Outcome. We could only see blue and orange dots overlap. However we can see some correlation between BMI-SkinThickness, Age-Pregnencies, Insulin-Glucose. But not too strong to be considered. Anyway we will confirm the same by plotting correlation matrix.

# In[ ]:


sns.heatmap(data.drop("Outcome", axis=1).corr(), annot=True)


# As we can see in the plot above there is no significant correlation between parameters.
# 
# Since there is very little relation between the parameters themselves, we will now analyse each parameter against the Target.
# 
# So lets look into Age and see how it is distributed

# In[ ]:


plt.hist(data["Age"])


# As the about plot suggests most of the observed women are young and middle aged. Let us look at the percentage of Outcome across the Age groups.
# 
# First, let us categorize each observation into particular age bucket.

# In[ ]:


bins = pd.Series([])
for i in data.index:
    if (data.loc[i:i,]["Age"] <= 25).bool(): bins = bins.append(pd.Series(["20-25"]))
    elif (data.loc[i:i,]["Age"] <= 30).bool(): bins = bins.append(pd.Series(["26-30"]))
    elif (data.loc[i:i,]["Age"] <= 35).bool(): bins = bins.append(pd.Series(["31-35"]))
    elif (data.loc[i:i,]["Age"] <= 40).bool(): bins = bins.append(pd.Series(["36-40"]))
    elif (data.loc[i:i,]["Age"] <= 45).bool(): bins = bins.append(pd.Series(["41-45"]))
    elif (data.loc[i:i,]["Age"] <= 50).bool(): bins = bins.append(pd.Series(["46-50"]))
    elif (data.loc[i:i,]["Age"] <= 55).bool(): bins = bins.append(pd.Series(["51-55"]))
    elif (data.loc[i:i,]["Age"] <= 60).bool(): bins = bins.append(pd.Series(["56-60"]))
    elif (data.loc[i:i,]["Age"] <= 65).bool(): bins = bins.append(pd.Series(["61-65"]))
    else: bins = bins.append(pd.Series([">65"]))
bins.reset_index(drop=True, inplace=True)
data["Ages"] = bins
data.head()


# Now calculate the no. of diabetic person in each age group

# In[ ]:


bindata1 = data[data["Outcome"]==1].groupby("Ages")[["Outcome"]].count()
bindata1.head()


# Calculate the % of diabetic person in each age group

# In[ ]:


bindata = data.groupby("Ages")[["Outcome"]].count()
bindata1["% Diabetic"] = (bindata1["Outcome"]/bindata["Outcome"])*100


# In[ ]:


sns.barplot(x=bindata1.index, y=bindata1["% Diabetic"])


# It is clearly evident that the percentage of women between 30 and 55, being diagnosed with diabetes, is well above our baseline model. We can say that the middle aged women are most likely to be diabetic than young or old women.
# 
# Let us try to visualize the distribution of all the other parameters side-by-side

# In[ ]:


fig = plt.figure(figsize=(20,3))
for i in np.arange(1,7):
        splt =  plt.subplot(1,7,i,title=data.columns[i])
        plt.boxplot(data[data.columns[i]])


# In the above distributions we see a lot of "0" values. zero values in Glucose Level, Blood Pressure, Skin thickness, Insulin and BMI are not a normal phenomenon. There might be some error in the observation made. This might skew our distributions hence we remove those zeros in each of the bi-variate analysis against the target variable. We can leaving out the "DiabetesPedigreeFunction" as we are not sure how it has been calculated. Sources suggest that this function helps us to identify the heriditary effect on a human that causes diabetes. However we do not know how accurately this function has been constructed and the no. of generations been considered for the study is unknown. It is safe to leave out that variable for EDA as it lacks explainability.
# 
# Ok, Now we can start inspecting the Glucose levels as we did for the Age.

# In[ ]:


gluData = data[data["Glucose"]!=0]


# We can do binning as the same way we did for Age but the range is quite large here. For Glucose we will use numpy,pandas functions to do binning.

# In[ ]:


bins = np.arange(min(gluData["Glucose"]),max(gluData["Glucose"]),10)
bins


# In[ ]:


gluData["Glucose Levels"] = pd.cut(gluData["Glucose"], bins=bins)
gluData.head()


# Now we have categorized individuals based on thier glucose levels. Lets group them to get % values

# In[ ]:


bindata1 = gluData[gluData["Outcome"]==1].groupby("Glucose Levels")[["Outcome"]].count()
bindata = gluData.groupby("Glucose Levels")[["Outcome"]].count()
bindata1["% Diabetic"] = (bindata1["Outcome"]/bindata["Outcome"])*100


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x=bindata1.index, y=bindata1["% Diabetic"])


# Higher the Glucose Levels higher the chance of testing Diabetes.
# 
# Similarly, We can do the same for other parameters as well.
# 
# For Blood Pressure,

# In[ ]:


pressData = data[data["BloodPressure"]!=0]


# In[ ]:


bins = np.arange(min(pressData["BloodPressure"]),max(pressData["BloodPressure"]),10)
pressData["BP Levels"] = pd.cut(pressData["BloodPressure"], bins=bins)
bindata1 = pressData[pressData["Outcome"]==1].groupby("BP Levels")[["Outcome"]].count()
bindata = pressData.groupby("BP Levels")[["Outcome"]].count()
bindata1["% Diabetic"] = (bindata1["Outcome"]/bindata["Outcome"])*100
plt.figure(figsize=(15,5))
sns.barplot(x=bindata1.index, y=bindata1["% Diabetic"])


# As we gain more insight from each parameter level, the EDA intutively takes us towards selecting an algorithm based on our individual analysis. A tree based model will mostly do best on these type of problems. One can apply Boosting techniques on top of it improve accuracy but keep in mind that the tree based models are easy to explain.
# 
# Thanks for reading!
