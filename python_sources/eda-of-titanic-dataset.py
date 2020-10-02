#!/usr/bin/env python
# coding: utf-8

# ## Hands-on Kaggle

# ### Importing the necessary libraries and setting up various paths.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import csv
import matplotlib.pyplot as plt
import seaborn as sns
import statistics


# In[ ]:


# defining directory paths
train_dir = "../input/train.csv"


# ### Preview of data.
# Get the preview of the data by reading the csv file, printing the length of the csv file and finding the number of NaN in data.

# In[ ]:


df = pd.read_csv(train_dir)
#checking for missing and na values
print("Total number of instance : ",len(df))

# code for printing NaN columns.


# The data contains a lot of missing values in columns **Age** and **Cabin**.
# to fill missing values we will fill those missing values in **Age** column with "-1".
# 
# **Cabin** Column contains 687 missing values out of a total of 891 values, we'll check what type of values cabin col has.

# In[ ]:


#filling in missing values with -1
df["Cabin"].unique()


# **Cabin** column has a lot of unique values, we can still use the alphabetical cabin type but it wont improve our model much thus dropping it would be the best.

# In[ ]:


# Drop cabin and ticket column.
df.info()
df.head(3)


# In[ ]:


# checking class distribution
df["Survived"].value_counts()
# Use this to plot a bar chart.


# **Correlation map :**

# In[ ]:


## Plot the heatmap using the sns library
plt.show()


# Now checking class distribution of **pclass**, i.e., how many people from each class survived.

# In[ ]:


df_survived = # Select only the survived people
df_notsurvived = # Select only the survived people
gb_pclass_surv = # Group by Pclass survivors
gb_pclass_notsurv = # Group by Pclass Non survivors
fig = plt.figure(figsize = (10,4))

## Add plots for both survivors and non survivors.


# The above figure shows that most of the people from class 3 did'nt survive while nearly equal no. of people from the 2nd class did and did not survive, while more people of the 1st class survived as compared to non survival rate, thus pclass is an important data for training the classifier.

# Also, the PassengerId column only tells about the id of the passenger travelling on the ship thus it is useless for training purpose thus dropping PassengerId. 

# In[ ]:


# Replace passenger ID


# Checking data in sibsp (sibling/spouse) and parch(parent/children), basically these columns gives information about how many family members the person is travelling with.

# In[ ]:


print("SibSp unqiue value counts :\n" + str(df["SibSp"].value_counts()))

fig = plt.figure(figsize = (15,5))
f1 = fig.add_subplot(1, 3, 1)
f1.set_ylim([0,700])
f2 = fig.add_subplot(1,3,2)
f2.set_ylim([0,700])
f3 = fig.add_subplot(1,3, 3)
f3.set_ylim([0,700])
df["SibSp"].value_counts().plot(kind= "bar", title = "(SibSp) Total", ax = f1)
df_survived["SibSp"].value_counts().plot(kind= "bar", title = "(SibSp) Survived", ax = f2)
df_notsurvived["SibSp"].value_counts().plot(kind= "bar", title =  "(SibSp) Not Survived", ax = f3)
plt.show()


# In[ ]:


print("Parch unique value counts : \n" + str(df["Parch"].value_counts()))

fig = plt.figure(figsize = (15,5))
f1 = fig.add_subplot(1, 3, 1)
f1.set_ylim([0,700])
f2 = fig.add_subplot(1,3,2)
f2.set_ylim([0,700])
f3 = fig.add_subplot(1,3, 3)
f3.set_ylim([0,700])
df["Parch"].value_counts().plot(kind= "bar", title = "(Parch) Total", ax = f1)
df_survived["Parch"].value_counts().plot(kind= "bar", title = "(Parch) Survived", ax = f2)
df_notsurvived["Parch"].value_counts().plot(kind= "bar", title =  "(Parch) Not Survived", ax = f3)
plt.show()


# Now, the columns **Sex** and **Embarked** are object type columns, thus we need to change them to numeric type.

# In[ ]:


# Code for replacing Sex and Embarked to codes.


# Using the **parch** and **sibsp** column we can make a new column named no. of family members onboard **(n_fam_mem)**. 
# And visualizing results.

# In[ ]:


df["n_fam_mem"] = df["SibSp"] + df["Parch"]
df_survived["n_fam_mem"] = df_survived["SibSp"] + df_survived["Parch"]
df_notsurvived["n_fam_mem"] = df_notsurvived["SibSp"] + df_notsurvived["Parch"]

fig = plt.figure(figsize = (15,5))
f1 = fig.add_subplot(1, 3, 1)
f1.set_ylim([0,600])
f2 = fig.add_subplot(1,3,2)
f2.set_ylim([0,600])
f3 = fig.add_subplot(1,3, 3)
f3.set_ylim([0,600])

df["n_fam_mem"].value_counts().plot(kind = "bar", title = "all", ax = f1)
df_survived["n_fam_mem"].value_counts().plot(kind = "bar", title = "Survived", ax = f2)
df_notsurvived["n_fam_mem"].value_counts().plot(kind = "bar", ax = f3, title = "Not Survived")


# Now we will divide the n_fam_mem into specific ranges or type, say single person (0), small family (1) or big family (2), 

# In[ ]:


def create_family_ranges(df):
    familysize = []
    for members in df["n_fam_mem"]:
        if members == 0:
            familysize.append(0)
        elif members > 0 and members <=4:
            familysize.append(1)
        elif members > 4:
            familysize.append(2)
    return familysize

famsize = #code for creating the family size
df["familysize"] = famsize


# The column **Age** contains continuous data, thus dividing it into particular ranges.

# In[ ]:


# Code for finding the average age of the passengers


# In[ ]:


def age_to_int(df):
    agelist = df["Age"].values.tolist()
    for i in range(len(agelist)):
        if agelist[i] < 18 and agelist[i] >= 0:
            agelist[i] = 0
        elif agelist[i] >= 18 and agelist[i] < 60:
            agelist[i] = 1
        elif agelist[i]>=60 and agelist[i]<200:
            agelist[i] = 2
        else:
            agelist[i] = -1
    ageint = pd.DataFrame(agelist)
    return ageint


# In[ ]:


# Code for converting age to categories and dropping age column


# Now the data in **Fare** seems like it is the total of what the passenger paid including the fare of the other family members, so we create a new column named actual_fare i.e., the fare divided by n_fam_mem + 1.

# In[ ]:


df["actual_fare"] = df["Fare"]/(df["n_fam_mem"]+1)

df["actual_fare"].plot()
df["actual_fare"].describe()


# Dividing the actual fare into 5 different ranges.

# **Fare Ranges = less than 7 , 7-14 , 14-30 , 30-50 , more than 50 **

# In[ ]:


def conv_fare_ranges(df): 
    fare_ranges = []
    for fare in df.actual_fare:
        if fare < 7:
            fare_ranges.append(0)
        elif fare >=7 and fare < 14:
            fare_ranges.append(1)
        elif fare >=14 and fare < 30:
            fare_ranges.append(2)
        elif fare >=30 and fare < 50:
            fare_ranges.append(3)
        elif fare >=50:
            fare_ranges.append(4)
    return fare_ranges
        
fare_ranges = conv_fare_ranges(df)
df["fare_ranges"] = fare_ranges


# In[ ]:


df_nonsurv_fare = df[df["Survived"]==0]
df_surv_fare = df[df["Survived"]==1]

fig = plt.figure(figsize = (15,5))
f1 = fig.add_subplot(1, 3, 1)
f1.set_ylim([0,500])
f2 = fig.add_subplot(1,3,2)
f2.set_ylim([0,500])
f3 = fig.add_subplot(1,3, 3)
f3.set_ylim([0,500])

df["fare_ranges"].value_counts().plot(kind="bar", title = "Fare Ranges all", ax = f1)
df_surv_fare["fare_ranges"].value_counts().plot(kind="bar", title =  "Survived", ax = f2)
df_nonsurv_fare["fare_ranges"].value_counts().plot(kind="bar", title = "Not Survived", ax = f3)


# Now the **name** column has unique data item in every row, but each name has a title with it. We can use the title and check if it relates to something.

# Below is a dictionary that maps to the title of a person to a label, this includes all the title present in the dataframe.

# In[ ]:


def name_to_int(df):
    name = df["Name"].values.tolist()
    namelist = []
    for i in name:
        index = 1
        inew = i.split()
        if inew[0].endswith(","):
            index = 1
        elif inew[1].endswith(","):
            index = 2
        elif inew[2].endswith(","):
            index = 3
        namelist.append(inew[index])
    print(set(namelist))
    
    titlelist = []
    
    for i in range(len(namelist)): 
        if namelist[i] == "Lady.":
            titlelist.append("Lady.")
        elif namelist[i] == "Ms.":
            titlelist.append("Ms.")
        elif namelist[i] == "Miss.":
            titlelist.append("Miss.")
        elif namelist[i] == "Dr.":
            titlelist.append("Dr.")
        elif namelist[i] == "Mr.":
            titlelist.append("Mr.")
        elif namelist[i] == "Jonkheer.":
            titlelist.append("Jonkheer.")
        elif namelist[i] == "Col.":
            titlelist.append("Col.")
        elif namelist[i] == "Mrs.":
            titlelist.append("Mrs")
        elif namelist[i] == "Sir.":
            titlelist.append("Sir.")
        elif namelist[i] == "Mlle.":
            titlelist.append("Mlle.")
        elif namelist[i] == "Capt.":
            titlelist.append("Capt.")
        elif namelist[i] == "the":
            titlelist.append("the")
        elif namelist[i] == "Don.":
            titlelist.append("Don.")
        elif namelist[i] == "Master.":
            titlelist.append("Master.")
        elif namelist[i] == "Rev.":
            titlelist.append("Rev.")
        elif namelist[i] == "Mme.":
            titlelist.append("Mme.")
        elif namelist[i] == "Major.":
            titlelist.append("Major.")
        else:
            titlelist.append("sometitle")
    print(set(namelist))
    return titlelist


# In[ ]:


titlelist = name_to_int(df)
df["titles"] = titlelist
df["titles"].value_counts()


# As we can see a lot of titles occur only one thus we will replace this by "sometitle"

# In[ ]:


df["titles"].replace(["Ms.","Jonkheer.","the","Don.","Capt.","Sir.","Lady.","Mme.","Col.","Major."],"sometitle", inplace = True)

df["titles"].replace("Mlle.","Miss.", inplace = True)


# In[ ]:


df["titles"].replace(["Mr.", "Miss.", "Mrs", "Master.", "Dr.", "Rev.", "sometitle"],[0,1,2,3,4,5,6], inplace = True)
df["titles"].astype("int64")


df.drop(["Name"], axis = 1, inplace = True)


# There is no object type data left. You can check this by df.info()

# We have cleaned and processed the data, we only need to get rid of a few original columns which we used to derive new columns. we need to drop :
# (Fare, n_fam_mem, actual_fare)

# In[ ]:


df.drop(["SibSp","Parch","Fare","n_fam_mem","actual_fare"], axis = 1, inplace = True)


# In[ ]:


df.info()
f,ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True,cmap = "Blues", linewidths=.5, fmt= '.2f',ax = ax)
plt.show()


# Next step is to dividing data into Training data and training labels, where labels is "Survived" column, and then splitting training data into train and test sets.
