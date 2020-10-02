#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import statistics


# In[ ]:


# defining directory paths
train_dir = "../input/train.csv"
test_dir = "../input/test.csv"


# In[ ]:


df = pd.read_csv(train_dir)
test_df = pd.read_csv(test_dir)
#checking for missing and na values
print("Total number of instance : ",len(df))
df.isna().sum()


# In[ ]:


df.head(3)


# The data contains a lot of missing values in columns **Age** and **Cabin**.
# to fill missing values we will fill those missing values in **Age** column with "-1".
# 
# **Cabin** Column contains 687 missing values out of a total of 891 values, we'll check what type of values cabin col has.

# In[ ]:


df.fillna(-1, inplace = True) #filling in missing values with -1
test_df.fillna(-1, inplace = True)
df["Cabin"].unique()


# **Cabin** column has a lot of unique values, we can still use the alphabetical cabin type but it wont improve our model much thus dropping it would be the best.

# In[ ]:


df.drop(["Cabin"], axis = 1, inplace = True)
test_df.drop(["Cabin"], axis = 1, inplace = True)
df.drop(["Ticket"], axis = 1, inplace = True)
test_df.drop(["Ticket"], axis = 1, inplace = True)
df.info()
df.head(10)


# In[ ]:


# checking class distribution
df["Survived"].value_counts().plot(kind = "bar")


# **Correlation map :**

# In[ ]:


f,ax = plt.subplots(figsize=(15, 13))
sns.heatmap(df.corr(), annot=True, cmap = "Blues", linewidths=.5, fmt= '.2f',ax = ax)
plt.show()


# Now checking class distribution of **pclass**, i.e., how many people from each class survived.

# In[ ]:


df_survived = df[df["Survived"]==1]
df_notsurvived = df[df["Survived"]==0]
gb_pclass_surv = df_survived.groupby("Pclass")["Survived"].sum()
#a = gb_pclass_surv.plot(kind= "bar")
gb_pclass_notsurv = df_notsurvived.groupby("Pclass")["Survived"].count()
#b = gb_pclass_notsurv.plot(kind= "bar")

fig = plt.figure(figsize = (10,4))
f1 = fig.add_subplot(1, 2, 1)
f1.set_ylim([0,400])
f2 = fig.add_subplot(1,2,2)
f2.set_ylim([0,400])
gb_pclass_surv.plot(kind= "bar", title = "Survived", ax = f1)
gb_pclass_notsurv.plot(kind= "bar", title = "Not Survived", ax = f2)


# The above figure shows that most of the people from class 3 did'nt survive while nearly equal no. of people from the 2nd class did and did not survive, while more people of the 1st class survived as compared to non survival rate, thus pclass is an important data for training the classifier.

# Also, the PassengerId column only tells about the id of the passenger travelling on the ship thus it is useless for training purpose thus dropping PassengerId. 

# In[ ]:


df.drop("PassengerId", axis = 1, inplace = True)
test_df.drop("PassengerId", axis = 1, inplace = True)


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


# * Now, the columns **Sex** and **Embarked** are object type columns, thus we need to change them to numeric type.

# In[ ]:


df["Sex"].replace("male", 0, inplace = True)
test_df["Sex"].replace("male", 0, inplace = True)
df["Sex"].replace("female", 1, inplace = True)
test_df["Sex"].replace("female", 1, inplace = True)

df["Embarked"].replace(["S","C","Q"],[0,1,2], inplace = True)
test_df["Embarked"].replace(["S","C","Q"],[0,1,2], inplace = True)


# Using the **parch** and **sibsp** column we can make a new column named no. of family members onboard **(n_fam_mem)**. 
# And visualizing results.

# In[ ]:


df["n_fam_mem"] = df["SibSp"] + df["Parch"]
test_df["n_fam_mem"] = df["SibSp"] + df["Parch"]
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
df_notsurvived["n_fam_mem"].value_counts().plot(kind = "bar", ax = f3)


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

famsize = create_family_ranges(df)
df["familysize"] = famsize

test_famsize = create_family_ranges(test_df)
test_df["familysize"] = test_famsize


# The column **Age** contains continuous data, thus dividing it into particular ranges.

# In[ ]:


df["Age"].where(df["Age"]!=-1).mean()


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


ageint = age_to_int(df)
df["Ageint"] = ageint
df.drop("Age", axis = 1, inplace = True)

test_ageint = age_to_int(test_df)
test_df["Ageint"] = test_ageint
test_df.drop("Age", axis = 1, inplace = True)


# Now the data in **Fare** seems like it is the total of what the passenger paid including the fare of the other family members, so we create a new column named actual_fare i.e., the fare divided by n_fam_mem + 1.

# In[ ]:


df["actual_fare"] = df["Fare"]/(df["n_fam_mem"]+1)

test_df["actual_fare"] = test_df["Fare"]/(test_df["n_fam_mem"]+1)

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

test_fare_ranges = conv_fare_ranges(test_df)
test_df["fare_ranges"] = test_fare_ranges


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

testtitlelist = name_to_int(test_df)
test_df["titles"] = testtitlelist


# As we can see a lot of titles occur only one thus we will replace this by "sometitle"

# In[ ]:


df["titles"].replace(["Ms.","Jonkheer.","the","Don.","Capt.","Sir.","Lady.","Mme.","Col.","Major."],"sometitle", inplace = True)
test_df["titles"].replace(["Ms.","Jonkheer.","the","Don.","Capt.","Sir.","Lady.","Mme.","Col.","Major."],"sometitle", inplace = True)

df["titles"].replace("Mlle.","Miss.", inplace = True)
test_df["titles"].replace("Mlle.","Miss.", inplace = True)

test_df["titles"].value_counts()


# In[ ]:


df["titles"].replace(["Mr.", "Miss.", "Mrs", "Master.", "Dr.", "Rev.", "sometitle"],[0,1,2,3,4,5,6], inplace = True)
df["titles"].astype("int64")

test_df["titles"].replace(["Mr.", "Miss.", "Mrs", "Master.", "Dr.", "Rev.", "sometitle"],[0,1,2,3,4,5,6], inplace = True)
test_df["titles"].astype("int64")

df.drop(["Name"], axis = 1, inplace = True)
test_df.drop(["Name"], axis = 1, inplace = True)


# There is no object type data left. You can check this by df.info()

# We have cleaned and processed the data, we only need to get rid of a few original columns which we used to derive new columns. we need to drop :
# (Fare, n_fam_mem, actual_fare)

# In[ ]:


df.drop(["SibSp","Parch","Fare","n_fam_mem","actual_fare"], axis = 1, inplace = True)
test_df.drop(["SibSp","Parch","Fare","n_fam_mem","actual_fare"], axis = 1, inplace = True)


# In[ ]:


df.info()
f,ax = plt.subplots(figsize=(15, 13))
sns.heatmap(df.corr(), annot=True,cmap = "Blues", linewidths=.5, fmt= '.2f',ax = ax)
plt.show()


# Next step is to dividing data into Training data and training labels, where labels is "Survived" column, and then splitting training data into train and test sets.

# In[ ]:


labels = df["Survived"]
data = df.drop("Survived", axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size = 0.1)


# **Defining and choosing the best from one of the classifiers given below : **
# 1. Logistic Regression
# 2. K-nearest Neighbors with n_neighbors = 3
# 3. K-nearest Neighbors with n_neighbors = 5
# 4. Random Forest Classifier
# 5. Decision Tree Classifier
# 6. Gradient Boosting Classifier
# 7. Support Vector Machine

# In[ ]:


final_clf = None
clf_names = ["Logistic Regression", "KNN(3)", "KNN(5)", "Random forest classifier", "Decision Tree Classifier",
            "Gradient Boosting Classifier", "Support Vector Machine"]
classifiers = []
scores = []


# In[ ]:


bestknn5 = None
bestknn3 = None
bestrf = None
bestgb = None
bestcvm = None
bestlr = None
bestdt = None
for i in range(10):
    
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size = 0.1)
    tempscores = []
    
    # logistic Regression
    lr_clf = LogisticRegression()
    lr_clf.fit(X_train, Y_train)
    tempscores.append((lr_clf.score(X_test, Y_test))*100)
    if lr_clf.score(X_test, Y_test)*100 > 85:
        bestlr = lr_clf
    
    # KNN n_neighbors = 3
    knn3_clf = KNeighborsClassifier()
    knn3_clf.fit(X_train, Y_train)
    tempscores.append((knn3_clf.score(X_test, Y_test))*100)
    if knn3_clf.score(X_test, Y_test)*100 > 85:
        bestknn3 = knn3_clf

    # KNN n_neighbors = 5
    knn5_clf = KNeighborsClassifier()
    knn5_clf.fit(X_train, Y_train)
    tempscores.append((knn5_clf.score(X_test, Y_test))*100)
    if knn5_clf.score(X_test, Y_test)*100 > 85:
        bestknn5 = knn5_clf

    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators = 100)
    rf_clf.fit(X_train, Y_train)
    tempscores.append((rf_clf.score(X_test, Y_test))*100)
    if rf_clf.score(X_test, Y_test)*100 > 85:
        bestrf = rf_clf

    # Decision Tree
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X_train, Y_train)
    tempscores.append((dt_clf.score(X_test, Y_test))*100)
    if dt_clf.score(X_test, Y_test)*100 > 85:
        bestdt = dt_clf

    # Gradient Boosting 
    gb_clf = GradientBoostingClassifier()
    gb_clf.fit(X_train, Y_train)
    tempscores.append((gb_clf.score(X_test, Y_test))*100)
    if gb_clf.score(X_test, Y_test)*100 > 85:
        bestgb = lr_clf

    #SVM
    svm_clf = SVC(gamma = "scale")
    svm_clf.fit(X_train, Y_train)
    tempscores.append((svm_clf.score(X_test, Y_test))*100)
    if svm_clf.score(X_test, Y_test)*100 > 85:
        bestsvm = svm_clf
    
    scores.append(tempscores)


# In[ ]:


scores = np.array(scores)
clfs = pd.DataFrame({"Classifier":clf_names})
clfs["iteration0"] = scores[0].T
clfs["iteration1"] = scores[1].T
clfs["iteration2"] = scores[2].T
clfs["iteration3"] = scores[3].T
clfs["iteration4"] = scores[4].T
clfs["iteration5"] = scores[5].T
clfs["iteration6"] = scores[6].T
clfs["iteration7"] = scores[7].T
clfs["iteration8"] = scores[8].T
clfs["iteration9"] = scores[9].T

means = clfs.mean(axis = 1)
means = means.values.tolist()

clfs["Average"] = means


# In[ ]:


clfs.head(10)


# Now Choosing the classifier accordingly,
# here we have used SVM.

# In[ ]:


final_clf = bestsvm


# New we use the trained classifier to make predictions on the testing data.
# 

# In[ ]:


test_data = test_df
predictions = final_clf.predict(test_data)
print(len(predictions))


# Saving final resutls into the csv file for submission.

# In[ ]:


final_csv = []
csv_title = ['PassengerId', 'Survived']
final_csv.append(csv_title)
for i in range(len(predictions)):
    passengerid = i + 892
    survived = predictions[i]
    temp = [passengerid, survived]
    final_csv.append(temp)

print(len(final_csv))

with open('submission_csv.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(final_csv)
file.close()

