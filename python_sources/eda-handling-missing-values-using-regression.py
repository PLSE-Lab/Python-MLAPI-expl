#!/usr/bin/env python
# coding: utf-8

# NOTE - V26 : Handling missing values in AGE column using Linear Regression. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from scipy import stats
from sklearn.model_selection import GridSearchCV
import statistics
from xgboost import XGBClassifier


# In[ ]:


# defining directory paths
train_dir = "../input/train.csv"
test_dir = "../input/test.csv"


# In[ ]:


df = pd.read_csv(train_dir)
test_df = pd.read_csv(test_dir)


# the dataset contains a lot of missing values in age and cabin columns, we will be taking care of those values later.

# In[ ]:


df.drop(["Ticket"], axis = 1, inplace = True)
test_df.drop(["Ticket"], axis = 1, inplace = True)
df.info()
df.head()


# **Now lets check the class distribution.**

# In[ ]:


# checking class distribution
print(df["Survived"].value_counts())
df["Survived"].value_counts().plot(kind = "pie")


# The PassengerId column only tells about the id of the passenger travelling on the ship thus it is useless for training purpose thus dropping PassengerId. 

# In[ ]:


df.drop("PassengerId", axis = 1, inplace = True)
test_df.drop("PassengerId", axis = 1, inplace = True)


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
gb_pclass_notsurv.plot(kind= "bar", title = "Not Survived", ax = f2);


# The above figure shows that most of the people from class 3 did'nt survive while nearly equal no. of people from the 2nd class did and did not survive, while more people of the 1st class survived as compared to non survival rate, thus pclass is an important data for training the classifier.

# Plotting the survival probability will clear it.

# In[ ]:


sns.catplot(x = 'Pclass', y = "Survived", data = df, kind = "bar");


# This clearly shows that people who have higher socioeconomic status have higher chances to survive and people with lower status have a lower chance of survival (survival rate nearly equal to 25%)

# This pclass1 is really important feature thus we will convert the Pclass column to dummy columns pclass1, pclass2, pclass3.

# In[ ]:


pclass_dum = pd.get_dummies(df["Pclass"])
test_pclass_dum = pd.get_dummies(test_df["Pclass"])

df = pd.concat([df, pclass_dum], axis = 1)
test_df = pd.concat([test_df, test_pclass_dum], axis = 1)

df.rename({1:"pclass1", 2:"pclass2", 3:"pclass3"}, axis = 1, inplace = True)
test_df.rename({1:"pclass1", 2:"pclass2", 3:"pclass3"}, axis = 1, inplace = True)

df.drop(["Pclass"], axis = 1, inplace = True)
test_df.drop(["Pclass"], axis = 1, inplace = True)


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


df["Sex"].replace("male", 0, inplace = True)
test_df["Sex"].replace("male", 0, inplace = True)
df["Sex"].replace("female", 1, inplace = True)
test_df["Sex"].replace("female", 1, inplace = True)

df["Embarked"].fillna("S", inplace = True)
test_df["Embarked"].fillna("S", inplace = True)

pclass_dum = pd.get_dummies(df["Embarked"])
test_pclass_dum = pd.get_dummies(test_df["Embarked"])

df = pd.concat([df, pclass_dum], axis = 1)
test_df = pd.concat([test_df, test_pclass_dum], axis = 1)

df.rename({"S":"embarked_s", "C":"embarked_c", "Q":"embarked_q"}, axis = 1, inplace = True)
test_df.rename({"S":"embarked_s", "C":"embarked_c", "Q":"embarked_q"}, axis = 1, inplace = True)

df.drop(["Embarked"], axis = 1, inplace = True)
test_df.drop(["Embarked"], axis = 1, inplace = True)


# Using the **parch** and **sibsp** column we can make a new column named no. of family members onboard **(n_fam_mem)**. 
# And visualizing results.

# In[ ]:


df["n_fam_mem"] = df["SibSp"] + df["Parch"] + 1
test_df["n_fam_mem"] = df["SibSp"] + df["Parch"] + 1
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
df_notsurvived["n_fam_mem"].value_counts().plot(kind = "bar", title = "Not Survived", ax = f3);


# Now we will divide the n_fam_mem into specific ranges or type, say single person (1), small family (2), medium family(2-4) or big family(>4), 

# In[ ]:


def create_family_ranges(df):
    familysize = []
    for members in df["n_fam_mem"]:
        if members == 1:
            familysize.append(1)
        elif members == 2:
            familysize.append(2)
        elif members>2 and members<=4:
            familysize.append(3)
        elif members > 4:
            familysize.append(4)
    return familysize

famsize = create_family_ranges(df)
df["familysize"] = famsize

test_famsize = create_family_ranges(test_df)
test_df["familysize"] = test_famsize


# Converting family size into dummies.

# In[ ]:


fsizedummies = pd.get_dummies(df["familysize"])
test_fsizedummies = pd.get_dummies(test_df["familysize"])

df = pd.concat([df, fsizedummies], axis = 1)
test_df = pd.concat([test_df, test_fsizedummies], axis = 1)

df.rename({1:"fam_single",2:"fam_small",3:"fam_medium", 4:"fam_big"}, axis = 1, inplace = True)
test_df.rename({1:"fam_single",2:"fam_small",3:"fam_medium", 4:"fam_big"}, axis = 1, inplace = True)


# In[ ]:


df.head()


# ## Using Regression to predict missing Age values
# The Age column contains a lot of missing values in both the training and the testing dataset we will deal with the missing values in the following way : 
# 1. Fitting a linear model on the known values of Age from both the dataframes (training and testing)
# 2. Using the above model to predict the unknown values of Age on both the dataframes.

# In[ ]:


reg_df = df.drop(["Survived", "Name", "Cabin"], axis = 1)
reg_df_test = test_df.drop(["Name", "Cabin"], axis = 1)
    
age_reg_df = reg_df[reg_df["Age"].isna() == False]
age_reg_df_test = reg_df_test[reg_df_test["Age"].isna() == False]

new_age_df = age_reg_df.append(age_reg_df_test)
    
new_age_X = new_age_df.drop(["Age"], axis = 1)
new_age_y = new_age_df["Age"]

new_age_X["Fare"].fillna(df["Fare"].median(), inplace = True)

linear_reg_model = LinearRegression().fit(new_age_X, new_age_y)


# In[ ]:


# get indexes of rows that have NaN value
def get_age_indexes_to_replace(df):
    age_temp_list = df["Age"].values.tolist()
    indexes_age_replace = []
    age_temp_list = [str(x) for x in age_temp_list]
    for i, item in enumerate(age_temp_list):
        if item == "nan":
            indexes_age_replace.append(i)
    return indexes_age_replace

indexes_to_replace_main = get_age_indexes_to_replace(df)
indexes_to_replace_test = get_age_indexes_to_replace(test_df)

# make predictions on the missing values
def linear_age_predictions(reg_df, indexes_age_replace):
    reg_df_temp = reg_df.drop(["Age"], axis = 1)
    age_predictions = []
    for i in indexes_age_replace:
        x = reg_df_temp.iloc[i]
        x = np.array(x).reshape(1,-1)
        pred = linear_reg_model.predict(x)
        age_predictions.append(pred)
    return age_predictions

age_predictions_main = linear_age_predictions(reg_df, indexes_to_replace_main)
age_predictions_test = linear_age_predictions(reg_df_test, indexes_to_replace_test)

# fill the missing values with predictions
def fill_age_nan(df, indexes_age_replace, age_predictions):
    for i, item in enumerate(indexes_age_replace):
        df["Age"][item] =  age_predictions[i]
    return df

df = fill_age_nan(df, indexes_to_replace_main, age_predictions_main)
df_test = fill_age_nan(test_df, indexes_to_replace_test, age_predictions_test)


# In[ ]:


def age_to_int(df):
    agelist = df["Age"].values.tolist()
    for i in range(len(agelist)):
        if agelist[i] < 14: #children
            agelist[i] = 0
        elif agelist[i] >= 14 and agelist[i] < 25: #youth
            agelist[i] = 1
        elif agelist[i]>=25 and agelist[i]<60:# adult
            agelist[i] = 2
        elif agelist[i]>=60:# senior
            agelist[i] = 3
    ageint = pd.DataFrame(agelist)
    return ageint


# Converting Age to Categorical:

# In[ ]:


ageint = age_to_int(df)
df["Ageint"] = ageint
df.drop("Age", axis = 1, inplace = True)

test_ageint = age_to_int(test_df)
test_df["Ageint"] = test_ageint
test_df.drop("Age", axis = 1, inplace = True)


# Now lets check the survival rate of age groups.

# In[ ]:


fig = plt.figure(figsize = (15,5))
f1 = fig.add_subplot(1, 3, 1)
f1.set_ylim([0,400])
f2 = fig.add_subplot(1,3,2)
f2.set_ylim([0,400])
df["Ageint"][df["Survived"] == 1].value_counts().plot(kind = "pie", title = "Survived", ax = f1)
df["Ageint"][df["Survived"] == 0].value_counts().plot(kind = "pie", title = "Not Survived", ax = f2);


# As we can see from the figure survival rate of children was more than any other age group, while the survival rate of adults and seniors was very low.

# Now the data in **Fare** seems like it is the total of what the passenger paid including the fare of the other family members, so we create a new column named actual_fare i.e., the fare divided by n_fam_mem.

# In[ ]:


test_df["Fare"].fillna(test_df["Fare"].median(), inplace = True)

df["actual_fare"] = df["Fare"]/df["n_fam_mem"]

test_df["actual_fare"] = test_df["Fare"]/test_df["n_fam_mem"]

df["actual_fare"].plot()
df["actual_fare"].describe()


# Dividing the actual fare into 5 different ranges.

# **Fare Ranges = less than 7 , 7-14 , 14-30 , 30-50 , more than 50 **

# In[ ]:


def conv_fare_ranges(df): 
    fare_ranges = []
    for fare in df["actual_fare"]:
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


# Fare Ranges and Survival plot :

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
df_nonsurv_fare["fare_ranges"].value_counts().plot(kind="bar", title = "Not Survived", ax = f3);


# Now looking at the **Cabin** feature, it consists of a lot of missing values and values with cabin no. and type.

# In[ ]:


df["Cabin"].fillna("unknown", inplace = True)
test_df["Cabin"].fillna("unknown", inplace = True)


# In[ ]:


cabins = [i[0]  if i!= 'unknown' else 'unknown' for i in df['Cabin']]
test_cabins = [i[0]  if i!= 'unknown' else 'unknown' for i in test_df['Cabin']]

df.drop(["Cabin"], axis = 1, inplace = True)
test_df.drop(["Cabin"], axis = 1, inplace = True)

df["cabintype"] = cabins
test_df["cabintype"] = test_cabins


# Now lets check the distribution of **cabins** according to the **class**.

# In[ ]:


fig = plt.figure(figsize = (15,5))
f1 = fig.add_subplot(1, 3, 1)
f1.title.set_text('Upper class')
f2 = fig.add_subplot(1,3,2)
f2.title.set_text('Middle class')
f3 = fig.add_subplot(1,3, 3)
f3.title.set_text('Lower class')
sns.catplot(y="pclass1",x="cabintype",data = df, kind = "bar",order = ['A','B','C','D','E','F','G','unknown'], ax = f1)
sns.catplot(y="pclass2",x="cabintype",data = df, kind = "bar",order = ['A','B','C','D','E','F','G','unknown'], ax = f2)
sns.catplot(y="pclass3",x="cabintype",data = df, kind = "bar",order = ['A','B','C','D','E','F','G','unknown'], ax = f3)
plt.close(2)
plt.close(3)
plt.close(4)


# Here we can see that most of the upper class people got known cabins(A-E), middle class got a lot of known cabins and  very few unknown cabins while the lower class got the most unknown cabins, thus cabins directly relate to the socioeconomic status of the person.

# Now lets check the probability of survival in accordance with the cabins.

# In[ ]:


sns.catplot(y="Survived",x="cabintype",data = df, kind = "bar",order = ['A','B','C','D','E','F','G','unknown']);


# As we can see the survival probability of people with unknown cabins are much less than people with known cabins.

# But the data contains most of the values for unknown cabins thus it would be better to drop these data columns.

# In[ ]:


df.drop(["cabintype"], axis = 1, inplace = True)
test_df.drop(["cabintype"], axis = 1, inplace = True)


# Next, the **name** column has unique data item in every row, but each name has a title with it. We can use the title and check if it relates to something.

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
        
    titlelist = []
    
    for i in range(len(namelist)): 
        titlelist.append(namelist[i])
    return titlelist


# In[ ]:


titlelist = name_to_int(df)
df["titles"] = titlelist
df["titles"].value_counts()
testtitlelist = name_to_int(test_df)
test_df["titles"] = testtitlelist
df["titles"].value_counts()


# As we can see a lot of titles occur only one thus we will replace this by "sometitle"

# Then we will find the probability of survival by each title.

# In[ ]:


df["titles"].replace(["Jonkheer.","the","Don.","Capt.","Sir.","Col.","Major.","Dr.","Rev."], "sometitle", inplace = True)
test_df["titles"].replace(["Jonkheer.","the","Don.","Capt.","Sir.","Col.","Major.","Dr.","Rev.","Dona."],"sometitle", inplace = True)

df["titles"].replace(["Mlle.","Lady.","Mme.","Ms."],"Miss.", inplace = True)
test_df["titles"].replace(["Mlle.","Lady.","Mme.","Ms."],"Miss.", inplace = True)

plot = sns.catplot(y="Survived",x="titles",data = df, kind = "bar",order = ["Mr.","Miss.","Mrs.","Master.","sometitle"])
plot.set_ylabels("Survival Probability")


# This shows that **women(Miss., Mrs)** and the **children(Master.)** have the most survival probability.

# In[ ]:


df["titles"].replace(["Mr.", "Miss.", "Mrs.", "Master.","sometitle"],[0,1,2,3,4], inplace = True)
df["titles"].astype("int64")

test_df["titles"].replace(["Mr.", "Miss.", "Mrs.", "Master.", "sometitle"],[0,1,2,3,4], inplace = True)
test_df["titles"].astype("int64")

df.drop(["Name"], axis = 1, inplace = True)
test_df.drop(["Name"], axis = 1, inplace = True)


# There is no object type data left. You can check this by df.info()

# We have cleaned and processed the data, we only need to get rid of a few original columns which we used to derive new columns. we need to drop :
# (Fare, n_fam_mem, actual_fare)

# In[ ]:


df.drop(["Fare","n_fam_mem","actual_fare"], axis = 1, inplace = True)
test_df.drop(["Fare","n_fam_mem","actual_fare"], axis = 1, inplace = True)


# In[ ]:


df.info()


# Next step is to define multiple classifiers and check which one works the best.

# In[ ]:


labels = df["Survived"]
data = df.drop("Survived", axis = 1)


# ## Performance of Different Classifiers

# **Defining and choosing the best from one of the classifiers given below : **
# 1. Logistic Regression
# 2. K-nearest Neighbors with n_neighbors = 3
# 3. XGBoost
# 4. Random Forest Classifier
# 5. Decision Tree Classifier
# 6. Gradient Boosting Classifier
# 7. Support Vector Machine

# In[ ]:


final_clf = None
clf_names = ["Logistic Regression", "KNN(3)", "XGBoost Classifier", "Random forest classifier", "Decision Tree Classifier",
            "Gradient Boosting Classifier", "Support Vector Machine"]


# In[ ]:


classifiers = []
scores = []
for i in range(10):
    
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size = 0.1)
    tempscores = []
    
    # logistic Regression
    lr_clf = LogisticRegression()
    lr_clf.fit(X_train, Y_train)
    tempscores.append((lr_clf.score(X_test, Y_test))*100)
    
    # KNN n_neighbors = 3
    knn3_clf = KNeighborsClassifier(n_neighbors = 3)
    knn3_clf.fit(X_train, Y_train)
    tempscores.append((knn3_clf.score(X_test, Y_test))*100)

    # XGBoost
    xgbc = XGBClassifier(n_estimators=15, seed=41)
    xgbc.fit(X_train, Y_train)
    tempscores.append((xgbc.score(X_test, Y_test))*100)

    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators = 100)
    rf_clf.fit(X_train, Y_train)
    tempscores.append((rf_clf.score(X_test, Y_test))*100)

    # Decision Tree
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X_train, Y_train)
    tempscores.append((dt_clf.score(X_test, Y_test))*100)

    # Gradient Boosting 
    gb_clf = GradientBoostingClassifier()
    gb_clf.fit(X_train, Y_train)
    tempscores.append((gb_clf.score(X_test, Y_test))*100)

    #SVM
    svm_clf = SVC(gamma = "scale")
    svm_clf.fit(X_train, Y_train)
    tempscores.append((svm_clf.score(X_test, Y_test))*100)
    
    scores.append(tempscores)


# In[ ]:


scores = np.array(scores)
clfs = pd.DataFrame({"Classifier":clf_names})
for i in range(len(scores)):
    clfs['iteration' + str(i)] = scores[i].T

means = clfs.mean(axis = 1)
means = means.values.tolist()

clfs["Average"] = means


# In[ ]:


clfs.set_index("Classifier", inplace = True)
print("Accuracies : ")
clfs["Average"].head(10)


# Now Choosing the classifier accordingly, After training and assessing multiple times it turns out that SVM is best classifier and provides best results. We do get better results while training with the Xgboost clf but while submission SVM performs much better.

# **ENSEMBLING MODELS**

# Here we will use ensemble learning to get better results : we will train the SVM classifier for a total of 5 times on different splits of data and then use them all to find the mode of the predictions and then use that mode.

# In[ ]:


# defining multiple SVM classifiers.
def create_multiple():    
    ensembles = []
    ensemble_scores = []
    for i in range(5):
        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size = 0.07)
        svm_clf = SVC(gamma = "scale")
        svm_clf = svm_clf.fit(X_train, Y_train)
        ensemble_scores.append((svm_clf.score(X_test, Y_test))*100)
        ensembles.append(svm_clf)
    return ensembles, ensemble_scores
SVM_ensembles, SVM_ensemble_scores = create_multiple()


# In[ ]:


def print_ensemble_score(ensemble_scores, model_name):    
    e_score = 0
    for i in range(len(ensemble_scores)):
        e_score = e_score + ensemble_scores[i]
    print("SCORE (ENSEMBLE MODELS) " +str(model_name)+ " : " + str(e_score/len(ensemble_scores)))
    return

print_ensemble_score(SVM_ensemble_scores, "SVM")


# We will use the above 5 classifiers to make predictions on our test data.

# In[ ]:


def per_model_prediction(ensembles):    
    test_data = test_df
    predictions_ensembles = []
    for clf in ensembles:
        temppredictions = clf.predict(test_data)
        predictions_ensembles.append(temppredictions)
    return predictions_ensembles


# In[ ]:


def get_predictions_modes(predictions_ensembles):    
    final_predictions_list = []
    for i in range(len(predictions_ensembles[0])):
        temp = [predictions_ensembles[0][i], predictions_ensembles[1][i], predictions_ensembles[2][i], predictions_ensembles[3][i], predictions_ensembles[4][i]]
        final_predictions_list.append(temp)

    final_predictions_list = np.array(final_predictions_list)
    pred_modes = stats.mode(final_predictions_list, axis = 1)

    final_predictions = []
    for i in pred_modes[0]:
        final_predictions.append(i[0])
    
    return final_predictions


# In[ ]:


SVM_predictions_ensembles = per_model_prediction(SVM_ensembles)
SVM_final_predictions = get_predictions_modes(SVM_predictions_ensembles)


# Saving final resutls into the csv file for submission.

# In[ ]:


passengerid = [892 + i for i in range(len(SVM_final_predictions))]
sub = pd.DataFrame({'PassengerId': passengerid, 'Survived':SVM_final_predictions})
sub.to_csv('submission.csv', index = False)

