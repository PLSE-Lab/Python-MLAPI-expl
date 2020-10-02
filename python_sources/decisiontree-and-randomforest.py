#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import tree
df = pd.read_csv('../input/xAPI-Edu-Data.csv')


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.tail(4).T


# In[ ]:


df.info()


# In[ ]:


df.rename(index=str, columns={'gender':'Gender', 'NationalITy':'Nationality','raisedhands':'RaisedHands', 'VisITedResources':'VisitedResources'},inplace=True)


# In[ ]:


from sklearn.cross_validation import train_test_split

# Generate the training set.  Set random_state to be able to replicate results.
train = df.sample(frac=0.7, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = df.loc[~df.index.isin(train.index)]


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


from sklearn.cross_validation import train_test_split

# Generate the training set.  Set random_state to be able to replicate results.
train = df.sample(frac=0.7, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = df.loc[~df.index.isin(train.index)]


# In[ ]:


train['Class'].value_counts()


# In[ ]:


train['Nationality'].value_counts()


# In[ ]:


pd.crosstab(train['Class'], train['Topic'])


# In[ ]:


pd.crosstab(train['Class'],train['Gender'])


# In[ ]:


train['Gender'][train['Gender'] == "F"] = 0
train['Gender'][train['Gender'] == "M"] = 1
train['Gender'] = train['Gender'].astype(int)

train["Class"][train["Class"] == "L"] = 0
train["Class"][train["Class"] == "M"] = 1
train["Class"][train["Class"] == "H"] = 2
train['Class'] = train['Class'].astype(int)


# In[ ]:


ax = sns.boxplot(x='Class', y='Discussion', data=train)
ax = sns.swarmplot(x='Class', y = 'Discussion', data=train, color='.25')
plt.show()


# In[ ]:


sns.factorplot('ParentschoolSatisfaction','Class',data=train)


# In[ ]:


train['ParentschoolSatisfaction'][train['ParentschoolSatisfaction'] == "Good"] = 0
train['ParentschoolSatisfaction'][train['ParentschoolSatisfaction'] == "Bad"] = 1
train['ParentschoolSatisfaction'] = train['ParentschoolSatisfaction'].astype(int)


# In[ ]:


sns.factorplot('Relation','Class',data=train)


# In[ ]:


train['Relation'][train['Relation'] == "Father"] = 0
train['Relation'][train['Relation'] == "Mum"] = 1
train['Relation'] = train['Relation'].astype(int)


# In[ ]:


sns.swarmplot(x='Class', y='AnnouncementsView', data=train)


# In[ ]:


Raised_hand = sns.swarmplot(x="Class", y="RaisedHands", data=train)


# In[ ]:


ax = sns.boxplot(x='Class', y='VisitedResources', data=train)
ax = sns.swarmplot(x='Class', y = 'VisitedResources', data=train, color='.25')
plt.show()


# In[ ]:


sns.factorplot('StudentAbsenceDays', 'Class', data=train)


# In[ ]:


train['StudentAbsenceDays'][train['StudentAbsenceDays'] == "Under-7"] = 0
train['StudentAbsenceDays'][train['StudentAbsenceDays'] == "Above-7"] = 1
train['StudentAbsenceDays'] = train['StudentAbsenceDays'].astype(int)


# In[ ]:


from sklearn import tree
target = train['Class'].values
features_one = train[["Gender", "ParentschoolSatisfaction", "Relation", "AnnouncementsView", "RaisedHands", "VisitedResources", "StudentAbsenceDays"]].values
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features, target)

print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))


# In[ ]:


# add one more attrinute into the features 
features_two = train[["Gender", "Discussion", "ParentschoolSatisfaction", "Relation", "AnnouncementsView", "RaisedHands", "VisitedResources", "StudentAbsenceDays"]].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)
print(my_tree_two.feature_importances_)
print(my_tree_two.score(features_two, target))


# In[ ]:


# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# Building and fitting my forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=5, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_two, target)
print(my_forest.feature_importances_)

# Print the score of the fitted random forest
print(my_forest.score(features_two, target))


# In[ ]:


#Start testing. Convert the object attribute in the testing set to integer.
test['Gender'][test['Gender'] == "F"] = 0
test['Gender'][test['Gender'] == "M"] = 1
test['Gender'] = test['Gender'].astype(int)

test["Class"][test["Class"] == "L"] = 0
test["Class"][test["Class"] == "M"] = 1
test["Class"][test["Class"] == "H"] = 2
test['Class'] = test['Class'].astype(int)

test['ParentschoolSatisfaction'][test['ParentschoolSatisfaction'] == "Good"] = 0
test['ParentschoolSatisfaction'][test['ParentschoolSatisfaction'] == "Bad"] = 1
test['ParentschoolSatisfaction'] = test['ParentschoolSatisfaction'].astype(int)

test['Relation'][test['Relation'] == "Father"] = 0
test['Relation'][test['Relation'] == "Mum"] = 1
test['Relation'] = test['Relation'].astype(int)

test['StudentAbsenceDays'][test['StudentAbsenceDays'] == "Under-7"] = 0
test['StudentAbsenceDays'][test['StudentAbsenceDays'] == "Above-7"] = 1
test['StudentAbsenceDays'] = test['StudentAbsenceDays'].astype(int)


# In[ ]:


#Create testing features.
test_features= test[["Gender", "Discussion", "ParentschoolSatisfaction", "Relation", "AnnouncementsView", "RaisedHands", "VisitedResources", "StudentAbsenceDays"]].values
print(test_features)


# In[ ]:


# Makeing prediction for the test set, and print the prediction.
my_prediction = my_forest.predict(test_features)
print(my_prediction)


# In[ ]:


my_solution = pd.DataFrame(my_prediction, columns = ["Class"])
print(my_solution)


# In[ ]:


my_solution.shape


# In[ ]:


my_solution.to_csv("my_solution_two.csv")


# In[ ]:


from sklearn.metrics import confusion_matrix
results = confusion_matrix(my_solution, test["Class"])
print(results)


# In[ ]:


from sklearn.metrics import classification_reportdf_cm = pd.DataFrame(results, index = [i for i in "LMH"],
                  columns = [i for i in "LMH"])
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True)


# In[ ]:


from sklearn.metrics import classification_report
y_true = test["Class"]
y_pred = my_prediction
target_names = ['class 0(L)', 'class 1(M)', 'class 2(H)']
print(classification_report(y_true, y_pred, target_names=target_names))

