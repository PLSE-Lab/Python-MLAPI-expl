#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import random as rnd


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV


# In[ ]:


# Load in the train and test datasets

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
unchanged_data = test_df


# In[ ]:


print(train_df.columns.values)


# In[ ]:


# preview the data
train_df.head()


# In[ ]:


sns.countplot(x = "Sex", hue = "Survived", data = train_df )


# In[ ]:


sns.countplot(x = "Pclass", hue = "Sex", data = train_df)


# In[ ]:


sns.countplot(x = "Parch", hue ="Survived",data = train_df)


# In[ ]:


sns.countplot(x = "Embarked", hue="Survived", data = train_df)


# In[ ]:


train_df.describe()


# In[ ]:


print(train_df.keys())
print(test_df.keys())


# In[ ]:


def null_table(train_df, test_df):
    print("Training Data Frame Imputation")
    print(pd.isnull(train_df).sum())
    print(" ")
    print("Testing Data Frame Imputation")
    print(pd.isnull(test_df).sum())

null_table(train_df, test_df)


# In[ ]:


train_df.drop(labels = ["Cabin", "Ticket"], axis = 1, inplace = True)
test_df.drop(labels = ["Cabin", "Ticket"], axis = 1, inplace = True)

null_table(train_df, test_df)


# In[ ]:


copy = train_df.copy()
copy.dropna(inplace = True)
sns.distplot(copy["Age"])


# Let's impute the NaN values with a median age

# In[ ]:


train_df["Age"].fillna(train_df["Age"].median(), inplace = True)
test_df["Age"].fillna(test_df["Age"].median(), inplace = True) 
train_df["Embarked"].fillna("S", inplace = True)
test_df["Fare"].fillna(test_df["Fare"].median(), inplace = True)

null_table(train_df, test_df)


# In[ ]:


train_df.count()


# In[ ]:


test_df.count()


# # Survival Distribution based on Class

# In[ ]:


sns.barplot(x="Pclass", y="Survived", data=train_df)
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Classes")
plt.title("Distribution of Survival Based on Class")
plt.show()

total_survived_one = train_df[train_df.Pclass == 1]["Survived"].sum()
total_survived_two = train_df[train_df.Pclass == 2]["Survived"].sum()
total_survived_three = train_df[train_df.Pclass == 3]["Survived"].sum()
total_survived_class = total_survived_one + total_survived_two + total_survived_three

print("Count of people who survived: " + str(total_survived_class))
print("Proportion of Class 1 Passengers who survived:") 
print(total_survived_one/total_survived_class)
print("Proportion of Class 2 Passengers who survived:")
print(total_survived_two/total_survived_class)
print("Proportion of Class 3 Passengers who survived:")
print(total_survived_three/total_survived_class)


# # Data Prep for Model Building

# In[ ]:


train_df.sample(5)


# In[ ]:


train_df.sample(5)


# # NaN Values

# In[ ]:


def nullValueCount(train_df, test_df):
    print("Training Data")
    print(pd.isnull(train_df).sum())
    print("\n")
    print("Testing Data")
    print(pd.isnull(test_df).sum())
    
nullValueCount(train_df, test_df)


# # One Hot Encoding

# In[ ]:


train_df.loc[train_df["Sex"] == "male", "Sex"] = 0
train_df.loc[train_df["Sex"] == "female", "Sex"] = 1

train_df.loc[train_df["Embarked"] == "S", "Embarked"] = 0
train_df.loc[train_df["Embarked"] == "C", "Embarked"] = 1
train_df.loc[train_df["Embarked"] == "Q", "Embarked"] = 2

test_df.loc[test_df["Sex"] == "male", "Sex"] = 0
test_df.loc[test_df["Sex"] == "female", "Sex"] = 1

test_df.loc[test_df["Embarked"] == "S", "Embarked"] = 0
test_df.loc[test_df["Embarked"] == "C", "Embarked"] = 1
test_df.loc[test_df["Embarked"] == "Q", "Embarked"] = 2


# In[ ]:


test_df.sample(10)


# # Combining SibSp and Parch

# In[ ]:


train_df["FamSize"] = train_df["SibSp"] + train_df["Parch"] + 1
test_df["FamSize"] = test_df["SibSp"] + test_df["Parch"] + 1


# In[ ]:


train_df["Solo"] = train_df.FamSize.apply(lambda x: 1 if x == 1 else 0)
test_df["Solo"] = test_df.FamSize.apply(lambda x: 1 if x == 1 else 0)


# In[ ]:


for name in train_df["Name"]:
    train_df["Title"] = train_df["Name"].str.extract("([A-Za-z)]+)\.", expand = True)
    
for name in test_df["Name"]:
    test_df["Title"] = test_df["Name"].str.extract("([A-Za-z)]+)\.", expand = True)


# # Let's check if the lamda function ran correctly or not.

# In[ ]:


train_df["Title"].sample(5)


# In[ ]:


title_replacements = {"Mlle": "Other", 
                      "Major": "Other", 
                      "Col": "Other", 
                      "Sir": "Other", 
                      "Don": "Other", 
                      "Mme": "Other",
                      "Jonkheer": "Other", 
                      "Lady": "Other", 
                      "Capt": "Other", 
                      "Countess": "Other", 
                      "Ms": "Other", 
                      "Dona": "Other", 
                      "Rev": "Other", 
                      "Dr": "Other"}

train_df.replace({"Title": title_replacements}, inplace=True)
test_df.replace({"Title": title_replacements}, inplace=True)


# # Re-coding the titles with numbers

# In[ ]:


train_df.loc[train_df["Title"] == "Miss", "Title"] = 0
train_df.loc[train_df["Title"] == "Mr", "Title"] = 1
train_df.loc[train_df["Title"] == "Mrs", "Title"] = 2
train_df.loc[train_df["Title"] == "Master", "Title"] = 3
train_df.loc[train_df["Title"] == "Other", "Title"] = 4

test_df.loc[test_df["Title"] == "Miss", "Title"] = 0
test_df.loc[test_df["Title"] == "Mr", "Title"] = 1
test_df.loc[test_df["Title"] == "Mrs", "Title"] = 2
test_df.loc[test_df["Title"] == "Master", "Title"] = 3
test_df.loc[test_df["Title"] == "Other", "Title"] = 4


# In[ ]:


set(train_df["Title"])


# In[ ]:


train_df.sample(5)


# # Training Data Model Building

# In[ ]:


features = ["Pclass", "Sex", "Age", "Embarked", "Fare", "FamSize", "Solo",
            "Title"]

X_train = train_df[features]
y_train = train_df["Survived"]
X_test_final = test_df[features]


# # Train/Test Split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 33)


# # SVC

# In[ ]:


svc_clf = SVC()
svc_clf.fit(X_train, y_train)
pred_svc_clf = svc_clf.predict(X_test)
accuracy_svc = accuracy_score(y_test, pred_svc_clf)

print(accuracy_svc)


# # Linear SVC

# In[ ]:


linsvc_clf = LinearSVC()
linsvc_clf.fit(X_train, y_train)
pred_linsvc_clf = linsvc_clf.predict(X_test)
accuracy_linsvc = accuracy_score(y_test, pred_linsvc_clf)

print(accuracy_linsvc)


# # Random Forest

# In[ ]:


rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
pred_rf_clf = rf_clf.predict(X_test)
accuracy_rf = accuracy_score(y_test, pred_rf_clf)

print(accuracy_rf)


# # Logistic Regression

# In[ ]:


logreg_clf = LogisticRegression()
logreg_clf.fit(X_train, y_train)
pred_logreg = logreg_clf.predict(X_test)
accuracy_logreg = accuracy_score(y_test, pred_logreg)

print(accuracy_logreg)


# # K-Neighbors

# In[ ]:


knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
pred_knn = knn_clf.predict(X_test)
accuracy_knn = accuracy_score(y_test, pred_knn)

print(accuracy_knn)


# # Gaussian Naive Bayes

# In[ ]:


gnb_clf = GaussianNB()
gnb_clf.fit(X_train, y_train)
pred_gnb = gnb_clf.predict(X_test)
accuracy_gnb = accuracy_score(y_test, pred_gnb)

print(accuracy_gnb)


# # Decision Tree

# In[ ]:


dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
pred_dt = dt_clf.predict(X_test)
accuracy_dt = accuracy_score(y_test, pred_dt)

print(accuracy_dt)


# # eXtreme Gradient Boosting (XGBoost)
# Since we haven't imported the XGBoost package, we will call it first.

# In[ ]:


from xgboost import XGBClassifier

xg_clf = XGBClassifier(objective="binary:logistic", n_estimators=38, seed=33)
xg_clf.fit(X_train, y_train)
pred_xgb = xg_clf.predict(X_test)
accuracy_xgb = accuracy_score(y_test, pred_xgb)

print(accuracy_xgb)


# After testing multiple parameters for n_estimators, it seems that n_estimators = 38 yields the best accuracy for XGBoost.

# # Evaluation based on accuracy scores

# In[ ]:


model_performance = pd.DataFrame({
    "Model": ["SVC", "Linear SVC", "Random Forest", 
              "Logistic Regression", "K Nearest Neighbors", "Gaussian Naive Bayes",  
              "Decision Tree", "XGBClassifier"],
    "Accuracy": [accuracy_svc, accuracy_linsvc, accuracy_rf, 
              accuracy_logreg, accuracy_knn, accuracy_gnb, accuracy_dt, 
                 accuracy_xgb]
})

model_performance.sort_values(by="Accuracy", ascending=False)


# Clearly, Random Forest performs the best given this dataset. I also noticed that in some iterations the XGBClassifier ran with a better score. So in my opinion both models perform well, with the Random Forest having a slightly better perfomance on an average.

# # Optimization using GridSearchCV

# In[ ]:


rf_clf = RandomForestClassifier()

parameters = {"n_estimators": [4, 5, 6, 7, 8, 9, 10, 15], 
              "criterion": ["gini", "entropy"],
              "max_features": ["auto", "sqrt", "log2"], 
              "max_depth": [2, 3, 5, 10], 
              "min_samples_split": [2, 3, 5, 10],
              "min_samples_leaf": [1, 5, 8, 10]
             }

grid_cv = GridSearchCV(rf_clf, parameters, scoring = make_scorer(accuracy_score))
grid_cv = grid_cv.fit(X_train, y_train)


# In[ ]:


print("GridSearchCV results:")
grid_cv.best_estimator_


# In[ ]:


rf_clf_optimized = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=10, max_features='log2', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=3,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
rf_clf_optimized.fit(X_train, y_train)
pred_rf_clf = rf_clf_optimized.predict(X_test)
accuracy_rf = accuracy_score(y_test, pred_rf_clf)

print(accuracy_rf)


# # Sumbit Results to Kaggle

# In[ ]:


submission_predictions = rf_clf_optimized.predict(X_test_final)


# In[ ]:


submission_predictions_df = pd.DataFrame(submission_predictions)
submission_predictions_df.count()


# In[ ]:


finalSubmission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": submission_predictions
    })

finalsubmission.to_csv("titanicSubmission.csv", index=False)
print(finalSubmission.shape)

