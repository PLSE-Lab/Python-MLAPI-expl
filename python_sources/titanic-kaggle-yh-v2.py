#!/usr/bin/env python
# coding: utf-8

# # Titanic Challenge for kaggle.com
# 
# Attempt number 2.
# 
# Our previous attempt has a score of 0.78947. We did not do any feature engineering, and simply run RandomForest classifier tuned with RandomSearchCV,
# 
# After reading around, especially https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner, I am inspired to try a few ways to improve the features and model.
# 
# 

# ## 1) Import libraries

# In[ ]:


import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# ## 2) Import train and test dataset

# In[ ]:


# data will be used for training and validation
# test will be used for final evaluation
data = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
data.head()


# In[ ]:


data.describe(include="all")


# In[ ]:


test.describe(include="all")


# ## 3) Initial Data Analysis

# ### our data
# the only difference between train and test dataset is that test dataset has **no** "Survived" label, since that's what we are trying to predict.

# In[ ]:


print("train dataset has " + str(len(data)) + " passengers")
print("test dataset has " + str(len(test)) + " passengers")


# In[ ]:


# exploring our feature data types
data.dtypes


# In[ ]:


# see how many feature and label data are missing for both train and test
pd.DataFrame({"train": data.isna().sum(),
              "test": test.isna().sum()})


# ### Data Summary 
# 
# 1. Raw features: 
# 
#   * Pclass, 
#   * Name, 
#   * Sex, 
#   * Age, 
#   * Sibsp (no. of siblings), 
#   * Parch (no. of parent/child), 
#   * Ticket, 
#   * Fare, 
#   * Cabin, 
#   * Embarked
# 
#  
# 2. We are trying to use our features to predict our label - "Survived"
# 
#  
# 3. The only difference between train and test dataset is that test dataset has no "Survived" label, since that's what we are trying to predict.
# 
# 
# 4. There are 891 dataset in train and 418 in test, which is on the low side.
# 
#  
# 5. Data is relatively complete - with most (77%) missing "Cabin" data, and some (20%) missing "Age", and a handful missing "Embarked" and "Fare".
# 
# 6. Features Data Type:
# 
#   * Numerical: Age (Continuous), Fare (Continuous), SibSp (Discrete), Parch (Discrete)
#   * Categorical: Survived, Sex, Embarked, Pclass, Ticket, Cabin

# ## 4) Feature Analysis
# 
# We want to go through each feature to get a sense of the data and form some baseline hypothesis

# ### Sex Feature
# We want to see if there is a difference in survival rate between sexes

# In[ ]:


sns.barplot(x="Sex", y="Survived", data=data);


# In[ ]:


print("Percentage of females who survived:", str(round(data["Survived"][data["Sex"] == 'female'].value_counts(normalize = True)[1]*100, 2)),"%")

print("Percentage of males who survived:", str(round(data["Survived"][data["Sex"] == 'male'].value_counts(normalize = True)[1]*100, 2)),"%")


# "Sex" appears to be an important feature. Female has a significantly higher rate of survival.

# ### Age Feature

# In[ ]:


senior = data["Age"][data["Age"] > 60].count()
adult = data["Age"][data["Age"] <= 60][data["Age"] > 40].count()
youngadult = data["Age"][data["Age"] <= 40][data["Age"] > 20].count()
teenager = data["Age"][data["Age"] <= 20][data["Age"] > 12].count()
child = data["Age"][data["Age"] <= 12][data["Age"] > 4].count()
toddler = data["Age"][data["Age"] <= 4][data["Age"] > 1].count()
baby = data["Age"][data["Age"] <= 1][data["Age"] >= 0].count()
missing = data["Age"].isna().sum()
total = senior+adult+youngadult+teenager+child+toddler+baby+missing

print("Passengers by Age Group in train set")
print("senior:", senior)
print("adult:", adult)
print("youngadult:", youngadult)
print("teenager:", teenager)
print("child:", child)
print("toddler:", toddler)
print("baby:", baby)
print("missing:", missing)
print("total:", total)


# In[ ]:


print("Percentage of senior who survived:", str(round(data["Survived"][data["Age"] > 60].value_counts(normalize = True)[1]*100, 2)),"%")
print("Percentage of adult who survived:", str(round(data["Survived"][data["Age"] <= 60][data["Age"] > 40].value_counts(normalize = True)[1]*100, 2)),"%")
print("Percentage of youngadult who survived:", str(round(data["Survived"][data["Age"] <= 40][data["Age"] > 20].value_counts(normalize = True)[1]*100, 2)),"%")
print("Percentage of teen who survived:", str(round(data["Survived"][data["Age"] <= 20][data["Age"] > 12].value_counts(normalize = True)[1]*100, 2)),"%")
print("Percentage of child who survived:", str(round(data["Survived"][data["Age"] <= 12][data["Age"] > 4].value_counts(normalize = True)[1]*100, 2)),"%")
print("Percentage of toddler who survived:", str(round(data["Survived"][data["Age"] <= 4][data["Age"] > 1].value_counts(normalize = True)[1]*100, 2)),"%")
print("Percentage of baby who survived:", str(round(data["Survived"][data["Age"] <= 1][data["Age"] > 0].value_counts(normalize = True)[1]*100, 2)),"%")
print("Percentage of age-missing who survived:", str(round(data["Survived"][data["Age"].isna()].value_counts(normalize = True)[1]*100, 2)),"%")


# A few things to note:
# 1. younger passenger has a higher rate of survival - which makes sense since younger passengers might be prioritised in rescue boat
# 2. those with age missing has lower survival rate than other age group except for senior. Perhaps we can put them in a category on their own.

# ### Pclass feature
# Pclass refers to the passenger class - 1,2,3 with 1 being the highest.

# In[ ]:



sns.barplot(x="Pclass", y="Survived", data=data)

#print percentage of people by Pclass that survived
print("Percentage of Pclass = 1 who survived:", data["Survived"][data["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", data["Survived"][data["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", data["Survived"][data["Pclass"] == 3].value_counts(normalize = True)[1]*100)


# ### Fare feature
# We can test if passengers paying higher fare has higher rate of survival.
# We can also see if it makes sense to group fare into different categories; it is now a continuous variable

# In[ ]:


data["Fare"].min(), data["Fare"].max(), data["Fare"].mean()


# In[ ]:


data["Fare"].plot.hist();


# In[ ]:


# grouping fare
high_fare = data["Fare"][data["Fare"] > 200].count()
mid_fare = data["Fare"][data["Fare"] <= 200][data["Fare"] > 50].count()
low_fare = data["Fare"][data["Fare"] <= 50][data["Fare"] > 25].count()
dirt_fare = data["Fare"][data["Fare"] <= 25].count()

# print("high: ", high_fare)
high_fare, mid_fare, low_fare, dirt_fare


# In[ ]:


# testing hypothesis: percentage of high-fare group who survived
print("Percentage of high-fare who survived:", data["Survived"][data["Fare"] > 200].value_counts(normalize = True)[1]*100)
print("Percentage of mid-fare who survived:", data["Survived"][data["Fare"] <= 200][data["Fare"] > 50].value_counts(normalize = True)[1]*100)
print("Percentage of low-fare who survived:", data["Survived"][data["Fare"] <= 50][data["Fare"] > 25].value_counts(normalize = True)[1]*100)
print("Percentage of dirt-fare who survived:", data["Survived"][data["Fare"] <= 25].value_counts(normalize = True)[1]*100)


# Higher-fared passengers have significantly higher survival rate

# ### SibSp features
# Number of siblings. Not sure what hypothesis we can form out of this, lets explore!

# In[ ]:


# survival rate by no. of siblings
sns.barplot(x="SibSp", y="Survived", data=data);


# There's a marked difference between those with 1-2 siblings vs those with more or 0 sibling.
# 
# Perhaps when a family has many siblings, someone needed to sacrifice.

# ### Parch feature
# No. of Parent/Child
# 

# In[ ]:


sns.barplot(x="Parch", y="Survived", data=data);


# #### Cabin feature
# 
# Cabin has the most missing data. And its data might overlap with "Pclass"

# In[ ]:


has_cabin = data["Cabin"][data["Cabin"] != None].count()
no_cabin = data["Cabin"].isna().sum()
has_cabin, no_cabin


# In[ ]:


# explore survival rate of those with recorded cabins vs without

print("Percentage of has_cabin who survived:", data["Survived"][data["Cabin"] != None].value_counts(normalize = True)[1]*100)

print("Percentage of no_cabin who survived:", data["Survived"][data["Cabin"].isna()].value_counts(normalize = True)[1]*100)


# There seems to be a difference between those with recorded cabins and those without. We can consider grouping them into boolean of 1 and 0.

# In[ ]:


# explore survival rate by cabin (A to G)
# we are filling up missing with "Z" since str.contains doesn't work on missing value
data["Cabin"].fillna("Z", inplace=True)


# In[ ]:


cabin_class = ["A", "B", "C", "D", "E", "F", "G", "Z"]

for cabin in cabin_class:
    print("% of cabin class", cabin, "who survived: ", 
          str(round(data["Survived"][data["Cabin"].str.contains(cabin)].value_counts(normalize = True)[1]*100)), "%",
          "out of ", data["Cabin"].str.contains(cabin).sum(), "passengers")


# The sample size for each cabin size is pretty small, so it might not make sense to classify based on different classes since it might result in overfitting? So for now, we will stick to classifying by has_cabin and no_cabin.

# ### Embarked feature
# Embarked means where did the passenger come on board - there are 3 categories: Southampton (S), Cherbourg (C), and Queenstown (Q).
# 
# 

# In[ ]:


print("Number of people embarking in Southampton (S):", data["Embarked"][data["Embarked"] == "S"].count())

print("Number of people embarking in Cherbourg (C):", data["Embarked"][data["Embarked"] == "C"].count())

print("Number of people embarking in Queenstown (Q):", data["Embarked"][data["Embarked"] == "Q"].count())


# since the majority of people embarked from S, we will fill up missing value with S as well

# In[ ]:


print("Percentage of S who survived:", data["Survived"][data["Embarked"] == "S"].value_counts(normalize = True)[1]*100)
print("Percentage of C who survived:", data["Survived"][data["Embarked"] == "C"].value_counts(normalize = True)[1]*100)
print("Percentage of Q who survived:", data["Survived"][data["Embarked"] == "Q"].value_counts(normalize = True)[1]*100)


# ### Name feature
# 
# On first glance, name should not matter in a passenger's survival rate.
# 
# However, our name data contains titles such as "Countess, Rev, Don, Master" should might be a signifer for social status.

# In[ ]:


#create a combined group of both datasets
combine = [data, test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(data['Title'], data['Sex'])


# In[ ]:


# test our hypothesis if certain titles have higher survival rate
# note that sample size is limited, so we will only be testing a few

print("Master", data["Survived"][data["Title"] == "Master"].value_counts(normalize = True)[1]*100)
print("Miss", data["Survived"][data["Title"] == "Miss"].value_counts(normalize = True)[1]*100)
print("Mrs", data["Survived"][data["Title"] == "Mrs"].value_counts(normalize = True)[1]*100)
print("Dr", data["Survived"][data["Title"] == "Dr"].value_counts(normalize = True)[1]*100)


# From a quick look, it seems that female titles have higher survival rate. 
# 
# Since this information is already captured by "Sex" feature, we will be dropping title and name.

# ## 5) Feature Engineering
# 
# Feature Engineering is the process of transforming raw data into features that better represent the underlying problem to the predictive model
# 
# Now that we have explored our data, we need to prepare our data for analysis:
# 1. Decide what features we want to drop / add
# 2. Decide what features we want to make changes: e.g. grouping of cabin, fare, age 
# 3. fill missing values for Cabin, Age, Embarked, Fare
# 4. change non-numerical values to numerical values

# In[ ]:


#get a list of the features within the dataset
print(data.columns)
print(test.columns)


# ### Splitting our data into X and y 
# 
# * y is our label - "Survived"
# * X is our features

# In[ ]:


# Split into X and y
X = data.drop("Survived", axis=1)
y = data["Survived"]


# ### Build data transform pipeline
# We want to build a function that can transform our data - so that we can use the same function to apply to train, validation, and eventually test set,

# In[ ]:


# writing our function

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def transform_data(data):
    # create new feature: HasCabin
    data["Cabin"].fillna("Z", inplace=True) # we previously filled missing with Z on train, but haven't done for test
    data["HasCabin"] = data["Cabin"].str.contains("Z").astype('int')
    
    # create new feature: AgeGroup
    data["Age"] = data["Age"].fillna(-0.5)
    bins = [-1, 0, 1, 4, 12, 20, 40, 60, np.inf]
    labels = ['AgeMissing', 'Baby', 'Toddler', 'Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']
    data['AgeGroup'] = pd.cut(data["Age"], bins, labels = labels)    

    # create new feature: FareGroup
    data["Fare"] = data["Fare"].fillna(-0.5)
    bins_2 = [-1, 0, 25, 50, 200, np.inf]
    labels_2 = ['FareMissing', 'DirtFare', 'LowFare', 'MidFare', 'HighFare']
    data['FareGroup'] = pd.cut(data["Fare"], bins_2, labels = labels_2)     
    
    # remove "PassengerId", "Name", "Ticket", "Title", "Cabin", "Age", "Fare"
    data = data.drop("PassengerId", axis=1)
    data = data.drop("Name", axis=1)
    data = data.drop("Ticket", axis=1)
    data = data.drop("Title", axis=1)
    data = data.drop("Cabin", axis=1)
    data = data.drop("Age", axis=1)
    data = data.drop("Fare", axis=1)                                                            
    
    # fill na with pandas
    # HasCabin, AgeGroup, and FareGroup already filled above, so only need to fill Embarked
    data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)
    
    # transform "Sex", "Embark", "AgeGroup", "FareGroup", "Pclass", "HasCabin"
    # One Hot Encoding transforms categories into different feature columns of 1 and 0
    # note even though "Pclass" data is in numbers (1,2,3), they are Categorical features (instead of numerical like Age) hence we need to encode it
    one_hot = OneHotEncoder()
    transformer = ColumnTransformer([("one_hot",
                                      one_hot,
                                      ["Sex", "Embarked", "AgeGroup", "FareGroup", "Pclass", "HasCabin"])],
                                    remainder="passthrough")
    data = transformer.fit_transform(data)
    
    return data


# ### Transform on X to get X_tf

# In[ ]:


# applying our function to transform X_train

X_tf = transform_data(X)
pd.DataFrame(X_tf)
# note this converts to numpy array, and not pd


# Lets see our transformed data is done correctly.
# 
# From above, our transformed data has 25 columns (starting from 0).
# 
# It should have:
# 
# * 1x "SibSp",
# * 1x "Parch",
# * 2x "Sex", 
# * 3x "Embarked", 
# * 8x "AgeGroup", 
# * 5x "FareGroup", 
# * 3x "Pclass",
# * 2x "HasCabin"

# In[ ]:


# number of columns we should have
1+1+2+3+8+5+3+2


# In[ ]:


# check that number of rows is intact
len(X_tf), len(X)


# In[ ]:


# check if any missing value
pd.DataFrame(X_tf).isna().sum()


# ### Splitting data train and validation
# 
# In general, it's good practice to split data into training and validation set **before** applying any feature engineering.
# 
# But since our sample size is small, we decide to split after feature engineering is applied.

# In[ ]:


# Split data into train and validation sets
from sklearn.model_selection import train_test_split
np.random.seed(17)
X_train, X_val, y_train, y_val = train_test_split(X_tf, y, test_size=0.2)


# In[ ]:


# confirm splitting is done right
len(X_train), len(X_val), len(y_train), len(y_val)


# ## 6) Modeling
# 
# We're going to try 4 different machine learning models that are used for Classification problem:
# 
# 1. Logistic Regression
# 2. K-Nearest Neighbours Classifier
# 3. Random Forest Classifier
# 4. Gradient Boosting Classifier
# 5. Support Vector Machines
# 6. Stochastic Gradient Descent
# 7. Decision Tree
# 
# We are using training data (X_train, y_train) for **fitting / training** the model 
# 
# Then, we use validation data (X_val, y_val) to **validate and evaluate** how accuracy our model is.
# * we use default `score` method of each model to get accuracy score
# * in essense, what `score` method does is it predicts on X_val data and produces a y_prediction data. 
# * and then, it compares y_prediction (Machine's guess) with y_val (actual answer) to see how accurate is the prediction

# In[ ]:


# Creating a function to fit and score across the models

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Put models in a dictionary
models = {"Logistic Regression": LogisticRegression(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier(),
          "Gradient Boosting": GradientBoostingClassifier(),
          "SVC": SVC(),
          "SGD": SGDClassifier(),
          "Decision Tree": DecisionTreeClassifier()}


def fit_and_score(models, X_train, X_val, y_train, y_val):
    """
    Fits and evaluates given machine learning models.
    models : a dict of differetn Scikit-Learn machine learning models
    """
    # Set random seed
    np.random.seed(17)
    # Make a dictionary to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_val, y_val)
    return model_scores


# In[ ]:


# Executing our function, and we will see the "accuracy" score (using default scoring method of each model)
fit_and_score(models=models,
              X_train=X_train,
              X_val=X_val,
              y_train=y_train,
              y_val=y_val)


# ## 7) Evaluation 
# 
# ### Additional Metrics to evaluate different models
# 
# We want to have more metrics in addition to "accuracy" to give us better assess performance of each model.
# 
# For now, we will evaluate our 3 most accurate models - Random Forest, SVC, Decision Tree

# In[ ]:


# creating a evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_preds(y_true, y_preds):
    """
    perform evaluation comparison on y_true labels vs y_preds labels
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    metric_dict = {"accuracy": round(accuracy, 2),
                    "precision": round(precision, 2),
                    "recall": round(recall, 2),
                    "f1": round(f1, 2)}
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")
    return metric_dict


# In[ ]:


# RandomForest

np.random.seed(19)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_preds = rf.predict(X_val)  # ML's prediction using X_val

# evaluate using our evaluation function on validation set
rf_metrics = evaluate_preds(y_val, y_preds)  # compares y_preds with y_val/y_true
rf_metrics


# In[ ]:


# SVC

np.random.seed(19)
svc = SVC()
svc.fit(X_train, y_train)
y_preds = svc.predict(X_val)  # ML's prediction using X_val

# evaluate using our evaluation function on validation set
svc_metrics = evaluate_preds(y_val, y_preds)  # compares y_preds with y_val/y_true
svc_metrics


# In[ ]:


# Decision Tree

np.random.seed(19)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_preds = decision_tree.predict(X_val)  # ML's prediction using X_val

# evaluate using our evaluation function on validation set
decision_tree_metrics = evaluate_preds(y_val, y_preds)  # compares y_preds with y_val/y_true
decision_tree_metrics


# In[ ]:


# Lets compare our different models with new metrics

compare_metrics = pd.DataFrame({"Random Forest": rf_metrics,
                                "SVC": svc_metrics,
                                "Decision Tree": decision_tree_metrics})
compare_metrics.plot.bar(figsize=(10,8));


# ### Result:
# 
# There is a whole science in making sense of the different metrics.
# 
# 
# 1)
# 
# In general, if there is **class imbalance**, precision and recall are more important than accuracy.
# 
# Class imbalance means e.g. in "Sex", if our data has 95% male and 5% female, it makes it difficult for machine learning to predict accurately especially for minority class.
# 
# In our data, due to our small sample size, there are quite a few features that have imbalanced classes, e.g. Pclass, FareGroup, etc.
# 
# 
# 2)
# 
# If false positive predictions are worse than false negatives (depend on the project), aim for higher precision. If false negative predictions are worse than false positives, aim for higher recall.
# 
# For us, we are no preference on false positive or false negative, since we are not predicting if a patient has heart disease, for instance.
# 
# 
# 3)
# In our case, both Random Forest and SVC are very close in metrics.
# 
# * they are equal in F1-score. F1-score is a combination of precision and recall.
# * SVC has higher accuracy.
# 
# Thus, we will be using SVC for further tuning.

# ## 8) Tuning Hyperparameters of our Model
# 
# Tuning hyperparameters helps to optimise the performance of a model. 
# 
# Hyperparameters are parameters that are external to our model's learning that can potentially helps improve its ability to predict.
# 
# We will be using RandomSearchCV and GridSearchCV for tuning.

# In[ ]:


# check available hyperparameters for SVC
svc.get_params()


# In[ ]:


# tuning hyperparameters by RandomSearchCV

from sklearn.model_selection import RandomizedSearchCV
grid = {"kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
        "degree": [0,1,2,3,4,5,6],
        "class_weight": ["balanced", None],
        "C": [100, 10, 1.0, 0.1, 0.001]}

np.random.seed(17)

svc = SVC()
rs_svc = RandomizedSearchCV(estimator=svc,
                            param_distributions=grid,  # what we defined above
                            n_iter=10, # number of combinations to try
                            cv=5,   # number of cross-validation split
                            verbose=2)
rs_svc.fit(X_train, y_train);


# In[ ]:


# checking out best parameters we find from tuning
rs_svc.best_params_


# In[ ]:


# evaluating our model tuned with RandomSearchCV
rs_y_preds = rs_svc.predict(X_val)

# evaluate predictions
rs_metrics = evaluate_preds(y_val, rs_y_preds)


# In[ ]:


# tuning hyperparameters by GridSearchCV
from sklearn.model_selection import GridSearchCV


grid_2 = {"kernel": ["linear", "rbf"],
        "gamma": ["scale"],
        "degree": [1,2,3],
        "class_weight": [None],
        "C": [100, 10, 1.0]}

np.random.seed(17)

svc = SVC()

# Setup GridSearchCV
gs_svc = GridSearchCV(estimator=svc,
                      param_grid=grid_2,
                      cv=5,
                      verbose=2)

# Fit the GSCV version of clf
gs_svc.fit(X_train, y_train);


# In[ ]:


gs_svc.best_params_


# In[ ]:


# evaluating our model tuned with GridSearchCV
gs_y_preds = gs_svc.predict(X_val)

# evaluate predictions
gs_metrics = evaluate_preds(y_val, gs_y_preds)


# In[ ]:


# Lets compare our different model metrics

compare_metrics = pd.DataFrame({"baseline SVC": svc_metrics,
                                "random search": rs_metrics,
                                "grid search": gs_metrics})
compare_metrics


# In[ ]:


# with bar graph
compare_metrics.plot.bar(figsize=(10,8));


# Admittedly, this is my first time tuning SVC model. So I have no idea if I'm doing this right.
# 
# The results are close between without tuning and tuning wih RandomSearchCV. We'll be using the latter since it gives a higher F1-score.

# ## 9) Testing and Submitting
# 
# Our final objective is use our best model to predict on test data (with no labels), and upload our prediction to Kaggle to evaluate.
# 
# Our model is choice is SVC tuned with RandomSearchCV.

# In[ ]:


# our raw test data
test.head()


# In[ ]:


# tranforming test data
test_tf = transform_data(test)
pd.DataFrame(test_tf)


# In[ ]:


# check no missing value
pd.DataFrame(test_tf).isna().sum()


# In[ ]:


# run our prediction with our rs_svc model

test_preds = rs_svc.predict(test_tf)
pd.DataFrame(test_preds)


# In[ ]:


submission = pd.DataFrame({"PassengerId": test["PassengerId"], 
                          "Survived": test_preds})
submission


# In[ ]:


# check length

len(submission), len(test)


# In[ ]:


submission.to_csv("submission_v2.csv", index=False)


# ### 10) After thoughts
# 
# Our submission has an accuracy score of 0.77033.
# 
# Interestingly, our previous submission using RandomForest without doing any feature engineering received a higher score of 0.78947.
# 
# So looks like all our deliberate tinkering has actually made a worse model!
# 
# Credits to Nadin Tamer for sharing her very beginner-friendly notebook: https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner 

# In[ ]:




