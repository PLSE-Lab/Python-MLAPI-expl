#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Predicting heart disease using machine learning
# This notebook looks into using various Python-based machine learning and data science libraries in an attempt to build a machine learning model capable of predicting whether or not someone has heart disease based on their medical attributes.
# 
# We're going to take the following approach:
# 
# 1. Problem definition
# 2. Data
# 3. Evaluation
# 4. Features
# 5. Modelling
# 6. Experimentation

# ## 1. Problem Definition
# In a statement,
# 
# Given clinical parameters about a patient, can we predict whether or not they have heart disease?

# ## 2. Data
# The original data came from the Cleavland data from UCI Machine Learning Repository.
# 
# https://archive.ics.uci.edu/ml/datasets/Heart+Disease
# There is also a version of it available in Kaggle.
# 
# https://www.kaggle.com/ronitf/heart-disease-uci

# ## 3. Evaluation
# If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursue the project

# ## 4. Features
# This is where you'll get different information about each of the features in your data.
# 
# Create data dictionary
# 
# 1. age - age in years
# 2. sex(1 = male; 0 = female)
# 3. cp - chest pain type
# 4. trestbps - resting blood pressure (in mm Hg on admission to the hospital)
# 5. cholserum - cholestoral in mg/dl
# 6. fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 7. restecg - resting electrocardiographic results
# 8. thalach - maximum heart rate achieved
# 9. exang - exercise induced angina (1 = yes; 0 = no)
# 10. oldpeak - ST depression induced by exercise relative to rest
# 11. slope - the slope of the peak exercise ST segment
# 12. ca - number of major vessels (0-3) colored by flourosopy
# 13. thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
# 14. target - 1 or 0

# ## Preparing the tools
# We're going to use Panda, Matplotlib, NumPy for data analysis and manipulation.

# In[ ]:


# Import all the tools we need

# Regular EDA(exploratory data analysis) plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# We want our plots to appear inside of our notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Models from scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve


# ## Load data

# In[ ]:


df = pd.read_csv("../input/heart-disease-uci/heart.csv")
df.shape


# ## Data Exploration (EDA)
# The goal here is to find out more about the data and become a subject matter expert on the dataset you're working with.
# 
# 1. What questions are you trying to solve?
# 2. What kind of data do we have and how do we treat different types?
# 3. Whats missing from the data and how do you deal with it?
# 4. Where are the outliers and why should you care about them?
# 5. How can you add, change or remove features to get more out of your data?

# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


# Let's find out how many of each class there
df["target"].value_counts()


# In[ ]:


df["target"].value_counts().plot(kind="bar", color=["salmon", "lightblue"]);


# In[ ]:


df.info()


# In[ ]:


# Check for missing values
df.isna().sum()


# In[ ]:


df.describe()


# ## Heart Disease Frequency acording to Sex

# In[ ]:


df.sex.value_counts()


# In[ ]:


# Compare target columns with sex column
pd.crosstab(df.target, df.sex)


# In[ ]:


# Create a plot of crosstab
pd.crosstab(df.target, df.sex).plot(kind="bar",
                                   figsize=(10, 6),
                                   color=["salmon", "lightblue"]);
plt.title("Heart Disease Frequency for Sex")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female", "Male"])
plt.xticks(rotation = 0);


# In[ ]:


# Heart Disease Frequency per Chest Pain Type
df.cp.value_counts()


# In[ ]:


# Compare target column with chest pain type
pd.crosstab(df.cp, df.target)


# In[ ]:


# Creating another plot of crosstab
pd.crosstab(df.cp, df.target).plot(kind="bar",
                                  figsize=(10, 6),
                                  color=["salmon", "lightblue"]);
plt.title("Heart Disease Frequency for Chest Pain")
plt.xlabel("Chest Pain Type")
plt.ylabel("Amount")
plt.legend(["No Disease", "Disease"])
plt.xticks(rotation = 0);


# ## Age vs Max Heart Rate for Heart Disease

# In[ ]:


# Create another figure
plt.figure(figsize=(10, 6))

# Scatter with positive examples
plt.scatter(df.age[df.target==1],
           df.thalach[df.target==1],
           c="salmon");

# Scatter with negative examples
plt.scatter(df.age[df.target==0],
           df.thalach[df.target==0],
           c="lightblue");

# Add some helpfull info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);


# In[ ]:


# Check the distribution of the age column with a histogram
df.age.plot.hist();


# In[ ]:


df.head()


# In[ ]:


# Make a correlation matrix
df.corr()


# In[ ]:


# Let's make our correlation matrix a little prettier
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                annot=True,
                linewidths=0.5,
                fmt=".2f",
                cmap="YlGnBu")


# ## 5. Modelling

# In[ ]:


df.head()


# In[ ]:


# Split data into X and y
X = df.drop("target", axis = 1)
y = df["target"]


# In[ ]:


X


# In[ ]:


y


# In[ ]:


# Split data into train and test sets
np.random.seed(12)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.2)


# In[ ]:


X_train


# In[ ]:


y_train, len(y_train)


# Now we've got our data split into training & test sets, it's timme to build a machine learning model.
# 
# We'll train it (find the patterns) on the training sets.
# 
# And we'll test it (use the patterns) on the test sets.
# 
# We're going to try 3 different machine learning models:
# 
# 1. Logistic Regression
# 2. K-Nearest Neighbours Classifier
# 3. Random Forest Classifier

# In[ ]:


# Put models in a dictionary
models = {"Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier()}

# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learninf models.
    models : a dict of different Scikit-Learn machine learning models
    X_train : training data (no labels)
    X_test : testing data (no labels)
    y_train : training labels
    y_test : testing labels
    """
    # Set random seed
    np.random.seed(12)
    # Make a dictionary to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores


# In[ ]:


model_scores = fit_and_score(models=models,
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test)
model_scores


# # Model Comparison

# In[ ]:


model_compare = pd.DataFrame(model_scores, index=["accuracy"])
model_compare.T.plot.bar();


# Now we've got a baseline model... and we know a model's first predictions aren't always what we should based our next steps off.
# What should do?
# 
# Let's look at the following:
# * Hyperparameter tuning
# * Feature importance
# * Confussion matrix
# * Cross-validation
# * Precision
# * Recall
# * F1 score
# * Classification report
# * ROC curve
# * Area under the curve (AUC)
# 
# ### Hyperparameter tuning

# In[ ]:


# Let's tune KNN

train_scores = []
test_scores = []

# Create a list of different values for n_neighbors
neighbors = range(1, 21)

# Setup KNN instance
knn = KNeighborsClassifier()

# Loop through diffetent n_neighbors
for i in neighbors:
    knn.set_params(n_neighbors=i)
    
    # Fit the algorithm
    knn.fit(X_train, y_train)
    
    # Update the training scores list
    train_scores.append(knn.score(X_train, y_train))
    
    #Update the test scores list
    test_scores.append(knn.score(X_test, y_test))


# In[ ]:


train_scores


# In[ ]:


test_scores


# In[ ]:


plt.plot(neighbors, train_scores, label = "Train score")
plt.plot(neighbors, test_scores, label = "Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")


# ## Hyperparameter tuning with RandomizedSearchCV
# 
# We're going to tune:
# * LigisticRegression()
# * RandomForestClassifier()
# ... using RandomizedSearchCV

# In[ ]:


# Create a hyperparameter grid for LogisticRegression
log_reg_grid = {"C": np.logspace(-4, 4, 20),
               "solver": ["liblinear"]}

# Create a hyperparameter grid for RandomForestClassifier
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
          "max_depth": [None, 3, 5, 10],
          "min_samples_split": np.arange(2, 20, 2),
          "min_samples_leaf": np.arange(1, 20, 2)}


# Now we've got hyperparameters grids setup for each of our models, lets tune them using RandomizedSearchCV

# In[ ]:


# Turn LogisticRegression
np.random.seed(12)

# Setup random hyperparameters search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

# Fit random hyperparameter search model for LogisticRegression
rs_log_reg.fit(X_train, y_train)


# In[ ]:


rs_log_reg.best_params_


# In[ ]:


rs_log_reg.score(X_test, y_test)


# Now we've tuned LogisticRegression(), let's do the same for RandomForestClassifier()...

# In[ ]:


# Setup random seed
np.random.seed(12)

# Setup random hyperparameters search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                          param_distributions=rf_grid,
                          cv=5,
                          n_iter=100,
                          verbose=True)

# Fit random hyperparameters search model for RandomForestClassifier()
rs_rf.fit(X_train, y_train)


# In[ ]:


rs_rf.best_params_


# In[ ]:


rs_rf.score(X_test, y_test)


# ## Hyperparameters Tuning with GreadSearchCV

# In[ ]:


# Different hyperparameters for our LogisticRegression model
log_reg_grid = {"C": np.logspace(-4, 4, 30),
               "solver": ["liblinear"]}

# Setup grid hyperparameters search for LogisticRegression
gs_log_reg = GridSearchCV(LogisticRegression(),
                         param_grid=log_reg_grid,
                         cv=5,
                         verbose=True)

# Fit grid hyperparameters search model
gs_log_reg.fit(X_train, y_train)


# In[ ]:


gs_log_reg.best_params_


# In[ ]:


# Evaluate the grid search LogisticRegression Model
gs_log_reg.score(X_test, y_test)


# ## Evaluating our tuned machine learning classifier, beyond accuracy
# 
# * ROC curve nad AUC score
# * Confussion matrix
# * Classification report
# * Precision
# * Recall
# * F1-score
# 
# ...and it would be great if cross-validation was used where possible.
# 
# To make comparison and evaluate our trained model, first we need to make predictions.

# In[ ]:


# Make predictions with tuned model
y_preds = gs_log_reg.predict(X_test)


# In[ ]:


y_preds


# In[ ]:


y_test


# In[ ]:


# Plot ROC curve and calculate AUC metric
plot_roc_curve(gs_log_reg, X_test, y_test);


# In[ ]:


# Confussion matrix
print(confusion_matrix(y_test, y_preds))


# In[ ]:


sns.set(font_scale=1.5)

def plot_conf_mat(y_test, y_preds):
    """
    Plots a nice looking cinfusion matrix using Seaborn's heatmap()
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                    annot=True,
                    cbar=False)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    
plot_conf_mat(y_test, y_preds)


# Now we've got a ROC curve, an AUC metric and a confusion matrix, let's get a classification report as well as cross-validated precision, recall abd F1-score.

# In[ ]:


print(classification_report(y_test, y_preds))


# ### Calculate evaluation metrics using cross-validation
# 
# We're going to calculate accuracy, precision, recall and f1-score of our model using cross-validataion and to do so we'll be using 'cross_val_score'

# In[ ]:


# Check best hyperparameters
gs_log_reg.best_params_


# In[ ]:


# Create a new classifier with best parameters
clf = LogisticRegression(C=0.1082636733874054,
                        solver="liblinear")


# In[ ]:


# Cross-validated accuracy
cv_acc = cross_val_score(clf,
                        X,
                        y,
                        cv=5,
                        scoring="accuracy")
cv_acc=np.mean(cv_acc)
cv_acc


# In[ ]:


# Cross-validated precision
cv_prec = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="precision")
cv_prec=np.mean(cv_prec)
cv_prec


# In[ ]:


# Cross-validated recall
cv_recall = cross_val_score(clf,
                           X,
                           y,
                           cv=5,
                           scoring="recall")
cv_recall=np.mean(cv_recall)
cv_recall


# In[ ]:


# Cross-validated F1-score
cv_f1 = cross_val_score(clf,
                       X,
                       y,
                       cv=5,
                       scoring="f1")
cv_f1=np.mean(cv_f1)
cv_f1


# In[ ]:


# Visualize cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                         "Precision": cv_prec,
                         "Recall": cv_recall,
                         "F1-score": cv_f1},
                         index=[0])

cv_metrics.T.plot.bar(title="Cross-validated classification metrics",
                     legend=False);


# ### Feature Importance
# 
# Feature Importance is another as asking, "which features contributed dmost to the outcomes of the model and how did thet contribute?"
# 
# Finding featute importance is different for each machine learning model.
# One way to find feature importance is to search for("MODEL NAME") feature importance.
# 
# Let's find the feature importance for our LogisticRegression model...

# In[ ]:


# Fit an instance of LogisticRegression
gs_log_reg.best_params_

clf = LogisticRegression(C=0.1082636733874054,
                        solver="liblinear")

clf.fit(X_train, y_train)


# In[ ]:


# Check coef_
clf.coef_


# In[ ]:


# Match coef's of features to column
feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
feature_dict


# In[ ]:


# Visualize feature importance
feature_df = pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title="Feature importance",
                     legend=False);


# In[ ]:


pd.crosstab(df["sex"], df["target"])


# In[ ]:


pd.crosstab(df["slope"], df["target"])


# ## 6. Experimentation
# 
# * Could i collect more data?
# * Could i try a better model? Like CatBoost or XGBoost?
# * Could i improve the current model? (beyong what we've done so far)

# In[ ]:




