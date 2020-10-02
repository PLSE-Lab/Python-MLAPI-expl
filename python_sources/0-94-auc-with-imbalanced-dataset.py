#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ### **Loading dataset into pandas dataframe**

# In[ ]:


df_card = pd.read_csv('../input/creditcard.csv')


# ### Head of dataset

# In[ ]:


df_card.head()


# ### Describing the dataset

# In[ ]:


df_card[['Time', 'Amount']].describe()


# #### Checking the data types of dataframe columns

# In[ ]:


df_card.info()


# ### Very imbalanced classes. There are a lot more regular transactions than fraudulent
# 
# 1. 0    284315
# 2. 1       492

# In[ ]:


df_card['Class'].value_counts()


# In[ ]:


print("Fraudulent transactions account for {:.2f}% of the dataset"
      .format(df_card['Class'].value_counts()[1]/len(df_card)*100))


# ### Fraudulent transactions have a slightly higher mean value

# In[ ]:


df_card[['Amount', 'Class']].groupby('Class').mean()


# ### Even though regular transactions have a higher transaction amount

# In[ ]:


df_card[['Amount', 'Class']].groupby('Class').max()


# ### The transaction amount that repeated the most is a very small value: **$1.00**. Would it be fraudulent or not?

# In[ ]:


df_card['Amount'].value_counts()


# ### Sorting the fraudulent transactions by Amount

# In[ ]:


df_card[df_card['Class'] == 1][['Amount', 'Class']].sort_values(by='Amount', ascending=False).head(10)


# ### Now we can check that the most repeated amount for fraudulent transactions is $1.00! This could indicate that this is just a "checking" amount, a value used to test if the transaction is approved.
# 

# In[ ]:


df_card[df_card['Class'] == 1][['Amount', 'Class']]['Amount'].value_counts()


# ### **Exploratory Data Analysis (EDA)**

# In[ ]:


def get_transactions_average():
    ## Fraudulent transactions mean
    fraudulent_transactions_mean = df_card[df_card['Class'] == 1]['Amount'].mean()
    ## Regular transactions mean
    normal_transactions_mean = df_card[df_card['Class'] == 0]['Amount'].mean()
    ## Creating an array with the mean values
    return [fraudulent_transactions_mean, normal_transactions_mean]


# In[ ]:


# Get the mean values for each transaction type
mean_arr = get_transactions_average()
# Calculate the overall mean
overall_mean = df_card['Amount'].mean()


# #### Plotting the mean values in a bar plot. We can check the regular transactions are aroud the overall mean value, but the fraudulent ones are slightly above the mean

# In[ ]:


fig = plt.figure(figsize=(10, 8))
## Labels to replace the elements' indexes in the x-axis
xticks_labels = ['Fraudulent transactions', 'Regular transactions']
## X-axis elements
xticks_elements = [item for item in range(0,len(mean_arr))]
ax = plt.gca()
## Plot the bar char custom bar colors
plt.bar(xticks_elements, mean_arr, color='#2F4F4F')
## Map the xticks to their string descriptions, then rotate them to make them more readable
plt.xticks(xticks_elements, xticks_labels, rotation=70)
## Draw a horizontal line to show the overall mean to compare with each category's mean
plt.axhline(overall_mean, color='#e50000', animated=True, linestyle='--')
## Annotate the line to explain its purpose
ax.annotate('Overall Mean', xy=(0.5, overall_mean), xytext=(0.5, 110),
            arrowprops=dict(facecolor='#e50000', shrink=0.05))
## Set the x-axis label
plt.xlabel('Transactions')
## Set the y-axis label
plt.ylabel('Average amount in $ Dollar')
## Show the plot
plt.show()


# In[ ]:


# Describing the amount values for the fraulent transactions
describe_arr = df_card[df_card['Class'] == 1]['Amount'].describe()
describe_arr


# In[ ]:


## Creates a new figure
plt.figure(figsize=(10, 8))
## Filter out the fraudulent transactions from dataframe
df_fraudulent = df_card[df_card['Class']==1]
## Creates a boxplot from fraudulent transactions data
sns.boxplot(x="Class", y="Amount", 
                 data=df_fraudulent, palette='muted')
## Most values are clustered around small values, but the max transactions amount is smaller 
## than those from regular transactions


# In[ ]:


## Creates a new figure
plt.figure(figsize=(10, 8))
## Filter out the normal transactions from dataframe
df_regular = df_card[df_card['Class']==0]
## Creates a boxplot from the regular transactions data
sns.boxplot(x="Class", y="Amount", 
                 data=df_regular, palette='muted')

## Most transactions are grouped around small amounts 


# In[ ]:


## Creates a new figure 
plt.figure(figsize=(10, 8))
## Draw a distribution plot (histogram) from amount values
sns.distplot(df_card['Amount'], kde=True, hist=True, norm_hist=True)
## Check that most of the transactions are clustered around small values


# In[ ]:


## Creates a new figure 
plt.figure(figsize=(10, 8))
## Draw a distribution plot (histogram) from fraudulent transactions data
sns.distplot(df_fraudulent['Amount'], kde=True, hist=True, norm_hist=True)
## Check that most transactions are clustered around $0 and $500.


# In[ ]:


df_card.head()


# ### Preparing data for model 

# In[ ]:


## Dataset split import
from sklearn.model_selection import train_test_split


# In[ ]:


## Scale the amount feature before fitting the models
sc= StandardScaler()
df_card["scaled_amount"]=  sc.fit_transform(df_card.iloc[:,29].values.reshape(-1,1))
## Drops the old amount, once the scaled one has been added to the dataframe
df_card.drop('Amount', axis=1, inplace=True)


# In[ ]:


## Set the features to the X variable
X = df_card.drop(['Time', 'Class'], axis=1)
## Set the target column to the y_target variable
y_target = df_card['Class']


# ### Models

# In[ ]:


## Models and evaluation metrics imports
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# ## Model utility functions

# In[ ]:


## Split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y_target, random_state=42)


# In[ ]:


## This is a generic function to calculate the auc score which is used several times in this notebook
def evaluate_model_auc(model, X_test_parameter, y_test_parameter):
    ## The predictions
    y_pred = model.predict(X_test_parameter)
    ## False positive rate, true positive rate and treshold
    fp_rate, tp_rate, treshold = roc_curve(y_test_parameter, y_pred)
    ## Calculate the auc score
    auc_score = auc(fp_rate, tp_rate)
    ## Returns the score to the model
    return (auc_score)


# In[ ]:


## This is a generic function to plot the area under the curve (AUC) for a model
def plot_auc(model, X_test, y_test):
    ## Predictions
    y_pred = model.predict(X_test)
    
    ## Calculates auc score
    fp_rate, tp_rate, treshold = roc_curve(y_test, y_pred)
    auc_score = auc(fp_rate, tp_rate)
    
    ## Creates a new figure and adds its parameters
    plt.figure()
    plt.title('ROC Curve')
    ## Plot the data - false positive rate and true positive rate
    plt.plot(fp_rate, tp_rate, 'b', label = 'AUC = %0.2f' % auc_score)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


# In[ ]:


## This is a generic utility function to calculate a model's score
def evaluate_model_score(model, X_test, y_test):
    ## Return the score value to the model
    return model.score(X_test, y_test)


# In[ ]:


## This is a generic function to create a classification report and return it to the model. The target
## variables have been mapped to the transaction types
def evaluate_classification_report(model, y_test):
    return classification_report(y_test, model.predict(X_test), target_names=['Regular transaction',
                                                                      'Fraudulent transaction'])


# In[ ]:


## This utility function evaluates a model using some common metrics such as accurary and auc. Also, it
## prints out the classification report for the specific model
def evaluate_model(model_param, X_test_param, y_test_param):
    print("Model evaluation")
    print("Accuracy: {:.5f}".format(evaluate_model_score(model_param, X_test_param, y_test_param)))
    print("AUC: {:.5f}".format(evaluate_model_auc(model_param, X_test_param, y_test_param)))
    print("\n#### Classification Report ####\n")
    print(evaluate_classification_report(model_param, y_test_param))
    plot_auc(model_param, X_test_param, y_test_param)


# In[ ]:


## This is a shared function used to print out the results of a gridsearch process
def gridsearch_results(gridsearch_model):
    print('Best score: {} '.format(gridsearch_model.best_score_))
    print('\n#### Best params ####\n')
    print(gridsearch_model.best_params_)


# In[ ]:


# Returns the Random Forest model which the n_estimators returns the highest score in order to improve 
# the results of the default classifier
# min_estimator - min number of estimators to run
# max_estimator - max number of estimators to run
# X_train, y_train, X_test, y_test - splitted dataset
# scoring function: accuracy or auc
def model_selection(min_estimator, max_estimator, X_train_param, y_train_param,
                   X_test_param, y_test_param, scoring='accuracy'):
    scores = [] 
    ## Returns the classifier with highest accuracy score
    if (scoring == 'accuracy'):
        for n in range(min_estimator, max_estimator):
            rfc_selection = RandomForestClassifier(n_estimators=n, random_state=42).fit(X_train_param, y_train_param)
            score = evaluate_model_score(rfc_selection, X_test_param, y_test_param)
            print('Number of estimators: {} - Score: {:.5f}'.format(n, score))
            scores.append((rfc_selection, score))
            
    ## Returns the classifier with highest auc score
    elif (scoring == 'auc'):
         for n in range(min_estimator, max_estimator):
            rfc_selection = RandomForestClassifier(n_estimators=n, random_state=42).fit(X_train_param, y_train_param)
            score = evaluate_model_auc(rfc_selection, X_test_param, y_test_param)
            print('Number of estimators: {} - AUC: {:.5f}'.format(n, score))
            scores.append((rfc_selection, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)[0][0]


# 
# ### **Dealing with imbalanced classes**
# 

# In[ ]:


## Importing SMOTE 
from imblearn.over_sampling import SMOTE
## Importing resample
from sklearn.utils import resample


# ### SMOTE
# Models like RFC and SVC have a parameter that penalizes imbalanced datasets in order to get more accurate results. However, we are going to balance the data using a technique called SMOTE to create synthetic data points from the monirity class using KNearest Neighbors.

# In[ ]:


## Making a copy of the dataset (could've been done using df.copy())
dataset = df_card[df_card.columns[1:]]
## Defines the features to the dataset_features variable
dataset_features = dataset.drop(['Class'], axis=1)
## Defines the target feature to the dataset_target variable
dataset_target = dataset['Class']


# In[ ]:


## Split the data once again
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(dataset_features,
                                                   dataset_target,
                                                   random_state=42)


# In[ ]:


## This function generates a balanced X_train and y_train from the original dataset to fit the model
def get_balanced_train_data(df):
    sm = SMOTE(random_state=42, ratio = 1.0)
    X_train_res, y_train_res = sm.fit_sample(X_train_2, y_train_2)
    ## Returns balanced X_train & y_train
    return (X_train_res, y_train_res)


# In[ ]:


## Calling the function to get scalled training data
(X_train_resampled, y_train_resampled) = get_balanced_train_data(df_card)


# ## **SVM** 
# With default parameters

# In[ ]:


## Creating a SVC model with default parameters
svc = svm.SVC()
svc.fit(X_train_2, y_train_2)


# In[ ]:


## Evaluating the model
evaluate_model(svc, X_test_2, y_test_2)


# ### Cross validation with parameter tuning
# Setting parameters

# In[ ]:


## Parameters grid to be tested on the model
parameters = {
    'C': [1, 5, 10, 15],
    'degree':[1, 2, 3, 5],
    'kernel': ['linear'],
    'class_weight': ['balanced', {0:1, 1:10}, {0:1, 1:15}, {0:1, 1:20}],
    'gamma': [0.01, 0.001, 0.0001, 0.00001]
    }


# In[ ]:


## Creates a gridsearch to find the best parameters for this dataset.
clf = GridSearchCV(estimator=svm.SVC(random_state=42),
                   ## Passes the parameter grid as argument (these parameters will be tested
                   ## when this model is created)
                   param_grid=parameters,
                   ## Run the processes in all CPU cores
                   n_jobs=-1,
                   ## Set the scoring method to 'roc_auc'
                   scoring='roc_auc')


# In[ ]:


## Fit the gridsearch model to the data
# clf.fit(X_train_2[:5000], y_train_2[:5000])


# In[ ]:


## Find the model with the best score achieved and the best parameters to use
# gridsearch_results(clf)


# ### Using the optimal parameters

# In[ ]:


## Creates a SVC model with the optimal parameters found in the previous step
svc_grid_search = svm.SVC(C=1,
                          kernel='linear',
                          degree=1,
                          class_weight={0:1, 1:10},
                          gamma=0.01,
                          random_state=42)
svc_grid_search.fit(X_train_2[:5000], y_train_2[:5000])


# In[ ]:


## Evaluate the model
evaluate_model(svc_grid_search, X_test_2, y_test_2)


# ## **Random Forest Classifier**
# With default parameters

# In[ ]:


## Creates a Random Forest Classifier with default parameters
model_rfc = RandomForestClassifier().fit(X_train_2, y_train_2)


# In[ ]:


## Evaluate the model
evaluate_model(model_rfc, X_test, y_test)


# ### Random Forest Classifier presentes a good performance right out of the box, but can we improve it? Let's test using the balanced dataset using parameter tuning

# ### Selecting a model with best # of estimators

# In[ ]:


## Creating a model selecting the best number of estimators
rfc_model = model_selection(5, 15, X_train, y_train, X_test, y_test, scoring='auc')


# In[ ]:


## Evaluate the model
evaluate_model(rfc_model, X_test, y_test)


# ### Training with balanced dataset

# In[ ]:


## Select the model with the best number of estimators using the balanced dataset
rfc_smote = model_selection(5, 15, X_train_resampled, y_train_resampled,
                     X_test_2, y_test_2, scoring='auc')


# In[ ]:


## Evaluate the model with AUC metric
evaluate_model(rfc_smote, X_test_2, y_test_2)


# In[ ]:


## Show the most important features from the dataset
sorted(rfc_smote.feature_importances_, reverse=True)[:5]


# In[ ]:


## Itemgetter import
from operator import itemgetter


# In[ ]:


## Loading features and importance
features = [i for i in X.columns.values]
importance = [float(i) for i in rfc_smote.feature_importances_]
feature_importance = []

## Creating a list of tuples concatenating feature names and its importance
for item in range(0, len(features)):
    feature_importance.append((features[item], importance[item]))

## Sorting the list
feature_importance.sort(key=itemgetter(1), reverse=True)

## Printing the top 5 most important features
feature_importance[:5]


# ### **Grid search random forest**

# In[ ]:


## Parameters to use with the RFC model
parameters_rfc = { 
    'n_estimators': [5, 6, 7, 8, 9, 10, 13, 15],
#     'class_weight': ['balanced'],
    'max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}


# In[ ]:


## Gridsearch to get the best parameters for RFC
rfc_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42,
                                                               n_jobs=-1),
                               param_grid=parameters_rfc,
                               cv=10, 
                               scoring='roc_auc',
                               return_train_score=True)


# In[ ]:


## Train the gridsearch model
## Using only part of the dataset because the entire data takes too long to train; the same
## applies to the other models
## Takes too long
rfc_grid_search.fit(X_train_2[:10000], y_train_2[:10000])


# In[ ]:


## Check the results of cross validation
cv_results = pd.DataFrame(rfc_grid_search.cv_results_)
## Sort the values to get the best result
cv_results.sort_values(by='rank_test_score').head()


# In[ ]:


## Model with the best score and the best parameters
#
gridsearch_results(rfc_grid_search)


# ### Training model with optimal parameters

# In[ ]:


## RFC model using the parameters found by gridsearch 
rfc = RandomForestClassifier(random_state=42,
                            n_estimators=7, min_samples_leaf=1, max_depth=5)
## Fit the data
rfc.fit(X_train_2, y_train_2)


# In[ ]:


## Evaluate the model
evaluate_model(rfc, X_test_2, y_test_2)


# ### With balanced dataset

# In[ ]:


## Running gridsearch again to find the best results for the scalled dataset
rfc_grid_search_balanced = GridSearchCV(estimator=RandomForestClassifier(random_state=42,
                                                               n_jobs=-1),
                               param_grid=parameters_rfc,
                               cv=10,
                               scoring='roc_auc',
                               return_train_score=True)


# In[ ]:


## Fitting the data
## Takes too long
# rfc_grid_search_balanced.fit(X_train_resampled[:5000], y_train_resampled[:5000])


# In[ ]:


## Best score and best parameters
#gridsearch_results(rfc_grid_search_balanced)

#13 4 None


# It takes too long to run the gridsearch process on this model for this dataset. For this reason, I decided to run on my local machine and wait until the process finishes. After completed, I obtained the following parameters:
# 1. **n_estimators**: 13
# 2. **min_samples_leaf**: 4
# 3. **max_depth**: default (None)
# 
# Note that a lower **max_depth** parameter will lower the precision for the 'balanced' dataset.

# ### Training model with optimal parameters

# In[ ]:


## Creating a new model with the selected parameters
rfc_balanced = RandomForestClassifier(random_state=42, 
                            n_estimators=13, min_samples_leaf=4, max_depth=None)
rfc_balanced.fit(X_train_resampled, y_train_resampled)


# In[ ]:


## Evaluate the model
evaluate_model(rfc_balanced, X_test_2, y_test_2)


# ## **Logistic regression**

# In[ ]:


## Parameters grid for Logistic Regression model
param_grid_lreg = {
    'C': [0.001, 0.01, 0.1, 1, 10, 15],
    'class_weight': ['balanced', {0:1, 0:10}, {0:1, 1:15}, {0:1, 1:20}],
    'penalty': ['l1', 'l2']
}


# In[ ]:


## Running gridsearch to find best parameters for Logistic Regression model
lreg_grid_search = GridSearchCV(estimator=LogisticRegression(random_state=42),
                               param_grid=param_grid_lreg, cv=10, scoring='roc_auc')


# In[ ]:


## Fitting the data (it takes a long time)
# lreg_grid_search.fit(X_train_2[:2000], y_train_2[:2000])


# In[ ]:


## Printing the best results for this model
# gridsearch_results(lreg_grid_search)


# ### Parameter tuning 

# In[ ]:


## Creating a model with gridsearch parameters
lreg = LogisticRegression(C=1, penalty='l1', random_state=42,
                         class_weight={0:1, 1:10})
## Fitting the model
lreg.fit(X_train_2, y_train_2)


# In[ ]:


## Evaluate the model
evaluate_model(lreg, X_test_2, y_test_2)


# ## Final words
# As you can see, we can get good results with some models out of the box such as Random Forest Classifier. However, the imbalanced nature of this dataset might impact the overall result. For this reason, I've tried several classifier algorithms along with different parameter tuning and a varied set of evaluation metrics in order to achieve stable results.
# 
# This is just an example of how to use some basic machine learning techniques such as: data manipulation, EDA, data scaling, balancing (SMOTE), gridsearch, cross validation, and model evaluation. 

# In[ ]:




