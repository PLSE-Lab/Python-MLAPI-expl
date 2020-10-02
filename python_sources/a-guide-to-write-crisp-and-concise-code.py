#!/usr/bin/env python
# coding: utf-8

# ### Motivation
# 
# Bit late to the party but the COVID crisis has given me sufficient time to do the things I love to do :) That being said I have a few other ulterior motives as well releasing this kernel :D. What I have noticed in my career in Data Science, is that beginners or enthusiasts who want to step into this field are often mislead by bootcamp courses offering them to magically transform them into a competent data science in **X** days/months (insert relevant term advertised by specific bootcamp company). Well this kernel aims to showcase two things on the Heart Disease Prediction dataset:
# 
# 1. A starter kernel showing basic steps in a classification problem. Although, even kaggle datasets do not really portray the true picture of messy datasets in the industry, but I have tried to depict my typical process approaching a classification problem.
# 
# 2. Coding conventions! Even some experienced coders not adhering to this. Typically, I would have even broken this kernel down to 3 different kernels - Data Prep, EDA, Modelling. But for the sake of posterity, I have made a single kernel. Comments have been inserted at relevant places to explain the logic, docstrings for functions have been added, markdowns to segment the code have all been done to show newcomers the importance of writing structured and clean code. Markdowns are your friend, especially if you use kernels/notebooks!
# 
# Hopefully, this kernel helps someone! 
# 
# **PS : You can play around with the code and achieve a higher accuracy than the one in this kernel, but that was not the intent of this kernel**

# ### 1. Import required libraries
# Keep all your imports at the top! This clearly shows the reader which libraries you have used.

# In[ ]:


# Selective library imports
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Selectively import functions
from math import sqrt
from IPython.display import display
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report, confusion_matrix


# ### 2. Set configurations
# 
# You would want to specify the settings for your kernel/notebook at the top. Also specify the constants (if any) like paths to files. Here, since its only one file we are working with, this is not necessary.

# In[ ]:


# Disable warnings. This is not a good practice generally, but for the sake of aesthetics we are disabling this :D
warnings.filterwarnings("ignore")

# Suppress scientific notation
pd.options.display.float_format = '{:20,.2f}'.format

# Set plot sizes
sns.set(rc={'figure.figsize':(11.7,8.27)})

# Set plotting style
plt.style.use('fivethirtyeight')


# ### 2. Write User-Defined Functions
# 
# My rule is, if I have to repeat a piece of code more than twice, I functionize them. Always include a docstring with your function. Here, all the data processing/modelling related functions have a lot of parameters which enable the reader to experiment with a variety of settings. Need a different train/test split? No problem, change the random_state in the function! Need a different scorer? Not to worry! Use the parameter scorer in the functions...

# In[ ]:


def plot_dist(df, var, target, var_type='num'):
    
    '''Function helper to facet on target variable'''
    
    if var_type == 'num':
        sns.distplot(df.query('target == 1')[var].tolist() , color="red", label="{} for target == 1".format(var))
        sns.distplot(df.query('target == 0')[var].tolist() , color="skyblue", label="{} for target == 0".format(var))
        plt.legend()
        
    else:
        fig, ax = plt.subplots(1,2)
        sns.countplot(data=df.query('target == 1') , color="salmon", x=var, label="{} for target == 1".format(var), ax=ax[0])
        sns.countplot(data=df.query('target == 0') , color="skyblue", x=var, label="{} for target == 0".format(var), ax=ax[1])
        fig.legend()
        fig.show()


# In[ ]:


def process_data(df, test_size=0.3, random_state=1, scale=True, scaler=MinMaxScaler(), feature_selection=True, k=10):
    
    '''Function helper to generate train and test datasets and apply transformations if any'''
    
    
    # Dummify columns
    dummy_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
    df = pd.get_dummies(df, columns=dummy_cols)
    
    
    # All the columns
    cols = df.columns.tolist()
    
    # X cols
    cols = [col for col in cols if 'target' not in col] 
    
    # Subset x and y
    X = df[cols]
    y = df['target']
    
    # Feature selection
    if feature_selection == True:
        
        k_best = SelectKBest(score_func=chi2, k=k)
        selector = k_best.fit(X, y)
        selection_results = pd.DataFrame({'feature' : cols, 'selected' : selector.get_support()})
        selected_features = list(selection_results.query('selected == True')['feature'])
        X = X[selected_features]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    
    # Make a copy to apply on. Else Set-copy warning will be displayed
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()

    # Scale columns if needed
    if scale == True:
        scale_cols = ['age', 
                      'trestbps', 
                      'chol', 
                      'thalach', 
                      'oldpeak']
        
        # If any features are dropped from feature selection we need to account for that
        scale_cols = list(set(selected_features) & set(scale_cols))
        
        # Define scaler to use
        scaler = scaler

        # Apply scaling
        X_train_copy.loc[:, scale_cols] = scaler.fit_transform(X_train[scale_cols])
        X_test_copy.loc[:, scale_cols] = scaler.transform(X_test[scale_cols])
      
    # Return train and tests
    return X_train_copy, X_test_copy, y_train, y_test


# In[ ]:


def select_model(X_train, y_train, cv=3, nruns=3, scorer='recall'):
    
    '''Function helper to automate selection of best baseline model without hyperparameter tuning'''

    record_scorer = []
    iter_scorer = []
    model_name = []
    model_accuracy = []

    # Specify estimators
    estimators = [('logistic_regression' , LogisticRegression()), 
                  ('random_forest' , RandomForestClassifier(n_estimators=100)),
                  ('lightgbm' , LGBMClassifier(n_estimators=100)), 
                  ('xgboost' , XGBClassifier(n_estimators=100))]


    scorer = scorer
    
    # Iterate through the number of runs. Default is 3.
    for run in range(nruns):
        print('Running iteration %s with %s as scoring metric' % ((run + 1), scorer))

        for name, estimator in estimators:

            print('Fitting %s model' % name)

            # Run cross validation
            cv_results = cross_val_score(estimator, X_train, y_train, cv=cv, scoring=scorer)

            # Append all results in list form which will be made into a dataframe at the end.
            iter_scorer.append((run + 1))
            record_scorer.append(scorer)
            model_name.append(name)
            model_accuracy.append(cv_results.mean())

        print()

    # Use ordered dictionary to set the dataframe in the exact order of columns declared.
    results = pd.DataFrame(OrderedDict({'Iteration' : iter_scorer, 
                                        'Scoring Metric' : record_scorer, 
                                        'Model' : model_name, 
                                        'Model Accuracy' : model_accuracy}))
    
    # Pivot to view results in a more aesthetic form
    results_pivot = results.pivot_table(index=['Iteration', 'Scoring Metric'], columns=['Model'])
    
    # Display the results
    print('\nFinal results : ')
    display(results_pivot)

    # Get the mean performance
    performance = results_pivot.apply(np.mean, axis=0)
    performance = performance.reset_index()
    performance.columns = ['metric', 'model', 'performance']
    
    # Get the mean performance
    performance = results_pivot.apply(np.mean, axis=0)
    performance = performance.reset_index()
    performance.columns = ['metric', 'model', 'performance']
    best_model = performance.loc[performance['performance'].idxmax()]['model']

    # Return the pivot 
    return results_pivot, best_model


# In[ ]:


def tune_model(X_train, X_test, y_train, y_test, best_model, scorer='recall'):
    
    # Define parameters for each model
    grid = {'logistic_regression' : {'model' : LogisticRegression(class_weight='balanced', random_state=42), 
                                    'params' : {'C' : [0.01, 0.1, 1, 10, 100]}},

            'random_forest' : {'model' : RandomForestClassifier(class_weight='balanced', random_state=42), 
                            'params' : {'n_estimators' : [100, 200, 300], 
                                        'max_depth' : [3, 5, 7], 
                                        'max_features' : ['log2', 5, 'sqrt']}},

            'lightgbm' : {'model' : LGBMClassifier(class_weight='balanced', random_state=42), 
                        'params' : {'n_estimators' : [100, 200, 300], 
                                    'max_depth' : [3, 5, 7], 
                                    'boosting_type' : ['gbdt', 'dart', 'goss']}},

            'xgboost' : {'model' : XGBClassifier(nthread=-1), 
                        'params' : {'n_estimators' : [100, 200, 300], 
                                    'max_depth' : [3, 5, 7], 
                                    'scale_pos_weight' : [5, 10, 20]}}                        
                                
        }

    # Select the best model
    model = grid[best_model]['model']

    # Define the grid
    params = grid[best_model]['params']

    # 3 Fold Cross Validation
    grid = GridSearchCV(model, cv=3, param_grid=params, scoring=scorer, n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)
    

    return(grid)


# In[ ]:


def model_performance(X_train, X_test, y_train, y_test, grid):
      
    # Select the model with the best paramters
    model = grid.best_estimator_

    # Fit the model on the data
    model.fit(X_train, y_train)
    

    # Get the training predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test) 
    
    # Get the train and test probabilities
    train_probabilities = model.predict_proba(X_train)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]

    # Get the accuracy for train and test
    print('Accuracy score for training is : %s' % accuracy_score(y_train, train_predictions))
    print('Accuracy score for testing is : %s' % accuracy_score(y_test, test_predictions))
    
    # Get the classification report for train and test
    print('\nClassification report for training is : \n%s' % classification_report(y_train, train_predictions))
    print('Classification report for testing is : \n%s' % classification_report(y_test, test_predictions))
    
    # Get the confusion matrix for train and test
    print('\nConfusion matrix for training is : \n%s' % confusion_matrix(y_train, train_predictions))
    print('Confusion matrix for testing is : \n%s' % confusion_matrix(y_test, test_predictions))
    
    # Get the ROC AUC for train and test
    print('\nROC AUC score for training is : %s' % roc_auc_score(y_train, train_probabilities))
    print('ROC AUC score for testing is : %s' % roc_auc_score(y_test, test_probabilities))


# ### 3. Import the required dataset
# 
# Ideally I keep paths at the config section. This is an exception as I am dealing with a single file only here. Also, I always use relative paths as it becomes easier when distributing code. 

# In[ ]:


df = pd.read_csv('../input/heart.csv')


# ### 4. About the data : 
# 
# Depending on the type of data, you may want to see additional things other than shape, describe, dtypes, value_counts and missing values.

# #### 4.1 Shape

# In[ ]:


df.shape


# #### 4.2 Statistics

# In[ ]:


# Summary statistic
df.describe()


# #### 4.3 Types of Columns

# In[ ]:


# Data types of columns
df.dtypes


# #### 4.4 Target Proportion

# In[ ]:


# Distribution of target
df['target'].value_counts([0])


# #### 4.5 Missing Values

# In[ ]:


df.isnull().sum()


# #### 4.6 Display head

# In[ ]:


# first few rows of data
df.head()


# ### 5. Exploratory Data Analysis
# 
# For numerical columns, density/histogram plots are used. For categorical columns, bar plots are used. You may want to see other kind of plots such as corelation plots, boxplots etc too.

# #### 5.1 Age

# In[ ]:


plot_dist(df, 'age', 'target')


# #### 5.2 Sex

# In[ ]:


plot_dist(df, 'sex', 'target', 'cat')


# #### 5.3 CP

# In[ ]:


plot_dist(df, 'cp', 'target', 'cat')


# #### 5.4 Trestbps

# In[ ]:


plot_dist(df, 'trestbps', 'target')


# #### 5.5 Chol

# In[ ]:


plot_dist(df, 'chol', 'target')


# #### 5.6 Fbs

# In[ ]:


plot_dist(df, 'fbs', 'target', 'cat')


# #### 5.7 Restecg

# In[ ]:


plot_dist(df, 'restecg', 'target', 'cat')


# #### 5.8 Thalach

# In[ ]:


plot_dist(df, 'thalach', 'target')


# #### 5.9 Exang

# In[ ]:


plot_dist(df, 'exang', 'target', 'cat')


# #### 5.10 Oldpeak

# In[ ]:


plot_dist(df, 'oldpeak', 'target')


# #### 5.11 Slope

# In[ ]:


plot_dist(df, 'slope', 'target', 'cat')


# #### 5.12 Ca

# In[ ]:


plot_dist(df, 'ca', 'target', 'cat')


# #### 5.13 Thal

# In[ ]:


plot_dist(df, 'thal', 'target', 'cat')


# ### 6. Model Building
# 
# This block specifies the training process which is quite simple. The pipeline is:
# **Split train/test -> Find best baseline model -> Tune parameters of best baseline model -> Evaluate model performance.**
# 
# Almost all data pipelines are of similar structure. I haven't tried other techniques like stacking here. You may also want to look into that!

# #### 6.1 Split into train and test

# In[ ]:


# Get the train and test datasets
X_train, X_test, y_train, y_test = process_data(df, test_size=0.3, random_state=100, scale=True, scaler=MinMaxScaler(), feature_selection=True, k=10)


# In[ ]:


# View the train dataset
X_train.head()


# In[ ]:


# View the test dataset
X_test.head()


# #### 6.2 Best baseline model

# In[ ]:


# Display results of each model
results_pivot, best_model = select_model(X_train, y_train, cv=5, nruns=10, scorer='balanced_accuracy')
results_pivot


# #### 6.3 Hyperparameter Tuning

# In[ ]:


# Print and tune the best model
print('Tuning model for {}'.format(best_model))
grid = tune_model(X_train, X_test, y_train, y_test, best_model, scorer='balanced_accuracy')
grid


# ### 7. Model Performance
# 
# My go-to metrics for evaluating classification models are:
# 
# **Accuracy(unreliable in cases of skewed class proportions of target), classification report showing precision/recall/f1, confusion matrix and roc_auc_score.**
# 
# You may want to look at other metrics such as lift, kappa etc

# In[ ]:


# Display model performance
model_performance(X_train, X_test, y_train, y_test, grid)

