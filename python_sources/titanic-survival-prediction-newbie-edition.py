#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Never was a fan of Titanic movie so I don't have too much pre-knowledge of it but familiar with the event of course. Dived a little deeper for more context and understanding here [Titanic Encyclopedia Entry](https://www.encyclopedia-titanica.org/titanic/). The goal of this notebook is to build a machine learning model that accurately predicts who survived and who did not survive based on statistical data provided on the passengers who were aboard the Titanic. 

# In[ ]:


import numpy as np 
import pandas as pd
import pandas_profiling

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from statistics import mean 

# Data Viz
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')


# # Data Exploration
# Let's explore the data at a high-level to understand what we're working with, anticipate any cleaning we would have to do, and build a general instinct for what features we should leverage to create a prediction. I also used the following resource to understand what each of the columns [Titanic Meta](http://campus.lakeforest.edu/frank/FILES/MLFfiles/Bio150/Titanic/TitanicMETA.pdf).
# 
# The profile below gives us plenty of information about the data we're looking at. We can see that we're missing data for the **Age**, **Cabin** and the **Embarked** columns so we may have to do some cleaning there. There are 5 columns that are categorical which potentially means we will need to do some encoding as well. The **Cabin** and the **Ticket** columns have a high cardinality so we know type of encoding we may need to use since they're both categorical. The correlation graphs give us a high-level of where we should look for features. 
# 
# Taking a closer look we can also see off rip that the **Ticket** and **PassengerId** columns wouldn't prove too useful for our predictions because there's no pattern there it causes too much noise. I would also say the same for the **Name** column but we can see an underlying pattern in the titles: *Mr.*, *Mrs.* *Master*, etc. The **Cabin**  column is interesting because we can extrapolate a pattern if we only looked at the *letter* in the cabin number unfortunately though **Cabin** is missing over 3/4s of its values. My decision here would be too also disregard this column as well. 

# In[ ]:


df_train.profile_report(style={'full_width': True})


# ## Data Visualization
# Let's dive a little deeper to see if we can identify stronger patterns and relationships by using graphs.

# In[ ]:


# Embarked is missing 2 data entries so let's fill those with the Mode value 
df_embarked = df_train['Embarked'].copy()
df_embarked.fillna(df_embarked.mode(), inplace=True)
df_train['EmbarkedFilled'] = df_embarked
sns.barplot(x="EmbarkedFilled", y="Survived", hue="Sex", data=df_train)


# In[ ]:


sns.barplot(x='Pclass', y='Survived', hue="Sex", data=df_train)


# In[ ]:


# Age is interesting so let's do some light transformation and look at the Age column via age groups
df_age = df_train['Age'].copy()

# Age column is missing some data approximately 20% so let's do some pre-processing and fill in the missing data with the median age.
median_age = df_age[pd.notna(df_age)].median()
df_age.fillna(median_age, inplace=True)

# Next let's decide a real life grouping for the ages:
# baby- 0-1; toddler- 1-3; preschool- 3-5; gradeschooler- 5-12; teen- 12-18; young adult- 18-21; adult- 21-54; senior- 55+ 
bins = (-1, 1, 3, 5, 12, 18, 21, 54, 100)
age_groups = pd.cut(df_age, bins=bins, labels=['baby', 'todd', 'pre', 'grade', 'teen', 'ya', 'adult', 'senior'])
# Add it to the original df so we can graph
df_train['AgeGroup'] = age_groups

# Graph the age grouping
sns.barplot(x='AgeGroup', y='Survived', hue='Sex', data=df_train)


# In[ ]:


# Given there are 248 unique fare amounts paid that may be too disperse and noisy let's group them into bins based on the quantiles.
# I was trying to think of ways to naturally group them similar to how we did with ages but there's no obvious way. For example if there
# were price tiers that we can derive from the fare amounts that would've helped but we don't have any indication of a tier. I mean who
# was setting the pricing on this trip sheeesh. 

df_fare = df_train['Fare'].copy()
bins = (-1, 8, 14, 31, 600)
df_train['FareGroup'] = pd.cut(df_fare, bins=bins)
sns.barplot(x='FareGroup', y='Survived', hue='Sex', data=df_train)


# In[ ]:


sns.pointplot(x='Parch', y='Survived', hue='Sex', data=df_train)


# In[ ]:


sns.pointplot(x='SibSp', y='Survived', hue='Sex', data=df_train)


# # Feature Engineering

# ## Cleaning & Pre-Processing
# 
# Before we start feature engineering we must first decide on each of the columns we want to use and if we want to generate new features based on the given features available, such as the name title from the name column. In addition we must fill in any missing data for any of the features we will use. Lastly we must ensure that we perform the same cleaning on the test data as well. 
# 
# The features we will be going with are: **AgeGroup**, **Title**, **FareGroup**, **Embarked**, **Pclass** and **Sex**. The columns that have missing data are **AgeGroup** and **Embarked** while the **Title**, **AgeGroup** and **FareGroup** need to be transformed. In addition let's transform the **Pclass** column by converting the string value into it's numerical value. The remaining columns in the data will be dropped. 

# In[ ]:


def fillAges(df):
    '''Fill in missing data for Age column by using the median'''
    df_age = pd.DataFrame(df['Age'])
    imputer = SimpleImputer(strategy='median')
    df['Age'] = imputer.fit_transform(df_age)
    return df
    
def fillEmbarked(df):
    '''Fill in missing data for the Embarked column by using the mode'''
    df_embarked = pd.DataFrame(df['Embarked'])
    imputer = SimpleImputer(strategy='most_frequent')
    df['Embarked'] = imputer.fit_transform(df_embarked)
    return df
    
def addAgeGroup(df):
    '''Group ages in bins and create a new series for it in the dataframe'''
    # baby- 0-1; toddler- 1-3; preschool- 3-5; gradeschooler- 5-12; teen- 12-18; young adult- 18-21; adult- 21-54; senior- 55+ 
    bins = (-1, 1, 3, 5, 12, 18, 21, 54, 100)
    age_group = pd.cut(df['Age'], bins=bins, labels=['baby', 'todd', 'pre', 'grade', 'teen', 'ya', 'adult', 'senior'])
    df['AgeGroup'] = age_group
    return df

def addFareGroup(df):
    '''Group fare in bins and create a new series it in the dataframe'''
    bins = (-1, 8, 14, 31, 600)
    df['FareGroup'] = pd.cut(df_fare, bins=bins, labels=['1','2','3','4'])
    return df

def addTitle(df):
    '''Add title column based on the Name column'''
    name = df['Name'].copy()
    df['Title'] = name.apply(lambda x: x.split(', ')[1].split(' ')[0])
    return df

def dropUnused(df, cols_to_keep):
    '''Drop the unused columns'''
    return df[cols_to_keep]

def clean(df, cols_to_keep):
    '''Clean the data and include the columns we dont want to drop in the end'''
    df = fillAges(df)
    df = fillEmbarked(df)
    df = addAgeGroup(df)
    df = addFareGroup(df)
    df = addTitle(df)
    df = dropUnused(df, cols_to_keep)
    return df

df_train = clean(df_train, ['AgeGroup', 'Title', 'FareGroup', 'Embarked', 'Parch', 'SibSp', 'Sex', 'Pclass', 'Survived'])
df_test = clean(df_test, ['AgeGroup', 'Title', 'FareGroup', 'Embarked', 'Parch', 'SibSp', 'Sex', 'Pclass', 'PassengerId']) # We need to keep the PassengerId for submission


# ## Encoding
# Now that we cleaned the data let's encode the categorical features so that we can move on to modeling. The categorical features left are **Embarked**, **Title**, **FareGroup**, **AgeGroup**, and **Sex**. 
# 
# To decide on the best encoding method for each feature let's think about each at a high-level. **Embarked** has a cardinality of 3, **Title** has a cardinality of 17, **Sex** has a cardinality of 2, **AgeGroup** with a cardinality of 8 and **FareGroup** with a cardinality of 6. The **Embarked** and **Sex** feature has no natural ordering usually **Title** does but under the circumstances of the Titanic it may have less importance. **FareGroup** and **AgeGroup** have been transformed into bins with labels so we can leverage the labels directly in order to encode. 

# In[ ]:


def encode(df_train, df_test, features):
    '''Encode the categorical columns'''
    
    # combine so that we can fit over the training and test data to ensure best results for transform
    df_combined = pd.concat([df_train[features], df_test[features]])
    for feature in features:
        lbl_encoder = LabelEncoder()
        lbl_encoder = lbl_encoder.fit(df_combined[feature])
        df_train[feature] = lbl_encoder.transform(df_train[feature])
        df_test[feature] = lbl_encoder.transform(df_test[feature])
        
    return (df_train, df_test)


categorical_features = ['Embarked', 'Sex', 'Title', 'AgeGroup', 'FareGroup']
df_train, df_test = encode(df_train, df_test, categorical_features)


# ## Sanity Check 
# As a sanity check let's make sure all the values are cleaned, pre-processed and encoded. If not then we missed a step. 

# In[ ]:


df_train.profile_report(style={'full_width': True})


# # Modeling
# 

# ## Test Data
# Now that we have cleaned the data, perform pre-processing and some minor feature engineering strategies let move on to modeling. But first we must prepare the test data we will be using for hyperparamter tuning and model prediction validation.

# In[ ]:


X_train_all = df_train.drop(['Survived'], axis=1)
y_train_all = df_train['Survived']

test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X_train_all, y_train_all, test_size=test_size, random_state=1)


# ## Model Selection
# For the selection strategy we're going to keep it light and use the default paramters for the models. We're going to first do a plain comparison between each model's predict on the test data. Then we're going to run cross-validation on each of the models for a higher rate of confidence on the winning model. 

# In[ ]:


def simple_predict(X_train, y_train, X_test, y_test, pred_models):
    models = []
    score_results = []

    for m in prediction_models:
        model = eval(m)()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        models.append(m)
        score_results.append(accuracy_score(y_test, y_pred))
        
    return pd.DataFrame({'model': models, 'result': score_results})

predict(X_train, y_train, X_test, y_test, ['RandomForestClassifier', 'SGDClassifier', 'GradientBoostingClassifier'])


# In[ ]:


def cross_validation(X_all, y_all, pred_models):
    '''Cross validate across all the different classifiers. By default use 4 folds'''
    models = []
    result_1 = []
    result_2 = []
    result_3 = []
    result_4 = []
    means = []
    
    for m in pred_models:
        model = eval(m)()
        results = cross_val_score(model, X_all, y_all, cv=4, scoring='accuracy')
        models.append(m)
        result_1.append(results[0])
        result_2.append(results[1])
        result_3.append(results[2])
        result_4.append(results[3])
        means.append(mean(results))
        
    return pd.DataFrame({'model': models, 
                         'result1': result_1,
                         'result2': result_2,
                         'result3': result_3,
                         'result4': result_4, 
                         'mean': means})

cross_validation(X_train_all, y_train_all, ['RandomForestClassifier', 'SGDClassifier', 'GradientBoostingClassifier'])
        


# ## Predict Test Data
# Now that we have our results from the classifiers above. Let's now predict the test data using the winning model. 

# In[ ]:


# Separate the passenger Ids for submission
pass_id_series = df_test['PassengerId']
df_test = df_test.drop(['PassengerId'], axis=1)

# Predict
rfc = RandomForestClassifier()
rfc.fit(X_train_all, y_train_all)
predictions = rfc.predict(df_test)

# Setup final data frame for csv output
df_output = pd.DataFrame({'PassengerId': pass_id_series, 'Survived': predictions})
print(df_output.head())


# # Conclusion
# While this was a good first attempt at creating a prediction model for the Titanic competition I feel like there is still more I could've done to improve the accuracy of the prediction. There are still some strategies left on the table in terms of *parameter tuning*, comparing more models  and creating custom ensemble models. I also could've invested in more graphs to reveal more depth on feature correlation to our prediction target as well. Instinctively I also feel like there is a win somewhere with a feature *interaction*, combining 2 or more features. One day I'll come back and see if I can improve it more. 
# 
# If you made it here thanks for following along I hope this proved helpful to you.

# In[ ]:


# output to csv
df_output.to_csv('submission.csv', index=False)

