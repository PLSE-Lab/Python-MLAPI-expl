#!/usr/bin/env python
# coding: utf-8

# # TSA Claims Classification (Approve / Settle / Deny)  
# ## (Part 2/2)
# 
# ## Introduction
# In this kernel, we continue to work toward building a model to predict whether a TSA claim is approved, settled, or denied. This is mostly a practice exercise, but it could have some very neat real-world uses!   
#   
# This is a continuation from part one [>here<](https://www.kaggle.com/perrychu/tsa-claims-classification-part-1)
# 
# In part one, we covered data cleaning, resulting in tsa_claims_clean.csv. Now we'll pull in the clean dataset as an input and us it for feature engineering and modeling.

# ## Setup
# Let's start by importing the necessary packages:

# In[1]:


#Data / vizualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Feature engineering
from datetime import datetime
from collections import defaultdict

#Modeling
##Utilities
from sklearn.model_selection import KFold, train_test_split
import sklearn.metrics as skm
import pickle
##Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
import xgboost as xgb


# And load our clean dataset:

# In[2]:


df = pd.read_csv("../input/tsa-claims-classification-part-1/tsa_claims_clean.csv", low_memory=False)

print(len(df))
df.head(3)


# # Part 2: Feature Engineering

# ## Make date variables
# First let's expand our date strings into DateTime objects, and then convert dates to useful numeric features.
# 
# We have two dates - when an incident happened, and when the claim for that incident was received. Using the full date is likely too specific - it doesn't generalize well for future claims. 
# 
# Specifically, we are going to leave out the year field because it is capturing forward progress in time rather than recurring patterns. For example, knowing the weather for each of 12 months in 2008 may not be very helpful for predicting weather in 2020... however knowing weather for 12 Januaries from 2008-2019 is much more likely to be useful for predicting weather in January 2020.
# 
# Excluding year for each date, we can look at the month [0-12], day of the month [0-31], and day of the year [0-366]. These features should capture any seasonal or annually recurring trends in the data.
# 
# Next, we can also look at the difference between the dates - in other words, how long someone waited before they reported their claim.
# 
# There are probably other ways to think about feature engineering, but this feels like a decent set for now.

# In[3]:


df["Date_Received"] = pd.to_datetime(df.Date_Received,format="%Y-%m-%d")
df["Month_Received"] = df.Date_Received.dt.month
df["DayMonth_Received"] = df.Date_Received.dt.day
df["DayYear_Received"] = df.Date_Received.dt.dayofyear

df["Incident_Date"] = pd.to_datetime(df.Incident_Date,format="%Y-%m-%d")
df["Incident_Month"] = df.Incident_Date.dt.month
df["Incident_DayMonth"] = df.Incident_Date.dt.day
df["Incident_DayYear"] = df.Incident_Date.dt.dayofyear

df["Report_Delay"] = (df.Date_Received - df.Incident_Date).dt.days

date_var = ["Report_Delay",
        "Month_Received","DayYear_Received","DayMonth_Received",
        "Incident_Month","Incident_DayYear","Incident_DayMonth"]


# ## Convert text categories to numeric
# Next, we need to modify our categorical text data - Airports, Airlines, Claim Type, and Claim Site. Our models require numbers as input - we would like to represent items in the same category (e.g. United Airlines, or LAX) with a similar number. 
# 
# One way to do this is to create dummy variables (aka one-hot encoding). In this method, we would create one column for each unique label in a category (United Airlines, American Airlines, Delta Airlines, etc.) and put 1 in the column correspoding to the category of that row. For our data, a few columns have a ton of possible values... which creates a ton of columns. We can try this approach anyway, but there is probably a better way.
# 
# Another option is to give each category a numeric rank. For example, if there are 200 possible airlines, we could number them from 1 to 200. However, if our models will learn using this ordering, we want the ordering to make sense! So we can do the ordering in a determined way, like ranking them by how often they appear in the claims data. For example, if United appears in the most claims, we give assign it the number 1, then 2 for second most, 3 for third most, etc.
# 
# Along the same lines, we could simply use the frequency count without flattening into rank. For example if United appears in 10k claims, then we assign it the number 10k, then if the next most is 9k claims we assign it 9k, etc. Under this plan, the number assigned to categories may not be unique. For example, two airlines may both appear in 10 claims. However, this is ok for our application - instead of differentiating by distinct airline this feature is now differentiating by airline popularity. 
# 
# Ok, so lets actually go through appllying each of these methods.
# 

# ### Helper functions
# Lets break out the calculation details into helper functions so our code is more easily understood.
# 
# The first two functions calculate mappings from text category to frequency rank and frequency count.
# 
# The third function applies a calculated mapping to a column.
# 
# The fourth function does a few things:
# * runs the mapping calculation
# * applies the mapping
# * repeats for multiple specified columns
# * inserts the new mapped columns in the data frame with a new name to avoid overwriting the original text data
# 
# It also nicely handles taking in a training set vs. testing set.

# In[4]:


#Frequency rank
#Default is max rank + 1
def get_count_rank(var_column):
    val_count = var_column.value_counts(dropna=False)
    conversion_dict = defaultdict(lambda: len(val_count)+1, zip(val_count.index, range(len(val_count.values))))
        
    return conversion_dict
    
#Frequency
#Default is zero
def get_count(var_column):
    val_count = var_column.value_counts(dropna=False)
    conversion_dict = defaultdict(lambda: 0, zip(val_count.index, val_count.values))
    
    return conversion_dict

#Apply conversion from text to numeric
def apply_conversion(var_column, conversion_dict):

    return var_column.map(lambda x: conversion_dict[x])

def create_numeric(train_df, test_df, conversion_func, columns, postfix="_"):
    '''
    Applies conversion function over the training set to train and test sets.
    Inputs:
        train_df: training data (input to conversion func, then applied)
        test_df: test data (conversion applied)
        conversion_func: returns a map assigning numeric value to each text category
        columns: columns to apply
        postfix: postfix to add to created columns (may overwrite, e.g. on empty string)
        
    '''
    maps = {}
    new_columns = []
    
    for name in columns:
        new_name = name+postfix
        new_columns.append(new_name)
        
        maps[name] = conversion_func(train_df[name])
        
        train_df[new_name] = apply_conversion(train_df[name], maps[name])
        test_df[new_name] = apply_conversion(test_df[name],maps[name])

    text_count_var = [x + "_Count" for x in string_categories]
    
    return train_df, test_df, new_columns


#  ### Conversion 
# Now we  apply each of the conversions!
# 
# First we'll split the data into a training/validation set and a holdout test set. The conversions will be calculated on the training set and then applied to both training and holdout sets. 

# In[34]:


df, df_holdout = train_test_split(df, test_size = .2, shuffle = True, random_state = 1)

string_categories = ["Claim_Type","Claim_Site","Airport_Code_Group","Airline_Name"]

#Category complaint counts
df, df_holdout, text_count_var = create_numeric(df, df_holdout, get_count, string_categories, "_Count")

#Category complaint rank
df, df_holdout, text_rank_var = create_numeric(df, df_holdout, get_count_rank, string_categories, "_Rank")


print("Training / Validation:", len(df), "\nTest:", len(df_holdout))
df_holdout.head(5)


# # Part 3: Modeling
# 
# Now that we've got our features, we can finally do some modeling. There are two things we want to optimize - the set of features to use, and the model to use on those features. Ideally we would do a giant grid search iterating over every model and every set of features... however, that would take too long to run, and might not be very interpretable.
# 
# Instead, we'll first do two distinct steps:
# * Feature selection - look at how a baseline model performs with different feature sets 
# * Model selection - look at how different models perform with our best feature set 
# 
# Note: After feature selection, we may consider going back to feature engineering (depending on our results and how much time we have!). This is an interative process, not a one-way street!
# 
# For each run, we'll track multiple performance metrics: Accuracy (global), Precision / Recall / F1 (for each class). I'm going to assume we're familiar with these classification metrics - but its worth reading about if you're not! 

# ## Helper fuctions
# Again, we'll use some helper functions to clean up our code.
# 
# The first function neatly prints out the scoring metrics.
# 
# The second function executes a cross-validation loop for a given model and dataset, then calls the first function to print the resulting average scores.

# In[25]:


def print_scores(scores_array, ylabel_strings):
    ''' Prints a table with headings for output of sklearn.metrics.precision_recall_fscore_support
        Inputs: scores_array -> np.array of scores from skm.
                ylabel_strings -> the target labels
    '''
    #Each row is a score in the output, transpose to get features across rows
    array = np.transpose(scores_array) 
    macro_avg = np.average(array,axis=0)
    labels = sorted(ylabel_strings)
    
    max_len = str(np.max([len(s) for s in ylabel_strings]))
        
    print(("\n{:>"+max_len+"} {:>10s} {:>10s} {:>10s} {:>10s}").format("","Precision","Recall","F1","Support"))
    
    for i in range(len(labels)):
        print(("{:>"+max_len+"} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.0f}")
              .format(labels[i],array[i][0],array[i][1],array[i][2],array[i][3]))
    
    print(("{:>"+max_len+"} {:>10.5f} {:>10.5f} {:>10.5f} {:>10.0f}")
          .format("Avg/Tot",macro_avg[0],macro_avg[1],macro_avg[2],macro_avg[3]))
    
def validation_loop(model,X,Y,k=5,rand_state=1):
    ''' Runs k-fold validation loop for input model, X, Y. Prints classification accuracy 
             and the following per-label metrics: precision, recall, f1, support.
        Inputs: 
                ylabel_strings -> the target labels
    '''
    test_accs, test_scores = [], []
    train_accs, train_scores = [], []
    
    i=1

    for train_ind, test_ind in KFold(k,shuffle=True,random_state=rand_state).split(X,Y):
        #print("Starting {} of {} folds".format(i,k))

        model.fit(X[train_ind],Y[train_ind])
        
        #Test metrics
        pred = model.predict(X[test_ind])
        acc = skm.accuracy_score(Y[test_ind],pred)
        test_accs.append(acc)
        score = skm.precision_recall_fscore_support(Y[test_ind],pred)
        test_scores.append(score)
        
        #Train metrics
        pred = model.predict(X[train_ind])
        acc = skm.accuracy_score(Y[train_ind],pred)
        train_accs.append(acc)
        score = skm.precision_recall_fscore_support(Y[train_ind],pred)
        train_scores.append(score)
        
        i+=1
    
    print("\nAvg. Train Metrics")
    print ("Accuracy: {:.5f}".format(np.average(train_accs)))
    print_scores(np.average(train_scores,axis=0),np.unique(Y))
    
    print("\nAvg. Validation Metrics")
    print ("Accuracy: {:.5f}".format(np.average(test_accs)))
    print_scores(np.average(test_scores,axis=0),np.unique(Y))
    
    


# ## Feature Selection
# 
# We'll use random forest (with 250 trees) as our baseline model. I picked random forest because it tends decent results with default parameters on most datasets, and I suspect our data set has a lot of non-linear relationships which are more difficult for logistic regression to handle. The downside is RF takes longer to run.
# 
# ### Baseline
# We'll baseline with only one feature (claim value).
# 
# Random Forest ends up performing slightly better (~51.7% accuracy) than naively guessing the most common class "Denied" (46.7%). It manages to pick up signal on the minority classes (Approved / Settled) so this is much more useful than the naive guess model.
# 
# Note: The cross-validation function prints both training and validation set metrics. The validation set metrics are what matters for picking between models. Training set metrics are used as a rough reference for where we are on the bias-variance curve (and should always be higher). Bias-variance is worth learning about if you're not familiar!

# In[7]:


features = ["Claim_Value"]
target = "Status"

model_df=df[[target]+features].dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = RandomForestClassifier(n_estimators=250, min_samples_split=10, n_jobs=-1)

validation_loop(model,X,Y,rand_state=1)


# ### Base + Date Features
# Let's try adding Date features. Our results improve a little, to ~52.7% accuracy.

# In[8]:


features = ["Claim_Value"]+date_var
target = "Status"

model_df=df[[target]+features].dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = RandomForestClassifier(n_estimators = 250, min_samples_split=10, n_jobs=-1)

validation_loop(model,X,Y,rand_state=1)


# ### Base + Categorical Features
# Let's try adding our Categorical features. Here, we had a few different representations. Let's try each style separately (and a combination), then use the best one.
# 
# Results (accuracy):
# * Dummy Vars: ~53.69%
# * Frequency Count: ~54.22%
# * Frequency Rank: ~54.19%
# * Count & Rank: ~54.16%
# 
# Frequency Count is best, so we'll use that going forward.

# #### Dummy Vars (aka One-hot encoding)

# In[19]:


dummies_df = pd.get_dummies(df[["Claim_Type","Claim_Site","Airport_Code_Group","Airline_Name"]],prefix=["Type","Site","Airport","Airline"])

features = ["Claim_Value"] + list(dummies_df.columns)
target = "Status"

model_df=df[["Status","Claim_Value"]].join(dummies_df).dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = RandomForestClassifier(n_estimators = 250, min_samples_split=10, n_jobs=-1)

validation_loop(model,X,Y,rand_state=1)


# #### Frequency Count

# In[20]:


features = ["Claim_Value"] + text_count_var 
target = "Status"

model_df=df[[target]+features].dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = RandomForestClassifier(n_estimators = 250, min_samples_split=10, n_jobs=-1)

validation_loop(model,X,Y,rand_state=1)


# #### Frequency Rank

# In[21]:


features = ["Claim_Value"] + text_rank_var
target = "Status"

model_df=df[[target]+features].dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = RandomForestClassifier(n_estimators = 250, min_samples_split=10, n_jobs=-1)

validation_loop(model,X,Y,rand_state=1)


# #### Frequency Count + Rank

# In[22]:


features = ["Claim_Value"] + text_count_var + text_rank_var
target = "Status"

model_df=df[[target]+features].dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = RandomForestClassifier(n_estimators = 250, min_samples_split=10, n_jobs=-1)

validation_loop(model,X,Y,rand_state=1)


# ### Base + Date + Categorical Features
# Let's try all of the features (using our best categorical representation).
# 
# Accuracy is 56.5% - we'll use these features for model selection.

# In[24]:


features = ["Claim_Value"]+date_var+text_count_var
target = "Status"

model_df=df[[target]+features].dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = RandomForestClassifier(n_estimators = 250, min_samples_split=10, n_jobs=-1)

validation_loop(model,X,Y,rand_state=1)


# ### Feature Selection Results
# * Baseline: 51.4%
# * Base + Date: 52.7%
# * Base + Category: 54.2%
# * Base + Category + Date: 56.5%
# 
# The best result was using all of the features. If desired, we could go back and create more features, then do more validation to figure out whether those features improve the model.
# 
# For now, lets stick with what we have. We can record this data set as a .csv for future use.

# In[14]:


features = ["Claim_Value"]+date_var+text_count_var
target = "Status"
csv_df=df[[target]+features].dropna()

csv_df.to_csv("tsa_model_features.csv",index=False)


# ## Model Selection
# Now that we have a solid feature set, lets test some other models.
# 
# Our dataset seems non-linear, which tree-based models should handle best. Aside from Random Forest, we can try gradient boosted trees. We'll use the XGBoost implementation.
# 
# We can also try two linear models - Logistic Regression and Naive Bayes - just to see how they do.

# ### Random Forest

# In[30]:


features = ["Claim_Value"]+date_var+text_count_var
target = "Status"

model_df=df[[target]+features].dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = RandomForestClassifier(n_estimators = 250, min_samples_split=10, n_jobs=-1)

validation_loop(model,X,Y,rand_state=1)


# ### XGBoost

# In[31]:


features = ["Claim_Value"]+date_var+text_count_var
target = "Status"

model_df=df[[target]+features].dropna()

X = model_df[features].reset_index(drop=True)
Y = model_df[target].reset_index(drop=True)

model = xgb.XGBClassifier(n_estimators = 30000,
                          learning_rate = .2,
                          max_depth = 4,
                          objective = "multi:softmax",
                          subsample=1,
                          min_child_weight=1,
                          colsample_bytree=.8,
                          random_state = 1,
                          n_jobs = -1
                         )

test_accs, test_scores = [], []
train_accs, train_scores = [], []

logloss = []
ntrees = []

i=1
k=5
rand_state = 1

for train_ind, test_ind in KFold(k,shuffle=True,random_state=rand_state).split(X,Y):
    #print("Starting {} of {} folds".format(i,k))

    eval_set=[(X.iloc[train_ind],Y.iloc[train_ind]),(X.iloc[test_ind],Y.iloc[test_ind])] 
    fit_model = model.fit( 
                    X.iloc[train_ind], Y.iloc[train_ind], 
                    eval_set=eval_set,
                    eval_metric='mlogloss',
                    early_stopping_rounds=50,
                    verbose=False
                   )
    
    logloss.append(model.best_score)
    ntrees.append(model.best_ntree_limit)
    
    #Test metrics
    pred = model.predict(X.iloc[test_ind],ntree_limit=model.best_ntree_limit)
    acc = skm.accuracy_score(Y.iloc[test_ind],pred)
    test_accs.append(acc)
    score = skm.precision_recall_fscore_support(Y.iloc[test_ind],pred)
    test_scores.append(score)
    #print(acc)
    #print(skm.classification_report(Y[test_ind],pred))

    #Train metrics
    pred = model.predict(X.iloc[train_ind],ntree_limit=model.best_ntree_limit)
    acc = skm.accuracy_score(Y.iloc[train_ind],pred)
    train_accs.append(acc)
    score = skm.precision_recall_fscore_support(Y.iloc[train_ind],pred)
    train_scores.append(score)

    i+=1

print("\nAvg. Train Metrics")
print ("Accuracy: {:.5f}".format(np.average(train_accs)))
print_scores(np.average(train_scores,axis=0),np.unique(Y))

print("\nAvg. Validation Metrics")
print ("Accuracy: {:.5f}".format(np.average(test_accs)))
print_scores(np.average(test_scores,axis=0),np.unique(Y))

print("\nLogloss:", np.average(logloss), "Std Dev:", np.std(logloss))
print("Best number of trees", ntrees)


# ### Logistic Regression

# In[32]:


features = ["Claim_Value"]+date_var+text_count_var 
target = "Status"

model_df=df[[target]+features].dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = LogisticRegression(C=100)

validation_loop(model,X,Y,rand_state=1)


# ### Naive Bayes

# In[33]:


features = ["Claim_Value"]+date_var+text_count_var 
target = "Status"

model_df=df[[target]+features].dropna()

X = np.array(model_df[features])
Y = np.array(model_df[target])

model = GaussianNB()

validation_loop(model,X,Y,rand_state=1)


# ### Model Selection Results
# 
# Results (accuracy):
# * Random Forest: 56.4%
# * XG Boost: 57.5%
# * Logistic Regression: 47.5%
# * Gaussian NB: 31.3%
# 
# From our testing, we see the linear models did significantly worse than the trees. Gradient boosted trees outperformed random forest, so we'll choose this model for our final test.

# ## Final Model Test
# Finally, we'll test the XGBoost model on the holdout test to estimate how it does on new data. 
# 
# The holdout set accuracy is 57.6% - pretty much in line with our validation accuracy.

# In[35]:


features = ["Claim_Value"]+date_var+text_count_var
target = "Status"

model_df=df[[target]+features].dropna()

X = model_df[features].reset_index(drop=True)
Y = model_df[target].reset_index(drop=True)

model_df_holdout = df_holdout[[target]+features].dropna()
X_holdout = model_df_holdout[features]
Y_holdout = model_df_holdout[target]

model =  xgb.XGBClassifier(max_depth = 6,
                           subsamples = 1,
                           min_child_weight=6,
                           colsample_bytree=0.6,
                           n_estimators=107,
                           learning_rate=.1,
                           objective = "multi:softmax",
                           random_state = 1,
                           n_jobs = -1)

fit_model = model.fit(X, Y)

print("Training")
pred = model.predict(X)
print(skm.accuracy_score(Y,pred))
print(skm.classification_report(Y, pred))

print("\nHoldout")
pred = model.predict(X_holdout)
print(skm.accuracy_score(Y_holdout,pred))
print(skm.classification_report(Y_holdout, pred))


# We can also try to make some inferences from the model. Due to being a tree-based model, we don't have directional relationships between variables and our prediction. However, we can look at the features that the tree found most important.
# 
# Claim Value, Claim Site, and Claim Type give the most information gain. We can dig into these variables a bit more. 

# In[38]:


xgb.plot_importance(fit_model, importance_type="gain");


# In[ ]:




