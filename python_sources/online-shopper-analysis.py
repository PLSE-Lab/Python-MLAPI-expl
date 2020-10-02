#!/usr/bin/env python
# coding: utf-8

# # Online Shopper's Intention
# 
# <br>
# 
# Goal: build a classification model that will predict whether a shopper is a paying customer based on the available features (10 numerical and 7 categorical features, not including the "revenue" category). 
# 
# <br>
# 
# Basic steps will involve: data cleaning -> data exploration -> model pipeline -> model examination
# 
# <br>
# 
# The data were downloaded from Kaggle [here](https://www.kaggle.com/roshansharma/online-shoppers-intention) and the explanation of the features can be found on there.

# ## Data Cleaning
# 
# 
# .head(), .shape, .describe(), .info(), .isnull().sum(), .columns, .dtypes(), .fillna(value = ), .drop(..., axis = 1) - to drop columns, pd.read_csv()
# 
# check out numerical and categorical values - does everything look like it makes sense?
# 
# ### Load the libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import random
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load the dataset

# In[ ]:


shoppers = pd.read_csv("../input/online-shoppers-intention/online_shoppers_intention.csv")


# ### Examine the data

# In[ ]:


type(shoppers)


# In[ ]:


shoppers.head()


# Look at the shape of the dataset to understand how much data you're working with

# In[ ]:


shoppers.shape ## 12330 rows, 18 columns


# 12330 rows = 12330 samples

# Describe the numerical features

# In[ ]:


shoppers.describe()


# None of the numerical features appear as though they have "incorrect" values (e.g. negative values in a feature that could only have positive values).
# 
# One thing that's noticeable is that the mean value in most categories is a lot smaller than the max value, suggesting strongly right skewed distributions of values in each category. 
# 
# Notice that "Region", "Browser", "OperatingSystems", and "TrafficType" are being categorized as a numeric features, but they're actually numbers representing categories.

# In[ ]:


shoppers.columns ## just the column names


# Looking for null values in the columns

# In[ ]:


shoppers.isnull().sum()


# Not many null values, but suspicially 14 samples have null values in the first 8 features, and these might be important features for our model. But 14 samples represents a small portion of the total dataset, to the easiest way to fix those missing values is to drop those samples from the dataset rather than finding a way to impute them. 

# In[ ]:


shoppers = shoppers.dropna(axis = 0)


# In[ ]:


shoppers.isnull().sum() ## no remaining null values


# In[ ]:


shoppers.shape ## should be 14 rows less


# In[ ]:


shoppers.info()


# Define the types of features in the data - categorical or numeric

# In[ ]:


cat_cols = ['Month', 'OperatingSystems', 'Browser', 'Region',
           'TrafficType', 'VisitorType', 'Weekend', 'Revenue']
for col in cat_cols:
    shoppers[col] = shoppers[col].astype('category')


# In[ ]:


shoppers.dtypes 


# Look at the categorical values of the category features and see if there are any weird values that might be data errors

# In[ ]:


for col in cat_cols:
    print(shoppers[col].unique())


# Things to notice:
# 
# 1) the month column only has 10 months represented, January and April are missing.
# 
# 2) there is an "other" value in the returning visitor category, let's see how many samples have "other" as the return visitor value

# In[ ]:


sum(shoppers['VisitorType'] == 'Other') ## 85 have this other type


# 85 samples have the visitor type listed as "Other" - that's not insignificant I suppose and "Other" might have some actually important value, so let's leave it. It shouldn't interfere too much. 
# 
# There are also duration values that are negative - what would that even represent in the real world? Not sure, could be an error. There are only 33 of those entries, so we can remove. 

# In[ ]:


len(shoppers.loc[shoppers['ProductRelated_Duration'] == -1, 'ProductRelated_Duration'])


# In[ ]:


len(shoppers.loc[shoppers['Informational_Duration'] == -1, 'Informational_Duration'])


# In[ ]:


len(shoppers.loc[shoppers['Administrative_Duration'] == -1, 'Administrative_Duration'])


# In[ ]:


## okay so looks like the -1 are all in the same row so let's drop those rows
shoppers = shoppers.loc[shoppers['Administrative_Duration'] != -1, ]


# In[ ]:


## how many shoppers spent zero amount of time on the website - these ones maybe are
## rounded down because they spent so little time on the site (<0)
sum(shoppers[['ProductRelated_Duration',
              'Informational_Duration',
              'Administrative_Duration']].sum(axis = 1) == 0)


# ## Data Exploration
# 
# Now that data are clean and there are no missing values we can begin the exploratory data analysis
# 
# ### EDA Questions
# 
# * Where do shoppers come from (region)?
# * Do new or returning customers spend longer on the website?
# * Which part of the website (admin/product/information) are new vs returning customers more likely to spend their time? 
# * Does site visitation increase around special days? Which special days? Does this change based on the type of customer (new vs. returning)? 
# * What special day attracts more new users? 
# * What month of the year are there more users? Is this related to special days? 
# 

# ### What region do shoppers come from?
# 
# To answer this we have to count the number of values of each region and then plot the values in an ordered histogram -- can do this all at once using `sns.countplot`

# In[ ]:


plt.figure(figsize = (16,8))
sns.countplot(x = "Region", data = shoppers, 
              order = shoppers['Region'].value_counts().index)
plt.title("Site Visitors by Region", fontsize = 16)
plt.xlabel("Region", fontsize = 16)
plt.ylabel("Number of Visitors", fontsize = 16)


# The region with the most visitors is region 1 - but we don't know where exactly that is. But maybe if you were in charge of these data you would. 

# ### Do new or returning customers spend longer on the website?
# 
# Group the data by `visitortype` and then add up their total time spent on the site (across informational, product, administrative types) and then plot a boxplot comparing the returning versus new customer. Then perform a permutation test to see if there is a statistical difference between new and returning customers time on the site (if it looks like there might be one).

# In[ ]:


shoppers.columns


# Columns you want are "Administrative_Duration", "Informational_Duration", 'ProductRelated_Duration"

# In[ ]:


dur = shoppers[['Administrative_Duration', 
          'Informational_Duration', 
          'ProductRelated_Duration']].sum(axis = 1)


# In[ ]:


site_duration = pd.DataFrame({"VisitorType": shoppers['VisitorType'],
                           "TotalDuration": dur
                           })


# In[ ]:


plt.figure(figsize = (16,8))
sns.boxplot('VisitorType', 'TotalDuration', data = site_duration)
plt.title("Time on Site by Visitor Type", fontsize = 16)
plt.xlabel("Visitor Type", fontsize = 16)
plt.ylabel("Total Time on Site (seconds)", fontsize = 16)
plt.ylim(10, 10000)


# Appears as though returning visitors spend longer on the site. I would say we could take the log to make the spread of the durations easier to see but the values include zeros and negatives, so it wouldn't work. 

# To check whether returning visitors spend a significantly longer time on the site than new visitors we can do a permutation test

# In[ ]:


ret_duration = list(site_duration.loc[
    site_duration['VisitorType'] == 'Returning_Visitor', 'TotalDuration'])


# In[ ]:


new_duration = list(site_duration.loc[
    site_duration['VisitorType'] == 'New_Visitor', 'TotalDuration'])


# In[ ]:


import copy
import random


# In[ ]:


## write a permutation function for mean

def perm_mean(group_1, group_2, p): ## two lists and a numeric value for the number of permutations
    """Returns the p-value for a permutation test of difference in means between
    two groups"""
    
    ## observed difference in means
    obs_mean = np.abs(np.average(group_1) - np.average(group_2))
    
    ## pool the observations into a single list
    pooled_groups = list(group_1 + group_2)
    
    ## make a copy that can be randomly shuffled for the permutations
    pooled_copy = copy.copy(pooled_groups)
    
    ## a space to save permutation output
    perm_means = []
    
    ## permutations
    for i in range(0, p):
        ## randomly shuffle the pooled observations
        random.shuffle(pooled_copy)
        
        ## calculate differences in mean for each permutation
        perm_means.append(
            np.abs(np.average(
                pooled_copy[0:len(group_1)]) - np.average(pooled_copy[len(group_1):])))

    ## calculate the p-value as proportion of the permuted means that had a larger
    ## difference in means than the observed difference in means
    p_value = sum(perm_means >= obs_mean)/p
    
    return p_value


# In[ ]:


perm_mean(ret_duration, new_duration, 1000)


# A value of 0 means that none of the permutations had a difference in means greater than the observed difference in means. In addition, the mean duration on the site of the returning customers is larger than the mean duration on the site of new customers. This leads us to determine that returning customers spend significantly more time on the site than new customers, which makes sense!

# ### Which part of the website (admin/product/information) are new vs returning customers more likely to spend their time? 
# 
# Will want to do as similar thing as above except we break up the box plots into the page type 

# In[ ]:


shoppers.columns


# In[ ]:


duration = list(shoppers['Administrative_Duration']) + list(
    shoppers['ProductRelated_Duration']) + list(
    shoppers['Informational_Duration'])


# In[ ]:


import itertools


# In[ ]:


duration_type = list(
    itertools.repeat('Administrative', len(shoppers['Administrative_Duration']))) + list(
    itertools.repeat('ProductRelated', len(shoppers['ProductRelated_Duration']))) + list(
    itertools.repeat('Informational', len(shoppers['Informational_Duration'])))


# In[ ]:


visitor_type = list(shoppers['VisitorType'])*3


# In[ ]:


duration_info = pd.DataFrame({"Visitor Type": visitor_type,
                           "Duration Type": duration_type,
                           "Duration": duration})


# In[ ]:


duration_info


# In[ ]:


plt.figure(figsize = (16,8))
sns.boxplot('Visitor Type', 'Duration', data = duration_info, hue = 'Duration Type')
plt.title("Duration on Each Page Type by Visitor Type", fontsize = 16)
plt.xlabel("Visitor Type", fontsize = 16)
plt.ylabel("Duration on Page (seconds)", fontsize = 16)
plt.show()


# In[ ]:


plt.figure(figsize = (16,8))
sns.boxplot('Visitor Type', 'Duration', data = duration_info, hue = 'Duration Type')
plt.title("Duration on Each Page Type by Visitor Type", fontsize = 16)
plt.xlabel("Visitor Type", fontsize = 16)
plt.ylabel("Duration on Page (seconds)", fontsize = 16)
plt.ylim((-100, 3000))
plt.show()


# Again, the distributions are so right skewed but you can see from the graph when it's zoomed in that returning customers spend longer on the product related portion of the website, and it appears as though new visitors spend a little longer than returning visitors on the administrative parts of the website. 

# What percent of their visit time does each customer spends in each section

# In[ ]:


duration_info ## ok something got fucked up


# In[ ]:


shoppers


# In[ ]:


duration_info['Total Duration'] = list(site_duration['TotalDuration'])*3


# In[ ]:


duration_info


# In[ ]:


## percentages
duration_info['Duration Percent'] = duration_info['Duration']/duration_info['Total Duration']


# In[ ]:


duration_info


# In[ ]:


min(duration_info['Duration Percent'])


# Because some people spent "no time" on the website (their duration was zero - is this truly zero or is it just a very very small amount of time that got rounded to zero? Not sure). We can remove the rows that have zero because we can't get a percentage out of them

# In[ ]:


duration_info = duration_info.dropna()


# In[ ]:


plt.figure(figsize = (16,8))
sns.boxplot('Visitor Type', 'Duration Percent', data = duration_info, hue = 'Duration Type',
              palette = "colorblind")
plt.title("Duration on Each Page Type by Visitor Type", fontsize = 16)
plt.xlabel("Visitor Type", fontsize = 16)
plt.ylabel("Duration on Page (seconds)", fontsize = 16)
plt.show()


# The vast majority of users time is spent on product related pages, with more time spent on administrative pages by new visitors than returning visitors. 

# ### What percent of visitors to the site were new vs. returning?

# In[ ]:


percent_ret = round(shoppers['VisitorType'].value_counts()['Returning_Visitor']/len(shoppers['VisitorType'])*100, 2)
percent_new = round(shoppers['VisitorType'].value_counts()['New_Visitor']/len(shoppers['VisitorType'])*100, 2)

print(f"{percent_ret} % visitors were returning")
print(f"{percent_new} % visitors were new")


# ### Did visitation change across the year? 
# 
# Need to count the number of visitors per month
# <br>
# 
# But first need to fix the probelem of the missing two months and make sure that the months will plot in chronological order (by ordering them)

# In[ ]:


ordered_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 
                  'Nov', 'Dec']


# In[ ]:


## this bit might not be the cleanest and I wish I could figure out a better way
## dataframe that counts the entries in each group (visitor type & month)
month_info = shoppers.groupby(['VisitorType', 'Month']).count()


# In[ ]:


## only need one of those columns
month_info = pd.DataFrame(month_info.iloc[:, 1])


# In[ ]:


## turns the index into columns
month_info.reset_index(inplace = True)


# In[ ]:


## change column names
month_info.columns = ['Visitor Type', 'Month', 'Num. Visitors']


# In[ ]:


month_info['Num. Visitors'].fillna(value = 0, inplace = True)


# we do still have the problem of missing months - Jan and Apr

# In[ ]:


month_info['Month']


# In[ ]:


month_info['Visitor Type'] = month_info['Visitor Type'].astype(str)


# In[ ]:


month_info['Month'] = month_info['Month'].astype(str)


# In[ ]:


to_add = pd.DataFrame([], columns = month_info.columns)


# In[ ]:


to_add['Visitor Type'] = ['New_Visitor', 'New_Visitor', 
                          'Other', 'Other', 
                          'Returning_Visitor', 'Returning_Visitor']


# In[ ]:


to_add['Month'] = ['Jan', 'Apr']*3


# In[ ]:


to_add['Num. Visitors'] = [0]*6


# In[ ]:


to_add


# In[ ]:


month_info = month_info.append(to_add)


# In[ ]:


month_info['Month'] = pd.Categorical(month_info['Month'], categories = ordered_months,
                                    ordered = True)


# In[ ]:


month_info


# In[ ]:


plt.figure(figsize=(12, 8))
sns.lineplot(x = 'Month', y = "Num. Visitors", hue = "Visitor Type", data = month_info, 
            hue_order = ["Returning_Visitor", "New_Visitor", "Other"], sizes=(2.5, 2.5))
plt.title("Number of Monthly Visitors by Visitor Type", fontsize=16)
plt.xlabel("Month", fontsize=16)
plt.xticks(rotation=45)
plt.ylabel("Number of Visitors", fontsize=16)
plt.show()


# What the graph is showing is spikes in the number of visitors around Mar, May, and November/December. It also shows that the number of new visitors and the number of returning visitors are fairly in sync - there isn't a particular time during the year where more new visitors seem to be attracted and returning visitors aren't. This indicates that in terms of attracting visitors overall, the influence of time of the year is about the same for new and returning visitors. In terms of attracting new visitors overall, it appears as though March, May, and November attract a lot of new visitors (and December to and extent). Spikes in November and December could be related to the holiday season, but spikes in March and May are more questionable - why? 

# ### Special days
# Are any of these spikes near "special days"?

# In[ ]:


shoppers.columns


# In[ ]:


## closeness to special days in may? 
np.average(shoppers.loc[shoppers['Month'] == 'May', 'SpecialDay'])


# In[ ]:


np.average(shoppers.loc[shoppers['Month'] == 'Mar', 'SpecialDay'])


# In[ ]:


np.average(shoppers.loc[shoppers['Month'] == 'Nov', 'SpecialDay'])


# In[ ]:


np.average(shoppers.loc[shoppers['Month'] == 'Dec', 'SpecialDay'])


# In[ ]:


shoppers['SpecialDay'].unique()


# In[ ]:


shoppers.loc[shoppers['SpecialDay'] > 0, 'Month'].unique()


# This is showing that ONLY February and May are "close to special days" which seems wrong since we know there are special days (e.g. Christmas, Father's Day) that occur on other days
# Hmmm... 
# This indicates to me that in this case the "Special Day" data that this company is collecting could be more informative if you now what they consider to be a "special day", but as is, it's not particularly useful

# ### Which visitors generated revenue? 

# Moving on then to looking at the distribution of our target variable and the thing that we are trying to predict: whether or not a site visitor generated revenue for the company

# Based on our exploratory data analysis if you asked me to guess what would be the best predictor of whether a visitor generated revenue or not is whether that visitor is a returning visitor (returning visitor = higher probability of revenue), followed by the month that visitor accessed the site (if it's May, Mar, Nov, Dec = higher probability of revenue)

# In[ ]:


plt.figure(figsize=(12, 8))
sns.countplot(x = 'Revenue', data = shoppers)
plt.title("How many visitors generated revenue overall?", fontsize=16)
plt.xlabel("Revenue Generated?", fontsize=16)
plt.xticks(rotation=45)
plt.ylabel("Number of Visitors", fontsize=16)
plt.show()


# It's evident that the majority of visitors to the site do not generate revenue (as to be expected)
# 
# <br>
# 
# This also means that our data are imbalanced

# In[ ]:


print(round(sum(shoppers['Revenue'])/len(shoppers['Revenue']), 2)*100, "% of visitors generate revenue")


# So only 16% of visitors generate revenue - low! But i'm not sure how that is in comparison to other sites - perhaps it's actually relatively high!
# 
# Since there is an imblanaced data in our label values (more falses than trues) we will use a stratified sampling approach when splitting the data into training and test data and use F1 score to assess model performance.

# ## Preparing the data for modeling

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# ### Feature Selection
# Target = "Revenue"
# 
# Features = All the other columns

# In[ ]:


shoppers.columns


# In[ ]:


shoppers.dtypes


# We have 7 categorical features, 10 numeric features, and 1 target variable (aka label or independent variable)
# 
# We must encode the categorical features so that they can be given to the model, since the model does not take python objects (strings)

# In[ ]:


cat_vars = list(shoppers.select_dtypes('category').columns)


# In[ ]:


num_vars = list(shoppers.select_dtypes('float').columns)


# In[ ]:


cat_vars = cat_vars[:-1] ## drop the revenue label


# In[ ]:


shoppers_dummies = pd.get_dummies(shoppers, columns = cat_vars)


# In[ ]:


shoppers_dummies.head()


# In[ ]:


shoppers_dummies.columns ## no revenue column because we already dropped it so no need to dop it now


# In[ ]:


X = shoppers_dummies ## independent variables


# In[ ]:


y = shoppers['Revenue'] ## dependent variable


# In[ ]:


y = y.astype(int) ## transform boolean to 0s and 1s


# ### Split the dataset into train/test
# 
# 70-30 train-test split -- because the data are imbalanced we stratify the train test split

# In[ ]:


## use the stratify argument here because of the uneven distribution of true and false values
## in the target variable (revenue)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size = 0.3, random_state = 42,
                                                stratify = y)


# ### Preprocess the data and build the model
# 
# Build a pipeline that will standardize the numeric values, tune the hyperparameters, and run classification model

# In[ ]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import svm
from xgboost import XGBClassifier
from itertools import compress


# Many choices, but I will use the classifiers Logisitic Regression, Decision Tree, and SVC

# In[ ]:


# define models to test:
base_models = [("clf", LogisticRegression(random_state=42)),
               ("clf", DecisionTreeClassifier(random_state=42)),
               ("clf", svm.SVC(random_state=42))]


# In[ ]:


svm.SVC().get_params().keys()


# ### Hyperparameters and GridSearch CV
# 
# In order to get the best performing model we can also tune the hyperparameters
# 
# Define the parameters to check for each of the different models

# In[ ]:


check_params_lr = {'clf__C': [0.1, 1, 10, 100],
                  'pca__n_components':[2, 3, 4, 5, 6]}
check_params_dt = {'pca__n_components':[2, 3, 4, 5, 6],
               'clf__criterion':['gini', 'entropy'],
               'clf__min_samples_split': [2,3,4],
               'clf__max_depth': np.arange(3,15)}
check_params_svc = {'pca__n_components': [2, 3, 4, 5, 6],
                   'clf__C': [0.1, 1, 10, 100]}

check_params = [check_params_lr, check_params_dt, check_params_svc]


# ### Build the model pipeline

# In[ ]:


def model_fit(model, params, X_train, y_train, X_test, y_test):
    
    pipe = Pipeline([('sc1', StandardScaler()),
                     ('pca', PCA()),
                    model])
    
    gs = GridSearchCV(estimator = pipe,
                     param_grid = params,
                     scoring = 'accuracy',
                     cv = 5)
    
    gs.fit(X_train, y_train)
    
    # evaluate the model on the test set
    y_true, y_pred = y_test, gs.predict(X_test)

    # get classification report for the gs model
    print(classification_report(y_true, y_pred))
    


# ### Fit the models
# 
# Run the pipeline for each model

# In[ ]:


for mod, param in zip(base_models, check_params):
    model_fit(mod, param, X_train, y_train, X_test, y_test)


# The best performing model (marginally) is the SVC. Since we have imbalanced data, we can look at the accuracy but it is more useful to look at the F1 score since that also takes into account incorrect predictions (which we may have in the minority class). The model performancees are good not great, so there is room for improvement in these models. Future directions would be to include a Random Forest classifier or an XGBoost classifier. Other options for improving the models would be dealing with the imbalanced data beyond stratifying the train-test datasplit. Dealing with imbalanced data includes weighting the data or over-sampling techniques (SMOTE and ADASYN).

# That's all for now!

# In[ ]:




