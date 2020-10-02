#!/usr/bin/env python
# coding: utf-8

# # EDA + LOGISTIC REGRESSION

# <p> We are given two datasets: census.csv and census_test.csv. We'll have two determine applyin EDA if census_test is either a random sample from census.csv or if it's manipulated. Once we've guessed that, we'll apply a logistic regression and measure it with different statistics. </p>

# # Exercise 1: EDA analysis

# In[ ]:


#First, we import the libraries we need

get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import seaborn as sns


# We are given two datasets: census and census_test. Applying EDA, we have to guess if census_test was randomly selected from census. If not, it may be biased. 
# 
# First of all, well import census dataset and give a quick view to the variables it contains

# In[ ]:


df_census = pd.read_csv("../input/census.csv")
df_census_test = pd.read_csv("../input/census_test.csv")
df_census.head(10)


# We'll check dtypes that were automatically guessed by pandas library

# In[ ]:


df_census.dtypes


# In[ ]:


df_census.greater_than_50k.unique()


# From the previous types, we see that:
#     <ul>
#     <li>age: discrete variable.</li> 
#     <li>workclass: numeric variable.</li>
#     <li>education: nominal variable.</li>
#     <li>education_num: ordinal variable.</li>
#     <li>marital_status: nominal variable</li>
#     <li>occupation: nominal variable.</li>
#     <li>relationship: nominal variable</li>
#     <li>race: nominal variable.</li>
#     <li>gender: nominal variable</li>
#     <li>hours_per_week: discrete variable.</li>
#     <li>native_country: nominal variable.</li>
#     <li>greater_than_50k: binary data.</li>
#     </ul>

# First of all, we are going to clean all trailing and leading whitespaces

# In[ ]:


#For df_census
df_census_selected = df_census.select_dtypes(include = ["object"])
df_census[df_census_selected.columns] = df_census_selected.apply(lambda x: x.str.strip())

#For df_cesus_test

df_census_selected = df_census_test.select_dtypes(include = ["object"])
df_census_test[df_census_selected.columns] = df_census_selected.apply(lambda x: x.str.strip())


# Since there's categorical data, we'll take advantage of the "category" dtype pd.Series

# In[ ]:


df_census = df_census.astype(
{
        "workclass" : "category", 
        "education" : "category",
        "marital_status" : "category",
        "occupation" : "category",
        "relationship" : "category",
        "race" : "category",
        "native_country" : "category"
}
)

df_census_test = df_census_test.astype(
{
        "workclass" : "category", 
        "education" : "category",
        "marital_status" : "category",
        "occupation" : "category",
        "relationship" : "category",
        "race" : "category",
        "native_country" : "category"
}
)


# <p> In order to understand the dataset, we'll check out the categories in each variable </p>

# In[ ]:


for col_name in df_census.columns:
    if(isinstance(df_census[col_name].dtype, pd.core.dtypes.dtypes.CategoricalDtype)):
        print("Catergories for {} are:".format(col_name))
        for category in list(df_census[col_name].cat.categories):
            print("  -", category) 


# Next, we are going to compare statistics for numerical data.

# In[ ]:


pd.concat([df_census[["age", "hours_per_week"]].describe(), df_census_test[["age", "hours_per_week"]].describe() ], axis = 1)


# In[ ]:


sns.distplot(df_census["age"])


# In[ ]:


import scipy.stats as stats
print("H0 hypothesis test:",stats.normaltest(df_census["age"]))


# The former distributionn is not normal

# In[ ]:


sns.distplot(df_census_test["age"])


# In[ ]:


sns.distplot(df_census["hours_per_week"])


# In[ ]:


print("H0 hypothesis test:",stats.normaltest(df_census["hours_per_week"]))


# The former distribution isn't normal too

# In[ ]:


sns.distplot(df_census_test["hours_per_week"])


# From the previous comparison, we see that there isn't such a big difference between census and census_test. From this perspective, the census_test may seem randomly selected. Let's see what happens with categorical data.
# 
# To do that, we are going to calculate proportions for each value
# 

# <b>For greater_than_50k </b>

# In[ ]:


print("Proportions of 1s for census: ", df_census.greater_than_50k.mean())
print("Proportions of 1s for census test: ", df_census_test.greater_than_50k.mean())


# <b>For workclass </b>

# In[ ]:


df_1 = pd.DataFrame({ "census": (df_census.workclass.value_counts()/df_census.workclass.count())})
df_2 = pd.DataFrame({ "census": (df_census_test.workclass.value_counts()/df_census_test.workclass.count())})
df_2 =df_2.rename(columns = {"census" : "census_test"})
pd.concat([df_1, df_2], axis = 1)


# So far, so good, but we still have more work to do. We will group variables in order to
# analise the proportion of paychecks greater than 50K

# In[ ]:


df_census_mean = pd.DataFrame({"avg_over_50k" : df_census[["workclass",  "greater_than_50k"]].groupby(["workclass"])["greater_than_50k"].mean()})
df_census_mean_test = pd.DataFrame({"avg_over_50k_test" : df_census_test[["workclass",  "greater_than_50k"]].groupby(["workclass"])["greater_than_50k"].mean()})
pd.concat([df_census_mean, df_census_mean_test], axis = 1)


# We have 0 vs 28% of people in our test dataset that have a income greater than 50K without being paid!!! <b> How's that possible :S !</b>
# 
# Either it's manually modified or there's is an error in the dataset. Anyway, we should check the number of observations, because the fewer the number of observations, the greater the margin of error.

# In[ ]:


df_census_test[["workclass",  "greater_than_50k"]].groupby(["workclass"])["greater_than_50k"].value_counts()


# We'll check other variables

# <b>Education </b>

# In[ ]:


df_census_mean = pd.DataFrame({"avg_over_50k" : df_census[["education",  "greater_than_50k"]].groupby(["education"])["greater_than_50k"].mean()})
df_census_mean_test = pd.DataFrame({"avg_over_50k_test" : df_census_test[["education",  "greater_than_50k"]].groupby(["education"])["greater_than_50k"].mean()})
pd.concat([df_census_mean, df_census_mean_test], axis = 1)


# <b>Marital status </b>

# In[ ]:


df_census_mean = pd.DataFrame({"avg_over_50k" : df_census[["marital_status",  "greater_than_50k"]].groupby(["marital_status"])["greater_than_50k"].mean()})
df_census_mean_test = pd.DataFrame({"avg_over_50k_test" : df_census_test[["marital_status",  "greater_than_50k"]].groupby(["marital_status"])["greater_than_50k"].mean()})
pd.concat([df_census_mean, df_census_mean_test], axis = 1)


# Also, it seems that there is a 12% of difference between Married-AF-spouse. But this is not so relevant.

# In[ ]:


df_census_test[["marital_status",  "greater_than_50k"]].groupby(["marital_status"])["greater_than_50k"].value_counts()


# <b>Occupation </b>

# In[ ]:


df_census_mean = pd.DataFrame({"avg_over_50k" : df_census[["occupation",  "greater_than_50k"]].groupby(["occupation"])["greater_than_50k"].mean()})
df_census_mean_test = pd.DataFrame({"avg_over_50k_test" : df_census_test[["occupation",  "greater_than_50k"]].groupby(["occupation"])["greater_than_50k"].mean()})
pd.concat([df_census_mean, df_census_mean_test], axis = 1)


# Even if there are 66% of Armed-Forces from test dataset have an income over 50K, it may be that the sample took more armed forces observations, since there are few of them.

# In[ ]:


df_census["occupation"].value_counts()


# In[ ]:


gr_occupation = sns.barplot(y = "occupation", x ="greater_than_50k", data=df_census, estimator = np.mean)


# In[ ]:


gr_occupation = sns.barplot(y = "occupation", x ="greater_than_50k", data=df_census_test, estimator = np.mean)


# From the former graphics, we see that the margin of error in Armed-Forces is high, because there are very few observations. Because of that we can't state that census.csv is manipulated.

# <b>Relationship </b>

# In[ ]:


df_census_mean = pd.DataFrame({"avg_over_50k" : df_census[["relationship",  "greater_than_50k"]].groupby(["relationship"])["greater_than_50k"].mean()})
df_census_mean_test = pd.DataFrame({"avg_over_50k_test" : df_census_test[["relationship",  "greater_than_50k"]].groupby(["relationship"])["greater_than_50k"].mean()})
pd.concat([df_census_mean, df_census_mean_test], axis = 1)


# <b>Race </b>

# In[ ]:


df_census_mean = pd.DataFrame({"avg_over_50k" : df_census[["race",  "greater_than_50k"]].groupby(["race"])["greater_than_50k"].mean()})
df_census_mean_test = pd.DataFrame({"avg_over_50k_test" : df_census_test[["race",  "greater_than_50k"]].groupby(["race"])["greater_than_50k"].mean()})
pd.concat([df_census_mean, df_census_mean_test], axis = 1)


# <b>Gender </b>

# In[ ]:


df_census_mean = pd.DataFrame({"avg_over_50k" : df_census[["gender",  "greater_than_50k"]].groupby(["gender"])["greater_than_50k"].mean()})
df_census_mean_test = pd.DataFrame({"avg_over_50k_test" : df_census_test[["gender",  "greater_than_50k"]].groupby(["gender"])["greater_than_50k"].mean()})
pd.concat([df_census_mean, df_census_mean_test], axis = 1)


# <b>Native country </b>

# In[ ]:


df_census_mean = pd.DataFrame({"avg_over_50k" : df_census[["native_country",  "greater_than_50k"]].groupby(["native_country"])["greater_than_50k"].mean()})
df_census_mean_test = pd.DataFrame({"avg_over_50k_test" : df_census_test[["native_country",  "greater_than_50k"]].groupby(["native_country"])["greater_than_50k"].mean()})
pd.concat([df_census_mean, df_census_mean_test], axis = 1)


# One last plot to show the proportions

# In[ ]:


sns.countplot(y='occupation', hue='greater_than_50k', data=df_census)


# In[ ]:


sns.countplot(y='occupation', hue='greater_than_50k', data=df_census_test)


# <b> We can state that census_test.csv is not manipulated. Data distribution is very similar for both datasets and when there is a significant difference, the margin of error is high because there are very few observations (as in Armed-foces). Also, it seems that there is an error because there are people workclass Without-pay that have an income over 50K, while in the original dataset (census.csv), there were 0 people. 

# # Exercise 2: applying logistic regresion

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


# First of all, we have to deal with null values. A good aproach to deal with null values
# is applyin a heatmap

# In[ ]:


sns.heatmap(df_census.isnull())


# In[ ]:


df_census.dropna(inplace = True)


# <b> We are going to calculate a logistic regression using sklearn </b>

# In[ ]:


from sklearn.model_selection import train_test_split
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


# Because we can't compute independent categorical variables in a machine learning algorithm, we have to use the pandas method get_dummies.

# In[ ]:


df_census_dummies = pd.get_dummies(df_census[['workclass', 'education', 'marital_status',
       'occupation', 'relationship', 'race', 'gender',
       'native_country']])


# In[ ]:


df_census.drop(['workclass', 'education', 'marital_status',
       'occupation', 'relationship', 'race', 'gender',
       'native_country'], axis = 1, inplace=True)


# In[ ]:


#We get together original dataset with dummies

df_census_train = pd.concat([df_census_dummies, df_census], axis = 1)


# We'll split census dataset for training and test

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df_census_train.drop('greater_than_50k',axis=1), 
                                                    df_census_train['greater_than_50k'], test_size=0.30, 
                                                    random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[ ]:


# Our accuracy is:
logmodel_score = logmodel.score(X_test, y_test)
print("Model Score: " ,logmodel_score)


# In[ ]:


pd.DataFrame(metrics.confusion_matrix(y_test, predictions), columns = ["PREDICTED_FALSE","PREDICTED_TRUE" ], index = ["ACTUAL_FALSE", "ACTUAL_TRUE"])


# <b> Now let's calculate KS and GINI index </b>

# KS and GINI for train

# In[ ]:


a = logmodel.predict_proba(X_train)[:,1]
b = y_train

tot_bads=1.0*sum(b)
tot_goods=1.0*(len(b)-tot_bads)
elements_df = pd.DataFrame({'probability': a,'gbi': b})
pivot_elements_df = pd.pivot_table(elements_df, values='gbi', index=['probability'], aggfunc=[sum,len]).fillna(0)
max_ks = perc_goods = perc_bads = cum_perc_bads = cum_perc_goods = 0
cum_perc_bads_list = [0.0]
cum_perc_goods_list = [0.0]
cum_cp_minus = [0.0]
cum_cp_plus = [0.0]

for i in range(len(pivot_elements_df)):
    perc_goods =  ((pivot_elements_df['len'].iloc[i]['gbi'] - pivot_elements_df['sum'].iloc[i]['gbi']) / tot_goods)
    perc_bads = float(pivot_elements_df['sum']['gbi'].iloc[i]/ tot_bads)
    cum_perc_goods += perc_goods   
    cum_perc_bads += perc_bads

    
    cum_perc_bads_list.append(cum_perc_bads)
    cum_perc_goods_list.append(cum_perc_goods)
    cum_diff = cum_perc_bads-cum_perc_goods

    cum_cp_minus.append(0.0)    
    cum_cp_minus[-1] = cum_perc_bads_list[-1] - cum_perc_bads_list[-2]

    cum_cp_plus.append(0.0)
    cum_cp_plus[-1] = cum_perc_goods_list[-1] + cum_perc_goods_list[-2]
    
    
    if abs(cum_diff) > max_ks:
        max_ks = abs(cum_diff)

print('KS=',max_ks)


# In[ ]:


z_score = 0
for i in range(len(cum_cp_plus)):
    try:
        z_score +=  cum_cp_minus[i] * cum_cp_plus[i]
    except:
        pass
print('GINI=',1- z_score/100.0)


# KS and GINI for test

# In[ ]:


a = logmodel.predict_proba(X_test)[:,1]
b = y_test

tot_bads=1.0*sum(b)
tot_goods=1.0*(len(b)-tot_bads)
elements_df = pd.DataFrame({'probability': a,'gbi': b})
pivot_elements_df = pd.pivot_table(elements_df, values='gbi', index=['probability'], aggfunc=[sum,len]).fillna(0)
max_ks = perc_goods = perc_bads = cum_perc_bads = cum_perc_goods = 0
cum_perc_bads_list = [0.0]
cum_perc_goods_list = [0.0]
cum_cp_minus = [0.0]
cum_cp_plus = [0.0]

for i in range(len(pivot_elements_df)):
    perc_goods =  ((pivot_elements_df['len'].iloc[i]['gbi'] - pivot_elements_df['sum'].iloc[i]['gbi']) / tot_goods)
    perc_bads = float(pivot_elements_df['sum']['gbi'].iloc[i]/ tot_bads)
    cum_perc_goods += perc_goods   
    cum_perc_bads += perc_bads

    
    cum_perc_bads_list.append(cum_perc_bads)
    cum_perc_goods_list.append(cum_perc_goods)
    cum_diff = cum_perc_bads-cum_perc_goods

    cum_cp_minus.append(0.0)    
    cum_cp_minus[-1] = cum_perc_bads_list[-1] - cum_perc_bads_list[-2]

    cum_cp_plus.append(0.0)
    cum_cp_plus[-1] = cum_perc_goods_list[-1] + cum_perc_goods_list[-2]
    
    
    if abs(cum_diff) > max_ks:
        max_ks = abs(cum_diff)

print('KS=',max_ks)


# In[ ]:


z_score = 0
for i in range(len(cum_cp_plus)):
    try:
        z_score +=  cum_cp_minus[i] * cum_cp_plus[i]
    except:
        pass
print('GINI=',1- z_score/100.0)

