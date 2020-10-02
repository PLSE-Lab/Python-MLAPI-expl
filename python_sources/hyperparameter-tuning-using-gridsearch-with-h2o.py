#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">Hyperparameter tuning using GridSearch with H2O</font></center></h1>
# 
# 
# <img src="https://www.h2o.ai/wp-content/themes/h2o2018/templates/dist/images/h2o_logo.svg" width="400"></img>
# 
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Load packages</a>  
# - <a href='#3'>Read the data</a>  
# - <a href='#4'>Check the data</a>  
# - <a href='#5'>Data exploration</a>  
# - <a href='#6'>Feature engineering</a>  
# - <a href='#7'>Predictive Model</a>  
#     - <a href='#71'>Split the data</a> 
#     - <a href='#72'>Train GBM</a> 
#     - <a href='#73'>Parameter tuning</a>
#     - <a href='#74'>Grid Search</a>  
#     - <a href='#75'>Predict test data</a>   
# - <a href='#8'>Conclusions</a>
# - <a href='#9'>References</a>

# 
# # <a id="1">Introduction</a>  
# 
# 
# ## Dataset
# 
# This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from **April 2005** to **September 2005**. 
# 
# ## Content
# 
# There are 25 variables:
# 
# * **ID**: ID of each client
# * **LIMIT_BAL**: Amount of given credit in NT dollars (includes individual and family/supplementary credit
# * **SEX**: Gender (1=male, 2=female)
# * **EDUCATION**: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# * **MARRIAGE**: Marital status (1=married, 2=single, 3=others)
# * **AGE**: Age in years
# * **PAY_0**: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
# * **PAY_2**: Repayment status in August, 2005 (scale same as above)
# * **PAY_3**: Repayment status in July, 2005 (scale same as above)
# * **PAY_4**: Repayment status in June, 2005 (scale same as above)
# * **PAY_5**: Repayment status in May, 2005 (scale same as above)
# * **PAY_6**: Repayment status in April, 2005 (scale same as above)
# * **BILL_AMT1**: Amount of bill statement in September, 2005 (NT dollar)
# * **BILL_AMT2**: Amount of bill statement in August, 2005 (NT dollar)
# * **BILL_AMT3**: Amount of bill statement in July, 2005 (NT dollar)
# * **BILL_AMT4**: Amount of bill statement in June, 2005 (NT dollar)
# * **BILL_AMT5**: Amount of bill statement in May, 2005 (NT dollar)
# * **BILL_AMT6**: Amount of bill statement in April, 2005 (NT dollar)
# * **PAY_AMT1**: Amount of previous payment in September, 2005 (NT dollar)
# * **PAY_AMT2**: Amount of previous payment in August, 2005 (NT dollar)
# * **PAY_AMT3**: Amount of previous payment in July, 2005 (NT dollar)
# * **PAY_AMT4**: Amount of previous payment in June, 2005 (NT dollar)
# * **PAY_AMT5**: Amount of previous payment in May, 2005 (NT dollar)
# * **PAY_AMT6**: Amount of previous payment in April, 2005 (NT dollar)
# * **default.payment.next.month**: Default payment (1=yes, 0=no)
# 
# 
# ## H2O
# 
# H2O is an AI company with the vision to democratize the use of machine learning. They provide an flexible, scallable, robust platform that allows data loading, processing, visualization, feature engineering, model building and tuning. 
# 
# Few revolutionary products of H2O are:   
# * **H2O** - fully open source, distributed in-memory machine learning platform with linear scalability;  
# * **H2O AutoML** - used for automating the machine learning workflow, which includes automatic training and tuning of many models within a user-specified time-limit;  
# * **Sparkling Water** - allows users to combine the fast, scalable machine learning algorithms of H2O with the capabilities of Spark;
# * **H2O4GPU** - open source, GPU-accelerated machine learning package with APIs in Python and R that allows anyone to take advantage of GPUs to build advanced machine learning models;
# * **DriverlessAI** - employs the techniques of expert data scientists in an easy to use application that helps scale your data science efforts.  
# 
# 
# <a href="#0"><font size="1">Go to top</font></a>
# 

# # <a id="2">Load packages</a>
# 
# ## Load packages

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import time
import itertools
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Set parameters
# 
# Here we set few parameters for the analysis and models.

# In[ ]:


#DATA SPLIT IN TRAIN/VALIDATION/TEST
TRAIN_SIZE = 0.60  
VALID_SIZE = 0.20 
RANDOM_STATE = 2018
IS_LOCAL = False
import os
if(IS_LOCAL):
    PATH="../input/default-of-credit-card-clients-dataset"
else:
    PATH="../input"
print(os.listdir(PATH))


# <a href="#0"><font size="1">Go to top</font></a>
# 
# # <a id="3">Read the data</a>
# 
# For reading the data, we will use also H2O. First, we will initialize H2O.
# 
# * ## Initialize H2O
# H2O will first try to connect to an existing instance. If none available, will start one. Then informations about this engine are printed. At the end connection to the H2O server is attempted and reported.

# In[ ]:


h2o.init()


# More information are presented: the H2O cluster uptime, timezone, version, version age, cluster name, hardware resources allocated ( number of nodes, memory, cores), the connection url, H2O API extensions exposed and the Python version used.
# 
# ## Import the data
# 
# We already initialized the H2O engine, now we will use H2O to import the data.

# In[ ]:


data_df = h2o.import_file(PATH+"/UCI_Credit_Card.csv", destination_frame="data_df")


# <a href="#0"><font size="1">Go to top</font></a>
# 
# # <a id="4">Check the data</a>
# 
# We use also H2O function describe to check the data.

# In[ ]:


data_df.describe()


# There are **30,000** distinct credit card clients and there are **25** different features.
# 
# There are no missing data in the whole dataset.
# 
# The mean value for the amount of credit card limit is **167,484**. The standard deviation is unusually large, max value being **1M**.
# 
# Education level is mostly graduate school and university.
# 
# Most of the clients are either married or single (less frequent the other status).
# 
# Average age is 35.5 years, with a standard deviation of 9.2, min being 21 (must be a legal age) and max being 79.
# 
# The target value is **default.payment.next.month**. This indicates if a client will default payment next month.  As the value **0** for **default payment** means **not default** and value **1** means **default**, the mean of **0.221** means that there are **22.1%** of credit card contracts that will default next month (will verify this in the next sections of this analysis).
# 
# We will explore in more detail the data in the following section.

# <a href="#0"><font size="1">Go to top</font></a>
# 
# # <a id="5">Explore the data</a>
# 
# We will use another functions from H2O to explore the data.
# 
# Let's start by showing the distribution of features, grouped by  **default.payment.next.month**, which is the **target** value.
# 
# We start by looking how many default vs. not default cases are.
# 

# In[ ]:


df_group=data_df.group_by("default.payment.next.month").count()
df_group.get_frame()


# We can confirm that in only 22% of the cases the clients are defaulting.   
# 
# Let's explore all the features, grouped by **default.payment.next.month**.
# 
# ## Density plots for all features
# 
# We group first the data by **default.payment.next.month**. We use **kdeplot** to visualize the data distribution.
# 

# In[ ]:


features = [f for f in data_df.columns if f not in ['default.payment.next.month']]

i = 0
t0 = data_df[data_df['default.payment.next.month'] == 0].as_data_frame()
t1 = data_df[data_df['default.payment.next.month'] == 1].as_data_frame()

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(6,4,figsize=(16,24))

for feature in features:
    i += 1
    plt.subplot(6,4,i)
    sns.kdeplot(t0[feature], bw=0.5,label="Not default")
    sns.kdeplot(t1[feature], bw=0.5,label="Default")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# One can observe that there are several features that show a different distribution for values of target variable of `Default` or `Not default`.  The following features are most representative:  
# 
# * PAY_0;  
# * PAY_2;  
# * PAY_3;
# * PAY_4;  
# * PAY_5;
# * PAY_6;  
# 

# <a href="#0"><font size="1">Go to top</font></a>
# 
# ## Amount of credit limit 

# In[ ]:


d_df = data_df.as_data_frame()
plt.figure(figsize = (14,6))
plt.title('Amount of credit limit - Density Plot')
sns.set_color_codes("pastel")
sns.distplot(d_df['LIMIT_BAL'],kde=True,bins=200, color="blue")
plt.show()


# Largest group of amount of credit limit is apparently for amount of 50K. Let's verify this.

# In[ ]:


d_df['LIMIT_BAL'].value_counts().shape


# There are 81 distinct values for amount of credit limit.

# In[ ]:


d_df['LIMIT_BAL'].value_counts().head(5)


# Indeed, the largest number of credit cards are with limit of:  
# * **50,000** (**3365**), followed by:  
# * **20,000** (**1976**) and    
# * **30,000** (**1610**).  
# 

# <a href="#0"><font size="1">Go to top</font></a>
# 
# ## Amount of credit limit grouped by default payment next month
# 
# Let's visualize the density plot for amount of credit limit (**LIMIT_BAL**), grouped by **default payment next month**.

# In[ ]:


class_0 = d_df.loc[d_df['default.payment.next.month'] == 0]["LIMIT_BAL"]
class_1 = d_df.loc[d_df['default.payment.next.month'] == 1]["LIMIT_BAL"]
plt.figure(figsize = (14,6))
plt.title('Default amount of credit limit  - grouped by Payment Next Month (Density Plot)')
sns.set_color_codes("pastel")
sns.distplot(class_1,kde=True,bins=200, color="red")
sns.distplot(class_0,kde=True,bins=200, color="green")
plt.show()


# Most of defaults are for credit limits **0-100,000** (and density for this interval is larger for defaults than for non-defaults).   
# 
# Larger defaults number are for the amounts of **50,000**, **20,000** and **30,000**.

# <a href="#0"><font size="1">Go to top</font></a>
# 
# ## Credit limit vs Repayment status in Sept. 2005- April 2005, grouped by Default payment next month
# 
# Meaning of values for **PAY_0** (repayment status in Sept. 2005) is:    
# * -1=pay duly,   
# * 1=payment delay for one month,  
# * 2=payment delay for two months,  
# ...   
# 
# *  8=payment delay for eight months,   
# * 9=payment delay for nine months and above.  
# 
# For the other values **PAY_2**, **PAY_3** ... **PAY_6** the meaning is similar, referring to the respective months.   
# 

# In[ ]:


var = ['PAY_0',
       'PAY_2', 
       'PAY_3', 
       'PAY_4', 
       'PAY_5',
       'PAY_6']

for v in var:
    fig, ax = plt.subplots(ncols=1, figsize=(14,6))
    s = sns.boxplot(ax = ax, x=v, y="LIMIT_BAL", hue="default.payment.next.month",data=d_df, palette="PRGn",showfliers=False)
    plt.show();


# <a href="#0"><font size="1">Go to top</font></a>
# 
# ## Credit limit vs. sex
# 
# Let's check the credit limit distribution vs. sex. For the sex, 1 stands for male and 2 for female.

# In[ ]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
s = sns.boxplot(ax = ax1, 
                x="SEX", 
                y="LIMIT_BAL", 
                hue="SEX",data=d_df, 
                palette="PRGn",
                showfliers=True)
s = sns.boxplot(ax = ax2, 
                x="SEX", 
                y="LIMIT_BAL", 
                hue="SEX",data=d_df, 
                palette="PRGn",
                showfliers=False)
plt.show();


# The limit credit amount is quite balanced between sexes. The males have a slightly smaller Q2 and larger Q3 and Q4 and a lower mean. The female have a larger outlier max value (1M NT dollars).

# <a href="#0"><font size="1">Go to top</font></a>
# 
# ## Features correlation
# 
# 
# For the numeric values, let's represent the features correlation.
# 
# 
# Let's check the correlation of Amount of bill statement in April - September 2005.
# 
# We will use **H2O** correlation function **cor()**.
# 

# In[ ]:


var = ['BILL_AMT1',
       'BILL_AMT2',
       'BILL_AMT3',
       'BILL_AMT4',
       'BILL_AMT5',
       'BILL_AMT6']

plt.figure(figsize = (8,8))
plt.title('Amount of bill statement (Apr-Sept) \ncorrelation plot (Pearson)')
corr = data_df[var].cor().as_data_frame()
corr.index = var
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,vmin=-1, vmax=1)
plt.show()


# Correlation is decreasing with distance between months. Lowest correlations are between Sept-April.
# 
# 
# Let's check the correlation of Amount of previous payment in April - September 2005.

# In[ ]:


var = ['PAY_AMT1', 
       'PAY_AMT2', 
       'PAY_AMT3', 
       'PAY_AMT4', 
       'PAY_AMT5']

plt.figure(figsize = (8,8))
plt.title('Amount of previous payment (Apr-Sept) \ncorrelation plot (Pearson)')
corr = data_df[var].cor().as_data_frame()
corr.index = var
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,vmin=-1, vmax=1)
plt.show()


# There are no correlations between amounts of previous payments for April-Sept 2005.
# 
# Let's check the correlation between Repayment status in April - September 2005.

# In[ ]:


var = ['PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

plt.figure(figsize = (8,8))
plt.title('Repayment status (Apr-Sept) \ncorrelation plot (Pearson)')
corr = data_df[var].cor().as_data_frame()
corr.index = var
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,vmin=-1, vmax=1)
plt.show()


# Correlation is decreasing with distance between months. Lowest correlations are between Sept-April.

# <a href="#0"><font size="1">Go to top</font></a>
# 
# ## Sex, Education, Age and Marriage
# 
# 
# Let's show sex, education, age and marriage distributions.
# 
# We start by showing the boxplots with age distribution grouped by marriage status and sex.
# 
# Marriage status meaning is:
# 
# * 0 : unknown (let's consider as others as well)
# * 1 : married
# * 2 : single
# * 3 : others
# 
# Sex meaning is:
# 
# * 1 : male
# * 2 : female
# 

# In[ ]:


def boxplot_variation(feature1, feature2, feature3, width=16):
    fig, ax1 = plt.subplots(ncols=1, figsize=(width,6))
    s = sns.boxplot(ax = ax1, x=feature1, y=feature2, hue=feature3,
                data=d_df, palette="PRGn",showfliers=False)
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show();


# In[ ]:


boxplot_variation('MARRIAGE','AGE', 'SEX',8)


# It looks like Married status 3 (others), with mean values over 40 and Q4 values over 60 means mostly vidowed or divorced whilst Married status 0 could be not specified or divorced, as Q1 values are above values for married of both sexes.
# 
# Married males have mean age above married women. Unmarried males have mean value for age above unmarried women as well but closer. Q3 abd Q4 values for married man are above corresponding values for married women.
# 
# 
# Let's show the boxplots with age distribution grouped by education and marriage.
# 
# Education status meaning is:
# 
# * 1 : graduate school
# * 2 : university
# * 3 : high school
# * 4 : others
# * 5 : unknown
# * 6 : unknow

# In[ ]:


boxplot_variation('EDUCATION','AGE', 'MARRIAGE',12)


# <a href="#0"><font size="1">Go to top</font></a>
# 
# ## Age, sex and credit amount limit
# 
# 
# Let's show the  boxplots with credit amount limit distribution grouped by age and sex.

# In[ ]:


boxplot_variation('AGE','LIMIT_BAL', 'SEX',16)


# Mean, Q3 and Q4 values are increasing for both male and female with age until aroung 35 years and then they are oscilating and get to a maximum of Q4 for males at age 64.
# 
# Mean values are generally smaller for males than for females, with few exceptions, for example at age 39, 48, until approximately 60, where mean values for males are generally larger than for females.

# ## Marriage status, education level and credit amount limit
# 
# 
# Let's show the  boxplots with credit amount limit distribution grouped by marriage status and education level.

# In[ ]:


boxplot_variation('MARRIAGE','LIMIT_BAL', 'EDUCATION',12)


# <a href="#0"><font size="1">Go to top</font></a>
# 
# # <a id="6">Feature engineering</a>  
# 
# 
# Let's add few new features.

# In[ ]:


data_df['EDUCATION_SEX'] = data_df['EDUCATION'] + "_" + data_df['SEX']
data_df['SEX_MARRIAGE'] = data_df['SEX'] + "_" + data_df['MARRIAGE']
data_df['EDUCATION_MARRIAGE'] = data_df['EDUCATION'] + "_" + data_df['MARRIAGE']
data_df['EDUCATION_MARRIAGE_SEX'] = data_df['EDUCATION'] + "_" + data_df['MARRIAGE'] + "_" + data_df['SEX']


# Let's check the shape again.

# In[ ]:


data_df.shape


# <a href="#0"><font size="1">Go to top</font></a>
# 
# # <a id="7">Predictive Model</a>  
# 
# 
# 
# 

# ## <a id="71">Split the data</a>
# 
# Let's start by spliting the data in train, validation and test sets. We will use 60%, 20% and 20% splits.   
# 
# Please not here that we can directly split in train, validation and test set, we do not need to do it in two steps.

# In[ ]:


train_df, valid_df, test_df = data_df.split_frame(ratios=[TRAIN_SIZE, VALID_SIZE], seed=2018)
target = "default.payment.next.month"
train_df[target] = train_df[target].asfactor()
valid_df[target] = valid_df[target].asfactor()
test_df[target] = test_df[target].asfactor()
print("Number of rows in train, valid and test set : ", train_df.shape[0], valid_df.shape[0], test_df.shape[0])


# ## <a id="72">Train GBM</a>
# 
# 
# We will use a GBM model provided by H2O framework (H2OGradientBoostingEstimator) for prediction of the **target** (**default.payment.next.month**).  
# 
# The training predictors columns, the target values and the dataframe are specified as parameters of the **train** function.
# 

# In[ ]:


# define the predictor list - all the features analyzed before (all columns but 'default.payment.next.month')
predictors = features
# initialize the H2O GBM 
gbm = H2OGradientBoostingEstimator()
# train with the initialized model
gbm.train(x=predictors, y=target, training_frame=train_df)


# ### Model evaluation
# 
# Let's inspect the model already trained. We can print the summary:

# In[ ]:


gbm.summary()


# This shows that we used 50 trees, 50 internal trees. It is also showing the min and max tree depth (5,5), the min and max number of leaves (16,32) and the mean values for tree depth and number of leaves.
# 
# We can also inspect the model further, looking to other informations.
# 
# Let's see the model performance for the train set.

# In[ ]:


print(gbm.model_performance(train_df))


# For the train set, the Gini coefficient obtained is 0.64, the AUC score is 0.82. LogLoss is 0.39.
# 
# Let's check now the performance of the model with the validation set.

# In[ ]:


print(gbm.model_performance(valid_df))


# For the validation set, the Gini score obtained is 0.55, the AUC score is 0.77 and LogLoss is 0.43.
# 
# 
# To summarize, we obtained AUC score 0.82 for train set and 0.77 for test set.
# 
# Let's use the validation set to tune the parameters.
# 
# <a href="#0"><font size="1">Go to top</font></a>
# 
# ## <a id="73">Parameters tuning</a>
# 
# 

# In[ ]:


tuned_gbm  = H2OGradientBoostingEstimator(
    ntrees = 2000,
    learn_rate = 0.02,
    stopping_rounds = 25,
    stopping_metric = "AUC",
    col_sample_rate = 0.65,
    sample_rate = 0.65,
    seed = RANDOM_STATE
)      
tuned_gbm.train(x=predictors, y=target, training_frame=train_df, validation_frame=valid_df)


# Let's check the validation AUC score.

# In[ ]:


tuned_gbm.model_performance(valid_df).auc()


# The performance was not improved (AUC actually decreased).
# 
# <a href="#0"><font size="1">Go to top</font></a>
# 
# 
# 
# ## <a id="74">Grid Search</a>
# 
# Let's try Hyperparamater tuning using Grid Search. 
# 
# The hyperparameters grid is specified in the **hyper_params**, for each parameters beign given the set of values we want to explore.

# In[ ]:


grid_search_gbm = H2OGradientBoostingEstimator(
    stopping_rounds = 25,
    stopping_metric = "AUC",
    col_sample_rate = 0.65,
    sample_rate = 0.65,
    seed = RANDOM_STATE
) 

hyper_params = {
    'learn_rate':[0.01, 0.02, 0.03],
    'max_depth':[4,8,16,24],
    'ntrees':[50, 250, 1000]}

grid = H2OGridSearch(grid_search_gbm, hyper_params,
                         grid_id='depth_grid',
                         search_criteria={'strategy': "Cartesian"})
#Train grid search
grid.train(x=predictors, 
           y=target,
           training_frame=train_df,
           validation_frame=valid_df)


# We will explore the model, printing the parameters which give the best AUC.

# In[ ]:


grid_sorted = grid.get_grid(sort_by='auc',decreasing=True)


# In[ ]:


print(grid_sorted)


# Let's pick the best model, selected by AUC:

# In[ ]:


best_gbm = grid_sorted.models[0]


# Let's inspect the best model obtained with Hyperparameters tuning using GridSearch.

# In[ ]:


print(best_gbm)


# The best model has an AUC score of 0.82 for the train set and of 0.78 for validation set.
# 
# Let's show the variable importance.

# In[ ]:


best_gbm.varimp_plot()


# <a href="#0"><font size="1">Go to top</font></a>
# 
# ## <a id="75">Predict test data</a>
# 
# Let's use the best model to predict the target value for the test data.

# In[ ]:


pred_val = (best_gbm.predict(test_df[predictors])[0]).as_data_frame()
true_val = (test_df[target]).as_data_frame()
prediction_auc = roc_auc_score(pred_val, true_val)
prediction_auc


# The AUC score for test data is 0.698.
# 
# 
# <a href="#0"><font size="1">Go to top</font></a>

# # <a id="8">Conclusions</a>
# 
# We initialized the H2O engine, using some H2O features for importing and inspecting the data.  
# 
# We then continued visualizing the features and understanding the relationship between different features.   
# 
# A small feature engineering step was added.
# 
# We then splited the data in train, validation and test set. We  trained a  predictive model using GBM model from H2O, starting with a simple model, following with simple parameter tuning and then we used GridSearch to find the best parameters for the model, looking to maximize the AUC score for the validation set.   
# 
# Finaly, with the best model, we atempted to predict the target value for the test data.   
# 
# The AUC score for the test data was <font color="red">**0.698**</font>.
# 
# <a href="#0"><font size="1">Go to top</font></a>

# # <a id="9">References</a>
# 
# [1] Default Credit Card Clients Dataset,  https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset/    
# [2] SRK, Getting started with H2O, https://www.kaggle.com/sudalairajkumar/getting-started-with-h2o    
# [3] H2O, the company, https://www.h2o.ai/company/  
# [4] H2O, the company (Wikipedia), https://en.wikipedia.org/wiki/H2O_(software)
# 
# 
# <a href="#0"><font size="1">Go to top</font></a>
