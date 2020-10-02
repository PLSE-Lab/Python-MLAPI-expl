#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">Credit Card Fraud Detection Predictive Models</font></center></h1>
# 
# 
# <img src="https://kaggle2.blob.core.windows.net/datasets-images/310/684/3503c6c827ca269cc00ffa66f2a9c207/dataset-card.jpg" width="400"></img>
# 
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Load packages</a>  
# - <a href='#3'>Read the data</a>  
# - <a href='#4'>Check the data</a>  
#     - <a href='#41'>Glimpse the data</a>  
#     - <a href='#42'>Check missing data</a>
#     - <a href='#43'>Check data unbalance</a>
# - <a href='#5'>Data exploration</a>
# - <a href='#6'>Predictive models</a>  
#     - <a href='#61'>RandomForrestClassifier</a> 
#     - <a href='#62'>AdaBoostClassifier</a>     
#     - <a href='#63'>CatBoostClassifier</a> 
#     - <a href='#64'>XGBoost</a> 
#     - <a href='#65'>LightGBM</a> 
# - <a href='#7'>Conclusions</a>
# - <a href='#8'>References</a>
# 

# # <a id="1">Introduction</a>  
# 
# The datasets contains transactions made by credit cards in **September 2013** by european cardholders. This dataset presents transactions that occurred in two days, where we have **492 frauds** out of **284,807 transactions**. The dataset is **highly unbalanced**, the **positive class (frauds)** account for **0.172%** of all transactions.  
# 
# It contains only numerical input variables which are the result of a **PCA transformation**.   
# 
# Due to confidentiality issues, there are not provided the original features and more background information about the data.  
# 
# * Features **V1**, **V2**, ... **V28** are the **principal components** obtained with **PCA**;  
# * The only features which have not been transformed with PCA are **Time** and **Amount**. Feature **Time** contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature **Amount** is the transaction Amount, this feature can be used for example-dependant cost-senstive learning.   
# * Feature **Class** is the response variable and it takes value **1** in case of fraud and **0** otherwise.  
# 
# 

# # <a id="2">Load packages</a>

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


import gc
from datetime import datetime 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn import svm
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb

pd.set_option('display.max_columns', 100)


RFC_METRIC = 'gini'  #metric used for RandomForrestClassifier
NUM_ESTIMATORS = 100 #number of estimators used for RandomForrestClassifier
NO_JOBS = 4 #number of parallel jobs used for RandomForrestClassifier


#TRAIN/VALIDATION/TEST SPLIT
#VALIDATION
VALID_SIZE = 0.20 # simple validation using train_test_split
TEST_SIZE = 0.20 # test size using_train_test_split

#CROSS-VALIDATION
NUMBER_KFOLDS = 5 #number of KFolds for cross-validation



RANDOM_STATE = 2018

MAX_ROUNDS = 1000 #lgb iterations
EARLY_STOP = 50 #lgb early stop 
OPT_ROUNDS = 1000  #To be adjusted based on best validation rounds
VERBOSE_EVAL = 50 #Print out metric result

IS_LOCAL = False

import os

if(IS_LOCAL):
    PATH="../input/credit-card-fraud-detection"
else:
    PATH="../input"
print(os.listdir(PATH))


# # <a id="3">Read the data</a>

# In[ ]:


data_df = pd.read_csv(PATH+"/creditcard.csv")


# # <a id="4">Check the data</a>

# In[ ]:


print("Credit Card Fraud Detection data -  rows:",data_df.shape[0]," columns:", data_df.shape[1])


# ## <a id="41">Glimpse the data</a>
# 
# We start by looking to the data features (first 5 rows).

# In[ ]:


data_df.head()


# Let's look into more details to the data.

# In[ ]:


data_df.describe()


# Looking to the **Time** feature, we can confirm that the data contains **284,807** transactions, during 2 consecutive days (or **172792** seconds).

# ## <a id="42">Check missing data</a>  
# 
# Let's check if there is any missing data.

# In[ ]:


total = data_df.isnull().sum().sort_values(ascending = False)
percent = (data_df.isnull().sum()/data_df.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()


# There is no missing data in the entire dataset.

# ## <a id="43">Data unbalance</a>

# Let's check data unbalance with respect with *target* value, i.e. **Class**.

# In[ ]:


temp = data_df["Class"].value_counts()
df = pd.DataFrame({'Class': temp.index,'values': temp.values})

trace = go.Bar(
    x = df['Class'],y = df['values'],
    name="Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)",
    marker=dict(color="Red"),
    text=df['values']
)
data = [trace]
layout = dict(title = 'Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)',
          xaxis = dict(title = 'Class', showticklabels=True), 
          yaxis = dict(title = 'Number of transactions'),
          hovermode = 'closest',width=600
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='class')


# Only **492** (or **0.172%**) of transaction are fraudulent. That means the data is highly unbalanced with respect with target variable **Class**.

# # <a id="5">Data exploration</a>

# ## Transactions in time

# In[ ]:


class_0 = data_df.loc[data_df['Class'] == 0]["Time"]
class_1 = data_df.loc[data_df['Class'] == 1]["Time"]
#plt.figure(figsize = (14,4))
#plt.title('Credit Card Transactions Time Density Plot')
#sns.set_color_codes("pastel")
#sns.distplot(class_0,kde=True,bins=480)
#sns.distplot(class_1,kde=True,bins=480)
#plt.show()
hist_data = [class_0, class_1]
group_labels = ['Not Fraud', 'Fraud']

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
fig['layout'].update(title='Credit Card Transactions Time Density Plot', xaxis=dict(title='Time [s]'))
iplot(fig, filename='dist_only')


# Fraudulent transactions have a distribution more even than valid transactions - are equaly distributed in time, including the low real transaction times, during night in Europe timezone.

# ## Transactions amount

# In[ ]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=data_df, palette="PRGn",showfliers=True)
s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=data_df, palette="PRGn",showfliers=False)
plt.show();


# In[ ]:


tmp = data_df[['Amount','Class']].copy()
class_0 = tmp.loc[tmp['Class'] == 0]['Amount']
class_1 = tmp.loc[tmp['Class'] == 1]['Amount']
class_0.describe()


# In[ ]:


class_1.describe()


# The real transaction have a larger mean value, larger Q1, smaller Q3 and Q4 and larger outliers; fraudulent transactions have a smaller Q1 and mean, larger Q4 and smaller outliers.
# 
# Let's plot the fraudulent transactions (amount) against time. The time is shown is seconds from the start of the time period (totaly 48h, over 2 days).

# In[ ]:


fraud = data_df.loc[data_df['Class'] == 1]

trace = go.Scatter(
    x = fraud['Time'],y = fraud['Amount'],
    name="Amount",
     marker=dict(
                color='rgb(238,23,11)',
                line=dict(
                    color='red',
                    width=1),
                opacity=0.5,
            ),
    text= fraud['Amount'],
    mode = "markers"
)
data = [trace]
layout = dict(title = 'Amount of fraudulent transactions',
          xaxis = dict(title = 'Time [s]', showticklabels=True), 
          yaxis = dict(title = 'Amount'),
          hovermode='closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='fraud-amount')


# ## Features correlation

# In[ ]:


plt.figure(figsize = (14,14))
plt.title('Credit Card Transactions features correlation plot (Pearson)')
corr = data_df.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Reds")
plt.show()


# As expected, there is no notable correlation between features **V1**-**V28**. There are certain correlations between some of these features and **Time** (inverse correlation with **V3**) and **Amount** (direct correlation with **V7** and **V20**, inverse correlation with **V1** and **V5**).
# 
# 
# Let's plot the correlated and inverse correlated values on the same graph.
# 
# Let's start with the direct correlated values: {V20;Amount} and {V7;Amount}.

# In[ ]:


s = sns.lmplot(x='V20', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})
s = sns.lmplot(x='V7', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})
plt.show()


# We can confirm that the two couples of features are correlated (the regression lines for **Class = 0** have a positive slope, whilst the regression line for **Class = 1** have a smaller positive slope).
# 
# Let's plot now the inverse correlated values.

# In[ ]:


s = sns.lmplot(x='V2', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})
s = sns.lmplot(x='V5', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})
plt.show()


# We can confirm that the two couples of features are inverse correlated (the regression lines for **Class = 0** have a negative slope while the regression lines for **Class = 1** have a very small negative slope).
# 

# ## Features density plot

# In[ ]:


var = data_df.columns.values

i = 0
t0 = data_df.loc[data_df['Class'] == 0]
t1 = data_df.loc[data_df['Class'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8,4,figsize=(16,28))

for feature in var:
    i += 1
    plt.subplot(8,4,i)
    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")
    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# For some of the features we can observe a good selectivity in terms of distribution for the two values of **Class**: **V4**, **V11** have clearly separated distributions for **Class** values 0 and 1, **V12**, **V14**, **V18** are partially separated, **V1**, **V2**, **V3**, **V10** have a quite distinct profile, whilst **V25**, **V26**, **V28** have similar profiles for the two values of **Class**.  
# 
# In general, with just few exceptions (**Time** and **Amount**), the features distribution for legitimate transactions (values of **Class = 0**)  is centered around 0, sometime with a long queue at one of the extremities. In the same time, the fraudulent transactions (values of **Class = 1**) have a skewed (asymmetric) distribution.

# # <a id="6">Predictive models</a>  
# 
# 

# ### Define predictors and target values
# 
# Let's define the predictor features and the target features. Categorical features, if any, are also defined. In our case, there are no categorical feature.

# In[ ]:


target = 'Class'
predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',       'Amount']


# ### Split data in train, test and validation set
# 
# Let's define train, validation and test sets.

# In[ ]:


train_df, test_df = train_test_split(data_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True )
train_df, valid_df = train_test_split(train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True )


# Let's start with a RandomForrestClassifier <a href='#8'>[3]</a>   model.

# ## <a id="61">RandomForestClassifier</a>
# 
# 
# ### Define model parameters
# 
# Let's set the parameters for the model.

# Let's run a model using the training set for training. Then, we will use the validation set for validation. 
# 
# We will use as validation criterion **GINI**, which formula is **GINI = 2 * (AUC) - 1**, where **AUC** is the **Receiver Operating Characteristic - Area Under Curve (ROC-AUC)** <a href='#8'>[4]</a>.  Number of estimators is set to **100** and number of parallel jobs is set to **4**.
# 
# We start by initializing the RandomForestClassifier.

# In[ ]:


clf = RandomForestClassifier(n_jobs=NO_JOBS, 
                             random_state=RANDOM_STATE,
                             criterion=RFC_METRIC,
                             n_estimators=NUM_ESTIMATORS,
                             verbose=False)


# Let's train the **RandonForestClassifier** using the **train_df** data and **fit** function.

# In[ ]:


clf.fit(train_df[predictors], train_df[target].values)


# Let's now predict the **target** values for the **valid_df** data, using **predict** function.

# In[ ]:


preds = clf.predict(valid_df[predictors])


# Let's also visualize the features importance.
# 
# ### Features importance

# In[ ]:


tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()   


# The most important features are **V17**, **V12**, **V14**, **V10**, **V11**, **V16**.
# 
# 
# ### Confusion matrix
# 
# Let's show a confusion matrix for the results we obtained. 

# In[ ]:


cm = pd.crosstab(valid_df[target].values, preds, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cm, 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix', fontsize=14)
plt.show()


# ### Type I error and Type II error
# 
# We need to clarify that confussion matrix are not a very good tool to represent the results in the case of largely unbalanced data, because we will actually need a different metrics that accounts in the same time for the **selectivity** and **specificity** of the method we are using, so that we minimize in the same time both **Type I errors** and **Type II errors**.
# 
# 
# **Null Hypothesis** (**H0**) - The transaction is not a fraud.  
# **Alternative Hypothesis** (**H1**) - The transaction is a fraud.  
# 
# **Type I error** - You reject the null hypothesis when the null hypothesis is actually true.  
# **Type II error** - You fail to reject the null hypothesis when the the alternative hypothesis is true.  
# 
# **Cost of Type I error** - You erroneously presume that the the transaction is a fraud, and a true transaction is rejected.  
# **Cost of Type II error** - You erroneously presume that the transaction is not a fraud and a ffraudulent transaction is accepted.  
# 
# The following image explains what **Type I error** and **Type II error** are:    
# 
# 
# <img src="https://i.stack.imgur.com/x1GQ1.png" width="600"/>
# 
# And this alternative image explains even better:  
# 
# <img src="https://i2.wp.com/flowingdata.com/wp-content/uploads/2014/05/Type-I-and-II-errors1.jpg" width="600"/>
# 
# 
# 
# Let's calculate the ROC-AUC score <a href='#8'>[4]</a>.
# 
# ### Area under curve

# In[ ]:


roc_auc_score(valid_df[target].values, preds)


# The **ROC-AUC** score obtained with **RandomForrestClassifier** is **0.85**.
# 
# 
# 
# 

# ## <a id="62">AdaBoostClassifier</a>
# 
# 
# AdaBoostClassifier stands for Adaptive Boosting Classifier <a href='#8'>[5]</a>.
# 
# ### Prepare the model
# 
# Let's set the parameters for the model and initialize the model.

# In[ ]:


clf = AdaBoostClassifier(random_state=RANDOM_STATE,
                         algorithm='SAMME.R',
                         learning_rate=0.8,
                             n_estimators=NUM_ESTIMATORS)


# ### Fit the model
# 
# Let's fit the model.

# In[ ]:


clf.fit(train_df[predictors], train_df[target].values)


# ### Predict the target values
# 
# Let's now predict the **target** values for the **valid_df** data, using predict function.

# In[ ]:


preds = clf.predict(valid_df[predictors])


# ### Features importance
# 
# Let's see also the features importance.

# In[ ]:


tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()   


# ### Confusion matrix
# 
# Let's visualize the confusion matrix.

# In[ ]:


cm = pd.crosstab(valid_df[target].values, preds, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cm, 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix', fontsize=14)
plt.show()


# Let's calculate also the ROC-AUC.
# 
# 
# ### Area under curve

# In[ ]:


roc_auc_score(valid_df[target].values, preds)


# The ROC-AUC score obtained with AdaBoostClassifier is **0.83**.

# ## <a id="63">CatBoostClassifier</a>
# 
# 
# CatBoostClassifier is a gradient boosting for decision trees algorithm with support for handling categorical data <a href='#8'>[6]</a>.
# 
# ### Prepare the model
# 
# Let's set the parameters for the model and initialize the model.

# In[ ]:


clf = CatBoostClassifier(iterations=500,
                             learning_rate=0.02,
                             depth=12,
                             eval_metric='AUC',
                             random_seed = RANDOM_STATE,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = VERBOSE_EVAL,
                             od_wait=100)


# In[ ]:


clf.fit(train_df[predictors], train_df[target].values,verbose=True)


# ### Predict the target values
# 
# Let's now predict the **target** values for the **val_df** data, using predict function.

# In[ ]:


preds = clf.predict(valid_df[predictors])


# ### Features importance
# 
# Let's see also the features importance.

# In[ ]:


tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()   


# ### Confusion matrix
# 
# Let's visualize the confusion matrix.

# In[ ]:


cm = pd.crosstab(valid_df[target].values, preds, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cm, 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix', fontsize=14)
plt.show()


# Let's calculate also the ROC-AUC.
# 
# 
# ### Area under curve

# In[ ]:


roc_auc_score(valid_df[target].values, preds)


# The ROC-AUC score obtained with CatBoostClassifier is **0.86**.

# ## <a id="63">XGBoost</a>

# XGBoost is a gradient boosting algorithm <a href='#8'>[7]</a>.
# 
# Let's prepare the model.

# ### Prepare the model
# 
# We initialize the DMatrix objects for training and validation, starting from the datasets. We also set some of the parameters used for the model tuning.

# In[ ]:


# Prepare the train and valid datasets
dtrain = xgb.DMatrix(train_df[predictors], train_df[target].values)
dvalid = xgb.DMatrix(valid_df[predictors], valid_df[target].values)
dtest = xgb.DMatrix(test_df[predictors], test_df[target].values)

#What to monitor (in this case, **train** and **valid**)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

# Set xgboost parameters
params = {}
params['objective'] = 'binary:logistic'
params['eta'] = 0.039
params['silent'] = True
params['max_depth'] = 2
params['subsample'] = 0.8
params['colsample_bytree'] = 0.9
params['eval_metric'] = 'auc'
params['random_state'] = RANDOM_STATE


# ### Train the model
# 
# Let's train the model. 

# In[ ]:


model = xgb.train(params, 
                dtrain, 
                MAX_ROUNDS, 
                watchlist, 
                early_stopping_rounds=EARLY_STOP, 
                maximize=True, 
                verbose_eval=VERBOSE_EVAL)


# The best validation score (ROC-AUC) was **0.984**, for round **241**.

# ### Plot variable importance

# In[ ]:


fig, (ax) = plt.subplots(ncols=1, figsize=(8,5))
xgb.plot_importance(model, height=0.8, title="Features importance (XGBoost)", ax=ax, color="green") 
plt.show()


# ### Predict test set
# 
# 
# We used the train and validation sets for training and validation. We will use the trained model now to predict the target value for the test set.

# In[ ]:


preds = model.predict(dtest)


# ### Area under curve
# 
# Let's calculate ROC-AUC.

# In[ ]:


roc_auc_score(test_df[target].values, preds)


# The AUC score for the prediction of fresh data (test set) is **0.974**.

# ## <a id="64">LightGBM</a>
# 
# 
# Let's continue with another gradient boosting algorithm, LightGBM <a href='#8'>[8]</a> <a href='#8'>[9]</a>.
# 
# 
# ### Define model parameters
# 
# Let's set the parameters for the model. We will use these parameters only for the first lgb model.

# In[ ]:


params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric':'auc',
          'learning_rate': 0.05,
          'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
          'max_depth': 4,  # -1 means no limit
          'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
          'max_bin': 100,  # Number of bucketed bin for feature values
          'subsample': 0.9,  # Subsample ratio of the training instance.
          'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
          'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
          'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
          'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
          'nthread': 8,
          'verbose': 0,
          'scale_pos_weight':150, # because training data is extremely unbalanced 
         }


# ### Prepare the model
# 
# Let's prepare the model, creating the **Dataset**s data structures from the train and validation sets.

# In[ ]:


dtrain = lgb.Dataset(train_df[predictors].values, 
                     label=train_df[target].values,
                     feature_name=predictors)

dvalid = lgb.Dataset(valid_df[predictors].values,
                     label=valid_df[target].values,
                     feature_name=predictors)


# ### Run the model
# 
# Let's run the model, using the **train** function.

# In[ ]:


evals_results = {}

model = lgb.train(params, 
                  dtrain, 
                  valid_sets=[dtrain, dvalid], 
                  valid_names=['train','valid'], 
                  evals_result=evals_results, 
                  num_boost_round=MAX_ROUNDS,
                  early_stopping_rounds=2*EARLY_STOP,
                  verbose_eval=VERBOSE_EVAL, 
                  feval=None)


# Best validation score  was obtained for round **85**, for which **AUC ~= 0.974**.
# 
# Let's plot variable importance.

# In[ ]:


fig, (ax) = plt.subplots(ncols=1, figsize=(8,5))
lgb.plot_importance(model, height=0.8, title="Features importance (LightGBM)", ax=ax,color="red") 
plt.show()


# Let's predict now the target for the test data.
# 
# ### Predict test data

# In[ ]:


preds = model.predict(test_df[predictors])


# ### Area under curve
# 
# Let's calculate the ROC-AUC score for the prediction.

# In[ ]:


roc_auc_score(test_df[target].values, preds)


# The ROC-AUC score obtained for the test set is **0.946**.

# ### Training and validation using cross-validation
# 
# Let's use now cross-validation. We will use cross-validation (KFolds) with 5 folds. Data is divided in 5 folds and, by rotation, we are training using 4 folds (n-1) and validate using the 5th (nth) fold.
# 
# Test set is calculated as an average of the predictions 

# In[ ]:


kf = KFold(n_splits = NUMBER_KFOLDS, random_state = RANDOM_STATE, shuffle = True)

# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
test_preds = np.zeros(test_df.shape[0])
feature_importance_df = pd.DataFrame()
n_fold = 0
for train_idx, valid_idx in kf.split(train_df):
    train_x, train_y = train_df[predictors].iloc[train_idx],train_df[target].iloc[train_idx]
    valid_x, valid_y = train_df[predictors].iloc[valid_idx],train_df[target].iloc[valid_idx]
    
    evals_results = {}
    model =  LGBMClassifier(
                  nthread=-1,
                  n_estimators=2000,
                  learning_rate=0.01,
                  num_leaves=80,
                  colsample_bytree=0.98,
                  subsample=0.78,
                  reg_alpha=0.04,
                  reg_lambda=0.073,
                  subsample_for_bin=50,
                  boosting_type='gbdt',
                  is_unbalance=False,
                  min_split_gain=0.025,
                  min_child_weight=40,
                  min_child_samples=510,
                  objective='binary',
                  metric='auc',
                  silent=-1,
                  verbose=-1,
                  feval=None)
    model.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                eval_metric= 'auc', verbose= VERBOSE_EVAL, early_stopping_rounds= EARLY_STOP)
    
    oof_preds[valid_idx] = model.predict_proba(valid_x, num_iteration=model.best_iteration_)[:, 1]
    test_preds += model.predict_proba(test_df[predictors], num_iteration=model.best_iteration_)[:, 1] / kf.n_splits
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = predictors
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
    del model, train_x, train_y, valid_x, valid_y
    gc.collect()
    n_fold = n_fold + 1
train_auc_score = roc_auc_score(train_df[target], oof_preds)
print('Full AUC score %.6f' % train_auc_score)                                    


# The AUC score for the prediction from the test data was 0.93.
# 
# We prepare the test prediction, from the averaged predictions for test over the 5 folds.

# In[ ]:


pred = test_preds


# # <a id="7">Conclusions</a>

# We investigated the data, checking for data unbalancing, visualizing the features and understanding the relationship between different features. 
# We then investigated two predictive models. The data was split in 3 parts, a train set, a validation set and a test set. For the first three models, we only used the train and test set.  
# 
# We started with **RandomForrestClassifier**, for which we obtained an AUC scode of **0.85** when predicting the target for the test set.  
# 
# We followed with an **AdaBoostClassifier** model, with lower AUC score (**0.83**) for prediction of the test set target values.    
# 
# We then followed with an **CatBoostClassifier**, with the AUC score after training 500 iterations **0.86**.    
# 
# We then experimented with a **XGBoost** model. In this case, se used the validation set for validation of the training model.  The best validation score obtained was   **0.984**. Then we used the model with the best training step, to predict target value from the test data; the AUC score obtained was **0.974**.
# 
# We then presented the data to a **LightGBM** model. We used both train-validation split and cross-validation to evaluate the model effectiveness to predict 'Class' value, i.e. detecting if a transaction was fraudulent. With the first method we obtained values of AUC for the validation set around **0.974**. For the test set, the score obtained was **0.946**.   
# With the cross-validation, we obtained an AUC score for the test prediction of  **0.93**.

# # <a id="8">References</a>
# 
# [1] Credit Card Fraud Detection Database, Anonymized credit card transactions labeled as fraudulent or genuine, https://www.kaggle.com/mlg-ulb/creditcardfraud  
# [2] Principal Component Analysis, Wikipedia Page, https://en.wikipedia.org/wiki/Principal_component_analysis  
# [3] RandomForrestClassifier, http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html  
# [4] ROC-AUC characteristic, https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve   
# [5] AdaBoostClassifier, http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html  
# [6] CatBoostClassifier, https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_catboostclassifier-docpage/  
# [7] XGBoost Python API Reference, http://xgboost.readthedocs.io/en/latest/python/python_api.html  
# [8] LightGBM Python implementation, https://github.com/Microsoft/LightGBM/tree/master/python-package  
# [9] LightGBM algorithm, https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/lightgbm.pdf   
# 
# 
