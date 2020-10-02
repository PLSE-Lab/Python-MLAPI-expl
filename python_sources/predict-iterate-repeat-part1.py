#!/usr/bin/env python
# coding: utf-8

# ### GOAL
# 
# The goal of this notebook is to work ouor way through getting better predictive results by iterating. We will apply a ML and data-driven feature engineering approach since all of the feature don't say much about themselves as they are.
# 
# Tuning our algorithm and ensembling or stacking different ones will be the last thing we will do.
# 
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import xgboost as xgb

from IPython.display import display
from collections import defaultdict


# In[ ]:


dtrain = pd.read_csv('../input/train-and-test-csv/train.csv')
dtest = pd.read_csv('../input/train-and-test-csv/test.csv')


# In[ ]:


def display_all(data):
    with pd.option_context('display.max_rows', 1000):
        with pd.option_context('display.max_columns', 1000):
            return display(data)


# ### What do we now about data to start with ?
# 
# *"In the train and test data, features that belong to similar groupings are tagged as such in the feature names **(e.g., ind, reg, car, calc)**. **In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features.** Features without these designations are either continuous or ordinal. Values of -1 indicate that the feature was missing from the observation. The target columns signifies whether or not a claim was filed for that policy holder."*

# ## Write something about target...
# 
# We see that there is almost 600K training and 900K test data observations, exactly 1.5x. As a good practice we mostly want to have a  balanced sample size to build models on and assess since they might generalize better with a similar sample size. 
# 
# Since we will be iterating quickly we want to work with a sample of data while applying our data driven feature engineering approach. For this purpose a similar ratio as train and test for subsampling can be taken.
# 
# **Huge Problem Imbalance:** As seen below our dependent variable highly imbalanced ~3.7%. However this is very likely to happen in insurance claim context. Better risk allocating insurance companies generally have this type of ratio whereas a number up to ~10% can be seen through different companies in the sector. This been said having this low ratio is making the problem even harder!

# In[ ]:


print(dtrain.shape)
print(dtest.shape)


# In[ ]:


#Target
print(dtrain.target.value_counts(normalize=True))


# ### Get a feel of the data
# 
# As we look at both training and test samples we see that most of the data is already processed and most of the columns are categorical, but still there are some numerical ones too. 
# 
# First we will make some bad assumptions since we know NOTHING yet about the data and iterate our way through talking with the models and try to come up with better predictors while testing them.

# In[ ]:


display_all(dtrain.head())


# In[ ]:


display_all(dtest.head())


# In[ ]:


dtrain.replace(to_replace=-1, value=np.nan, inplace=True)
dtest.replace(to_replace=-1, value=np.nan, inplace=True)


# In[ ]:


pred_columns = dtrain.columns[2:].values


# ### Let's output some summaries..
# 
# This will be helpful to see each variables in both train and test. That way we can look at to summaries to come up with new ideas. For example what to do NAs?

# In[ ]:


### Let's do the column dtype conversions
pred_columns = dtrain.columns[2:]
bin_cols = [c for c in pred_columns if 'bin' in c]
cat_cols = [c for c in pred_columns if 'cat' in c]
num_cols= [c for c in pred_columns if c not in bin_cols and c not in cat_cols]


# ### Show summaries for categorical and numerical features
# 
# 
# In categorical case, data in general show consistency between training and test. Which is good for us since data seems to be coming from the same generator.
# 
# We can also see that cardinalities for both train test data are the same, so we don't need to worry about missing category levels.
# 
# We can also see that similar group variables are have strong correlated features such as 
# 
# - Corr between ps_reg_02 and ps_reg_03: 0.7427425174286711
# - Corr between ps_car_12 and ps_car_13: 0.6705204199424284
# - ...

# In[ ]:


for c in pred_columns:
    if 'cat' in c or 'bin' in c:
        print(f'Column: {c.upper()}')
        print('Train Summary')
        print(f'Cardinality {len(dtrain[c].unique())}')
        print(dtrain[c].value_counts(dropna=False))
        print('Test Summary')
        print(f'Cardinality {len(dtest[c].unique())}')
        print(dtest[c].value_counts(dropna=False))
        print()
        print()


# In[ ]:


# For fast iteration we will be using a subsample of data
def subsample(data, ratio=0.5):
    subsample, _ = train_test_split(data, test_size =ratio, stratify=dtrain.target)
    return subsample


# In[ ]:


subsample = subsample(dtrain)


# In[ ]:


# If feature is not binary include in correlation matrix
corr_cols = [c for c in num_cols if len(dtrain[c].unique()) > 2]
plt.imshow((subsample[corr_cols].corr()), cmap='hot', interpolation='nearest')
plt.show()


# In[ ]:


corr_mat = np.array(subsample[corr_cols].corr())

i_ix = np.where((corr_mat > 0.4) | (corr_mat < -0.4))[0]
j_ix = np.where((corr_mat > 0.4) | (corr_mat < -0.4))[1]
for i, j in zip(i_ix, j_ix):
    if i != j:
        print(f'Corr between {corr_cols[i]} and {corr_cols[j]}: {corr_mat[i, j]}')


# ### ITERATIONS
# 
# In this part we will engineer features and run randomforest on a validation set and test it on our competition metric and try to if it adds some value to our model.
# 
# Random forest are fast and have good interpretations.

# ### GINI SCORE
# 
# Define competition metric

# In[ ]:


def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


# ### CV FOLD ON SUBSAMPLE
# 
# Having a good cv fold and using it for validation is important in order to see whether are engineered features or new methods pay off **(random state)**
# 
# Since we don't care about tuning parameters and since this is just the beginning we will use default xgboost. Also we want it to be shallow to get fast results.
# 
# This is our motivation:
# 
# [thanks to @Winks here]("https://datascience.stackexchange.com/questions/10640/how-to-perform-feature-engineering-on-unknown-features")
# 
# [also this video]("https://www.youtube.com/watch?v=bL4b1sGnILU&t=891s")
# 
# "First, run your boosting algorithm using only stumps, 1-level decision trees. Stumps are very weak, but Boosting makes it a reasonnable model. This will act as your baseline. Depending on the library you are using, you should be ale to display pretty easily which are the most used features, and you should plot them against the response (or do an histogram if the response is categorical) to identify some pattern. This might give you an intuition on what would be a good single feature transformation.
# 
# Next, run the Boosting algorithm with 2-level decision trees. This model is a lot more complex than the previous one; if two variables taken together have more power than taken individually, this model should outperform your previous one (again, not in term of training error, but on validation error!). Based on this, you should be able to extract which variable are often used together, and this should lead you to potential multi-feature transformations."
# 
# For each iterations we will be using the feedback function and score summary below, as well as other outputs from xgboost.

# In[ ]:


shuffleSplit = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state= 10)


# In[ ]:


subsample.reset_index(drop=True, inplace=True)


# In[ ]:


def xgb_feedback(params, nrounds, prc_subsample):
    val_scores = []
    for train_ix, val_ix in shuffleSplit.split(prc_subsample, prc_subsample.target):
        X_train, y_train = prc_subsample.loc[train_ix].drop(['id', 'target'], axis=1), prc_subsample.target.loc[train_ix]
        X_val, y_val = prc_subsample.loc[val_ix].drop(['id', 'target'], axis=1), prc_subsample.target.loc[val_ix]

        #create dmatrix
        dtrain = xgb.DMatrix(data=X_train, label=y_train, missing= np.nan)
        dval = xgb.DMatrix(data=X_val, label=y_val, missing= np.nan)

        #train
        model = xgb.train(params, dtrain, num_boost_round=nrounds)
        preds = model.predict(dval)
        score = gini_normalized(y_val, preds)
        val_scores.append(score)
    return val_scores, model


# In[ ]:


def score_summary(scores):return f'Mean: {np.mean(scores)} Std: {np.std(scores)}'


# In[ ]:


iter_performances = defaultdict()
def add_to_iter(scores, name):
    iter_performances[name]= scores


# In[ ]:


def plot_model(model):
    matplotlib.rcParams['figure.figsize'] = [10, 7]
    ax = xgb.plot_importance(model)


# ### 1) BASELINE - NO FEATURE ENGINEERING
# 
# Here we use data as it is. Including missing values as -1. Remember we can always call a new subsample but this might shadow our benchmarking and it wouldn't be unbiased so I will use the same subsample through this notebook but my processed subsamples will have the name prc_subsample.

# In[ ]:


#first process is just to put those NA -1s back
def prc1(data):
    data = data.fillna(-1)
    return data


# In[ ]:


prc_subsample = prc1(subsample)

params = {"objective":"binary:logistic", "max_depth":1}
val_scores, model = xgb_feedback(nrounds=100, params=params, prc_subsample=prc_subsample)

init_benchmark = score_summary(val_scores)

# So this will be our initial benchmark
print(init_benchmark)

add_to_iter(init_benchmark, "benchmark")


# **The Beauty of this approach that it takes seconds to run a model and see the feature importances. ps_car_13 is stated to be a very effective feature by most of the competitors out there and here it is fast and accurate. [see it here]("https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41489")**
# 
# So let's take a look at these top features...

# In[ ]:


matplotlib.rcParams['figure.figsize'] = [10, 7]
ax = xgb.plot_importance(model)


# ### 2) ITER 1: PS_CAR_13

# This distribution skewed to the right let's try log transform on this. Note for transformation on columns we will use {transform_name}_features and for other features we will use {fi}_features

# In[ ]:


subsample.ps_car_13.hist(bins=30)


# In[ ]:


np.log(subsample.ps_car_13).hist(bins=30)


# In[ ]:


round(subsample.ps_car_13**2 * 48400).hist(bins =100)


# In[ ]:


def prc2(data):
    data = data.copy()
    #log transform ps_car_13
    data['log_ps_car_13'] = np.log(data["ps_car_13"])
    #create ps_car_13 feature from the kernel link
    #thanks to @raddar
    data['f1_ps_car_13'] = round(subsample["ps_car_13"]**2 * 48400)
    #also log this
    data['log_f1_ps_car_13'] = np.log(round(subsample["ps_car_13"]**2 * 48400))
    return data


# In[ ]:


prc_subsample = prc2(subsample)

params = {"objective":"binary:logistic", "max_depth":1}
val_scores, model = xgb_feedback(nrounds=100, params=params, prc_subsample=prc_subsample)

iter1_benchmark = score_summary(val_scores)

# So this will be our initial benchmark
print(iter1_benchmark)

add_to_iter(iter1_benchmark, "iter1")


# We see a little bit of improvement but not much, but std of val score is still the same which is a good indicator. Our val score on average improved by 0.25%. But this improvement maybe due to np.nan so let's try again with also NAs as -1. In fact improvement was due to NA introduction. So NAs must be important....Let's create a column with number of NAs

# In[ ]:


prc_subsample = prc2(prc1(subsample))

params = {"objective":"binary:logistic", "max_depth":1}
val_scores, model = xgb_feedback(nrounds=100, params=params, prc_subsample=prc_subsample)

iter1_nas_benchmark = score_summary(val_scores)

# So this will be our initial benchmark
print(iter1_nas_benchmark)

add_to_iter(iter1_nas_benchmark, "iter1_na_as_neg1")


# In[ ]:


iter_performances


# ### 3) ITER 2 COLUMN: NUMBER OF NAs

# In[ ]:


def prc3(data):
    data= data.copy()
    data["number_of_nan"] = data.isnull().sum(axis=1)
    return data


# In[ ]:


prc_subsample = prc3(subsample)

params = {"objective":"binary:logistic", "max_depth":1}
val_scores, model = xgb_feedback(nrounds=100, params=params, prc_subsample=prc_subsample)

iter2 = score_summary(val_scores)

# So this will be our initial benchmark
print(iter2)

add_to_iter(iter2, "iter2")


# In[ ]:


iter_performances


# In[ ]:


plot_model(model)


# ### 4) ITER 3 : PS_CAR_15 + PS_REG_03

# In[ ]:


# thanks to Pascal Nagel's kernel
def recon(reg):
    if np.isnan(reg):
        return reg
    else:
        integer = int(np.round((40*reg)**2)) # gives 2060 for our example
        for f in range(28):
            if (integer - f) % 27 == 0:
                F = f
        M = (integer - F)//27
        return F, M
# Using the above example to test
ps_reg_03_example = 1.1321312468057179
print("Federative Unit (F): ", recon(ps_reg_03_example)[0])
print("Municipality (M): ", recon(ps_reg_03_example)[1])


# In[ ]:


def prc4(data):
    data = data.copy()
    data["f1_ps_car_15"] = 1 / np.exp(data["ps_car_15"])
    data["f2_ps_car_15"] = (data["ps_car_15"])**2 
    data['ps_reg_F'] = data['ps_reg_03'].apply(lambda x: recon(x) if np.isnan(x) else recon(x)[0])
    data['ps_reg_M'] = data['ps_reg_03'].apply(lambda x: recon(x) if np.isnan(x) else recon(x)[1])
    return data


# In[ ]:


prc_subsample = prc4(subsample)

params = {"objective":"binary:logistic", "max_depth":1}
val_scores, model = xgb_feedback(nrounds=100, params=params, prc_subsample=prc_subsample)

iter3 = score_summary(val_scores)

# So this will be our initial benchmark
print(iter3)

add_to_iter(iter3, "iter3")


# In[ ]:


iter_performances


# In[ ]:


plot_model(model)


# ### 5) ITER  4 - ["ps_ind_06_bin","ps_ind_07_bin", "ps_ind_08_bin", "ps_ind_09_bin"] OHE
# 
# We can see that there is only 1 observation of 1 per row...
# 
# We see that these single preprocesses and feature engineering have relative lifts on validation sore

# In[ ]:


sum(subsample[["ps_ind_06_bin","ps_ind_07_bin", "ps_ind_08_bin", "ps_ind_09_bin"]].sum(axis =1) > 2)


# In[ ]:


def prc5(data):
    data = data.copy()
    arr = np.array(data[["ps_ind_06_bin","ps_ind_07_bin", "ps_ind_08_bin", "ps_ind_09_bin"]])
    data["ps_ind_bin_6789"] = arr.dot(np.array([6, 7, 8, 9]))
    return data


# In[ ]:


prc_subsample = prc5(subsample)

params = {"objective":"binary:logistic", "max_depth":1}
val_scores, model = xgb_feedback(nrounds=100, params=params, prc_subsample=prc_subsample)

iter4 = score_summary(val_scores)

# So this will be our initial benchmark
print(iter4)

add_to_iter(iter4, "iter4")


# In[ ]:


iter_performances


# In[ ]:


plot_model(model)


# ### MID-WAY PIPE
# 
# Let's pipe all of our processes so far except for NAs being = -1.

# In[ ]:


prc_subsample = prc5(prc4(prc3(prc2(subsample))))


# In[ ]:


params = {"objective":"binary:logistic", "max_depth":1}
val_scores, model = xgb_feedback(nrounds=100, params=params, prc_subsample=prc_subsample)

mid_score = score_summary(val_scores)

# So this will be our initial benchmark
print(mid_score)

add_to_iter(mid_score, "mid_way_score")


# In[ ]:


iter_performances


# In[ ]:


plot_model(model)


# In[ ]:


def midway_prc(data):return prc5(prc4(prc3(prc2(data))))


# ### CONTINUE HERE
# 
# - Do different types of encoding on important features
# - Check interactions by 2-split subsample trees
# - Check Ordinality
# - Check OHE effect

# ### SUBMISSION 
# 
# Let's make a submission to see how well we are doing so far with all the help of others...
# 
# We will not tune our parameters superbly since there is way much more things to do...
# 
# So far we explored stuff on subsample but it's not coincidence that we get better results on whole data since more data means better results. No proper tuning with a fast and dirty approach gives ~0.69 in LB
data = midway_prc(dtrain)params = {"objective":"binary:logistic", "max_depth":m, "eta":0.1}
val_scores, model = xgb_feedback(nrounds=95, params=params, prc_subsample=data)
print(f'param {m}, avg val {np.mean(val_scores)}')data = midway_prc(dtrain)
test = midway_prc(dtest)params = {"objective":"binary:logistic", "max_depth":3, "eta":0.1, "num_rounds":95}def xgb_submission(params, train, test):
    X_train, y_train = train.drop(['id', 'target'], axis=1), train.target
    X_test = test.drop(['id'], axis=1)

    #create dmatrix
    dtrain = xgb.DMatrix(data=X_train, label=y_train, missing= np.nan)
    dtest = xgb.DMatrix(data=X_test, missing= np.nan)

    #train
    model = xgb.train(params, dtrain, num_boost_round=95)
    preds = model.predict(dtest)
    return model, predsparamsmodel, preds = xgb_submission(params, data, test)plot_model(model)