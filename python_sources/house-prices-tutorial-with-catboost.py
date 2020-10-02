#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# # House Prices Tutorial - With Catboost
# 
# This kernel was born during one of our KaggleDays meetup events in Cologne. The workshop was related to the task feature engineering even though many of us just used a brute force approach, this kernel has become much more detailed. During the last weeks (or months :-O) I tried out several ideas and consequently the content changed over time. ;-) Finally I found a stable structure to play with and the kernel has become my personal catboost tutorial that I like to share with the Kaggle community. 
# 
# 
# <img src="https://images.unsplash.com/photo-1551969014-7d2c4cddf0b6?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2778&q=80" width="900px">
# 
# 
# 
# If you like to work with it, just fork and build uppon this sceleton. Of course you can change the model, build a model zoo etc.. Feel free to choose your personal learning path. ;-)
# 
# And please... **don't forget to UPVOTE! ;-p**

# ### Table of contents
# 
# 1. [Prepare to start](#prepare) (complete)
#     * [Packages](#packages) (complete)
#     * [Helper methods](#helpers) (complete)
# 2. [Peek at the data](#peek) (complete)
#     * [Training data](#training) (complete)
#     * [Test data](#test) (complete)
#     * [Submission](#submission) (complete)
# 3. [Data exploration & cleaning](#eda) 
#     * [Log-transforming the target distribution](#targets) (complete)
#     * [Dropping nan-features](#nanfeatures) (complete)
#     * [Finding categorical features in numerical candidates](#num_to_cat_candidates) (complete)
#     * [Dropping useless categorical candidates](#useless_cat) (complete)
#     * [Fusing seldom categorical levels](#fusion)
# 4. [Data preparation](#dataprep)
#     * [Dealing with outliers](#outliers) (complete)
#     * [Imputing missing values](#impute) (somehow complete)
#     * [Generating obvious new features](#obvious)
# 5. [Baseline predictions with catboost](#baseline)
#     * [Trying to demystify the learning process](#demytify)
#     * [Generating hold-out-data](#holdout) (complete)
#     * [A colorful bouquet of hyperparameters](#hyparams)
#     * [Running catboost & feature importances](#run_catboost) (somehow complete)
# 6. [Feature selection](#feature_selection)
#     * [Dropping features with almost no importance](#fs_no_imp)
# 7. [Feature engineering](#engineering)
#     * [Feature interaction](#interaction)
# 8. [Some further insights](#furtherinsights)
# 9. [Outlook](#outlook)

# # Prepare to start <a class="anchor" id="prepare"></a>
# 
# ## Packages <a class="anchor" id="packages"></a>

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set()


from catboost import CatBoostRegressor, Pool, cv

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.feature_selection import RFECV

import eli5
from eli5.sklearn import PermutationImportance

import hyperopt

from numpy.random import RandomState
from os import listdir


import shap
# load JS visualization code to notebook
shap.initjs()


# In[ ]:


listdir("../input")


# ## Helper methods <a class="anchor" id="helpers"></a>

# In[ ]:


def run_catboost(traindf, testdf, holddf, params, n_splits=10, n_repeats=1,
                 plot=False, use_features=None, plot_importance=True):
    
    
    folds = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)
    p_hold = np.zeros(holddf.shape[0])
    y_hold = holddf.target
    p_test = np.zeros(testdf.shape[0])
    
    if use_features is None:
        use_features = testdf.columns.values
    
    cat_features = np.where(testdf.loc[:, use_features].dtypes=="object")[0]
    x_hold = holddf.loc[:, use_features]
    x_test = testdf.loc[:, use_features]
    
    feature_importance_df = pd.DataFrame(index=use_features)
    
    m = 0
    cv_scores = []
    for train_idx, dev_idx in folds.split(traindf):
        x_train, x_dev = traindf.iloc[train_idx][use_features], traindf.iloc[dev_idx][use_features]
        y_train, y_dev = traindf.target.iloc[train_idx], traindf.target.iloc[dev_idx]

        train_pool = Pool(x_train, y_train, cat_features=cat_features)
        dev_pool = Pool(x_dev, y_dev, cat_features=cat_features)
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=dev_pool, plot=plot)

        # bagging predictions for test and hold out data:
        p_hold += model.predict(x_hold)/(n_splits*n_repeats)
        log_p_test = model.predict(x_test)
        p_test += (np.exp(log_p_test) - 1)/(n_splits*n_repeats)

        # predict for dev fold:
        y_pred = model.predict(x_dev)
        feature_importance_df.loc[:, "fold_" + str(m)] = model.get_feature_importance(train_pool)
        cv_scores.append(np.sqrt(mse(y_dev, y_pred)))
        m+=1

    print("hold out rmse: " + str(np.sqrt(mse(y_hold, p_hold))))
    print("cv mean rmse: " + str(np.mean(cv_scores)))
    print("cv std rmse: " + str(np.std(cv_scores)))
    
    feature_importance_df["mean"] = feature_importance_df.mean(axis=1)
    feature_importance_df["std"] = feature_importance_df.std(axis=1)
    feature_importance_df = feature_importance_df.sort_values(by="mean", ascending=False)
    
    if plot_importance:
        plt.figure(figsize=(15,20))
        sns.barplot(x=feature_importance_df["mean"].values, y=feature_importance_df.index.values);
        plt.title("Feature importances");
        plt.show()
    
    results = {"last_model": model,
               "last_train_pool": train_pool,
               "feature_importance": feature_importance_df, 
               "p_hold": p_hold,
               "p_test": p_test,
               "cv_scores": cv_scores}
    return results


# # Peek at the data <a class="anchor" id="peek"></a>

# ## Training data <a class="anchor" id="training"></a>

# In[ ]:


train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv", index_col=0)
train.head()


# In[ ]:


train.shape


# ## Test data <a class="anchor" id="test"></a>

# In[ ]:


test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv", index_col=0)
test.head()


# In[ ]:


test.shape


# In[ ]:


train.shape[0] / test.shape[0]


# The training data has almost the same size as the test data!

# ## Submission <a class="anchor" id="submission"></a>

# In[ ]:


submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv", index_col=0)
submission.head()


# # Data exploration <a class="anchor" id="eda"></a>

# ## Log-transforming the target distribution <a class="anchor" id="targets"></a>

# In[ ]:


plt.figure(figsize=(20,5))
sns.distplot(train.SalePrice, color="tomato")
plt.title("Target distribution in train")
plt.ylabel("Density");


# The sale price distribution is right-skewed and shows extreme outliers. We should log-transform the target values, as we will use some loss built on mean-squared-error which assumes that our target distribution is normal.

# In[ ]:


train[train.SalePrice <=0 ]


# We don't have to worry about negative or zero house prices. :-)

# In[ ]:


plt.figure(figsize=(20,5))
sns.distplot(np.log(train.SalePrice), color="tomato")
plt.title("Log Target distribution in train")
plt.ylabel("Density");


# In[ ]:


train["LogSalePrice"] = train.SalePrice.apply(np.log)


# ## Dropping nan-features <a class="anchor" id="nanfeatures"></a>

# In[ ]:


combined = train.drop(["SalePrice", "LogSalePrice"], axis=1).append(test)
nan_percentage = combined.isnull().sum().sort_values(ascending=False) / combined.shape[0]
missing_val = nan_percentage[nan_percentage > 0]


# In[ ]:


plt.figure(figsize=(20,5))
sns.barplot(x=missing_val.index.values, y=missing_val.values * 100, palette="Reds_r");
plt.title("Percentage of missing values in train & test");
plt.ylabel("%");
plt.xticks(rotation=90);


# We can see that PoolQC, MiscFeature, Alley etc. have more than 80 % missing values. In the description, we can see that this often tells us "no pool", "no miscfeature" etc.. In my opinion it's difficult to say if such a feature is important. For this reason, let's not drop them and plugin "None" later in the analysis. For numerical features we need to find an approriate strategy.

# ## Finding categorical features in numerical candidates <a class="anchor" id="num_to_cat_candidates"></a>

# In[ ]:


num_candidates = list(combined.dtypes[combined.dtypes!="object"].index.values)
num_candidates


# Can we be sure that each numerical candidate is indeed a numerical feature? A feature that measures the area is a numerical feature but what about MSSubClass? It holds different groups as is not numerical but rather categorical. A first attempt could be to assume that all numerical features should have various different values and not only a few.

# In[ ]:


unique_counts = combined.loc[:, num_candidates].nunique().sort_values()

plt.figure(figsize=(20,5))
sns.barplot(unique_counts.index, unique_counts.values, palette="Oranges_r")
plt.xticks(rotation=90);
plt.yscale("log")


# With this rule in our mind, we find:
# 
# * Features that count the number of rooms. This seems to be more categorical as you can't have 3.245 bedrooms for example.
# * The same holds for the number of fireplaces and garage cars.
# * Features like OverallCond and OverallQual can be considered as groups and are categorical variables with a natural kind of order (ordinal features). 
# * Every feature with SF or area can be considered as numerical. 
# * I think that we should treat temporal features like YrSold as categorical.

# But what about MisVal and 3SsnPorch? Reading in the description, we can see that the latter is related to area. The further stands for the value in $ of a miscellaneous feature like a tennis court. Consequently both are numerical.

# In[ ]:


cat_candidates = list(combined.dtypes[combined.dtypes=="object"].index.values)


# In[ ]:


num_to_cats = ["BsmtHalfBath", "HalfBath", "KitchenAbvGr", "BsmtFullBath", "Fireplaces", "FullBath", "GarageCars",
               "BedroomAbvGr", "OverallCond", "OverallQual", "TotRmsAbvGrd", "MSSubClass", "YrSold", "MoSold", 
               "GarageYrBlt", "YearRemodAdd"]

for feat in num_to_cats:
    num_candidates.remove(feat)
    cat_candidates.append(feat)
    combined[feat] = combined[feat].astype("object")
    train[feat] = train[feat].astype("object")
    test[feat] = test[feat].astype("object")


# In[ ]:


num_candidates


# In[ ]:


len(num_candidates)


# ## Useless categorical candidates <a class="anchor" id="useless_cat"></a>

# Let's first look for numerical features in our categorical candidates:

# In[ ]:


cat_candidates = combined.dtypes[combined.dtypes=="object"].index.values
cat_candidates


# Comparing with the description this looks fine! It seems that there are no numerical features in categorical candidates. Some categorical features might still be completely useless as only some levels occur most of the time (no diversity). To get more insights, we can compute the frequency of the most common level in the train & test data:

# In[ ]:


frequencies = []
for col in cat_candidates:
    overall_freq = combined.loc[:, col].value_counts().max() / combined.shape[0]
    frequencies.append([col, overall_freq])

frequencies = np.array(frequencies)
freq_df = pd.DataFrame(index=frequencies[:,0], data=frequencies[:,1], columns=["frequency"])
sorted_freq = freq_df.frequency.sort_values(ascending=False)

plt.figure(figsize=(20,5))
sns.barplot(x=sorted_freq.index[0:30], y=sorted_freq[0:30].astype(np.float), palette="Blues_r")
plt.xticks(rotation=90);


# ### Insights
# 
# * Many categorical candidates have one major level that occupies > 80 % of the data. That's bad. What should we learn from such a feature? Let's pick some examples:

# In[ ]:


example = "Utilities"
combined.loc[:,example].value_counts()


# This is a completely useless feature!  

# In[ ]:


example = "Street"
combined.loc[:,example].value_counts()


# In[ ]:


example = "Condition2"
combined.loc[:,example].value_counts()


# Ahh! A second problem arises! ;-) Do you see the seldom levels of the Condition2 feature?! They only have one sample. We will find a lot of levels with very low frequencies or some that are only present in train or test. We have to deal with this problem later. For now, let's drop only "Utilities".

# In[ ]:


cats_to_drop = ["Utilities"]


# In[ ]:


combined = combined.drop(cats_to_drop, axis=1)
train = train.drop(cats_to_drop, axis=1)
test = test.drop(cats_to_drop, axis=1)


# In[ ]:


cat_candidates = combined.dtypes[combined.dtypes=="object"].index.values
cat_candidates


# ## Fusing seldom categorical levels <a class="anchor" id="fusion"></a>
# 
# ### Levels present in test but not in train

# In[ ]:


def build_map(useless_levels, plugin_level, train_levels):
    plugin_map = {}
    for level in useless_levels:
        plugin_map[level] = plugin_level
    for level in train_levels:
        plugin_map[level] = level
    return plugin_map


# In[ ]:


def clean_test_levels(train, test):
    for col in test.columns:
        train_levels = set(train[col].unique())
        test_levels = set(test[col].unique())
        in_test_not_in_train = test_levels.difference(train_levels)
        if len(in_test_not_in_train)>0:
            close_to_mean_level = train.groupby(col).LogSalePrice.mean() - train.SalePrice.apply(np.log).mean()
            close_to_mean_level = close_to_mean_level.apply(np.abs)
            plugin_level = close_to_mean_level.sort_values().index.values[0]
            in_test_not_in_train = list(in_test_not_in_train)
            plugin_map = build_map(in_test_not_in_train, plugin_level, train_levels)
            test[col] = test[col].map(plugin_map)
    return train, test


# In[ ]:


#train, test = clean_test_levels(train, test)


# In[ ]:


test["MSSubClass"].value_counts()


# # Data preparation <a class="anchor" id="dataprep"></a>

# ## Dealing with outliers <a class="anchor" id="outliers"></a>

# In[ ]:


fig, ax = plt.subplots(len(num_candidates),3,figsize=(20,len(num_candidates)*6))

for n in range(len(num_candidates)):
    feat = num_candidates[n]
    ax[n,0].scatter(train[feat].values, np.log(train.SalePrice.values), s=4)
    ax[n,0].set_ylabel("Log SalePrice")
    ax[n,0].set_xlabel(feat);
    ax[n,1].scatter(np.log(train[feat].values+1), np.log(train.SalePrice.values), s=4)
    ax[n,1].set_ylabel("Log SalePrice")
    ax[n,1].set_xlabel("Log" + feat);
    sns.distplot(test[feat].dropna(), kde=True, ax=ax[n,2], color="limegreen")
    ax[n,2].set_title("Distribution in test")


# ### Insights
# 
# * Even with log transformed features there are some really strange outliers that are often related to area features. 
# 
# Let's try to clean up a bit! :-)

# In[ ]:


outlier_ids = set()
outlier_ids = outlier_ids.union(set(train[train.LotArea > 60000].index.values))
outlier_ids = outlier_ids.union(set(train[train.LotFrontage > 200].index.values))
outlier_ids = outlier_ids.union(set(train[(train.LotFrontage > 150) & (train.SalePrice.apply(np.log) < 11)].index.values))
outlier_ids = outlier_ids.union(set(train[train.GrLivArea > 4500].index.values))
outlier_ids = outlier_ids.union(set(train[train["1stFlrSF"] > 4000].index.values))
outlier_ids = outlier_ids.union(set(train[train.MasVnrArea > 1400].index.values))
outlier_ids = outlier_ids.union(set(train[train["BsmtFinSF1"] > 5000].index.values))
outlier_ids = outlier_ids.union(set(train[train.TotalBsmtSF > 6000].index.values))
outlier_ids = outlier_ids.union(set(train[(train.OpenPorchSF > 500) & (np.log(train.SalePrice) < 11)].index.values))


# In[ ]:


outlier_ids


# In[ ]:


train.shape


# In[ ]:


train = train.drop(list(outlier_ids))
combined = combined.drop(list(outlier_ids))


# In[ ]:


train.shape


# ## Imputing missing values <a class="anchor" id="impute"></a>

# In[ ]:


def impute_na_trees(df, col):
    if df[col].dtype == "object":
        df[col] = df[col].fillna("None")
        df[col] = df[col].astype("object")
    else:
        df[col] = df[col].fillna(0)
    return df


# In[ ]:


for col in combined.columns:
    combined = impute_na_trees(combined, col)


# In[ ]:


num_candidates = combined.dtypes[combined.dtypes!="object"].index.values
len(num_candidates)


# In[ ]:


cat_candidates = combined.dtypes[combined.dtypes=="object"].index.values
len(cat_candidates)


# In[ ]:


combined.isnull().sum().sum()


# ## Generating obvious new features <a class="anchor" id="obvious"></a>

# In[ ]:


combined["TotalSF"] = combined["1stFlrSF"] + combined["2ndFlrSF"] + combined["TotalBsmtSF"] 
combined["GreenArea"] = combined["LotArea"] - combined["GrLivArea"] - combined["GarageArea"]


# # Baseline predictions with catboost <a class="anchor" id="baseline"></a>
# 
# This data has many categorical features and one model that can deal nicely with these kind of features is [catboost](https://catboost.ai/). If you like to read more about it, you can find a [paper here](https://arxiv.org/abs/1706.09516).
# 
# ## Trying to demystify the learning process <a class="anchor" id="demytify"></a>
# 
# **Under construction**
# 
# * binary decision trees as base predictors

# ## Generating hold-out-data <a class="anchor" id="holdout"></a>
# 
# I like to use the hold out data later for ensembling. 

# In[ ]:


traindf = combined.iloc[0:train.shape[0]].copy()
traindf.loc[:, "target"] = train.LogSalePrice
testdf = combined.iloc[train.shape[0]::].copy()


# In[ ]:


traindf, holddf = train_test_split(traindf, test_size=0.25, random_state=0)
print((traindf.shape, holddf.shape, testdf.shape))


# ## A colorful bouquet of hyperparameters <a class="anchor" id="hyparams"></a>

# In[ ]:


org_params = {
    'iterations': 10000,
    'learning_rate': 0.08,
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'logging_level': 'Silent',
    'use_best_model': True,
    'loss_function': 'RMSE',
    'od_type': 'Iter',
    'od_wait': 1000,
    'one_hot_max_size': 20,
    'l2_leaf_reg': 100,
    'depth': 3,
    'rsm': 0.6,
    'random_strength': 2,
    'bagging_temperature': 10
}


# ## Running catboost and feature importances <a class="anchor" id="run_catboost"></a>

# In[ ]:


results = run_catboost(traindf, 
                       testdf,
                       holddf,
                       org_params,
                       plot=True,
                       n_splits=5,
                       n_repeats=3)
p_hold = results["p_hold"]
p_test = results["p_test"]
feature_importance_df = results["feature_importance"]


# In[ ]:


feature_importance_df.head()


# In[ ]:


feature_importance_df.tail()


# ### Insights
# 
# * There are features that were never used by catboost.
# * There is a lot of variation of the feature importance between folds! What does this mean?
#     * How useful a feature is to fit the training data depends on the data itself and how big the dataset is.
#     * If you increase the number of splits the most important features will change! Try it out!
#     * Remember that our test dataset is the same size as the training dataset. It's very likely that we will overfit to our training data and that we will not be able to generalize well on the test data. 
# * What can we do to reduce this problem?
#     * We should drop features that almost have no importance!
#     * We should include more randomess to our KFold as well (run it several times, make averaged predictions with each KFold)
#     * We should use different models that contribute to our prediction.
# 
# Let's figure it out by playing with these concepts. ;-)
#     

# In[ ]:


submission.loc[testdf.index.values, "SalePrice"] = p_test
submission = submission.reset_index()
submission.head()


# In[ ]:


submission.to_csv("submission_before_feature_selection.csv", index=False)
submission = submission.set_index("Id")
submission.head()


# Using this submission we get a score around 0.141.

# In[ ]:


plt.figure(figsize=(5,5))
plt.scatter(holddf.target, p_hold, s=10, color="deepskyblue")
plt.xlabel("Target value")
plt.ylabel("Predicted value");


# In[ ]:


hold_predictions = pd.DataFrame(holddf.target.values, index=holddf.index, columns=["target"])
hold_predictions["catboost_org_features"] = p_hold
hold_predictions.head()


# # Feature selection <a class="anchor" id="feature_selection"></a>
# 
# ## Dropping features with almost no importance <a class="anchor" id="fs_no_imp"></a>

# In[ ]:


plt.figure(figsize=(20,5))
sns.barplot(feature_importance_df["mean"].index, feature_importance_df["mean"].values, palette="Reds_r")
plt.xticks(rotation=90);


# In[ ]:


to_drop = ["Condition2", "RoofMatl", "PoolArea", "3SsnPorch", "LandSlope", "LowQualFinSF",
           "Electrical", "MiscFeature"]


# In[ ]:


combined = combined.drop(to_drop, axis=1)
traindf = traindf.drop(to_drop, axis=1)
testdf = testdf.drop(to_drop, axis=1)
holddf = holddf.drop(to_drop, axis=1)


# # Feature engineering <a class="anchor" id="engineering"></a>

# ## Feature Interaction <a class="anchor" id="interaction"></a>
# 
# With catboost we can also discover feature interactions with a depth > 1. This way we can see which features are often found as a combination in single trees:

# In[ ]:


cat_features = np.where(testdf.dtypes == "object")[0]


# In[ ]:


x_train, x_dev = traindf.drop("target", axis=1), holddf.drop("target", axis=1)
y_train, y_dev = traindf.target, holddf.target

train_pool = Pool(x_train, y_train, cat_features=cat_features)
dev_pool = Pool(x_dev, y_dev, cat_features=cat_features)
model = CatBoostRegressor(**org_params)
model.fit(train_pool, eval_set=dev_pool, plot=True)


# I really love the next feature: ;-)

# In[ ]:


interaction = model.get_feature_importance(train_pool, type="Interaction")
column_names = testdf.columns.values 
interaction = pd.DataFrame(interaction, columns=["feature1", "feature2", "importance"])
interaction.feature1 = interaction.feature1.apply(lambda l: column_names[int(l)])
interaction.feature2 = interaction.feature2.apply(lambda l: column_names[int(l)])
interaction.head(20)


# ## Generating new features based on interactions

# In[ ]:


interaction["feature1_type"] = interaction.feature1.apply(
    lambda l: np.where(testdf[l].dtype=="object", 0, 1)
)
interaction["feature2_type"] = interaction.feature2.apply(
    lambda l: np.where(testdf[l].dtype=="object", 0, 1)
)
interaction.head()


# In[ ]:


interaction["combination"] = interaction.feature1_type + interaction.feature2_type
interaction.combination.value_counts()


# In[ ]:


numerical_combi = interaction[interaction.combination==2].copy()
for n in numerical_combi.index.values:
    feat1 = numerical_combi.loc[n].feature1
    feat2 = numerical_combi.loc[n].feature2
    if traindf[feat1].max() > traindf[feat2].max():
        traindf.loc[:, feat2 + "_" + feat1 + "_frac"] = traindf[feat2] / traindf[feat1]
    else:
        traindf.loc[:, feat1 + "_" + feat2 + "_frac"] = traindf[feat1] / traindf[feat2]
    traindf.loc[:, feat2 + "_" + feat1 + "_mult"] = traindf[feat2] * traindf[feat1]
    traindf.loc[:, feat2 + "_" + feat1 + "_add"] = traindf[feat2] + traindf[feat1]
    traindf.loc[:, feat2 + "_" + feat1 + "_sub"] = traindf[feat2] - traindf[feat1]


# In[ ]:


mixed_combi = interaction[interaction.combination==1].copy()
for n in mixed_combi.index.values:
    feat1 = mixed_combi.loc[n].feature1
    feat2 = mixed_combi.loc[n].feature2
    if traindf[feat1].dtype=="object":
        traindf.loc[:, "grouped_" + feat1 + "_mean_" + feat2] = traindf[feat1].map(
            traindf.groupby(feat1)[feat2].mean())
        traindf.loc[:, "grouped_" + feat1 + "_std_" + feat2] = traindf[feat1].map(
            traindf.groupby(feat1)[feat2].std())


# In[ ]:


traindf.head()


# ## Happy kaggling! ;-)
