#!/usr/bin/env python
# coding: utf-8

# # Error in data fixing and initial preprocessing
# 
# In this notebook I will focus solely on features and understanding them.
# It is not strictly a beginner notebook, though you should be able to follow through. __Please at ;east read dataset description before checking this kernel!__
# 
# If you are looking for strictly beginners notebooks, check [this](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python), [this](https://www.kaggle.com/bsivavenu/house-price-calculation-methods-for-beginners) or [this](https://www.kaggle.com/apapiu/regularized-linear-models).
# 
# __Upvote if you found this notebook useful and want to pay kudos for this work, thanks!__
# 
# Let's start by defining __what I will not do in this notebook__:
# 
# - Finding importance of each feature in the context of the main task (predicting house price)
# - Transformations of targets and creative feature engineering
# - Finding outliers
# - Preprocessing of features to improve final score (though employing those steps __should be beneficial to the task__) like normalization or standardization
# 
# Things listed above can be found in other great kernels (like the very popular [COMPREHENSIVE DATA EXPLORATION WITH PYTHON](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)) and I wouldn't like to repeat points made over there.
# 
# __Instead this kernel tries to focus on data from another perspective:__
# - Proper encoding of different features
# - Errors in data and contradictory deata description (some where missed by other publicly available kernels)
# - Explain reasoning and intuition standing behind each decision
# - Readable and easy to follow code following newest Python guidelines (many Kaggle solutions do not care about code quality unfortuantely)
# 
# Okay, I hope you will have some fun and you will learn something as we go on this journey!
# 
# Let's start by setting up some good old imports of libraries we'll need further down the line (you probably know most of them, don't you?):

# In[ ]:


from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import missingno as msno

sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

training_dataset_path = Path("../input/train.csv")
testing_dataset_path= Path("../input/test.csv")

train = pd.read_csv(training_dataset_path)
test = pd.read_csv(testing_dataset_path)
print(f"Training dataset shape: {train.shape}")
print(f"Testing dataset shape: {test.shape}")


# # 1. Divide dataset into nominal, ordinal and numerical features
# 
# We can move on to dividing our dataset based on features
# category, high level split would consist of:
# 
# - __numerical features__ (containing numbers)
# - __categorical__ (other like objects of type `string`)
# 
# Last four lines of code display names of features belonging to one of those categories:

# In[ ]:


categorical_features = train.select_dtypes(include=[np.object]).columns.values
numerical_features = train.select_dtypes(include=[np.number]).columns.values

feature_names = {"Categorical": categorical_features, "Numerical": numerical_features}
# For readability sake, I transpose dataframe so categories are columns and features are rows
feature_types = pd.DataFrame.from_dict(feature_names, orient="index").T

# Fillna so last rows do not contain nans
display(feature_types.fillna(''))
display(feature_types.count())


# __Categorical__ features can be further divided into:
# 
# - __Ordinal__ - clear ordering of variable's values can be inferred. Intuitive example of such values
# could be a triple: `['Bad', 'Average', 'Good']` with respective encodings `[-1, 0, 1]`. 
# - __Nominal__ - variable's values cannot be reasonably ordered. Here hair colour would be a perfect example.
# It would be hard to justify why `blonde` should be rated higher than `dark` hair. You should perform [One-Hot Encoding](https://en.wikipedia.org/wiki/One-hot) for those variables.
# 
# By employing this scheme, we can encode additional knowledge into the data. It often helps models as they don't have to infer it by themselves or sometimes they are even unable to do so. Yeah, you should take some time with encoding, trust me.
# 
# To create those distinctions one needs to manually explore each categorical trait. In my opinion here are
# the ordinal variables:

# In[ ]:


ordinal_features = np.array([
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "HeatingQC",
    "KitchenQual",
    "Functional",
    "FireplaceQu",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "PoolQC",
    "Utilities",
    "BsmtFinType1",
    "BsmtFinType2",
    "LandSlope",
    "Electrical",
    "Fence"
])


# There are a few features I have decided to exclude from this group due to
# lack of necessary housing knowledge and/or unclear descriptions, namely:
# 
# - __Heating__
# - __RoofMatl__
# - __Exterior1st__
# 
# In case of reasonable doubt __I would rather NOT__ encode feature specific
# knowledge (ordinality in this case).
# Furthermore I think those three traits listed above vary  to infer any ordering.
# 
# Our `nominal_features` will be a difference between `categorical` and it's `ordinal` subset, simply defined like:

# In[ ]:


nominal_features = np.setdiff1d(categorical_features, ordinal_features)


# Finally trivial recovery of nominal features by calculating the difference of
# sets, leaves us with the following separation:

# In[ ]:


feature_names = {
    "Ordinal": ordinal_features,
    "Nominal": nominal_features,
    "Numerical": numerical_features,
}

feature_types = pd.DataFrame.from_dict(feature_names, orient="index").T
display(feature_types.fillna(''))
display(feature_types.count())


# Features seem to be appropriately divided (later we will encode them appropriately). Now it's time for...

# # 2. Locating missing values and fixing where possible
# 
# I will join training and testing dataset for this section as it will be easier to work with it. 

# In[ ]:


full = pd.concat([train, test], keys=['train', 'test'], sort=False)
display(full.head())
display(full.tail())


# For faster analysis I will implement functions describing NaN values in our
# dataset (will be used extensively along the way).

# In[ ]:


def nan_rows(dataset: pd.DataFrame):
    nan_rows_count = dataset.isnull().any(axis=1).sum()
    return nan_rows_count, nan_rows_count / len(dataset) * 100


def nan_features(dataset: pd.DataFrame):
    nans_per_feature = dataset.isnull().sum().sort_values(ascending=False)
    nan_features = nans_per_feature[nans_per_feature != 0].reset_index()
    nan_features.columns = ["Feature", "NaNs"]
    return nan_features


def nan_count(dataset: pd.DataFrame):
    return dataset.isnull().sum().sum()


def display_nan_statistics(
    dataset: pd.DataFrame, remove_target: bool = True, target_name: str = "SalePrice"
):
    if remove_target:
        df = dataset.drop(target_name, axis=1)
    else:
        df = dataset
    print("Dataset contains {} NaNs".format(nan_count(df)))
    print("NaN rows: {} | In percentage: {}".format(*nan_rows(df)))
    print("NaNs per feature:")
    display(nan_features(df))


display_nan_statistics(full)


# ## 2.1 Imputting values according to description
# 
# For some features __NaN__ means absence of a trait, not a missing value. Usually you can read more about it in dataset's description. If you can't find such information, you should assume values are missing.
# 
# I will imput absent values with `ValueAbsent`, accordingly to `data_description.txt` provided for this competition:

# In[ ]:


described_features = [
  "PoolQC",
  "MiscFeature",
  "Alley",
  "Fence",
  "FireplaceQu",
  "GarageCond",
  "GarageQual",
  "GarageFinish",
  "GarageType",
  "BsmtCond",
  "BsmtExposure",
  "BsmtQual",
  "BsmtFinType2",
  "BsmtFinType1",
]

full.fillna(value={feature: "ValueAbsent" for feature in described_features}, inplace=True)

display_nan_statistics(full)


# One can see only __~17%__ of data is missing values right now.
# 
# __A little cheat now__: example with missing __Electrical__ feature can be safely deleted, because:
# - It is in the training dataset (not required for making prediction)
# - Only one example will not affect predictive power of the algorithm one might later build
# - I cannot do any sensible analysis with __only one value__ missing and we should assume it's missing completely at random, hence can be safely deleted.
# 
# You should always be careful when deleting examples as it may change true distribution of underlying data. Usually it should be avoided (unless you have a good reason to do otherwise) as too large standard error might be introduced by doing so. 
# 
# For each feature one should decide what's the mechanism standing behind missingnes (__Missing Completely At Random (MCAR)__ / __Missing At Random (MAR)__ or __Missing Not At Random__) and act approprietly (see [here](https://www.theanalysisfactor.com/mar-and-mcar-missing-data/) for some description). I will say a few words more about it at the end of this notebook.
# 
# 

# In[ ]:


full = full[full["Electrical"].notnull()]


# ## 3. Re-input falsely described absent values
# 
# As you shall soon see there are some data points contradicting correctness of 
# data description given to us by the competition creators. I call those `falsely described absent values` as in reality those are __missing__, not absent as `data_description.txt` claims.
# 
# I think you will understand the idea as I go through some examples:
# 
# ### 3.1 Garage
# 
# __GarageYrBlt__ is not described in the data itself, but it may be that `NaNs` mean the garage is missing. 
# 
# To verify this hypothesis I am going to take a subset of our dataset
# linked to garage features, change imputed values to NaNs again and check missingness correlation.

# In[ ]:


garage_data = full.filter(regex=".*Garage.*")
garage_data = garage_data.replace(["ValueAbsent", 0], np.NaN)
msno.heatmap(garage_data)


# Correlation is pretty high, hypothesis seems almost perfectly in-line with data.
# 
# You should spot something strange though; __GarageType's missingness__ does not
# perfectly correlate with other variables (although it should according to the description). All in all, if there is no garage (indicated by say `GarageType`) there should be no garage quality (`GarageQual`), right?
# 
# You will see this pattern more than once so I have created a function finding examples where one feature has `NaN` and other has `non-NaN` value even though both values should be equal:

# In[ ]:


def falsely_described_nans(dataset):
  # If any feature in the dataset is NaN while other is not it will be returned.
  indices = []
  features = dataset.columns.tolist()
  for index, nan_feature in enumerate(features):
    for non_nan_feature in features[index:]:
      df = dataset[
                dataset[nan_feature].isnull() &
                dataset[non_nan_feature].notnull()
              ]
      if not df.empty:
        indices.extend(tuple(df.index.tolist()))

  return dataset.loc[list(set(indices))].copy()


# **Remember: according to data description all `NaN`s above should mean absence of
# garage, 
# hence all of them should have `NA` in the same data points!** 
# 
# See below to be sure it really isn't the case here:

# In[ ]:


false_nans = falsely_described_nans(garage_data)
display(false_nans)


# Data for this example seems to be flawed , so I am going to leave values of example `666` untouched. It cannot be deleted (as it's in the `test` set). Other data points will be left as they were (so imputed according to description). 
# 
# Additionally I will impute `GarageYrBlt` with unique value `0` so models are able to learn it means garage missingness.
# One could argue that missing garage is N times worse than garage built in a certain year. Usually, the later the garage was built, the worse for prices of the house. IMO this imputation is a good pragmatic choice for future predictive tasks.

# In[ ]:


# Change GarageYrBlt values to zero
full.fillna(value={"GarageYrBlt": 0}, inplace=True)
# Get index and replace our imputed ValueAbsent with NaN 
full.at[false_nans.index, ["GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond"]] = np.NaN

display_nan_statistics(full)


# ### 3.2 Basement
# ****
# There are some predictors (variables) revolving around basement of the house. Their missing values should not cotradict each other either. 
# 
# Let's see how this goes...

# In[ ]:


basement_data = full.filter(regex=".*Bsmt.*").copy()
basement_data.replace(["ValueAbsent"], np.NaN, inplace=True)
msno.heatmap(basement_data)


# Matter seems more complicated than it was previously. I will focus on the described data
# for now. Remember: __Either all variables should be missing or none for each example__. 
# 
# If the case was different, it would mean the data is __trully missing__ and not __absent__ as description tells us:

# In[ ]:


described_features = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
basement_described_data = basement_data[described_features]

false_nans = falsely_described_nans(basement_described_data)
display(false_nans)

display(msno.heatmap(basement_described_data))


# Once again, there are some records in which we should not trust as they are self-contradictory (according to
# `data_description.txt` at least). Same thing applies: we change all those feature values to `NaN`s for those specific examples.
# 
# You may also leave their data instead on imputting `NaN`s everywhere like I did, but I do not trust those records and would rather get them fully imputed (which might be the wrong move, it's your call).

# In[ ]:


full.at[
    false_nans.index,
    ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"],
] = np.NaN

display_nan_statistics(full)


# ### 3.3 Fake year (great name indeed)
# 
# Year is another type of feature where multiple predictors depend on each other.

# In[ ]:


year_features=[ "YearBuilt", "YearRemodAdd", "GarageYrBlt" ]
full[year_features].describe()


# Maximum value for `GarageYrBlt` is way above the possible one (2010, dataset
# creation date), we should inspect anything above this threshold:

# In[ ]:


fake_years = full[full["GarageYrBlt"] > 2010]
display(fake_years)


# I agree with conclusions from [this
# kernel](https://www.kaggle.com/laurenstc/top-2-of-leaderboard-advanced-fe) 
# and
# it probably is a typo and the year was meant to be 2007, I'm going to change it accordingly:

# In[ ]:


full.at[fake_years.index, 'GarageYrBlt'] = 2007


# Furthermore it should not be possible for __YearRemodAdd__ to be smaller than
# __YearBuilt__, cell below verifies it:

# In[ ]:


fake_remodel = full[full["YearBuilt"] > full["YearRemodAdd"]][["YearBuilt", "YearRemodAdd"]]
fake_remodel


# It is hard to pinpoint the mistake here, my guess would be the house was not
# remodelled
# at all so I will change `YearRemodAdd` to reflect my assumption making it equal to `YearBuilt` value (you may decide to go the other wayy around and set both values to `2001`):

# In[ ]:


full.at[fake_remodel.index, 'YearRemodAdd'] = 2002


# # 3.4 Fake pools, masonry and parking lots
# 
# If there is no pool quality there should be no pool (or so the description says). Same goes for masonry or parking lot.
# 
# Below is the code inspecting whether the description has hidden some `NaN`s from us once again:

# In[ ]:


# Go through all the possible contradictions to description:
def contradictive_area_quality(
    dataset,
    area_feature: str,
    quality_feature: str,
    absent_value_string: str = "ValueAbsent",
):
    area_no_quality = full[
        (dataset[area_feature] != 0) & (full[quality_feature] == absent_value_string)
    ].copy()

    quality_no_area = full[
        (dataset[area_feature] == 0) & (full[quality_feature] != absent_value_string)
    ].copy()

    return pd.concat([area_no_quality, quality_no_area])

print("Contradictive pools:")
pools = contradictive_area_quality(full, "PoolArea", "PoolQC")
display(pools[["PoolArea", "PoolQC"]])

print("Contradictive masonry veneer:")
veneers = contradictive_area_quality(full, "MasVnrArea", "MasVnrType", "None")
display(veneers[["MasVnrArea", "MasVnrType"]])

print("Lot Area false variables")
zero_lot_data = full[full["LotArea"] == 0].copy()
display(zero_lot_data)


# Of course it did, let's fix it the same way it's been already done:

# In[ ]:


# Pool Area looks legit, I suppose poll quality is simply missing
full.at[pools.index, "PoolQC"] = np.NaN

# 1.0 veneer area is ridiculously small, maybe a simple mistakes, change to 0
indices = pd.MultiIndex.from_tuples([("train", 773), ("train", 1230), ("test", 992)])
full.at[indices, "MasVnrArea"] = 0

# MasVnrType does not seem random, input NaN and leave it for imputation
indices = pd.MultiIndex.from_tuples([("train", 688), ("train", 1241), ("test", 859)])
full.at[indices, "MasVnrArea"] = np.NaN

# MasVnrArea looks legit, the type seems to be simply missing
indices = pd.MultiIndex.from_tuples(
    [("train", 624), ("train", 1300), ("train", 1334), ("test", 209)]
)
full.at[indices, "MasVnrType"] = np.NaN


# I have not found other features contradicting `description.txt` because of missingness. One might check for other contradictory data, though it would require much more work and might amount to manual inspection of many records.
# 
# If you did it, please post a link to your kernel below, would love to see!

# ## 4. Map ordinal features
# 
# The last thing I am going to do, for now at least,
# is changing ordinal variables to it's numeric counterparts. I have ordered each of the variables from best (higher score) to worst (lowest score). I have decided that absence of features __is worse__ than poor quality. IMO this assumption makes intuitive sense, though does not always hold...
# 
# Let's imagine beautiful house with rotten pool, it would be better to have it absent in this case, wouldn't it? You might want to check out if this occurs in the dataset (like above, would be cool to see someone do this) and maybe you can find a better encoding scheme for those features than I did?

# In[ ]:


ordinal_mapping = {
  "Ex": 2,
  "Gd": 1,
  "TA": 0,
  "Fa": -1,
  "Po": -2,
  "ValueAbsent": -3, # our designed NaN placeholder

  "Gd":	1,
  "Av":	0,
  "Mn":	-1,
  "No":	-2,

  "GLQ": 3,
  "ALQ": 2,
  "BLQ": 1,
  "Rec": 0,
  "LwQ": -1,
  "Unf": -2,

  "Typ" : 3,
  "Min1": 2,
  "Min2": 1,
  "Mod" : 0,
  "Maj1": -1,
  "Maj2": -2,
  "Sev" : -3,
  "Sal" : -4,

  "Fin": 0,
  "RFn": -1,
  "Unf": -2,

  "Y": 1,
  "P": 0,
  "N": -1,

  "AllPub": 1,
  "NoSewr": 0,
  "NoSeWa": -1,
  "ELO": -2,

  "Gtl": 1,
  "Severe": -1
}

full.replace({"LandSlope": "Sev"}, "Severe")
full = full.replace(ordinal_mapping)

display_nan_statistics(full)


# Finally, let's split our dataset and save it separately to `train.csv` and `test.csv` after our initial analysis.

# In[ ]:


initially_preprocessed_train = "../input/initially_preprocessed_train.csv"
initially_preprocessed_test = "../input/initially_preprocessed_test.csv"

train = full.loc["train", :]
test = full.loc["test", :]

# Can't save here, you might do it locally though :)
# train.to_csv(initially_preprocessed_train)
# test.to_csv(initially_preprocessed_test)


# __Important:__ I am not going to `one-hot` dataset for now. It's a simple one-liner with `pandas` and I think analysis of missingness mechanisms should be done before tinkering with this representation. Furthermore, it would unnecessarily increase dataset size, hence I left it out of this kernel (and it's a pretty easy, boring task even).
# 
# Oh, and speaking of data imputation...

# # 5. Data imputation and final words
# 
# Now, when we know what values are missing and where, we can move on to imputation.
# 
# A great source of knowledge on the topic can be found in [Flexible Imputation of Missing Data](https://www.amazon.com/Flexible-Imputation-Missing-Interdisciplinary-Statistics/dp/1439868247) by __Stef Van Buuren__. Inside it you can find why mean imputation is not good for your data (usually), what's MICE package is actually about, what's up with those MCAR, MAR, MNAR and what other SOTA approaches of handling those problems exist. __Highly recommended to at least skim through!__
# 
# #### If you want me to make a kernel on data imputation this time using R language or you found this kernel useful, please upvote it. If you have some comments I would love to hear them, thanks for staying with me! :D
