#!/usr/bin/env python
# coding: utf-8

# **Introduction**

# First of all, I have to say that I am a newbie. I am learning (or re-learning, since I studied a bit of it at the University... 20 years ago) Machine Learning, and enjoying very much kaggle while doing so.
# 
# My kernel has very few original things. It is essentially a work of going through a lot of interesting kernels and trying to get the best from each of them and put it all together. Most things are a compendium of all the techniques I have seen while studying available kernels on the competition. I have extracted things, and owe them almost all my result, from:
# * [https://www.kaggle.com/gunesevitan/in-depth-eda-and-stacking-with-house-prices](http://)
# * [https://www.kaggle.com/humananalog/xgboost-lasso](http://)
# * [https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard](http://)
# * [https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset](http://)
# * [https://www.kaggle.com/apapiu/regularized-linear-models](http://)
# * [https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python](http://)
# * [https://www.kaggle.com/yw6916/top-9-eda-stacking-ensemble](http://)
# * [https://www.kaggle.com/nwhidden/ames-house-prices-regression-practice](http://)
# * [https://www.kaggle.com/chiranjeevbit/complete-eda-and-predict-house-price](http://)
# 
# And I also upvoted every one of them, of course!
# 
# I tried to achieve two things with the kernel:
# 1. To combine every interesting technique I learnt. This is dangerous, since it could happen that mathematically combining some of them may not make much sense. I tried to validate the result by measuring the scores on every change, since I still do not understand the implications of every parameter out there.
# 2. To generalize, where possible, the techniques to make them domain independent. Getting a fully generic algorithm is quite hard, but the kernel could serve as a good code base for any other regression problem.
# 
# Please, any feedback in the comments is more than welcome. 

# Imports of all the libraries used along the code

# In[ ]:


import warnings

import numpy as np
import pandas as pd
from mlxtend.regressor import StackingCVRegressor
from scipy.special import boxcox1p
from scipy.stats import skew
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# %matplotlib inline
warnings.filterwarnings('ignore')


# **First steps**
# 
# First of all, load the data sets, both the train and the test. I create one joint collection, called *all_data*, so that all the modifications to the features I will do will be applied to both the train and the test set at the same time.

# In[ ]:


###############################################################
# Load the data                                               #
###############################################################
train_data_path = 'input/train.csv'
test_data_path = 'input/test.csv'

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

train_data_size = train_data.shape[0]
test_data_size = test_data.shape[0]

all_data = pd.concat((train_data, test_data)).reset_index(drop=True)


# I arrange here all the variables in the code that help me take decisions, to make changes easier.

# In[ ]:


missing_data_threshold = 80
none_category_threshold = 2
predominant_value_threshold = 98
feature_correlation_threshold = 0.80
prediction_correlation_threshold = 0.60
skewness_threshold = 0.75
boxcox_lambda = 0.15
n_folds = 7
#blending weights
stacked_pred_weight = 0.4
gbr_pred_weight = 0.1
xgb_pred_weight = 0.2
ridge_pred_weight = 0.1
elastic_pred_weight = 0.05
lasso_pred_weight = 0.05
lr_pred_weight = 0.1


# Variable 'predicted_feature' will help in keeping the code independent from the exact feature we want to predict. This will not made the code generic, but will help if in the future I want to resuse this code for another problem.

# In[ ]:


predicted_feature = 'SalePrice'

y = train_data[predicted_feature]


# **Feature engineering**
# 
# ***'Categorize' some numerical features***
# 
# I want to organize the features in two collections, one for numerical features and another one for categorical features. However, there are some features that even though they contain numbers, and would be automatically classified as numerical, in essence they are representing categories with those numbers. I transform the numbers into strings so that those features are detected as categorical from now on.

# In[ ]:


###############################################################
# Organize features into collections                          #
###############################################################

# Some of the numerical features are really categorical features that encode the different categories with numbers.
# Converting them to 'str' will help later in classifying them as categorical.
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['YrSold'] = all_data['YrSold'].apply(str)
all_data['YearBuilt'] = all_data['YearBuilt'].apply(str)
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].apply(str)
all_data = all_data.replace({"MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun", 7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}})

numeric_features = [f for f in all_data.columns if all_data.dtypes[f] != 'object']
numeric_features_filtered = numeric_features

categorical_features = [f for f in all_data.columns if all_data.dtypes[f] == 'object']


# And now I define the "features" collection, that initially has all the categorical and numerical features in the dataset, but will be heavily modified during the process. This way, I will use *all_data[features]* along the code to refer to the dataset I am working with, but avoiding the modification of the original datasets, so that I can check the original data at any time of the process (and I had to do it quite a few times).
# 
# As a first step, I drop the 'Id' and 'SalePrice' columns, since I will not need them during the data analysis/feature engineering phase.

# In[ ]:


# I define the set of features that will be used for training. This allows me to make any changes I need easily
# and at the same time, leave the original information unchanged.
features = numeric_features_filtered + categorical_features
features.remove('Id')
features.remove('SalePrice')


# ***Missing data***
# 
# First of all, I will handle missing data. There are many features that have missing values, and even some of them which miss almost all of the values. First, tell me which features miss values and how many:

# In[ ]:


###############################################################
# Analyze and handle missing data                             #
###############################################################
# Calculate the amount of missing values in each feature
missing_perc = (all_data.isnull().sum() / len(all_data)) * 100
missing_perc = missing_perc[(missing_perc.index != predicted_feature) & (missing_perc > 0)]
missing_perc.sort_values(inplace=True, ascending=False)

missing_abs = all_data.isnull().sum()
missing_abs = missing_abs[(missing_abs.index != predicted_feature) & (missing_abs > 0)]
missing_abs.sort_values(inplace=True, ascending=False)

missing_data = pd.DataFrame({'Missing_Ratio': missing_perc, 'Missing_Total': missing_abs})
print("\nFeatures with missing values:")
print('{}'.format(missing_data))


# This is the result:
# 
#                     Missing_Ratio  Missing_Total
#     PoolQC            99.657417           2909
#     MiscFeature       96.402878           2814
#     Alley             93.216855           2721
#     Fence             80.438506           2348
#     FireplaceQu       48.646797           1420
#     LotFrontage       16.649538            486
#     GarageQual         5.447071            159
#     GarageCond         5.447071            159
#     GarageFinish       5.447071            159
#     GarageType         5.378554            157
#     BsmtExposure       2.809181             82
#     BsmtCond           2.809181             82
#     BsmtQual           2.774923             81
#     BsmtFinType2       2.740665             80
#     BsmtFinType1       2.706406             79
#     MasVnrType         0.822199             24
#     MasVnrArea         0.787941             23
#     MSZoning           0.137033              4
#     BsmtFullBath       0.068517              2
#     BsmtHalfBath       0.068517              2
#     Utilities          0.068517              2
#     Functional         0.068517              2
#     Electrical         0.034258              1
#     BsmtUnfSF          0.034258              1
#     Exterior1st        0.034258              1
#     Exterior2nd        0.034258              1
#     TotalBsmtSF        0.034258              1
#     GarageArea         0.034258              1
#     GarageCars         0.034258              1
#     BsmtFinSF2         0.034258              1
#     BsmtFinSF1         0.034258              1
#     KitchenQual        0.034258              1
#     SaleType           0.034258              1
# 
# There are several different situations. There are features that miss most of the values, so they are not very representative to do the training, and there are other features that just miss some of the values and some handling of these missed values should be implemented to improve the training accuracy. 
# 
# First, I try to implement some generic decisions, that will let me handle most of the cases:
# 1. I will drop features that miss more than 80% of the values
# 2. For categorical features, if they miss more than 2% of the values, I fill them with 'None', making it a new category by itsef, while if the missing values represent less than 2% percent of the values, I just fill them with the most common value in the category.
# 
# There are some other features that are not covered by these criteria, and I decided to fill them manually. Maybe these cases could be generalized to find a domain agnostic criteria too.

# In[ ]:


# Handle variable by variable
# We discard variables with more than 80% percent of data missing
for f in missing_data[missing_data.Missing_Ratio > missing_data_threshold].index:
    features.remove(f)

# For categorical values, if the number of na represent more than 2%, we fill it with "None",
# but if it is less than 2%, we fill it with the most common value in the dataset
for f in missing_data[(missing_data.Missing_Ratio <= missing_data_threshold) & (missing_data.Missing_Ratio > none_category_threshold)].index:
    if f in categorical_features:
        all_data[f] = all_data[f].fillna("None")

for f in missing_data[(missing_data.Missing_Ratio <= none_category_threshold)].index:
    if f in categorical_features:
        all_data[f] = all_data[f].fillna(all_data[f].mode()[0])


# Some features are domain specific and I decide how to fill them manually
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
all_data["GarageYrBlt"] = all_data["GarageYrBlt"].fillna(0)
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data["BsmtFullBath"] = all_data["BsmtFullBath"].fillna(0)
all_data["BsmtHalfBath"] = all_data["BsmtHalfBath"].fillna(0)
all_data["BsmtUnfSF"] = all_data["BsmtUnfSF"].fillna(0)
all_data["TotalBsmtSF"] = all_data["TotalBsmtSF"].fillna(0)
all_data["GarageArea"] = all_data["GarageArea"].fillna(0)
all_data["GarageCars"] = all_data["GarageCars"].fillna(0)
all_data["BsmtFinSF2"] = all_data["BsmtFinSF2"].fillna(0)
all_data["BsmtFinSF1"] = all_data["BsmtFinSF1"].fillna(0)


# ***Generate new features based on existing ones***
# 
# Now I generate some additional features based on the existing features. As the comment in the code says, this is another of the domain specific parts, that needs a careful study of the feature set, some domain knowledge and some common sense. 
# 
# I have generated two sets of new features. The first set, refers to summaries of the most important features of the houses (total square feet, total square feet with good quality, total baths, total porch area and overall rating), something that every buyer would like to know about any house. The second set is a set of flags that denote if the house has some "complementary" features such as pool, second floor, garage, etc. Somehow this information is already present in the data (i.e., PoolArea = 0 means no pool), but this way I make it explicit.

# In[ ]:


###############################################################
# Generate new variables from the existing ones               #
###############################################################
# I generate new features by combining existing features in the dataset.
# I saw no way to automatize this. It is completely domain dependant.
all_data["TotalSF"] = all_data["TotalBsmtSF"] + all_data["GrLivArea"]
all_data["TotalSF_HQ"] = all_data["TotalBsmtSF"] + all_data["GrLivArea"] - all_data["LowQualFinSF"] - all_data["BsmtUnfSF"]
all_data["TotalBath"] = all_data["BsmtFullBath"] + all_data["FullBath"] + (all_data["BsmtHalfBath"]/2) + (all_data["HalfBath"]/2)
all_data["TotalPorchSF"] = all_data["OpenPorchSF"] + all_data["EnclosedPorch"] + all_data["3SsnPorch"] + all_data["ScreenPorch"]
all_data["OverallRating"] = all_data["OverallQual"] * all_data["OverallCond"]

features.append('TotalSF')
features.append('TotalSF_HQ')
features.append('TotalBath')
features.append('TotalPorchSF')
features.append('OverallRating')

all_data["HasPool"] = all_data["PoolArea"].apply(lambda x: 1 if x > 0 else 0)
all_data["Has2ndFloor"] = all_data["2ndFlrSF"].apply(lambda x: 1 if x > 0 else 0)
all_data["HasGarage"] = all_data["GarageArea"].apply(lambda x: 1 if x > 0 else 0)
all_data["HasBsmt"] = all_data["TotalBsmtSF"].apply(lambda x: 1 if x > 0 else 0)
all_data["HasFireplace"] = all_data["Fireplaces"].apply(lambda x: 1 if x > 0 else 0)

features.append('HasPool')
features.append('Has2ndFloor')
features.append('HasGarage')
features.append('HasBsmt')
features.append('HasFireplace')


# ***Drop features that are not informative***
# 
# It is necessary to keep working on the data set with the features. 
# 
# First, I drop the features that have a predominant value (over 98% of the cases), because they are of little use to predict any price. It results in dropping the following features:
# 
# Variables that have a very predominant value. Check if could be candidates to drop:
# * Variable: 3SsnPorch: value "0" appears 2882 times, 98.7324426173347% of the examples
# * Variable: LowQualFinSF: value "0" appears 2879 times, 98.6296676944159% of the examples
# * Variable: PoolArea: value "0" appears 2906 times, 99.55464200068516% of the examples
# * Variable: Condition2: value "Norm" appears 2889 times, 98.97225077081193% of the examples
# * Variable: Heating: value "GasA" appears 2874 times, 98.45837615621788% of the examples
# * Variable: RoofMatl: value "CompShg" appears 2876 times, 98.52689277149709% of the examples
# * Variable: Street: value "Pave" appears 2907 times, 99.58890030832477% of the examples
# * Variable: Utilities: value "AllPub" appears 2918 times, 99.9657416923604% of the examples
# * Variable: HasPool: value "0" appears 2906 times, 99.55464200068516% of the examples

# In[ ]:


###############################################################
# Select and discard the most relevant/useless features       #
###############################################################
# Automatically drop features that have a value present in more than 98% of the examples
# These features are not very informative
print('\nVariables that have a very predominant value. Check if could be candidates to drop')
for i in all_data[features].columns:
    most_frequent_value = all_data[i].mode().iloc[0]
    most_frequent_value_repetitions = all_data[i].value_counts().iloc[0]
    most_frequent_value_perc = most_frequent_value_repetitions / len(all_data) * 100
    if most_frequent_value_perc > predominant_value_threshold:
        print('Variable: {}: value "{}" appears {} times, {}% of the examples'.format(i, most_frequent_value,
                                                                                      most_frequent_value_repetitions,
                                                                                      most_frequent_value_perc))
        features.remove(i)


# I have some doubts wether the *HasPool*/*PoolArea* features could help predicting the price of the most expensive houses. I need to check the data. In case it seems so, it would be enough to re-append them to the *features* collection so that the following steps still take them into acount.

# ***Encode categorical features***
# 
# Then I encode the categorical values. This converts categorical features into numerical so that the regression algorithms can use them in the calculations. 
# 
# I used a piece of code that assigned to the different values of a feature the ordinal number according to the average sale price of the houses with that value. This is, if a feature has values A, B and C, and the average sale prices for houses for these values are 10000, 15000 and 12000, respectively, the encoding algorithm would substitute A with 3, B with 1 and C with 2, like this:
# 
# * A --> Average SalePrice=10000 --> 3
# * B --> Average SalePrice=15000 --> 1
# * C --> Average SalePrice=12000 --> 2
# 
# However, I decided to modify this substitution and instead of the ordinal, I assign directly the average sale price of the category value. In my impression, this keeps the ordering information while also including information about the proportion in which each category value affects the final sale price. The tests I did seem to give slightly better results. There is a caveat here (represented by an 'if' clause in the code), that is the fact that you can find a value in the test set that does not appear in the train set, so there is no information about the SalePrice for that category value. In this case, I decided to assign the mean SalePrice for all houses in the train set. In the Ames dataset this happens only once, with the value 150 of feature MsSubClass.

# In[ ]:


# Encode the categorical features.
# I assign the mean of the SalePrice for houses with each value in that category
def encode(frame, feature):
    numeric_values = pd.DataFrame()
    numeric_values['categories'] = frame[feature].unique()
    numeric_values.index = numeric_values.categories
    category_value = frame[[feature, predicted_feature]].groupby(feature).mean()[predicted_feature]
    if np.isnan(category_value).any():
        category_value[np.isnan(category_value) == True] = frame[predicted_feature].mean()
    numeric_values['spmean'] = category_value
    numeric_values = numeric_values['spmean'].to_dict()
    for cat, value in numeric_values.items():
        frame.loc[frame[feature] == cat, feature + '_E'] = value

new_features = []
features_to_delete = []
for q in features:
    if q in categorical_features:
        encode(all_data, q)
        features_to_delete.append(q)
        new_features.append(q + '_E')

# I substitute the categorical features by their encoded versions
for x in features_to_delete:
    features.remove(x)
features = features + new_features


# ***Detect correlated features***
# 
# Now that all the features are numeric, it is possible to study the correlations between the features, and between the features and the SalePrice.
# 
# I study first the correlation between features. Highly correlated features should be avoided since they negatively affect the predictions. 

# In[ ]:


#
# Detect correlations between training features
#
corr_features = pd.DataFrame({'Feature_A': [], 'Feature_B': [], 'corr_score': []})

corrmat = all_data[features].corr()
for i in range(2, len(corrmat.columns)):
    for j in range(1, i):
        feature_1 = corrmat.columns[i]
        feature_2 = corrmat.columns[j]
        if corrmat[feature_1][feature_2] > feature_correlation_threshold:
            corr_features.loc[len(corr_features)] = [feature_1, feature_2, corrmat[feature_1][feature_2]]

corr_learning_features = corr_features.loc[(corr_features.Feature_A != predicted_feature) & (corr_features.Feature_B != predicted_feature)]
print("\nThere are {} pairs of training features potentially correlated".format(len(corr_learning_features)))
for index, row in corr_learning_features.sort_values('corr_score', ascending=False).iterrows():
    print('{} y {} : {}'.format(row['Feature_A'], row['Feature_B'], row['corr_score']))


# The results are the following:
# 
# There are 14 pairs of training features potentially correlated
# * SaleType_E y SaleCondition_E : 0.9210433865680396
# * Exterior2nd_E y Exterior1st_E : 0.9116839166782535
# * Has2ndFloor y 2ndFlrSF : 0.9064688408171122
# * HasFireplace y Fireplaces : 0.8996251260698431
# * GarageCars y GarageArea : 0.8898902241956981
# * TotalSF y GrLivArea : 0.8717698431418889
# * FireplaceQu_E y HasFireplace : 0.8693916712558053
# * YearBuilt_E y GarageYrBlt_E : 0.8499587140172883
# * TotalSF_HQ y TotalSF : 0.8408043871234457
# * BsmtFinType2_E y HasBsmt : 0.8291587577005163
# * TotalSF y TotalBsmtSF : 0.8271178158193125
# * GarageCond_E y HasGarage : 0.8206178473298337
# * GarageQual_E y GarageCond_E : 0.8108106519177948
# * TotRmsAbvGrd y GrLivArea : 0.8083544205418542
# 

# I suppose it would be possible to automatically decide which features to drop, but in this case I decided to do it manually. I chose the minimum amount of features that would eliminate the maximum amount of the highest correlated features, and only for those features that clearly representing the same concept, such GarageCArs and GarageArea. If unsure, I decided to maintain the features in the dataset.

# In[ ]:


# I drop only some of the features, that are clearly overlapping
features.remove('GarageCond_E')
features.remove('GarageArea')
features.remove('GarageQual_E')


# ***Detect correlation between features and the SalePrice***
# 
# And next, I detect correlation between the remaining features and the SalePrice. This will point out which of the features in the data set are more representative to decide the price of a house, and I will use this information to add some new very important features to the collection. 

# In[ ]:


#
# And now detect correlations between training features and the predicted feature
#
corr_features = pd.DataFrame({'Feature_A': [], 'Feature_B': [], 'corr_score': []})

corrmat = all_data[:train_data.shape[0]][features + [predicted_feature]].corr('spearman')
for i in range(1, len(corrmat.columns)):
    feature_1 = corrmat.columns[i]
    feature_2 = predicted_feature
    if corrmat[feature_1][feature_2] > prediction_correlation_threshold:
        corr_features.loc[len(corr_features)] = [feature_1, feature_2, corrmat[feature_1][feature_2]]

prediction_correlated_features = []
print("\nThere are {} features potentially correlated with {}".format(len(corr_features), predicted_feature))
for index, row in corr_features.sort_values('corr_score', ascending=False).iterrows():
    if row['Feature_A'] != predicted_feature:
        print('{}: {}'.format(row['Feature_A'], row['corr_score']))
        prediction_correlated_features.append(row['Feature_A'])


# ***Generate polinomial features***
# 
# Correlation are detected and now it is time to generate the new features. I will get each of the features that have a correlation factor above the threshold and generate three new features: the square, the cube and the square root of each of them. These are called polinomial features and will let the prediction function adapt better to the SalePrice distribution.

# In[ ]:


#
# Using the numeric features that are more correlated with the feature to predict, we generate new polinomial features
#
for f in prediction_correlated_features:
    new_feature_name = "{}_2".format(f)
    all_data[new_feature_name] = all_data[f] ** 2
    features.append(new_feature_name)

    new_feature_name = "{}_3".format(f)
    all_data[new_feature_name] = all_data[f] ** 3
    features.append(new_feature_name)

    new_feature_name = "{}_sqrt".format(f)
    all_data[new_feature_name] = np.sqrt(all_data[f])
    features.append(new_feature_name)


# With the threshold I chose (0.60), these are the features I used to create the polinomical features.
# 
# There are 16 features potentially correlated with SalePrice
# * TotalSF: 0.8149837586936304
# * OverallQual: 0.8098285862017292
# * Neighborhood_E: 0.7557789170655119
# * GrLivArea: 0.7313095834659141
# * TotalSF_HQ: 0.7071092354480142
# * TotalBath: 0.7037310128090049
# * GarageCars: 0.6907109670497433
# * ExterQual_E: 0.6840137963904297
# * BsmtQual_E: 0.678026253071665
# * KitchenQual_E: 0.6728485475386916
# * YearBuilt: 0.6526815462850586
# * FullBath: 0.6359570562496957
# * GarageYrBlt: 0.6340952202911952
# * GarageFinish_E: 0.6339736230180727
# * TotalBsmtSF: 0.6027254448924095
# 
# 

# ***Outliers management***
# 
# The set of features is ready, but I will do a couple last steps analysing their values. 
# 
# The first step is regarding outliers. There are two houses in the training set that have a huge amount of square feet that does not correspond to their sale price. Most of the kernels I studied decided to drop them, and so did I.

# In[ ]:


###############################################################
# Transform numeric features                                  #
###############################################################
#Deleting outliers
# Outlier detection is also domain specific
for i in all_data[(all_data['GrLivArea']>4000) & (all_data['SalePrice']<300000)].index:
    all_data = all_data.drop(i)
    train_data = train_data.drop(i)
    y = y.drop(i)
    train_data_size -= 1


# ***Normalize features distributions***
# 
# The second step regarding the features values affects their statistical distribution. The more the distributions of the features look like a normal/Gaussian distribution, the better the predictions will be, so I check the skewness of all the final features, and those above a threshold are normalized using the boxcox function. Logarithm is an alternative used in many kernels, but the boxcox function seemed to provide slighlty better results.

# In[ ]:


# Check the skew of all numerical features
# For variable that show a high skewness, I aplly boxcox to normalize the distribution
numeric_feats = all_data[features].dtypes[all_data[features].dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' : skewed_feats})
skewness = skewness[abs(skewness) > skewness_threshold]
lam = boxcox_lambda
for feat in skewness.index:
    all_data[feat] = boxcox1p(all_data[feat], lam)


# So, the data is ready and now it is time to train the models. It is time to build them.

# **Model building**
# 
# First, I split the all_data set back into the training and testing sets.

# In[ ]:


###############################################################
# Build the model                                             #
###############################################################
# Regenerate the training and test datasets, using the set of features developed
train = all_data[['Id'] + features][:train_data_size].as_matrix()
test = all_data[['Id'] + features][-test_data_size:].as_matrix()


# It is also important to make sure that the variable we want to predict is normally distributed. I apply the log to it. Some kernels show that the Johnson distribution adapts better to the initial distribution of the SalePrice, and should make a better job normalizing it, but I got a bit better results using the log.

# In[ ]:


# Normalize the feature to predict
y = np.log1p(y)


# ***Model parameter tuning***
# 
# I have chosen seven different models. A simple logistic regression, Lasso, ElasticNet, Ridge, XGBRegressor, GradientBoostingRegressor and a stacked combination of all of them. But as a newbie that I am, I have no experience to decide how to wisely choose the parameters to apply in each of them. So I have to rely on something that allows me to measure objectively how well they perform. I will go one by one, and use cross validation on each of the models. I start with a for loop with coarse grained values (or several nested for loops if I want to tune several parameters), and once I see which values give better results, I repeat the steps, but with values in the range of the ones that gave the best scores in the first round, more fine grained. This is a very time consuming and resource intensive method, and only allows to fit a few parameters, but works reasonably well if you do not have any other heuristic to apply. 
# 
# I leave the commented code showing the parameter tuning loops, although it is not necessary to make the final predictions.

# In[ ]:


# First of all, I have to calculate the parameters for the models I will use to predict.
# I will use cross validation and measure the results using the same score as in the competition

kf = KFold(n_folds, shuffle=True, random_state=42)
scorer = make_scorer(mean_squared_error, greater_is_better = False)

# LogisticRegression
lr_model = make_pipeline(StandardScaler(), LinearRegression())
# score = np.sqrt(-cross_val_score(lr_model, train, y, scoring=scorer, cv = kf))
# print("\nLinear Regression Score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# Lasso Regression
# alphas = [0.00000001, 0.000003, 0.000009, 0.00001, 0.00003, 0.00009, 0.0001, 0.0003, 0.0009, 0.001, 0.003, 0.009, 0.01]
# for a in alphas:
# for a in np.arange(0.0005, 0.003, 0.00001):
#     lasso_model = Lasso(alpha=a, random_state=1)
#     score = np.sqrt(-cross_val_score(lasso_model, train, y, scoring=scorer, cv=kf))
#     if score.mean() < 1.1174:
#         print("Lasso Score with alpha {}: {} ({})".format(a, score.mean(), score.std()))

lasso_model = make_pipeline(StandardScaler(), Lasso(alpha=0.00096, random_state=1))
# score = np.sqrt(-cross_val_score(lasso_model, train, y, scoring=scorer, cv=kf))
# print("Lasso Score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


#ElasticNet
# l1_ratios = [0.0001, 0.0003, 0.0009, 0.001, 0.003, 0.009, 0.01, 0.03, 0.09, 0.1, 0.3, 0.9, 1, 3, 9, 10, 30, 90]
# # for a in alphas:
# for a in np.arange(0.001, 0.0095, 0.0005):
#     for l1 in l1_ratios:
#         ENet = ElasticNet(alpha=a, l1_ratio=l1, random_state=3)
#         score = np.sqrt(-cross_val_score(ENet, train, y, scoring=scorer, cv = kf))
#         if score.mean() < 0.1175:
#             print("ENet Score with alpha {} and l1_ratio {}: {} ({})".format(a, l1, score.mean(), score.std()))

elastic_model = make_pipeline(StandardScaler(), ElasticNet(alpha=0.0033, l1_ratio=0.3, random_state=3))
# score = np.sqrt(-cross_val_score(elastic_model, train, y, scoring=scorer, cv = kf))
# print("ENet Score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# Ridge Regression
# En primera ronda los mejores resultados fueron 6, 2, 6  y 3,2,3
# alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30]
# degree = [1,2,3]
# coef = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6]
# for a in np.arange(2.0, 8.0, 0.2):
#     for d in [2]:
#         for c in np.arange(3.0, 4.0, 0.1):
#             Ridge = KernelRidge(alpha=a, kernel='polynomial', degree=d, coef0=c)
#             score = np.sqrt(-cross_val_score(Ridge, train, y, scoring=scorer, cv = kf))
#             if score.mean() < 0.11417:
#                 print("Ridge Score with alpha {}, degree {} and coef {}: {} ({})".format(a, d, c, score.mean(), score.std()))

ridge_model = make_pipeline(StandardScaler(), KernelRidge(alpha=4.3, kernel='polynomial', degree=2, coef0=3.9))
# score = np.sqrt(-cross_val_score(ridge_model, train, y, scoring=scorer, cv = kf))
# print("Ridge Score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# XGBRegressor
# l_rates = [0.0005, 0.001, 0.003, 0.009, 0.01, 0.03, 0.09, 0.1, 0.3, 0.9]
# # for l in l_rates:
# for l in np.arange(0.001, 0.009, 0.0005):
#     incrementing_steps = 0
#     last_score = 999999999
#     for est in range(3000, 10000, 250):
#         XGB = XGBRegressor(n_estimators=est, learning_rate=l, n_jobs=-1, nthread=-1)
#         score = np.sqrt(-cross_val_score(XGB, train, y, scoring=scorer, cv = kf))
#         if score.mean() < 0.12:
#             print("XGB Score with learning rate {} and num estimators {}: {:.6f} ({:.6f})".format(l, est, score.mean(), score.std()))
#         if score.mean() > last_score:
#             incrementing_steps += 1
#             print("This try was worse than the previous one. incrementing_steps = {}".format(incrementing_steps))
#         else:
#             incrementing_steps = 0
#         last_score = score.mean()
#         if incrementing_steps == 3:
#             print("Stop exploring more estimators, because it has gone worse for the last three")
#             break
#     print("Finished for learning rate {}".format(l))

xgb_model = XGBRegressor(learning_rate=0.0085, n_estimators=3500, n_jobs=-1, nthread=-1)
# score = np.sqrt(-cross_val_score(xgb_model, train, y, scoring=scorer, cv=kf))
# print("XGB Score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# GradientBoostingRegressor
# for est in range(400, 2000, 200):
#     for a in np.arange(0.05, 1, 0.05):
#         for d in range(2, 6, 1):
#             for sub in np.arange(0.5, 1, 0.1):
#                 # skgbm_model = GradientBoostingRegressor(loss='huber', n_estimators=700, alpha=.5, max_depth=3, subsample=.6)
#                 skgbm_model = GradientBoostingRegressor(loss='huber', n_estimators=est, alpha=a, max_depth=d, subsample=sub)
#                 score = np.sqrt(-cross_val_score(skgbm_model, train, y, scoring=scorer, cv=kf))
#                 print("skgbm_model with est {}, alpha {:.3f}, d {} and sub {:.1f}: Score: {:.4f} ({:.4f})".format(est, a, d, sub, score.mean(), score.std()))

gbr_model = GradientBoostingRegressor(loss='huber', n_estimators=600, alpha=0.5, max_depth=2, subsample=0.6)
# score = np.sqrt(-cross_val_score(gbr_model, train, y, scoring=scorer, cv=kf))
# print("GradientBoostingRegressor Score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

# Stacked model
stacked_model = StackingCVRegressor(regressors=(lr_model, lasso_model, elastic_model, ridge_model, gbr_model), meta_regressor=xgb_model, use_features_in_secondary=True)
# score = np.sqrt(-cross_val_score(stacked_model, train, y, scoring=scorer, cv=kf))
# print("Stacked Score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# ***Make predictions and blend them***
# 
# Up to this point, the models are defined, so I just have to fit them using the training set and make the predictions of the data in the test set. I blend the results of the different models. I did not find a way to fine tune the weights to apply to each of the predictions of each of the models. I have chosen them based on what I read in other kernels and the results I measured for each model in the previous step. There could probably be better combinations. 

# In[ ]:


# And finally, use the parameterized models to predict the results and blend them
lr_model.fit(train, y)
lr_pred = lr_model.predict(test)

lasso_model.fit(train, y)
lasso_pred = lasso_model.predict(test)

elastic_model.fit(train, y)
elastic_pred = elastic_model.predict(test)

ridge_model.fit(train, y)
ridge_pred = ridge_model.predict(test)

xgb_model.fit(train, y)
xgb_pred = xgb_model.predict(test)

gbr_model.fit(train, y)
gbr_pred = gbr_model.predict(test)

stacked_model.fit(train, y)
stacked_pred = stacked_model.predict(test)


ensemble = stacked_pred * stacked_pred_weight + \ 
           gbr_pred * gbr_pred_weight +            xgb_pred * xgb_pred_weight +            ridge_pred * ridge_pred_weight +            elastic_pred * elastic_pred_weight +            lasso_pred * lasso_pred_weight +            lr_pred * lr_pred_weight

final_ensemble = np.floor(np.expm1(ensemble))


# And, to wrap up, generate the submission information for the competition.

# In[ ]:


# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.
output = pd.DataFrame({'Id': all_data['Id'][-test_data_size:], 'SalePrice': final_ensemble})
output.to_csv('submission.csv', index=False)


# **Conclusions/Next steps**
# 
# There are a bunch of things that would be worth trying and maybe could help in improving the results. To name a few:
# * I could try using different models and see if any of them provides better results, or complements the ones I already used.
# * I could repeat the model parameter tuning phase, as I changed some small details in the feature engineering phase after I had all the models already parameterized. Maybe this could result is slightly better tuned models. 
# * I could split the train set into a train and test set, and use it to fine tune the blending weights.
# * There are wonderful visualizations that are very helpful in the feature engineering phase. As my objective was to create a kernel as automatic as possible, I preferred to use raw data, but adding them would improve the decisions that had to be taken manually.
# * Retry the Johnson function in the SalePrice feature, as it should provide better results than the log.
# 
# There is a big score difference between the score that this kernel gets if I execute it inside kaggle, or if I execute the code in my laptop and I upload the submission file. I have to look for the reasons for that gap.
# 
# As I said at the beginning, any feedback will be highly appreciated.
# 
