#!/usr/bin/env python
# coding: utf-8

# <h1>House Price - Data Preparation</h1>

# In this script the preprocessing of the data of the Kaggle House Price Competion will be shown.
# <br>
# <h2>Problem Description:</h2>
# <p>The Goal is to predict the sale price for houses based on various features like building characteristics, size and location.</p>
# <h2>Data Preprocessing Process:</h2>
# <p>The following data preprocessing process will consist of the following steps:</p>
# <br>
# <h3>Data Exploration</h3>
#     <ul style="list-style-type:circle">
#     <li>Variable Identification</li>
#         <ul style="list-style-type:square">
#         <li>DataFrame size and shape</li>
#         <li>Label</li>
#         <li>Features (Numeric, Categorical)</li>
#         </ul>
#         <br>
#     <li>Univariate Analysis</li>
#         <ul style="list-style-type:square">
#         <li>Label</li>
#         <li>Numeric features</li>
#         <li>Categorical features</li>
#         </ul>
#         <br>
#      <li>Bivariate Analysis</li>
#         <ul style="list-style-type:square">
#         <li>Numeric features</li>
#         <li>Categorical features</li>
#         <li>Correlation Matrix</li>   
#         </ul>
# <h3>Data Manipulation</h3>
#     <ul style="list-style-type:square">
#     <li>Label Manipulation</li>
#     <li>Cutting Outliers</li>
#     <li>Impute missing values</li>
#     <li>Create new features</li>
#     <li>Turn some numeric features into categorical features</li>
#     <li>Correct Feature Skewness</li>
#     <li>Create Dummy Variables</li>
#     </ul>

# <h2>Data Exploration</h2>
# 
# In the first part of the analysis we will look at the data. We will investigate the size and shape of the test dataset and identify the variables. Then we will examine the distributions of the label and of each single feature. Afterwards we will explore the relationship between the features and the label and look at their correlation.

# We import all necessary modules.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
import warnings
warnings.filterwarnings('ignore')


# We load the training data und test data.

# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# <h3>Variable Identification</h3>

# First, we investigate the training dataset. We look at the shape of the dataset, the name of the columns, the type of the columns and have a first look at the amount of missing data.

# In[ ]:


df_train.shape


# In[ ]:


df_train.columns


# In[ ]:


df_train.info()


# We see that the training dataset consists of 1460 observations and 81 columns, our label column "SalePrice" and 80 features. We have 38 numeric features and 43 string features. We observe 19 features with missing values in the training dataset.

# Next we check for duplicate rows by comparing the number of unique ids (column "Id") with the total length of the dataframe.

# In[ ]:


unique_ids = len(set(df_train.Id))
total_ids = df_train.shape[0]
print("There are " + str(total_ids - unique_ids) + " duplicate Ids in the train dataset.")


# <h3>Univariate Analysis</h3>
# <h4>Label</h4>

# We will have a closer look at the label column of the dataset, the "SalePrice". We calculate the summary statistics, including the minimum and maximum value, the mean, the median and the standard deviation. The distribution can be visualized by a histogram. 

# In[ ]:


df_train["SalePrice"].describe()


# In[ ]:


sns.distplot(df_train["SalePrice"], bins=30, fit=norm)
plt.show()


# The histogram indicates that the distribution of the SalePrice exhibits a positive skewness and that the peakedness deviates from normal distribution. The skewness can be visualized by a probability plot of the SalePrice against the quantiles of the normal distribution.

# In[ ]:


stats.probplot(df_train["SalePrice"], plot=plt)
plt.show()


# A normal distributed label is a requirement for a linear regression. Therefore we will try transforming the label later in the data manipulation part.

# <h4>Numerical features</h4>

# After investigating the SalePrice, we will examine the other features. Our features have two different data types: numeric and string features. We will split them into numeric and categorical features for further analysis. We will look at histograms for the numerical features and countplots for the categorical features.

# In[ ]:


numeric_features = df_train.dtypes[df_train.dtypes != "object"].index
numeric_features = numeric_features.drop("SalePrice")


# In[ ]:


categorical_features = df_train.dtypes[df_train.dtypes == "object"].index


# In[ ]:


f = pd.melt(df_train, value_vars=numeric_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size = 5)
g = g.map(sns.distplot, "value")
plt.show()


# The histograms of the numeric data show that there are numeric values present in the dataset that consists of few discrete values, e.g. "OverallQual". These features will be transformed later in the data manipulation part from numerical to categorical.
# <br>
# <br>
# We can see that many features with continuous values do not exhibit normal distributions, e.g. "LotFrontage". We will try to improve the skewness of these features in the data manipulation part.
# <br>
# <br>
# Furthermore, it is visible that the most common value (mode) of some features is "0", e.g. for "2ndFlrSF". This occurs for features which give a measure of a characteristic that is not present for each house.

# <h4>Categorical features</h4>
# <br>
# We create a countplot for each categorical feature.

# In[ ]:


f = pd.melt(df_train, value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable", col_wrap=3, sharex=False, sharey=False, size = 5)
g = g.map(sns.countplot, "value")
plt.show()


# The countplots of the categorical data show the number of the different categories for each feature and their frequency distribution. There are features, where the different categories occur with similar frequency, e.g. for "Alley"; and there are feature, where one category is very dominant, e.g. for "Utilities".

# In[ ]:


df_train["Alley"].value_counts()


# In[ ]:


df_train["Utilities"].value_counts()


# <h3>Bivariate Analysis</h3>
# <br>
# In this part we will investigate the relationship between the label and the features. Therefore we will plot scatter plots for the numerical features in dependence of the label and boxplots for the categorical features in dependence of the label. Afterwards we will look at the correlations between all numerical columns.

# <h4>Numeric features</h4>
# <br>
# We create a scatter plot for each numerical feature in dependence of the label.

# In[ ]:


def regplot(x, y, **kwargs):
    sns.regplot(x=x, y=y)
    x=plt.xticks(rotation=90)
f = pd.melt(df_train, id_vars=['SalePrice'], value_vars=numeric_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(regplot, "value", "SalePrice")
plt.show()


# Scatter plots are useful to identify linear relationships between the label and the features. For example, the feature "GrLivArea" shows a linear relationship with the label. The SalePrice grows, when the feature "GrLivArea" increases. Scatter plots are also a very good way to spot outliers. In the discussed scatter plot of "GrLivArea" against the SalePrice, we can see two outliers for very high SalePrices of above 700.000 and two more outliers for smaller SalePrices and big "GrLivArea" values of above 4000. These oberservations will be cut out later in the data manipulation part.

# <h4>Categorical features</h4>
# <br>
# We create a boxplot for each categorical feature in dependence of the label.

# In[ ]:


def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)
f = pd.melt(df_train, id_vars=['SalePrice'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "SalePrice")
plt.show()


# The boxplots illustrate if the categorical features are a good indicator to group the observations into high and low SalePrices. For some features like "ExterQual" the ranges of the boxes for the different categories do not overlap at all, for other features like "BsmtFinType2" all boxes lie in the same range.

# <h4>Correlation matrix</h4>
# <br>
# The correlation matrix shows the correlation between all numerical columns in the dataset. This helps to illustrate the influence of the different features on the label. A correlation of "1" means completely correlated, a correlation of "0" means no correlation at all. Furthermore, for a linear regression it is assumed that the features are all independent and not correlated at all.

# In[ ]:


corr = df_train.corr()
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=.8, square=True);
plt.show()


# In[ ]:


print(corr.iloc[-1].sort_values(ascending=False).drop("SalePrice").head(10))


# There are 10 numerical features present in the test dataset that have a strong correlation above 0.5 with the label.

# In[ ]:


print(corr.loc["GarageCars"].sort_values(ascending=False).drop("GarageCars").head(1))


# There are features present in the dataset that are strongly correlated. For example the "GarageCars" and "GarageArea". This correlation can be explained logically. The number of cars that fit in the garage depends strongly on the area of the garage.

# <h4>Data Exploration Conclusion:</h4>
# <br>
# The first part of the analysis, the data exploration is essential for the whole analysis and the later following modeling steps. In this part we got to know the dataset. We investigated the distribution of the label, looked at all features separately, as well as at their relationship with the label. In this part we collected all the information that we needed to start the data manipulation. 

# <h2>Data Manipulation</h2>
# <br>
# The second part of this analysis is the data manipulation process. In this part we will correct the skewness of the label, cut outliers, impute missing values, create new features, turn some numeric features into categorical features, correct the skewness of some features and create dummy variables for the categorical features.

# <h4>Label Manipulation</h4>
# <br>
# As discussed in the data exploration part, we found that the SalePrice is positively skewed. This skewness can be corrected by using the logarithmic values of the SalePrice.

# In[ ]:


df_train["SalePrice"] = np.log1p(df_train["SalePrice"])


# In[ ]:


sns.distplot(df_train["SalePrice"], bins=30, fit=norm)
plt.show()


# In[ ]:


stats.probplot(df_train["SalePrice"], plot=plt)
plt.show()


# The histogram of the logarithmic SalePrice and the probability plot of the logarithmic SalePrice against the quantiles of the normal distribution indicate a less skewed distribution. We can see in the probability plot that specially the two smallest und two biggest SalePrices show a high deviation from the normal distribution. Therefore we drop the corresponding observations.

# In[ ]:


print(df_train["SalePrice"].sort_values().head())


# In[ ]:


print(df_train["SalePrice"].sort_values().tail())


# In[ ]:


df_train = df_train[df_train["SalePrice"] < 13.5]
df_train = df_train[df_train["SalePrice"] > 10.5]


# Finally, the probability plot indicates that the distribution of the logarithmic Sale Price is fairly normal.

# In[ ]:


stats.probplot(df_train["SalePrice"], plot=plt)
plt.show()


# We dropped four observations. Now the train dataset consists of 1456 oberservations.

# In[ ]:


df_train.shape


# <h4>Cutting Outliers</h4>
# <br>
# 
# The scatter plot of the feature "GrLivArea" against the SalePrice showed a linear relationship between both variables. We also identified some outliers in the graph. After changing the SalePrice to its logarithmic values, we want display the new scatter plot to choose the right limits to cut out the outliers. We can identify two clear outliers with values of "GrLivArea" of above 4000.

# In[ ]:


sns.regplot(x="GrLivArea", y="SalePrice", data=df_train);
plt.show()


# In[ ]:


df_train = df_train[df_train["GrLivArea"] < 4000]


# In[ ]:


sns.regplot(x="GrLivArea", y="SalePrice", data=df_train);
plt.show()


# After dropping these two observations, there remain 1454 oberservations in the train dataset.

# In[ ]:


df_train.shape


# <h4>Combining the train and test datasets for further manipulations</h4>
# <br>
# The next manipulation steps like imputing missing values and feature manipulation needs to be done for the train dataset, as well as for the test dataset. Therefore we concatenate both dataframes. The test dataframe contains of 1459 observations.

# In[ ]:


df_test.shape


# In[ ]:


df = pd.concat([df_train, df_test])


# The new concatenated dataframe contains 2913 observations.

# In[ ]:


df.shape


# <h4>Impute missing values</h4>
# <br>
# We will determine the features that exhibit missing values and visualize the percentage of missing values.

# In[ ]:


df_na = pd.DataFrame()
df_na["Feature"] = df.columns
missing = ((df.isnull().sum() / len(df)) * 100).values
df_na["Missing"] = missing
df_na = df_na[df_na["Feature"] != "SalePrice"]
df_na = df_na[df_na["Missing"] != 0]
df_na=df_na.sort_values(by="Missing", ascending=False)
print(df_na)


# In[ ]:


sns.barplot(x="Feature", y="Missing", data=df_na)
plt.xticks(rotation=90)
plt.show()


# The features with missing values can be divided in different groups for which we can apply the same imputing method.  The categorical features can be divided into two groups. For the first group of categorical features, the feature description explains that a missing values indicates that this features in not present. That is why the missing values for this group will be filled with the string "None".

# In[ ]:


df['Alley'] = df['Alley'].fillna('None')
df['MasVnrType'] = df['MasVnrType'].fillna('None')
df['BsmtQual'] = df['BsmtQual'].fillna('None')
df['BsmtCond'] = df['BsmtCond'].fillna('None')
df['BsmtExposure'] = df['BsmtExposure'].fillna('None')
df['BsmtFinType1'] = df['BsmtFinType1'].fillna('None')
df['BsmtFinType2'] = df['BsmtFinType2'].fillna('None')
df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
df['GarageType'] = df['GarageType'].fillna('None')
df['GarageFinish'] = df['GarageFinish'].fillna('None')
df['GarageQual'] = df['GarageQual'].fillna('None')
df['GarageCond'] = df['GarageCond'].fillna('None')
df['Fence'] = df['Fence'].fillna('None')
df['MiscFeature'] = df['MiscFeature'].fillna('None')
df['PoolQC'] = df['PoolQC'].fillna('None')


# There is a second group of categorical features, where the missing values do not indicate, that the feature is not present. The missing values will therefore be filled with the most common value, the mode.

# In[ ]:


df['MSZoning'] = df['MSZoning'].fillna('RL')
df["Exterior1st"] = df["Exterior1st"].fillna('VinylSd')
df["Exterior2nd"] = df["Exterior2nd"].fillna('VinylSd')
df['Electrical'] = df['Electrical'].fillna('SBrkr')
df['KitchenQual'] = df['KitchenQual'].fillna('TA')
df['Functional'] = df['Functional'].fillna('Typ')
df['SaleType'] = df['SaleType'].fillna('WD')
df['Utilities'] = df['Utilities'].fillna('AllPub')


# Also the numerical features can be divided into two groups for filling the missing values. The first group contains a numeric measure (e.g. number or area) of a feature that can be present or not present. Therefore missing values will be imputed by zero.

# In[ ]:


df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(0)
df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(0)
df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(0)
df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)
df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)
df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0)
df['GarageCars'] = df['GarageCars'].fillna(0)
df['GarageArea'] = df['GarageArea'].fillna(0)
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)


# There is one numeric feature left, where the missing values will be filled by the median values grouped by the neighborhood.

# In[ ]:


df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


# We check a second time for missing values. The dataframe df_na is empty, so there are no more missing values for the features.

# In[ ]:


df_na = pd.DataFrame()
df_na["Feature"] = df.columns
missing = ((df.isnull().sum() / len(df)) * 100).values
df_na["Missing"] = missing
df_na = df_na[df_na["Feature"] != "SalePrice"]
df_na = df_na[df_na["Missing"] != 0]
df_na=df_na.sort_values(by="Missing", ascending=False)
print(df_na)


# <h4>Create new features</h4>
# <br>
# We create one new feature "TotalSF", giving the total area of the house, by combing the areas of the basement, first and second floor, "TotalBsmtSF", "1stFlrSF" and "2ndFlrSF".

# In[ ]:


df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']


# <h4>Turn some numeric features into categorical features</h4>
# <br>
# During the data exploration of the numeric features, we found that there were some features that contained few discrete numeric values. These features will be turned into categorical features.

# In[ ]:


num_to_cat=["BedroomAbvGr", "BsmtFullBath", "BsmtHalfBath", "Fireplaces", "FullBath",
            "GarageCars", "HalfBath", "KitchenAbvGr", "MoSold", "MSSubClass", "OverallCond", 
            "OverallQual", "TotRmsAbvGrd", "YrSold"]

df[num_to_cat] = df[num_to_cat].apply(lambda x: x.astype("str"))


# We confirm the tranformation by illustrating the boxplots of these new categorical features.

# In[ ]:


def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
f = pd.melt(df_train, id_vars=['SalePrice'], value_vars=num_to_cat)
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "SalePrice")
plt.show()


# <h4>Correct Feature Skewness</h4>
# <br>
# For applying a linear regression to the dataset, also the numeric features are assumed to be normally distributed. We calculate the skewness of the numeric features. 

# In[ ]:


numeric_features = df.dtypes[df.dtypes != "object"].index
numeric_features = numeric_features.drop("SalePrice")

skew_before = df[numeric_features].apply(lambda x: skew(x.dropna()))
print(skew_before)


# We want to correct strong positively skewed distributions by taking their logarithmic values. To see if this operation lowers the skewness, we calculate the skew of the logarithmic values, take the difference before and after the transformation and then apply the logarithm only to those features, where the skewness decreased.

# In[ ]:


df_log = np.log1p(df[numeric_features])
skew_after = df_log[numeric_features].apply(lambda x: skew(x.dropna()))
skew_diff = (abs(skew_before)-abs(skew_after)).sort_values(ascending=False)
df[skew_diff[skew_diff>0].index] = np.log1p(df[skew_diff[skew_diff>0].index])
skew_new = df[numeric_features].apply(lambda x: skew(x.dropna()))

print(skew_new)


# <h4>Create Dummy Variables</h4>
# <br>
# We transform all categorical features to indicator variables, where "1" and "0" indicate if a category is present or not.

# In[ ]:


df = pd.get_dummies(df, drop_first=True)


# After the transformation the dataframe exhibits 23 numerical und 312 indicator variables.

# In[ ]:


df.info()


# <h4>Split data back into train and test data</h4>
# <br>
# After the dataset is cleaned, we need to divide the dataframe again into train and test data.

# In[ ]:


train = df.iloc[:1454]
test = df.iloc[1454:].drop("SalePrice", axis=1)


# In[ ]:


train.info()


# In[ ]:


test.info()


# <h3>Conclusion:</h3>
# <br>
# During this analysis we investigated and cleaned the test and train datasets of the Kaggle House Price Competion. The analysis consisted of two parts: the data exploration and the data manipulation. In the first part, we identified the label and the features, looked at their distributions and relationships. In the second part we cutted outliers, imputed missing values, created new features, corrected the skewness of the label and the features. The train and test datasets are cleaned and ready for modeling.

# <h3>Note:</h3>
# <br>
# Before I wrote this skript I read several good kernels to learn and to get ideas. Here are some which I found helpful:
# <br>
# <a href="https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python">"Comprehensive data exploration with Python"
#  by Pedro Marcelino</a>
# <br>
# <a href="https://www.kaggle.com/dgawlik/house-prices-eda">"House Prices EDA"
#  by Dominik Gawlik</a>
# <br>
# <a href="https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset">"A study on Regression applied to the Ames dataset" by juliencs</a>
# <br>
# <a href="https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard">"Stacked Regressions : Top 4% on LeaderBoard" by Serigne</a>
