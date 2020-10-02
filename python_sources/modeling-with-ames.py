#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#loading in necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


#loading data sets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


#printing dimensions of data sets
print ("Dimensions of training data:", train.shape)
print ("Dimensions of testing data:", test.shape)


# In[ ]:


#preview of training set
train.head()


# In[ ]:


#printing all integer type variables in the training set
print(train.dtypes[train.dtypes=='int64'])


# In[ ]:


#printing all float type variables in the training set
print(train.dtypes[train.dtypes=='float64'])


# In[ ]:


#printing all string types variables in training set
print(train.dtypes[train.dtypes=='object'])


# In[ ]:


#printing preview of testing set
test.head()


# In[ ]:


#printing all integer type variables in the testing set
print(test.dtypes[test.dtypes=='int64'])


# In[ ]:


#printing all float type variables in the testing set
print(test.dtypes[test.dtypes=='float64'])


# In[ ]:


#printing all string type variables in the testing set
print(test.dtypes[test.dtypes=='object'])


# # Data Processing

# __Outlier Detection__

# In[ ]:


#Checking to see if Null values present in training set

#Displaying ratio of missing variables to total number of rows in training data
training_data_missing_values_ratio = np.round(train.isnull().sum().loc[train.isnull().sum()>0,]/(len(train)) * 100.0,1)
training_data_missing_values_ratio = training_data_missing_values_ratio.reset_index()
training_data_missing_values_ratio.columns = ['column', 'ratio']
training_data_missing_values_ratio.sort_values(by=['ratio'], ascending=False)


# In[ ]:


#Checking to see if Null values present in testing set

#Displaying ratio of missing variables to total number of rows in testing data
testing_data_missing_values_ratio = np.round(test.isnull().sum().loc[test.isnull().sum()>0,]/(len(test)) * 100.0,1)
testing_data_missing_values_ratio = testing_data_missing_values_ratio.reset_index()
testing_data_missing_values_ratio.columns = ['column', 'ratio']
testing_data_missing_values_ratio.sort_values(by=['ratio'], ascending=False)


# In[ ]:


#Displaying a few plots with Sale Price
f, ax = plt.subplots(figsize=(8, 4))
sns.regplot(x=train.GrLivArea, y=train.SalePrice)
plt.title("GrLivArea vs SalePrice")


# In[ ]:


#Noticed a few outliers, so I will be dropping them from our data set
train = train.drop(train.loc[(train.GrLivArea>4000) & (train.SalePrice < 200000),].index)

f, ax = plt.subplots(figsize=(8, 4))
sns.regplot(x=train.GrLivArea, y=train.SalePrice)
plt.title("GrLivArea vs SalePrice")


# In[ ]:


#PLotting distribution of Sale Price
f, ax = plt.subplots(figsize=(10, 6))
sns.distplot(train.SalePrice)


# In[ ]:


#Heavily right skewed
from scipy import stats
f, ax = plt.subplots(figsize=(10, 6))
stats.probplot(train.SalePrice, plot=plt)


# In[ ]:


#kurtosis validates that the distribution is not normal
from scipy.stats import kurtosis
kurtosis(train.SalePrice)


# In[ ]:


#Take the logarithm of Sale Price to get a more normal distribution
f, ax = plt.subplots(figsize=(10, 6))
sns.distplot(np.log(train.SalePrice))


# In[ ]:


#QQ-Plot appears more normal
f, ax = plt.subplots(figsize=(10, 6))
stats.probplot(np.log(train.SalePrice), plot=plt)


# In[ ]:


#kurtosis is closer to 0, it appears more normal
kurtosis(np.log(train.SalePrice))


# __EDA__

# In[ ]:


#Getting all variables that have numerical values
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.columns


# In[ ]:


#Getting all variables that have string values
categorical_features = train.select_dtypes(include=[np.object])
categorical_features.columns


# In[ ]:


#Examining the correlation between all numerical values with Sale Price
numeric_features.corr()['SalePrice'].sort_values(ascending = False)


# In[ ]:


#Correlation Matrix
corr = train.iloc[:,1:].corr()
f, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap = "YlGnBu")
plt.title("Correlation Matrix")


# In[ ]:


#Looking at highly correlated variables 
high_corr_cols = numeric_features.corr()['SalePrice'].sort_values(ascending = False)[numeric_features.corr()[
    'SalePrice'].sort_values(ascending = False)>0.5].index.tolist()
high_corr_cols


# In[ ]:


#Heat map of highly correlated variables with Sale Price
corr = train[high_corr_cols].corr()
f, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
            annot=True,
        cmap = "YlGnBu")
plt.title("Correlation Matrix")


# In[ ]:


#Imputing missing values in categorical variables for plotting purposes
for cols in categorical_features:
    train[cols] = train[cols].astype('category')
    if train[cols].isnull().any():
        train[cols] = train[cols].cat.add_categories(['MISSING'])
        train[cols] = train[cols].fillna('MISSING')

#Mass plotting of categorical variables to see if relationships with Sale Price exist
def mass_boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)
f = pd.melt(train, id_vars=['SalePrice'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(mass_boxplot, "value", "SalePrice")


# In[ ]:


train2 = train
#some numerical values that are actually categorical values
cols_to_conv_to_categorical = ["MSSubClass", "OverallQual", "OverallCond", "MoSold", "YrSold"]
for cols in cols_to_conv_to_categorical:
    train2[cols] = train2[cols].apply(str)

#plotting many boxplots to see relationship with Sale Price
def mass_boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)

f = pd.melt(train2, id_vars=['SalePrice'], value_vars=cols_to_conv_to_categorical)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(mass_boxplot, "value", "SalePrice")


# In[ ]:


#plotting more categorical variables with Sale Price
for cols in ["YearRemodAdd", "YearBuilt", "GarageYrBlt"]: 
    data = pd.concat([train2['SalePrice'], train2[cols]], axis=1)
    f, ax = plt.subplots(figsize=(40, 20))
    fig = sns.boxplot(x=cols, y="SalePrice", data=data)
    plt.xticks(rotation=90, fontsize = 20)
    title_name = "SalePrice Across " + cols
    plt.title(str(title_name), fontsize = 25)


# There definitely appears to be an increasing relationship between the overall quality of the house and its sale price.

# In[ ]:


#Taking a closer look at the distribution of Sale Price across OverAllQual
data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)
data["OverallQual"] = data["OverallQual"].apply(int)
f, ax = plt.subplots(figsize=(12, 8))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.title("SalePrice vs. OverAllQual")


# In[ ]:


#getting the order of neighborhood based by median sale price
neighborhood_order = train[['Neighborhood','SalePrice']].groupby([
    'Neighborhood']).describe()['SalePrice']['50%'].sort_values(ascending=True).index


# In[ ]:


#Plotting Sale Price across Neighborhoods
data = pd.concat([train['SalePrice'], train['Neighborhood']], axis=1)
f, ax = plt.subplots(figsize=(20, 12))
fig = sns.boxplot(x='Neighborhood', y="SalePrice", data=data, order = neighborhood_order)
fig.axis(ymin=0, ymax=800000);
plt.title("SalePrice Across Neighborhoods")


# After the boxplots, I saw many any relationships in the ordering of the categorical variables. So I converted the following variables into ordinal and examined how their relationship with SalePrice will perform in modeling.

# In[ ]:


train_feature_check = pd.read_csv('../input/train.csv')
train_feature_check = train_feature_check.drop(train_feature_check.loc[(train_feature_check.GrLivArea>4000) & (train_feature_check.SalePrice < 200000),].index)
print("Original dimension:", train_feature_check.shape)


# In[ ]:


def conv_to_numeric(df, column_list, mapper):
    for cols in column_list:
        df[str("o")+cols] = df[cols].map(mapper)
        df[str("o")+cols].fillna(0, inplace = True)         


# In[ ]:


#remapping categorical variables with numerical values
convert_col_1 = ["ExterQual", "ExterCond","BsmtQual", "BsmtCond", "HeatingQC","KitchenQual","FireplaceQu",
                  "GarageQual","GarageCond","PoolQC"]
mapper = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None':0}

conv_to_numeric(train_feature_check, convert_col_1, mapper)

convert_col_2 = ["BsmtExposure"]
mapper = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None':0}
conv_to_numeric(train_feature_check, convert_col_2, mapper)

convert_col_3 = ["BsmtFinType1", "BsmtFinType2"]
mapper = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1 ,'None':0}
conv_to_numeric(train_feature_check, convert_col_3, mapper)

convert_col_4 = ["GarageFinish"]
mapper = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None':0}
conv_to_numeric(train_feature_check, convert_col_4, mapper)

convert_col_5 = ["Fence"]
mapper = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'None':0}
conv_to_numeric(train_feature_check, convert_col_5, mapper)


# In[ ]:


convert_col = convert_col_1 + convert_col_2 + convert_col_3 + convert_col_4 + convert_col_5
new_features_col=[]

for cols in convert_col:
    cols = str("o")+cols
    new_features_col.append(cols)


# In[ ]:


#plotting variables
for cols in new_features_col:
    data = pd.concat([train_feature_check['SalePrice'], train_feature_check[cols]], axis=1)
    f, ax = plt.subplots(figsize=(12, 8))
    fig = sns.boxplot(x=cols, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)


# In[ ]:


#created new variable TotalSqFt 
train_feature_check["TotalSqFt"] = train_feature_check["TotalBsmtSF"] + train_feature_check["1stFlrSF"] + train_feature_check["2ndFlrSF"]
new_features_col.append("TotalSqFt")
new_features_col.append("SalePrice")


# In[ ]:


print("Training dimension:", train.shape)
print("Training with new features dimension:", train_feature_check.shape)


# In[ ]:


#correlation matrix with new features
corr = train_feature_check[new_features_col].corr()
f, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        annot=True,
        cmap = "YlGnBu")
plt.title("Correlation Matrix for New Features")


# In[ ]:


#combining training and testing set to address missing values
train_row = train.shape[0]
test_row = test.shape[0]
y = train["SalePrice"]
all_data = pd.concat([train,test]).reset_index(drop=True)
all_data = all_data.drop(['SalePrice'], axis = 1)
print("Size of concatenated train and test datasets:", all_data.shape)


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


data_missing_values_ratio = np.round(
    all_data.isnull().sum().loc[all_data.isnull().sum()>0,]/(len(all_data)) * 100.0,3
)
data_missing_values_counts = all_data.isnull().sum().loc[all_data.isnull().sum()>0,].reset_index()
data_missing_values_counts.columns = ['column', 'counts']

data_missing_values_ratio = data_missing_values_ratio.reset_index()
data_missing_values_ratio.columns = ['column', 'ratio']

data_missing_values = pd.merge(left = data_missing_values_ratio , right = data_missing_values_counts, on = 'column', how='inner')
data_missing_values = data_missing_values.sort_values(by=['ratio'], ascending=False).reset_index(drop=True)
data_missing_values


# In[ ]:


ind = np.arange(data_missing_values.shape[0])
width = 0.1
fig, ax = plt.subplots(figsize=(20,8))
rects = ax.barh(ind, data_missing_values.counts.values[::-1], color='b')
ax.set_yticks(ind)
ax.set_yticklabels(data_missing_values.column.values[::-1], rotation='horizontal')
ax.set_xlabel("Missing Observations Count")
ax.set_title("Missing Observations Counts")
plt.show()


# Majority of the imputation will be based on the data dictionary. Some will require some assumptions and intuition.

# __PoolQC:__ Pool Quality. The data description mentions NA which means "No Pool". We will replace these NA values with "None".

# In[ ]:


all_data["PoolQC"] = all_data["PoolQC"].fillna("None")


# __MiscFeature:__ Miscellaneous feature not covered in other categories. The data description mentions NA which means "None". We will replace these NA values with "None".

# In[ ]:


all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")


# __Alley:__ Type of alley access to property. The data description mentions NA which means "None". We will replace these NA values with "None".

# In[ ]:


all_data["Alley"] = all_data["Alley"].fillna("None")


# __Fence__: Describes quality of fence. The data description mentions NA which means "No Fence". We will replace these NA values with "None".
#     

# In[ ]:


all_data["Fence"] = all_data["Fence"].fillna("None")


# __FireplaceQu__: Describes quality of the Fireplace. The data description mentions NA which means "No Fireplace". We will replace these NA values with "None".
#     

# In[ ]:


all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")


# __LotFrontage__: Linear feet of street connected to property. We will replace the NA values with the median value.

# In[ ]:


all_data["LotFrontage"].describe()


# In[ ]:


all_data["LotFrontage"] = all_data["LotFrontage"].fillna(all_data["LotFrontage"].describe()['50%'])


# __GarageCond__: Garage condition. The data description mentions NA which means "No Garage". We will replace these NA values with "None".

# In[ ]:


all_data["GarageCond"] = all_data["GarageCond"].fillna("None")


# __GarageQual__: Garage quality. The data description mentions NA which means "No Garage". We will replace these NA values with "None".

# In[ ]:


all_data["GarageQual"] = all_data["GarageQual"].fillna("None")


# __GarageYrBlt:__ Year garage was built. We will assume that these houses do not have a garage. We will replace these NA values with "None". I also noticed an outlier in the year built for a garage. The year 2207 is an invalid date. I replaced this year with the year that the house was built in.

# In[ ]:


np.sort(all_data["GarageYrBlt"].unique().tolist())


# In[ ]:


all_data.loc[all_data["GarageYrBlt"] > 2016,]["GarageYrBlt"] 


# In[ ]:


all_data.loc[(all_data["GarageYrBlt"] > 2016),]["YearBuilt"]


# In[ ]:


all_data.loc[all_data["GarageYrBlt"] > 2016,"GarageYrBlt"]  =  all_data.loc[(all_data["GarageYrBlt"] > 2016),"YearBuilt"]


# In[ ]:


all_data["GarageYrBlt"] = all_data["GarageYrBlt"].fillna(0)


# __GarageFinish__: Interior finish of the garage. The data description mentions NA which means "No Garage". We will replace these NA values with "None".
# 
# 

# In[ ]:


all_data["GarageFinish"] = all_data["GarageFinish"].fillna("None")


# __GarageType:__ Garage location. The data description mentions NA which means "No Garage". We will replace these NA values with "None".

# In[ ]:


all_data["GarageType"] = all_data["GarageType"].fillna("None")


# __BsmtExposure:__ Refers to walkout or garden level walls. The data description mentions NA which means "No Basement". We will replace these NA values with "None".

# In[ ]:


all_data["BsmtExposure"] = all_data["BsmtExposure"].fillna("None")


# __BsmtCond:__ Evaluates the general condition of the basement. The data description mentions NA which means "No Basement". We will replace these NA values with "None".

# In[ ]:


all_data["BsmtCond"] = all_data["BsmtCond"].fillna("None")


# __BsmtQual:__ Evaluates the height of the basement. The data description mentions NA which means "No Basement". We will replace these NA values with "None".

# In[ ]:


all_data["BsmtQual"] = all_data["BsmtQual"].fillna("None")


# __BsmtFinType2:__ Rating of basement finished area (if multiple types). The data description mentions NA which means "No Basement". We will replace these NA values with "None".

# In[ ]:


all_data["BsmtFinType2"] = all_data["BsmtFinType2"].fillna("None")


# __BsmtFinType1:__ Rating of basement finished area. The data description mentions NA which means "No Basement". We will replace these NA values with "None".

# In[ ]:


all_data["BsmtFinType1"] = all_data["BsmtFinType1"].fillna("None")


# __MasVnrType:__ Masonry veneer type. The data description mentions NA which means "No Masonry veneer". We will replace these NA values with "None".

# In[ ]:


all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")


# __MasVnrArea:__ Masonry veneer area in square feet. We will replace the NA values with the median value.

# In[ ]:


all_data["MasVnrArea"].describe()


# In[ ]:


all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(all_data["MasVnrArea"].describe()['50%'])


# __MSZoning:__ Identifies the general zoning classification of the sale. We will impute the missing zones with 'RL' as it the most frequently occuring zoning classification.

# In[ ]:


all_data["MSZoning"].value_counts()


# In[ ]:


all_data["MSZoning"] = all_data["MSZoning"].fillna(all_data["MSZoning"].value_counts()[0])


# __Utilities:__ Type of utilities available. As we can see the distribution of type of utilities do not differentiate except for one "NoSeWa" and two NA values. This variable does not provide meaningful information in predicting SalePrice, so I will be dropping this variable.

# In[ ]:


all_data["Utilities"].value_counts()


# In[ ]:


all_data = all_data.drop(["Utilities"], axis = 1)


# __Functional:__ Home functionality (Assume typical unless deductions are warranted). We will impute the missing functional with 'Typ' as it is the most frequently occuring type of Home functionality.

# In[ ]:


all_data["Functional"].value_counts()


# In[ ]:


all_data["Functional"] = all_data["Functional"].fillna(all_data["Functional"].value_counts()[0])


# __BsmtHalfBath:__ Basement half bathrooms. We will assume that these houses have missing values because there are no basements so we will replace these NA values with zero.

# In[ ]:


all_data["BsmtHalfBath"] = all_data["BsmtHalfBath"].fillna(0)


# __BsmtFullBath:__ Basement full bathrooms. We will assume that these houses have missing values because there are no basements so we will replace these NA values with zero.

# In[ ]:


all_data["BsmtFullBath"] = all_data["BsmtFullBath"].fillna(0)


# __GarageCars:__ Size of garage in car capacity. We will assume that these houses have missing values because there are  no garages so we will replace these NA values with zero.

# In[ ]:


all_data["GarageCars"] = all_data["GarageCars"].fillna(0)


# __Exterior2nd:__ Exterior covering on house (if more than one material). We will impute the missing exterior convering type with 'VinylSd' as it the most frequently occuring exterior covering on house (if more than one material).
# 

# In[ ]:


all_data["Exterior2nd"].value_counts()


# In[ ]:


all_data["Exterior2nd"] = all_data["Exterior2nd"].fillna(all_data["Exterior2nd"].value_counts()[0])


# __Exterior1st:__ Exterior covering on house. We will impute the missing exterior convering type with 'VinylSd' as it the most frequently occuring exterior covering on house.
# 

# In[ ]:


all_data["Exterior1st"].value_counts()


# In[ ]:


all_data["Exterior1st"] = all_data["Exterior1st"].fillna(all_data["Exterior1st"].value_counts()[0])


# __KitchenQual:__ Kitchen quality. We will impute the missing kitchen quality with 'TA' as it the most frequently occuring kitchen quality.
# 

# In[ ]:


all_data["KitchenQual"].value_counts()


# In[ ]:


all_data["KitchenQual"] = all_data["KitchenQual"].fillna(all_data["KitchenQual"].value_counts()[0])


# __Electrical:__ Electrical system.  We will impute the missing kitchen quality with 'SBrkr' as it the most frequently occuring electrical system.
# 

# In[ ]:


all_data["Electrical"].value_counts()


# In[ ]:


all_data["Electrical"] = all_data["Electrical"].fillna(all_data["Electrical"].value_counts()[0])


# __BsmtUnfSF:__ Unfinished square feet of basement area. We will assume that these houses have missing values because there are no basements so we will replace these NA values with zero.
# 

# In[ ]:


all_data["BsmtUnfSF"] = all_data["BsmtUnfSF"].fillna(0)


# __BsmtFinSF2:__ Type 2 finished square feet. We will assume that these houses have missing values because there are no basements so we will replace these NA values with zero.
# 

# In[ ]:


all_data["BsmtFinSF2"] = all_data["BsmtFinSF2"].fillna(0)


# __BsmtFinSF1:__ Type 1 finished square feet. We will assume that these houses have missing values because there are no basements so we will replace these NA values with zero.
# 

# In[ ]:


all_data["BsmtFinSF1"] = all_data["BsmtFinSF1"].fillna(0)


# __SaleType:__ Type of sale. We will impute the missing type of sale with 'WD' as it the most frequently occuring type of sale.

# In[ ]:


all_data["SaleType"].value_counts()


# In[ ]:


all_data["SaleType"] = all_data["SaleType"].fillna(all_data["SaleType"].value_counts()[0])


# __TotalBsmtSF:__ Total square feet of basement area. We will assume that these houses have missing values because there are no basements so we will replace these NA values with zero.

# In[ ]:


all_data["TotalBsmtSF"] = all_data["TotalBsmtSF"].fillna(0)


# __GarageArea:__ Size of garage in square feet. We will assume that these houses have missing values because there are no garages so we will replace these NA values with zero.

# In[ ]:


all_data["GarageArea"] = all_data["GarageArea"].fillna(0)


# In[ ]:


#Check to see if any other missing values
all_data.isnull().sum().loc[all_data.isnull().sum()>0,]


# In[ ]:


print("All Data Dimensions:", all_data.shape)


# In[ ]:


all_data["TotalSqFt"] = all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]


# In[ ]:


#remapping categorical variables for training and testing set
convert_col_1 = ["ExterQual", "ExterCond","BsmtQual", "BsmtCond", "HeatingQC","KitchenQual","FireplaceQu",
                  "GarageQual","GarageCond","PoolQC"]
mapper = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None':0}

conv_to_numeric(all_data, convert_col_1, mapper)

convert_col_2 = ["BsmtExposure"]
mapper = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None':0}
conv_to_numeric(all_data, convert_col_2, mapper)

convert_col_3 = ["BsmtFinType1", "BsmtFinType2"]
mapper = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1 ,'None':0}
conv_to_numeric(all_data, convert_col_3, mapper)

convert_col_4 = ["GarageFinish"]
mapper = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None':0}
conv_to_numeric(all_data, convert_col_4, mapper)

convert_col_5 = ["Fence"]
mapper = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'None':0}
conv_to_numeric(all_data, convert_col_5, mapper)


# In[ ]:


print("All Data New Dimensions:", all_data.shape)


# Some variables are not labeled as categorical variables, even when they are intended to be so I will be fixing those variables.

# In[ ]:


cols_to_conv_to_categorical = cols_to_conv_to_categorical + ["YearRemodAdd", "YearBuilt", "GarageYrBlt"]

for cols in cols_to_conv_to_categorical:
    all_data[cols] = all_data[cols].astype(str)


# In[ ]:


from scipy.stats import skew
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# In[ ]:


#Displaying an example of skewness in data
sns.distplot(all_data["LotArea"])


# In[ ]:


from scipy.stats import skew
skew(all_data["LotArea"])


# In[ ]:


#After applying the Box Cox Transformation, we eliminate majority of the skewness and normalize the variable.
from scipy.special import boxcox1p
all_data_LotArea_boxcox_transform = boxcox1p(all_data["LotArea"], 0.15)
skew(all_data_LotArea_boxcox_transform)


# In[ ]:


sns.distplot(all_data_LotArea_boxcox_transform)


# In[ ]:


#Apply this to all features that exhibit skewness of over 
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)


# In[ ]:


all_data = pd.get_dummies(all_data)
print(all_data.shape)


# In[ ]:


ntrain = all_data[:train_row]
ntest = all_data[train_row:]


# In[ ]:


ntrain = ntrain.drop(['Id'], axis = 1)
ntest = ntest.drop(['Id'], axis = 1)


# Modeling

# In[ ]:


#import required packages
from sklearn.linear_model import ElasticNet, Lasso,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor, plot_importance
import time
from mlxtend.regressor import StackingCVRegressor

RANDOM_SEED = 1


# In[ ]:


#defining number of folds
n_folds = 5

def cross_val_rmse(model):
    """This function will be used to perform cross validation and gather the average RMSE across five-folds for a model"""
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(ntrain.values)
    rmse= np.sqrt(-cross_val_score(model, ntrain.values, np.log(y).values, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# Getting a benchmark for each model

# In[ ]:


models = [
          Ridge(alpha=0.5, random_state=RANDOM_SEED),
          ElasticNet(alpha=0.005, random_state=RANDOM_SEED),
          Lasso(alpha = 0.005, random_state=RANDOM_SEED),
          XGBRegressor(random_state=RANDOM_SEED)
         ]
model_name = ["Ridge","ElasticNet","Lasso","XGBoost"]

for name, model in zip(model_name, models):
    model_test = make_pipeline(RobustScaler(), model)
    score = cross_val_rmse(model_test)
    print(name, ": {:.4f} ({:.4f})".format(score.mean(), score.std()))


# In[ ]:


def grid_search_function(func_X_train, func_X_test, func_y_train, func_y_test, parameters, model):
    grid_search = GridSearchCV(model, parameters,  scoring='neg_mean_squared_error')
    regressor = grid_search.fit(func_X_train,func_y_train)
    return regressor


# In[ ]:


def train_test_split_function(X,y, test_size_percent):
    """Fucntion to perform train_test_split"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percent, random_state=RANDOM_SEED)
    return X_train, X_test, y_train, y_test


# In[ ]:


starttime = time.monotonic()
parameters = {'ridge__alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20], 'ridge__random_state':[RANDOM_SEED],
             'ridge__max_iter':[100000000]}
pipe = Pipeline(steps=[('rscale',RobustScaler()), ('ridge',Ridge())])
X_train, X_test, y_train, y_test = train_test_split(ntrain, 
                                                             np.log(y), 
                                                             test_size = 0.20,
                                                             random_state=RANDOM_SEED
                                                            )

ridge_regressor = grid_search_function(X_train, X_test, y_train, y_test, 
                                     parameters, 
                                     model = pipe)

print("That took ", (time.monotonic()-starttime)/60, " minutes")

print("\nBest Params:",ridge_regressor.best_estimator_)

print("\nBest Score:",np.sqrt(-ridge_regressor.best_score_))


# In[ ]:


ridge_regressor.best_estimator_.steps


# In[ ]:


starttime = time.monotonic()
parameters = {}
pipe = Pipeline(steps=[('ridge',Ridge(alpha=10, copy_X=True, fit_intercept=True, max_iter=100000000, normalize=False, random_state=1, solver='auto', tol=0.001))])

X_train, X_test, y_train, y_test = train_test_split(ntrain, 
                                                             np.log(y), 
                                                             test_size = 0.20,
                                                             random_state=RANDOM_SEED
                                                            )
ridge_model = grid_search_function(X_train, X_test, y_train, y_test, 
                                     parameters, 
                                     model = pipe)

print("That took ", (time.monotonic()-starttime)/60, " minutes")

print("\nBest Params:",ridge_model.best_estimator_)

print("\nBest Score:",np.sqrt(-ridge_model.best_score_))


# In[ ]:


starttime = time.monotonic()
parameters = {}
pipe = Pipeline(steps=[('enet',ElasticNet(alpha=0.0005, copy_X=True, fit_intercept=True, l1_ratio=0.5,
      max_iter=100000000, normalize=False, positive=False,
      precompute=False, random_state=1, selection='cyclic', tol=0.0001,
      warm_start=False))])
X_train, X_test, y_train, y_test = train_test_split(ntrain, 
                                                             np.log(y), 
                                                             test_size = 0.20,
                                                             random_state=RANDOM_SEED
                                                            )

enet_model = grid_search_function(X_train, X_test, y_train, y_test, 
                                     parameters, 
                                     model = pipe)

print("That took ", (time.monotonic()-starttime)/60, " minutes")

print("\nBest Params:",enet_model.best_estimator_)

print("\nBest Score:",np.sqrt(-enet_model.best_score_))


# In[ ]:


starttime = time.monotonic()
parameters = {}
pipe = Pipeline(steps=[('lasso',Lasso(alpha=0.0005, copy_X=True, fit_intercept=True, max_iter=100000000,
   normalize=False, positive=False, precompute=False, random_state=1,
   selection='cyclic', tol=0.0001, warm_start=False))])

X_train, X_test, y_train, y_test = train_test_split(ntrain, 
                                                             np.log(y), 
                                                             test_size = 0.20,
                                                             random_state=RANDOM_SEED
                                                            )

lasso_model = grid_search_function(X_train, X_test, y_train, y_test, 
                                     parameters, 
                                     model = pipe)

print("That took ", (time.monotonic()-starttime)/60, " minutes")

print("\nBest Params:",lasso_model.best_estimator_)

print("\nBest Score:",np.sqrt(-lasso_model.best_score_))


# In[ ]:


##UNCOMMENT to run

#starttime = time.monotonic()
#parameters = {'xgb__random_state':[RANDOM_SEED],
#             'xgb__gamma':[0,0.1], 
#              'xgb__learning_rate':[0.01,0.05,0.1],
#             'xgb__n_jobs':[-1], 
#              'xgb__n_estimators':[500,1000,2000],
#             'xgb__reg_lambda':[0,0.5,1],
#              'xgb__reg_alpha':[0,0.5,1]
#              }
#
#pipe = Pipeline(steps=[('xgb',XGBRegressor())])
#X_train, X_test, y_train, y_test = train_test_split(ntrain, 
#                                                             np.log(y), 
#                                                             test_size = 0.20,
# random_state=RANDOM_SEED)
#
#xgb_regressor = grid_search_function(X_train, X_test, y_train, y_test, 
#                                     parameters, 
#                                     model = pipe)
#
#print("That took ", (time.monotonic()-starttime)/60, " minutes")
#
#print("\nBest Params:",xgb_regressor.best_estimator_)
#
#print("\nBest Score:",np.sqrt(-xgb_regressor.best_score_))

#####Output###
####That took  66.38542943511857  minutes
####
####Best Params: Pipeline(memory=None,
####     steps=[('xgb', XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
####       colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
####       max_depth=3, min_child_...
####       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
####       silent=True, subsample=1))])
####
####Best Score: 0.12259807102070201
##
###xgb_regressor.best_estimator_.steps
####[
#### ('xgb', XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
####         colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
####         max_depth=3, min_child_weight=1, missing=None, n_estimators=1000,
####         n_jobs=-1, nthread=None, objective='reg:linear', random_state=1,
####         reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
####         silent=True, subsample=1))]


# In[ ]:


starttime = time.monotonic()
parameters = {}
pipe = Pipeline(steps=[
 ('xgb', XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
         colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
         max_depth=3, min_child_weight=1, missing=None, n_estimators=1000,
         n_jobs=-1, nthread=None, objective='reg:linear', random_state=1,
         reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
         silent=True, subsample=1))])

X_train, X_test, y_train, y_test = train_test_split(ntrain, 
                                                             np.log(y), 
                                                             test_size = 0.20,
                                                             random_state=RANDOM_SEED
                                                            )

xgb_regressor = grid_search_function(X_train, X_test, y_train, y_test, 
                                     parameters, 
                                     model = pipe)

print("That took ", (time.monotonic()-starttime)/60, " minutes")

print("\nBest Params:",xgb_regressor.best_estimator_)

print("\nBest Score:",np.sqrt(-xgb_regressor.best_score_))


# Best Models

# In[ ]:


Ridge_model = Ridge(alpha=10, copy_X=True, fit_intercept=True, max_iter=100000000,
   normalize=False, random_state=1, solver='auto', tol=0.001)

Enet_model = ElasticNet(alpha=0.0005, copy_X=True, fit_intercept=True, l1_ratio=0.5,
      max_iter=100000000, normalize=False, positive=False,
      precompute=False, random_state=1, selection='cyclic', tol=0.0001,
      warm_start=False)

lasso_model = Lasso(alpha=0.0005, copy_X=True, fit_intercept=True, max_iter=100000000,
   normalize=False, positive=False, precompute=False, random_state=1,
   selection='cyclic', tol=0.0001, warm_start=False)

xgb_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
         colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
         max_depth=3, min_child_weight=1, missing=None, n_estimators=1000,
         n_jobs=-1, nthread=None, objective='reg:linear', random_state=1,
         reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
         silent=True, subsample=1)


# In[ ]:


#running a stacking model
regression_stacker = StackingCVRegressor(regressors = [
    Enet_model, Ridge_model, xgb_model],
                                         meta_regressor = lasso_model,
                                         cv=3)

score = cross_val_rmse(regression_stacker)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# __Scoring Models__

# In[ ]:


Ridge_model.fit(ntrain.values, np.log(y).values)
y_pred = Ridge_model.predict(ntest.values)
y_train_pred =  Ridge_model.predict(ntrain.values)
exp_y_pred = np.exp(y_pred)
submission = pd.read_csv("../input/sample_submission.csv")
submission.SalePrice = exp_y_pred
submission.to_csv("ridge_submission.csv",index = False)
print("Training Score:", np.sqrt(mean_squared_error(np.log(y).values, y_train_pred)))


# In[ ]:


#Examining magnitudes of features for Ridge regression
predictors = ntrain.columns

coef = pd.Series(Ridge_model.coef_,predictors).sort_values()
coef2 = coef[coef!=0]
f, ax = plt.subplots(figsize=(60, 20))
coef2.plot(kind='bar', title='Model Coefficients')


# In[ ]:


Enet_model.fit(ntrain.values, np.log(y).values)
y_pred = Enet_model.predict(ntest.values)
y_train_pred =  Enet_model.predict(ntrain.values)
exp_y_pred = np.exp(y_pred)
submission = pd.read_csv("../input/sample_submission.csv")
submission.SalePrice = exp_y_pred
submission.to_csv("elastic_net_submission.csv",index = False)
print("Training Score:", np.sqrt(mean_squared_error(np.log(y).values, y_train_pred)))


# In[ ]:


#Examining magnitudes of features for Elastic Net regression
predictors = ntrain.columns

coef = pd.Series(Enet_model.coef_,predictors).sort_values()
coef2 = coef[coef!=0]
f, ax = plt.subplots(figsize=(30, 12))
coef2.plot(kind='bar', title='Model Coefficients')


# In[ ]:


lasso_model.fit(ntrain.values, np.log(y).values)
y_pred = lasso_model.predict(ntest.values)
y_train_pred =  lasso_model.predict(ntrain.values)
exp_y_pred = np.exp(y_pred)
submission = pd.read_csv("../input/sample_submission.csv")
submission.SalePrice = exp_y_pred
submission.to_csv("lasso_submission.csv",index = False)
print("Training Score:", np.sqrt(mean_squared_error(np.log(y).values, y_train_pred)))


# In[ ]:


#Examining magnitudes of features for Lasso regression
predictors = ntrain.columns

coef = pd.Series(lasso_model.coef_,predictors).sort_values()
coef2 = coef[coef!=0]
f, ax = plt.subplots(figsize=(30, 12))
coef2.plot(kind='bar', title='Model Coefficients')


# In[ ]:


xgb_model.fit(ntrain.values, np.log(y).values)
y_pred = xgb_model.predict(ntest.values)
y_train_pred =  xgb_model.predict(ntrain.values)
exp_y_pred = np.exp(y_pred)
submission = pd.read_csv("../input/sample_submission.csv")
submission.SalePrice = exp_y_pred
submission.to_csv("xgboost_submission.csv",index = False)
print("Training Score:", np.sqrt(mean_squared_error(np.log(y).values, y_train_pred)))


# In[ ]:


#Examining feature importance (most important features) of XGBoost
xgb_feature_importance_df = pd.DataFrame({"column_names":ntrain.columns, "feature_importance": xgb_model.feature_importances_})
xgb_feature_importance_filtered_df = xgb_feature_importance_df.loc[xgb_feature_importance_df.feature_importance>0.003].sort_values('feature_importance',ascending=False)
f, ax = plt.subplots(figsize=(20, 12))
sns.barplot(x="feature_importance", y="column_names", data=xgb_feature_importance_filtered_df)


# In[ ]:


regression_stacker.fit(ntrain.values, np.log(y).values)
y_pred = regression_stacker.predict(ntest.values)
y_train_pred =  regression_stacker.predict(ntrain.values)
exp_y_pred = np.expm1(y_pred)
submission = pd.read_csv("../input/sample_submission.csv")
submission.SalePrice = exp_y_pred
submission.to_csv("stacking_exp1m_submission.csv",index = False)
print("Training Score:", np.sqrt(mean_squared_error(np.log(y).values, y_train_pred)))

