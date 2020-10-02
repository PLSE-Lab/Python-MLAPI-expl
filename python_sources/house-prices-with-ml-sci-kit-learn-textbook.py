#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Loading training data
AUTOPATH = "../input"

def load_training_data():
    csv_path = os.path.join(AUTOPATH, "train.csv")
    return csv_path

def load_testing_data():
    csv_path = os.path.join(AUTOPATH, "test.csv")
    return csv_path

training_path = load_training_data()


# In[ ]:


train = pd.read_csv(training_path)
train.head()


# In[ ]:


train.info()


# In[ ]:


sns.heatmap(train.isnull())


# In[ ]:


null_cols = train.columns[train.isnull().any()].tolist()
for col in null_cols:
    print(str(col),": ", sum(train[col].isnull()))


# ## Fixing 'LotFrontage' Column

# In[ ]:


train["LotFrontage"].describe()


# In[ ]:


train["LotFrontage"].fillna(train["LotFrontage"].mean(), inplace=True)
sns.distplot(train["LotFrontage"])


# In[ ]:


drop_cols = ["PoolQC", "Id", "Alley", "Fence", "MiscFeature", "FireplaceQu"]


# ## Fixing 'Alley' Column
# It seems that NaN in the 'Alley' column is supposed to represent "No alley access", which makes sense since most houses are close to or adjacent to another house.

# In[ ]:


def getValsUniq(col):
    print(train[col].value_counts())
    print(train[col].unique())


# In[ ]:


def fillAlley(a):
    if pd.isnull(a):
        return "NA"
    else:
        return a
    
train["Alley"] = train["Alley"].apply(fillAlley)
train["Alley"].value_counts()


# ## Fixing 'MasVnrType' and 'MasVnrArea' Columns
# Assuming that NaN is for None (None is also the mode).

# In[ ]:


getValsUniq("MasVnrType")


# In[ ]:


train["MasVnrType"].fillna("TEST", inplace=True)


# In[ ]:


train[train["MasVnrType"] == "TEST"]["MasVnrArea"]


# In[ ]:


train["MasVnrType"].replace({"TEST": "None"})
train["MasVnrArea"].fillna(0, inplace=True)
print(train["MasVnrType"].isnull().value_counts())
print(train["MasVnrArea"].isnull().value_counts())


# ## Fixing 'BsmtQual' and 'BsmtCond'

# In[ ]:


getValsUniq("BsmtQual")


# In[ ]:


getValsUniq("BsmtCond")


# In[ ]:


train["BsmtQual"].fillna("TEST", inplace=True)


# In[ ]:


train[train["BsmtQual"] == "TEST"]["BsmtCond"].unique()


# Going to assume that NaN is for No basement since all NaN values in BsmtQual associate with NaN in BsmtCond where there is no basement.

# In[ ]:


train["BsmtQual"].replace({"TEST": "NA"}, inplace=True)
train["BsmtCond"].fillna("NA", inplace=True)
print(train["BsmtQual"].value_counts())
print(train["BsmtCond"].value_counts())


# ## Fixing 'BsmtExposure', 'BsmtFinType1',  and 'BsmtFinType2'

# In[ ]:


getValsUniq("BsmtExposure")


# In[ ]:


# Above are 5 unique values, so NaN likely represents No Basement
train["BsmtExposure"].fillna("NA", inplace=True)


# In[ ]:


getValsUniq("BsmtFinType1")


# In[ ]:


# There are 7 unique values, NaN likely represent No Basement
train["BsmtFinType1"].fillna("NA", inplace=True)


# In[ ]:


getValsUniq("BsmtFinType2")


# In[ ]:


# Same as Type1 above
train["BsmtFinType2"].fillna("NA", inplace=True)


# ## Fixing 'Electrical' Column

# In[ ]:


getValsUniq("Electrical")


# In[ ]:


# Need to look further later to determine what value to fill
# For now, I will fill w/ mode
train["Electrical"].fillna("SBrkr", inplace=True)


# ## Fixing 'FireplaceQu' Column

# In[ ]:


getValsUniq("FireplaceQu")


# In[ ]:


train["FireplaceQu"].fillna("TEST", inplace=True)


# In[ ]:


train[train["FireplaceQu"] == "TEST"]["Fireplaces"].unique()


# In[ ]:


# Since all NaN values in FireplaceQu are associated w/ 0 fireplaces
# NaN values should be filled w/ No Fireplace
train["FireplaceQu"].replace({"TEST": "NA"})
train["FireplaceQu"].isnull().unique()


# ## Fixing 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', and 'GarageCond'

# In[ ]:


garageNullCols = ["GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond"]
for col in garageNullCols:
    getValsUniq(col)


# In[ ]:


train["GarageType"].fillna("TEST", inplace=True)


# In[ ]:


train[train["GarageType"] == "TEST"]["GarageArea"].unique()


# When there is no garage type, there is no garage area, so it is NA.

# In[ ]:


train["GarageType"].replace({"TEST": "NA"})
for col in garageNullCols:
    if col != "GarageYrBlt":
        train[col].fillna("NA", inplace=True)
    else:
        train[col].fillna(0, inplace=True)
        
for col in garageNullCols:
    print(train[col].isnull().value_counts())


# ## Fixing 'PoolQC', 'Fence', and 'MiscFeature'

# In[ ]:


extrasNullCols = ["PoolQC", "Fence", "MiscFeature"]
for col in extrasNullCols:
    getValsUniq(col)


# In[ ]:


train["PoolQC"].fillna("TEST", inplace=True)
train["PoolQC"].unique()


# In[ ]:


train[train["PoolQC"] == "TEST"]["PoolArea"].unique()


# In[ ]:


train["PoolQC"].replace({"TEST": "NA"})
for col in extrasNullCols:
    train[col].fillna("NA", inplace=True)


# In[ ]:


sum(train.isnull().any())


# In[ ]:


from sklearn.preprocessing import LabelEncoder
for col in train.columns:
    if train[col].dtypes == 'object':
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col])


# In[ ]:


sns.set(rc={'figure.figsize': (120,120)})
sns.set(font_scale=5)
sns.heatmap(train.corr())


# In[ ]:


plt.figure(figsize=(15,30))
plt.barh(train.corr().columns, train.corr()["SalePrice"])
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.xticks(np.arange(-1,1,step=0.2))


# In[ ]:


y=train["SalePrice"]
X=train.drop(["Id", "SalePrice"], axis=1, inplace=False)


# In[ ]:


from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_X, train_y)


# In[ ]:


preds = lr.predict(test_X)


# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

def getMetrics (test_ys, preds):
    acc = sum(1 - (abs(preds - test_ys) / test_ys)) / len(preds)
    mae = mean_absolute_error(test_ys, preds)
    mse = mean_squared_error(test_ys, preds)
    rmse = mse ** 0.5
    
    print("Model Accuracy:", acc)
    print("Mean Absolute Error:", mae)
    print("Mean Square Error:", mse)
    print("Root Mean Square Error:", rmse)
    
    plt.figure(figsize=(15,15))
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.scatter(test_ys, preds, s=45)
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")
    plt.title("Predicted vs. Actual Values")


# In[ ]:


getMetrics(test_y, preds)


# In[ ]:


test_y.describe()


# In[ ]:


testing_path = load_testing_data()
test = pd.read_csv(testing_path)
test.head()


# In[ ]:


test.info()


# In[ ]:


testNullCols = test.columns[test.isnull().any()].tolist()
print(len(null_cols))
print(len(testNullCols))
testNullCols == null_cols


# In[ ]:


len(test.columns) == len(train.columns) - 1


# In[ ]:


nullObjectCols = []
nullIntFloatCols = []
for col in test.columns:
    if test[col].dtypes == "object" and col in testNullCols:
        nullObjectCols.append(col)
    elif (test[col].dtypes == "int64" or test[col].dtypes == "float64") and col in testNullCols:
        nullIntFloatCols.append(col)
        
print("NULL OBJECT COLS:\n",str(nullObjectCols))
print("NULL INT/FLOAT COLS:\n",str(nullIntFloatCols))


# In[ ]:


for col in nullObjectCols:
    print(col, ":", test[col].unique())
    
# Compare the number of categories w/ given list. Some NaN may correspond with NA in the feature descs.


# In[ ]:


for col in nullObjectCols:
    print(col, ":", sum(test[col].isnull()))


# In[ ]:


colsWithoutNA = np.array(["MSZoning", "Utilities", "Exterior1st", "Exterior2nd", "Functional", "SaleType"])
colsWithNA = np.array([col for col in nullObjectCols if col not in colsWithoutNA])
for col in colsWithoutNA:
    test[col].fillna(test[col].mode().tolist()[0], inplace=True)
    
for col in colsWithNA:
    test[col].fillna("NA", inplace=True)


# In[ ]:


for col in nullIntFloatCols:
    print(col, ":", sum(test[col].isnull()))


# In[ ]:


test[test["GarageType"] == "NA"]["GarageArea"].unique()


# In[ ]:


test[test["GarageType"] == "NA"]["GarageCars"].unique()


# In[ ]:


colsWithZero = [col for col in nullIntFloatCols if col != "LotFrontage"]


# In[ ]:


for col in colsWithZero:
    test[col].fillna(0, inplace=True)


# In[ ]:


test["LotFrontage"].describe()


# In[ ]:


test["LotFrontage"].fillna(test["LotFrontage"].median(), inplace=True)


# In[ ]:


sum(test.isnull().any())


# In[ ]:


for col in test.columns:
    if test[col].dtypes == "object":
        le = LabelEncoder()
        test[col] = le.fit_transform(test[col])


# In[ ]:


test.info()


# In[ ]:


testing_ids = test["Id"]
submission_preds = lr.predict(test.drop("Id", axis=1, inplace=False))


# In[ ]:


output = pd.DataFrame({'Id': testing_ids,
                       'SalePrice': submission_preds})
output.to_csv('sample_submission.csv', index=False)

