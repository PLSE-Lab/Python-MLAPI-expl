#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <h2 style="color:blue" align="left"> 1. Import necessary Libraries </h2>

# In[ ]:


# Read Data
import numpy as np                     # Linear Algebra (calculate the mean and standard deviation)
import pandas as pd                    # manipulate data, data processing, load csv file I/O (e.g. pd.read_csv)

# Visualization
import seaborn as sns                  # Visualization using seaborn
import matplotlib.pyplot as plt        # Visualization using matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly                          # Visualization using Plotly
import plotly.express as px
import plotly.graph_objs as go

# style
plt.style.use("fivethirtyeight")       # Set Graphs Background style using matplotlib
sns.set_style("darkgrid")              # Set Graphs Background style using seaborn

# ML model building; Pre Processing & Evaluation
from sklearn.model_selection import train_test_split                     # split  data into training and testing sets
from sklearn.linear_model import LinearRegression, Lasso, Ridge          # Linear Regression, Lasso and Ridge
from sklearn.linear_model import LogisticRegression                      # Logistic Regression
from sklearn.tree import DecisionTreeRegressor                           # Decision tree Regression
from sklearn.ensemble import RandomForestRegressor                       # this will make a Random Forest Regression
from sklearn import svm                                                  # this will make a SVM classificaiton
from sklearn.svm import SVC                                              # import SVC from SVM
import xgboost
from sklearn.metrics import confusion_matrix, classification_report      # this creates a confusion matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import roc_curve,auc                                # ROC
from sklearn.preprocessing import StandardScaler                         # Standard Scalar
from sklearn.model_selection import GridSearchCV                         # this will do cross validation
from sklearn.decomposition import PCA                                    # to perform PCA to plot the data

import warnings                                                          # Ignore Warnings
warnings.filterwarnings("ignore")


# <h2 style="color:blue" align="left"> 2. Load data </h2>

# In[ ]:


# Import first 5 rows
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


display(train.head())
display(test.head())


# In[ ]:


# checking dimension (num of rows and columns) of dataset
print("Training data shape (Rows, Columns):",train.shape)
print("Test data shape (Rows, Columns):",test.shape)


# ### Checking for Numerical and Categorical features

# In[ ]:


# check dataframe structure like columns and its counts, datatypes & Null Values
train.info()


# In[ ]:


train.dtypes.value_counts()


# - Our dataset features consists of three datatypes
#      1. float
#      2. integer
#      3. object
# - Total numerical features are 38
# - Total categorical features are 43
# - Also we don't have complete data for all of our features

# In[ ]:


# check dataframe structure like columns and its counts, datatypes & Null Values
test.info()


# In[ ]:


test.dtypes.value_counts()


# - Our dataset features consists of three datatypes
#      1. float
#      2. integer
#      3. object
# - Total numerical features are 37
# - Total categorical features are 43
# - Also we don't have complete data for all of our features

# In[ ]:


# Gives number of data points in each variable
train.count()


# In[ ]:


# merge the train and test data and inspect the data type
merged = pd.concat([train, test], axis=0, sort=True)
display(merged.dtypes.value_counts())
print('Dimensions of data:', merged.shape)


# In[ ]:


# merge the train and test data and inspect the data type
merged = pd.concat([train, test], axis=0, sort=True)
display(merged.dtypes.value_counts())
print('Dimensions of data:', merged.shape)


# <h2 style="color:blue" align="left"> 3. EDA (Exploratory Data Analysis) </h2>
# 
# - EDA is a way of **Visualizing, Summarizing and interpreting** the information that is **hidden in rows and column** format.

# In[ ]:


train['MSZoning'].value_counts()


# ### Train Dataset
# #### a. Numeric Features

# In[ ]:


numeric_cols_train = train.select_dtypes(include=[np.number])
display(numeric_cols_train.head())
print('\n')
numeric_cols_train.columns


# In[ ]:


numeric_cols_train.shape


# #### b. Categorical Features

# In[ ]:


categorical_cols_train = train.select_dtypes(include=[np.object])
display(categorical_cols_train.head())
print('\n')
categorical_cols_train.columns


# ### Test Dataset
# #### a. Numeric Features

# In[ ]:


numeric_cols_test = test.select_dtypes(exclude='object')
display(numeric_cols_test.head())
print('\n')
numeric_cols_test.columns


# #### b. Categorical Features

# In[ ]:


categorical_cols_test = test.select_dtypes(include=[np.object])
display(categorical_cols_test.head())
print('\n')
categorical_cols_test.columns


# ### Steps involved in EDA:
# 1. Find Unwanted Columns
# - Find Missing Values
# - Find Features with one value
# - Explore the Categorical Features
# - Find Categorical Feature Distribution
# - Relationship between Categorical Features and Label
# - Explore the Numerical Features
# - Find Discrete Numerical Features
# - Relation between Discrete numerical Features and Labels
# - Find Continous Numerical Features
# - Distribution of Continous Numerical Features
# - Relation between Continous numerical Features and Labels
# - Find Outliers in numerical features
# - Explore the Correlation between numerical features

# #### 1. Find Unwanted Columns

# - There is no unwanted column present in given dataset to remove.
# 
#      EX: ID

# - Checking missing values by below methods:
# 
#      1. df.isnull().sum()
#         - It returns null values for each column
#           
#      2. isnull().any()
#         - It returns True if column have NULL Values
#         - It returns False if column don't have NULL Values
#           
#      3. Heatmap()
#         - Missing value representation using heatmap.
#           
#      4. Percentage of Missing values

# In[ ]:


# Listing Number of missing values by feature column wise
train_missing = train.isnull().sum().sort_values(ascending=False)
train_missing = train_missing[train_missing > 0]
train_missing


# In[ ]:


# Listing Number of missing values by feature column wise
test_missing = test.isnull().sum().sort_values(ascending=False)
test_missing = test_missing[test_missing > 0]
test_missing


# In[ ]:


# any() check null values by columns
train.isnull().any()


# In[ ]:


plt.figure(figsize=(17,10))
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[ ]:


plt.figure(figsize=(17,10))
sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# - All columns have **more than 1 unique value.** No feature found with one value.
# 
# 
# - There could be chance of only one category in a particular feature. In Categorical features, suppose gender column we have only one value ie male.Then there is no use of that feature in dataset. 

# In[ ]:


# Percentage of Missing values in train dataset
total = train.isnull().sum().sort_values(ascending=False)
percent = ((train.isnull().sum()/train.isnull().count())*100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, join='outer', keys=['Total Missing Count', '% of Total Observations'])
missing_data.index.name =' Numeric cols'

missing_data.head(20)


# In[ ]:


# Percentage of Missing values in train dataset
total = test.isnull().sum().sort_values(ascending=False)
percent = ((test.isnull().sum()/test.isnull().count())*100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, join='outer', keys=['Total Missing Count', '% of Total Observations'])
missing_data.index.name =' Numeric cols'

missing_data.head(20)


# #### 3. Find Features with one value

# In[ ]:


for column in train.columns:
    print(column,train[column].nunique())


# #### 4. Explore the Categorical Features

# In[ ]:


categorical_features = [feature for feature in train.columns if train[feature].dtypes=='O']
categorical_features


# In[ ]:


for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(train[feature].unique())))


# #### 5. Find Categorical Feature Distribution

# In[ ]:


fig, axes = plt.subplots(round(len(categorical_cols_train.columns) / 3), 3, figsize=(12, 30))

for i, ax in enumerate(fig.axes):
    if i < len(categorical_cols_train.columns):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
        sns.countplot(x=categorical_cols_train.columns[i], alpha=0.7, data=categorical_cols_train, ax=ax)

fig.tight_layout()


# #### 6. Relationship between Categorical Features and Label

# In[ ]:


# Find out the relationship between categorical variable and dependent varaible
plt.figure(figsize=(15,70), facecolor='white')
plotnumber =1
for feature in categorical_features:
    ax = plt.subplot(11,4,plotnumber)
    data=train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plotnumber+=1
plt.show()


# #### 7. Explore the Numerical Features

# In[ ]:


numerical_features = train.select_dtypes(exclude='object')
numerical_features


# #### 8. Find Discrete Numerical Features

# In[ ]:


discrete_feature=[feature for feature in numerical_features if len(train[feature].unique())<25]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# #### 9. Find Continous Numerical Features

# In[ ]:


continuous_features=[feature for feature in numerical_features if feature not in discrete_feature+['target']]
print("Continuous feature Count {}".format(len(continuous_features)))


# In[ ]:


continuous_features


# #### 10. Distribution of Continous Numerical Features

# In[ ]:


fig, ax = plt.subplots(5,4, figsize=(16,16))
sns.distplot(train['LotFrontage'], bins = 20, ax=ax[0,0]) 
sns.distplot(train.LotArea, bins = 20, ax=ax[0,1]) 
sns.distplot(train.YearBuilt, bins = 20, ax=ax[0,2]) 
sns.distplot(train.YearRemodAdd, bins = 20, ax=ax[0,3])
sns.distplot(train.MasVnrArea, bins = 20, ax=ax[1,0]) 
sns.distplot(train.BsmtFinSF1, bins = 20, ax=ax[1,1]) 
sns.distplot(train.BsmtUnfSF, bins = 20, ax=ax[1,3])
sns.distplot(train.TotalBsmtSF, bins = 20, ax=ax[2,0])
sns.distplot(train['1stFlrSF'], bins = 20, ax=ax[2,1])
sns.distplot(train['2ndFlrSF'], bins = 20, ax=ax[2,2])
sns.distplot(train.GrLivArea, bins = 20, ax=ax[2,3])
sns.distplot(train.GarageYrBlt, bins = 20, ax=ax[3,0])
sns.distplot(train.GarageArea, bins = 20, ax=ax[3,1])
sns.distplot(train.WoodDeckSF, bins = 20, ax=ax[3,2])
sns.distplot(train.OpenPorchSF, bins = 20, ax=ax[3,3])
sns.distplot(train.SalePrice, bins = 20, ax=ax[4,2])
plt.show()


# - it seems all continuous features are not normally distributed
# 
# - **MasVnrArea, BsmtFinSF1, BsmtUnfSF, 2ndFlrSF, GrLivArea & SalePrice** are **right skewed**
# 
# - **YearBuilt,GarageYrBlt,GarageArea,YearRemodAdd** is **left skewed**.

# #### 11. Relation between Continous numerical Features and Labels

# In[ ]:


plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for feature in continuous_features:
    data=train.copy()
    ax = plt.subplot(12,3,plotnumber)
    plt.scatter(train[feature],train['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plotnumber+=1
plt.show()


# #### 12. Find Outliers in numerical features

# In[ ]:


# boxplot on numerical features to find outliers
plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for numerical_feature in numerical_features:
    ax = plt.subplot(15,3,plotnumber)
    sns.boxplot(train[numerical_feature])
    plt.xlabel(numerical_feature)
    plotnumber+=1
plt.show()


# #### 13. Explore the Correlation between numerical features

# #### Correlation
# - Now we'll try to find which features are strongly correlated with SalePrice.

# In[ ]:


(train.corr()**2)["SalePrice"].sort_values(ascending = False)[1:]


# #### Correlation Heat Map

# In[ ]:


plt.figure(figsize = (14,12))
plt.title('Correlation of Numeric Features with Sale Price', y=1, size=16)
sns.heatmap(train.corr(), square = True, vmax=0.8)


# #### Selected HeatMap

# In[ ]:


corr = numeric_cols_train.drop('SalePrice', axis=1).corr()
plt.figure(figsize=(17, 14))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


# - A lot of features seems to be correlated between each other but some of them such as **YearBuild/GarageYrBlt** may just indicate a price inflation over the years. As for **1stFlrSF/TotalBsmtSF**, it is normal that the more the 1st floor is large (considering many houses have only 1 floor), the more the total basement will be large.
# 
# - There is a strong negative correlation between **BsmtUnfSF and BsmtFinSF2**.

# In[ ]:


plt.figure(figsize=(14,12))
corr = numeric_cols_train.corr()
k= 11
cols = corr.nlargest(k,'SalePrice')['SalePrice'].index
print(cols)
cm = np.corrcoef(train[cols].values.T)
sns.heatmap(cm, cmap='viridis', vmax=.8, annot=True, linewidth=2, linecolor="white", xticklabels = cols.values,
            yticklabels = cols.values, square=True, annot_kws = {'size':12})


# - 'GarageCars' and 'GarageArea' are strongly correlated variables. It is because the number of cars that fit into the garage is a consequence of the garage area. 'GarageCars' and 'GarageArea' are like twin brothers. So it is hard to distinguish between the two. Therefore, we just need one of these variables in our analysis (we can keep 'GarageCars' since its correlation with 'SalePrice' is higher).
# 
# 
# - 'TotRmsAbvGrd' and 'GrLivArea', twins

# #### 14. Descriptive statistics

# In[ ]:


# descriptive statistics (numerical columns)
train.describe()


# - for **each feature** it provides below:
# 
#      1. count       --->  total no of data points
#      2. statistics  --->  mean, standard deviation
#      3. min & max   --->  values of feature
#      4. percentile  --->  25%, 50%, 75%

# <h2 style="color:green" align="left"> 5. Data Visualization </h2>
# 
# - Used below **visualisation libraries**
# 
#      1. Matplotlib
#      2. Seaborn (statistical data visualization)
#      
# ### 1. Categorical
# 
# - Categorical data :
# 
#      1. Numerical Summaries
#      2. Histograms
#      3. Pie Charts
# 
# 
# ### 2. Univariate Analysis
# 
# - Univariate Analysis : data consists of **only one variable (only x value)**.
# 
#      1. Line Plots / Bar Charts
#      2. Histograms
#      3. Box Plots 
#      4. Count Plots
#      5. Descriptive Statistics techniques
#      6. Violin Plot

# ### 1. Histogram

# In[ ]:


# Histogram for "SalePrice"
plt.figure(figsize=(9,7))
sns.distplot(train['SalePrice'])


# - With this information we can see that the prices are skewed right and some outliers lies above ~500,000. We will eventually want to get rid of the them to get a normal distribution of the independent variable (SalePrice) for machine learning

# In[ ]:


# Histogram for "Numerical Features in train dataset"
numeric_cols_train.hist(figsize=(16, 20), bins=50, xlabelsize=7, ylabelsize=7);


# In[ ]:


# Histogram for "Numerical Features in test dataset"
numeric_cols_test.hist(figsize=(16, 20), bins=50, xlabelsize=7, ylabelsize=7);


# ### 3. Bivariate Analysis
# 
# - **Bivariate Analysis** : data involves **two different variables**.
# 
#      1. Bar Charts
#      2. Scatter Plots
#      3. FacetGrid
#      
# 
# -  There are **three** types of bivariate analysis
# 
#      1. Numerical & Numerical
#      2. Categorical & Categorical
#      3. Numerical & Categorical

# ### 1. Line Plot

# #### YrSold Vs SalePrice

# In[ ]:


# Line Plot between "YrSold" and "SalePrice"
plt.figure(figsize=(7,6))
sns.lineplot(x=train['YrSold'], y=train['SalePrice'])

plt.xlabel('YrSold', fontsize=15, fontweight='bold')
plt.ylabel('SalePrice', fontsize=15, fontweight='bold')

plt.title('YrSold Vs SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# #### YearBuilt Vs SalePrice

# In[ ]:


# Line Plot between "YearBuilt" and "SalePrice"
plt.figure(figsize=(7,6))
sns.lineplot(x=train['YearBuilt'], y=train['SalePrice'])

plt.xlabel('YearBuilt', fontsize=15, fontweight='bold')
plt.ylabel('SalePrice', fontsize=15, fontweight='bold')

plt.title('YearBuilt Vs SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# #### MoSold Vs SalePrice

# In[ ]:


# Line Plot between "MoSold" and "SalePrice"
plt.figure(figsize=(7,6))
sns.lineplot(x=train['MoSold'], y=train['SalePrice'])

plt.xlabel('MoSold', fontsize=15, fontweight='bold')
plt.ylabel('SalePrice', fontsize=15, fontweight='bold')

plt.title('MoSold Vs SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# #### OverallQual Vs SalePrice

# In[ ]:


train['OverallQual'].value_counts()


# In[ ]:


# Line Plot between "OverallQual" and "SalePrice"
plt.figure(figsize=(7,6))
sns.lineplot(x=train['OverallQual'], y=train['SalePrice'])

plt.xlabel('OverallQual', fontsize=15, fontweight='bold')
plt.ylabel('SalePrice', fontsize=15, fontweight='bold')

plt.title('OverallQual Vs SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# #### SaleCondition Vs SalePrice

# In[ ]:


# Line Plot between "SaleCondition" and "SalePrice"
plt.figure(figsize=(7,6))
sns.lineplot(x=train['SaleCondition'], y=train['SalePrice'])

plt.xlabel('SaleCondition', fontsize=15, fontweight='bold')
plt.ylabel('SalePrice', fontsize=15, fontweight='bold')

plt.title('SaleCondition Vs SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# ### 2. Scatter Plot

# #### OverallQual Vs SalePrice

# In[ ]:


# Scatter Plot between "OverallQual" and "SalePrice"
plt.figure(figsize=(7,6))
sns.scatterplot(x=train['OverallQual'], y=train['SalePrice'])

plt.xlabel('OverallQual', fontsize=15, fontweight='bold')
plt.ylabel('SalePrice', fontsize=15, fontweight='bold')

plt.title('OverallQual Vs SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# #### GrLivArea Vs SalePrice

# In[ ]:


# Scatter Plot between "GrLivArea" and "SalePrice"
plt.figure(figsize=(7,6))
sns.scatterplot(train.GrLivArea, train.SalePrice)

plt.xlabel('GrLivArea', fontsize=15, fontweight='bold')
plt.ylabel('SalePrice', fontsize=15, fontweight='bold')

plt.title('GrLivArea Vs SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# #### GarageArea Vs SalePrice

# In[ ]:


# Scatter Plot between "GarageArea" and "target" variable
plt.figure(figsize=(7,6))
sns.scatterplot(train.GarageArea, train.SalePrice)

plt.xlabel('GarageArea', fontsize=15, fontweight='bold')
plt.ylabel('SalePrice', fontsize=15, fontweight='bold')

plt.title('GarageArea Vs SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# #### TotalBsmtSF Vs SalePrice

# In[ ]:


# Scatter Plot between "TotalBsmtSF" and "SalePrice"
plt.figure(figsize=(7,6))
sns.scatterplot(train.TotalBsmtSF, train.SalePrice)

plt.xlabel('TotalBsmtSF', fontsize=15, fontweight='bold')
plt.ylabel('SalePrice', fontsize=15, fontweight='bold')

plt.title('TotalBsmtSF Vs SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# ####  1stFlrSF Vs SalePrice

# In[ ]:


# Scatter Plot between "1stFlrSF" and "SalePrice"
plt.figure(figsize=(7,6))
sns.scatterplot(train['1stFlrSF'], train.SalePrice)

plt.xlabel('1stFlrSF', fontsize=15, fontweight='bold')
plt.ylabel('SalePrice', fontsize=15, fontweight='bold')

plt.title('1stFlrSF Vs SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# #### SalePrice vs MasVnrArea

# In[ ]:


# Scatter Plot between "MasVnrArea" and "SalePrice"
plt.figure(figsize=(7,6))
sns.scatterplot(train.MasVnrArea, train.SalePrice)

plt.xlabel('MasVnrArea', fontsize=15, fontweight='bold')
plt.ylabel('SalePrice', fontsize=15, fontweight='bold')

plt.title('MasVnrArea Vs SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# In[ ]:


# Scatter Plot between "MasVnrArea" and "SalePrice"
plt.figure(figsize=(7,6))
sns.scatterplot(np.sqrt(train['MasVnrArea']),np.log(train['SalePrice']))

plt.xlabel('Sqrt_MasVnrArea', fontsize=15, fontweight='bold')
plt.ylabel('log_SalePrice', fontsize=15, fontweight='bold')

plt.title('Sqrt_MasVnrArea Vs log_SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# #### MSSubClass Vs SalePrice

# In[ ]:


# Scatter Plot between "MSSubClass" and "SalePrice"
plt.figure(figsize=(7,6))
plt.scatter(train['MSSubClass'], train['SalePrice'])

plt.xlabel('MSSubClass', fontsize=15, fontweight='bold')
plt.ylabel('SalePrice', fontsize=15, fontweight='bold')

plt.title('MSSubClass Vs SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# #### LotFrontage Vs SalePrice

# In[ ]:


# Scatter Plot between "LotFrontage" and "SalePrice"
plt.figure(figsize=(7,6))
plt.scatter(train['LotFrontage'], train['SalePrice'])

plt.xlabel('LotFrontage', fontsize=15, fontweight='bold')
plt.ylabel('SalePrice', fontsize=15, fontweight='bold')

plt.title('LotFrontage Vs SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# - Observed **Outliers**

# #### LotArea Vs SalePrice

# In[ ]:


# Scatter Plot between "LotArea" and "SalePrice"
plt.figure(figsize=(7,6))
plt.scatter(train['LotArea'], train['SalePrice'])

plt.xlabel('LotArea', fontsize=15, fontweight='bold')
plt.ylabel('SalePrice', fontsize=15, fontweight='bold')

plt.title('LotArea Vs SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# - Observed **Outliers** and **non-linear** relationship

# In[ ]:


# Scatter Plot between "LotArea" and "SalePrice"
plt.figure(figsize=(7,6))
plt.scatter(np.log(train['LotArea']), np.log(train['SalePrice']))

plt.xlabel('log_LotArea', fontsize=15, fontweight='bold')
plt.ylabel('log_SalePrice', fontsize=15, fontweight='bold')

plt.title('log_LotArea Vs log_SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# #### TotRmsAbvGrd Vs SalePrice

# In[ ]:


# Scatter Plot between "TotRmsAbvGrd" and "SalePrice"
plt.figure(figsize=(7,6))
plt.scatter(train['TotRmsAbvGrd'], train['SalePrice'])

plt.xlabel('TotRmsAbvGrd', fontsize=15, fontweight='bold')
plt.ylabel('SalePrice', fontsize=15, fontweight='bold')

plt.title('TotRmsAbvGrd Vs SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# #### CentralAir Vs SalePrice

# In[ ]:


# Scatter Plot between "CentralAir" and "SalePrice"
plt.figure(figsize=(7,6))
plt.scatter(train['CentralAir'], train['SalePrice'])

plt.xlabel('CentralAir', fontsize=15, fontweight='bold')
plt.ylabel('SalePrice', fontsize=15, fontweight='bold')

plt.title('CentralAir Vs SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# #### KitchenAbvGr Vs SalePrice

# In[ ]:


# Scatter Plot between "KitchenAbvGr" and "SalePrice"
plt.figure(figsize=(7,6))
plt.scatter(train['KitchenAbvGr'], train['SalePrice'])

plt.xlabel('KitchenAbvGr', fontsize=15, fontweight='bold')
plt.ylabel('SalePrice', fontsize=15, fontweight='bold')

plt.title('KitchenAbvGr Vs SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# #### OpenPorchSF Vs SalePrice

# In[ ]:


# Scatter Plot between "OpenPorchSF" and "SalePrice"
plt.figure(figsize=(7,6))
plt.scatter(train['OpenPorchSF'], train['SalePrice'])

plt.xlabel('OpenPorchSF', fontsize=15, fontweight='bold')
plt.ylabel('SalePrice', fontsize=15, fontweight='bold')

plt.title('OpenPorchSF Vs SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# In[ ]:


# Scatter Plot between "OpenPorchSF" and "SalePrice"
plt.figure(figsize=(7,6))
plt.scatter(np.log(train['OpenPorchSF']),np.log(train['SalePrice']))

plt.xlabel('log_OpenPorchSF', fontsize=15, fontweight='bold')
plt.ylabel('log_SalePrice', fontsize=15, fontweight='bold')

plt.title('log_OpenPorchSF Vs log_SalePrice', fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# ### 3. Box Plot

# In[ ]:


# Box Plot for "BsmtExposure" & "SalePrice"
plt.figure(figsize = (10, 6))
sns.boxplot(x='BsmtExposure', y='SalePrice', data=train)


# In[ ]:


# Box Plot for "SaleCondition" & "SalePrice"
plt.figure(figsize = (12, 6))
sns.boxplot(x='SaleCondition', y='SalePrice', data=train)


# In[ ]:


# Box Plot for "Neighborhood" & "SalePrice"
plt.figure(figsize=(15, 9))
sns.boxplot(x='Neighborhood', y="SalePrice", data=train)
plt.xticks(rotation=90)


# ### 4. Count Plot

# In[ ]:


# Count Plot for "Neighborhood"
plt.figure(figsize = (15, 9))
sns.countplot(x = 'Neighborhood', data = train)
xt = plt.xticks(rotation=45)


# In[ ]:


num_feat = set(train._get_numeric_data().columns)
feat = set(train.columns)
cat_feat = list(feat-num_feat)
print("total categoricalfeatures : "+str(len(cat_feat)))

y='SalePrice'
for i,j in enumerate(cat_feat):
    
    sns.catplot(x=j, y=y, data=train, alpha=0.5)
    plt.xticks(rotation=90)


# ### 3. Multivariate Analysis
# 
# - 1. Pair Plot
#     
#     Pair Plot between 'SalePrice' and correlated variables

# In[ ]:


sns.set()
columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']
sns.pairplot(train[columns], height = 2, kind ='scatter', diag_kind='kde')


# <h2 style="color:blue" align="left"> 6. Detect outliers using IQR </h2>

# In[ ]:


train_IQR = train[['MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', '2ndFlrSF', 'GrLivArea', 'YearBuilt', 'GarageYrBlt',
                   'GarageArea', 'YearRemodAdd']]
Q1 = train_IQR.quantile(0.25)
Q3 = train_IQR.quantile(0.75)
IQR = Q3 - Q1
IQR


# In[ ]:


# Here are the outliers

train_IQR_clean = train_IQR[~((train_IQR < (Q1 - 1.5*IQR)) | (train_IQR > (Q3 + 1.5*IQR))).any(axis=1)]


# In[ ]:


# boxplot() showing outlier
box = train_IQR_clean
plt.figure(figsize=(12,10))
sns.boxplot(data=box)
plt.show()


# 

# ### Drop Features have more missing values

# In[ ]:


# Drop columns of train dataset with more than 25% of missing data

train.drop(columns=['Id','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], inplace=True, axis=1)


# In[ ]:


train.shape


# In[ ]:


# Drop columns of test dataset with more than 25% of missing data

test.drop(columns=['Id','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], inplace=True, axis=1)


# In[ ]:


test.shape


# ### filling nan values of categorical features by their mode and numeric ones by thier mean.

# In[ ]:


df = [train, test]


# In[ ]:


# train data

  # Numeric Features
    
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())  # float
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())  # float
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())     # float

  # Categorical Features
    
train['GarageCond'] = train['GarageCond'].fillna(train['GarageCond'].mode()[0])
train['GarageQual'] = train['GarageQual'].fillna(train['GarageQual'].mode()[0])
train['GarageFinish'] = train['GarageFinish'].fillna(train['GarageFinish'].mode()[0])
train['GarageType'] = train['GarageType'].fillna(train['GarageType'].mode()[0])
train['BsmtFinType2'] = train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0])
train['BsmtExposure'] = train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0])
train['BsmtFinType1'] = train['BsmtFinType1'].fillna(train['BsmtFinType1'].mode()[0])
train['BsmtQual'] = train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])
train['BsmtCond'] = train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])
train['MasVnrType'] = train['MasVnrType'].fillna(train['MasVnrType'].mode()[0])
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

# test data

  # Numeric Featurestrain
    
test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].mean())  # float
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean())  # float
test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mean())     # float
test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mean())
test['BsmtFullBath'] = test['BsmtFullBath'].fillna(test['BsmtFullBath'].mean())
test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mean())
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean())
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean())
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())
test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].mean())

  # Categorical Features
    
test['GarageCond'] = test['GarageCond'].fillna(test['GarageCond'].mode()[0])
test['GarageQual'] = test['GarageQual'].fillna(test['GarageQual'].mode()[0])
test['GarageFinish'] = test['GarageFinish'].fillna(test['GarageFinish'].mode()[0])
test['GarageType'] = test['GarageType'].fillna(test['GarageType'].mode()[0])
test['BsmtQual'] = test['BsmtQual'].fillna(test['BsmtQual'].mode()[0])
test['BsmtCond'] = test['BsmtCond'].fillna(test['BsmtCond'].mode()[0])
test['BsmtFinType2'] = test['BsmtFinType2'].fillna(test['BsmtFinType2'].mode()[0])
test['BsmtExposure'] = test['BsmtExposure'].fillna(test['BsmtExposure'].mode()[0])
test['BsmtFinType1'] = test['BsmtFinType1'].fillna(test['BsmtFinType1'].mode()[0])
test['MasVnrType'] = test['MasVnrType'].fillna(test['MasVnrType'].mode()[0])
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])
test['Utilities'] = test['Utilities'].fillna(test['Utilities'].mode()[0])
test['Functional'] = test['Functional'].fillna(test['Functional'].mode()[0])
test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])
test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])


# In[ ]:


plt.figure(figsize=(17,10))
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[ ]:


plt.figure(figsize=(17,10))
sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# <h2 style="color:blue" align="left"> 7. Check & Reduce Skewness </h2>

# - Skewness tells us about the symmetry in a distribution.
# 
# * If the **skewness** is **between -0.5 to +0.5** then we can say data is **fairly symmetrical**.
#   
# * If the **skewness** is **between -1 to -0.5 or 0.5 to 1** then data is **moderately skewed**.
#   
# * If the **skewness** is **less than -1 and greater than +1** then our data is **heavily skewed**.

# ### Train set

# In[ ]:


numeric_cols_train.skew()


# - In our above data,
#     1. LotArea
#     2. LowQualFinSF
#     3. 3SsnPorch
#     4. PoolArea
#     5. MiscVal
# - Are highly positively,right skewed.

# ### Checking Skewness for feature "SalePrice"

# In[ ]:


sns.distplot(train['SalePrice'])
Skew_SalePrice = train['SalePrice'].skew()
plt.title("Skew:"+str(Skew_SalePrice))


# In[ ]:


# SalePrice right skewed; log transform)
sns.distplot(np.log(train['SalePrice']+1))
Skew_SalePrice_Log = np.log(train['SalePrice']+1).skew()
plt.title("Skew:"+str(Skew_SalePrice_Log))


# ### a. Checking Skewness for feature "LotArea"

# In[ ]:


# Checking the skewness of "LotArea" attributes
train['LotArea'].skew()


# In[ ]:


# calculating the square for the column df['LotArea'] column
log_LotArea_train = np.log(train['LotArea'])
log_LotArea_train.skew()


# ### b. Checking Skewness for feature "LowQualFinSF"

# In[ ]:


# Checking the skewness of "LowQualFinSF" attributes
train['LowQualFinSF'].skew()


# In[ ]:


# calculating the square for the column df['LowQualFinSF'] column
recipr_LowQualFinSF_train = np.reciprocal(train['LowQualFinSF'])
recipr_LowQualFinSF_train.skew()


# ### c. Checking Skewness for feature "3SsnPorch"

# In[ ]:


# Checking the skewness of "3SsnPorch" attributes
train['3SsnPorch'].skew()


# In[ ]:


# performing the log transformation using numpy
recipr_3SsnPorch_train = np.reciprocal(train['3SsnPorch'])
recipr_3SsnPorch_train.skew()


# ### d. Checking Skewness for feature "PoolArea"

# In[ ]:


# Checking the skewness of "PoolArea" attributes
train['PoolArea'].skew()


# In[ ]:


# performing the log transformation using numpy
cuberoot_PoolArea_train = np.cbrt(train['PoolArea'])
cuberoot_PoolArea_train.skew()


# ### e. Checking Skewness for feature "MiscVal"

# In[ ]:


# Checking the skewness of "MiscVal" attributes
train['MiscVal'].skew()


# In[ ]:


# performing the log transformation using numpy
recipr_MiscVal_train = np.reciprocal(train['MiscVal'])
recipr_MiscVal_train.skew()


# In[ ]:


train_skew = pd.concat([log_LotArea_train, recipr_LowQualFinSF_train, recipr_3SsnPorch_train,
                        recipr_MiscVal_train], axis=1)
train_skew


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


train.drop(['LotArea','LowQualFinSF','3SsnPorch','MiscVal'], inplace=True, axis=1)
train.head()


# In[ ]:


new_train = pd.concat([train, train_skew], axis=1)
new_train.head()


# In[ ]:


new_train.shape


# ### Test set

# In[ ]:


numeric_cols_test.skew()


# - In our above data,
#     1. LowQualFinSF
#     2. 3SsnPorch
#     3. PoolArea
#     4. MiscVal
# - Are highly positively,right skewed.

# ### a. Checking Skewness for feature "LowQualFinSF"

# In[ ]:


# Checking the skewness of "LowQualFinSF" attributes
test['LowQualFinSF'].skew()


# In[ ]:


# performing the cube root transformation using numpy
cube_root_LowQualFinSF_test = np.cbrt(test['LowQualFinSF'])
cube_root_LowQualFinSF_test.skew()


# ### b. Checking Skewness for feature "3SsnPorch"

# In[ ]:


# Checking the skewness of "3SsnPorch" attributes
test['3SsnPorch'].skew()


# In[ ]:


# performing the cube root transformation using numpy
cube_root_3SsnPorch_test = np.cbrt(test['3SsnPorch'])
cube_root_3SsnPorch_test.skew()


# ### c. Checking Skewness for feature "PoolArea"

# In[ ]:


# Checking the skewness of "PoolArea" attributes
test['PoolArea'].skew()


# In[ ]:


# performing the cube root transformation using numpy
recipr_PoolArea_test = np.reciprocal(test['PoolArea'])
recipr_PoolArea_test.skew()


# ### d. Checking Skewness for feature "MiscVal"

# In[ ]:


# Checking the skewness of "MiscVal" attributes
test['MiscVal'].skew()


# In[ ]:


# performing the cube root transformation using numpy
cube_root_MiscVal_test = np.cbrt(test['MiscVal'])
cube_root_MiscVal_test.skew()


# <h2 style="color:blue" align="left"> 7. Model building and Evaluation </h2>

# ### OneHotEncoding

# In[ ]:


# Transform discrete values to columns with 1 and 0s
train_OHE = pd.get_dummies(train)

# Do the same for competition data
test_OHE = pd.get_dummies(test)


# In[ ]:


display(train_OHE.head())
display(test_OHE.head())


# In[ ]:


plt.figure(figsize=(14,12))
corr = numeric_cols_train.corr()
k= 11
cols = corr.nlargest(k,'SalePrice')['SalePrice'].index
print(cols)
cm = np.corrcoef(train[cols].values.T)
sns.heatmap(cm, cmap='viridis', vmax=.8, annot=True, linewidth=2, linecolor="white", xticklabels = cols.values,
            yticklabels = cols.values, square=True, annot_kws = {'size':12})


# In[ ]:


print("Training Data Shape (Rows,Columns):",train_OHE.shape)
print("Competition Data Shape (Rows,Columns):", test_OHE.shape)


# In[ ]:


# There is a differece between features in the training data set and the test data set
# We will try dropping the features that are not present in both sets

missingFeatures_train = list(set(train_OHE.columns.values) - set(test_OHE.columns.values))
train_OHE = train_OHE.drop(missingFeatures_train, axis=1)

missingFeatures_test = list(set(test_OHE.columns.values) - set(train_OHE.columns.values))
test_OHE = test_OHE.drop(missingFeatures_test, axis=1)


# In[ ]:


print("Training Data Shape (Rows,Columns):",train_OHE.shape)
print("Competition Data Shape (Rows,Columns):", test_OHE.shape)


# In[ ]:


# Independant variable
X = train_OHE                     # All rows & columns exclude Target features

# Dependant variable
y = train['SalePrice']        # Only target feature


# In[ ]:


# split  data into training and testing sets of 80:20 ratio
# 20% of test size selected
# random_state is random seed
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=4)


# In[ ]:


# shape of X & Y test / train
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# <h2 style="color:green" align=left> 1. Linear Regression / Lasso / Ridge </h2>

# In[ ]:


LinReg = LinearRegression()
LinReg.fit(X_train, y_train)


# In[ ]:


# Lasso Regression
lasso = Lasso()
lasso.fit(X_train, y_train)


# In[ ]:


# Ridge Regression
ridge = Ridge()
ridge.fit(X_train, y_train)


# In[ ]:


y_pred_LinReg = LinReg.predict(X_test)
y_pred_lasso = lasso.predict(X_test)
y_pred_ridge = ridge.predict(X_test)


# In[ ]:


y_pred_train_LinReg = LinReg.predict(X_train)


# In[ ]:


print("Linear Regression : Train Score {:.2f} & Test Score {:.2f}".format(LinReg.score(X_train, y_train), LinReg.score(X_test, y_test)))
print("Lasso Regression : Train Score {:.2f} & Test Score {:.2f}".format(lasso.score(X_train, y_train), lasso.score(X_test, y_test)))
print("Ridge Regression : Train Score {:.2f} & Test Score {:.2f}".format(ridge.score(X_train, y_train), ridge.score(X_test, y_test)))


# In[ ]:


print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2")

print("""LinearRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_LinReg)),
            mean_squared_error(y_test, y_pred_LinReg),
            mean_absolute_error(y_test, y_pred_LinReg),
            r2_score(y_test, y_pred_LinReg)))

print("""LassoRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
            mean_squared_error(y_test, y_pred_lasso),
            mean_absolute_error(y_test, y_pred_lasso),
            r2_score(y_test, y_pred_lasso)))

print("""RidgeRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
            mean_squared_error(y_test, y_pred_ridge),
            mean_absolute_error(y_test, y_pred_ridge),
            r2_score(y_test, y_pred_ridge)))


# In[ ]:


# Plotting Predictions
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,4))

ax1.scatter(y_pred_LinReg, y_test, s=20)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax1.set_ylabel("True")
ax1.set_xlabel("Predicted")
ax1.set_title("Linear Regression")

ax2.scatter(y_pred_lasso, y_test, s=20)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax2.set_ylabel("True")
ax2.set_xlabel("Predicted")
ax2.set_title("Lasso Regression")

ax3.scatter(y_pred_ridge, y_test, s=20)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax3.set_ylabel("True")
ax3.set_xlabel("Predicted")
ax3.set_title("Ridge Regression")
fig.suptitle("True vs Predicted")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[ ]:


plt.figure(figsize=(9,8))

plt.scatter(y_pred_train_LinReg, y_train, c = "blue",  label = "Training data")
plt.scatter(y_pred_LinReg, y_test, c = "black",  label = "Test data")
plt.plot(y_train, y_train, alpha=0.3, c='r')

plt.xlabel("Predicted values")
plt.ylabel("Real values")

plt.title("Linear regression")

plt.legend(loc = "upper left")

plt.show()


# In[ ]:


plt.figure(figsize=(9,8))

plt.scatter(y_train, y_pred_train_LinReg)
plt.plot(y_train, y_train, alpha=0.3, c='r')

plt.xlabel("Sales", fontsize=15, fontweight='bold')
plt.ylabel("Predicted Sales", fontsize=15, fontweight='bold')

plt.title("Training Data Fit", fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# In[ ]:


plt.figure(figsize=(9,8))
plt.scatter(y_test, y_pred_LinReg)
plt.plot(y_test, y_test, alpha=0.3, c='r')

plt.xlabel("Sales", fontsize=15, fontweight='bold')
plt.ylabel("Predicted Sales", fontsize=15, fontweight='bold')

plt.title("Test Data Fit", fontsize=18, fontweight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


# In[ ]:


a = (y_test-y_pred_LinReg)
sns.distplot(a)


# In[ ]:


plt.scatter(y_train-y_pred_train_LinReg, y_train)
plt.title("Residual Vs  Actual SalePrice for train data")


# In[ ]:


plt.scatter(y_test-y_pred_LinReg, y_test)
plt.title("Residual Vs  Actual SalePrice for test data")
plt.show()


# <h2 style="color:green" align=left> 2. Decision Tree </h2>

# In[ ]:


DTR = DecisionTreeRegressor()
DTR.fit(X_train, y_train)


# In[ ]:


y_pred_DTR = DTR.predict(X_test)


# In[ ]:


print("Train Score {:.2f} & Test Score {:.2f}".format(DTR.score(X_train, y_train), DTR.score(X_test, y_test)))


# In[ ]:


print("Model\t\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2")
print("""Decision Tree Regressor \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_DTR)),
            mean_squared_error(y_test, y_pred_DTR),
            mean_absolute_error(y_test, y_pred_DTR),
            r2_score(y_test, y_pred_DTR)))

plt.scatter(y_test, y_pred_DTR)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

plt.xlabel("Predicted")
plt.ylabel("True")

plt.title("Decision Tree Regressor")

plt.show()


# In[ ]:


# Tuning Hyperparameter max_depth & min_sam_split of DecisionTreeRegressor
max_d = list(range(1,10))
min_sam_split = list(range(10,50,15))
gridcv = GridSearchCV(DTR, param_grid={'max_depth':max_d, 'min_samples_split':min_sam_split}, n_jobs=-1)
gridcv.fit(X_train, y_train)


# In[ ]:


print("Parameters :", gridcv.best_params_)
print("Train Score {:.2f} & Test Score {:.2f}".format(gridcv.score(X_train, y_train), gridcv.score(X_test, y_test)))


# <h3 style="color:green" align="left"> 3. Random Forest Regressor </h3>

# In[ ]:


rf = RandomForestRegressor()
rf.fit(X_train,y_train)


# In[ ]:


y_pred = rf.predict(X_test)


# In[ ]:


print("Train Score {:.2f} & Test Score {:.2f}".format(rf.score(X_train, y_train), rf.score(X_test, y_test)))


# <h3 style="color:green" align="left"> 4. Logistic Regression </h3>

# In[ ]:


LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)


# In[ ]:


y_pred_test_Log = LogReg.predict(X_test)


# In[ ]:


print("Train Score {:.2f} & Test Score {:.2f}".format(LogReg.score(X_train, y_train), LogReg.score(X_test, y_test)))


# <h3 style="color:green" align="left"> 5. XGBoost </h3>

# In[ ]:


import xgboost
reg_xgb = xgboost.XGBRegressor()
reg_xgb.fit(X_train,y_train)


# In[ ]:


# predicting X_test
y_pred_xgb = reg_xgb.predict(X_test)


# In[ ]:


print("Train Score {:.2f} & Test Score {:.2f}".format(reg_xgb.score(X_train,y_train),reg_xgb.score(X_test,y_test)))


# ### Score Summary :

# In[ ]:


models = [LinReg, DTR, rf, reg_xgb]
names = ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "XGBoost"]
rmses = []

for model in models:
    rmses.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

x = np.arange(len(names)) 
width = 0.3

fig, ax = plt.subplots(figsize=(10,7))
rects = ax.bar(x, rmses, width)
ax.set_ylabel('RMSE')
ax.set_xlabel('Models')

ax.set_title('RMSE with Different Algorithms')

ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45)

fig.tight_layout()


# ### Submission

# In[ ]:


df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


y_pred_test_OHE = rf.predict(test_OHE)


# In[ ]:


submission = pd.DataFrame({'Id': df.Id, 'SalePrice': y_pred_test_OHE})
submission.to_csv('Housing_submission.csv', index=False)

