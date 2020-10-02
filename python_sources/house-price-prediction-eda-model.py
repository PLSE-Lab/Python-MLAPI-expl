#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # **Getting Data**

# In[ ]:


train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")


# ### Getting View of Submission file

# In[ ]:


submission_look = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission_look.head()


# # **EDA**

# ### Looking into Data

# In[ ]:


print("\n TRAIN FILE \n")
display(train.head())
print("\n TEST FILE \n")
display(test.head())


# ### Shape of Data

# In[ ]:


print("\n TRAIN FILE \n")
display(train.shape)
print("\n TEST FILE \n")
display(test.shape)


# ### Statistical Description

# In[ ]:


print("\n TRAIN FILE \n")
display(train.describe())
print("\n TEST FILE \n")
display(test.describe())


# ### Columns

# In[ ]:


print("\n TRAIN FILE \n")
display(train.columns)
print("\n TEST FILE \n")
display(test.columns)


# ### NULL VALUES

# In[ ]:


print("\n TRAIN FILE \n")
display(train.isnull().sum())
print("\n TEST FILE \n")
display(test.isnull().sum())


# ### Missing Data(in percentage)

# In[ ]:


def missing_data_percentage(a):
    total = a.isnull().sum().sort_values(ascending=False)
    percent = (a.isnull().sum()/a.isnull().count()).sort_values(ascending=False)
    b = pd.concat([total, percent], axis=1, keys=['Total NULL values', 'Percentage'])
    return b

print("\n TRAIN FILE \n")
display(missing_data_percentage(train))
print("\n TEST FILE \n")
display(missing_data_percentage(test))


# In[ ]:


a = missing_data_percentage(train)


# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
plt.xticks(rotation='90')
sns.barplot(x=a.index, y=a['Percentage']*100)
plt.xlabel('Features', fontsize=1)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# ### Data Type of columns

# In[ ]:


print("\n TRAIN FILE \n")
display(train.dtypes)
print("\n TEST FILE \n")
display(test.dtypes)


# ### Sales Price Analysis

# In[ ]:


print("\n Description \n")
display(train['SalePrice'].describe())


# #### Distribution Curve

# In[ ]:


sns.distplot(train['SalePrice'])


# #### Data is skewed and dense at bottom, So checking for skewness and kurtosis

# In[ ]:


print("Skewness Value:",train['SalePrice'].skew())
print("Kurtosis value:", train['SalePrice'].kurt())


# #### HeatMap
#     To get important features

# In[ ]:


corrmat = train.corr()
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# #### Pairplot

# In[ ]:


cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols])


# In[ ]:


sns.distplot(np.log(train['SalePrice']), fit=norm);
fig = plt.figure()
res = stats.probplot(np.log(train['SalePrice']), plot=plt)


# In[ ]:


#train['SalePrice'] = np.log(train['SalePrice'])
#train['GrLivArea'] = np.log(train['GrLivArea'])
#train['TotalBsmtSF'] = np.log(train['TotalBsmtSF'])


# > We need to now apply transformations to our data, remove columns 

# # Data Preprocessing

# ## Preprocessing Train File

# In[ ]:


def preprocess(df):
    df.dropna(inplace=True, axis=1)
    df.reset_index(inplace=True,drop=True) 
    object_type_columns = df.select_dtypes(include=['object']).columns
    print("Object_Type_Columns:\n",object_type_columns)
    display("Top 10 most affecting factors")
    corrmat = train.corr()
    k = 10
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    display('Before Label Encoding Dataframe')
    display(df)
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    for i in object_type_columns:
        df[i] = le.fit_transform(df[i])
    display('Making Data ready for Fitting in Model')
    return df


# In[ ]:


preprocess(train)


# In[ ]:


train_cols = train.columns


# ## Preprocessing Test File

# In[ ]:


test1 = train.drop(['SalePrice'],axis=1)
test_new = test[test1.columns]
display(test_new.columns)
display(test_new.dtypes)


# In[ ]:


#test_new[['MSZoning']].idxmax()


# In[ ]:


def preprocess_test(df):
    #df = df.replace({np.nan:'-'})
    object_type_columns = df.select_dtypes(include=['object']).columns
    float_type_columns = df.select_dtypes(include=['float']).columns
    print("Object_Type_Columns:\n",object_type_columns)
    print("\nFloat_Type_Columns:\n",float_type_columns)
    display('Before Label Encoding Dataframe')
    display(df)
    
    for o in object_type_columns:
        values = df[o].value_counts().idxmax()
        df[o] = df[o].fillna(value=values)
    
    for f in float_type_columns:
        values = df[f].mean()
        df[f] = df[f].fillna(value=values)
        
    from sklearn import preprocessing
    #from sklearn.preprocessing import OneHotEncoder
    le = preprocessing.LabelEncoder()
    #ohe = OneHotEncoder(handle_unknown='ignore')
    for i in object_type_columns:
        df[i] = le.fit_transform(df[i])
    display('Making Data ready for Fitting in Model')
    return df


# In[ ]:


preprocess_test(test_new)


# # Regression Model 

# ## Linear Regression

# In[ ]:


X = train.drop(['SalePrice'],axis=1)
#test1 = test[X.columns]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
y = np.array(train['SalePrice']).reshape(-1,1)
y = sc.fit_transform(y)


# In[ ]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[ ]:


'''viz_train = plt
viz_train.scatter(X_train, y_train, color='red')
viz_train.plot(X_train, regressor.predict(X_train), color='blue')
viz_train.show()

# Visualizing the Test set results
viz_test = plt
viz_test.scatter(X_test, y_test, color='red')
viz_test.plot(X_train, regressor.predict(X_train), color='blue')
viz_test.show()'''


# In[ ]:


regressor.score(X_train, y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# In[ ]:


a = mean_squared_error(y_test, y_pred)


# In[ ]:


rmse = a**0.5
rmse


# In[ ]:


from sklearn.tree import DecisionTreeRegressor  
  
# create a regressor object 
regressorTree = DecisionTreeRegressor(random_state = 0)  
  
# fit the regressor with X and Y data 
regressorTree.fit(X_train, y_train) 


# In[ ]:


regressorTree.score(X_train, y_train)


# In[ ]:


y_pred_tree = regressor.predict(X_test)


# In[ ]:


DTRrmse = (mean_squared_error(y_test, y_pred_tree))**0.5
DTRrmse


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

regressorGB = GradientBoostingRegressor(
    max_depth=5,
    n_estimators=10,
    learning_rate=0.8
)
regressorGB.fit(X_train, y_train)


# In[ ]:


errors = [mean_squared_error(y_test, y_pred) for y_pred in regressorGB.staged_predict(X_test)]
best_n_estimators = np.argmin(errors)


# In[ ]:


regressorGB.score(X_train, y_train)


# In[ ]:


y_predGB = regressorGB.predict(X_test)


# In[ ]:


GBrmse = (mean_squared_error(y_test, y_predGB))**0.5

GBrmse


# In[ ]:


predictions = regressorGB.predict(test_new)


# In[ ]:


predictions_transform = sc.inverse_transform(predictions)


# In[ ]:


predictions_transform


# In[ ]:


submission = pd.DataFrame(data = { 'Id' : test_new['Id'], 'SalePrice' : predictions_transform})


# In[ ]:


submission


# In[ ]:


submission.to_csv('Submission.csv', index=False)

