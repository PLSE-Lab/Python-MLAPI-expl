#!/usr/bin/env python
# coding: utf-8

#  As usuall I checked the training data set, treated the missing values, encoded the categorical values
#  and created the model.
#   Then i went ahead with the test data set(submission data). After creating the dummy variables i realised the features of the two data sets does not match. Further more there were categorical values which were in the test set and not in the training set and viceversa these values must be removed. 
#   
#    1. Training model will not see this values
#    2. Features will not match with the training and the testing data after encoding the categorical variables  
#    
#   Feature engineering must be done for both data sets hand in hand 
#   
#   By using Lasso selected the most important featues out of 250+ columns and created a two models using RandomForest 
#   (which was overfitting with trainning and test scores of 97 and 86) I think the best model is using Gradient boosting
#   scores of 94 and 91.
#   
#     Will be doing a model with Catboost, since we dont need to encode categorical variables it might be quite interesting to compare the performance. 
#     
#   
#   
#   
#      
# 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#loading Data
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


print(df.shape)
print(df_test.shape)


# In[ ]:


#pd.options.display.max_columns = None
df.head(5)


# In[ ]:


df.describe()


# In[ ]:


# get only null columns
nullcol = df.columns[df.isna().any()]


# In[ ]:


df[nullcol].isnull().sum()


# missing values as a percentage 

# In[ ]:


df[nullcol].isnull().sum() * 100 / len(df)


# In[ ]:


#dropping the columns where the missing values are over 40%

df.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis=1, inplace=True)


# In[ ]:


df.shape


# In[ ]:


#checking the data types of missing value columns
nullcol = df.columns[df.isna().any()]
df[nullcol].dtypes


# In[ ]:


# selecting columns where the type is strings with missing values

objcols= df[nullcol].select_dtypes(['object']).columns
objcols


# In[ ]:


#replacing the missing values of the strings with the mode
df[objcols] = df[objcols].fillna(df.mode().iloc[0])


# In[ ]:


#checking the columns
df[objcols].isnull().sum() * 100 / len(df)


# In[ ]:


#imputing numeric values

#get numerical features by dropping categorical features from the list
num_null=(nullcol.drop(objcols))

df[num_null] = df[num_null].fillna(df.mean().iloc[0])


# In[ ]:



df.columns[df.isna().any()]


# In[ ]:


#numerical data
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

num_cols = df.select_dtypes(include=numerics)

#categorical data
string_cols = df.select_dtypes(exclude =numerics)


# In[ ]:



print(num_cols.shape)
print(string_cols.shape)


# In[ ]:


#correlaation of nymerical data
corr = num_cols.corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr,cmap='coolwarm')
    


# In[ ]:



num_cols[num_cols.columns[1:]].corr()['SalePrice']


# # Checking for highly correlated feaures

# In[ ]:


corr_matrix = num_cols.corr().abs()
high_corr_var=np.where(corr_matrix>0.8)
high_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]

high_corr_var


# In[ ]:


def corrank(X):
        import itertools
        dff = pd.DataFrame([[(i,j),X.corr().loc[i,j]] for i,j in list(itertools.combinations(X.corr(), 2))],columns=['pairs','corr'])    
        display(dff.sort_values(by='corr',ascending=False))

corrank(num_cols) # prints a descending list of correlation pair (Max on top)


# In[ ]:





# In[ ]:


df.groupby(['YrSold','MoSold']).Id.count().plot(kind='bar',figsize=(12,4))


# In[ ]:


# this show a seasonal pattren where in 6,7 months the sales rises


# In[ ]:


chart = pd.melt( df, value_vars = num_cols )

gp = sns.FacetGrid(chart, col = 'variable', col_wrap=4, sharex=False, sharey=False)
gp = gp.map(sns.distplot,'value')


# In[ ]:


# though the above distribution is for numerical features the bars means the values
## are not continuos but discrete similar to categorcial


# In[ ]:


#print unique values for categorical columns

for cols in string_cols:
         print( cols)
         print( df[cols].unique())   


# In[ ]:


string_cols.columns


# # checking the trainnig data set and the test data (submission data)
# 
# 

# In[ ]:


df_test.isnull().sum()


# In[ ]:


test_nullcol = df_test.columns[df_test.isna().any()]
df_test[test_nullcol].isnull().sum() * 100 / len(df_test)


# In[ ]:


#first imputing the missing values in the test data set

test_nullcol = df_test.columns[df_test.isna().any()]
df_test.isnull().sum()
df_test[test_nullcol].isnull().sum()

#string columns in test 
test_objcols= df_test[test_nullcol].select_dtypes(['object']).columns


#replacing the missing values of the strings with the mode
df_test[test_objcols] = df_test[test_objcols].fillna(df_test.mode().iloc[0])


#imputing numeric values

#get numerical features by dropping categorical features from the list
test_num_null=(test_nullcol.drop(test_objcols))

#replacing with the mean
df_test[test_num_null] = df[test_num_null].fillna(df_test.mean().iloc[0])


# In[ ]:


df_test.columns[df_test.isna().any()]


# In[ ]:


#drop features that were removed from trainig set
df_test.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis=1, inplace=True)


# In[ ]:


df_test.shape


# In[ ]:





# In[ ]:


#numerical data
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

#numerical features in the test set
test_num_cols = df_test.select_dtypes(include=numerics)

#categorical data features
test_string_cols = df_test.select_dtypes(exclude =numerics)


# In[ ]:





# check whether the featues of test and training match

# In[ ]:


#checking whether columns match one to one
set(test_string_cols.columns.values) - set(string_cols.columns.values)


# Need to check whether there are categorical values in only one data set and remove it
# 

# In[ ]:


# test and training
for cols in test_string_cols:
    no = set(df_test[cols].unique() ) - set( df[cols].unique()) 
    
    if len(no) > 0:
        print(no, " ",cols)
                


# In[ ]:


# training and test

for cols in test_string_cols:
    no =  set( df[cols].unique()) -set(df_test[cols].unique() )
    
    if len(no) > 0:
        print(no, " ",cols)


# need to remove this values from the training data set cause when generating dummy variables
# there would be a feature mismatch

# In[ ]:


for cols in test_string_cols:
    no =  set( df[cols].unique()) -set(df_test[cols].unique() )
    if len(no) > 0:
        arr=list(no)
        df.drop( df[df[cols].isin(arr)].index, inplace=True)


# In[ ]:


test_string_cols.columns


# In[ ]:


for cols in test_string_cols:
         print( cols)
         print( df_test[cols].unique())   


# In[ ]:


print(df.shape)
print(df_test.shape)


# In[ ]:


# two data sets have the same number of features (trainin has 1 more of the target )


# In[ ]:





# # lets build the the model on the training data set

# In[ ]:


#encoding catogorical columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[ ]:


#encoding all categorical features

df_encoded = df[string_cols.columns].apply(le.fit_transform)


# In[ ]:


df_encoded.head(5) 


# In[ ]:


#Creating Dummy variables 

dummy = pd.get_dummies(df_encoded, columns= df_encoded.columns)


# In[ ]:


display(dummy.tail(5))


# In[ ]:


print(dummy.shape)
print(df.shape)


# In[ ]:


# merge the dummy variables to the data set 
df = pd.concat([df,dummy], axis=1)


# In[ ]:


# removing the orginal string columns since the values are represented in the dummy columns
df.drop(string_cols.columns, axis=1, inplace=True)


# In[ ]:


#removeing Id'
del df['Id']


# In[ ]:


df.head()


# In[ ]:


print("Shape of the Final Dataset", df.shape)


# In[ ]:





# In[ ]:


plt.figure(figsize=(10,6))
sns.regplot(df['GrLivArea'],df['SalePrice'] )
plt.show()


# In[ ]:


#remove outliers - houses that are more than 4000sft

df.drop(df[df.GrLivArea > 4000].index,inplace=True)


# In[ ]:


#assiging X and target label y
y = df['SalePrice']
X = df.drop('SalePrice',axis=1)


# In[ ]:


print(y.shape)
print(X.shape)


# In[ ]:


#splitting data to training and testing
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=40)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


#getting numeric columns for standerdising
num_cols.drop('Id',axis=1,inplace=True)
num_cols.drop('SalePrice',axis=1,inplace=True)


# In[ ]:


num_cols.columns
X_test[num_cols.columns]


# In[ ]:


#standerdiasing
#standardising  only the numerical columns [excluding dummy varables]

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()


X_train[num_cols.columns] = scale.fit_transform( X_train[num_cols.columns])
X_test[num_cols.columns] = scale.fit_transform( X_test[num_cols.columns])


# In[ ]:


print(X_test.shape)
print(X_train.shape)


# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# In[ ]:


param_grid ={'alpha':[0.1,1,5,10,25],'max_iter':[50000]}
lasso = GridSearchCV(Lasso(), cv=5, param_grid=param_grid, scoring='neg_mean_squared_error')
lasso.fit(X_train, y_train)
alpha = lasso.best_params_['alpha']


# In[ ]:


alpha


# In[ ]:


#using Lasso to get important features
lmodel = Lasso(alpha=25 ).fit(X_train, y_train)

print("Train Score ",lmodel.score(X_train,y_train))
print("Test Score ",lmodel.score(X_test,y_test))

print("Number of Features Used ",np.sum(lmodel.coef_ !=0))


#  model achives a training score of 94 and testing score of 90
#  and is not overfitting
# 

# In[ ]:


X_train.shape


# In[ ]:



important_f = pd.DataFrame(X.columns.values,columns =['Features'])
important_f['wht']= lmodel.coef_
important_f['Abs']= important_f['wht'].abs()
important_f = important_f.sort_values( by =['Abs'], ascending = False)

ColsUsed  = important_f[important_f['Abs'] !=0 ]
ColsUsed['Features']


# In[ ]:


#using random forest with selected features
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 10, random_state = 42)
rf.fit(X_train[ColsUsed['Features']], y_train)


# In[ ]:





# In[ ]:


print("Train Score :", rf.score(X_train[ColsUsed['Features']], y_train))
print("Test  Score :",rf.score(X_test[ColsUsed['Features']], y_test))


# In[ ]:


#using gradient boosting with 161 features out of 300+
import xgboost as xgb

regr =xgb.XGBRegressor(colsample_bytree=0.2, gamma=0, learning_rate=0.05,max_depth=5,
                      min_child_weight=1,n_estimators=100,reg_alpha=0.9,
                      reg_lambda=0.6,subsample=0.2,seed=42,silent=1)


# In[ ]:


regr.fit( X_train[ColsUsed['Features']],  y_train)


# In[ ]:


print(regr.score(X_train[ColsUsed['Features']],y_train))
print(regr.score(X_test[ColsUsed['Features']],y_test))


# In[ ]:


# training score of 94 and testing 91


# # working on submission data set

# In[ ]:


df_test.shape


# In[ ]:


test_string_cols


# In[ ]:


#encoding the test set categorcial columns

df_test_encoded = df_test[test_string_cols.columns].apply(le.fit_transform)


# In[ ]:


#Creating Dummy variables 

dummy = pd.get_dummies(df_test_encoded, columns= df_test_encoded.columns)


# In[ ]:


# merge the dummy variables to the data set 
df_test = pd.concat([df_test,dummy], axis=1)


# In[ ]:


print(df_test.shape)
print(dummy.shape)


# In[ ]:


# removing the orginal string columns since the values are represented in the dummy columns
df_test.drop(test_string_cols.columns, axis=1, inplace=True)


# In[ ]:


print(df_test.shape)


# In[ ]:


print(df_test.shape)


# In[ ]:


# create output dataframe with the prediction as SalesPrice and the Id for the important features

Output =pd.DataFrame(regr.predict(df_test[ColsUsed['Features']]),columns=['SalePrice'], index = df_test['Id'] )


# In[ ]:


Output.head()


# In[ ]:





# In[ ]:


Output.to_csv('Submission.csv')

