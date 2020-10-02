#!/usr/bin/env python
# coding: utf-8

#     Import the training and test data for the BigMart Sale

# In[100]:


import numpy as np
import pandas as pd

#Read files:
train = pd.read_csv("../input/Train.csv")
test = pd.read_csv("../input/Test.csv")


#     As we have different files containing the train the test data, we need to merge them. Print the shape/number of columns and rows of the merged dataset

# In[102]:


train['source']='train'
test['source']='test'
data = pd.concat([train, test],ignore_index=True)
print (train.shape, test.shape, data.shape)
print (data.head())


# Check whether the dataset has any missing columns

# In[103]:


data.apply(lambda x: sum(x.isnull()))


# In[104]:


data.describe()


# * Missing values are present in Item_Outlet_Sales and Item_Weight columns
# * Item_Outlet_Sales has 5681 missing entries and Item_Weight has 2439 missing entries.
# * Also the column present in the test data named "Outlet_Size" has missing values
# * The minimum value of "Item_Visibility" column is zero "0".

# In[105]:


data.apply(lambda x: len(x.unique()))


# The above output shows the following points:
# 1.   There are 1559 of different products
# 1.   There are 16 types of item types
# 1.   There are 10 different outlets

# In[106]:


#Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
#Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
#Print frequency of categories
for col in categorical_columns:
    print ('\nFrequency of Categories for varible %s'%col)
    print (data[col].value_counts())


# Following are the observations from the above results:
# 1.  For column "Item_Fat_Content" , there are 2 more values for  "Low Fat" .i.e "LF" and "low fat". These can be merged together. Similarly two values "Regular" and "reg" which is kind of redundant data
# 1. Too many categories in "Item_Type" column.  We can try to merge them into common category.

# Lets fill the missing for "Item_Weight" column by the mean weight of each item.

# In[107]:


itemAvgWt = data.pivot_table(values='Item_Weight', index='Item_Identifier')
getBooleanData = data['Item_Weight'].isnull() 
print ('Orignal #missing: %d'% sum(getBooleanData))
data.loc[getBooleanData,'Item_Weight'] = data.loc[getBooleanData,'Item_Identifier'].apply(lambda x: itemAvgWt.loc[x] )
print ('Final #missing: %d'% sum(data['Item_Weight'].isnull()))


# Lets fill the missing for "Outlet_Size" column by the mean weight of each item.

# In[109]:


#Import mode function:
from scipy.stats import mode

getBooleanData = data['Outlet_Size'].isnull() 
data['Outlet_Size'].fillna('Small',inplace=True)
outletSizeMode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x).mode[0]) )
print ('Mode for each Outlet_Type:')
print (outletSizeMode)

#Impute data and check #missing values before and after imputation to confirm
print ('\nOrignal #missing: %d'% sum(miss_bool))
data.loc[getBooleanData,'Outlet_Size'] = data.loc[getBooleanData,'Outlet_Type'].apply(lambda x: outletSizeMode.loc[x])
print (sum(data['Outlet_Size'].isnull()))


# In[110]:


data.pivot_table(values='Item_Outlet_Sales',index='Outlet_Type')


# Remove the values having "0" in column Item_Visibility

# In[111]:


#Determine average visibility of a product
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')

#Impute 0 values with mean visibility of that product:
miss_bool = (data['Item_Visibility'] == 0)

print ('Number of 0 values initially: %d'%sum(miss_bool))
data.loc[miss_bool,'Item_Visibility'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg.loc[x])
print ('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))


#     The columns "Item_Type" has 16 different categories. Lets club them together fewer amount of categories

# In[112]:


data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
print(data['Item_Type_Combined'].value_counts())
#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()


#     Lets determine the operation years of the store

# In[114]:


#Years:
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()


# Lets modify the redundant entries of "Item_Fat_Content" into similar ones

# In[115]:


#Change categories of low fat:
print ('Original Categories:')
print (data['Item_Fat_Content'].value_counts())

print ('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print (data['Item_Fat_Content'].value_counts())


# In[116]:


#Mark non-consumables as separate category in low_fat:
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()


#         Lets start encoding the non-numeric columns 

# In[117]:


#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])


# In[118]:


#One Hot Coding:
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])
data.dtypes


# In[119]:


data[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10)


# In[120]:


#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)


# In[121]:


#Mean based:
mean_sales = train['Item_Outlet_Sales'].mean()

#Define a dataframe with IDs for submission:
base1 = test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales

#Export submission file
base1.to_csv("alg0.csv",index=False)


# In[123]:


#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
from sklearn import cross_validation, metrics

def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print ("\nModel Report")
    print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)


# In[124]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
alg1 = LinearRegression(normalize=True)
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')


# In[ ]:




