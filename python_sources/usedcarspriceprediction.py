#!/usr/bin/env python
# coding: utf-8

# ## Used Cars Price Prediction

# ### Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# ### Importing Data

# In[ ]:


train = pd.read_excel("../input/Data_Train.xlsx")
test  = pd.read_excel("../input/Data_Test.xlsx")


# In[ ]:


print('Train Shape:',train.shape)
train.head(2)


# In[ ]:


print('Test Shape: ',test.shape)
test.head(2)


# #### Data Properties
# Checking for na values

# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# ### Preprocessing Data

# In[ ]:


# large proportion of New_Price is Empty, Can't think of any other Solution so dropping it.
def dropNewPrice(x):
    x.drop('New_Price',axis=1, inplace=True)


# In[ ]:


# Spliting "Name" Feature to 'Brand', 'CarName' and 'Model'.
def splitName(x):
    x['Brand']   = x['Name'].apply(lambda x: x.split(' ')[0].strip())
    x['CarName'] = x['Name'].apply(lambda x: x.split(' ')[1].strip())
    x['Model'] = x['Name'].apply(lambda x:' '.join(x.split(' ')[2:]))
    x.drop(['Name'],axis=1, inplace=True)


# #### Splitting Features and Imputing NaN values

# In[ ]:


#Splitting Power, Engine, & Mileage to remove Units    
def splitIn(x):
    x['Power'  ].replace('null bhp',np.nan,inplace=True)
    x['Mileage'].replace('0.0 kmpl',np.nan,inplace=True)
    for i in ['Power', 'Engine', 'Mileage']:      
        x[i] = x[i].apply(lambda x: float(x.split()[0].strip()) if not pd.isna(x) else x)


# In[ ]:


#Imputing Power, Engine, Seats & Mileage by grouping.
def imputeNaN(x):
    for i in ['Power', 'Engine', 'Seats','Mileage']:
        x[i] = x.groupby(['Model'])[i].transform(lambda y: y.fillna(y.mean()))
        #Some Values will still be left with na.
        x[i].fillna(x[i].mean(), inplace=True)
    


# In[ ]:


def preprocessData(data):
    dropNewPrice(data)
    splitName(data)
    splitIn(data)
    imputeNaN(data)


# In[ ]:


preprocessData(train)
preprocessData(test)


# In[ ]:


train.head()


# ### Looking for Outliers

# In[ ]:


_, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,7))
sns.boxplot( x=train['Kilometers_Driven'],ax=axes[0][0])
sns.boxplot( x=train['Mileage']          ,ax=axes[0][1])
sns.boxplot( x=train['Engine']           ,ax=axes[1][0])
sns.boxplot( x=train['Power']            ,ax=axes[1][1])


# #### Getting rid of outliers by Quantile filtering

# In[ ]:


def filterByQuantiles(x):
    for i in ['Kilometers_Driven', 'Mileage', 'Power', 'Engine']:
        upper_lim = x[i].quantile(.98)
        lower_lim = x[i].quantile(.02)
        return x[(x[i] < upper_lim) & (x[i] > lower_lim)]


# In[ ]:


# def logTrans(x):
#     for i in ['Mileage', 'Power', 'Engine']:
#         x[i] = x[i].transform(np.log)


# In[ ]:


# def binCols(x):
#     for i in ['Kilometers_Driven', 'Mileage', 'Power', 'Engine']:
#         x[i] = pd.qcut(x[i],5).astype(str)


# In[ ]:


train = filterByQuantiles(train)
# logTrans(train)
# logTrans(test)


# ### Encoding Categorical Values

# #### Label Encoding:  'Owner_Type'

# In[ ]:


train['Owner_Type'].replace(['First', 'Second', 'Fourth & Above', 'Third'],[0,1,3,2],inplace=True)
test['Owner_Type'].replace( ['First', 'Second', 'Fourth & Above', 'Third'],[0,1,3,2],inplace=True)


# #### Combining Train & Test

# In[ ]:


#Creating a Flag Column in both datasets
train['train'] =1
test['train']  =0
combined = pd.concat([train,test])


# In[ ]:


#Reducing no. of Categorical Values
def changeBrandLabel(x):
    for col in ['Brand', 'Model','CarName']:
        threshold = 20 if col=='Model' else 60
        counts = x[col].value_counts()
        mask = x[col].isin(counts[counts > threshold].index)
        x.loc[~mask, col] = "Other"


# In[ ]:


changeBrandLabel(combined)


# Dummy Encoding of Categorical Values

# In[ ]:


df = pd.get_dummies(data    = combined,                                                                                 columns = ['Brand', 'Model', 'Location', 'Fuel_Type', 'Transmission','CarName'],                    drop_first = True )


# In[ ]:


print('New Columns:',df.shape[1])


# #### Splitting Train & Test

# In[ ]:


train = df[df['train'] == 1]
train.drop(['train'], axis=1, inplace=True)

test = df[df['train'] == 0]
test.drop(['train','Price'], axis=1, inplace=True)


# In[ ]:


print(train.shape)


# In[ ]:


print(test.shape)


# ### Splitting data into Dependent & Independent variable sets

# In[ ]:


# Dependent Variable
Y_train_data = train.loc[:,['Price']].values
Y_train_data = np.log1p(train.Price)

# Independent Variables of Train Set
X_train_data = train.loc[:,train.columns != 'Price'].values

# Independent Variables of Test Set
X_test = test.iloc[:,:].values


# #### Splitting Train data Into Training & Validation Set

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train_data, Y_train_data, test_size = 0.30, random_state = 1)


# ### Feature Scaling

# In[ ]:


# #Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()

# #Scaling Original Training Data
# X_train_data = sc.fit_transform(X_train_data)


# In[ ]:


# #Reshaping vector to array for transforming
# Y_train_data = Y_train_data.reshape((len(Y_train_data), 1))
# Y_train_data = sc.fit_transform(Y_train_data)
# #converting back to vector
# Y_train_data = Y_train_data.ravel()


# In[ ]:


# X_test = sc.transform(X_test)

# # Scaling Splitted training and val sets
# X_train = sc.fit_transform(X_train)
# X_val = sc.fit_transform(X_val)

# #Reshaping vector to array for transforming
# Y_train = Y_train.reshape((len(Y_train), 1)) 
# Y_train = sc.fit_transform(Y_train)
# #converting back to vector
# Y_train = Y_train.ravel()


# ### Model Training

# In[ ]:


# from sklearn.ensemble import RandomForestRegressor

# #Initializing regressor
# my_model = RandomForestRegressor(random_state=1)

# #Fitting the regressor on Train data
# my_model.fit(X_train,Y_train)

# #Predicting for Validation data
# # Y_pred = sc.inverse_transform(my_model.predict(X_val))
# Y_pred = my_model.predict(X_val)


# In[ ]:


from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=4, min_child_weight=2, n_jobs=4)
my_model.fit(X_train, Y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_val, Y_val)], 
             verbose=False)
Y_pred = my_model.predict(X_val)


# ### Model Evaluation

# In[ ]:


def score(y_pred, y_true):
    error = np.square(np.log10(y_pred +1) - np.log10(y_true +1)).mean() ** 0.5
    score = 1 - error
    return score


# In[ ]:


#Eliminating negative values in prediction for score calculation
for i in range(len(Y_pred)):
     if Y_pred[i] < 0:
        Y_pred[i] = 0
        
y_true = Y_val


# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from statistics import mean
        
print("Score: ",score(Y_pred,y_true))
print("Mean Absolute Error:",mean_absolute_error(y_true, Y_pred))
# print("Cross Validation Score:",mean(cross_val_score(my_model, X_train, Y_train, cv=5)))


# ### Model Training on Full Train Set

# In[ ]:


# #Initializing a new regressor
# rf2 = RandomForestRegressor()

# #Fitting the regressor with complete training data(X_train_data,Y_train_data)
# rf2.fit(X_train_data,Y_train_data)

# #Predicting the target(Price) for predictors in the test data
# Y_pred2 = rf2.predict(X_test)

# #converting target to original state
# Y_pred2 = np.exp(Y_pred2)-1 

# Y_pred2 = Y_pred2.round(2)
# #Eliminating negative values in prediction for score calculation
# for i in range(len(Y_pred2)):
#     if Y_pred2[i] < 0:
#         Y_pred2[i] = 0


# In[ ]:


# Y_pred2 = my_model.predict(X_test)
# Y_pred2 = np.exp(Y_pred2)-1 

# Y_pred2 = Y_pred2.round(2)
# #Eliminating negative values in prediction for score calculation
# for i in range(len(Y_pred2)):
#     if Y_pred2[i] < 0:
#         Y_pred2[i] = 0


# ### Saving Prediction Results

# In[ ]:


# pd.DataFrame(Y_pred2, columns = ['Price']).to_excel("predictions.xlsx", index=False)

