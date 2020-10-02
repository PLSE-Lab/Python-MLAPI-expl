#!/usr/bin/env python
# coding: utf-8

# # Black Friday Sales Analysis

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
sns.set_context('talk')


# #### Loading the Dataset 

# In[ ]:



test = pd.read_csv("../input/blackfriday/test.csv")
train = pd.read_csv("../input/blackfriday/train.csv")


# In[ ]:





# #### Removing the Outliers 

# In[ ]:


sns.boxplot(train["Purchase"], orient='v')


# In[ ]:


q1,q3 = np.percentile(train["Purchase"], [25,70])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
print(lower_bound)
print(upper_bound)


# In[ ]:


train.drop(train[train["Purchase"] > 20085.5].index, inplace = True)


# ### Adding a new column User_Category
# 
# 

# The total (train + test) number of Unique User_ID = 5891 <br>
# The total number of Unique User_ID in the train dataset = 5891<br>
# The total number of Unique User_ID in the test dataset= 5891
# 
# 

# In[ ]:


user_id_mapping = {}
average_purchase_per_customer = train.groupby('User_ID')['Purchase'].mean()
values = average_purchase_per_customer.iteritems()
np.percentile(average_purchase_per_customer, [5, 20, 50, 85, 100])


# In[ ]:


for key, val in values:
    if val <= 6702:
        user_id_mapping[key] = 1
    elif val <= 7798:
        user_id_mapping[key] = 2
    elif val <= 9069:
        user_id_mapping[key] = 3
    elif val <= 10996:
        user_id_mapping[key] = 4
    else:
        user_id_mapping[key] = 5   

def get_customer_category(user_id):
    if user_id in user_id_mapping:
        return user_id_mapping[user_id]
    return 3


# In[ ]:


train['User_Category'] = [get_customer_category(train['User_ID'][i]) for i in train.index]
test['User_Category'] = [get_customer_category(test['User_ID'][i]) for i in test.index]


# ### Adding a new column Product_Category 

# The total (train + test) number of Unique Product_ID = 3631 <br>
# The total number of Unique Product_ID in the train dataset = 3631<br>
# The total number of Unique Product_ID in the test dataset= 3631
# 

# In[ ]:


product_id_mapping = {}
product_id_avg_purchase = train.groupby('Product_ID')['Purchase'].mean()
values = product_id_avg_purchase.iteritems()
np.percentile(product_id_avg_purchase, [30, 60, 75, 90, 100])


# In[ ]:


for key, val in values:
    if val <= 5792:
        product_id_mapping[key] = 1
    elif val <= 7527:
        product_id_mapping[key] = 2
    elif val <= 10069:
        product_id_mapping[key] = 3
    elif val <= 13562:
        product_id_mapping[key] = 4
    else:
        product_id_mapping[key] = 5 
    
def get_product_category(product_id):
    if product_id in product_id_mapping:
       return product_id_mapping[product_id]
    return 2


# In[ ]:


train['Product_Category'] = [get_product_category(train['Product_ID'][i]) for i in train.index]
test['Product_Category'] = [get_product_category(test['Product_ID'][i]) for i in test.index]


# #### Combining the train and test datasets 

# In[ ]:


train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test], sort=False)


# Dealing with missing values

# In[ ]:


data.isnull().sum()


# As an item can belong to more than one product category we are filling the missing values in Product_Category_2 and Product_Category_3 with -2 

# In[ ]:


data['Product_Category_2']= data['Product_Category_2'].fillna(-2).astype("int")
data['Product_Category_3']= data['Product_Category_3'].fillna(-2).astype("int")


# ## 1. Data Exploration

# In[ ]:


sns.distplot(train["Purchase"])


# In[ ]:


plt.figure(figsize = (23, 4))

plt.subplot2grid((1, 3), (0, 0))
sns.barplot('Age', 'Purchase', data = train, order = ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'])

plt.subplot2grid((1, 3), (0, 1))
sns.barplot('City_Category', 'Purchase', data = train, order = ['A', 'B', 'C'])

plt.subplot2grid((1, 3), (0, 2))
sns.barplot('Gender', 'Purchase', data = train)


# In[ ]:


plt.figure(figsize = (20, 6))

plt.subplot2grid((1, 2), (0, 0))
sns.countplot(data['Product_Category_1'])

plt.subplot2grid((1, 2), (0, 1))
sns.barplot(data['Product_Category_1'], data['Purchase'])


# In[ ]:


plt.figure(figsize = (20, 6))

plt.subplot2grid((1, 2), (0, 0))
sns.countplot(data['Product_Category_2'])

plt.subplot2grid((1, 2), (0, 1))
sns.barplot(data['Product_Category_2'], data['Purchase'])


# In[ ]:


plt.figure(figsize = (20, 6))

plt.subplot2grid((1, 2), (0, 0))
sns.countplot(data['Product_Category_3'])

plt.subplot2grid((1, 2), (0, 1))
sns.barplot(data['Product_Category_3'], data['Purchase'])


# In[ ]:


plt.figure(figsize = (9,9))
sns.set(font_scale=1)
sns.heatmap(train.corr(), annot=True, cmap="Blues")


# ## 2. Data Pre-Processing

# #### Label encoding or creating dummies for Categorical Variables

# Gender 

# In[ ]:


gender_dict = {'F': 0, 'M': 1}
data['Gender'] = data['Gender'].apply(lambda line: gender_dict[line])
data['Gender'].value_counts()


# Label encoding age with a mid value in range. <br>
# An assumption made : 0-17 age group is considered as a teenage group from 13-17

# In[ ]:


Age_dict = {'0-17': 15, '18-25': 21, '26-35': 30, '36-45': 40, '46-50': 48, '51-55': 53, '55+': 60}
data['Age'] = data['Age'].map(Age_dict)
data['Age'].value_counts()


# In[ ]:


Stay_In_Current_City_Years_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4+': 4}
data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].map(Stay_In_Current_City_Years_dict)
data['Stay_In_Current_City_Years'].value_counts()


# In[ ]:


data = pd.get_dummies(data,columns=["City_Category"],drop_first=False)


# In[ ]:


Product_Price_mean = data.groupby(['Product_ID'])['Purchase'].agg(['mean']).reset_index()
Product_Price_mean.rename(columns ={'mean': 'Product_Price_mean'}, inplace = True)
Product_Price_mean.head()


# In[ ]:


data = pd.merge(data, Product_Price_mean)
data.loc[data['Product_Price_mean']!=data['Product_Price_mean'],'Product_Price_mean'] = 0


# <br>

# In[ ]:


train = data.loc[data['source'] == 'train']
test = data.loc[data['source'] == 'test']


# In[ ]:


submission = pd.DataFrame()
submission['User_ID'] = test['User_ID']
submission['Product_ID'] = test['Product_ID']


# In[ ]:


test.drop(['source','Purchase','Product_ID','User_ID'],axis=1,inplace=True)
train.drop(["source"],axis=1,inplace=True)


# In[ ]:





# ## 3. Data Modelling

# In[ ]:


from sklearn.model_selection import train_test_split
X = train.drop(['Purchase','Product_ID','User_ID' ], axis =1)
y = train['Purchase']
X.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state=942)


# #### A function to score the Models 

# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

df = pd.DataFrame(columns=['Model Name','RMSE Score','R2 score'])

def modelscore(target_test, predicted, i , df):
    
    #RMSE
    mse = mean_squared_error(target_test, predicted)
    rmse = np.sqrt(mse)
    
    #R2 score 
    r2score = r2_score(target_test, predicted)
    
    if(i == 0):
        df = df.append({'Model Name': 'Linear Regressio', 'RMSE Score': rmse, 'R2 score': r2score}, ignore_index=True)
    elif (i == 1):
        df = df.append({'Model Name': 'Ridge Regression', 'RMSE Score': rmse, 'R2 score': r2score}, ignore_index=True)
    elif (i == 2):
        df = df.append({'Model Name': 'Lasso Regression', 'RMSE Score': rmse, 'R2 score': r2score}, ignore_index=True)
    elif (i == 3):
        df = df.append({'Model Name': 'Elastic Net Regression', 'RMSE Score': rmse, 'R2 score': r2score}, ignore_index=True)
    elif (i == 4):
        df = df.append({'Model Name': 'Decision Tree Regressor', 'RMSE Score': rmse, 'R2 score': r2score}, ignore_index=True)
    print (df)
    return df


# ### 3.1 Linear Regression 

# In[ ]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression(normalize=True)
LR.fit(X_train, y_train)


# In[ ]:


LRpred = LR.predict(X_test)   
scores = cross_val_score(LR, X, y, scoring = 'r2', cv = 5)
print()
print ("Linear Regression's Cross Validation Score:", np.sqrt(scores).mean())
print()
df = modelscore(y_test, LRpred, 0, df)


# In[ ]:


rankings = LR.coef_.tolist()
features = list(X)
d = dict(zip(features,rankings))
d = dict(zip(features,rankings))
d = pd.DataFrame(list(d.items()), columns=["features", "ranking"])
d = dict(zip(features,rankings))
d = pd.DataFrame(list(d.items()), columns=["features", "ranking"])
d.sort_values(["ranking"], ascending=False)


# ### 3.2 Ridge Regression 

# In[ ]:


alphas = 10**np.linspace(10,-2,100)*0.5
from sklearn.linear_model import RidgeCV
ridgecv = RidgeCV(alphas = alphas, scoring = 'r2', normalize = True)
ridgecv.fit(X, y)
ridgecv.alpha_


# In[ ]:


from sklearn.linear_model import Ridge
RR = Ridge(alpha = ridgecv.alpha_, normalize = True)
RR.fit(X_train, y_train)


# In[ ]:


RRpred = RR.predict(X_test)
scores = cross_val_score(RR, X, y, scoring = 'r2', cv = 5)
print()
print ("Ridge Regression's Cross Validation Score:", np.sqrt(scores).mean())
print()
df = modelscore(y_test, RRpred, 1, df)


# In[ ]:


rankings = RR.coef_.tolist()
features = list(X)
d = dict(zip(features,rankings))
d = dict(zip(features,rankings))
d = pd.DataFrame(list(d.items()), columns=["features", "ranking"])
d = dict(zip(features,rankings))
d = pd.DataFrame(list(d.items()), columns=["features", "ranking"])
d.sort_values(["ranking"], ascending=False)


# ### 3.3 Lasso Regression 

# In[ ]:


alphas = 10**np.linspace(10,-2,100)*0.5
from sklearn.linear_model import LassoCV
lassocv = LassoCV(alphas = alphas, normalize = True)
lassocv.fit(X, y)
lassocv.alpha_


# In[ ]:


from sklearn.linear_model import Lasso
LaR = Lasso(alpha = lassocv.alpha_,normalize = True)
LaR.fit(X_train, y_train)


# In[ ]:


LaRpred = LaR.predict(X_test)

scores = cross_val_score(LaR, X, y, scoring = 'r2', cv = 5)
print()
print ("Lasso Regression's Cross Validation Score:", np.sqrt(scores).mean())
print()

df = modelscore(y_test, LaRpred, 2, df)


# In[ ]:


rankings = LaR.coef_.tolist()
features = list(X)
d = dict(zip(features,rankings))
d = dict(zip(features,rankings))
d = pd.DataFrame(list(d.items()), columns=["features", "ranking"])
d = dict(zip(features,rankings))
d = pd.DataFrame(list(d.items()), columns=["features", "ranking"])
d.sort_values(["ranking"], ascending=False)


# ### 3.4 Elastic Net 

# In[ ]:


from sklearn.linear_model import ElasticNet
ENR = ElasticNet(alpha = 0.00001, l1_ratio = 0.9, max_iter = 5, normalize = True)
ENR.fit(X_train,y_train)


# In[ ]:


ENRpred = ENR.predict(X_test)
scores = cross_val_score(ENR, X, y, scoring = 'r2', cv = 5)
print()
print ("Elastic Net Regression's Cross Validation Score:", np.sqrt(scores).mean())
print()
df = modelscore(y_test, ENRpred, 3, df)


# In[ ]:


rankings = ENR.coef_.tolist()
features = list(X)
d = dict(zip(features,rankings))
d = dict(zip(features,rankings))
d = pd.DataFrame(list(d.items()), columns=["features", "ranking"])
d = dict(zip(features,rankings))
d = pd.DataFrame(list(d.items()), columns=["features", "ranking"])
d.sort_values(["ranking"], ascending=False)


# ### 3.5 Decision Tree Regressor 

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth = 8, min_samples_leaf = 100, min_samples_split= 2, max_leaf_nodes = 500)
model.fit(X_train, y_train)


# In[ ]:


dtrpred = model.predict(X_test)
scores = cross_val_score(model, X, y, scoring = 'r2', cv = 5)
print()
print ("Decision Tree Regression's Cross Validation Score:", np.sqrt(scores).mean())
print()
df = modelscore(y_test, dtrpred, 4, df)


# In[ ]:


rankings = model.feature_importances_.tolist()
features = list(X)
d = dict(zip(features,rankings))
d = dict(zip(features,rankings))
d = pd.DataFrame(list(d.items()), columns=["features", "ranking"])
d = dict(zip(features,rankings))
d = pd.DataFrame(list(d.items()), columns=["features", "ranking"])
d.sort_values(["ranking"], ascending=False)


# In[ ]:


from xgboost.sklearn import XGBRegressor
xgb_reg = XGBRegressor(learning_rate = 0.01 , n_estimators = 500, max_depth= 8)
xgb_reg.fit(X_train, y_train)


# In[ ]:


y_pred = xgb_reg.predict(X_test)
df = modelscore(y_test, y_pred, 4, df)


# In[ ]:


test_target = xgb_reg.predict(test)  
submission['Purchase'] = test_target
submission.to_csv  ('checkscore.csv', index= False)


# RMSE on Test Data set: 2670.2065114129
# 

# In[ ]:





# In[ ]:


import pandas as pd
test = pd.read_csv("../input/blackfriday/test.csv")
train = pd.read_csv("../input/blackfriday/train.csv")

