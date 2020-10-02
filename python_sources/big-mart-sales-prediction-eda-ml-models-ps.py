#!/usr/bin/env python
# coding: utf-8

# Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Upload Datasets

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/big-mart-sales-prediction/train_v9rqX0R.csv')
test = pd.read_csv('/kaggle/input/big-mart-sales-prediction/test_AbJTz2l.csv')


# Exploring Data Information

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape, test.shape


# In[ ]:


train.columns, test.columns


# In[ ]:


train.info(), test.info()


# In[ ]:


train.isnull().sum() , test.isnull().sum()


# In[ ]:


train.dtypes, test.dtypes


# ### Combine test and train data

# In[ ]:


df = pd.concat([train, test], axis = 0)


# In[ ]:


df.shape


# ## Null values

# Columns containing null values are: Item_weight and Outlet-size

# In[ ]:


df.isnull().sum()


# In[ ]:


df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())


# In[ ]:


df.isnull().sum()


# In[ ]:


sns.heatmap(df.isnull(), yticklabels = False, cmap = 'viridis', cbar = False)


# The test data does not contain the target variable i.e. Item_Outlet_Sales hence show the null values. 

# # EDA
# * Univariate Analysis
# * Bivariate analysis

# ### Univariate Analysis

# 1. **Target Variable**

# In[ ]:


sns.distplot(train['Item_Outlet_Sales'], kde = False, hist_kws={ "linewidth": 3, "alpha": 1, "color": "g"})


# In[ ]:


obj_col = df.select_dtypes('object').columns
obj_col


# In[ ]:


num_col = df.select_dtypes(exclude='object').columns
num_col


#  <font color='red'>2.  **Numerical Variable**

# In[ ]:


fig = plt.figure(figsize = (15, 4))
for i in range(1,4):
    plt.subplot(1, 3, i)
    sns.distplot(df[df[['Item_MRP', 'Item_Visibility', 'Item_Weight']].columns[i-1]], kde = False, color = 'b')


# **Observations**   
# 
# * There seems to be no clear-cut pattern in Item_Weight. The single value at 12.79 is due to the filling with the average. so all the missing values are now replaced with the 12.79. 
# * Item_Visibility is right-skewed and should be transformed to curb its skewness.
# * We can clearly see 4 different distributions for Item_MRP. It is an interesting insight.
# 

# 3. Categorical Variables

# In[ ]:


sns.countplot(df['Item_Fat_Content'])


# **Observations**  
# * In the Item_Fat_Content, there are different categories refering to Low Fat and Regular, so we have to combine them.

# In[ ]:


df['Item_Fat_Content'].value_counts()


# In[ ]:


df['Item_Fat_Content'] = df['Item_Fat_Content'].map({'LF': 'Low Fat', 'low fat' : 'Low Fat', 'reg': 'Regular', 'Low Fat': 'Low Fat', 'Regular': 'Regular'})


# In[ ]:


df['Item_Fat_Content'].value_counts()


# In[ ]:


plt.figure(figsize = (15, 4))
sns.countplot(df['Item_Type'])
plt.xticks(rotation = 45)
plt.show();


# In[ ]:


plt.figure(figsize = (15, 4))

plt.subplot(1,2,1)
sns.countplot(df['Outlet_Identifier'])
plt.xticks(rotation = 45)

plt.subplot(1,2,2)
sns.countplot(df['Outlet_Size'])
plt.xticks(rotation = 45)


# In[ ]:


plt.figure(figsize = (15, 4))

plt.subplot(1,2,1)
sns.countplot(df['Outlet_Establishment_Year'])
plt.xticks(rotation = 45)

plt.subplot(1,2,2)
sns.countplot(df['Outlet_Type'])
plt.xticks(rotation = 45)


# **Observations**  
# 
# * Lesser number of observations in the data for the outlets established in the year 1998 as compared to the other years.
# * Supermarket Type 1 seems to be the most popular category of Outlet_Type.

# 4. Target Variable vs Independent Numerical Variables

# In[ ]:


plt.figure(figsize = (15, 5))
sns.scatterplot(x = 'Item_Weight', y = 'Item_Outlet_Sales', data = df)


# In[ ]:


plt.figure(figsize = (15, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x = 'Item_Visibility', y = 'Item_Outlet_Sales', data = df)

plt.subplot(1,2,2)
sns.scatterplot(x= 'Item_MRP', y = 'Item_Outlet_Sales', data = df)


# **Observations**  
# 
# * Item_Outlet_Sales is spread well across the entire range of the Item_Weight without any obvious pattern.
# * In Item_Visibility vs Item_Outlet_Sales, there is a string of points at Item_Visibility = 0.0 which seems strange as item visibility cannot be completely zero. We will take note of this issue and deal with it in the later stages.
# * In the third plot of Item_MRP vs Item_Outlet_Sales, we can clearly see 4 segments of prices that can be used in feature engineering to create a new variable.

# 5. Target Variable vs Independent Categorical Variables

# In[ ]:


fig = plt.figure(figsize = (15, 12))

ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
sns.boxplot(x = 'Item_Type', y = 'Item_Outlet_Sales', data = df, ax=ax1)
plt.xticks(rotation = 45)
ax1.set_title(' Target vs Item_Type')

ax2 = plt.subplot2grid((2,2), (1,0), colspan=1)
sns.boxplot(x = 'Item_Fat_Content', y = 'Item_Outlet_Sales', data = df, ax=ax2)
plt.xticks(rotation = 45)
ax2.set_title(' Target vs Item_Fat_Content')

ax3 = plt.subplot2grid((2,2), (1,1), colspan=1)
sns.boxplot(x = 'Outlet_Identifier', y = 'Item_Outlet_Sales', data = df, ax=ax3)
plt.xticks(rotation = 45)
ax3.set_title(' Target vs Outlet_Identifier')

plt.show();


# **Observations**
# 
# * Distribution of Item_Outlet_Sales across the categories of Item_Type is not very distinct. 
# * The distribution for OUT010 and OUT019 categories of Outlet_Identifier are quite similar and very much different from the rest of the categories of Outlet_Identifier.

# In[ ]:


plt.figure(figsize = (10, 12))

plt.subplot(3, 1, 1)
sns.boxplot( x = 'Outlet_Size', y = 'Item_Outlet_Sales', data = df)

plt.subplot(3, 1, 2)
sns.boxplot( x = 'Outlet_Location_Type', y = 'Item_Outlet_Sales', data = df)

plt.subplot(3, 1, 3)
sns.boxplot( x = 'Outlet_Type', y = 'Item_Outlet_Sales', data = df)

plt.show();


# **Observations**
# 
# * Tier 1 and Tier 3 locations of Outlet_Location_Type look similar.
# * In the Outlet_Type plot, Grocery Store has most of its data points around the lower sales values as compared to the other categories.

# 'Outlet_Size has 4016 missing values. As it is categorical feature so generally the missing values can be filled with the mode, however it would be better to fill them with the most appropirate category by visualizing the similarity of distribution.  

# In[ ]:


df['Outlet_Size'] = df['Outlet_Size'].fillna('NAN')


# In[ ]:


df['Outlet_Size'].value_counts()


# In[ ]:


sns.violinplot( x = 'Outlet_Size', y = 'Item_Outlet_Sales', data = df)


# The distribution of NAN is similar to the 'small' category. so replacing it with the small. 

# In[ ]:


df['Outlet_Size'].replace('NAN', 'Small', inplace = True)


# 'Item_Visibility variable contains so many '0' values which can't ideally be true. so replace them with the mean of the non-zero values.

# In[ ]:


# mean of the Item_Visibility non-zero values. 
Item_vis_mean = df[df['Item_Visibility'] != 0]['Item_Visibility'].mean()
Item_vis_mean


# In[ ]:


df['Item_Visibility'].replace(0.0, Item_vis_mean, inplace = True)


# In[ ]:


sns.scatterplot(x = 'Item_Visibility', y = 'Item_Outlet_Sales', data = df)


# ### Feature Engineering
# Creating New features
# 1. Item_new_type: Broader categories for the variable Item_Type.
# 2. Item_category: Categorical variable derived from Item_Identifier.
# 3. Outlet_Years: Years of operation for outlets.
# 4. price_per_unit_wt: Item_MRP/Item_Weight
# 5. Item_MRP_clusters: Binned feature for Item_MRP.

# 1. 'Item_new_Type' new column has been created in place of Item_type

# In[ ]:


df['Item_Type'].value_counts()


# In[ ]:


df['Item_new_type'] = df['Item_Type']


# All the items in the Item_Type are grouped into three categories: perishable, non perishable and no food

# In[ ]:


non_perishable = ["Baking Goods", "Canned", "Frozen Foods", "Hard Drinks", "Health and Hygiene", "Household", "Soft Drinks"]
perishable = ["Breads", "Breakfast", "Dairy", "Fruits and Vegetables", "Meat", "Seafood"]

df['Item_new_type'] = df['Item_new_type'].apply(lambda x: 'perishable' if x in perishable else ('non perishable' if x in non_perishable else 'no food'))


# In[ ]:


df['Item_new_type'].value_counts()


# In[ ]:


sns.countplot(df['Item_new_type'])


# In[ ]:


sns.boxplot(x = 'Item_new_type', y = 'Item_Outlet_Sales', data = df)


# 2. 'Item_Identifier' column does not have any correlation with the target variable. So, either it can be re-classified into 3 categories ('FD', 'DR', 'NC') based on the first two alphabets, or drop the column. 

# In[ ]:


df['Item_Identity'] =  [x[:2] for x in df['Item_Identifier']]


# In[ ]:


df['Item_Identity'].value_counts()


# 3. 'Age' is calculated from the 'outlet_establishment year'.

# In[ ]:


df['Outlet_Establishment_Year'].value_counts()


# In[ ]:


df['Age'] = [2013-x for x in df['Outlet_Establishment_Year']]


# In[ ]:


df['Age'].hist()


# In[ ]:


sns.scatterplot(x= 'Age', y= 'Item_Outlet_Sales', data = df)


# 4. price per unit weight = Item_MRP/Item_Weight

# In[ ]:


df['price_per_unit_wt'] = df['Item_MRP']/df['Item_Weight']


# In[ ]:


sns.lmplot(x= 'price_per_unit_wt', y= 'Item_Outlet_Sales', data = df)


# 5. Item_MRP_clusters: Binned feature for Item_MRP.   
# As there are 4 categories so divide into 4 bins

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize = (10, 8))
sns.scatterplot(x= 'Item_MRP', y= 'Item_Outlet_Sales', hue = 'Item_Fat_Content',
                size = 'Item_Weight', data = df)
plt.plot([69,69], [0,5000])
plt.plot([137,137], [0,6500])
plt.plot([203,203], [0,9500])


# In[ ]:


df['Item_MRP_bins'] = pd.cut(df['Item_MRP'], bins = [25,69,137,203,270], 
                            labels = ['a', 'b', 'c', 'd'], right = True)


# Now lets drop the columns that has been replaced with their modification to avoid multicollinearity
# 
#     'Item_Identifier',  'Item_MRP', 'Item_Type','Outlet_Establishment_Year' , 'Item_Weight'
# 

# In[ ]:


df_final = df[['Item_Fat_Content',  'Item_Outlet_Sales',
        'Item_Visibility', 'Outlet_Identifier',
       'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type', 'Item_new_type',
       'Item_Identity', 'Age', 'price_per_unit_wt', 'Item_MRP_bins']]


# In[ ]:


df_final.info()


# In[ ]:


obj_col_final = df_final.select_dtypes('object').columns
obj_col_final


# In[ ]:


for col in obj_col_final:
    df_final[col] = df_final[col].astype('category')


# In[ ]:


df_final.info()


# In[ ]:


cat_col = df_final.select_dtypes('category').columns
cat_col


# In[ ]:


df_final = pd.get_dummies(data = df_final, columns = cat_col, drop_first=True)


# In[ ]:


df_final.info()


# **Now split the data into train and test after cleaning and preprocessing. 
# train, test
# ((8523, 12), (5681, 11))**

# In[ ]:


df_final.shape


# In[ ]:


train_final = df_final.iloc[:8523, :]
test_final = df_final.iloc[8523:, :]


# In[ ]:


train_final.shape, test_final.shape


# In[ ]:


test_final.drop('Item_Outlet_Sales', axis = 1, inplace = True)


# In[ ]:


test_final.shape


# # ML algorithms
# lets now find the best model by applying on the training data only and checking the accuracy score. 

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = train_final.copy()


# In[ ]:


X.drop('Item_Outlet_Sales', axis = 1, inplace = True)


# In[ ]:


y = train_final['Item_Outlet_Sales']


# In[ ]:


X.shape , y.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)


# # 1. Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_lr = lr.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error

lr_score = np.sqrt(mean_squared_error(y_test, y_lr))
lr_score


# # CrossVal for Linear Regression

# In[ ]:


from sklearn.model_selection import cross_val_score

score = cross_val_score(lr, X_train, y_train, cv = 10, scoring = 'neg_mean_squared_error')


# In[ ]:


lr_score_cross = np.sqrt(-score)


# In[ ]:


np.mean(lr_score_cross), np.std(lr_score_cross)


# # 2. Ridge Regression

# In[ ]:


from sklearn.linear_model import Ridge
r = Ridge(alpha= 0.05, solver = 'cholesky')
r.fit(X_train, y_train)
y_r = r.predict(X_test)
r_score = np.sqrt(mean_squared_error(y_test, y_r))
print(r_score)


# # Cross Val Ridge

# In[ ]:


score = cross_val_score(r, X_train, y_train, cv = 10, scoring = 'neg_mean_squared_error')
r_score_cross = np.sqrt(-score)
np.mean(r_score_cross), np.std(r_score_cross)


# # 3. Lasso

# In[ ]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.01)
lasso.fit(X_train, y_train)
y_lasso = lasso.predict(X_test)
lasso_score = np.sqrt(mean_squared_error(y_test, y_lasso))
print(lasso_score)


# # Cross Val Lasso

# In[ ]:


score = cross_val_score(lasso, X_train, y_train, cv = 10, scoring = 'neg_mean_squared_error')
lasso_score_cross = np.sqrt(-score)
np.mean(lasso_score_cross), np.std(lasso_score_cross)


# # 4. Elastic Net

# In[ ]:


from sklearn.linear_model import ElasticNet

en = ElasticNet(alpha = 0.01, l1_ratio= 0.5)
en.fit(X_train, y_train)
y_en = en.predict(X_test)
en_score = np.sqrt(mean_squared_error(y_test, y_en))
print(en_score)


# # Cross Val Elastic Net

# In[ ]:


score = cross_val_score(en, X_train, y_train, cv = 10, scoring='neg_mean_squared_error')
en_score_cross = np.sqrt(-score)
np.mean(en_score_cross), np.std(en_score_cross)


# # 5. Stochastic gradient

# In[ ]:


from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(penalty='l2', max_iter= 100, alpha = 0.05)
sgd.fit(X_train, y_train)
y_sgd = sgd.predict(X_test)
sgd_score = np.sqrt(mean_squared_error(y_test, y_sgd))
print(sgd_score)


# # Cross Val Stochastic gradient

# In[ ]:


score = cross_val_score(sgd, X_train, y_train, cv = 10, scoring='neg_mean_squared_error')
sgd_score_cross = np.sqrt(-score)
np.mean(sgd_score_cross), np.std(sgd_score_cross)


# # 6. Support Vector Regression

# In[ ]:


from sklearn.svm import SVR
svr = SVR(epsilon=15, kernel='linear')
svr.fit(X_train, y_train)
y_svr = svr.predict(X_test)
svr_score = np.sqrt(mean_squared_error(y_test, y_svr))
print(svr_score)


# # Cross Val SVR

# In[ ]:


score = cross_val_score(svr, X_train, y_train, cv = 10, scoring='neg_mean_squared_error')
svr_score_cross = np.sqrt(-score)
np.mean(svr_score_cross), np.std(svr_score_cross)


# # 7. Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
y_dtr = dtr.predict(X_test)
dtr_score = np.sqrt(mean_squared_error(y_test, y_dtr))
print(dtr_score)


# # Cross Val Decision Tree

# In[ ]:


score = cross_val_score(dtr, X_train, y_train, cv= 10, scoring = 'neg_mean_squared_error')
dtr_score_cross = np.sqrt(-score)
np.mean(dtr_score_cross), np.std(dtr_score_cross)


# # 8. Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)
rf_score = np.sqrt(mean_squared_error(y_test, y_rf))
print(rf_score)


# # Cross Val Random Forest

# In[ ]:


score = cross_val_score(rf, X_train, y_train, cv = 10, scoring= 'neg_mean_squared_error')
rf_score_cross = np.sqrt(-score)
np.mean(rf_score_cross), np.std(rf_score_cross)


# # 9. Bagging Regression

# In[ ]:


from sklearn.ensemble import BaggingRegressor

br = BaggingRegressor(max_samples = 70)
br.fit(X_train, y_train)
y_br = br.predict(X_test)
br_score = np.sqrt(mean_squared_error(y_test, y_br))
print(br_score)


# # Cross Val Bagging Regression

# In[ ]:


score = cross_val_score(br, X_train, y_train, cv = 10, scoring='neg_mean_squared_error')
br_score_cross = np.sqrt(-score)
np.mean(br_score_cross), np.std(br_score_cross)


# # 10. Ada Boosting

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor()
ada.fit(X_train, y_train)
y_ada = ada.predict(X_test)
ada_score = np.sqrt(mean_squared_error(y_test, y_ada))
print(ada_score)


# # Cross Val Ada Boosting

# In[ ]:


score = cross_val_score(ada, X_train, y_train, cv = 10 , scoring = 'neg_mean_squared_error')
ada_score_cross = np.sqrt(-score)
np.mean(ada_score_cross), np.std(ada_score_cross)


# # 11. Gradient Boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
y_gbr = gbr.predict(X_test)
gbr_score = np.sqrt(mean_squared_error(y_test, y_gbr))
print(gbr_score)


# ## Cross Val Gradient Boosting

# In[ ]:


score = cross_val_score(gbr, X_train, y_train, cv =10, scoring='neg_mean_squared_error')
gbr_score_cross = np.sqrt(-score)
np.mean(gbr_score_cross) , np.std(gbr_score_cross)


# # 12. XG Boosting

# In[ ]:


from xgboost import XGBRegressor

xgb = XGBRegressor()
xgb.fit(X_train, y_train)
y_xgb = xgb.predict(X_test)
xgb_score = np.sqrt(mean_squared_error(y_test, y_xgb))
print(xgb_score)


# ## Cross Val XGB
# 

# In[ ]:


score = cross_val_score(xgb, X_train, y_train, cv = 10, scoring='neg_mean_squared_error')
xgb_score_cross = np.sqrt(-score)
np.mean(xgb_score_cross), np.std(xgb_score_cross)


# # Comparison

# In[ ]:


name = ['Linear Regression','Linear Regression CV','Ridge Regression','Ridge Regression CV','Lasso Regression',
     'Lasso Regression CV','Elastic Net Regression','Elastic Net Regression CV','SGD Regression','SGD Regression CV',
     'SVM','SVM CV','Decision Tree','Decision Tree Regression','Random Forest','Random Forest CV','Ada Boost','Ada Boost CV',
     'Bagging','Bagging CV','Gradient Boost','Gradient Boost CV', 'XGboost', 'XGBoost CV']


# In[ ]:


model = pd.DataFrame({'RMSE': [lr_score, lr_score_cross, r_score, r_score_cross, 
                              lasso_score, lasso_score_cross, en_score, en_score_cross, 
                              sgd_score, sgd_score_cross, svr_score, svr_score_cross, 
                              dtr_score, dtr_score_cross, rf_score, rf_score_cross, 
                               ada_score, ada_score_cross, br_score, br_score_cross, 
                              gbr_score, gbr_score_cross, xgb_score, xgb_score_cross]}, index = name)


# In[ ]:


model['RMSE'] = [np.mean(x) for x in model['RMSE']]


# In[ ]:


model['RMSE'].sort_values()


# ## Gradient boosting is doing great job

# lets do grid search to tupe hyper parameter

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


gb = GradientBoostingRegressor(max_depth=7, n_estimators=200, learning_rate=0.01)
param = [{'min_samples_split' : [5,9,13], 
         'max_leaf_nodes' : [3,5,7,9],
         'max_features':[8,10,15,18]}]
gs = GridSearchCV(gb, param, cv = 5, scoring= 'neg_mean_squared_error')
gs.fit(X_train, y_train)


# In[ ]:


gs.best_estimator_


# In[ ]:


gb = gs.best_estimator_


# # Now train our model on Training Data

# In[ ]:


train_final.shape


# In[ ]:


X_train_final = train_final.drop('Item_Outlet_Sales', axis = 1)


# In[ ]:


y_train_final = train_final['Item_Outlet_Sales']


# In[ ]:


X_train_final.shape, y_train_final.shape


# In[ ]:


# fitting model 
gb.fit(X_train_final, y_train_final)


# In[ ]:


test_final.shape


# In[ ]:


test_predict = gb.predict(test_final)


# In[ ]:


test_predict.shape


# In[ ]:


sample_result = pd.read_csv('/kaggle/input/big-mart-sales-prediction/sample_submission_8RXa3c6.csv')
sample_result.head()


# In[ ]:


del sample_result['Item_Outlet_Sales']


# In[ ]:


sample_result['Item_Outlet_Sales'] = test_predict


# In[ ]:


sample_result


# In[ ]:


sample_result.to_csv('submission.csv', index = False)


# In[ ]:




