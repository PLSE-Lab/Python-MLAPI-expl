#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import math
import statsmodels.formula.api as smf
from scipy import stats


# In[6]:


data = pd.read_csv('../input/imports_85_data.csv')
data.head()


# In[ ]:


data.shape
data.columns


# In[162]:


data.columns = ['symboling',' normalized_losses','make','fuel_type','aspiration','num_of_doors','body_style','drive_wheels','engine_location','wheel_base','length','width','height','curb_weight','engine_type','num_of_cylinders','engine_size','fuel_system','bore','stroke','compression_ratio','horsepower','peak_rpm','city_mpg','highway_mpg','price']
data.columns
data.head(10)


# In[163]:


data=pd.get_dummies(data, columns=['make','fuel_type','aspiration','drive_wheels','engine_location',
 'engine_type','fuel_system'])
data.head()


# In[164]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2)

train.shape


# In[165]:


test.shape


# In[166]:


train.count()


# In[167]:


def getMissingFeatures(df):
    ser_counts=df.count()
    data_size=len(df)
    data_missing_features=[]
    for idx in ser_counts.index:
        if(ser_counts[idx]<data_size):
            data_missing_features.append(idx)
    return data_missing_features  


# In[168]:


getMissingFeatures(train)


# In[169]:


train[' normalized_losses'].hist(bins=100)


# In[170]:


train['bore'].hist(bins=100)


# In[171]:


train['stroke'].hist(bins=100)


# In[172]:


train['horsepower'].hist(bins=100)


# In[173]:


train['peak_rpm'].hist(bins=100)


# In[174]:


train['price'].hist(bins=100)


# In[175]:


train.fillna(value=train.median()[[' normalized_losses']], inplace=True)
train.fillna(value=train.median()[['bore']], inplace=True)
train.fillna(value=train.median()[['stroke']], inplace=True)
train.fillna(value=train.median()[['horsepower']], inplace=True)
train.fillna(value=train.median()[['peak_rpm']], inplace=True)
train.fillna(value=train.median()[['price']], inplace=True)


# In[176]:


train[train.price.isnull()]


# In[177]:


plt.figure(figsize=(16,16))
sns.set_palette("PuBuGn_d")
g= sns.heatmap(train.corr(),linewidths=.5,cmap="YlGnBu")


# In[178]:


#corr = numeric_features.corr()
#print(corr['price'].sort_values(ascending=False)[:5],'\n')
#print(corr['price'].sort_values(ascending=False)[-5:])


# In[179]:


train.engine_size.unique()


# In[180]:


engine_pivot = train.pivot_table(index='engine_size', values='price', aggfunc=np.median)


# In[181]:


engine_pivot


# In[182]:


engine_pivot.plot(kind='bar', color='blue')
plt.xlabel('symboling')
plt.ylabel('median price')
plt.xticks(rotation=0)
plt.show()


# In[183]:


target =np.log(train.price)
plt.scatter(x=train['curb_weight'], y=target)
plt.ylabel('price')
plt.xlabel('curb weight')
plt.show()


# In[184]:


plt.scatter(x=train['highway_mpg'], y=target)
plt.ylabel('price')
plt.xlabel('highway mpg')
plt.show()


# In[185]:


categoricals = train.select_dtypes(exclude=[np.number])
categoricals.describe()


# In[186]:


def getCatFeaturesWithNulls(df):
    ser_dtypes = df.dtypes
    str_col_with_nulls = [x for x in ser_dtypes.index
                              if ser_dtypes[x] in ['object']
                                and len(df[df[x].notnull()])< len(df)]
    return str_col_with_nulls


# In[187]:


getCatFeaturesWithNulls(train)


# In[188]:


lst_cat_features = list(categoricals.columns.values)
lst_cat_features


# In[189]:


for feature in lst_cat_features:
    print(train[feature].value_counts())


# In[190]:


train["num_of_doors"] = train["num_of_doors"].astype('category')
train["num_of_doors_cat"] =train["num_of_doors"].cat.codes
print("value counts")
print(train["num_of_doors_cat"].value_counts())


# In[191]:


train["body_style"] = train["body_style"].astype('category')
train["body_style_cat"] =train["body_style"].cat.codes
print("value counts")
print(train["body_style_cat"].value_counts())


# In[192]:


train["num_of_cylinders"] = train["num_of_cylinders"].astype('category')
train["num_of_cylinders_cat"] =train["num_of_cylinders"].cat.codes
print("value counts")
print(train["num_of_cylinders_cat"].value_counts())


# In[193]:


train.head()


# In[194]:


del train['num_of_doors']
del train['num_of_cylinders']
del train['body_style']


# In[195]:


train.head()


# In[196]:


y_train = train['price'].values
print('y_train - type:' , type(y_train))
print('y_train - shape:',y_train.shape)
print('y_train - array:', y_train)


# In[197]:


x_features = [ feature for feature in train.columns if feature not in ['price','Id']]
print(x_features)


# In[198]:


x_train = train[x_features].values
print('x_train - type:', type(x_train))
print('x_train - shape:',x_train.shape)
print('x_train - array:', x_train)


# In[199]:


from sklearn.linear_model import LinearRegression
lm_model = LinearRegression()
lm_model.fit(x_train, target)
print('Linear model coefficients:', lm_model.coef_)
print('Linear model intercept:', lm_model.intercept_)


# In[200]:


from sklearn.metrics import mean_squared_error, r2_score
pred_x_train = lm_model.predict(x_train)
mse_x_train = mean_squared_error(target, pred_x_train)
r2_x_train = r2_score(target, pred_x_train)
print('Root mean squared error x train:',np.sqrt(mse_x_train))
print('R-square x train :',r2_x_train)


# In[201]:


getMissingFeatures(test)


# In[202]:


test.fillna(value=test.median()[[' normalized_losses']], inplace=True)
test.fillna(value=test.median()[['bore']], inplace=True)
test.fillna(value=test.median()[['stroke']], inplace=True)
test.fillna(value=test.median()[['price']], inplace=True)            
            
test.head()


# In[203]:


test["num_of_doors"] = test["num_of_doors"].astype('category')
test["num_of_doors_cat"] =test["num_of_doors"].cat.codes
print("value counts")
print(test["num_of_doors_cat"].value_counts())


# In[204]:


test["body_style"] = test["body_style"].astype('category')
test["body_style_cat"] =test["body_style"].cat.codes
print("value counts")
print(test["body_style_cat"].value_counts())


# In[205]:


test["num_of_cylinders"] = test["num_of_cylinders"].astype('category')
test["num_of_cylinders_cat"] =test["num_of_cylinders"].cat.codes
print("value counts")
print(test["num_of_cylinders_cat"].value_counts())


# In[206]:


del test['num_of_doors']
del test['num_of_cylinders']
del test['body_style']
test.head()


# In[207]:


y_test = test['price'].values
print('y_test - type:' , type(y_test))
print('y_test - shape:',y_test.shape)
print('y_test - array:', y_test)


# In[208]:


x_features1 = [ feature for feature in test.columns if feature not in ['price','Id']]
print(x_features1)


# In[209]:


x_test = test[x_features1].values
print('x_test - type:', type(x_test))
print('x_test - shape:',x_test.shape)
print('x_test - array:', x_test)
target1 =np.log(test.price)


# In[1]:


pred_x_test = lm_model.predict(x_test)
mse_x_test = mean_squared_error(target1, pred_x_test)
r2_x_test = r2_score(target1, pred_x_test)
print('Root mean squared error x test:', np.sqrt(mse_x_test))
print('R-square x test :', r2_x_test)


# In[ ]:





# In[ ]:




