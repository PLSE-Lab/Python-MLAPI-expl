#!/usr/bin/env python
# coding: utf-8

# In[114]:


import pandas as pd
import numpy as np


# In[115]:


data = pd.read_csv('../input/Melbourne_housing_FULL.csv')
pd.set_option('display.max_columns', None) # display all columns
data.head(5)


# In[116]:


data.describe(include='all').T


# In[117]:


data.shape


# I want first by using intuition to drop all the columns that are not relevant to the **price** including:
# + Lattitude
# + Longtitude

# In[118]:


data.drop(columns=['Lattitude','Longtitude'], inplace=True)


# # Data cleaning #

# ### - Distance ###

# In[119]:


data.Distance.value_counts(dropna=False)


# There is only 1 missing value. The mode of this feature has significant counts. Hence, it is plausible to replace this missing value with the mode of this feature.

# In[120]:


data.Distance.fillna(data.Distance.mode(), inplace=True)


# ### - Postcode ###

# In[121]:


data[data.Postcode.isnull()]


# It occurs that this row has a lots of missing data. The solution is just drop this because it seems useless.

# In[122]:


data.drop(index=29483, inplace=True)


# In[123]:


data.reset_index().drop(columns='index', inplace=True)


# In[124]:


data.iloc[29483]


# ### - Regionname ###

# In[125]:


data.Regionname.value_counts(dropna=False)


# In[126]:


data[data.Regionname.isnull()]


# It also shows that these two rows misses alot of features and we want to drop them as above.

# In[127]:


data.drop(index=[18523, 26888], inplace=True)
data.reset_index().drop(columns='index', inplace=True)


# We might want to take a look at the data again to see which columns need cleaning

# In[128]:


data.describe(include='all').T


# ### - YearBuilt ###

# In[129]:


data.YearBuilt[data.YearBuilt >= 2020].value_counts(dropna=False)


# This is a typo and the possible data is 2016 instead of 2106.

# In[130]:


data.YearBuilt.replace(2106, 2016, inplace=True)


# In[131]:


data.YearBuilt.value_counts(dropna=False)


# In[132]:


data.YearBuilt.corr(data.Price)


# This feature has too maning NaNs and very low correlation with Price. I will drop this column.

# In[133]:


data.drop(columns='YearBuilt', inplace=True)


# ### - BuildingArea ###

# In[134]:


data.BuildingArea.value_counts(dropna=False)


# This feature also misses a lot of data. But this is a very important feature as a guess, I cannot drop this column. Instead, I will drop all of the missing rows.

# In[135]:


data = data[pd.notnull(data['BuildingArea'])]


# In[136]:


data.loc[data.BuildingArea.idxmax()]


# I notice this is an outlier with very high price and very large Area, totally distighes itself from all of the others. I would drop this for the sake of regression later.

# In[137]:


data.drop(index=data.BuildingArea.idxmax(), inplace=True)


# In[138]:


data.describe(include='all').T


# We dropped alot of data but an amount of the other NaNs has been removed as well.

# ### - Landsize ###

# In[139]:


data.Landsize.value_counts(dropna=False)


# According to the data, we might guess the NaNs data can be replaced with the mode = 0, implying the house only have building area which also makes sense.

# In[140]:


data.Landsize.fillna(value=0, inplace=True) # using mode() does not work -> resort to hardcode value 0


# ### - Car ###

# In[141]:


data.Car.value_counts(dropna=False)


# In this case, it makes sense to replace the NaNs with 0.

# In[142]:


data.Car.fillna(value=0, inplace=True)


# ### - Price ###

# This feature is important as it is the output of the upcoming regression model. To train the model, we need all of its data, and the NaNs can be replaced with anything. But at the cost of droping all of the NaNs rows in price, we might miss out on data. Hence, I suggest creating another dataframe for regression while using the old dataframe for data exploration.

# In[143]:


import copy
data_regr = copy.copy(data[pd.notnull(data['Price'])])


# In[144]:


data_regr.describe(include='all').T


# I want to check out other columns to see if there is any problems in the data

# In[145]:


data.Rooms.value_counts()


# In[146]:


data.Type.value_counts()


# In[147]:


data.Method.value_counts()


# In[148]:


data.CouncilArea.value_counts()


#  # Data Exploration #

# In[149]:


data.describe().T


# In[150]:


import seaborn as sns
import matplotlib.pyplot as plt
from jupyterthemes import jtplot

jtplot.style(theme='grade3')


# In[151]:


corr = data[['Rooms', 'Price', 'Distance', 
             'Postcode', 'Bedroom2', 'Bathroom', 
             'Car', 'Landsize', 'BuildingArea', 'Propertycount']].corr()


# In[152]:


sns.heatmap(corr)


# Looking at the correlation heatmap, we see that:
# + **Bedroom2** and **Rooms** have very strong correlation (near to 1)
# + **Bathroom** and **Rooms** also have decent correlation

# In[153]:


data[['Rooms','Bedroom2','Bathroom','Price']].corr()


# We want a closer look in the correlation matrix of these specific features. Because **Rooms** and **Bedroom2** has near 1 correlation value, we should remove one of them to reduce the dimension. According to the matrix, we choose **Rooms** to keep because it has higher correlation value with **Price**

# Among the features, the number of rooms have very high correlation with **Price** but not the **Landsize**.

# In[154]:


data.drop(columns='Bedroom2', inplace=True)


# In[155]:


data_regr.drop(columns='Bedroom2', inplace=True)


# In[156]:


data.describe(include='all').T


# In[157]:


#TODO: Date, methode, type


# # Regression #

# In[158]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error


# There are many categorical features in the data, I have to first deal with them one by one before I can perform regression. Let's take a look at the categorical variables:

# In[159]:


data_regr.describe(include=['O']).T


# In Australia, Suburbs are contained within Regions and each Suburb as their own Postcode. Knowing this information, we can:
# + drop the **Postcode** feature
# + decide between **Suburb** and **Regionname** feature to use for the regression model

# **Suburb** has 324 unique values and dealing with them takes lot of hard work while **Regionname** has only 8 unique value and *One-hot encoding* would suffice. I decide to also drop the Suburb column.

# In[160]:


data_regr.drop(columns=['Suburb','Postcode'], inplace=True)


# For **Date**, it is safe to say that price won't change much within a year. We will analyse this later if time allows. I will use only *year* in this feature.

# In[161]:


import datetime
def to_year(date_str):
    return datetime.datetime.strptime(date_str.strip(),'%d/%m/%Y').year


# In[162]:


data_regr['Date'] = data_regr.Date.apply(to_year)


# In[163]:


data_regr.Date.value_counts()


# For **Address**, this feature has too many unique value and it is not good if we use *one-hot encode* technique. We might have more interest in the *Street* and hope that the unique value wil decrease such that we can use *one-hot encode*.

# In[164]:


import re
def to_street(str):
    return re.sub('[^A-Za-z]+', '', str)


# In[165]:


data_regr.Address.apply(to_street).value_counts().count()


# It still has up to 6000 unique value. For regression, this column is too diversed. Hence, I drop it.

# In[166]:


data_regr.drop(columns='Address', inplace=True)


# In[167]:


counts = data_regr.SellerG.value_counts()
counts


# This is also a diversed feature with up to 268 unique values. But we can see that it has many value occurs only once. Let s group all of these values into 'Other' to see if we can deal with this feature.

# In[168]:


data_regr.SellerG[data['SellerG'].isin(counts[counts < 100].index)] = 'less than 100'
data_regr.SellerG[data['SellerG'].isin(counts[(counts >= 100) & (counts < 200)].index)] = '100 - 200'
data_regr.SellerG[data['SellerG'].isin(counts[(counts >= 200) & (counts < 500)].index)] = '200 - 500'
data_regr.SellerG[data['SellerG'].isin(counts[(counts >= 500) & (counts < 1000)].index)] = '500 - 1000'
data_regr.SellerG[data['SellerG'].isin(counts[counts > 1000].index)] = 'over 1000'


# In[169]:


data_regr.SellerG.value_counts()


# As we already have the Geo information of the house in the **Regionname** , the **CouncilArea** with 33 unique values can be dropped for convenience.

# In[170]:


data_regr.drop(columns='CouncilArea', inplace=True)


# Now we have finished dealing with our Categorical variables, let's take a look again at them:

# In[171]:


data_regr.describe(include=['O']).T


# In[172]:


data.head()


# In[173]:


data = data.reset_index().drop(columns='index') # do not use inplace=True if combine
data_regr = data_regr.reset_index().drop(columns='index')


# In[174]:


data_regr.head()


# ## One-hot Encode

# In[175]:


categoricals = ['Type', 'Method', 'SellerG', 'Regionname', 'Date']
for feature in categoricals:
    df = copy.copy(pd.get_dummies(data_regr[feature], drop_first=True))
    data_regr = pd.concat([data_regr, df], axis=1)
    data_regr.drop(columns=feature, inplace=True)


# In[176]:


data_regr.head()


# Noted that in this case we only take k-1 dummies to reduce the number of dimensions.

# In[177]:


data_regr.shape


# ## Linear Regression without PCA ##

# ### Hold out ###

# In[178]:


model_HO = linear_model.LinearRegression()


# In[179]:


train, test = train_test_split(data_regr, test_size = 0.2, random_state=512)


# In[180]:


train.shape


# In[181]:


test.shape


# In[182]:


X_train = train.loc[:, data_regr.columns != 'Price']
y_train = train.Price

X_test = test.loc[:, data_regr.columns != 'Price']
y_test = test.Price


# In[183]:


model_HO.fit(X_train.values, y_train.values)


# In[184]:


predict_train = model_HO.predict(X_train.values)
mean_squared_error(y_train, predict_train)


# In[185]:


predict_test = model_HO.predict(X_test.values)
mean_squared_error(y_test, predict_test)


# In[186]:


fig, ax = plt.subplots()
ax.scatter(y_test, predict_test)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# ### Cross validation ###

# In[187]:


model_CV = linear_model.LinearRegression()


# In[188]:


y = data_regr.Price
X = data_regr.loc[:, data_regr.columns != 'Price']
predicted = cross_val_predict(model_CV, X.values, y.values, cv=5)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# In[189]:


mean_squared_error(y.values, predicted)


# According to the graph and the high value, this model does not perform well. It is possible that the model is too simple for this problem.

# ## Linear Regression with PCA ##

# In[190]:


from sklearn.decomposition import PCA


# In[191]:


pca = PCA(n_components=10)


# In[192]:


X_new = pd.DataFrame(pca.fit_transform(X.values))


# ### Hold out ###

# In[193]:


model_HO_PCA = linear_model.LinearRegression()


# In[194]:


dataPCA = pd.concat([X_new, y], axis=1)


# In[195]:


train, test = train_test_split(dataPCA, test_size = 0.2, random_state=512)


# In[196]:


X_train = train.loc[:, train.columns != 'Price']
y_train = train.Price

X_test = test.loc[:, test.columns != 'Price']
y_test = test.Price


# In[197]:


model_HO_PCA.fit(X_train.values, y_train.values)


# In[198]:


predict_train = model_HO_PCA.predict(X_train.values)
mean_squared_error(y_train, predict_train)


# In[199]:


predict_test = model_HO_PCA.predict(X_test.values)
mean_squared_error(y_test, predict_test)


# In[200]:


fig, ax = plt.subplots()
ax.scatter(y_test, predict_test)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# ### Cross validation ###

# In[201]:


model_CV_PCA = linear_model.LinearRegression()


# In[202]:


y = dataPCA.Price
X = dataPCA.loc[:, dataPCA.columns != 'Price']
predicted = cross_val_predict(model_CV_PCA, X.values, y.values, cv=5)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# In[203]:


mean_squared_error(y.values, predicted)


# As expected, the original model with all of the data does not perform as very well, the model with PCA couldnt perform better.

# In[204]:


X = data_regr.loc[:, data_regr.columns != 'Price']
y = data_regr.Price


# In[205]:


model = linear_model.LinearRegression()


# In[206]:


model.fit(X.values, y.values)


# In[207]:


predict = model.predict(X.values)


# In[208]:


sns.residplot(predict, y.values)


# the Residual Plot shows that this model is not a good fit for the problem.

# In[209]:


a = (y.values - predict)


# In[210]:


fig, ax = plt.subplots()
ax.scatter(data_regr.Rooms.values, a)
ax.set_xlabel('Rooms')
ax.set_ylabel('Residual')
plt.show()


# In[211]:


fig, ax = plt.subplots()
ax.scatter(data_regr.Distance.values, a)
ax.set_xlabel('Distance')
ax.set_ylabel('Residual')
plt.show()


# In[212]:


fig, ax = plt.subplots()
ax.scatter(data_regr.Bathroom.values, a)
ax.set_xlabel('Bathroom')
ax.set_ylabel('Residual')
plt.show()


# In[213]:


fig, ax = plt.subplots()
ax.scatter(data_regr.Car.values, a)
ax.set_xlabel('Car')
ax.set_ylabel('Residual')
plt.show()


# In[214]:


fig, ax = plt.subplots()
ax.scatter(data_regr.Landsize.values, a)
ax.set_xlabel('Landsize')
ax.set_ylabel('Residual')
plt.show()


# In[215]:


fig, ax = plt.subplots()
ax.scatter(data_regr.BuildingArea.values, a)
ax.set_xlabel('BuildingArea')
ax.set_ylabel('Residual')
plt.show()


# In[216]:


fig, ax = plt.subplots()
ax.scatter(data_regr.Propertycount.values, a)
ax.set_xlabel('Propertycount')
ax.set_ylabel('Residual')
plt.show()


# It shows that the variables have great heteroskedasticity value, especially the **BuildingArea** and the **Landsize**. They have many outliers. I want to eliminate all of these outliers and fit the model again to see the result.

# In[223]:


data_regr = data_regr[data_regr.BuildingArea < 3000]


# In[224]:


data_regr = data_regr.reset_index()


# In[225]:


data_regr.drop(columns='index', inplace=True)


# In[228]:


data_regr.describe().T


# In[227]:


data_regr = data_regr[data_regr.Landsize < 3000]
data_regr = data_regr.reset_index()
data_regr.drop(columns='index', inplace=True)


# In[229]:


X = data_regr.loc[:, data_regr.columns != 'Price']
y = data_regr.Price
model = linear_model.LinearRegression()
model.fit(X.values, y.values)
predict = model.predict(X.values)
mean_squared_error(y.values, predict)


# In[230]:


fig, ax = plt.subplots()
ax.scatter(y, predict)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# The model still does not perform really good. We might need a non-linear model for this problem.

# In[ ]:




