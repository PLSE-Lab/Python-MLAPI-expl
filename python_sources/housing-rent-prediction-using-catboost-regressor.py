#!/usr/bin/env python
# coding: utf-8

# ## Import Packages & Load Data

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns',None)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style='ticks',color_codes=True,font_scale=1.5)
color = sns.color_palette()
sns.set_style('darkgrid')

from scipy import stats
from scipy.stats import skew, norm, probplot, boxcox
from scipy.special import boxcox1p


# In[ ]:


train = pd.read_csv('../input/housing-rent-dataset/housing_train.csv')
train.head()


# In[ ]:


train_Id = train['id']
train.drop('id',axis=1,inplace = True)


# # Exploratory Data Analysis

# In[ ]:


train.shape


# In[ ]:


train.describe()


# In[ ]:


def analysis(df, target):
    instance = df.shape[0]
    types=df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.T.apply(pd.Series.unique,1)
    nulls= df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(pd.Series.nunique)
    null_perc = (df.isnull().sum()/instance)*100
    skewness = df.skew()
    kurtosis = df.kurt()
    
    corr = df.corr()[target]
    str = pd.concat([types, counts,uniques, nulls,distincts, null_perc, skewness, kurtosis, corr], axis = 1, sort=False)
    corr_col = 'corr '  + target
    cols = ['types', 'counts','uniques', 'nulls','distincts', 'null_perc', 'skewness', 'kurtosis', corr_col ]
    str.columns = cols
    return str


# In[ ]:


details = analysis(train,'price')
details


# #### Types of data
# Int : 10  
# float : 3   
# object : 9
# 
# #### Highly skewed and heavy tail distributed features:
# price  
# beds   
# sqfeet  
# bath   
# electric_vehicle_charge  
# possibly outliers  
# We may apply boxcox or log transformation on the above feature
# 
# #### Features with Null values:
# laundry_options  
# parking_options	   
# description  
# lat  
# long  
# state  
# 
# No much correlation between independent features and target
# 

# In[ ]:


# lets see our numerical features
num_features = [feat for feat in train.columns if train[feat].dtypes != 'O']
num_features


# In[ ]:


# lets see our categorical features
cat_features = [feat for feat in train.columns if train[feat].dtypes == 'O']
cat_features


# ### Analyzing Sqfeet feature

# In[ ]:


plt.figure(figsize=(15,8))

sns.distplot(train['sqfeet'],kde=True,norm_hist=True)
plt.title('Skewness: {} and Kurtosis: {}'.format(train['sqfeet'].skew(),train['sqfeet'].kurtosis()))


# The feature sqfeet is highly positively skewed and has a heavy tailed distribution. There is high probability of outliers presence.

# In[ ]:


sns.jointplot(x=train['sqfeet'],y=train['price'],height=10,ratio=3)
print('sqfeet Skewness: {:.3f} and Kurtosis: {:.3f}'.format(train['sqfeet'].skew(),train['sqfeet'].kurtosis()))
print('price Skewness: {:.3f} and Kurtosis: {:.3f}'.format(train['price'].skew(),train['price'].kurtosis()))
plt.title('Correlations: {}'.format(train['sqfeet'].corr(train['price'])))
train.shape


# In[ ]:


data=train.copy()
data= data[data['sqfeet']<data['sqfeet'].quantile(0.9996)]
data = data[data['price']<data['price'].quantile(0.9996)]

sns.jointplot(x=data['sqfeet'],y=data['price'],height=10,ratio=3)
print('sqfeet Skewness: {:.3f} and Kurtosis: {:.3f}'.format(data['sqfeet'].skew(),data['sqfeet'].kurtosis()))
print('price Skewness: {:.3f} and Kurtosis: {:.3f}'.format(data['price'].skew(),data['price'].kurtosis()))
plt.title('Correlations: {:.3f}'.format(data['sqfeet'].corr(data['price'])))
data.shape


# ### Analyzing beds feature

# In[ ]:


train['beds'].value_counts()


# In[ ]:


data = train.copy()
data.loc[data['beds']>8]='8+'
plt.figure(figsize=(12,8))
sns.countplot(data['beds'].astype('str').sort_values())
plt.title('Beds distribution',fontsize=12)
print('Skewness: {:.3f} \nKurtosis: {:.3f}'.format(train['beds'].skew(),train['beds'].kurtosis()))
plt.show()


# In[ ]:


data = train.copy()
data = data[data['price']<data['price'].quantile(0.9996)]
data = data[data['beds']<10]
plt.figure(figsize=(15,8))
sns.boxplot(x='beds',y='price',data=data)
plt.title('Correlations: {:.3f}'.format(data['beds'].corr(data['price'])))
print('price Skewness: {:.3f} and Kurtosis: {:.3f}'.format(data['beds'].skew(),data['beds'].kurtosis()))


# ### Analyzing Bath Feature

# In[ ]:


train['baths'].value_counts()


# In[ ]:


data = train.copy()
data.loc[data['baths']> 8.5]='9+'
plt.figure(figsize=(12,8))
sns.countplot(data['baths'].astype('str').sort_values())
plt.title('Bathss distribution',fontsize=12)
print('Skewness: {:.3f} \nKurtosis: {:.3f}'.format(train['baths'].skew(),train['baths'].kurtosis()))
plt.show()


# In[ ]:


data = train.copy()
data = data[data['price']<data['price'].quantile(0.9996)]
data = data[data['baths']<10]
plt.figure(figsize=(15,8))
sns.boxplot(x='baths',y='price',data=data)
plt.title('Correlations: {:.3f}'.format(data['baths'].corr(data['price'])))
print('price Skewness: {:.3f} and Kurtosis: {:.3f}'.format(data['baths'].skew(),data['baths'].kurtosis()))


# ### Analyzing boolean features

# In[ ]:


bool_features = [x for x in num_features if train[x].nunique() < 3]
bool_features


# In[ ]:


# we will use median for skewed and mean for normally distributed feature
skewed_bool_features = ['wheelchair_access',
 'electric_vehicle_charge',
 'comes_furnished']


# In[ ]:


for feature in skewed_bool_features:
    data=train.copy()
    data = data[data['price']<data['price'].quantile(0.9996)]
    data.groupby(feature)['price'].median().plot.bar()
    plt.xlabel(feature)
    plt.title('Skeweness of feature {} is {:.6f}'.format(feature,data[feature].skew()))
    plt.ylabel('price')
    print('Correlation: {:.3f}'.format(data[feature].corr(data['price'])))
    plt.show()


# In[ ]:


normal_bool_features=['cats_allowed',
 'dogs_allowed',
 'smoking_allowed']
for feature in normal_bool_features:
    data=train.copy()
    data = data[data['price']<data['price'].quantile(0.9996)]
    data.groupby(feature)['price'].mean().plot.bar()
    plt.xlabel(feature)
    plt.title('Skeweness of feature {} is {:.6f}'.format(feature,data[feature].skew()))
    plt.ylabel('price')
    print('Correlation: {:.3f}'.format(data[feature].corr(data['price'])))
    plt.show()


# ### Analyzing categorical featuress

# In[ ]:


# lets see our categorical features
cat_features = [feat for feat in train.columns if train[feat].dtypes == 'O']
cat_features = [feat for feat in cat_features if 'url' not in feat]
cat_features


# #### Analyzing Type Feature

# In[ ]:


train['type'].unique()


# In[ ]:


data = train.copy()
data = data[data['price']<data['price'].quantile(0.9996)]
fig=plt.figure(figsize=(18,14))
plt.subplots_adjust(hspace=0.5)
ax1 = fig.add_subplot(211);
sns.countplot(data['type'])
plt.title('Type distribution',fontsize=12)
plt.xticks(rotation=90)
ax2 =fig.add_subplot(212);
sns.boxplot(x='type',y='price',data=data)
plt.xticks(rotation=90)
plt.show()


# ####  Analyzing parking_options Feature

# In[ ]:


data = train.copy()
data = data[data['price']<data['price'].quantile(0.9996)]
fig=plt.figure(figsize=(18,14))
plt.subplots_adjust(hspace=0.5)
ax1 = fig.add_subplot(211);
sns.countplot(data['parking_options'])
plt.title('parking_options distribution',fontsize=12)
plt.xticks(rotation=90)
ax2 =fig.add_subplot(212);
sns.boxplot(x='parking_options',y='price',data=data)
plt.xticks(rotation=90)
plt.show()


# ####  Analyzing laundry_options Feature

# In[ ]:


data = train.copy()
data = data[data['price']<data['price'].quantile(0.9996)]
fig=plt.figure(figsize=(18,14))
plt.subplots_adjust(hspace=0.6)
ax1 = fig.add_subplot(211);
sns.countplot(data['laundry_options'])
plt.title('laundry_options distribution',fontsize=12)
plt.xticks(rotation=90)
ax2 =fig.add_subplot(212);
sns.boxplot(x='laundry_options',y='price',data=data)
plt.xticks(rotation=90)
plt.show()


# ####  Analyzing region Feature

# In[ ]:


train['region'].value_counts()


# ####  Analyzing state Feature

# In[ ]:


train['state'].value_counts()


# In[ ]:


data = train.copy()
data = data[data['price']<data['price'].quantile(0.9996)]
plt.figure(figsize=(15,8))
sns.countplot(data['state'])
plt.title('state distribution',fontsize=12)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


data = train.copy()
data.loc[data['state']=='ga'].sort_values(by='price',ascending=False).head() # remove the outlier


# In[ ]:


train.drop(110953,inplace=True)
train.loc[data['state']=='ga'].sort_values(by='price',ascending=False).head()


# In[ ]:


temp=pd.DataFrame(train.groupby('state')['price'].mean()).reset_index()
plt.figure(figsize=(12,6))
sns.barplot(x='state',y='price',data=temp)
plt.xticks(rotation=90)


# ### Correlation heatmap

# In[ ]:


fig, ax = plt.subplots(figsize=(15,12))
corr= data.corr()
sns.heatmap(corr,annot=True,fmt='.1f',ax=ax,cmap='BrBG')
sns.set(font_scale=1.45)
plt.show()


# ## Feature Engineering and Outlier Removal

# In[ ]:


train=train[train['price']<train['price'].quantile(0.9996)]
train=train[train['sqfeet']<train['sqfeet'].quantile(0.9996)]
train.shape


# From the above figure, it is evident that there is almost 0 correlations among the features. We will try to remove the outliers inorder to make features useful for our model

# In[ ]:


train.drop(train.index[train['sqfeet']==0],inplace=True)
train.drop(train.index[train['price']==0],inplace=True)
train.shape


# In[ ]:


# Let's create some useful feature


# In[ ]:


train['pp_sqfeet'] = train['price'] / train['sqfeet']

print('Correlation with price: {:.3f}'.format(train['price'].corr(train['pp_sqfeet'])))


# In[ ]:


fig, ax = plt.subplots(figsize=(15,12))
corr= train.corr()
sns.heatmap(corr,annot=True,fmt='.1f',ax=ax,cmap='BrBG')
sns.set(font_scale=1.45)
plt.show()


# In[ ]:


plt.figure(figsize=(12,6))

sns.distplot(train['pp_sqfeet'])
plt.title('Skewness: {} and Kurtosis: {}'.format(train['pp_sqfeet'].skew(),train['pp_sqfeet'].kurtosis()))


# In[ ]:


temp=pd.DataFrame(train.groupby('region')['pp_sqfeet'].mean()).reset_index()
temp.head()


# ### Removing outliers using pp_sqfeet

# In[ ]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key,subdf in df.groupby('region'):
        m=np.mean(subdf.pp_sqfeet)
        st=np.std(subdf.pp_sqfeet)
        reduced_df = subdf[(subdf.pp_sqfeet > (m-st)) & (subdf.pp_sqfeet <= (m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out


# In[ ]:


data=train.copy()
data.shape


# In[ ]:


data=remove_pps_outliers(data)


# In[ ]:


data.shape


# In[ ]:


data['pp_sqfeet'].describe()


# In[ ]:


# Lets find out the effect of removing outliers
# our main dataframe train is not free from outliers and data dataframe has been treated with outlier


# In[ ]:


fig=plt.figure(figsize=(12,6))
ax1=fig.add_subplot(121);
sns.distplot(train['pp_sqfeet'])
plt.title('Skewness: {:.3f} and Kurtosis: {:.3f}'.format(train['pp_sqfeet'].skew(),train['pp_sqfeet'].kurtosis()))

ax2=fig.add_subplot(122);
sns.distplot(data['pp_sqfeet'])
plt.title('Skewness: {:.3f} and Kurtosis: {:.3f}'.format(data['pp_sqfeet'].skew(),data['pp_sqfeet'].kurtosis()))

print('Correlation with price: {:.3f}'.format(train['price'].corr(train['pp_sqfeet'])))
print('Correlation with price: {:.3f}'.format(data['price'].corr(data['pp_sqfeet'])))

# skewness and kurtosis have been substantially reduced. Also, Correlation has improved. 


# In[ ]:


fig=plt.figure(figsize=(12,6))
ax1=fig.add_subplot(121);
sns.distplot(train['price'])
plt.title('Skewness: {:.3f} and Kurtosis: {:.3f}'.format(train['price'].skew(),train['price'].kurtosis()))

ax2=fig.add_subplot(122);
sns.distplot(data['price'])
plt.title('Skewness: {:.3f} and Kurtosis: {:.3f}'.format(data['price'].skew(),data['price'].kurtosis()))

print('Correlation with price: {:.3f}'.format(train['price'].corr(train['sqfeet'])))
print('Correlation with price: {:.3f}'.format(data['price'].corr(data['sqfeet'])))

# Also, we have improved on other numerical features


# In[ ]:


fig=plt.figure(figsize=(12,6))
ax1=fig.add_subplot(121);
sns.distplot(train['sqfeet'])
plt.title('Skewness: {:.3f} and Kurtosis: {:.3f}'.format(train['sqfeet'].skew(),train['sqfeet'].kurtosis()))

ax2=fig.add_subplot(122);
sns.distplot(data['sqfeet'])
plt.title('Skewness: {:.3f} and Kurtosis: {:.3f}'.format(data['sqfeet'].skew(),data['sqfeet'].kurtosis()))


# In[ ]:


fig, ax = plt.subplots(figsize=(15,12))
corr= data.corr()
sns.heatmap(corr,annot=True,fmt='.1f',ax=ax,cmap='BrBG')
sns.set(font_scale=1.45)
plt.show()


# In[ ]:


# Clearly, we have improved dataset data, now let us go ahead and remove outliers from our main dataset train


# In[ ]:


train=remove_pps_outliers(train)
train.shape


# In[ ]:


train=train.loc[(train['beds']<9)] # houses with beds more than 9 are not useful, because we are trying to create more generalized model
train.shape


# In[ ]:


train=train.loc[(train['baths']<9)] # houses with baths more than 9 are not useful, because we are trying to create more generalized model
train.shape


# In[ ]:


temp= pd.DataFrame(train.groupby('beds')['sqfeet'].mean().reset_index())
temp['bedsbysqfeet']= temp['sqfeet']/temp['beds'] 
temp


# In[ ]:


# creating new feature beds_X_feet
data=train.copy()
data['beds_x_sqfeet'] = data['beds'] * data['sqfeet']

print('Correlation with price: {:.3f}'.format(data['beds_x_sqfeet'].corr(data['price'])))
data.shape


# In[ ]:



data = data[data['beds']!=0]
data['sqfeet_p_bed'] = data['sqfeet'] / data['beds']


print('Correlation with price: {:.3f}'.format(data['sqfeet_p_bed'].corr(data['price'])))
data.shape


# In[ ]:


# It is not usual that a bedroom has area less than 150 sqft. So, we are going to remove those from our dataset. By doing that we have improved our correlation of the feature beds_x_sqfeet  


# In[ ]:


data.loc[data['sqfeet_p_bed']<150]


# In[ ]:


data=data[data['sqfeet_p_bed']>150]


# In[ ]:


print('Correlation with price: {:.3f}'.format(data['sqfeet_p_bed'].corr(data['price'])))
data.shape


# In[ ]:


print('Correlation with price: {:.3f}'.format(data['beds_x_sqfeet'].corr(data['price'])))
data.shape


# In[ ]:


data.loc[data['price']<50]


# In[ ]:


data=data[data['price']>50]
data.shape


# In[ ]:


train['beds_x_sqfeet'] = train['beds'] * train['sqfeet']
train= train[(train['sqfeet']/train['beds'])>150]


# In[ ]:


train.shape


# In[ ]:


print('Correlation with price: {:.3f}'.format(train['beds_x_sqfeet'].corr(train['price'])))


# In[ ]:


train.pp_sqfeet.describe()


# In[ ]:


fig, ax = plt.subplots(figsize=(15,12))
corr= train.corr()
sns.heatmap(corr,annot=True,fmt='.1f',ax=ax,cmap='BrBG')
sns.set(font_scale=1.45)
plt.show()


# In[ ]:


train.columns


# In[ ]:


data=train.copy()
data.shape


# In[ ]:


# creating new feature beds_x_baths
data['beds_x_baths'] = data['beds'] * data['baths']
print('Correlation: {:.3f}'.format(data['price'].corr(data['beds_x_baths'])))


# In[ ]:


data['beds_by_baths'] = data['beds'] / data['baths']
data.loc[data['beds_by_baths']==np.inf]
print('Correlation: {:.3f}'.format(data['price'].corr(data['beds_by_baths'])))


# In[ ]:


m=0
summ=0
for i in data['beds_by_baths']:
    if i == np.Inf:
        pass
    else:
        summ+=i
m = summ/len(data)
m


# In[ ]:


data['beds_by_baths'].replace(np.Inf,m,inplace=True)


# In[ ]:


data.loc[data['beds_by_baths']==np.inf]


# In[ ]:


print('Correlation: {:.3f}'.format(data['price'].corr(data['beds_by_baths'])))


# In[ ]:


#usually there is no house with bathrooms greater than bedrooms. But for generalizing I have considered the difference to be less than one


# In[ ]:


data=data[~((data['baths']-data['beds'])>1)]


# In[ ]:


print('Correlation: {:.3f}'.format(data['price'].corr(data['beds_x_baths'])))


# In[ ]:


fig, ax = plt.subplots(figsize=(15,12))
corr= data.corr()
sns.heatmap(corr,annot=True,fmt='.1f',ax=ax,cmap='BrBG')
sns.set(font_scale=1.45)
plt.show()


# In[ ]:


train['beds_x_baths'] = train['beds'] * train['baths']


# In[ ]:


train_unused = train[['url', 'region_url',  'image_url', 'description', 'lat',
       'long']]


# In[ ]:


train = train[[ 'region', 'price', 'type', 'sqfeet', 'beds',
       'baths', 'cats_allowed', 'dogs_allowed', 'smoking_allowed',
       'wheelchair_access', 'electric_vehicle_charge', 'comes_furnished',
       'laundry_options', 'parking_options','state', 'pp_sqfeet', 'beds_x_sqfeet','beds_x_baths']]


# In[ ]:


details = analysis(train,'price')
details


# In[ ]:


for i in bool_features:
    print(train[i].value_counts())


# In[ ]:


# we transform the boolean features to categorical, as we are using Catboost Regressor
for col in bool_features:
    train[col] = train[col].astype('object',copy=False)


# ## Handle Missing Data

# In[ ]:


train['laundry_options']= train.groupby('type')['laundry_options'].transform(lambda i : i.fillna(i.mode()[0]))


# In[ ]:


train['parking_options']= train.groupby('type')['parking_options'].transform(lambda i : i.fillna(i.mode()[0]))


# In[ ]:


train[train['state'].isna()]


# In[ ]:


train.state[train['region']=='columbus'].head()


# In[ ]:


train['state']=train['state'].fillna('ga')


# In[ ]:


details = analysis(train,'price')
details


# In[ ]:


y = train['price']
X = train.copy()
X.drop('price',axis=1,inplace=True)


# In[ ]:


skew_data=details[abs(details['skewness'])> 0.75]
skew_data.drop('price',inplace=True)
skew_data.drop('cats_allowed',inplace=True)
skew_data.drop('dogs_allowed',inplace=True)
skew_data.drop('smoking_allowed',inplace=True)
skew_data.drop('electric_vehicle_charge',inplace=True)
skew_data.drop('comes_furnished',inplace=True)
skew_data.drop('wheelchair_access',inplace=True)

skew_data


# In[ ]:


# i dont understand why object types have skewness and kurtosis. It is woring fine in Jupyter Notebook


# ### Applying boxcox1p transformation to skewed features

# In[ ]:



from scipy.special import boxcox1p
skewed_features = skew_data.index
lam = 0.5
for feat in skewed_features:
    train[feat] = boxcox1p(train[feat], lam)


# In[ ]:


details = analysis(train,'price')
details


# ## Model using CatBoost Regressor

# In[ ]:


# splitting data
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test =train_test_split(X,y,train_size=0.75,random_state=25) 


# In[ ]:


categorical_features_indices = np.where(X.dtypes == 'object')[0]


# In[ ]:


categorical_features_indices


# In[ ]:


from catboost import CatBoostRegressor
model=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.25, loss_function='RMSE')
model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_test, y_test),plot=True)


# In[ ]:


y_prdeict = model.predict(X_test)


# In[ ]:


y_test.head()


# In[ ]:


y_prdeict[:5]


# In[ ]:


model.score(X_test,y_test)

