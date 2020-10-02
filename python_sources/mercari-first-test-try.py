#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 12,6
import warnings
warnings.filterwarnings('ignore')
seq_col_brew = sns.color_palette("YlGnBu_r", 4)
sns.set_palette(seq_col_brew)


# In[ ]:


dataset = pd.read_csv('../input/train.tsv', sep='\t', usecols=['item_condition_id', 'shipping', 'brand_name', 'price', 'category_name'])


# In[ ]:


dataset.head()


# In[ ]:


x = dataset[['item_condition_id', 'shipping', 'brand_name', 'category_name']]
y = dataset['price']


# In[ ]:


dataset.info(memory_usage='deep')


# In[ ]:


x.isnull().sum()


# In[ ]:


x.shape


# In[ ]:


x['brand_name'].fillna('Other', inplace=True)


# In[ ]:


x.isnull().sum()


# In[ ]:


dataset[dataset['category_name'].isnull()]


# In[ ]:


x.isnull().sum()


# In[ ]:


x.shape


# In[ ]:


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.80, random_state = 0)


# In[ ]:


x_train.info()


# In[ ]:


x_train.shape


# In[ ]:


x_train.isnull().sum()


# In[ ]:





# In[ ]:


x_train.head()


# ## Data Analysis

# In[ ]:


#To check Gaussian Distribution.
sns.distplot(y)


# In[ ]:


#Log distribution
sns.distplot(np.log(y + 1))


# In[ ]:


sns.distplot(y[y < 100])


# In[ ]:


y.mode()


# In[ ]:


y.mean()


# In[ ]:


y.median()


# In[ ]:


#How do you know what a high standard deviation is? 
#https://www.researchgate.net/post/What_do_you_consider_a_good_standard_deviation
y.std()


# In[ ]:


#Viewing the an independent variable and it's different distribution 
x_train.item_condition_id.hist()


# In[ ]:


#1 if shipping fee is paid by seller and 0 by buyer
x_train.shipping.hist()


# In[ ]:


#The seller takes a hit by paying for the shipping him/herself
sns.barplot(x='shipping', y='price', data=dataset)


# In[ ]:


dataset.describe(include=['O']) 


# In[ ]:


dataset.describe()


# In[ ]:


sns.kdeplot(data=y, shade=True, bw=.85)


# In[ ]:


sns.barplot(x='item_condition_id', y='price', hue='shipping', data=dataset)


# In[ ]:


sns.barplot(x='item_condition_id', y='price', data=dataset)


# ## Summary Data Analysis
# 1. There are outliers in the price data (could I apply feature scaling without having the y?)
# 2. 5 condition id is more expensive - meaning 5 is higher quality
# 3. Majourity of items are either in 1 or 2 condition

# In[ ]:





# In[ ]:


x_train["brand_name"].value_counts()


# In[ ]:





# In[ ]:


x_train.isnull().sum()


# In[ ]:


#Splitting the categories into sub categories


# In[ ]:


category_columns = ['Top_Level_Category'] + ['Second_Level_Category'] + ['Third_Level_Category']


# In[ ]:


category_columns


# In[ ]:





# In[ ]:


x_train.head()


# In[ ]:


new_categories = x_train['category_name'].str.extract('(\w*)\/(\w*)\/(\w*)', expand=True)


# In[ ]:


new_categories.columns


# In[ ]:


new_categories.columns = category_columns


# In[ ]:


x_train = pd.concat([x_train, new_categories], axis=1)


# In[ ]:


x_train.head()


# In[ ]:


columns = ['Top_Level_Category', 'Second_Level_Category', 'Third_Level_Category']


# In[ ]:


x_train.isnull().sum()


# In[ ]:


new_categories.isnull().sum()


# In[ ]:





# In[ ]:


for col in columns:
   x_train[col].fillna('Other', inplace=True)


# In[ ]:


x_train.head()


# In[ ]:


x_train = x_train.drop('category_name', axis=1)


# In[ ]:





# In[ ]:


x_train['brand_name'] = x_train['brand_name'].astype('category').cat.codes
x_train['Top_Level_Category'] = x_train['Top_Level_Category'].astype('category').cat.codes
x_train['Second_Level_Category'] = x_train['Second_Level_Category'].astype('category').cat.codes
x_train['Third_Level_Category'] = x_train['Third_Level_Category'].astype('category').cat.codes


# In[ ]:


x_train.head()


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[ ]:


'''linreg = LinearRegression(n_jobs=-1)
cv_scores = (linreg, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
print(cv_scores.mean(), cv_scores.std())'''


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
pred = np.ones((submission.shape[0]))


# In[ ]:


pred


# In[ ]:


pred * y_train.mean()


# In[ ]:


pred = pred * y_train.mean()


# In[ ]:


submission.shape


# In[ ]:





# In[ ]:


submission['price'] = pred


# In[ ]:


submission.to_csv('sample_submission.csv', index=False)

