#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


df = pd.read_csv("/kaggle/input/kc-housesales-data/kc_house_data.csv")


# In[ ]:


print(df.shape)
df.info()
df.describe()


# In[ ]:


df.head()


# # Data Cleaning

# In[ ]:


#dropping date and id as they won't affect the prcie prediction
df.drop(['date','id'], inplace = True, axis=1)


#  Creating an age column so, age will become a numerical variable and can be used in prediction. So, drop 'yr_built'.

# In[ ]:


df['age'] = 2020-df['yr_built']


# In[ ]:


df.drop('yr_built', axis=1, inplace=True)


# # Exploaratory Data Analysis
# ## Univariate Analysis

# In[ ]:


import seaborn as sns
sns.distplot(df['price'])
print("skewness :", df['price'].skew())
print("kurtosis :",df['price'].kurt())


# This shows that the 'price' variable is positively skewed, so while training the model we'll have to use log transformation of 'price'

# In[ ]:


df_boxplot = df[['sqft_living', 'sqft_above', 'sqft_basement', 'sqft_living15', 'age']]
df_barplot = df[['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'grade']]


# In[ ]:


for i in df_boxplot.columns:
    #sns.set(style='white')
    plt.figure(figsize=(15,5))
    #print("boxplot of %s" %(i))
    sns.boxplot(x=i, data=df)
    plt.show()


# In[ ]:


for i in df_barplot.columns:
    plt.figure(figsize=(10,5))
    cat_num = df[i].value_counts()
    sns.barplot(x=cat_num.index, y=cat_num)
    plt.show()


# ## Bivariate Analysis

# In[ ]:


sns.set()
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True)


# The heatmap above suggests following independent variables don't affect the price much:
# * sqft_lot
# * condition
# * yr_renovated
# * zipcode
# * long
# * sqft_lot15
# 
# So, let's drop these from our dataframe

# In[ ]:


df.drop(['sqft_lot','condition','yr_renovated','zipcode','long','sqft_lot15'],axis=1, inplace=True)


# Also, dropping the outliers from the dataframe, using z-score method

# # Splitting the Data
#  We're going to split the data between training and test sets, in a 75:25 ratio.

# In[ ]:


X = df.drop('price',1)
y=df['price']
y = np.log(y) #since the price distribution is positively skewed, thus, doing logarithmic transformation


# In[ ]:


X


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)


# # Training the Model and predictions

# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)


# In[ ]:


sns.set_style("whitegrid")
sns.regplot(y_test,predictions)


# # Evaluation and Understanding Results

# In[ ]:


from sklearn.metrics import r2_score
print("score : ",r2_score(y_test,predictions))


# The accuracy score of our model is 0.7697 which means 76.97% of the predications made our correct.
