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


# <h2 style="color:blue" align="left"> 1. Load Data </h2>

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('/kaggle/input/weight-height/weight-height.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


df.count()


# In[ ]:


df['Gender'].value_counts()


# <h2 style="color:blue" align="left"> 2. Missing Values </h2>

# In[ ]:


df.isnull().sum()


# <h2 style="color:blue" align="left"> 3. EDA(Exploratory Data Analysis) </h2>

# In[ ]:


df.describe()


# In[ ]:


a = pd.DataFrame(df['Weight'])
b = pd.DataFrame(df['Height'])


# In[ ]:


import statsmodels.api as sms
model = sms.OLS(b,a).fit()
model.summary()


# In[ ]:


sns.heatmap(df.corr(), annot=True, cmap='viridis')


# In[ ]:


sns.countplot(df.Gender)


# In[ ]:


plt.figure(figsize=(7,6))
sns.boxplot(x='Gender', y='Height', data=df)


# In[ ]:


plt.figure(figsize=(7,6))
sns.boxplot(x='Gender', y='Weight', data=df)


# In[ ]:


sns.pairplot(df, hue='Gender', size=4)


# <h2 style="color:green" align="left"> Univariate Analysis ---->  plotting only a single feature </h2>

# ### a. histogram

# In[ ]:


plt.figure(figsize=(5, 4))
sns.distplot(df['Height']);
plt.axvline(df['Height'].mean(),color='blue',linewidth=2)

plt.figure(figsize=(5, 4))
sns.distplot(df['Weight']);
plt.axvline(df['Weight'].mean(),color='red',linewidth=2)


# In[ ]:


plt.figure(figsize=(7,6))
males['Height'].plot(kind='hist',bins=50, alpha=0.3,color='blue')
females['Height'].plot(kind='hist',bins=50, alpha=0.3,color='red')
plt.title('Height distribution')
plt.legend(['Males','Females'])
plt.xlabel('Height in')
plt.axvline(males['Height'].mean(),color='blue',linewidth=2)
plt.axvline(females['Height'].mean(),color='red',linewidth=2);


# 
# ### b. kde plot (kernel distribution estimation)

# In[ ]:


plt.figure(figsize=(7,6))
df.Height.plot(kind="kde", title='Univariate: Height KDE', color='c');


# In[ ]:


plt.figure(figsize=(7,6))
df.Weight.plot(kind="kde", title='Univariate: Height KDE', color='c');


# ### c. Boxplot

# In[ ]:


sns.boxplot(df.Weight)


# In[ ]:


sns.boxplot(df.Height)


# 
# <h2 style="color:green" align="left"> Bivariate Analysis ---->  plotting two variables </h2>
# 
# ###         a. Scatter plot

# In[ ]:


df.plot(figsize=(8,7), kind='scatter',x='Height',y='Weight');


# - From above graph observed, there is a linear relationship b/n Height and Weight. As height increases weight also increases.

# In[ ]:


males=df[df['Gender']=='Male']
females=df[df['Gender']=='Female']
fig,ax = plt.subplots()
males.plot(figsize=(9,8), kind='scatter', x='Height', y='Weight', ax=ax, color='blue',alpha=0.3, title='Male and Female Distribution')
females.plot(figsize=(9,8), kind='scatter', x='Height', y='Weight', ax=ax, color='red', alpha=0.3, title='Male and Female Populations');


# - Observed from graph, compared to men womens are less height and weight.

# <h2 style="color:green" align="left"> Outliers </h2>

# <h3 style='color:purple'> 1. Detect outliers using IQR </h3>
# ### Height

# In[ ]:


Q1 = df.Height.quantile(0.25)
Q3 = df.Height.quantile(0.75)
Q1, Q3


# In[ ]:


IQR = Q3 - Q1
IQR


# In[ ]:


lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
lower_limit, upper_limit


# - Here are the outliers

# In[ ]:


df[(df.Height<lower_limit)|(df.Height>upper_limit)]


# ### Remove ouliers from Height column

# In[ ]:


df_no_outlier_height = df[(df.Height>lower_limit)&(df.Height<upper_limit)]
df_no_outlier_height


# ### Weight

# In[ ]:


Q1 = df.Weight.quantile(0.25)
Q3 = df.Weight.quantile(0.75)
Q1, Q3


# In[ ]:


IQR = Q3 - Q1
IQR


# In[ ]:


lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
lower_limit, upper_limit


# In[ ]:


df[(df.Height<lower_limit)|(df.Height>upper_limit)]


# ### Remove ouliers from Height column

# In[ ]:


df_no_outlier_Weight = df[(df.Height>lower_limit)&(df.Height<upper_limit)]
df_no_outlier_Weight


# <h2 style="color:blue" align="left"> Data Preprocessing </h2>

# ### Converting Categorical Variables to Numeric by using Pandas get_Dummies

# In[ ]:


df[['Female','Male']] = pd.get_dummies(df['Gender'])
df.head()


# In[ ]:


df.drop('Gender',axis=1,inplace=True)


# In[ ]:


df.head()


# ### Standard Scalar

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


X = df.drop('Height',axis=1)
y = df['Height']


# ### Train the model

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ### Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
LinReg = LinearRegression()
LinReg.fit(X_train, y_train)


# In[ ]:


y_pred = LinReg.predict(X_test)
y_pred


# In[ ]:


y_test


# In[ ]:


LinReg.score(X_test, y_test)


# In[ ]:


print(LinReg.coef_)
print(LinReg.intercept_)


# In[ ]:


from sklearn.metrics import r2_score,mean_squared_error
r2_score(y_test,y_pred)


# In[ ]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:


plt.figure(figsize=(7,6))
sns.scatterplot(X_train.Weight, y_train)
plt.plot(X_train.Weight, LinReg.predict(X_train), c='r')


# In[ ]:


plt.figure(figsize=(7,6))
sns.scatterplot(X_test.Weight, y_test,color='r')
plt.plot(X_test.Weight,y_pred, c='b')


# ### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.predict(X_test)


# In[ ]:


model.score(X_test, y_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
Log = LogisticRegression()


# In[ ]:


Output = pd.DataFrame(X_test['Weight'], y_test)
Output

