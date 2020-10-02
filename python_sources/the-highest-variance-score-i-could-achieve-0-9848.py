#!/usr/bin/env python
# coding: utf-8

# # **The highest *Variance Score* I could achieve (0.9848)**
# 
# In this Kernel, I am going to share my thoughts and findings for the Diamonds dataset.  
# I would be greatful if you leave your comments about any sections you like or dislike/disagree.   
# 
# **General Steps**:
# 1. [**Exploratory Data Analysis (EDA)**](#there_you_go_1)
# 2. [**Data Cleaning**](#there_you_go_2)
# 3. [**Preprocessing and Feature Engineering**](#there_you_go_3)
# 4. [**Training Machine Learning Algorithms**](#there_you_go_4)
# 5. [**Hyperparameter Tuning**](#there_you_go_5)

# In[ ]:


import numpy as np
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Importing data visualization libraries

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')
df.head()


# In[ ]:


# The Unnamed:0 column does not add any useful information. Let's simply delete it.

df.drop('Unnamed: 0', axis=1,inplace=True)


# In[ ]:


# High level information about the dataset

df.info()


# **53940 rows and 10 columns (9 features & 1 target).  
# As you can see above, there are 3 columns with Categorical data type: cut, color and clarity.  
# We will deal with them in a bit.  
# Also, it looks like there is no missing value at all (*superficially though*).**

# In[ ]:


# Let's get some high level insight about our numerical features

df.describe()


# **There is something fishy here!  
# The min for x, y and z are zero which does not make sense.  
# On top of that, the min for depth (which is multiplication of x, y and z) is not zero!  
# We need to take care of this issue, for sure.**

# In[ ]:


# Let's see how many rows (instances) has either x, y or z values equal to zero

len(df[(df['x']==0) | (df['y']==0) | (df['z']==0)])


# Just 20 rows out of 53,940 rows have either x, y or z values equal to zero.  
# Dropping all of them should not be harmful to our final results.

# In[ ]:


df = df[(df['x']!=0) & (df['y']!=0) & (df['z']!=0)]


# In[ ]:


# Now let's get some high level insight about our categorical features

df.describe(exclude=[np.number])


# In[ ]:


# Let's find out all the unique categories of the categorical features

category_list = ['cut','color', 'clarity']
for cat in category_list:
    print(f"Unique values of {cat} column: {df[cat].unique()}\n")


# According to the description of the features, **cut**, **color** and **clarity** are all Oridinal categorical features. There are various ways to convert Ordinal features into Numerical. OrdinalEncoder() is the most famous one. However, I am going to use factorize method (You can check out the following article to figure out why: [https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9](http://)).  
# If there were any Nominal category, I would have used pd.get_dummies or OneHotEncoding (just a reminder). 

# In[ ]:


# Pay attention that the order of categories are from the worst to the best.

cut = pd.Categorical(df['cut'], categories=['Fair','Good','Very Good','Premium','Ideal'], ordered=True)
labels_cut, unique = pd.factorize(cut, sort=True)
df['cut'] = labels_cut

color = pd.Categorical(df['color'], categories=['J','I','H','G','F','E','D'], ordered=True)
labels_color, unique = pd.factorize(color, sort=True)
df['color'] = labels_color

clarity = pd.Categorical(df['clarity'], categories=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'], ordered=True)
labels_clarity, unique = pd.factorize(clarity, sort=True)
df['clarity'] = labels_clarity


# In[ ]:


# Let's take a look at our data frame now

df.head()


# In[ ]:


# Alright way better now. Time to check the corrolations between the features

plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),annot=True,square=True,cmap='RdYlGn')


# The target variable price has the highest correlation with **carat(0.92)**. 
# 
# price is highly correlated to **x(0.89)**, **y(0.87)** and **z(0.87)** and also to themselves. At this point, there are two different approaches one can take. Either keep x, y and z as separate features, or drop them all and add ***xyz or volume*** as a new feature to the dataframe. I have tried both methods and it turned out keeping x, y and z, results in better performance.
# 
# price has surprisingly negative correlation with **cut(-0.053)**, **color(-0.17)**, **clarity(-0.15)** and **depth(-0.011)**.

# In[ ]:


# How's the distribution of the target variale

plt.figure(figsize=(12,5))
sns.distplot(df['price'],bins=50)


# In[ ]:


# Seems like there are lots of outliers in the price variable, for better visualization, I will use the boxplot:

plt.figure(figsize=(10,3))
sns.boxplot(data=df,x=df['price'])


# In[ ]:


# Let's calculate the max in the boxplot above 

iqr = df['price'].quantile(q=0.75) - df['price'].quantile(q=0.25)
maximum = df['price'].quantile(q=0.75) + 1.5*iqr
maximum


# In[ ]:


# percentage of the instances above the maximum

len(df[df['price']>maximum])/len(df)*100


# **6.55%** is relatively a considerable number. Therefore, for now we will keep these instances in the dataframe rather than dropping them.  

# In[ ]:


# Separating features form the target variable
# Then, spliting the data into train and test datasets prior to feature scaling to avoid data leakage.
# I chose the test_size=0.1 since the number of instances are big enough.

X = df.drop('price',axis=1)
y = df['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# The next step is to perform feature scaling.  
# There are different methods to choose between, **StandardScaler**, **MinMax Scaler**, **MaxAbs Scaler** and **Robust Scaler**.  
# I chose ***StandardScaler*** since it returned better results (you can try other ones and play with them yourself).

# In[ ]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# In[ ]:


# Importing all the required libraries

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
import xgboost
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Instanciating all the models that we are going to apply

lr = LinearRegression()
lasso = Lasso()
ridge = Ridge()
svr = SVR()
knr = KNeighborsRegressor()
dtree = DecisionTreeRegressor()
rfr = RandomForestRegressor()
abr = AdaBoostRegressor(n_estimators=1000)
mlpr = MLPRegressor()
xgb = xgboost.XGBRegressor()


# In[ ]:


# For the sake of automation, let's create a function to train the model and generate the variance score 

def R2_function(regressor,X_train,y_train,X_test,y_test):
    regressor.fit(X_train,y_train)
    predictions = regressor.predict(X_test)
    return (metrics.explained_variance_score(y_test,predictions))


# In[ ]:


models_list = [lr, lasso, ridge, svr, knr, dtree, rfr, abr, mlpr, xgb]

for model in models_list:
    print(f'{model} R2 score is: {R2_function(model,X_train,y_train,X_test,y_test)} \n')


# The top three models are **RandomForestRegressor(R2=0.9846)**, **XGBRegressor(R2=0.9830)** and **KNeighborsRegressor(R2=0.9717)**.  
# 
# Now it is time to perform **Hyperparameter Tuning** by using **GridSearchCV**.

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


# Random Forest GridSearch

params_dict = {'n_estimators':[20,40,60,80,100], 'n_jobs':[-1],'max_features':['auto','sqrt','log2']}
rfr_GridSearch = GridSearchCV(estimator=RandomForestRegressor(), param_grid=params_dict,scoring='r2')
rfr_GridSearch.fit(X_train,y_train)


# In[ ]:


rfr_GridSearch.best_params_


# In[ ]:


rfr_GridSearch_BestParam = RandomForestRegressor(max_features='auto',n_estimators=100,n_jobs=-1)

rfr_GridSearch_BestParam.fit(X_train,y_train)
predictions = rfr_GridSearch_BestParam.predict(X_test)
print(f"R2 score: {metrics.explained_variance_score(y_test,predictions)}")
print(f"Mean absolute error: {metrics.mean_absolute_error(y_test,predictions)}")
print(f"Mean squared error: {metrics.mean_squared_error(y_test,predictions)}")
print(f"Root Mean squared error: {np.sqrt(metrics.mean_squared_error(y_test,predictions))}")


# In[ ]:


# Residual Histogram

sns.distplot((y_test-predictions),bins=100)


# **Thanks for reading and agian, I would appreciate your comments for improvement purposes.**

# In[ ]:




