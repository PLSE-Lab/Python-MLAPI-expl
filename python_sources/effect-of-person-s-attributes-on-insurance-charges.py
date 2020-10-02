#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[ ]:


#import libraries 
#structures
import numpy as np
import pandas as pd

#visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
from mpl_toolkits.mplot3d import Axes3D

#get model duration
import time
from datetime import date

#analysis
from sklearn.metrics import confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Description of data

# In[ ]:


#load dataset
data = '../input/insurance/insurance.csv'
dataset = pd.read_csv(data)
dataset.shape


# The red wine data consists of 1338 rows and 7 columns.

# In[ ]:


dataset.dtypes


# In[ ]:


dataset.describe()


# In[ ]:


dataset.head()


# # Data Cleaning

# In[ ]:


dataset.isnull().sum()


# In[ ]:


#check for unreasonable data
dataset.applymap(np.isreal)


# So we can see there is no null value in our dataset. <br>
# From the dataset, we can see that we need to encode the features ('sex', 'smoker', 'region') otherwise we won't be able to do analysis on these features.

# # Encoding Data

# First, lets check how many unique values in each of these features.

# In[ ]:


dataset.sex.unique()


# So we can encode 'sex' feature as '0' for female & '1' for male.

# In[ ]:


dataset.smoker.unique()


# Same for 'smoker' feature, '0' for no & '1' for yes.

# In[ ]:


dataset.region.unique()


# Then for 'region' feature, '0' for northeast, '1' for northwest, '2' for southeast & '3' for southwest.

# In[ ]:


le = LabelEncoder()


# In[ ]:


X = dataset
a = dataset['sex']
b = dataset['smoker']
c = dataset['region']


# In[ ]:


X['sex'] = le.fit_transform(X['sex'])

a = le.transform(a)


# In[ ]:


X['smoker'] = le.fit_transform(X['smoker'])

b = le.transform(b)


# In[ ]:


X['region'] = le.fit_transform(X['region'])

c = le.transform(c)


# In[ ]:


dataset = X


# In[ ]:


dataset.head()


# In[ ]:


dataset.dtypes


# # Data Visualization

# In[ ]:


sns_plot = sns.pairplot(dataset)


# In[ ]:


sns_plot = sns.distplot(dataset['charges'])


# # Pre-processing

# In[ ]:


#set x and y
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = dataset.iloc[:,0:6]
y = dataset['charges']

#stadardize data
X_scaled = StandardScaler().fit_transform(X)

#get feature names
X_columns = dataset.columns[:6]

#split train and test data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42)


# In[ ]:


dataset.head()


# ## Pearson's Correlation

# In[ ]:


#get correlation map
corr_mat=dataset.corr()


# In[ ]:


#visualise data
plt.figure(figsize=(13,5))
sns_plot=sns.heatmap(data=corr_mat, annot=True, cmap='GnBu')
plt.show()


# # Applying Machine Learning Models

# ## Linear Regression

# In[ ]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[ ]:


# import model
from sklearn.linear_model import LinearRegression

#instantiate
linReg = LinearRegression()

start_time = time.time()
# fit out linear model to the train set data
linReg_model = linReg.fit(X_train, y_train)
today = date.today()
print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:


#get coefficient values
coeff_df = pd.DataFrame(linReg.coef_, X_columns, columns=['Coefficient'])  
coeff_df


# Among all the features, it can be seen that features: 'age', 'bmi' and 'smoker' have highest effect on the insurance charges. <br>
# It makes sense because in the real world - age, obesity and smoking habits have significant influences over your health risks and thus higher insurance charges.

# In[ ]:


#validate model
y_pred = linReg.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(10)


# In[ ]:


df1.plot(kind='bar',figsize=(8,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


# print the intercept and coefficients
print('Intercept: ',linReg.intercept_)
print('r2 score: ',linReg.score(X_train, y_train))


# In[ ]:


# define input
X2 = sm.add_constant(X)

# create a OLS model
model = sm.OLS(y, X2)

# fit the data
est = model.fit()


# In[ ]:


# make some confidence intervals, 95% by default
est.conf_int()


# In[ ]:


print(est.summary())


# In[ ]:




