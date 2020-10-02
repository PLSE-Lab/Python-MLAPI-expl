#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Loading the data and importing libraries

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


advert = pd.read_csv('/kaggle/input/advertising-dataset/advertising.csv')
advert.head()


# In[ ]:


advert.info()


# In[ ]:


advert.columns


# ### Exploratory Data Analysis

# In[ ]:


import seaborn as sns
sns.distplot(advert.Sales)


# From the above, we can say that "sales" column is normally distributed.

# In[ ]:


sns.distplot(advert.Newspaper)


# From the above, we can say that "Newspaper" column is right skewed, and even the price at which newspaper is sold is very less

# In[ ]:


sns.distplot(advert.Radio)


# From the above, we can say that for the "Radio" column the distribution is more of uniform, that is the spend on radio is more of a uniform distribution.

# In[ ]:


sns.distplot(advert.TV)


# From the above, we can say that for the "TV" column the distribution is more of uniform, that is the spend on TV is more of a uniform distribution. Moreover, spend on TV is much higher as compare to those of Radio and Newspaper

# ### Exploring relationship between Predictors and Response

# In[ ]:


sns.pairplot(advert, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=7, 
            aspect=0.7, kind='reg')


# From the above, we can say that spend on TV is highly correlated with Sales, but for Radio and Newpaper, nothing like that can be said, but there is weak realtion between Radio and Sales, Newspaper and Sales

# In[ ]:


# Correlation value between TV and Sales
advert.TV.corr(advert.Sales)


# In[ ]:


# Correlation value between Radio and Sales
advert.Radio.corr(advert.Sales)


# In[ ]:


# Correlation value between Newspaper and Sales
advert.Newspaper.corr(advert.Sales)


# In[ ]:


advert.corr()


# In[ ]:


# Heatmap for showing the correlation values
sns.heatmap(advert.corr(), annot=True)


# As only the TV variable is highly correlated with Sales, so we'll use only these two variables to construct the model, and the model which we'll build is Simple Linear Regression.

# ### Creating the Simple Linear Regression Model

# In[ ]:


X = advert[['TV']]
X.head()


# In[ ]:


print(type(X))
print(X.shape)


# In[ ]:


y = advert.Sales
print(type(y))
print(y.shape)


# In[ ]:


# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[ ]:


# Displaying the shape of each of the train test dataframe
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


# Creating the model
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(X_train, y_train)


# ### Interpreting model coefficients

# In[ ]:


print(linreg.intercept_)
print(linreg.coef_)


# ### Making predictions with our model
# 

# In[ ]:


y_pred = linreg.predict(X_test)
y_pred[:5]


# ### Model evaluation metrics

# In[ ]:


from sklearn import metrics


# In[ ]:


# Calculating RMSE (Root mean Squared Error)
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:




