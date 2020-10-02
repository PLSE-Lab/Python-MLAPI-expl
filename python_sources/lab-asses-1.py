#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import ResidualsPlot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


rng= np.random.RandomState(1)
rng


# In[ ]:


df1 =pd.DataFrame({'x1': [rng.randint(500,2000) for i in rng.rand(50)],'x2': [rng.randint(100,500) for i in rng.rand(50)]})


x3df = pd.DataFrame({'x3': [rng.randint(0,50) for i in rng.rand(50)]+df1['x1']*3})

df1['x3']=x3df['x3']

y= df1['x3']+df1['x2']

df1['y']=y


df1


# In[ ]:


df1['x1'].corr(df1['y'])


# In[ ]:


df1['x2'].corr(df1['y'])


# In[ ]:


df1['x3'].corr(df1['y'])


# In[ ]:


corr = df1.corr(method='pearson')
corr.head()


# In[ ]:



df1.plot(kind="scatter", # or `us_gdp.plot.scatter(`
    x='x1',
    y='y',
    title="Graph of Y vs X1",
    figsize=(12,8)
)
plt.title("From %d to %d" % (
    df1['x1'].min(),
    df1['x1'].max()
),size=8)
plt.suptitle("Graph of Y vs X1",size=12)
plt.ylabel("Y")


# In[ ]:


df1.plot(kind="scatter",
    x='x2',
    y='y',
    title="Graph of Y vs X2",
    figsize=(12,8)
)
plt.title("From %d to %d" % (
    df1['x2'].min(),
    df1['x2'].max()
),size=8)
plt.suptitle("Graph of Y vs X2",size=12)
plt.ylabel("Y")


# In[ ]:


#independent and dependent vars
X_data = df1[['x1','x2',]]
Y_data = df1['y']

#import train-test split
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)


# In[ ]:


reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)


# In[ ]:


print("Regression Coefficients")
pd.DataFrame(reg.coef_,index=X_train.columns,columns=["Coefficient"])


# In[ ]:



# Make predictions using the testing set
test_predicted = reg.predict(X_test)
test_predicted


# In[ ]:


df2 = X_test.copy()
df2['predicted Y']=test_predicted
df2['Actual Y']=y_test


# In[ ]:


#shouldnt this graph show a stronger link between test and actual?
sns.residplot(test_predicted, y_test, lowess=True, color="g")


# In[ ]:


#trying to see if this plot gives better results
# it does, whats the diff

df2.plot.scatter(
    x='predicted Y',
    y='Actual Y',
    figsize=(12,8)
)

plt.suptitle("Predicted Y vs Actual Y",size=12)
plt.ylabel("Actual Y")
plt.xlabel("Predicted Y")


# In[ ]:


print('R squared score is %.2f' % r2_score(y_test, test_predicted))

#this shows how stong the relationship between the dependent and independent vars are. A higher score implies (1) 
#high strong relationship


# In[ ]:


# x1 and x2 combined can determine with a large degree of accuracy, the value of  Y


# In[ ]:


reg.intercept_


# Regression formula
# 
# Y = 18.41 + 3.0*x1+ 1.0*x2
# 
# 
# x1=30
# x2=40
# 
# Y= 18.41+ (3*30) + 40
# Y= 148.41

# In[ ]:


#mean absolute error
mean_absolute_error(y_test, test_predicted)


# In[ ]:


#root mean square errpr
rmse = sqrt( mean_squared_error(y_test, test_predicted))
rmse


# In[ ]:


mean_squared_error(y_test, test_predicted)


# In[ ]:




