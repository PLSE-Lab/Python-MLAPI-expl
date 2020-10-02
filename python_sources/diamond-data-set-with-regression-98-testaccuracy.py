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


import pandas as pd
import numpy as np


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')


# In[ ]:


dataset = pd.read_csv('/kaggle/input/diamonds/diamonds.csv',)


# In[ ]:


dataset.describe()


# In[ ]:


dataset.head()


# In[ ]:


#removing index column at 0
dataset = dataset.iloc[:,1:]
dataset.head()


# In[ ]:


sns.pairplot(dataset)


# Outliers are detected in x ,y, z columns that distort the graphs. Some column show skewed distribution 

# In[ ]:


dataset.cut.value_counts().plot(kind='bar')


# Very unbalanced feature classes.

# In[ ]:


bar_plot_of_cuts = dataset.groupby('cut').mean().price.plot.barh()
bar_plot_of_cuts.set_xlim(7.5,8.2)


# Premium and ideal cut is not having high mean price that is strange. Also very good is price low than the good cut. very strange behaviour. I will leave it for sumbody having domain knowledge.

# In[ ]:


bar_plot_of_color = dataset.groupby('color').mean().price.plot.barh()
bar_plot_of_color.set_xlim(7.5,8.2)
bar_plot_of_color.set_title(' Color vs price')


# Some random category labels

# In[ ]:


bar_plot_of_carat = dataset.groupby('price').mean().carat.plot.hist(bins=20)
#bar_plot_of_carat.set_ylim(7.5,8.1)
bar_plot_of_carat.set_title(' Carat vs price')


# Price being maximum for 0.7 carat diamonds

# 

# In[ ]:


sns.heatmap(dataset.corr(),cmap="BrBG")


#  High Correlation is detected

# In[ ]:


# removing  outliers 
dataset = dataset[(dataset['x']<dataset['x'].quantile(.99)) & 
                  (dataset['y']<dataset['y'].quantile(.99)) & 
                  (dataset['z']<dataset['z'].quantile(.99)) ]

# Removing 0s in x y and z columns
dataset = dataset[(dataset['x']>0.01) & (dataset['y']>0.01) & (dataset['z']>0.01) ]


# #Log transformation of the numerical columns 
# for i in dataset.columns:
#     if dataset[i].dtype != 'O' :
#         dataset[i] = np.log(dataset[i] + 1)

# In[ ]:


#Getting dummies for categorical columns
df = pd.get_dummies(dataset,drop_first=True)


# In[ ]:


#Seggregating features and labels
X = df.drop(['price'],1)
y = df['price']


# In[ ]:


# adding 2nd degree polynomial to the features
from sklearn.preprocessing import PolynomialFeatures
Pl = PolynomialFeatures(degree=2)
X = Pl.fit_transform(X)


# In[ ]:


# test and train split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25,random_state=142)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


model =lm.fit(X_train,y_train)


# In[ ]:


# Accuracy 
y_pred = model.predict((X_test))
model.score(X_test,y_test)


# In[ ]:


# RMSE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print("RMSE: {}".format(np.sqrt(mean_squared_error((y_test),(y_pred)))))
print("R2  : {}".format(np.sqrt(r2_score((y_test),(y_pred)))))


# In[ ]:


# RMSE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print("RMSE: {}".format(np.sqrt(mean_squared_error(np.exp(y_test)-1,np.exp(y_pred)-1))))
print("R2  : {}".format(np.sqrt(r2_score(np.exp(y_test)-1,np.exp(y_pred)-1))))


# In[ ]:


plt.scatter(((y_test)),((y_test))-((y_pred)),alpha=.2)
plt.xlabel('Y_Test')
plt.ylabel('Y_Pred')
plt.title('Residuals of test set')
plt.show()


# In[ ]:



sns.distplot((np.exp(y_test)-1)-(np.exp(y_pred)-1)).set_title('Histogram of residuals')


# In[ ]:


plt.scatter((np.exp(y_test)-1),(np.exp(y_pred)-1))
plt.xlabel('Y_Test')
plt.ylabel('Y_Pred')
plt.title(' Actual vs Predicted on Test set')


# In[ ]:


output = pd.Series((np.exp(y_pred)-1))


# In[ ]:


output.to_csv('Final Output.csv')


# In[ ]:


model.coef_


# In[ ]:




