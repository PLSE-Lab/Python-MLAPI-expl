#!/usr/bin/env python
# coding: utf-8

# **Context**
# 
# The insurance.csv dataset contains 1338 observations (rows) and 7 features (columns). The dataset contains 4 numerical features (age, bmi, children and expenses) and 3 nominal features (sex, smoker and region).

# Let us first import all the packages we need.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
from plotly import tools
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# Using Pandas let us read the dataset.

# In[ ]:


df = pd.read_csv('../input/insurance.csv')


# Have a look at the data using head()

# In[ ]:


df.head()


# info() gives information about each column

# In[ ]:


df.info()


# We can see that there are no missing values.
# Categorical columns include sex, smoker and region.
# Numerical columns include age, bmi, children and expenses.
# Input features for the model are age, sex, bmi, children, smoker, region.
# Output of the model is expenses.

# In[ ]:


df.describe()


# In[ ]:


df.describe(exclude=['int64','float64'])


# Let us do some EDA through visualization

# In[ ]:


df_bar_male = df[df['sex']=='male'].groupby(by =['age'])['expenses'].sum()
df_plot_male = pd.DataFrame(df_bar_male,columns=['Age','expenses'])
df_plot_male['Age']=df_plot_male.index
df_plot_male.reset_index(level=0, inplace=True)
df_plot_male.drop(columns='age',inplace=True)

df_bar_female = df[df['sex']=='female'].groupby(by =['age'])['expenses'].sum()
df_plot_female = pd.DataFrame(df_bar_female,columns=['Age','expenses'])
df_plot_female['Age']=df_plot_male.index
df_plot_female.reset_index(level=0, inplace=True)
df_plot_female.drop(columns='age',inplace=True)


# In[ ]:


x0 = df_plot_male['Age']
y0 = df_plot_male['expenses']
x1 = df_plot_female['Age']
y1 = df_plot_female['expenses']
male = go.Bar(
    x=x0,y=y0,
    opacity=0.75,
    name = 'male'
)
female = go.Bar(
    x=x1,y=y1,
    opacity=0.75,
    name = 'female'
)
data = [male,female]
layout = go.Layout(barmode='stack',    xaxis = dict(
        range=[18,65]))
fig = go.Figure(data=data)
py.iplot(fig, filename='grouped bars')


# In[ ]:


sns.jointplot(x=df.age,y=df.expenses,data = df)


# In[ ]:


plt.figure(figsize=(15,16))
sns.jointplot(x=df.bmi,y=df.expenses,data = df)


# In[ ]:


df_encoded = pd.get_dummies(df)


# In[ ]:


df_encoded.head()


# Removing sex_male column since if a 0 in sex_female indicates sex male only and we don't want redundant columns that may overfit our model. This applies to smoker column too.

# In[ ]:


df_encoded.drop(columns=['sex_male','smoker_yes'],inplace=True)


# In[ ]:


X = df_encoded.drop(columns='expenses')

y = df_encoded['expenses']

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 101)
model = LinearRegression()
model.fit(X_train,y_train)


# In[ ]:


model.coef_


# In[ ]:


model.intercept_


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[ ]:


train_predict = model.predict(X_train)

mae_train = mean_absolute_error(y_train,train_predict)

mse_train = mean_squared_error(y_train,train_predict)

rmse_train = np.sqrt(mse_train)

r2_train = r2_score(y_train,train_predict)

mape_train = mean_absolute_percentage_error(y_train,train_predict)


# In[ ]:


test_predict = model.predict(X_test)

mae_test = mean_absolute_error(test_predict,y_test)

mse_test = mean_squared_error(test_predict,y_test)

rmse_test = np.sqrt(mean_squared_error(test_predict,y_test))

r2_test = r2_score(y_test,test_predict)

mape_test = mean_absolute_percentage_error(y_test,test_predict)


# In[ ]:


print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('TRAIN: Mean Absolute Error(MAE): ',mae_train)
print('TRAIN: Mean Squared Error(MSE):',mse_train)
print('TRAIN: Root Mean Squared Error(RMSE):',rmse_train)
print('TRAIN: R square value:',r2_train)
print('TRAIN: Mean Absolute Percentage Error: ',mape_train)
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('TEST: Mean Absolute Error(MAE): ',mae_test)
print('TEST: Mean Squared Error(MSE):',mse_test)
print('TEST: Root Mean Squared Error(RMSE):',rmse_test)
print('TEST: R square value:',r2_test)
print('TEST: Mean Absolute Percentage Error: ',mape_test)


# In[ ]:




