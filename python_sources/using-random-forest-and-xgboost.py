#!/usr/bin/env python
# coding: utf-8

# ** If you like my notebook, please upvote my work!**
# 
# **If you use parts of this notebook in your scripts/notebooks, giving some kind of credit for instance link back to this notebook would be very much appreciated. Thanks in advance! :) **
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Importing important libraries

# In[ ]:


import calendar
import seaborn as sb
import xgboost as xgb
import plotly.express as px
import pandas_profiling as pp
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_log_error,make_scorer
from sklearn.model_selection import train_test_split,GridSearchCV


# # Loading the training dataset

# In[ ]:


#Reading the file
df_train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")


# # Creating a profile report

# In[ ]:


pp.ProfileReport(df_train)


# # Checking for null values

# In[ ]:


df_train.isnull().sum(axis=0)


# No data cleaning is required since no Null values are found!

# In[ ]:


df_train.columns


# # Data visualization:

# ## Heatmap of all the continuous values in the file.

# In[ ]:


corr = df_train[['temp','atemp','humidity', 'windspeed','casual', 'registered',
                 'count']].corr()
f,axes = plt.subplots(1,1,figsize = (8,8))
sb.heatmap(corr,square=True,annot = True,linewidth = .5,center = 1.4,ax = axes)


# ## Line plot for all continuous values in file 

# In[ ]:


y = ['casual','registered','count']
list_continuous = ['temp','atemp','humidity','windspeed']
n=3
s= 15
f,axes = plt.subplots(4,3,figsize = (s,s))
counter = 0
for i in list_continuous:
    for j in y:
        sb.lineplot(x = i , y = j , data  = df_train, ax = axes[counter//n][counter%n])
        counter+=1


# ### The conclusions drawn are:
# 1. We can see that temp and atemp have a very strong positive correlation therefore we can use drop atemp as a variable without any loss of information. 
# 
# 2. We can infer from the correlaton matrix and lineplots that windspeed has no significant correlation with the casual,registered or count which we wish to predict so we can remove that

# ## Data visualizaton for non continuous variables in data

# First we have to separate the individual date and time for each data point into hour,day,month and year.

# In[ ]:


df_train['Date'] = pd.DatetimeIndex(df_train['datetime']).date
df_train['Hour'] = pd.DatetimeIndex(df_train['datetime']).hour
df_train['Day'] = pd.DatetimeIndex(df_train['datetime']).day
df_train['Month'] = pd.DatetimeIndex(df_train['datetime']).month
df_train['Year'] = pd.DatetimeIndex(df_train['datetime']).year
df_train['Weekday'] = pd.DatetimeIndex(df_train['datetime']).weekday_name


# In[ ]:


a = []
for i in df_train.index:
    a.append('Total Count : '+str(df_train['count'][i]))
df_train['count_vis'] = a


# In[ ]:


fig = px.line(x = 'Date', y = "count", data_frame = df_train,color = 'Hour',
              range_y = (0,1150),hover_data = ['Hour','Date','casual','registered'],
              title = 'Interactive LinePlot of the whole dataset(Hover for more details)',
              hover_name = 'count_vis', text = None,height = 670,width = 980)
fig.show()


# **The sudden periodic changes between the differrent regions is due to the missing data.These are the regions in which the regions we have to predict the result.**

# ## 1. Season

# In[ ]:


f,axes = plt.subplots(1,3,figsize = (17,7))
sb.despine(left = True)
x = 'season'

sb.barplot(x = x , y = 'casual' , data = df_train, saturation = 1, ax =  axes[0])
sb.barplot(x = x , y = 'registered' , data = df_train, saturation = 1, ax = axes[1])
sb.barplot(x = x , y = 'count' , data = df_train, saturation = 1, ax = axes[2])


# ## 2. Holiday

# In[ ]:


f,axes = plt.subplots(1,3,figsize = (17,7))
sb.despine(left = True)
x = 'holiday'

sb.barplot(x = x , y = 'casual' , data = df_train, saturation = 1, ax =  axes[0] ,)
sb.barplot(x = x , y = 'registered' , data = df_train, saturation = 1, ax = axes[1])
sb.barplot(x = x , y = 'count' , data = df_train, saturation = 1, ax = axes[2])


# ## 3. Working day

# In[ ]:


f,axes = plt.subplots(1,3,figsize = (17,7))
sb.despine(left = True)
x = 'workingday'

sb.barplot(x = x , y = 'casual' , data = df_train, saturation = 1, ax =  axes[0] ,)
sb.barplot(x = x , y = 'registered' , data = df_train, saturation = 1, ax = axes[1])
sb.barplot(x = x , y = 'count' , data = df_train, saturation = 1, ax = axes[2])


# Combining the graphs of casual and registered into one of may make it seem like that holiday and workingday have no dependence on count but we can clearly see that holiday increases the casual amount by upto 40% and a reverse kind of trend is observed in working day so it is reasonable to take two different models one for casual and another for registered.

# Therefore what I will attempt to do is make two separate models for the casual and the registerd training them separately and then adding the result to get the count.

# ## 4. Weather

# In[ ]:


f,axes = plt.subplots(1,3,figsize = (17,7))
sb.despine(left = True)
x = 'weather'

sb.barplot(x = x , y = 'casual' , data = df_train, saturation = 1, ax =  axes[0] )
sb.barplot(x = x , y = 'registered' , data = df_train, saturation = 1, ax = axes[1])
sb.barplot(x = x , y = 'count' , data = df_train, saturation = 1, ax = axes[2])


# ## 5. Date and Time

# ### 5.a. Hour

# In[ ]:


f,axes = plt.subplots(1,3,figsize = (19,7))
sb.despine(left = True)
x = 'Hour'

sb.barplot(x = x , y = 'casual' , data = df_train, saturation = 1, ax =  axes[0] ,)
sb.barplot(x = x , y = 'registered' , data = df_train, saturation = 1, ax = axes[1])
sb.barplot(x = x , y = 'count' , data = df_train, saturation = 1, ax = axes[2])


# We can see that in the final graph there are two prominent peaks.
# 1. At 8 a.m
# 2. At 5 p.m
# Where as the trend of casual is mostly the same throughout the day. 
# 
# **From this we can conclude that the registered are mostly people going on their jobs which explains the peaks at the start and end of office hours. Clearly these people would have a more definite and predictible schedule and are therefore more likely to be registered.** In order to test this hypothesis we plot some more graphs.

# In[ ]:


df_train.groupby('Weekday').count().index


# In[ ]:


df_train_temp = df_train.groupby(['Hour','Weekday']).mean().reset_index()
dic = {'Weekday':['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday',
                  'Sunday']}
dic1 = {'registered':'Average count of registered poeple commuting.',
        'count': 'Average people commuting','Hour':'Hour of the day',
        'Weekday':'Day of the week'}
fig = px.line(x = 'Hour', y = "registered", data_frame = df_train_temp.reset_index(),
              color = 'Weekday',hover_data = ['count'],category_orders = dic,
              title = 'Interactive LinePlot of the registered separated by weekday(Hover for more details)',
              labels = dic1,range_y = [0,550],height = 670,width = 980)
fig.show()


# Clearly We can see that on the days of saturday and sunday,the general trend more or less follows the same trend as of casual where as on weekdays there is a completely different trend of two peaks at 8 am and 5 pm which confirms that those peaks are due to the workpeople commuting.

# In[ ]:


df_train_temp = df_train.groupby(['Hour','Weekday']).mean().reset_index()
dic = {'Weekday':['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday',
                  'Sunday']}
dic1 = {'casual':'Average count of casual poeple commuting.',
        'count': 'Average people commuting','Hour':'Hour of the day',
        'Weekday':'Day of the week'}
fig = px.line(x = 'Hour', y = "casual", data_frame = df_train_temp.reset_index(),
              color = 'Weekday',hover_data = ['count'],category_orders = dic,
              title = 'Interactive LinePlot of the casual separated by weekday(Hover for more details)',
              labels = dic1,range_y = [0,550],height = 670,width = 980)
fig.show()


# We can observe that on the days of saturday and sunday,there is a surge in the demand. This makes sense as these days are holidays for most of the poeple which results in higher people commuting (probably for leisure activities).

# ### 5.b. Day

# In[ ]:


f,axes = plt.subplots(1,3,figsize = (19,7))
sb.despine(left = True)
x = 'Day'

sb.barplot(x = x , y = 'casual' , data = df_train, saturation = 1, ax =  axes[0] ,)
sb.barplot(x = x , y = 'registered' , data = df_train, saturation = 1, ax = axes[1])
sb.barplot(x = x , y = 'count' , data = df_train, saturation = 1, ax = axes[2])


# **From the above graphs we can conclude that the feature day has hardly any influence over the features registered and count.**

# ### 5.c. Month

# In[ ]:


f,axes = plt.subplots(1,3,figsize = (19,7))
sb.despine(left = True)
x = 'Month'
#order = ['January','February','March','April','May','June','July','August','September','October','November','December']
plot = sb.barplot(x = x , y = 'casual' , data = df_train, saturation = 1, ax =  axes[0])
sb.barplot(x = x , y = 'registered' , data = df_train, saturation = 1, ax = axes[1])
sb.barplot(x = x , y = 'count' , data = df_train, saturation = 1, ax = axes[2])


# ### 5.d. Year 

# In[ ]:


f,axes = plt.subplots(1,3,figsize = (19,7))
sb.despine(left = True)
x = 'Year'

sb.barplot(x = x , y = 'casual' , data = df_train, saturation = 1, ax =  axes[0] ,)
sb.barplot(x = x , y = 'registered' , data = df_train, saturation = 1, ax = axes[1])
sb.barplot(x = x , y = 'count' , data = df_train, saturation = 1, ax = axes[2])


# We can see that overal the company made growth from the year 2011 to the year 2012.

# In[ ]:


df_train.describe()


# In[ ]:


df_train.columns


# # One Hot Encoding for each of the categorical data columns and removing unnecesary ones

# ## 1. Season

# In[ ]:


for i in df_train.groupby('season').count().index:
    s = 's'+str(i)
    a=[]
    for j in df_train.season:
        if j==i:
            a.append(1)
        else:
            a.append(0)
    df_train[s]=a
df_train.sample(5)


# ## 2. Weather 

# In[ ]:


for i in df_train.groupby('weather').count().index:
    s = 'w'+str(i)
    a=[]
    for j in df_train.weather:
        if j==i:
            a.append(1)
        else:
            a.append(0)
    df_train[s]=a
df_train.sample(5)


# ## 3. Hour

# In[ ]:


for i in df_train.groupby('Hour').count().index:
    s = 'Hour'+str(i)
    a=[]
    for j in df_train.Hour:
        if j==i:
            a.append(1)
        else:
            a.append(0)
    df_train[s]=a
df_train.sample(5)


# ## 4.Month

# In[ ]:


for i in df_train.groupby("Month").count().index:
    s = 'Month' + str(i)
    a = []
    for j in df_train.Month:
        if j==i:
            a.append(1)
        else:
            a.append(0)
    df_train[s] = a
df_train.sample(5)


# In[ ]:


df_train.columns


# ## Removing unnecessary columns

# In[ ]:


df_train = df_train[['Hour0', 'Hour1', 'Hour2', 'Hour3', 'Hour4', 'Hour5',
       'Hour6', 'Hour7', 'Hour8', 'Hour9', 'Hour10', 'Hour11', 'Hour12',
       'Hour13', 'Hour14', 'Hour15', 'Hour16', 'Hour17', 'Hour18', 'Hour19',
       'Hour20', 'Hour21', 'Hour22', 'Hour23','Month1', 'Month2', 'Month3',
       'Month4', 'Month5', 'Month6', 'Month7', 'Month8', 'Month9', 'Month10',
       'Month11', 'Month12','Year','s1','s2','s3','s4','holiday','workingday',
        'w1','w2','w3','w4','temp','humidity','casual','registered']]


# In[ ]:


df_train.describe()


# In[ ]:


df_train.columns


# In[ ]:


df_train.describe()


# # Preparing training and testing sets

# ### 1. Training set

# In[ ]:


df_train_x = df_train.drop('casual',axis = 1).drop('registered',axis=1)
df_train_x.describe()


# Training set will be the same for both the models.

# ### 2. Training set

# In[ ]:


df_reg_train_y = df_train['registered']
df_reg_train_y.describe


# In[ ]:


df_cas_train_y = df_train['casual']
df_cas_train_y.describe


# # Machine learning model

# ### Splitting data into train and test sets

# In[ ]:


x1_train, x1_test, y1_train, y1_test = train_test_split(df_train_x, df_reg_train_y,
                                                        test_size=0.15, random_state=42)
x2_train, x2_test, y2_train, y2_test = train_test_split(df_train_x, df_cas_train_y,
                                                        test_size=0.15, random_state=42)


# ### Using polynomial on the dataset

# In[ ]:


poly = PolynomialFeatures(degree=2)
poly_x1_train = poly.fit_transform(x1_train)
poly_x1_test = poly.fit_transform(x1_test)
poly_x2_train = poly.fit_transform(x2_train)
poly_x2_test = poly.fit_transform(x2_test)


# ### Trying different models to see which one works best for the given data.

# In[ ]:


rf = RandomForestRegressor()
xg = xgb.XGBRegressor()
parameter = {"max_depth": [1,2,3,4,5,6],
             "eta": [0.01,0.03,0.05],
             "alpha":[0],'n_estimators': [100,500,800,1000,1200,1400]}

parameters = {'n_estimators':[50,100,150,200,250],
              'min_impurity_decrease':[0.0,0.001,0.01],
              'max_depth':[20,40,60,80,100]}

models = ['Normal Linear Regression: ','Linear Regression over polynomial: ',
          'Random Forest Regressor: ','XG Boosting: ']


# ### Defining a custom scorer function for the models

# In[ ]:


def custom_scorer(y_true,y_pred):
    for i in range(len(y_pred)):
        if y_pred[i]<0:
            y_pred[i] = 1
    return np.sqrt(mean_squared_log_error(y_true, y_pred ))
scorer = make_scorer(custom_scorer,greater_is_better = False)


# Here I have defined a scorer function as we are using mean squared log loss which does not work on negative values and the models were sometimes predicting negative values which messes with the scores and since we know that these values will always be positive we just replace them with the minimum value in the train set which is 1.

# ## Using different models for registered

# In[ ]:


predict = []
reg = LinearRegression().fit(x1_train, y1_train)
pre_reg = reg.predict(x1_test)

reg_poly = LinearRegression().fit(poly_x1_train, y1_train)
pre_reg_poly = reg_poly.predict(poly_x1_test)

rf_reg = GridSearchCV(rf, parameters, cv=5, verbose=2,scoring = scorer,n_jobs = -1)
rf_reg.fit(x1_train, y1_train)
pre_rf_reg = rf_reg.predict(x1_test)

xg_reg = GridSearchCV(xg,parameter,cv=5,verbose = 2 , scoring = scorer, n_jobs = -1)
xg_reg.fit(x1_train, y1_train)
pre_xg_reg = xg_reg.predict(x1_test)

predict.append(pre_reg)
predict.append(pre_reg_poly)
predict.append(pre_rf_reg)
predict.append(pre_xg_reg)


# In[ ]:


for prediction in range(len(predict)):
    pre = []
    for p in predict[prediction]:
        if p < 1:
            pre.append(1)
        else:
            pre.append(p)
    print(models[prediction]+str(np.sqrt(mean_squared_log_error(y1_test, pre ))))


# We can therefore conclude that the Random Forest Model works best for predicting registered.

# ## Using different models for casual 

# In[ ]:


predict = []
cas = LinearRegression().fit(x2_train, y2_train)
pre_cas = cas.predict(x2_test)

cas_poly = LinearRegression().fit(poly_x2_train, y2_train)
pre_cas_poly = cas_poly.predict(poly_x2_test)

rf_cas = GridSearchCV(rf, parameters, cv=5, verbose=2,scoring = scorer,n_jobs = -1)
rf_cas.fit(x2_train, y2_train)
pre_rf_cas = rf_cas.predict(x2_test)

xg_cas = GridSearchCV(xg,parameter,cv=5,verbose = 2 , scoring = scorer, n_jobs = -1)
xg_cas.fit(x2_train, y2_train)
pre_xg_cas = xg_cas.predict(x2_test)

predict.append(pre_cas)
predict.append(pre_cas_poly)
predict.append(pre_rf_cas)
predict.append(pre_xg_cas)


# In[ ]:


for prediction in range(len(predict)):
    pre = []
    for p in predict[prediction]:
        if p < 1:
            pre.append(1)
        else:
            pre.append(p)
    print(models[prediction]+str(np.sqrt(mean_squared_log_error(y2_test, pre ))))


# We can therefore conclude that the XGBoost Model works best for predicting casual.

# In[ ]:


print("For Random Forest Model: ")
print("\t Best Parametres for registered are: ",end='')
print(rf_reg.best_params_)
print("\t Best Parametres for casual are: ",end = '')
print(rf_cas.best_params_)
print("\nFor XGBoost Model: ")
print("\t Best Parametres for registered are: ",end='')
print(xg_reg.best_params_)
print("\t Best Parametres for casual are: ",end = '')
print(xg_cas.best_params_)


# # Plotting the residual plots

# In[ ]:


predict1 = []

reg1 = LinearRegression().fit(x1_train, y1_train)
pre_reg1 = reg1.predict(x1_test)

reg1_poly = LinearRegression().fit(poly_x1_train, y1_train)
pre_reg1_poly = reg1_poly.predict(poly_x1_test)

rf1 = RandomForestRegressor(n_estimators = 250,min_impurity_decrease = 0.001,
                            max_depth=60).fit(x1_train, y1_train)
pre_rf1 = rf1.predict(x1_test)

xg1 = xgb.XGBRegressor(alpha = 0, eta = 0.03, n_estimators = 1200, 
                       max_depth = 6).fit(x1_train,y1_train)
pre_xg1 = xg1.predict(x1_test)

for i in range(pre_reg1.size):
    if pre_reg1[i]<1:
        pre_reg1[i] = 1 
    if pre_reg1_poly[i]<1:
        pre_reg1_poly[i] = 1
    if pre_rf1[i]<1:
        pre_rf1[i] = 1
    if pre_xg1[i]<1:
        pre_xg1[i] = 1

predict1.append(pre_reg1)
predict1.append(pre_reg1_poly)
predict1.append(pre_rf1)
predict1.append(pre_xg1)

x1_final = x1_test.copy()
x1_final['Output'] = y1_test
x1_final['Linear'] = pre_reg1
x1_final['Lin_poly'] = pre_reg1_poly
x1_final['RF'] = pre_rf1
x1_final['XG'] = pre_xg1
x1_final['Resid'] = y1_test-pre_reg1
x1_final['Resid_poly'] = y1_test-pre_reg1_poly
x1_final['Resid_rf'] = y1_test - pre_rf1
x1_final['Resid_xg'] = y1_test - pre_xg1

for prediction in range(len(predict1)):
    print(models[prediction]+
          str(np.sqrt(mean_squared_log_error(y1_test,predict1[prediction] ))))


# In[ ]:


predict2 = []

reg2 = LinearRegression().fit(x2_train, y2_train)
pre_reg2 = reg2.predict(x2_test)

reg2_poly = LinearRegression().fit(poly_x2_train, y2_train)
pre_reg2_poly = reg2_poly.predict(poly_x2_test)

rf2 = RandomForestRegressor(n_estimators = 100,min_impurity_decrease = 0.001,
                            max_depth=40).fit(x2_train, y2_train)
pre_rf2 = rf2.predict(x2_test)

xg2 = xgb.XGBRegressor(alpha = 0, eta = 0.05, n_estimators = 800,
                       max_depth = 6).fit(x2_train,y2_train)
pre_xg2 = xg2.predict(x2_test)

for i in range(pre_reg2.size):
    if pre_reg2[i]<1:
        pre_reg2[i] = 1 
    if pre_reg2_poly[i]<1:
        pre_reg2_poly[i] = 1
    if pre_rf2[i]<1:
        pre_rf2[i] = 1
    if pre_xg2[i]<1:
        pre_xg2[i] = 1

predict2.append(pre_reg2)
predict2.append(pre_reg2_poly)
predict2.append(pre_rf2)
predict2.append(pre_xg2)

x2_final = x2_test.copy()
x2_final['Output'] = y2_test
x2_final['Linear'] = pre_reg2
x2_final['Lin_poly'] = pre_reg2_poly
x2_final['RF'] = pre_rf2
x2_final['XG'] = pre_xg2
x2_final['Resid'] = y2_test-pre_reg2
x2_final['Resid_poly'] = y2_test-pre_reg2_poly
x2_final['Resid_rf'] = y2_test - pre_rf2
x2_final['Resid_xg'] = y2_test - pre_xg2

for prediction in range(len(predict2)):
    print(models[prediction]+
          str(np.sqrt(mean_squared_log_error(y2_test, predict2[prediction]))))


# In[ ]:


name1 = ['Residual for casual without polynomial features'] *1633
name2 = ['Residual for casual with polynomial features'] *1633
name3 = ['Residual for registered without polynomial features'] *1633
name4 = ['Residual for registered with polynomial features'] *1633
dic = {'Lin': 'Output Predicted using linear model',
       'Lin_poly': 'Output Predicted using polynomial features',
       'RF' : 'Output Predicted using RandomForest Model', 
       'XG': 'Output Predicted using XGBoost Model',
       'Resid':'Deviation from predicted','Output':'Expected Output',
       'Resid_poly':'Deviation from predicted','Resid_rf':'Deviation from predicted',
       'Output':'Expected Output','Resid_xg':'Deviation from predicted'}
fig1 = px.scatter(data_frame = x1_final,x = 'Linear', y = 'Resid',hover_data = ['Output'],
                  labels = dic,hover_name = name3,color_discrete_sequence = ['red'])
fig2 = px.scatter(data_frame = x1_final,x = 'Lin_poly', y = 'Resid_poly',
                  hover_data = ['Output'],labels = dic,hover_name = name4,
                  color_discrete_sequence = ['blue'])
fig3 = px.scatter(data_frame = x2_final,x = 'Linear', y = 'Resid',hover_data = ['Output'],
                  labels = dic,hover_name = name1,color_discrete_sequence = ['darkgreen'])
fig4 = px.scatter(data_frame = x2_final,x = 'Lin_poly', y = 'Resid_poly',
                  hover_data = ['Output'],labels = dic,hover_name = name2,
                  color_discrete_sequence = ['gold'])

trace1 = fig1['data'][0]
trace2 = fig2['data'][0]
trace3 = fig3['data'][0]
trace4 = fig4['data'][0]


fig = make_subplots(rows=2, cols=2,horizontal_spacing =0.1,vertical_spacing  = 0.2,
                    row_titles = ['Linear Model','Polynomial Model'],
                    column_titles = ['Casual','Registered'],
                    x_title = 'Residual plots for Registered and Casual under different models (Hover for more details)')

fig.add_trace(trace3, row=1, col=1)
fig.add_trace(trace4, row=2, col=1)
fig.add_trace(trace1, row=1, col=2)
fig.add_trace(trace2, row=2, col=2)

fig.show()


# **Since the residual plots show a conical divergence therefore we can conclude that Linear Regression is definitely not a suitable model for the predicting in the above distribution of data**

# In[ ]:


name5 = ['Residual for casual using RandomForest Model'] *1633
name6 = ['Residual for casual using XGBoost Model'] *1633
name7 = ['Residual for registered using RandomForest Model'] *1633
name8 = ['Residual for registered using XGBoost Model'] *1633

dic = {'Lin': 'Output Predicted using linear model',
       'Lin_poly': 'Output Predicted using polynomial features',
       'RF' : 'Output Predicted using RandomForest Model',
       'XG': 'Output Predicted using XGBoost Model',
       'Resid':'Deviation from predicted','Output':'Expected Output',
       'Resid_poly':'Deviation from predicted','Resid_rf':'Deviation from predicted',
       'Output':'Expected Output','Resid_xg':'Deviation from predicted'}

fig5 = px.scatter(data_frame = x1_final,x = 'RF', y = 'Resid_rf',hover_data = ['Output'],
                  labels = dic,hover_name = name7,color_discrete_sequence = ['red'])
fig6 = px.scatter(data_frame = x1_final,x = 'XG', y = 'Resid_xg',hover_data = ['Output'],
                  labels = dic,hover_name = name8,color_discrete_sequence = ['blue'])
fig7 = px.scatter(data_frame = x2_final,x = 'RF', y = 'Resid_rf',hover_data = ['Output'],
                  labels = dic,hover_name = name5,color_discrete_sequence = ['darkgreen'])
fig8 = px.scatter(data_frame = x2_final,x = 'XG', y = 'Resid_xg',hover_data = ['Output'],
                  labels = dic,hover_name = name6,color_discrete_sequence = ['gold'])

trace5 = fig5['data'][0]
trace6 = fig6['data'][0]
trace7 = fig7['data'][0]
trace8 = fig8['data'][0]

fig = make_subplots(rows=2, cols=2,horizontal_spacing =0.1,vertical_spacing  = 0.2,
                    row_titles = ['Random Forest','XGBoost'],
                    column_titles = ['Casual','Registered'],
                    x_title = 'Residual plots for Registered and Casual under different models (Hover for more details)')

fig.add_trace(trace5, row=1, col=2)
fig.add_trace(trace6, row=2, col=2)
fig.add_trace(trace7, row=1, col=1)
fig.add_trace(trace8, row=2, col=1)
fig.show()


# **Since the residual plots of Random Forest regressor and XGBoost regressor do not show a conical divergence therefore we can conclude that they are suitable model for the predicting in the above distribution of data**

# ### Retraining the decision tree over the whole dataset for submission.

# In[ ]:


rf1 = RandomForestRegressor(n_estimators = 200,min_impurity_decrease = 0.001,
                            max_depth=80).fit(df_train_x,df_reg_train_y)
xg2 = xgb.XGBRegressor(alpha = 0, eta = 0.05, max_depth = 6,
                       n_estimators = 800).fit(df_train_x,df_cas_train_y)


# # Predicting output over test set

# ### Reading the test file

# In[ ]:


df_test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')


# In[ ]:


test=df_test
test.describe()


# ## Processing of the test file

# In[ ]:


test['mth'] = pd.DatetimeIndex(test['datetime']).month
test['Year'] = pd.DatetimeIndex(test['datetime']).year
test['dy'] = pd.DatetimeIndex(test['datetime']).day
test['hr'] = pd.DatetimeIndex(test['datetime']).hour

for i in test.groupby("season").count().index:
    s = 's' + str(i)
    a = []
    for j in test.season:
        if j==i:
            a.append(1)
        else:
            a.append(0)
    test[s] = a
for i in test.groupby("weather").count().index:
    s = 'w' + str(i)
    a = []
    for j in test.weather:
        if j==i:
            a.append(1)
        else:
            a.append(0)
    test[s] = a
for i in test.groupby('hr').count().index:
    s = 'Hour'+str(i)
    a=[]
    for j in test.hr:
        if j==i:
            a.append(1)
        else:
            a.append(0)
    test[s]=a
for i in test.groupby("mth").count().index:
    s = 'Month' + str(i)
    a = []
    for j in test.mth:
        if j==i:
            a.append(1)
        else:
            a.append(0)
    test[s] = a
test.sample(10)


# In[ ]:


test = test[['Hour0','Hour1','Hour2','Hour3','Hour4','Hour5','Hour6','Hour7','Hour8',
             'Hour9','Hour10','Hour11','Hour12','Hour13','Hour14','Hour15','Hour16',
             'Hour17','Hour18','Hour19','Hour20','Hour21','Hour22','Hour23','Month1',
             'Month2','Month3','Month4','Month5','Month6','Month7','Month8','Month9',
             'Month10','Month11','Month12','Year','s1','s2','s3','s4','holiday',
             'workingday','w1','w2', 'w3','w4','temp','humidity']]
test.describe


# ## Predicting the output over test set

# In[ ]:


pre_reg = rf1.predict(test)
pre_cas = xg2.predict(test)

final_predictions = pd.DataFrame(pre_cas+pre_reg,columns = ['cout'])

final_predictions.describe


# In[ ]:


s=[]
for j in final_predictions.cout:
    if int(j)<1:
        s.append(1)
    else:
        s.append(j)
final_predictions['count'] = s 


# **Since we know that the output should never be less than 1, replace all negative values with 1.**

# In[ ]:


final_predictions.describe


# In[ ]:


final_predictions['datetime']=df_test['datetime']
final_predictions = final_predictions[['datetime','count']]


# In[ ]:


final_predictions.describe()


# ## Exporting output to csv

# In[ ]:


final_predictions.to_csv('submission.csv',index=False)


# In[ ]:




