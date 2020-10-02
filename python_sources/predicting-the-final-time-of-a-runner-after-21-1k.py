#!/usr/bin/env python
# coding: utf-8

# Can we build a model that would accurately predict the final result of a runner given his splits until the 21st km, and his demographics? Such model can be useful for marathon broadcasting and coaching. 
# 
# Please feel free to comment and give suggestions for improvement.

# First, let's load the relevant libraries:

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from math import sqrt
from subprocess import check_output

plt.style.use('fivethirtyeight')

print(check_output(["ls", "../input"]).decode("utf8"))


# Read the data:

# In[ ]:


path = "../input/"
filename = 'marathon_results_2016.csv'
df = pd.read_csv(path + filename)


# Some feature engineering:
# 
#  1. Final result and half-way result in minutes
#  2. Time for each of the first 4 splits of 5km

# In[ ]:


def time_to_min(string):
    if string is not '-':
        time_segments = string.split(':')
        hours = int(time_segments[0])
        mins = int(time_segments[1])
        sec = int(time_segments[2])
        time = hours*60 + mins + np.true_divide(sec,60)
        return time
    else:
        return -1

df['Half_min'] = df.Half.apply(lambda x: time_to_min(x))
df['Full_min'] = df['Official Time'].apply(lambda x: time_to_min(x))
df['split_ratio'] = (df['Full_min'] - df['Half_min'])/(df['Half_min'])

df_split = df[df.Half_min > 0]

df['5K_mins'] = df['5K'].apply(lambda x: time_to_min(x))
df['10K_mins'] = df['10K'].apply(lambda x: time_to_min(x))
df['10K_mins'] = df['10K_mins'] - df['5K_mins'] 
df['15K_mins'] = df['15K'].apply(lambda x: time_to_min(x))
df['15K_mins'] = df['15K_mins'] - df['10K_mins'] -  df['5K_mins']
df['20K_mins'] = df['20K'].apply(lambda x: time_to_min(x))
df['20K_mins'] = df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']
df['25K_mins'] = df['25K'].apply(lambda x: time_to_min(x))
df['25K_mins'] = df['25K_mins'] - df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']
df['30K_mins'] = df['30K'].apply(lambda x: time_to_min(x))
df['30K_mins'] = df['30K_mins'] -df['25K_mins'] - df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']
df['35K_mins'] = df['35K'].apply(lambda x: time_to_min(x))
df['35K_mins'] = df['35K_mins'] -df['30K_mins'] -df['25K_mins'] - df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']
df['40K_mins'] = df['40K'].apply(lambda x: time_to_min(x))
df['40K_mins'] = df['40K_mins'] -  df['35K_mins'] -df['30K_mins'] -df['25K_mins'] - df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']

columns = ['20K_mins','15K_mins','10K_mins','5K_mins']
df['avg'] = df[columns].mean(axis = 1)
df['stdev'] = df[columns].std(axis = 1)

df_split = df[(~(df['5K'] == '-')) &(~(df['10K'] == '-'))&(~(df['15K'] == '-'))&(~(df['20K'] == '-'))&(~(df['25K'] == '-')) &(~(df['30K'] == '-')) &(~(df['35K'] == '-')) &(~(df['40K'] == '-'))]
df_split = df_split[df_split.split_ratio>0]


prediction_df = df_split[['Age','M/F','Half_min','Full_min','split_ratio','5K_mins','10K_mins','15K_mins','20K_mins','stdev']]


# We will need to manually calculate R square for our simple models. Therefor it is important to also verify that the manual calculation yields the same result as the sklearn built-in function:

# In[ ]:


def r_square(yhat,y):
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    return ssreg/sstot

rand_x = np.linspace(1,100,100)
rand_y = np.linspace(1,100,100) + np.random.normal(0,10,100)
plt.plot(rand_x,rand_y,'o')

test_reg = LinearRegression()
test_reg.fit(rand_x.reshape(-1, 1),rand_y)
pred_y = test_reg.predict(rand_x.reshape(-1, 1))
plt.plot(rand_x,pred_y)
plt.title('Just testing the R-square function')
print('Sklearn R square...',test_reg.score(rand_x.reshape(-1, 1),rand_y))
print('My R square...',r_square(pred_y,rand_y))


# Now we can test 2 simple approaches and use them as benchmark:
# 
#  1. Simply assume that the 2nd half of the race is identical to the 1st
#  2. Take into account the slowing down effect in 2nd part of the race,
#     and add a constant offset based on the median [slowing down
#     factor][1].
# 
#   [1]: https://www.kaggle.com/drgilermo/d/rojour/boston-results/negative-split-and-the-wall

# In[ ]:


Naive_benchmark = df_split['Half_min']*2
Naive_benchmark_split_ratio = df_split['Half_min']*(1 + np.median(df_split.split_ratio))

Naive_bacnmark_error = Naive_benchmark - df_split['Full_min']
Naive_bacnmark_split_ratio_error = Naive_benchmark_split_ratio - df_split['Full_min']

print('R Square for naive guess...',r_square(Naive_benchmark,df_split['Full_min']))
print('R Square for naive guess + slowing down factor...',r_square(Naive_benchmark_split_ratio,df_split['Full_min']))

print('Average Error for naive guess...',np.mean(Naive_bacnmark_error))
print('Average Error for naive guess + slowing down factor...',np.mean(Naive_bacnmark_split_ratio_error))

print('RMSE for naive guess...',sqrt(mean_squared_error(df_split['Full_min'], Naive_benchmark)))
print('RMSE Error for naive guess + slowing down factor...',sqrt(mean_squared_error(df_split['Full_min'], Naive_benchmark_split_ratio)))

sns.distplot(Naive_bacnmark_error, np.linspace(-70,40,200), kde = False)
sns.distplot(Naive_bacnmark_split_ratio_error, np.linspace(-70,40,200), kde = False)
plt.legend(['Naive Guess','Naive Guess + Slow Down Factor'])
plt.xlabel('Error in Minutes')
plt.ylabel('Number of Runners')


# We can see that adding the slowing down factor to the naive guess improves the RMSE dramatically. 
# 
# We can also see that although this clearly is an improvement (as can also be seen in the error histograms), it is not seen when using the R square as an error metric. Therefore I would focus on the RMSE from here on.

# Now let us see if we can improve this benchmark using regression algorithms, namely linear regression and gradient boosting regression.
# 
# First let's turn the categorical sex feature to a numeric binary feature:

# In[ ]:


def gender_to_numeric(value):
    if value == 'M':
        return 0
    else:
        return 1

prediction_df['M/F'] = prediction_df['M/F'].apply(lambda x: gender_to_numeric(x))


# ## Linear Regression

# In[ ]:


traindf, testdf = train_test_split(prediction_df, test_size = 0.2)

X_train = traindf[['Age','Half_min','5K_mins','10K_mins','15K_mins','20K_mins','stdev']]
y_train = traindf['Full_min']

X_test = testdf[['Age','Half_min','5K_mins','10K_mins','15K_mins','20K_mins','stdev']]
y_test = testdf['Full_min']

model = LinearRegression()
model.fit(X_train,y_train)
regression_prediction = model.predict(X_test)
regression_error = regression_prediction - y_test
print('R sqruare of regression...',model.score(X_test,y_test))
print('RMSE of regression...',sqrt(mean_squared_error(y_test, regression_prediction)))

sns.distplot(np.random.choice(Naive_bacnmark_error, round(0.2*len(Naive_bacnmark_error))), np.linspace(-40,40,200), kde = False)
sns.distplot(np.random.choice(Naive_bacnmark_split_ratio_error, round(0.2*len(Naive_bacnmark_error))), np.linspace(-40,40,200), kde = False)
sns.distplot(regression_error, np.linspace(-40,40,200), kde = False)
plt.xlim([-40,40])
plt.xlabel('Error in Minutes')
plt.ylabel('Number of Runners')
plt.legend(['Naive Guess','Naive Guess + Split offset','Linear Regression'])


# We can see that the regression is an improvement both in terms of R square as well as RMSE. The error histogram also looks slightly narrower. However, the difference is small compared to the educated guess. 

# ## Gradient Boosting Regression 

# In[ ]:


X_train = traindf[['Age','Half_min','5K_mins','10K_mins','15K_mins','20K_mins','M/F','stdev']]
y_train = traindf['Full_min']

X_test = testdf[['Age','Half_min','5K_mins','10K_mins','15K_mins','20K_mins','M/F','stdev']]
y_test = testdf['Full_min']

model = XGBRegressor(learning_rate=0.01, n_estimators = 2000)

model.fit(X_train,y_train)
xgb_regression_prediction = model.predict(X_test)
xgb_regression_error = xgb_regression_prediction - y_test
print('Gradient Boosting Regression R Square...',model.score(X_test,y_test))
print('RMSE of Graident Bossting Regression...',sqrt(mean_squared_error(y_test, xgb_regression_prediction)))


# The Gradient Boosting Regressor outperforms the linear regression. Let's now visually compare all the models.
# 

# ## Models Comparison

# In[ ]:


sns.distplot(regression_error, np.linspace(-40,40,200), kde = False)
sns.distplot(xgb_regression_error,np.linspace(-40,40,200), kde = False)
sns.distplot(np.random.choice(Naive_bacnmark_error, round(0.2*len(Naive_bacnmark_error))), np.linspace(-40,40,200), kde = False)
sns.distplot(np.random.choice(Naive_bacnmark_split_ratio_error, round(0.2*len(Naive_bacnmark_error))), np.linspace(-40,40,200), kde = False)

plt.xlabel('Error in minutes')
plt.legend(['Linear Regression','XGB Regression','Naive Guess','Naive Guess + split offset'], loc = 2)


# While the XGB model gives the best results in all metrics, we still don't see a significant improvement. 10 minutes error, which I would consider as big for our purpose, are still a likely outcome. The error distribution for the linear regression and XGB regression are a clear improvement from the educated guess. Moreover, it seems like the models suffer from the same bias and facing the same difficulties with certain runners as can be concluded from the long tail of the histogram. 

# In[ ]:


a = sqrt(mean_squared_error(y_test, xgb_regression_prediction))
b = sqrt(mean_squared_error(y_test, regression_prediction))
c = sqrt(mean_squared_error(df_split.Full_min, Naive_benchmark_split_ratio))
d = sqrt(mean_squared_error(df_split.Full_min, Naive_benchmark))


plt.bar([1,2,3,4],[a,b,c,d])
labels = ['XGB','Linear Regression','Educated Guess','Naive Guess']
plt.xticks([1,2,3,4], labels)
plt.ylabel('RMSE')
plt.title('Models Comparison')


# ## Feature Importance 
# 
# Let's normalize the features and explore their importance in the regression model

# In[ ]:


traindf, testdf = train_test_split(prediction_df.drop(['M/F','split_ratio'], axis = 1), test_size = 0.2)

X_train = traindf[['Age','Half_min','5K_mins','10K_mins','15K_mins','20K_mins','stdev']]
y_train = traindf['Full_min']

X_test = testdf[['Age','Half_min','5K_mins','10K_mins','15K_mins','20K_mins','stdev']]
y_test = testdf['Full_min']

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

model = LinearRegression(fit_intercept=True)
model.fit(X_train,y_train)
regression_prediction = model.predict(X_test)
regression_error = regression_prediction - y_test
print('R sqruare of regression...',model.score(X_test,y_test))
print('RMSE of regression...',sqrt(mean_squared_error(y_test, regression_prediction)))

plt.bar([1,2,3,4,5,6,7],  model.coef_)
plt.bar(8,model.intercept_)
plt.xticks([1,2,3,4,5,6,7,8],['Age','Half_min','5K_mins','10K_mins','15K_mins','20K_mins','Stdev','Intercept'])
plt.title('Regression Features')


# Unsurprisingly, the most important features are the DC offset (intercept) and the time at halfway.
# 
# The age and stability (as represented in the standard deviation of the first 4 segments) barely contribute to the regression. 
# 
# Interestingly, the contribution of the first 4 5K segments is negative. This is probably the 2nd order correction (the 1st order is the time at halfway), which represents the positive split effect (or slowing down factor).

# ## Regularization
# 
# Let's see if regularization optimization can change something:

# In[ ]:


traindf, testdf = train_test_split(prediction_df.drop(['M/F','split_ratio'], axis = 1), test_size = 0.2)
X_train = traindf[['Age','Half_min','5K_mins','10K_mins','15K_mins','20K_mins','stdev']]
y_train = traindf['Full_min']

X_test = testdf[['Age','Half_min','5K_mins','10K_mins','15K_mins','20K_mins','stdev']]
y_test = testdf['Full_min']

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

R = []
RMSE = []
for reg in np.logspace(-1,1,num = 10):
    model = Ridge(fit_intercept=True,alpha = reg)
    model.fit(X_train,y_train)
    ridge_regression_prediction = model.predict(X_test)
    ridge_regression_error = ridge_regression_prediction - y_test
    R.append(model.score(X_test,y_test))
    RMSE.append(sqrt(mean_squared_error(y_test, ridge_regression_prediction)))
    
    
plt.plot(np.logspace(-1,1,10), RMSE,'o')
plt.xlabel('L2 Regularization strength')
plt.ylabel('RMSE')
plt.title('Regularization')


# In[ ]:


model = Ridge(fit_intercept=True,alpha = 0.4)
model.fit(X_train,y_train)
ridge_regression_prediction = model.predict(X_test)
ridge_regression_error = ridge_regression_prediction - y_test
print('R Square...',model.score(X_test,y_test))
print('RMSE....',sqrt(mean_squared_error(y_test, ridge_regression_prediction)))


# Not really.

# ## Summary
# 
# The XGB model gives the best predictions. However, the results are still unsatisfying.
# 
# Any suggestions for improvement are welcomed! 
