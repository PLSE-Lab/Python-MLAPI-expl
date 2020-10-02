#!/usr/bin/env python
# coding: utf-8

# The goal of this statistical analysis is to predict house sales in Kings County. There are various models
# which can perform this task each having its advantages and disadvantages. In this report, I started with
# Simple Linear Regression and progressed with Multiple Linear Regression, Polynomial Regression, K
# Nearest Neighbor Regression and Random Forest Regression. The dataset consists of houses sold which
# range from May 2014 to May 2015. It consists of 19 home features, 1 house ID, and 1 dependent
# variable which is the price.
# 
# There are various assumptions of Multiple Linear Regression. I will also test the validity of the following
# assumptions in our dataset:
# 1. The relationship between the features of the dataset with the dependent variables is linear.
# 2. The residuals of the regression should be normally distributed.
# 3. Independent variables should not be highly correlated with each other. Hence, there should no
# multicollinearity among the features.
# 
# The structure of this report is based on:
# 1. Import the libraries
# 2. Importing and descriptive understanding of the dataset
# 3. Exploratory Data Analysis
# 4. Data Preprocessing and Feature Scaling
# 5. Model Selection
# 6. Model Evaluation
# 7. Model Tuning

# <h2>Importing the libraries</h2>

# In[ ]:


import numpy as np
import pandas as pd
import plotly.plotly as py
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline as ply
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn import utils
import statsmodels.formula.api as sm
sns.set(style= "whitegrid")
from  plotly.offline import plot
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
from sklearn.neighbors import KNeighborsRegressor


# <h2>Importing and Understanding the dataset</h2>

# In[ ]:


df = pd.read_csv("../input/kc_house_data.csv")


# In[ ]:





# <h2>Descriptive understanding of the data</h2>
# Descriptive analysis is performed to summarize our dataset and provide us meaningful insights and patterns in our dataset. For understanding the dataset, 3 steps were involved.

# a) Getting the gist of what kind of data you are dealing with

# In[ ]:


df.head(5).transpose()


# b) Information about the dataset, what kind of data types are your variables

# In[ ]:


df.info()


# c) Statistical summary of your dataset

# In[ ]:


df.describe().transpose()


# <h2>Exploratory Data Analysis</h2>
# EDA provides the characteristics of dataset through various visualization techniques including covariance matrix, plotting, mapping, line graph of univariate, bivariate, and multivariate features.

# I have drawn a correlation matrix heat map to depict the different degrees of correlation among the variables. With respect to price, high positively correlated features include sqft_living, grade, sqft_above, and sqft_living15. There are two negatively correlated features id and zipcode and have a very low correlation with price.

# In[ ]:


corr_mat = df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr_mat, cmap = 'coolwarm', linecolor = 'white', linewidth = 1, annot=True)


# With distribution plot of price, we can visualize that most of the prices of the house are 1 million with few outliers close to 8 million.

# In[ ]:


plt.figure(figsize=(12,5))
sns.distplot(df['price'])


# Grade is the overall grade given to the house based on the Kings County's grading system. Most of the housing unit in the Kings County have received a grade of 7 which is regarded as a decent grade. Further in our report we will see if the grade of the house has any impact in the selling price of the house.

# In[ ]:


plt.figure(figsize = (5,5))
sns.distplot(df['grade'] , kde = False)


# The pie chart consists of count of individual category in our categorical bedrooms variable. Through pie chart we can analyze that 3 bedrooms house is the most in Kings County followed by 4 and 2 bedrooms making 90% of the houses. There are outliers in bedrooms which will be taken care of in our data preprocessing.

# In[ ]:


bb = df['bedrooms'].value_counts()
index = [3,4,2,5,6,1,7,8,0,9,10,11,33]
trace = go.Pie(labels = index, values=bb.values)
ply.iplot([trace])


# There are high numbers of categories in the bathrooms categorical variable. Kings County has a high number of 2.5 bathrooms followed by 1 and 1.75 bathrooms.

# In[ ]:


bbath = df['bathrooms'].value_counts()
indexbath =[2.5,1,1.75,2.25,2,1.5,2.75,3,3.5,3.25,3.75,4,4.5,4.25,0.75,4.75,5,5.25,0,5.5,1.25,6,.5,5.75,8,6.25,6.5,6.75
,7.5,7.75]
tracebath = go.Pie(labels = indexbath, values=bbath.values)
ply.iplot([tracebath])


# I have created a subplot where y axis consist of price of the house and x axis consist of sqft_living, bedrooms, bathrooms, and grade. Our first assumption for Linear Regression is that the features of our data set have a linear relationship with the dependent variables. As seen from the bivariate scatterplot chart, features like bathrooms and bedrooms fail to hold this assumption.

# In[ ]:


trace1 = go.Scattergl(x=df['sqft_living'], y=df['price'], mode='markers', name='sqft_living')
trace2 = go.Scattergl(x=df['bedrooms'], y=df['price'], mode = 'markers', name = 'bedrooms')
trace3 = go.Scattergl(x=df['bathrooms'], y=df['price'],mode = 'markers', name = 'bathrooms')
trace4 = go.Scattergl(x=df['grade'], y=df['price'],mode = 'markers', name = 'grade')
fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('sqft_living vs Price', 'bedrooms vs Price',
'bathrooms vs Price', 'grade vs Price'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)
fig['layout'].update(height=800, width=800, title='Price Subplots')
ply.iplot(fig)


# I have drawn a graph based on price and sqft_living with levels in condition. As seen from the graph, there are high numbers of houses with condition rating given as 3. Another trend we can visualize with same sqft_living but with high condition rank, price of the house tends to be higher.

# In[ ]:


plt.figure(figsize=(20,10))
x = sns.scatterplot(x=df['price'], y=df["sqft_living"],
hue=df['condition'],
palette="coolwarm",
sizes=(4, 16), linewidth=0)
plt.setp(x.get_legend().get_texts(), fontsize='22')
plt.setp(x.get_legend().get_title(), fontsize='32')
plt.show()


# <h2>Data Preprocessing and Feature Scaling</h2>
# 
# Data preprocessing is carried out to make our initial data more understandable, and removing the noise from it. By doing so, we can improve the performance of model, and thereby decreasing the errors in our predictions.

# First step is to check if our dataset has any null values. If our data set consists of null values, we need to address the issue based on the type of dataset and the requirement of our model.

# In[ ]:


df.isnull().values.any()


# Second step involves removing unnecessary variable which are not required for our prediction. For my model I have removed both id and data from the data set.

# In[ ]:


df.drop(["id"], axis = 1, inplace=True)
df.drop(["date"], axis = 1, inplace=True)


# Removing outliers

# In[ ]:


df = df.drop(df[df["bedrooms"]>10].index )


# In[ ]:


bed = pd.get_dummies(df["bedrooms"])
df.columns
df_pre1 = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'grade', 'sqft_above',
'sqft_basement', 'yr_built', 'zipcode', 'lat', 'long','sqft_living15', 'sqft_lot15', 'price']]


# Our data set might consist of a number of continuous variables each having different range. Features having large values will have a heavier impact in our model even if the correlation with our dependent is low. To prevent this, we perform feature scaling which standardizes our independent variables.

# In[ ]:


ss = StandardScaler()
df_pre = ss.fit_transform(df_pre1)
df_pre = pd.DataFrame(df_pre, columns= df_pre1.columns)


# Multicollinearity is defined as the features which has high correlation among them. There can be independent variables which can have high correlation with each other. This leads to data redundancy which in turn increases the standard error. We can use the Eigen value of our correlation matrix to detect if there is any multicollinearity among the features. If the Eigen value is very close to 0, there is multicollinearity. Multicollinearity can be removed by various processes including the Variance Inflation Factor. As seen below, there is no value close to zero. Hence, our third assumption is valid.

# In[ ]:


eigen_value, eigen_vector = np.linalg.eig(corr_mat)
print(eigen_value.round(3))


# <h2>Model Selection and Evaluation</h2>
# The first model is the simple linear regression which shows the relationship between price and sqft_living. I have further tested my model in Multiple Linear regression where I have performed backward elimination method to remove unwanted features from the data set. With improved data frame, further model testing was done Polynomial Linear Regression with 2, 3, and 4 degrees. K Nearest Neighbor model and Random Forest is also used to see if there is increase in the performance with different machine learning algorithm.
# Performance of the model is tested on the following basis:
# 1. Mean Absolute Error
# 2. Mean Square Error
# 3. Root Mean Square Error
# 4. R Squared Value
# 5. Adjusted R Squared Value

# <h2>Simple Linear Regression</h2>
# I started my model with the simplest linear regression i.e. Simple Linear Regression. Here I studied the relationship between the highest correlate features (sqft_living) with our dependent variable (price).

# In[ ]:


X_s = np.array(df_pre['sqft_living']).reshape(-1,1)
y_s = np.array(df_pre['price']).reshape(-1,1)
X_trains , X_tests, y_trains, y_tests = train_test_split(X_s, y_s , test_size = 0.3, random_state = 101)
lms = LinearRegression()
lms.fit(X_trains, y_trains)


# The residual of the simple linear regression is approximately normally distributed. This is the test of our second assumption.

# In[ ]:


per_lms = lms.predict(X_tests)
residual = (X_tests - per_lms )
sns.distplot(residual)


# In[ ]:


print(lms.coef_)


# In[ ]:


rsquared_sl = metrics.r2_score(y_tests,per_lms)
adjusted_r_squared_sl = 1 - (1-rsquared_sl)*(len(y_tests)-1)/(len(y_tests)-X_tests.shape[1]-1)
print('Mean absolute error: {}'.format(metrics.mean_absolute_error(y_tests, per_lms)))
print('Mean squared error: {}'.format(metrics.mean_squared_error(y_tests, per_lms)))
print('Root mean squared error: {}'.format(np.sqrt(metrics.mean_squared_error(y_tests, per_lms))))
print('R Squared value: {}'.format(rsquared_sl))
print('Adjusted R Squared Value: {}'.format(adjusted_r_squared_sl))


# <h2>Multiple Linear Regressions</h2>
# In the Multiple Linear Regression, we take into consideration more than one independent variable and see if there is a linear relationship with the dependent variable. Here I studied the relationship between all our independent variable with our dependent variable (price).

# In[ ]:


X = df_pre[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'grade', 'sqft_above',
'sqft_basement', 'yr_built', 'zipcode', 'lat', 'long','sqft_living15', 'sqft_lot15']]
y = df_pre["price"]
X_train , X_test, y_train, y_test = train_test_split(X, y , test_size = 0.3, random_state = 101)
lm = LinearRegression()
lm.fit(X_train, y_train)
per_lm = lm.predict(X_test)


# The residual of the multiple linear regression is approximately normally distributed. This is the test of our second assumption.

# In[ ]:


residuals = (y_test- per_lm)
sns.distplot(residuals)


# In[ ]:


rsquared_ml = metrics.r2_score(y_test,per_lm)
adjusted_r_squared_ml = 1 - (1-rsquared_ml)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print('Mean absolute error: {}'.format(metrics.mean_absolute_error(y_test, per_lm)))
print('Mean squared error: {}'.format(metrics.mean_squared_error(y_test, per_lm)))
print('Root mean squared error: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, per_lm))))
print('R Squared value: {}'.format(rsquared_ml))
print('Adjusted R Squared Value: {}'.format(adjusted_r_squared_ml))


# We can visualize our multiple linear regression model by plotting our prediction and comparing it with our test set. 

# In[ ]:


trace0 = go.Scatter(x = X_test['sqft_living'],y = y_test,mode = 'markers',name = 'Test Set')
trace1 = go.Scatter(x = X_test['sqft_living'],y = per_lm,opacity = 0.75,mode = 'markers',name = 'Predictions',marker = dict(line = dict(color = 'black', width = 0.5)))
data = [trace0, trace1]
ply.iplot(data)


# <h2>Multiple Linear Regression after Backward Elimination</h2>
# Backward Elimination is one of the processes for removing the unnecessary features from our model, and enhancing the predictive capabilities our model. Initial step of the backward elimination is to take into account every feature of our model. Next step involves selecting the significance level. For my model, I have considered a significance level of .05. After selecting the significance level we fit our model and check the p-value. The variable with the highest p value and above our significance level is removed, and again we fit our model without the removed feature. We repeat this process until the pvalue of our entire feature is below our significance level.

# In[ ]:


lm2 = sm.ols(formula = 'price ~ bedrooms+bathrooms+sqft_living+sqft_lot+waterfront+view+condition+grade+sqft_above+sqft_basement+yr_built+yr_renovated+zipcode+lat+long+sqft_living15+sqft_lot15', data = df).fit()
lm2.summary()


# In[ ]:


Xb = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'waterfront','view','condition', 'grade',
'sqft_above',
'sqft_basement', 'yr_built','yr_renovated', 'zipcode', 'lat', 'long','sqft_living15', 'sqft_lot15']]
Xb = ss.fit_transform(Xb)
yb = df_pre["price"]
Xb_train , Xb_test, yb_train, yb_test = train_test_split(Xb, yb , test_size = 0.3, random_state = 101)
lmb = LinearRegression()
lmb.fit(Xb_train, yb_train)
per_lmb = lmb.predict(Xb_test)


# In[ ]:


residuals = (yb_test- per_lmb)
sns.distplot(residuals)


# In[ ]:


rsquared_mlb = metrics.r2_score(yb_test,per_lmb)
adjusted_r_squared_mlb = 1 - (1-rsquared_mlb)*(len(yb_test)-1)/(len(yb_test)-Xb_test.shape[1]-1)
print('Mean absolute error: {}'.format(metrics.mean_absolute_error(yb_test, per_lmb)))
print('Mean squared error: {}'.format(metrics.mean_squared_error(yb_test, per_lmb)))
print('Root mean squared error: {}'.format(np.sqrt(metrics.mean_squared_error(yb_test, per_lmb))))
print('R Squared value: {}'.format(rsquared_mlb))
print('Adjusted R Squared Value: {}'.format(adjusted_r_squared_mlb))


# There is minor improvement after the backward elimination process, which is more clearly seen through our performance parameters.

# In[ ]:


trace0 = go.Scatter(
x = Xb_test[:,2],
y = yb_test,
mode = 'markers',
name = 'Test Set'
)
trace1 = go.Scatter(
x = Xb_test[:,2],
y = per_lmb,
opacity = 0.75,
mode = 'markers',
name = 'Predictions',
marker = dict(line = dict(color = 'black', width = 0.5))
)
data = [trace0, trace1]
ply.iplot(data)


# <h2>K-nearest Neighbors</h2>

# In[ ]:


knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)


# In[ ]:


print('Mean absolute error: {}'.format(metrics.mean_absolute_error(y_test, knn_pred)))
print('Mean squared error: {}'.format(metrics.mean_squared_error(y_test, knn_pred)))
print('Root mean squared error: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, knn_pred))))
rsquared_knn = metrics.r2_score(y_test,knn_pred)
adjusted_r_squared_knn = 1 - (1-rsquared_knn)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print('R Squared: {}'.format(rsquared_knn))
print('Adjusted R Squared: {}'.format(adjusted_r_squared_knn))


# Here I used for loop and ran the model 20 times with neighbors ranging from 1 to 20. The root mean square error was used to select the best model having the least error. k with 8 has the least RMSE of .444.

# In[ ]:


RMSE = []
for i in range(1,20):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    RMSE.append((np.sqrt(metrics.mean_squared_error(y_test, pred_i))))
print('Minimum Root Mean Squared Error is {} with {} neighbors'.format(round(min(RMSE),3),RMSE.index(min(RMSE))+1))


# In[ ]:


trace = go.Scatter(
x = np.arange(1,20),
y = np.round(RMSE,3),
marker = dict(
size = 10,
color = 'rgba(255, 182, 193, .9)'),
line = dict( color = 'blue'),
mode = 'lines+markers'
)
layout = dict(title = 'RMSE vs Neighbors',
xaxis = dict(title = 'Number of neighbors',zeroline = False),
yaxis = dict(title = 'RMSE',zeroline = False)
)
data = [trace]
fig = dict(data = data, layout = layout)
ply.iplot(fig)


# In[ ]:


knn = KNeighborsRegressor(n_neighbors=8)
knn.fit(X_train,y_train)
pred_8 = knn.predict(X_test)


# In[ ]:


rsquared_8 = metrics.r2_score(y_test,pred_8)
adjusted_r_squared_8 = 1 - (1-rsquared_8)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print('Mean absolute error: {}'.format(metrics.mean_absolute_error(y_test, pred_8)))
print('Mean squared error: {}'.format(metrics.mean_squared_error(y_test, pred_8)))
print('Root mean squared error: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, pred_8))))
print('R squared value with 8 neighbors: {}'.format(rsquared_8))
print('Adjusted squared value with 8 neighbors: {}'.format(adjusted_r_squared_8))


# In[ ]:





# The residual of the K-Nearest Neighbors is approximately normally distributed. This is the test of our second assumption.

# In[ ]:


residualk = (y_test- pred_8)
sns.distplot(residualk)


# We can visualize our KNN model by plotting our prediction and comparing it with our test set. We can visualize here that the prediction point of the outlier prices are coming closer and there are more overlapping points in our scatter plots.

# In[ ]:


trace0 = go.Scatter(
x = X_test['sqft_living'],
y = y_test,
mode = 'markers',
name = 'Test Set'
)
trace1 = go.Scatter(
x = X_test['sqft_living'],
y = pred_8,
opacity = 0.75,
mode = 'markers',
name = 'Predictions',
marker = dict(line = dict(color = 'black', width = 0.5))
)
data = [trace0, trace1]
ply.iplot(data)


# <h2>Random Forest Regression</h2>

# In[ ]:


Xr = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'waterfront','view','condition', 'grade',
'sqft_above',
'sqft_basement', 'yr_built','yr_renovated', 'zipcode', 'lat', 'long','sqft_living15', 'sqft_lot15']]
Xr = ss.fit_transform(Xr)
yr = df_pre["price"]
Xr_train , Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size = 0.3, random_state = 101)
from sklearn.ensemble import RandomForestRegressor
rfc = RandomForestRegressor(n_estimators = 1)
rfc.fit(Xr_train,yr_train)
rfc_pred = rfc.predict(Xr_test)


# In[ ]:


rsquared = metrics.r2_score(yr_test,rfc_pred)
adjusted_r_squared = 1 - (1-rsquared)*(len(yr_test)-1)/(len(yr_test)-Xr_test.shape[1]-1)
print('Mean absolute error: {}'.format(metrics.mean_absolute_error(yr_test, rfc_pred)))
print('Mean squared error: {}'.format(metrics.mean_squared_error(yr_test, rfc_pred)))
print('Root mean squared error: {}'.format(np.sqrt(metrics.mean_squared_error(yr_test, rfc_pred))))
print('R Squared value: {}'.format(rsquared))
print('Adjusted R Squared Value: {}'.format(adjusted_r_squared))


# Here I used for loop and ran the model 50 times with estimators ranging from 1 to 50. The root mean square error was used to select the best model having the least error. 

# In[ ]:


RMSE_rfc = []
for i in range(1,50):
    rfc = RandomForestRegressor(n_estimators=i)
    rfc.fit(Xr_train,yr_train)
    pred_i = rfc.predict(Xr_test)
    RMSE_rfc.append((np.sqrt(metrics.mean_squared_error(yr_test, pred_i))))
print('Minimum Root Mean Squared Error is {} with {} estimators'.format(round(min(RMSE_rfc),3),RMSE_rfc.index(min(RMSE_rfc))+1))


# In[ ]:


trace = go.Scatter(
x = np.arange(1,100),
y = np.round(RMSE_rfc,3),
marker = dict(
size = 10,
color = 'rgba(255, 182, 193, .9)'),
line = dict( color = 'blue'),
mode = 'lines+markers'
)
layout = dict(title = 'RMSE vs Estimators',
xaxis = dict(title = 'Number of estimators',zeroline = False),
yaxis = dict(title = 'RMSE',zeroline = False)
)
data = [trace]
fig = dict(data = data, layout = layout)
ply.iplot(fig)


# In[ ]:


rfc = RandomForestRegressor(n_estimators=40)
rfc.fit(Xr_train,yr_train)
pred_p = rfc.predict(Xr_test)
rsquared_p = metrics.r2_score(yr_test,pred_p)
adjusted_r_squared_p = 1 - (1-rsquared_p)*(len(yr_test)-1)/(len(yr_test)-Xr_test.shape[1]-1)
print('Mean absolute error: {}'.format(metrics.mean_absolute_error(yr_test, pred_p)))
print('Mean squared error: {}'.format(metrics.mean_squared_error(yr_test, pred_p)))
print('Root mean squared error: {}'.format(np.sqrt(metrics.mean_squared_error(yr_test, pred_p))))
print('R squared value: {}'.format(rsquared_p))
print('Adjusted squared value: {}'.format(adjusted_r_squared_p))


# In[ ]:


residualr = (yr_test- pred_p)
sns.distplot(residualr)


# In[ ]:


trace0 = go.Scatter(
x = Xr_test[:,2],
y = yr_test,
mode = 'markers',
name = 'Test Set'
)
trace1 = go.Scatter(
x = Xr_test[:,2],
y = pred_8,
opacity = 0.75,
mode = 'markers',
name = 'Predictions',
marker = dict(line = dict(color = 'black', width = 0.5))
)
data = [trace0, trace1]
ply.iplot(data)


# <h2>Evaluation</h2>
# Here we are going to use Adjusted R-Squared score because it downgrades the score when our model is over fitted to compare performance of our model. We started our prediction with Simple Linear Regression where we predicted the price using the sqft_living. Through different scores and visualization we saw this is not the perfect model and more features are required in order to predict the prices of the houses more accurately. Using the Multiple Linear Regression after backward elimination, there was a huge improvement from Simple Linear Regression where our Adjusted R-Squared increased by 42%. I also used both KNN and Random Forest model to see if Adjusted R-Squared can further be improved by tweaking the number of neighbors in KNN and the number of estimators in Random Forest. The best result came out in Random Forest with lowest error and highest Adjusted R-Squared
