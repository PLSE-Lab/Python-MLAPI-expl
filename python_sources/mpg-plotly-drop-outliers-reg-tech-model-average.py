#!/usr/bin/env python
# coding: utf-8

# # Introduction[](http://)
#  * This kernel contains Prediction of Fuel Efficiency
#  * If you like my kernel, please upvote.
#  
# <hr>
#  * You will learn:
#    * Plotly
#    * Drop Outliers
#    * Regularization Technics
#    * Model Average
#    
# <hr>
# <br>
# <br>
# <font color = 'blue'>
# <b>Content: </b>
# 
# 1. [Prepare Problems](#1)
#     * [Load Libraries](#2)
#     * [Load Dataset](#3)    
# 1. [Descriptive Analysis](#4)
# 1. [Data Visualization](#5)
#     * [Pie Chart](#6)
#     * [Bar Plot](#7)
#     * [Bubble Plot](#8)
#     * [Histogram](#9)
#     * [Box Plot](#10)
#     * [Choropleth](#11)
# 1. [Detection Outliers](#12)
# 1. [Feature Engineering - 1](#13)
# 1. [Feature Engineering - 2](#14)
# 1. [Train and Test Split](#15)
# 1. [Standardization](#16)
# 1. [Regression Models](#17)
#     * [Linear Regression](#18)
# 1. [Regularization Technics](#19)
#     * [Ridge(L2)](#20)
#     * [Lasso(L1)](#21)
#     * [ElasticNet](#22) 
#     * [Compare Technis via Bar Plot](#23) 
# 1. [XGBOOST](#24)
# 1. [Model Averaging by XGB and Linear Regression](#25)
# 1. [Conclusion](#26)

# <a id = "1"></a><br>
# ## Prepare Problems
# This kernel contains Prediction of Fuel Efficiency

# <a id = "2"></a><br>
# ## Load Libraries

# In[ ]:


# Load Libraries:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.offline as pyo 
import plotly.graph_objs as go
import plotly.figure_factory as ff
#
from scipy import stats
from scipy.stats import norm, skew
#
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
#
import xgboost as xgb
#
import warnings
warnings.filterwarnings("ignore")
#
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a id = "3"></a><br>
# ## Load Dataset

# In[ ]:


data = pd.read_csv("/kaggle/input/autompg-dataset/auto-mpg.csv")
data.head()


# * Attribute Information:
# 
# * mpg: continuous
# * cylinders: multi-valued discrete (3-8)
# * displacement: continuous
# * horsepower: continuous
# * weight: continuous
# * acceleration: continuous
# * model year: multi-valued discrete (1970-1982)
# * origin: multi-valued discrete (USA - Japan - Euro)
# * car name: string (unique for each instance)

# <a id = "4"></a><br>
# ## Descriptive Analysis

# ### What did happen?

# ### Shape of Data

# In[ ]:


# row
print("row count:",data.shape[0])
# columns
print("column count:",data.shape[1])


# ### Info of Data
# We explore :
# * Data Types
# * Row Count
# * Column Count
# * Is There Missing Values

# In[ ]:


data.info()


# ### Describe of Data
# We explore :
# * Count
# * Mean
# * Std
# * Quartiles
# * Max

# ### Summary Statistics
# * Mean : Balance Point
# * Median : Middle Value "when ordered"
# * Variance : The average of the squared distance of the mean
# * Standard Deviation : The square root of the variance
# * Skewness : A measure that describes the contrast of one tail versus the other tail. For example, if there are more high values in your distribution than low values then your distribution is 'skewed' towards the high values.
# * Kurtosis : A measure of how 'fat' the tails in the distribution are.

# * mpg.mean = 23.514573
# * mpg.median(50%) = 23.0
# * So we can not say this is a "Normal Distribution"
# 

# In[ ]:


data.describe().T


# ## Missing Value Control

# In[ ]:


# But there are "?", (Missing Attribute Values: horsepower has 6 missing values)
#Just we did not see here
data.isnull().sum()


# ## Covariance 
# * Covariance is a measure of how much two random variables vary together.
# 
# * If two variables are independent, their covariance is 0. However, a covariance of 0 does not imply that the variables are independent.
# * (+) : positive relation
# * (-) : negtive relation

# In[ ]:


data.cov()


# ## Correlation 
# * Correlation is a standardized version of covariance.
# * Between -1 and 1
# * Zero = Not correlated
# * (+) : positive relation
# * (-) : negtive relation

# In[ ]:


data.corr()


# ## Let's see in the plot
# * We see in the plot
#     * clinders and origin can be categorical feature
#     * and also explore about correlations between features

# In[ ]:


sns.pairplot(data,markers="*");
plt.show()


# ## Distribution of Countries
# *  1 : USA
# *  2 : Eurepe
# *  3 : Japan

# In[ ]:


data["origin"].value_counts()


# As Percentage

# In[ ]:


data["origin"].value_counts(normalize=True)


# <a id = "5"></a><br>
# ## Data Visualization

# <a id = "6"></a><br>
# ## Pie Chart
# * Display the percentages of Data Origin

# In[ ]:


colors = ['#f4cb42', '#cd7f32', '#a1a8b5'] #gold,bronze,silver
#
origin_counts = data["origin"].value_counts(sort=True)
labels = ["USA", "Europe","Japan"]
values = origin_counts.values
#
pie = go.Pie(labels=labels, values=values, marker=dict(colors=colors))
layout = go.Layout(title='Origin Distribution')
fig = go.Figure(data=[pie], layout=layout)
pyo.iplot(fig)


# <a id = "7"></a><br>
# ## Bar Plot

# In[ ]:


trace0 = [go.Bar(x=data["model year"]+1900,y=data["mpg"],
                   marker=dict(color="rgb(17,77,117)"),
                   name="Total")]

layout = go.Layout(title="Consumption Gallon by Years",barmode="stack")
fig = go.Figure(data=trace0,layout=layout)   
pyo.iplot(fig) 


# <a id = "8"></a><br>
# ## Bubble Plot
# We will make circles of different sizes depending on the Cylinders

# In[ ]:


trace1 = [go.Scatter(x=data["horsepower"], y=data["weight"],
                   text=data["car name"],
                   mode="markers",
                   marker=dict(size=2*data["cylinders"],
                               color=data["cylinders"],
                               showscale=True))]
 
layout = go.Layout(title="Relation Horse Power & Weight",
                   xaxis=dict(title="Horse Power"),
                   yaxis=dict(title="Weight"),
                   hovermode="closest")
fig = go.Figure(data=trace1,layout=layout)
pyo.iplot(fig)


# <a id = "9"></a><br>
# ## Histogram

# In[ ]:


trace2 = [go.Histogram(x=data.mpg,
                         xbins=dict(start=0,end=50))]
layout = go.Layout(title="MPG")

fig = go.Figure(data=trace2,layout=layout)
pyo.iplot(fig)


# <a id = "10"></a><br>
# ## Box Plot
# We will display statistics
# 
# * A box plot is a graphical method to summarize a data set by visualizing the minimum value, 25th percentile, median, 75th percentile, the maximum value, and potential outliers. A percentile is the value below which a certain percentage of data fall. For example, if 75% of the observations have values lower than 685 in a data set, then 685 is the 75th percentile of the data. At the 50th percentile, or median, 50% of the values are lower and 50% are higher than that value.

# In[ ]:


trace3 = [go.Box(y=data["mpg"],name=data.columns[0]),
          go.Box(y=data["cylinders"],name=data.columns[1]),
          go.Box(y=data["displacement"],name=data.columns[2]),
          go.Box(y=data["horsepower"],name=data.columns[3]),
          go.Box(y=data["weight"],name=data.columns[4]),
          go.Box(y=data["acceleration"],name=data.columns[5]),
          go.Box(y=data["origin"],name=data.columns[7])]

pyo.iplot(trace3)


# <a id = "11"></a><br>
# ## Choropleth
# Let's determine how many origins have 

# In[ ]:


# I choice Germany instead of Euro
country_number = pd.DataFrame(index=["USA","DEU","JPN"],columns=["number","country"])
country_number["country"] = ["United States","Germany","Japan"]
country_number["number"] = [249,79,70]


# In[ ]:


country_number


# In[ ]:


worldmap = [dict(type = 'choropleth', locations = country_number['country'], locationmode = 'country names',
                 z = country_number['number'], autocolorscale = True, reversescale = False, 
                 marker = dict(line = dict(color = 'rgb(180,180,180)', width = 0.5)), 
                 colorbar = dict(autotick = False, title = 'Number of athletes'))]

layout = dict(title = 'Distribution of Data', geo = dict(showframe = False, showcoastlines = True, 
                                                                projection = dict(type = 'Mercator')))

fig = dict(data=worldmap, layout=layout)
pyo.iplot(fig, validate=False)


# <a id = "12"></a><br>
# ## Detection Outliers

# IQR = Q3 - Q1

# ![](https://lumileds.studysixsigma.com/wp-content/uploads/sites/14/2016/03/anat-1.png)

# ### Outlier for Horsepower

# In[ ]:


data["horsepower"] = data["horsepower"].replace("?","100")
data["horsepower"] = data["horsepower"].astype(float)


# In[ ]:


threshoold       = 2
horsepower_desc  = data["horsepower"].describe()
q3_hp            = horsepower_desc[6]
q1_hp            = horsepower_desc[4]
IQR_hp           = q3_hp - q1_hp
top_limit_hp     = q3_hp + threshoold * IQR_hp
bottom_limit_hp  = q1_hp - threshoold * IQR_hp
filter_hp_bottom = bottom_limit_hp < data["horsepower"]
filter_hp_top    = data["horsepower"] < top_limit_hp
filter_hp        = filter_hp_bottom & filter_hp_top


# In[ ]:


data = data[filter_hp]
data.shape


# ### Outlier for Acceleration

# In[ ]:


data.columns


# In[ ]:


acceleration_desc  = data["acceleration"].describe()
q3_acc             = acceleration_desc[6]
q1_acc             = acceleration_desc[4]
IQR_acc            = q3_acc - q1_acc
top_limit_acc      = q3_acc + threshoold * IQR_acc
bottom_limit_acc   = q1_acc - threshoold * IQR_acc
filter_acc_bottom  = bottom_limit_acc < data["acceleration"]
filter_acc_top     = data["acceleration"] < top_limit_acc
filter_acc         = filter_acc_bottom & filter_acc_top


# In[ ]:


data = data[filter_acc]
data.shape


# <a id = "13"></a><br>
# ## Feature Engineering - 1

# ### Skewness
# * In the tails, our points can be outliers, so we should make them gaussian distirbution

# ![](https://i1.wp.com/alevelmaths.co.uk/wp-content/uploads/2019/02/Skewness_1.png?w=761&ssl=1)

# ## Observe for Dependent Feature 

# In[ ]:


f,ax = plt.subplots(figsize = (20,7))
sns.distplot(data.mpg, fit=norm);
# we see that, data["mpg"] has positive skewness


# In[ ]:


(mu,sigma) = norm.fit(data["mpg"])
print("mu:{},sigma:{}".format(mu,sigma))


# ### How can we decide data is gaussian or not?
# *  1-Histogram
# *  2-We can understand Quantile Quatile plot 

# In[ ]:


# qq plot:
plt.figure(figsize = (20,7))
stats.probplot(data["mpg"],plot=plt)
plt.show
print("We expect that our data points will be on red line for gaussian distributin. We see dist tails")


# ## Log Transformations
# * The log transformation can be used to make highly skewed distributions less skewed.
# * http://onlinestatbook.com/2/transformations/log.html

# In[ ]:


data["mpg"] = np.log1p(data["mpg"])


# In[ ]:


f,ax = plt.subplots(figsize = (20,7))
sns.distplot(data["mpg"], fit=norm);


# ## Let's other features of Skewness
# * I will ignore if they are close to -1 or 1

# In[ ]:


# Let's other skewness of features
# if skew > 1  : positive skewness
# if skew > -1 : negative skewness

features = ['cylinders', 'displacement', 'horsepower', 'weight','acceleration','origin']
skew_list = []
for i in range(0,6):
    skew_list.append(skew(data.iloc[:,i]))
# So, features are good at skewness 
skew_list


# <a id = "14"></a><br>
# ## Feature Engineering - 2

# * I will apply One Hot Encoding for data["origin"]  and data["cylinders"]
# 
#    because these features have categorical values.
# 

# First, convert to object

# In[ ]:


data["origin"] = data["origin"].astype(str)
data["cylinders"] = data["cylinders"].astype(str)


# In[ ]:


data.drop(["car name"],axis=1,inplace=True)


# In[ ]:


# One Hot Encoding - 1
data = pd.get_dummies(data,drop_first=True)


# In[ ]:


# I decide to use model year as a fuaure, so i will apply one hot encoding
data["model year"] = data["model year"].astype(str)


# In[ ]:


# One Hot Encoding - 2
data = pd.get_dummies(data,drop_first=True)


# In[ ]:


data.info()


# <a id = "15"></a><br>
# ## Train and Test Split

# In[ ]:


y = data["mpg"]
x = data.drop(["mpg"],axis=1)


# In[ ]:


# Creating Train and Test Datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.90, random_state=42)


# <a id = "16"></a><br>
# ## Standardization

# In[ ]:


# Scale the data to be between -1 and 1
# Mean= 0
# Std = 1
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# <a id = "17"></a><br>
# ## Regression Models

# <a id = "18"></a><br>
# ## Linear Regression

# In[ ]:


lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print("LR Coef:",lr.intercept_)
print("LR Coef:",lr.coef_)
mse = mean_squared_error(y_test,y_pred)
print("MSE",mse)


# <a id = "19"></a><br>
# ## Regularization Technics
# * This technique discourages learning a more complex or flexible model, so as to avoid the risk of overfitting.
# * Minimize Sum Of Squared Errors

# <a id = "20"></a><br>
# ## 1- Ridge(L2)

# In[ ]:


ridge = Ridge(random_state=42, max_iter=10000)
alphas = np.logspace(-4,-0.5,30)
tuned_parameters = dict(alpha=alphas)


# In[ ]:


clf = GridSearchCV(ridge,tuned_parameters,cv=5,scoring="neg_mean_squared_error", refit=True)
clf.fit(X_train,y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]


# In[ ]:


print("Ridge Coef:",clf.best_estimator_.coef_)
ridge = clf.best_estimator_
print("Ridge Best Estimator:",ridge)


# In[ ]:


y_pred_ridge = clf.predict(X_test)
mse_ridge = mean_squared_error(y_test,y_pred_ridge)
print("Ridge MSE:",mse_ridge)


# <a id = "21"></a><br>
# ## 2- Lasso(L1)
# * This technic give zero for redundant features
# * Use this technic for feature selection

# In[ ]:


lasso = Lasso(random_state=42, max_iter=10000)
alphas = np.logspace(-4,-0.5,30)
tuned_parameters = dict(alpha=alphas)


# In[ ]:


clf = GridSearchCV(lasso,tuned_parameters,cv=5,scoring="neg_mean_squared_error", refit=True)
clf.fit(X_train,y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]


# In[ ]:


print("Lasso Coef:",clf.best_estimator_.coef_)
lasso = clf.best_estimator_
print("Lasso Best Estimator:",lasso)
print("Put Zero for redundat features:")


# In[ ]:


y_pred_lasso = clf.predict(X_test)
mse_lasso = mean_squared_error(y_test,y_pred_lasso)
print("Lasso MSE:",mse_lasso)


# <a id = "22"></a><br>
# ## ElasticNet
# * This technic takes positive ways of L1 and L2

# In[ ]:


parameters = dict(alpha=alphas,l1_ratio=np.arange(0.0,1,0.05))
eNet = ElasticNet(random_state=42, max_iter=10000)
clf = GridSearchCV(eNet,tuned_parameters,cv=5,scoring="neg_mean_squared_error", refit=True)
clf.fit(X_train,y_train)


# In[ ]:


print("Lasso Coef:",clf.best_estimator_.coef_)
eNet = clf.best_estimator_
print("Lasso Best Estimator:",eNet)


# In[ ]:


y_pred_eNet = clf.predict(X_test)
mse_eNet = mean_squared_error(y_test,y_pred_eNet)
print("Lasso MSE:",mse_eNet)


# <a id = "23"></a><br>
# ## Compare Technics via Bar Plot
# * As we see eNet gives the best result

# In[ ]:


technices = ["Ridge","Lasso","ElasticNet"]
results   = [0.01822,0.01844,0.01813]


# In[ ]:


fig = go.Figure(data=[go.Bar(
            x=technices, y=results,
            text=results,
            textposition='auto',)])
fig.show()


# <a id = "24"></a><br>
# ## XGBOOST

# In[ ]:


# objective: aim
# n_estimator: number of trees
model_xgb = xgb.XGBRegressor(objective="reg:linear",max_depth=5,min_child_weight=4,subsample=0.7,n_estimator=1000,learning_rate=0.07)
model_xgb.fit(X_train,y_train)


# In[ ]:


y_pred_xgb = model_xgb.predict(X_test)
mse_xgb = mean_squared_error(y_test,y_pred_xgb)
print("XGBOOST MSE:",mse_xgb)


# <a id = "25"></a><br>
# ## Model Averaging by XGB and Linear Regression

# In[ ]:


class AveragingModels():
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    # the reason of clone is avoiding affect the original base models
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]  
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)

averaged_models = AveragingModels(models = (model_xgb,lr))
averaged_models.fit(X_train,y_train)
y_pred_averaged_models = averaged_models.predict(X_test)
mse_averaged_models = mean_squared_error(y_test,y_pred_averaged_models)
print("Averaging Models MSE:",mse_averaged_models)


# <a id = "26"></a><br>
# ## Conclusion
# * Average Models gave the best score for our dataset.

# In[ ]:




