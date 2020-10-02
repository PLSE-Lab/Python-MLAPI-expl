#!/usr/bin/env python
# coding: utf-8

# # Predicting Train Ticket Prices

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(9,6)})


# ## Load the data

# In[ ]:


#Load data
data = pd.read_csv("../input/renfe.csv")
#Lets drop missing values for EDA
data = data.dropna(axis=0,how='any')


# For this simple analysis I just dropped any data points that were missing values.

# The code aggregates the origin and destination into one variable called route.

# In[ ]:


data['duration'] = (pd.to_datetime(data.end_date)-pd.to_datetime(data.start_date)).apply(lambda x: (x.seconds)/60)
#Some fares are too small and redundent combine them to for other
data.fare.replace(['Individual-Flexible','Mesa','Grupos Ida'],'Other',inplace=True)
data['route'] = data.origin+data.destination
# One hot encode routes and add to data frame
route_names = {}
i = 1
for route in data.route.unique().tolist():
    route_names["route_"+route] = "route"+str(i)
    i = i+1


# Lets split the start date to month, day and dayname. I did not split the end date because the duration of the trips was less than 1 day.

# In[ ]:


#add month
data['month'] = pd.to_datetime(data.start_date).apply(lambda x: x.month)
data['day'] = pd.to_datetime(data.start_date).apply(lambda x: x.day)
data['dayname'] = pd.to_datetime(data.start_date).apply(lambda x: x.day_name())


# In[ ]:


data = data.rename(index=str,columns=route_names)
data.head()


# In[ ]:


# Distribution of prices and train duration
fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(15,5))
_ = sns.distplot(data.price,ax=axs[0])
_ = sns.distplot(data.duration,ax=axs[1])


# The figures above show the distribution of price for a ticket and the duration for the trip.

# In[ ]:


# Lets check the count fo the categorical variables
fig,axs = plt.subplots(ncols=2,nrows=2,figsize=(15,15))
_ = sns.countplot(x=data.route,ax=axs[0,0]).set_xticklabels(rotation=90,labels=data.route.unique())
_ = sns.countplot(x=data.train_class,ax=axs[0,1]).set_xticklabels(rotation=90,labels=data.train_class.unique())
_ = sns.countplot(x=data.train_type,ax=axs[1,0]).set_xticklabels(rotation=90,labels=data.train_type.unique())
_ = sns.countplot(x=data.fare,ax=axs[1,1]).set_xticklabels(rotation=90,labels=data.fare.unique())
fig.subplots_adjust(hspace=.9)


# The categorical variables are unevenly distributed.

# In[ ]:


# Lets check how they might effect the price
fig,axs = plt.subplots(ncols=2,nrows=4,figsize=(15,15))
_ = sns.violinplot(x=data.route,y=data.price,ax=axs[0,0]).set_xticklabels(rotation=90,labels=data.route.unique())
_ = sns.violinplot(x=data.train_class,y=data.price,ax=axs[0,1]).set_xticklabels(rotation=90,labels=data.train_class.unique())
_ = sns.violinplot(x=data.train_type,y=data.price,ax=axs[1,0]).set_xticklabels(rotation=90,labels=data.train_type.unique())
_ = sns.violinplot(x=data.fare,y=data.price,ax=axs[1,1]).set_xticklabels(rotation=90,labels=data.fare.unique())
_ = sns.violinplot(x=data.month,y=data.price,ax=axs[2,0]).set_xticklabels(rotation=90,labels=data.month.unique())
_ = sns.violinplot(x=data.dayname,y=data.price,ax=axs[2,1]).set_xticklabels(rotation=90,labels=data.dayname.unique())
_ = sns.violinplot(x=data.day,y=data.price,ax=axs[3,0]).set_xticklabels(rotation=90,labels=data.day.unique())
fig.subplots_adjust(hspace=1.5)
fig.delaxes(axs[3,1])


# From the above figue, we can see that there are clear distinction between the classes for the average price of the train ticket.
# The distinction is not as obvious for the day plot. Months and dayname have different distributions as well. This suggesting that different months and days will have an impact on the price.

# In[ ]:


# Lets check the data ranges
fig,axs = plt.subplots(ncols=2,nrows=4,figsize=(15,15))
_ = sns.boxplot(x=data.route,y=data.price,ax=axs[0,0]).set_xticklabels(rotation=90,labels=data.route.unique())
_ = sns.boxplot(x=data.train_class,y=data.price,ax=axs[0,1]).set_xticklabels(rotation=90,labels=data.train_class.unique())
_ = sns.boxplot(x=data.train_type,y=data.price,ax=axs[1,0]).set_xticklabels(rotation=90,labels=data.train_type.unique())
_ = sns.boxplot(x=data.fare,y=data.price,ax=axs[1,1]).set_xticklabels(rotation=90,labels=data.fare.unique())
_ = sns.boxplot(x=data.month,y=data.price,ax=axs[2,0]).set_xticklabels(rotation=90,labels=data.month.unique())
_ = sns.boxplot(x=data.dayname,y=data.price,ax=axs[2,1]).set_xticklabels(rotation=90,labels=data.dayname.unique())
_ = sns.boxplot(x=data.day,y=data.price,ax=axs[3,0]).set_xticklabels(rotation=90,labels=data.day.unique())
fig.subplots_adjust(hspace=1.5)
fig.delaxes(axs[3,1])


# Box plots reveal similar story about the distribution of prices for each category.

# ## Lets check how our dates effects the price more closely.

# In[ ]:


fig,axs = plt.subplots(ncols=2,nrows=2,figsize=(15,15))
_ = sns.lineplot(x='dayname',y='price',data=data,ax=axs[0,0])
_ = sns.lineplot(x='month',y='price',data=data,ax=axs[0,1])
_ = sns.lineplot(x='day',y='price',data=data,ax=axs[1,0])
fig.delaxes(axs[1,1])


# This information suggests that there are differences in pricing given the dayname and price. We should treat these variables as categorical variables. On the other hand, day of the month will be treated as ordinal for model simplicity. It's interesting to note that the cheapest tickets are on the 15th of each month,saturday, and in july.

# ##  Lets perform basic linear regression to check which variables are statistically significant predictors
# 

# In[ ]:


data_i = data[['route','month','dayname','train_class','train_type','fare','duration','day','price']]
data_i.price = (data_i.price - data_i.price.mean())/data_i.price.std()
data_i.duration = (data_i.duration - data_i.duration.mean())/data_i.duration.std()


# In[ ]:


data_i.head()


# In[ ]:


formula = "price~C(route)+C(month)+C(dayname)+C(train_class)+C(train_type)+C(fare)+duration+day"
#Due to multico had to drop train_type and day
formula = "price~C(route)+C(month)+C(dayname)+C(train_class)+C(fare)+duration"

# Unable to run due to issues with statsmodel
# import statsmodels.formula.api as smf
# reg = smf.ols(formula = formula,data=data_i).fit()


# In[ ]:


#reg.summary()


#        "                            OLS Regression Results                            ",
#        "==============================================================================",
#        "Dep. Variable:                  price   R-squared:                       0.816",
#        "Model:                            OLS   Adj. R-squared:                  0.816",
#        "Method:                 Least Squares   F-statistic:                 3.858e+05",
#        "Date:                Sat, 22 Jun 2019   Prob (F-statistic):               0.00",
#        "Time:                        12:57:47   Log-Likelihood:            -1.3020e+06",
#        "No. Observations:             2269090   AIC:                         2.604e+06",
#        "Df Residuals:                 2269063   BIC:                         2.604e+06",
#        "Df Model:                          26                                         ",
#        "Covariance Type:            nonrobust                                         ",
#        "================================================================================",
#        "                                           coef    std err          t      P>|t|",
#        "--------------------------------------------------------------------------------",
#        "Intercept                                2.4169      0.057     42.167      0.000",
#        "C(route)[T.MADRIDBARCELONA]              0.0096      0.001     10.182      0.000",
#        "C(route)[T.MADRIDPONFERRADA]            -1.7084      0.002   -806.150      0.000",
#        "C(route)[T.MADRIDSEVILLA]               -1.2110      0.001  -1139.434      0.000",
#        "C(route)[T.MADRIDVALENCIA]              -1.6855      0.001  -1516.575      0.000",
#        "C(route)[T.PONFERRADAMADRID]            -1.7285      0.002   -834.537      0.000",
#        "C(route)[T.SEVILLAMADRID]               -1.2063      0.001  -1126.973      0.000",
#        "C(route)[T.VALENCIAMADRID]              -1.6672      0.001  -1495.629      0.000",
#        "C(month)[T.5]                           -0.1060      0.001   -135.860      0.000",
#        "C(month)[T.6]                           -0.1260      0.001   -101.805      0.000",
#        "C(month)[T.7]                           -0.1459      0.004    -40.722      0.000",
#        "C(dayname)[T.Monday]                    -0.1020      0.001    -97.370      0.000",
#        "C(dayname)[T.Saturday]                  -0.1679      0.001   -145.315      0.000",
#        "C(dayname)[T.Sunday]                     0.0258      0.001     23.941      0.000",
#        "C(dayname)[T.Thursday]                  -0.0904      0.001    -87.970      0.000",
#        "C(dayname)[T.Tuesday]                   -0.1406      0.001   -134.688      0.000",
#        "C(dayname)[T.Wednesday]                 -0.1104      0.001   -105.328      0.000",
#        "C(train_class)[T.Cama Turista]          -0.1204      0.059     -2.048      0.041",
#        "C(train_class)[T.Preferente]            -0.3115      0.057     -5.444      0.000",
#        "C(train_class)[T.Turista]               -1.2094      0.057    -21.141      0.000",
#        "C(train_class)[T.Turista Plus]          -0.8300      0.057    -14.509      0.000",
#        "C(train_class)[T.Turista con enlace]    -0.8561      0.057    -14.970      0.000",
#        "C(fare)[T.Flexible]                      0.6005      0.003    233.555      0.000",
#        "C(fare)[T.Other]                         3.5440      0.047     75.048      0.000",
#        "C(fare)[T.Promo]                        -0.5216      0.003   -202.514      0.000",
#        "C(fare)[T.Promo +]                      -0.2454      0.003    -74.600      0.000",
#        "duration                                -0.3481      0.001   -496.996      0.000",
#        "==============================================================================",
#        "Omnibus:                   186159.428   Durbin-Watson:                   1.568",
#        "Prob(Omnibus):                  0.000   Jarque-Bera (JB):           973915.556",
#        "Skew:                           0.217   Prob(JB):                         0.00",
#        "Kurtosis:                       6.180   Cond. No.                         887.",
#        "==============================================================================",
#        
#        "Warnings:",
#        "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.",

# In[ ]:


#_ = sns.distplot(reg.resid)


# We can see from the summary above that all of these variables are statistically significant in predicting the price of the ticket. The best forumla is price~C(route)+C(month)+C(dayname)+C(train_class)+C(fare)+duration. I dropped some variables as there was strong multicolinearity. Still, the model has a relatively high R-squared value. Now lets run random forest regression in scikit to improve the regression.

# In[ ]:


skdata  = data[['route','month','dayname','train_class','fare','price','duration']]
skdata.duration = (skdata.duration - skdata.duration.mean())/skdata.duration.std()
skdata = pd.get_dummies(skdata,columns=['route','month','dayname','train_class','fare'])


# In[ ]:


skdata.head()


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


linreg = LinearRegression()
rfreg = RandomForestRegressor(n_estimators=10)
nn = MLPRegressor(hidden_layer_sizes=(100,),
                 activation='relu',
                 solver='sgd',
                 learning_rate='adaptive',
                 learning_rate_init=.001,
                 verbose=True)
y = skdata.price
x = skdata[skdata.keys()[1:]]

print("Linear Regression R-Squared: ",cross_val_score(linreg,cv=10,X=x,y=y))
print("RF Regression R-Squared: ",cross_val_score(rfreg,cv=10,X=x,y=y))


# Unsurprisingly the random forest regression performs better than linear regression. Let's split the data into sets for validation and testing. 

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

y = skdata.price
x = skdata[skdata.keys()[1:]]

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=.3)

rf = RandomForestRegressor(n_estimators=10,
                           n_jobs=4)
cv  = KFold(n_splits=10,shuffle=True)

train_r = []
test_r = []
test_mse = []
i = 0
for train_idx,test_idx in cv.split(X=X_train,y=y_train):
        i+=1
        print("CV: ",i)
        
        
        # Random forest regression
        rf.fit(X=X_train.iloc[train_idx,:],y=y_train[train_idx])
        train_r.append(rf.score(X_train.iloc[train_idx,:],y=y_train[train_idx]))
        preds = rf.predict(X=X_train.iloc[test_idx,:])
        test_r.append(r2_score(y_train.iloc[test_idx],preds))
        test_mse.append(mean_squared_error(y_train.iloc[test_idx],preds))


# Lets test the predictive ability of the model by predicting the prices for the trainning set.

# In[ ]:


test_p = rf.predict(X_test)
restest = y_test - test_p
px = np.arange(0,len(test_p))

fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(15,5))
ax = sns.scatterplot(px[:100],test_p[:100],label="Predicted",ax=axs[0])
ax = sns.scatterplot(px[:100],y_test[:100],label="Truth",ax=axs[0])
ax = sns.scatterplot(px[:100],restest[:100],label="Res",ax=axs[0])
ax.set(title="Comparing Some Predicted, Truth and the Residuals")
ax2 = sns.distplot(restest,label="Residual Distribution")
_ = ax2.set(title="Residual Distribtion")


# We can see from the plots above, that the random forest regression model has good predictive ability for the testing data set. Further more the residuals appear to be normal which suggests that the model was able to capture most of signal in the data. This is a good model.

# In[ ]:


from scipy.stats import normaltest
from sklearn.metrics import explained_variance_score
print("PREDICTED MEAN SQUARED ERROR: ",mean_squared_error(y_test,test_p))
print("R2: ",r2_score(y_test,test_p))


# # Conclusion
# We were able to fit the price of the train tickets with an R2 of 0.91. This is a very good fit. The only other model I tested was vanilla linear regression which had a lower R2 at around 0.81. 
