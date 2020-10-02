#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

train_df=pd.read_csv('../input/train.csv')


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# As this is a highly skewed data, we will try to transform this data using either log, square-root or box-cox  transformation.

# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import boxcox, inv_boxcox

sns.distplot(train_df['count'])
plt.show()
#train_df['count']=train_df['count'].apply(lambda x:np.sqrt(x))
#train_df['count']=train_df['count'].apply(lambda x:np.sqrt(x))


# In[ ]:


#train_df['count']=train_df['count'].apply(lambda x:np.sqrt(x))


# Log and Square-root transformation do not bring up the desired distributions.
# Using Box_cox transformation for bringing the data to normality. (lambda=0.69). We will be 
# going with square root transformation due to the lambda value.

# In[ ]:


from scipy.stats import boxcox
train_df['count']=train_df['count'].apply(lambda x:np.log(x))
#train_df['count']=boxcox(train_df['count'])[0]
sns.distplot(train_df['count'])
plt.show()
print (train_df['count'])


# Univariate analysis of all variables
# Categorical data--> Season, Holiday, WorkingDay, Weather

# In[ ]:






cat_names=['season', 'holiday', 'workingday', 'weather']

i=0
for name in cat_names:
    i=i+1
    plt.subplot(2,2,i)
    sns.countplot(name,data=train_df) 
    
plt.show()


# Univariate analysis for continuous data

# In[ ]:



cont_names=['temp','atemp','humidity','windspeed']

        
#sns.boxplot(train_df['season'])   
i=0
for name in cont_names:
    i=i+1
    plt.subplot(2,2,i)
    sns.boxplot(name,data=train_df) 
    
plt.show()


# Some of the inferences that can be made:
# * Holiday and working day look  somewhat correlated. Can one of them be removed to avoid multi-collinearity?
# * Let's wait until we calculate thier correlation value
# * Not much can be inferred from season data. Majority of the data fall under 1 and 2, which is clear skies mist/cloudy.
# * Temp, Atemp, humidity look normally distributed. However, windspeed has a lot of outliers which will be analysed further.
# * doing a brief time-series analysis to see if there's any improvement in count over a period of time
# * moving average to be calculated for a period of 3/4 months as that is the no of months in one season

# In[ ]:




from datetime import datetime

train_df['datetime']=train_df['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
time_series_df=train_df
time_series_df.index=train_df['datetime']

import matplotlib.pyplot as plt

#Applying rolling average on a period of 60 days, as the typical weather lasts for around 3 months (20 days in training data of each month)
plt.plot(pd.rolling_mean(time_series_df['count'],60))
plt.show()


# As expected the total count grows over a period of time, therefore this dataset needs to incorporate changes in seasonality too.

# Calculating bivariate analysis on continuous data

# In[ ]:



i=1
for name_1 in cont_names:
    j=cont_names.index(name_1)


    while(j<len(cont_names)-1):


        plt.subplot(6,1,i)
        plt.title(name_1+' vs '+cont_names[j+1])
        sns.jointplot(x=name_1,y=cont_names[j+1],data=train_df) 
        j=j+1
        i=i+1
        plt.show()
            
    


# Not much can be inferred about the distribution of these variables except for variable 'temp' and 'atemp' that almost have
# similar context. We would be using the 'temp' and getting rid of the 'atemp' variables for better precision value and avoiding 
# multi-collinearity.
# 
# 

# Let us perfrom some feature engineering. The datetime column can be used to extract data like the month, day, hour which can be
# used in our model for making better predictions.

# In[ ]:




from datetime import datetime

#converting string dattime to datetime


#train_df['datetime']=train_df['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

new_df=train_df

new_df['month']=new_df['datetime'].apply(lambda x:x.month)
new_df['hour']=new_df['datetime'].apply(lambda x:x.hour)
new_df['day']=new_df['datetime'].apply(lambda x:x.day)
new_df['year']=new_df['datetime'].apply(lambda x:x.year)
#new_df['weekday_flag']=new_df['datetime'].apply(weekday_flag)
new_df['mnth+day']=new_df['datetime'].apply(lambda x:str(x.month)+'_'+str(x.day))


# In[ ]:


sns.swarmplot(x='hour',y='temp',data=new_df,hue='season')
plt.show()


# A non-linear relationship between temperature and day of the hour according to different seasons is evident from this chart.
# 

# In[ ]:


new_df.cov()
sns.heatmap(new_df.corr())
plt.show()


# A lot of inferences that we have already covered could be verified using the following heatmap

# In[ ]:


new_df.corr()


# A lot of inferences that we have already hypothesised could be verified using the following heatmap and correlation matrix.
# 
# Visualizing multi-variate distribution of target variable with other categorical data.

# In[ ]:



cat_names=['season', 'holiday', 'workingday', 'weather']
i=1
for name in cat_names:
    plt.subplot(2,2,i)
    sns.barplot(x=name,y='count',data=new_df,estimator=sum)
    i=i+1
    plt.show()


# *  With weather 1,2 and season 2,3 and working days the bicycle rental count is maximum.
# *  As per the analysis, we need to get rid off these variables to be inputted in our model:datetime,season,holiday,atemp,holiday(Working day) has better correlation with count,  weather,working day, hour,year has to be label encoded

# In[ ]:





final_df=new_df.drop(['datetime','season','holiday','atemp','holiday','windspeed','casual','registered','mnth+day','day'], axis=1)
final_df.head()


# Adding dummy varibles to categorical data

# In[ ]:



weather_df=pd.get_dummies(new_df['weather'],prefix='w',drop_first=True)
year_df=pd.get_dummies(new_df['year'],prefix='y',drop_first=True)
month_df=pd.get_dummies(new_df['month'],prefix='m',drop_first=True)
hour_df=pd.get_dummies(new_df['hour'],prefix='h',drop_first=True)
                     


final_df=final_df.join(weather_df)
final_df=final_df.join(year_df)
final_df=final_df.join(month_df)                     
final_df=final_df.join(hour_df)
                     
final_df.head()


# In[ ]:


final_df.columns


# Now that we have got our guns lock and loaded, it's time to shoot.
# lets begin the modelling process.
# 

# In[ ]:




X=final_df.iloc[:,final_df.columns!='count'].values
print (X)

Y=final_df.iloc[:,4].values

print (Y)


# **Choosing the appropriate model for regression**
# After trying multiple linear regression, poly linear regression, SVR, Decision Tree regression and RF regression,XGRegressor
# Out of these, we would be choosing the one having the best accuracy and aplying GridSearchCV for optimal hyperparmater tuning. XGBoost gives the maximum accuracy of R2 square (92.5%)

# In[ ]:



#from sklearn.ensemble import RandomForestRegressor
import xgboost as xg
from sklearn.model_selection import GridSearchCV

def grid_search():
    print ('lets go')

    xgr=xg.XGBRegressor(max_depth=8,min_child_weight=6,gamma=0.4)
    xgr.fit(X,Y)

    #rf=RandomForestRegressor(n_estimators=100,random_state=0)
    #rf.fit(X,Y)

    
    #parameters=[{'max_depth':[8,9,10,11,12],'min_child_weight':[4,5,6,7,8]}]
    #parameters=[{'gamma':[i/10.0 for i in range(0,5)]}]
    parameters=[{'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]}]

    grid_search= GridSearchCV(estimator=xgr, param_grid=parameters, cv=10,n_jobs=-1)


    print (1)
    grid_search=grid_search.fit(X,Y)
    print (2)
    best_accuracy=grid_search.best_score_
    best_parameters=grid_search.best_params_
    print (best_accuracy)
    print (best_parameters)



#if __name__ == '__main__':
   #grid_search()



# Grid search gives best accuracy for max_depth=8,min_child_weight=6,gamma=0.4,colsample_bytree=0.6,subsample=0.6
# Training the model again with these new parameters.

# In[ ]:



"""
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=300,max_features='auto',random_state=0)
rf.fit(X,Y)
"""

import xgboost as xg
xgr=xg.XGBRegressor(max_depth=8,min_child_weight=6,gamma=0.4,colsample_bytree=0.6,subsample=0.6)
xgr.fit(X,Y)


# Using the same pre-processing functions on the test data:

# In[ ]:




new_df=pd.read_csv('../input/test.csv')
new_df['datetime']=new_df['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))


new_df['month']=new_df['datetime'].apply(lambda x:x.month)
new_df['hour']=new_df['datetime'].apply(lambda x:x.hour)
new_df['day']=new_df['datetime'].apply(lambda x:x.day)
new_df['year']=new_df['datetime'].apply(lambda x:x.year)
#new_df['weekday_flag']=new_df['datetime'].apply(weekday_flag)
#new_df['mnth+day']=new_df['datetime'].apply(lambda x:str(x.month)+'_'+str(x.day))

print (new_df.head())


# In[ ]:



new_df=new_df.drop(['datetime','season','holiday','atemp','holiday','windspeed','day'], axis=1)
new_df.head()


# In[ ]:


#adding dummy varibles to categorical variables
weather_df=pd.get_dummies(new_df['weather'],prefix='w',drop_first=True)
yr_df=pd.get_dummies(new_df['year'],prefix='y',drop_first=True)
month_df=pd.get_dummies(new_df['month'],prefix='m',drop_first=True)
hour_df=pd.get_dummies(new_df['hour'],prefix='h',drop_first=True)


new_df=new_df.join(weather_df)
new_df=new_df.join(yr_df)
new_df=new_df.join(month_df)                     
new_df=new_df.join(hour_df)
                     
new_df.head()


# In[ ]:


X_test=new_df.iloc[:,:].values
X_test.shape


# Using the XGBoost Regressor for predictions:

# In[ ]:


#def invboxcox(y):
#    return(np.exp(np.log(0.69*y+1)/0.69))


# In[ ]:


y_output=xgr.predict(X_test)
y_output




op=pd.DataFrame({'count':np.exp(y_output)})
op.to_csv('sub1.csv')


# In[ ]:


print (np.exp(y_output))

