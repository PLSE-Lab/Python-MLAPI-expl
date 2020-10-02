#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# In[ ]:


#read in data and display first 5 rows
df=pd.read_csv('../input/train.csv')
df.head()


# In[ ]:


#print rows/cols in dataset
df.shape


# In[ ]:


#print info on column types
df.info()


# In[ ]:


#check for missing data, and output columns that have missing data
for col in df:
    if(df[col].isnull().any()):
        print(col)


# In[ ]:


#fills missing data with 0s
#GO BACK TO THIS, 0 may not be best fill for all missing data
df=df.fillna(0)


# In[ ]:


#check to see that missing data has been filled
df['host_response_rate'].head()


# In[ ]:


#summary stats on each of the numeric columns
df.describe()


# In[ ]:


#summary of categoricals
df.describe(include=['O'])


# In[ ]:


df[['property_type', 'log_price']].groupby(['property_type'], as_index=False).mean().sort_values(by='log_price',ascending=False).head(n=10)


# In[ ]:


df[['city', 'log_price']].groupby(['city'], as_index=False).mean().sort_values(by='log_price',ascending=False)


# In[ ]:


g=sns.regplot(x=df['review_scores_rating'],y=df['log_price'],fit_reg=False)


# In[ ]:


print (np.corrcoef(df['bedrooms'], df['accommodates']))


# In[ ]:


df[['accommodates', 'log_price']].groupby(['accommodates'], as_index=False).mean().sort_values(by='log_price',ascending=False)


# In[ ]:


def lat_center(row):
    if (row['city']=='NYC'):
        return 40.72
    
def long_center(row):
    if (row['city']=='NYC'):
        return -74

df['lat_center']=df.apply(lambda row: lat_center(row), axis=1)
df['long_center']=df.apply(lambda row: long_center(row), axis=1)


# In[ ]:


df['distance to center']=np.sqrt((df['lat_center']-df['latitude'])**2+(df['long_center']-df['longitude'])**2)


# In[ ]:


###2.18 find relationship btwn neighborhood and price###
#Filter only for NYC data
#Seems like soho is most expensive
#to test, we find coordinates of Soho, then find distance to Soho for each property
#and see if this correlates with price
pd.options.mode.chained_assignment = None 
ny=df[df['city']=='NYC']
ny.head(5)

#coordinates of soho
lat_ny=40.72
long_ny=-74
ny['distance to center']=np.sqrt((lat_ny-ny['latitude'])**2+(long_ny-ny['longitude'])**2)


# In[ ]:


soho_vs_price=sns.regplot(x=ny['distance to center'],y=ny['log_price'],fit_reg=True)
print (np.corrcoef(ny['distance to center'], ny['log_price']))

#Conclusion: not super strong relationship. -.37 correlation btwn distance to Soho and price


# In[ ]:


reviews_vs_price=sns.regplot(x=ny[ny['review_scores_rating']>0]['review_scores_rating'],y=ny[ny['review_scores_rating']>0]['log_price'], fit_reg=True)


# In[ ]:


print (np.corrcoef(ny['review_scores_rating'], ny['log_price']))

#no strong relationship btwn review scores and log price


# In[ ]:


ny[['room_type', 'log_price']].groupby(['room_type'], as_index=False).mean().sort_values(by='log_price',ascending=False)

#obvious relationship confirmed: entire apt is more expensive than private than share


# In[ ]:


print (np.corrcoef(ny['beds'], ny['accommodates']))

#high correlation btwn beds and accommodates. don't need both


# In[ ]:


#ny_model=ny[['log_price','distance to center', 'review_scores_rating','room_type','accommodates']]
#ny_model['room_type']=ny_model['room_type'].map({'Shared room':1,'Private room':2,'Entire home/apt':3}).astype(int)

#use one-hot-encoding with distance to center
categorical=['property_type','room_type','bed_type','cancellation_policy']
ny_model=pd.get_dummies(ny, columns=categorical)
ny_model.head(5)
ny_model.info()


# In[ ]:


#modeling

# Select only numeric data and impute missing values as 0
numerics = ['uint8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
ny_train_x=ny_model.select_dtypes(include=numerics).drop('log_price',axis=1).fillna(0).values
ny_train_y=ny_model['log_price'].values

#random forest
#RMSE is around .39, BETTER THAN algo w/o distance to center
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
cv_groups = KFold(n_splits=3)
regr = RandomForestRegressor(random_state = 0, n_estimators = 10)

for train_index, test_index in cv_groups.split(ny_train_x):
    
    # Train the model using the training sets
    regr.fit(ny_train_x[train_index], ny_train_y[train_index])
    
    # Make predictions using the testing set
    pred_rf = regr.predict(ny_train_x[test_index])
    
    # Calculate RMSE for current cross-validation split
    rmse = str(np.sqrt(np.mean((ny_train_y[test_index] - pred_rf)**2)))
    
    print("RMSE for current split: " + rmse)


# In[ ]:


from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, ny_train_x, ny_train_y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")


# In[ ]:


model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(ny_train_x, ny_train_y)
rmse_cv(model_lasso).mean()

