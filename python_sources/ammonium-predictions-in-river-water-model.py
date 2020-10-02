#!/usr/bin/env python
# coding: utf-8

# [](http://)Ammonium is a nitrogen containing compound. Although ammonium can be found in nature and is a natural byproduct of biological processes, its use in manufacturing processes and as a component of fertilizers can lead to an unnatural increase in environmental levels. High ammonium levels can be toxic to fish, and promote algal blooms, which can harbor toxic cyanobacteria and affect drinking water quality. The content of the dataset states that the maximum permissible value of Ammonium ions concentration (NH4) in Ukraine is 0.5 mg/cub. dm. The goal of the current project is to create a model that can predict ammonium concentrations at a station based on upstream concentrations. First, we will download the necessary packages, and perform some exploratory data analysis (EDA).

# In[ ]:


import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


ammonium_train=pd.read_csv('/kaggle/input/ammonium-prediction-in-river-water/train.csv')
ammonium_train.head()


# As per the description of the dataset, it appears as though there are a lot of missing values in columns (stations) 3-7. Dependent variables that have no associated independent variables are not useful for linear regression. We will deal with these missing values shortly.

# In[ ]:


ammonium_train.info()


# You can see exactly how may values are present in each column by looking at the "non-null count" column of the info function. You can see that stations 3-7 have significantly less non-null values than the target, 1 and 2 stations.

# In[ ]:


ammonium_train.describe()


# Stations 4 and 5 have much higher maximum values than the rest of the stations. The mean and max values for  the target station, station 1 and station 2 appear to increase as you move toward stations 4 and 5. This shows that the source of the ammonium is likely close to stations 4 and 5. This could mean that those stations are close to an aricultural or manufacturing area, where ammonium is used.

# Since stations 3-7 have 25% or less non-null values, we will only work with stations 1 and 2 to get a more accurate prediction. I will drop columns 3-7.

# In[ ]:


ammonium_train.drop(ammonium_train[['3','4','5','6','7']], axis=1, inplace=True)
ammonium_train.head()


# Next, we will determine which rows have incomplete data from our target, 1 and 2 stations, and drop those rows.

# In[ ]:


ammonium_train.count()


# In[ ]:


ammonium_train.dropna(inplace=True)
ammonium_train.count()


# Next, we will visualize the data. To get an idea of what the ammonium levels look like over time at each station, I created a line plot of ammonium levels over time.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(ammonium_train.index, ammonium_train['2'], color='blue', label='station 2')
plt.plot(ammonium_train.index, ammonium_train['1'], color='magenta', label='station 1')
plt.plot(ammonium_train.index, ammonium_train['target'], color='orange', label='target station')
plt.legend()

plt.ylabel('Ammonium \n concentration (mg/cub. dm)')
plt.title('Ammonium Concentrations Over Time')


# The data appears to be fluctuating over time , with values tending to increase toward the end of the dataset. This could be representative of the change of ammonium levels with the seasons, changes in some other unknown variable, or it could represent anthropogenic (human) factors altering the levels of ammonium in the stream. Since the data are represented as monthly averages of ammonium, the peaks may represent the growing season for agricultural areas. But there is no distinguishable pattern indicating a yearly fluctuation, so the source of ammonium may be from manufacturing instead.

# To determine the relationship of station 1 and station 2 ammonium concentrations to the target station, I plotted ammonium values from the two stations against the ammonium values for the target station.

# In[ ]:


plt.figure(figsize=(13,5))
ax1=plt.scatter(ammonium_train['1'], ammonium_train['target'], label='station 1', color='magenta')
plt.scatter(ammonium_train['2'], ammonium_train['target'], label='station 2', color='blue')
plt.legend()
plt.xlabel('Downstream ammonium \n concentration (mg/cub. dm)')
plt.ylabel('Target ammonium \n concentration (mg/cub. dm)')
plt.title('Relationship Between Downstream and \n Target Station Ammonium Concentrations')


# It appears as though there is a linear relationship between the 2 upstream stations and the target station, with some distortion appearing as downstream ammonium levels increase. A linear model may be appropriate to predict the ammonium levels at the target station. Before we begin creating the model, there are some assumptions of the data that must hold true in order for linear models to be as accurate as possible:
# 
# 1. the relationship between the dependent and independent variables must be linear
# 2. Homoscedastic. The variance of the residual must be the same for any value of X, meaning the residual plot should look uniform with no identifiable patterns
# 3. The dependent variable must be normally distributed
# 4. Observations must be independent of each other
# 
# Since we have two independent variables (station 1 and station 2), our first idea will be to perform a multiple linear regression. But first, let's see if the assumptions are met. 

# In[ ]:


from scipy import stats


# In[ ]:


fig = plt.figure(figsize = (10, 7))
sns.residplot(ammonium_train['1'], ammonium_train['target'], color='magenta', label='station 1')
sns.residplot(ammonium_train['2'], ammonium_train['target'], color='blue', label='station 2')

plt.title('Residual plot', size=24)
plt.xlabel('upstream stations ammonium', size=18)
plt.ylabel('target station ammonium', size=18)
plt.legend()


# In[ ]:


stats.shapiro(ammonium_train['target'])


# The residual plot above is showing heteroscedasticity, which violates one of the assumptions of linear regression. On top of this, the Shapiro-Wilk test is showing a very small p value, which implies that the data are not normally distributed. A power transform may help to alleviate these issues:

# In[ ]:


from sklearn.preprocessing import PowerTransformer


# In[ ]:


pt = PowerTransformer(method='yeo-johnson') 


# In[ ]:


values=ammonium_train[['target','1','2']]
pt.fit(values)
yeoj_transform=pt.transform(values)
ammonium_train_transformed=pd.DataFrame(data=yeoj_transform, columns=['target','1','2'])
ammonium_train_transformed.head()


# In[ ]:


ax1=plt.scatter(ammonium_train_transformed['1'], ammonium_train_transformed['target'], label='station 1')
plt.scatter(ammonium_train_transformed['2'], ammonium_train_transformed['target'], label='station 2')
plt.legend()
plt.xlabel('Downstream ammonium \n concentration (mg/cub. dm)')
plt.ylabel('Target ammonium \n concentration (mg/cub. dm)')
plt.title('Relationship Between Downstream and \n Target Station Ammonium Concentrations')


# In[ ]:


fig = plt.figure(figsize = (10, 7))
sns.residplot(ammonium_train_transformed['1'], ammonium_train_transformed['target'], color='magenta', label='station 1')
sns.residplot(ammonium_train_transformed['2'], ammonium_train_transformed['target'], color='blue', label='station 2')

# title and labels
plt.title('Residual plot', size=24)
plt.xlabel('upstream stations ammonium', size=18)
plt.ylabel('target station ammonium', size=18)
plt.legend()


# In[ ]:


stats.shapiro(ammonium_train_transformed['target'])


# The power transform appears to have aided in making the residual plot more homoscedastic, and it has also greatly increased the p value of the Shapiro-Wilk test. Although the data are still not completely normal, and heteroscedasticity is still present, we will continue to use linear regression on this data to see how well the model performs. The next step is to determine if stations 1 and 2 should be included in the regression. We do this by performing a correlation.

# In[ ]:


ammonium_train_transformed[['target','1','2']].corr()


# Since stations 1 and 2 appear to be highly correlated, we must drop one of them. Station 1 seems to be a better predictor of the target station, so we will use only this station in the analysis. However, if the goal is to determine ammonium at the target station as far upstream as possible, it might be better to use station 2. It all depends on what the goal of the regression is. For this model, I will use station 1 in the regression.

# In[ ]:


X=ammonium_train_transformed[['1']].values
y=ammonium_train_transformed['target'].values


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


from sklearn import linear_model


# In[ ]:


linreg = linear_model.LinearRegression()


# In[ ]:


linreg.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import r2_score, mean_squared_error as mse


# In[ ]:


y_pred_train=linreg.predict(X_train)


# In[ ]:


print("The Mean Squared Error on the Train set is:\t{:0.5f}".format(mse(y_train, y_pred_train)))
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_pred_train)))


# In[ ]:


plt.plot(X_train,y_pred_train)
plt.scatter(ammonium_train['1'], ammonium_train['target'], label='station 1')
plt.scatter(ammonium_train['2'], ammonium_train['target'], label='station 2')
plt.legend()


# Now to test on the test data

# In[ ]:


y_pred_test=linreg.predict(X_test)


# In[ ]:


print("The Mean Squared Error on the Test set is:\t{:0.1f}".format(mse(y_test, y_pred_test)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test)))


# The model appears to generalize well in the test set when split from the same csv file. Let's now take a look at how it does with the actual test csv file.

# In[ ]:


ammonium_test=pd.read_csv('/kaggle/input/ammonium-prediction-in-river-water/test.csv')
ammonium_test.head()


# In[ ]:


newest_pred=linreg.predict(ammonium_test[['1']])
ammonium_test['target station guesses']=newest_pred
ammonium_test.head()


# In[ ]:


print("The Mean Squared Error on the Test set is:\t{:0.1f}".format(mse(y[0:63],newest_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y[0:63], newest_pred)))


# The model does not generalize well to completely new data. This may be due to the fact that data were taken at different times of the year, where different variables might be present to influence downstream ammonium concentrations. In fact, in the test csv file, stations 1 and 2 are 10% less correlated than in the first csv file, indicating that there are other factors at play influencing downstream concentrations:

# In[ ]:


ammonium_test['1'].corr(ammonium_test['2'])


# In conclusion, a linear model was not the best model to use for predicting ammonium concentrations downstream. The relationship between upstream and downstream concentrations appears to be non-linear, and since multiple factors can influence ammonium concentrations in lotic systems, a more complex model may be needed to better represent the complexity of nature. Future attempts will be made to predict ammonium concentrations with machine learning models.

# Thank you for reviewing this notebook. Please offer any constructive criticism you might have, so that I can improve my methods :)
