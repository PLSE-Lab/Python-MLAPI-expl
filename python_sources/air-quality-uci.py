#!/usr/bin/env python
# coding: utf-8

# # Nitrogen Oxides(NOx) Level Analysis in Air Quality

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data = pd.read_csv('/kaggle/input/airquality-uci/AirQualityUCI.csv', sep=',', delimiter=";",decimal=",")


# In[ ]:


data.head()


# In[ ]:


data.tail()


# Checking the shape(dimensions) of the dataset.

# In[ ]:


data.shape


# Checking the Information about the dataset by using info()

# In[ ]:


data.info()


# From the info() we can see the following things:
# *   There are 17 columns
# *   There are 9471 records
# *   There are:
# > * Number of datetime column: 01
# > * Number of float datatype column(s): 15
# > * Number of object datatype column: 01
# 
# 
# 
# 
# 
# 
# 
# 

# ## **Data Cleaning**

# In[ ]:


#Deleting the Unnamed: 15 and Unnamed: 16 columns.
data = data.drop(["Unnamed: 15","Unnamed: 16"], axis=1)


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.info()


# Checking the missing data in our dataset:

# In[ ]:


data.isnull().any()


# In[ ]:


data.isnull().sum()


# In[ ]:


#Deleting all Null values in our dataset permanently.
data.dropna(inplace=True)
data.shape


# In[ ]:


data.set_index("Date", inplace=True) 
#setting Date column as new index of out dataframe.


# In[ ]:


data.head(1)


# In[ ]:


data.index = pd.to_datetime(data.index) #Converting the index in datetime datatype.
type(data.index)


# In[ ]:


data.head(1)


# In[ ]:


data['Time'] = pd.to_datetime(data['Time'],format= '%H.%M.%S').dt.hour #Selecting only Hour value from the 'Time' Column.
type(data['Time'][0])


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# ## **Handling Missing Data**

# In[ ]:


data.plot.box()
plt.xticks(rotation = 'vertical')
plt.show()


# Replacing -200 with NaN(null value) as it given in dataset information that Missing values are tagged with -200 value.

# In[ ]:


data.replace(to_replace= -200, value= np.NaN, inplace= True)


# In[ ]:


data.isnull().any()


# In[ ]:


data.isnull().sum()


# In[ ]:


plt.figure(figsize=(8,6))
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# In[ ]:


data.drop('NMHC(GT)', axis=1, inplace=True)


# In[ ]:


data.describe()


# In[ ]:


data.shape


# #### filling the null values with median

# In[ ]:


data.fillna(data.median(), inplace=True)


# In[ ]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# In[ ]:





# ### **Data Visualization**

# In[ ]:


sns.set_style('whitegrid')
eda_data = data.drop(['Time','RH','AH','T'], axis=1)
sns.pairplot(eda_data)


# **Histogram:**

# In[ ]:


data.hist(figsize = (20,20))
plt.show()


# From the histogram, we can observe the variability of each attribute.
# Also we can observe the skewness of data.

# **Distplot:**

# In[ ]:


data.drop(['Time','RH','AH','T'], axis=1).resample('M').mean().plot(figsize = (20,8))
plt.legend(loc=1)
plt.xlabel('Month')
plt.ylabel('All Toxic Gases in the Air')
plt.title("All Toxic Gases' Frequency by Month")


# In the above graph, you can see the frequency of all toxics that is usually in polluted air. The Brown line shows Nitrogen Oxides (NOx) and Yellow line shows NO2 which is part of NOx. It is a mixture of gases are composed of nitrogen and oxygen. Two of the most toxicologically significant compounds are nitric oxide (NO) and nitrogen dioxide (NO2). I chose Nitrogen Oxides(NOx) because these are one of the most dangerous forms of air pollution and are most relevant for air pollution. However, There are many others ways to measure air pollution, including PM10 (particulate matter around between 2.5 and 10 microns in diameter), carbon monoxide, sulfur dioxide, nitrogen dioxide, ozone (O3), etc.
# 
# NOx is produced from the reaction of nitrogen and oxygen gases in the air during combustion, especially at high temperatures. In areas of high motor vehicle traffic, such as in large cities, the amount of nitrogen oxides emitted into the atmosphere as air pollution can be significant.
# 
# It is mainly due to fossil fuel combustion from both stationary sources, i.e. power generation (21%), and mobile sources, i.e. transport (44%). Other atmospheric contributions come from non-combustion processes, for example nitric acid manufacture, welding processes and the use of explosives.
# 
# In addition, these create serious health issues. These mainly impact on respiratory conditions causing inflammation of the airways at high levels. Long term exposure can decrease lung function, increase the risk of respiratory conditions and increases the response to allergens. NOx also contributes to the formation of fine particles (PM) and ground level ozone, both of which are associated with adverse health effects.
# 
# Ref: https://www.epa.gov/no2-pollution/basic-information-about-no2#Effects

# In[ ]:


data['NOx(GT)'].resample('M').mean().plot(kind='bar', figsize=(18,6))
plt.xlabel('Month')
plt.ylabel('Total Nitrogen Oxides (NOx) in ppb')   # Parts per billion (ppb)
plt.title("Mean Total Nitrogen Oxides (NOx) Level by Month")


# We can see that initially, the Nitric Oxide levels are low but as the year pass, the Nitric Oxide level is increased.

# In[ ]:


plt.figure(figsize=(20,6))
sns.barplot(x='Time',y='NOx(GT)',data=data, ci=False)
plt.xlabel('Hours')
plt.ylabel('Total Nitrogen Oxides (NOx) in ppb') # Parts per billion (ppb)
plt.title("Mean Total Nitrogen Oxides (NOx) Frequency During Days")


# Here, the graph shows an average of Oxides of Nitrogen level with hours. It seems during the day, its level is high compared to night because of high use of transportations, phones, other electronics etc.
# 
# The Environmental Protection Agency (EPA) set a 1-hour NOx standard at the level of 100 parts per billion (ppb). (Ref: https://www.airnow.gov/index.cfm?action=pubs.aqiguidenox)
# 
# Here, this data shows, air has large amount of NOx compare to its standard measurement which is not good.

# In[ ]:


data.plot(x='NO2(GT)',y='NOx(GT)', kind='scatter', figsize = (10,6), alpha=0.3)
plt.xlabel('Level of Nitrogen Dioxide')
plt.ylabel('Level of Nitrogen Oxides (NOx) in ppb') # Parts per billion (ppb)
plt.title("Mean Total Nitrogen Oxides (NOx) Frequency During Days")
plt.tight_layout();


# Checking the Correlation between each attributes.

# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, linewidths=.20)


# KDEPlot of Nitric Oxide:

# In[ ]:


sns.kdeplot(data['NOx(GT)'])
plt.show()


# Plotting of Nitric oxide:

# In[ ]:


plt.figure(figsize=(20,5))
plt.plot(data['NOx(GT)'])
plt.show()


# According to above graph, the maximum level of Nitric Oxide is spotted between September 2004 to April 2005

# Let's check the trend of Nitrogen gas:

# In[ ]:


plt.figure(figsize=(21,8))
plt.plot(data['NO2(GT)'])
plt.show()


# From the beginning of 2004 to April 2005, there is much more level of Nitrogen gas.

# In[ ]:


data.shape


# ## **Machine Learning**

# ### Training a **Linear Regression** Model

# In[ ]:


X = data.drop(['NOx(GT)','Time'], axis=1)

y= data['NOx(GT)']


# ### Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ### Scaling and Transformation

# In[ ]:


from sklearn.preprocessing import RobustScaler

sc=RobustScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# ### Creating and Training the Model

# In[ ]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(x_train, y_train)


# ### Model Evaluation

# In[ ]:


print(lm.intercept_)


# In[ ]:


coeff_data = pd.DataFrame(lm.coef_, index=X.columns, columns=['Coefficient'])
coeff_data


# From above coefficient values, we can say: if 1 unit increases in Benzene (C6H6), NOx increases by 72.62. Same as, if 1 unit increases in Nitrogen Dioxide(NO2) and Relative Humidity(RH), Oxides of Nitrogen will increase by 71 points and 72.9 points, respectively.

# ####Prediction Model

# In[ ]:


prediction = lm.predict(x_test)
plt.scatter(y_test, prediction, c="blue", alpha=0.3)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Linear Regression Predicted vs Actual')


#  Accuracy on Training Dataset

# In[ ]:


score_train = lm.score(x_train, y_train)
score_train


#  Accuracy on Test Dataset

# In[ ]:


prediction = lm.predict(x_test)


# In[ ]:


score_test = lm.score(x_test, y_test)
score_test


# Residualt Histogram
# 

# In[ ]:


sns.distplot((y_test-prediction), bins=70, color="purple")


# In[ ]:


from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test, prediction))
print('MSE:',metrics.mean_squared_error(y_test, prediction))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[ ]:


coeff_data


# If we hold all other varibles constant and 1 point increases in CO(GT), NOx will increase by 49.81. Simillarly, If we hold all other varibles constant and 1 point increases in NO2(GT), NOx will increase by 1.48. and, If we hold all other varibles constant and 1 point increases in C6H6(GT), NOx will increase by 11.94.
# 

# ### **K-Nearest Neighbors Algorithm (KNN)**

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train,y_train)


# ####Prediction Model

# In[ ]:


prediction = knn.predict(x_test)
plt.scatter(y_test, prediction, c="black", alpha=0.3)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('K-nearest Neighbors Predicted vs Actual')


#  Accuracy on Training Dataset

# In[ ]:


knn_train = knn.score(x_train,y_train)
knn_train


# Accuracy on Test Dataset

# In[ ]:


kreg_test = knn.score(x_test,y_test)
kreg_test


# ### HyperParameter Tuning

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
kreg=KNeighborsRegressor()


# In[ ]:


para={'n_neighbors':np.arange(1,51),'weights':['uniform','distance'],
      'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],'leaf_size':np.arange(10,51)}


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


knn_cv=RandomizedSearchCV(kreg,para,cv=5,random_state=0)
knn_cv.fit(x_train,y_train)


# In[ ]:


print(knn_cv.best_score_)
print(knn_cv.best_params_)


# In[ ]:


kn=KNeighborsRegressor(weights='distance', n_neighbors= 12, leaf_size= 18, algorithm= 'brute')
kn.fit(x_train,y_train)


# ####Prediction Model

# In[ ]:


prediction = kn.predict(x_test)
plt.scatter(y_test, prediction, c="black", alpha=0.3)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('K-nearest Neighbors(Hyper) Predicted vs Actual')


# In[ ]:


#Accuracy on Training data
knn_train=kn.score(x_train,y_train)
knn


# In[ ]:


#Accuracy on Training data
knn_test=knn.score(x_test,y_test)
knn_test


# ### **Decision Tree Regressor Algorithm**

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dreg=DecisionTreeRegressor()
dreg.fit(x_train,y_train)


# ####Prediction Model

# In[ ]:


prediction = dreg.predict(x_test)
plt.scatter(y_test, prediction, c="green", alpha=0.3)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Decision Tree Predicted vs Actual')


# Accuracy on Training Data

# In[ ]:


dreg_train = dreg.score(x_train,y_train)
dreg_train


# Accuracy on Test Data

# In[ ]:


dreg_test = dreg.score(x_test,y_test)
dreg_test


# ### HyperParameter Tuning

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
d=DecisionTreeRegressor()


# In[ ]:


Param_grid={'splitter':['best','random'],'max_depth':[None,2,3,4,5],'min_samples_leaf':np.arange(1,9),
            'criterion':['mse','friedman_mse','mae'],'max_features':['auto','sqrt','log2',None]}


# In[ ]:


dt_cv=RandomizedSearchCV(d,Param_grid,cv=5,random_state=0)
dt_cv.fit(x_train,y_train)


# In[ ]:


print(dt_cv.best_score_)
print(dt_cv.best_params_)


# In[ ]:


dtr=DecisionTreeRegressor(splitter= 'best', min_samples_leaf= 8, max_features= 'auto', max_depth=None, criterion= 'mse')
dtr.fit(x_train,y_train)


# ####Prediction Model

# In[ ]:


prediction = dtr.predict(x_test)
plt.scatter(y_test, prediction, c="green", alpha=0.3)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Decision Tree(Hyper) Predicted vs Actual')


# In[ ]:


dtr_train=dtr.score(x_train,y_train)
dtr_train


# In[ ]:


#Accuracy on Test Data
dtr_test=dtr.score(x_test,y_test)
dtr_test


# ### **Random Forest Regressor Algorithm**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfreg=RandomForestRegressor(n_estimators=10,random_state=0)
rfreg.fit(x_train,y_train)


# ####Prediction Model

# In[ ]:


prediction = rfreg.predict(x_test)
plt.scatter(y_test, prediction, c="purple", alpha=0.3)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Random Forest Predicted vs Actual')


# Accuracy on Training data
# 

# In[ ]:


rfreg_train = rfreg.score(x_train,y_train)
rfreg_train


# Accuracy on Test data
# 

# In[ ]:


rfreg_test = rfreg.score(x_test,y_test)
rfreg_test


# ### HyperParameter Tuning

# In[ ]:


rh=RandomForestRegressor()


# In[ ]:


par={'n_estimators':np.arange(1,91),'criterion':['mse','mae'],'max_depth':[2,3,4,5,None],
     'min_samples_leaf':np.arange(1,9),'max_features':['auto','sqrt','log2',None]}


# In[ ]:


rh_cv=RandomizedSearchCV(rh,par,cv=5,random_state=0)
rh_cv.fit(x_train,y_train)


# In[ ]:


print(rh_cv.best_score_)
print(rh_cv.best_params_)


# In[ ]:


reg=RandomForestRegressor(n_estimators= 74, min_samples_leaf= 2, max_features= 'log2', max_depth=None, criterion= 'mse')
reg.fit(x_train,y_train)


# ####Prediction Model

# In[ ]:


prediction = reg.predict(x_test)
plt.scatter(y_test, prediction, c="purple", alpha=0.3)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Random Forest(Hyper) Predicted vs Actual')


# In[ ]:


#Accuracy on Training data
reg_train=reg.score(x_train,y_train)
reg_train


# In[ ]:


#Accuracy on Training data
reg_test=reg.score(x_test,y_test)
reg_test


# In[ ]:





# ### **SVM Algorithm**

# In[ ]:


from sklearn.svm import SVR
sreg = SVR(kernel='linear')
sreg.fit(x_train, y_train)


# ####Prediction Model

# In[ ]:


prediction = sreg.predict(x_test)
plt.scatter(y_test, prediction, c="brown", alpha=0.3)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('SVM Predicted vs Actual')


# Accuracy on Training data
# 

# In[ ]:


sreg_train = sreg.score(x_train,y_train)
sreg_train


# Accuracy on Test data

# In[ ]:


sreg_test = sreg.score(x_test,y_test)
sreg_test


# ### **XGBoost Algorithm**

# In[ ]:


from xgboost import XGBRegressor
xreg = XGBRegressor()
xreg.fit(x_train, y_train)


# ####Prediction Model

# In[ ]:


plt.scatter(y_test, xreg.predict(x_test),c="red", alpha=0.2)
plt.xlabel('NOx(GT)(y_test)')
plt.ylabel('XGBoost Predicted vs Actual')
plt.show()


# Accuracy on Training data
# 

# In[ ]:


xreg_train = xreg.score(x_train,y_train)
xreg_train


# Accuracy on Test data
# 

# In[ ]:


xreg_test = xreg.score(x_test,y_test)
xreg_test


# ###Hyper-parameter Optimization

# In[ ]:


booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]


# In[ ]:


## Hyper Parameter Optimization

n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }


# In[ ]:


# Set up the random search with 4-fold cross validation
from sklearn.model_selection import RandomizedSearchCV
random_cv = RandomizedSearchCV(estimator=xreg,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = -1,
            verbose = 5, 
            return_train_score = True,
            random_state=42)


# In[ ]:


random_cv.fit(x_train,y_train)


# In[ ]:


random_cv.best_estimator_


# In[ ]:


from xgboost import XGBRegressor
xreg = XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=10, min_child_weight=3, missing=None, n_estimators=100,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)
xreg.fit(x_train, y_train)


# ####Prediction Model

# In[ ]:


plt.scatter(y_test, xreg.predict(x_test),c="red", alpha=0.2)
plt.xlabel('NOx(GT)(y_test)')
plt.ylabel('HXBoost(Hyper) Predicted vs Actual')
plt.show()


# In[ ]:


#Accuracy on Training data
xreg_train = xreg.score(x_train,y_train)
xreg_train


# In[ ]:


#Accuracy on Test Data
xreg_test = xreg.score(x_test,y_test)
xreg_test


# XGBoost is good but is little bit bias.

# ## **Accuracies(After HyperParameter Tuning)**

# In[ ]:


results = pd.DataFrame({'Algorithm':['Linear Regression','K-Nearest Neighbour Regressor','Decision Tree Regressor', 'Random Forest Regressor',
                                     'Support Vector Machine Regressor', 'XGBoost'],
                        'Train Accuracy':[score_train, knn_train, dtr_train,reg_train, sreg_train,xreg_train],
                        'Test Accuracy':[score_test, knn_test, dtr_test,reg_test, sreg_test,xreg_test]})
results.sort_values('Test Accuracy', ascending=False)


# **Conclusion:** 
# * Gases like CO, NOx, titania, and Benzene are increased in the air over time.
# * The frequency of Oxides in Nitrogen is increasing
# * During the day Nitrogen Oxides' level is high compared to night
# * By looking at above table, we conclude that Random Forest regressor performed best and is not bias as there is not much difference between predicting the training and testing data.

# ##**Ensemble Learning**

# ###Applying Bagging on every algorithm one ny one.

# In[ ]:


from sklearn.ensemble import BaggingRegressor


# In[ ]:


m1=BaggingRegressor(LinearRegression()) #one method
m1.fit(x_train,y_train)


# In[ ]:


m2=BaggingRegressor(KNeighborsRegressor(weights='distance', n_neighbors= 12, leaf_size= 18, algorithm= 'brute'))
m2.fit(x_train,y_train)
  


# In[ ]:


m3=BaggingRegressor(DecisionTreeRegressor(splitter= 'best', min_samples_leaf= 8, max_features= 'auto', max_depth=None, criterion= 'mse')) #one method
m3.fit(x_train,y_train)


# In[ ]:


m4=BaggingRegressor(RandomForestRegressor(n_estimators= 74, min_samples_leaf= 2, max_features= 'log2', max_depth=None, criterion= 'mse')) #one method
m4.fit(x_train,y_train)
#here low bias and low variance than decision tree algo


# In[ ]:


m5=BaggingRegressor(SVR(kernel='linear')) #one method
m5.fit(x_train,y_train)


# In[ ]:


m6=BaggingRegressor(XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=10, min_child_weight=3, missing=None, n_estimators=100,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1))
m6.fit(x_train,y_train)


# ###**Accuracies (After applying Bagging Regressor)**

# In[ ]:


results = pd.DataFrame({'Algorithm':['Linear Regression','K-Nearest Neighbour Regressor','Decision Tree Regressor', 'Random Forest Regressor',
                                     'Support Vector Machine Regressor', 'XGBoost'],
                        'Train Accuracy':[m1.score(x_train,y_train), m2.score(x_train,y_train), m3.score(x_train,y_train),
                                          m4.score(x_train,y_train), m5.score(x_train,y_train), m6.score(x_train,y_train)],
                        'Test Accuracy':[m1.score(x_test,y_test) , m2.score(x_test,y_test), m3.score(x_test,y_test),
                                         m4.score(x_test,y_test), m5.score(x_test,y_test), m6.score(x_test,y_test)]})
results.sort_values('Test Accuracy', ascending=False)


# From above we can say that 
# * XGBoost Regressor Algorithm is having the best Accuracy(i.e., Prediction) on Test Data which is 0.9240.

# ##**Conclusions:**

#  
# * Gases like CO, NOx, titania, and Benzene are increased in the air over time.
# * The frequency of Oxides in Nitrogen is increasing.
# * During the day Nitrogen Oxides' level is high compared to night

# For this Air quality data analysis, we saw that NOx's ppb are increasing due to the air pollution causing factors as mentioned above and badly affects our health and enviroment. Before it becomes too dangeous for us, There are several intiatives has been tried successfully and some of them are as follow:
# 
# * Switching fuel that has reduce low NOx emmision. For instance, No. 2 oil instead of No. 6, distillate oil and natual gas.
# * Recircuclating flue gas which a waste gas produced at the power station and other big installation, with the combustion air supplied to the burners. This process of diluting the combustion air with flue gas, reduces both the oxygen concentration at the burners and the temperature and has reduced NOx emissions by 30 to 60%.
# * Water Injection and Water Emulsion, in which water is added to reduce temperature of the combustion. Water is mixed with fuel at mounted of the cylinder to inject water. This method can reduce NOx by 20-45%
# References:
# 
# https://www.pollutiononline.com/doc/nox-emission-reduction-strategies-0001
# https://www.marineinsight.com/tech/10-technologiesmethods-for-controlling-nox-sox-emissions-from-ships/
