#!/usr/bin/env python
# coding: utf-8

# # Objective
# 
# Hi! Welcome to the updated version of my first Kernel.
# Here, I have used the dataset "Countries of the World". The data comes from 227 countries and contains information about some of the important factors that govern economic development.  In this, I have done exploratory data analysis and made a predictive model for GDP per capita, as this is the key indicator of economic development of any country. For building a better model, I have performed hyperparameter tuning of XGBoost and RandomForest.  In addition, I have found the key features that determine GDP per capita. Also, I have checked some other ingredients of Economic development and the relationships between them. 

# ### Importing the necessary modules and loading the data:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from xgboost import plot_tree
from xgboost import plot_importance
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', 500)
countries=pd.read_csv('../input/countries of the world.csv', decimal=',')
import warnings
warnings.filterwarnings('ignore')


# ### Checking the data:
# 
# We begin our analysis with some exploratory data analysis. We first check the columns, the shape of the distribution and about their statistics. To get a glimpse of how the data looks like, one can use the `head` and `tail` methods.

# In[ ]:


countries.columns


# In[ ]:


countries.info()


# Checking the statistics can give key insights about the distribution of various variables. This also helps in deciding which imputation method to use while dealing with missing data (this is discussed later in this kernel).

# In[ ]:


countries.describe()


# Next, we quickly check how many rows for each feature has a null entry (this is just an easier way to get the sum of rows with missing data for a variable compared to the `info()` method).

# In[ ]:


countries.isnull().sum()


# If you are curious about the missing data for a particular column feature., this example shows one way to do that.

# In[ ]:


countries[countries['Net migration'].isnull()]


# Machine learning models in Python do not entertain categorical data which means if we put these variables in our model then it will give an error.  Thus it is essential to check the data types before proceeding further, as shown here.  If the dtype is an object, it needs to be encoded. One of the ways to encode them is by using One Hot encoding where we will use Pandas `get_dummies` function.  As for the current data we have only two categorical variables: Country and Region. We will get back to this point when we start building our model.

# In[ ]:


countries.dtypes


# For further analysis, I am keen which regions are there and the number of countries that fall into them, their population and their total area.  To do such operations we use `groupby` function which helps in segregating the data into groups and then one can apply any function to either selected columns or to all.  Here, I first grouped the data by 'Region' then counted the number of countries as well as the total population and area that fall under each region.

# In[ ]:


countries_grouped= countries.groupby('Region')[['Country','Population','Area (sq. mi.)']].agg({'Country':'count', 'Population':'sum','Area (sq. mi.)':'sum'})
countries_grouped


# In case you are curious about the region specific distribution of the countries and their details about the various features, you can find it here. To obtain data in that format I created a multi-index using `set_index` method as shown here. Here, I am only showing a few rows using `head` method.

# In[ ]:


countries_indexed= countries.set_index(['Region','Country']).sort_index()
countries_indexed.head(3)


# ### Dealing with the missing data:
# As seen above, we have 14 columns out of 20 that have missing entries. If we drop all these missing entries (as shown next), we lose 21% of out data.

# In[ ]:


countries_dropped=countries.dropna()
countries_dropped.info()


# In order to avoid loss of 21% of data, we fill the missing values in all 'NaN' containing columns. As median is a better statistical parameter here than mean because of some skewness of data in variables, we fill the missing values with the median of each region obtained for each variable.  Only in the case of  'Climate' column which has categorical data, the missing data is filled by the mode of that specific region. Dealing with missing values is a crucial step in any data analysis. Therefore, before deciding which parameter to use to fill the missing value, the distribution of the data should be carefully observed. Then we can check that we have filled all the missing values by using `isnull().sum()` function on dataframe (as done before).

# In[ ]:


countries['Climate'] = countries.groupby('Region')['Climate'].transform(lambda x: x.fillna(x.mode().max()))


# In[ ]:


lst=['Net migration','Infant mortality (per 1000 births)','GDP ($ per capita)','Literacy (%)','Phones (per 1000)','Arable (%)','Crops (%)','Other (%)','Birthrate','Deathrate','Agriculture','Industry','Service']
for col in countries.columns:
    if countries[col].isnull().sum()!=0:
        if col in lst:
            countries[col] = countries.groupby('Region')[col].transform(lambda x: x.fillna(x.median()))


# In[ ]:


countries.isnull().sum()


# ### Plotting and analysing the data:
# 
# After fixing all the missing values, it is time to check plots and see if we can find any interesting correlation between these variables. Here, we are using matplotlib and seaborn libraries for plotting. As can be observed from the heatmap below, GDP(% per capita) is showing a strong negative correlation with Infant mortality (per 1000 births), Birthrate and Agriculture. Also, it is showing a very strong positive correlation with Phones (per 1000) and a moderate positive correlation with Literacy(%) and Service. Moreover, it has some weak positive correlation with Net migration.  Additionally, Literacy(%) is showing strong negative correlation with Infant mortality (per 1000 births), Birthrate and Agriculture. In addition to GDP(% per capita), Literacy(%) is showing moderate positive correlation with Phones (per 1000), Service. Also, we find that agriculture is strongly positively correlated with Infant mortality (per 1000 births) and Birthrate, while it is negatively correlated with Service and Phone (per 1000) and as indicated before with  GDP(% per capita), Literacy(%).

# In[ ]:


fig,ax=plt.subplots(figsize=(14,14))
sns.heatmap(countries.corr(), annot=True, vmin=-1, vmax=1,fmt = ".2f", cmap = 'RdBu')
plt.show()


# One of the ways to visualize the correlation between various variables is by seaborn's pairplot. We try to see correlation between GDP(% per capita) and other factors that were showing positive or negative correlation with it. But, while doing that we can also observe the correlation among other factors as well. The diagonal subplots in pairplots are showing the region specific distribution of the variables.

# In[ ]:


sns.pairplot(countries, vars= ['GDP ($ per capita)','Infant mortality (per 1000 births)', 'Birthrate','Agriculture'],hue='Region',diag_kind="kde")


# In[ ]:


sns.pairplot(countries, vars= ['GDP ($ per capita)','Phones (per 1000)', 'Literacy (%)','Service'],hue='Region',diag_kind="kde")


# An interesting observation (but well-known and obvious) can be made from the following two bar graphs below: GDP (\$ per capita) for various regions and the population distribution of these regions. Sadly, some populated regions have low GDP (\$ per capita).

# In[ ]:


countries.groupby('Region')['GDP ($ per capita)'].sum().sort_values().plot(kind='bar')
plt.ylabel('GDP ($ per capita)')
plt.show()


# In[ ]:


countries.groupby('Region')['Population'].sum().sort_values().plot(kind='bar')
plt.ylabel('Population')
plt.show()


# ### Building a model for predicting GDP (per capita):
# As we have already observed from the pairplots that GDP ($ per capita) is showing non linear relationship with most of the variables, therefore, we move straight to play with Random Forest and  XGBoost for building a predictive model.  
# But, before we start building our model we have to get rid of categorical data that has `dtype` 'object', as machine learning models give errors with categorical data. We will encode them using get_dummies function. Here, we encode the 'Region' column of the dataframe. By encoding we create individual column for each value of the 'Region' column. After that we delete the 'Region' column from the new dataframe that we made after encoding. Moreover, we also drop 'Countries' column which is 'object' dtype and we do not need it further. 

# In[ ]:


countries2= pd.concat([countries,pd.get_dummies(countries['Region'], prefix='region')], axis=1).drop(['Region'],axis=1)
print(countries2.head())
print(countries2.dtypes)
print(countries2.shape)


# Here, we first try Random Forest. For checking the model's quality we are using Root Mean Square Error (RMSE) and R2. R2 is the proportion of variability that is explained by our model. It ranges from 0-1 for maximum proportion of variability explained . 

# In[ ]:


y = countries2['GDP ($ per capita)']
X = countries2.drop(['GDP ($ per capita)','Country'], axis=1)

forest_model = RandomForestRegressor(random_state=21)
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25, random_state=21)
rmse= np.sqrt(np.mean(-cross_val_score(forest_model, train_X, train_y,cv=5,  scoring='neg_mean_squared_error')))
print("RMSE : %f" % (rmse))
r2_score1= np.mean(cross_val_score(forest_model, train_X, train_y,cv=5,  scoring='r2'))
print("R2 score: %s" % '{:.2}'.format(r2_score1))


# Before proceeding further, we need the best parameters for our RandomForestRegressor. We try to find them using RandomSearchCV followed by GridSearchCV. RandomSearchCV helps to narrow down the number and range of parameters, which can then be tested using GridSearchCV. 

# In[ ]:


number_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': number_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# Total number of combinations available are 4320. In RandomizedSearchCV not all parameters are tried but a fixed number of combinations is tested using  `n_iter` . It is tradeoff between computing time and quality of the solution

# In[ ]:


forest_model = RandomForestRegressor(random_state=21)
rf_random = RandomizedSearchCV(estimator = forest_model, param_distributions = random_grid, 
                          cv = 5, n_jobs = -1,n_iter = 100, verbose = 0)
rf_random.fit(train_X, train_y)
print(rf_random.best_params_)


# Random search helps to narrow down the range for all the hyperparameters. This is then followed by GridSearch CV where we can test all the combinations of the parameters unlike Random search. 

# In[ ]:


param_grid = {'max_depth': [10, 20, 40, 60, 80],
              'max_features': ['sqrt'],
              'min_samples_leaf': [1, 3, 4, 5],
              'min_samples_split': [2, 4, 8],
              'n_estimators': [100, 300, 600, 1000],
             'bootstrap': [False, True]}
# Create a basic model
rf = RandomForestRegressor(random_state=21)
# Instantiate the grid search model
rf_grid = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 0)
rf_grid.fit(train_X, train_y)
print(rf_grid.best_params_)


# Next we check RMSE and R2 score for the hyperparameter tuned RandomForest. There is an improvement in RMSE score compared to the simplest RandomForestRegressor model used above.

# In[ ]:


rf_cv_random=RandomForestRegressor(random_state=21,n_estimators= 600, min_samples_split= 2, min_samples_leaf= 1, max_features= 'sqrt', max_depth= 40, bootstrap= False)
rf_cv_random.fit(train_X,train_y )
predictions=rf_cv_random.predict(test_X)
rmse3 = np.sqrt(mean_squared_error(test_y, predictions))
print("RMSE : %f" % (rmse3))
r23= r2_score(test_y,predictions)
print("R2 score: %s" % '{:.2}'.format(r23))


# Now, we try XGBoost and compare the RMSE from both the models.  As expected the RMSE reduces and R2 score increases for XGBoost.

# In[ ]:


model_x = XGBRegressor(random_state=21)
model_x.fit(train_X, train_y, verbose=False)
predictions4 = model_x.predict(test_X)

rmse4 = np.sqrt(mean_squared_error(test_y, predictions4))
print("RMSE: %f" % (rmse4))
r24= r2_score(test_y,predictions4)
print("R2 score: %s" % '{:.2}'.format(r24))


# Next, we try hyperparameter tuning for XGBoost. As this was taking a long time on Kaggle kernel, I performed hyperparameter tuning outside of kaggle kernel. The best parameters obtained : {'colsample_bytree': 0.6, 'gamma': 0.5, 'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 500, 'subsample': 0.6}

# In[ ]:


#param_grid = {'n_estimators':[100,500,1000],'learning_rate': [0.001,0.01,0.05,0.1,1], 'max_depth': [3, 4, 5,6], 'subsample': [0.6, 0.8, 1.0],'colsample_bytree': [0.6, 0.8, 1.0],'gamma': [0.5, 1, 1.5, 2, 5]}
#model_x = XGBRegressor(random_state=21)
#xgb = GridSearchCV(estimator = model_x, param_grid = param_grid, 
#                          cv = 5, n_jobs = -1, verbose = 0)
#xgb.fit(train_X, train_y)
#print(xgb.best_params_)


# In[ ]:


train_X = pd.DataFrame(data=train_X, columns=X.columns)
test_X = pd.DataFrame(data=test_X, columns=X.columns)
model_x2 = XGBRegressor(n_estimators=500, learning_rate=0.01,max_depth= 4, subsample=0.6,gamma= 0.5,colsample_bytree= 0.6,random_state=21)
model_x2.fit(train_X, train_y)
predictions2 = model_x2.predict(test_X)

rmse2 = np.sqrt(mean_squared_error(test_y, predictions2))
print("RMSE: %f" % (rmse2))
r22= r2_score(test_y,predictions2)
print("R2 score: %s" % '{:.2}'.format(r22))


# Now, using XGBoost's plot_importance function, we find the important features. 

# In[ ]:


plot_importance(model_x2)


# Now, we take the top 7 important features and use XGBoost for model prediction. This improved both RMSE and R2 scores suggesting that these are the key players. Including other features resulted in increasing RMSE and reducing R2.

# In[ ]:


y = countries2['GDP ($ per capita)']
X = countries2[['Population','Phones (per 1000)','Infant mortality (per 1000 births)','Agriculture','Net migration','Area (sq. mi.)','Birthrate']]
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25, random_state=21)
train_X = pd.DataFrame(data=train_X, columns=X.columns)
test_X = pd.DataFrame(data=test_X, columns=X.columns)
model_x3 = XGBRegressor(n_estimators=500, learning_rate=0.01,max_depth= 4, subsample=0.6,gamma= 0.5,colsample_bytree= 0.6,random_state=21)
model_x3.fit(train_X, train_y)
predictions3 = model_x3.predict(test_X)

rmse3 = np.sqrt(mean_squared_error(test_y, predictions3))
print("RMSE: %f" % (rmse3))
r23= r2_score(test_y,predictions3)
print("R2 score: %s" % '{:.2}'.format(r23))


# 

# ### Conclusion 
# In this analysis we build a predictive model using XGBoost which showed R2 score of 0.83. Also, from this dataset, we found the key features for predicting GDP (% per capita) are : Population, Phones (per 1000), Infant mortality (per 1000 births), Agriculture, Net migration, Area (sq. mi.) and Birthrate
# 
# Thanks for reading. Any feedback is highly appreciated.
# 
