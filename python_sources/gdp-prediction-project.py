#!/usr/bin/env python
# coding: utf-8

# ## GDP project - analyzing and predicting countrys' GDP based on socio-economic factors
# 
# ### Goals:
# 
# the goal of this project, as a student of economics and psychology, was to integrate my passion and knowledge in the field of ecomonics and with my developing programming skills. in order to do that, i've chosen to gather some data regarding countrys' GDP and it's related factors, analyze it, and try to fit it into a regression model which will help to predict it correctly.

# In[ ]:


import os
print(os.listdir("../input/countries-of-the-world"))
print(os.listdir("../input/additional-data"))


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# the data we're about to analyze is taken from https://www.kaggle.com/fernandol/countries-of-the-world and contains socioeconomic indicators as well as their Gross Domestic Product aka GDP.
# In the field of macroeconomics, the GDP is basically the main measure of the country's value of goods and services.
# GDP is considered a strong indicator for many aspects of the nation's economic system, and as such, a major area of research is trying to explore, study, and predict the factors which can contribute to one's GDP.
# To add some more variables which i think can help our model to be fully capble of predicting the GDP, i added a few more indicators taken from the World Bank such as Imports and Exports, the CIA factbook website and https://www.heritage.org/index/ranking for some indicators on market freedom.
# 
# with that in mind, let's start working.

# In[ ]:


dfc = pd.read_csv("../input/countries-of-the-world/countries_of_the_world.csv", decimal=',')
df_WB = pd.read_csv("../input/additional-data/additional_data_WB.csv")
df_rel = pd.read_csv("../input/additional-data/additional_data_religion.csv")
df_freedom = pd.read_csv("../input/additional-data/additional_data_freedom.csv")


# ## preprocessing the data:
# ### goals:
# #### * clean each data set, then merge it based on country name - we want to make sure our columns are organized and ready to be fully merged to one another in order to create a one database to work on
# #### * dealing with null values - we want to have zero null values, we will handle with them based on the characteristics of the specific variable and decide how to manage it.
# #### * dealing with outliers -  basically we wanna have a somewhat normal distributed variables. when trying to predict values with regression models, outliers can be problematic and skew the results to an unwanted direction. we'll deal with that when we finish with steps 1 and 2.

# In[ ]:


### changing the column name to fit the other data
dfc.rename(columns = {'Country':'Country Name'}, inplace=True)

### removing the space after each country name
dfc['Country Name'].values
dfc['Country Name'] = dfc['Country Name'].apply(lambda x: x.split(" ")[0] if len(x.split())==1 else x)
### merging the data with the WorldBank data
df_n_wb = dfc.merge(df_WB, on = 'Country Name', how='left')
df_n_wb.head()


# In[ ]:


### removing the space in the country name column of the religion data
df_rel['Country Name'] = df_rel['Country Name'].apply(lambda x: x.strip())

#merging the data with the religion data
df_n_wb_rel = df_n_wb.merge(df_rel, on = 'Country Name', how = 'left' )
df_n_wb_rel.drop(['Unnamed: 1'],axis=1, inplace=True)
df_n_wb_rel['Country Name'].values


# In[ ]:


### merging the data with the freedom index data
df = df_n_wb_rel.merge(df_freedom, on='Country Name', how='inner')

### rename some columns to be more comfortable to work with:

df.rename(columns = {'GDP ($ per capita)':'GDP', 'Country Name':'Country','Imports of goods and services  2013':'Imports', 'Exports of goods and services  2013':'Exports'
                    ,'Foreign direct investment, net inflows 2013': 'Net foreign invest'}, inplace=True)
df.Region = [x.strip() for x in df.Region]
df = df.rename(columns = lambda x: x.split('(')[0].strip())
df.shape


# In[ ]:


df.Country = df.Country.astype('category')
df.Region = df.Region.astype('category')
df['Main Religion'] = df['Main Religion'].astype('str')
df.info()


# In[ ]:


df['Main Religion'].value_counts()
df['Main Religion'] = df['Main Religion'].replace('Christian ', 'Christian').replace('Christian (Free Wesleyan Church claims over 30','Christian')
df['Main Religion'] = df['Main Religion'].replace('Muslim*', 'Muslim')
df['Main Religion'] = df['Main Religion'].replace('Buddist', 'Buddhist').replace('Buddhist', 'Buddhist').replace('BuddhismAnd','Buddhist')
df['Main Religion'] = df['Main Religion'].replace('Buddhist ', 'Buddhist')
df['Main Religion'] = df['Main Religion'].replace('syncretic (part Christian', 'Christian')
df['Main Religion'] = df['Main Religion'].replace('Zionist 40% (a blend of Christianity and indigenous ancestral worship)', 'Zionist')


# ## step 1 seems to be done, the data is clean and we're ready to move toward step 2 which is handling with null values.
# 
# ### the guidline for this step will be as follows:
# #### 1)for numerical data - we'll fill the null values based on the mean or median of the data, depends on what seems logically reasonable
# #### 2)for categirical data - we'll fill the null values based on the mode or a value that fit the best based on the knowledge we have

# In[ ]:


### presenting the number of null values by total and percentage:

total = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))
percent = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)/len(df))
data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
data


# In[ ]:


### Let's have a look at the variable Climate for it is a categorial one.
## let's find out what can fit the most to our null value for climate in each country

print(pd.pivot_table(df,index=['Climate'],values = 'Region', aggfunc='sum'))
print(df[df.Climate.isnull()]['Country'])
df.loc[3,['Climate']] = 3
df.loc[25,['Climate']] = 3
df.loc[31,['Climate']] =3
df.loc[64,['Climate']] =3
df.loc[74,['Climate']] =1
df.loc[77,['Climate']] =2.5
df.loc[79,['Climate']] =3
df.loc[80,['Climate']] =3
df.loc[88,['Climate']] =3
df.loc[92,['Climate']] =3
df.loc[94,['Climate']] =1
df.loc[97,['Climate']] =2
df.loc[113,['Climate']] =1
df.loc[117,['Climate']] =3
df.loc[121,['Climate']] =3
df.loc[132,['Climate']] =3
df.Climate.isnull().sum()


# Alright we handled the Climate column, let's move on and have a look at the other variables.
# 
# looks like government expenditure on education is null on nearly 40% of the data, it seems reasonable enoough to drop it
# 

# In[ ]:


df.drop(['Government expenditure on education % of GDP 2014'], axis=1, inplace=True)


# In[ ]:


total = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))
percent = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)/len(df))
data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
data


# In[ ]:


total = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))
percent = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)/len(df))
data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
data


# ### Now moving to the numerical variables null values.
# in order to fill the nulls, let's examine first what's the distribution for the top variables and see if we should use the median or the mean

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize = (10,3))
plt.subplot(1,3,1)
sns.distplot(df['ATMs per 100,000 adults 2013'])
plt.subplot(1,3,2)
sns.kdeplot(df['Exports'])
plt.subplot(1,3,3)
sns.kdeplot(df['Net foreign invest'])
plt.tight_layout()


# seems like the distributions are pretty much normal(except for ATM's which is a little bit skewd to the right but does'nt seem problematic), therefore i think using the mean will be just fine

# In[ ]:


columns = data.index[:16]
for column in columns:
    df[column] = df[column].fillna(df.groupby('Region')[column].transform('mean'))


# In[ ]:


total = pd.DataFrame(df.isnull().sum().sort_values(ascending=False))
percent = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)/len(df))
data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
data


# ### So, we took care of the null values.
# next we're gonna take a look at  the variables which we expect to have high correlation with the GDP.
# 
# 
# lets plot a heatmap to have a closer look at the details

# In[ ]:


corrs = df.corr()
fig, ax = plt.subplots(figsize = (12,12))
heatcor = sns.heatmap(corrs, cbar=True ,ax=ax).set(title = 'Correlation Map', xlabel = 'Columns', ylabel = 'Columns' )


# ### alright looks like we can get some insights from this map:
# looks like the "freedom" indicators are correlated quite well with the GDP
# 
# additionaly, ATM's, phones(maybe an inverse causality?), and Foreign investements and trades(imports and exports)
# 
# we'll take a closer look on the top 10 correlators
# 
# lets put it all in a nice correlation heatmap, this time smaller and more detailed

# In[ ]:


top_corrs = df.corr().nlargest(10, 'GDP').index
cm = np.corrcoef(df[top_corrs].values.T)
heatcor = sns.heatmap(cm, cbar = True, annot = True, cmap='BrBG', yticklabels = top_corrs.values, xticklabels=top_corrs.values)


# ### That's much better, now we can see much more clearly what type of correlations our variables have
# 
# well, i expected to see a strong correlation between imports and exports, but having it almost one is super interesting on the one hand, and can cause some problems of multicolinearity on the other, so i think we should seperate ways and stick with the Exports.
# 
# moving on, it looks like phone is the strongest correlator. again, this could be an inverse relationship, i.e, the more rich the country is, the more access to technology they have and not the other way around. let's keep that fact in mind.
# 
# what strikes me the most is the freedom aspect. 5 out of 7 indicators reflecting the economical freedom are being high at the top, and while i can think of some reasons why again that's maybe only an inverse causality, I also think it's reasonable enough to argue that that's not the case.

# ### Let's keep going on this direction and have a look at the correlations with more precise measures

# In[ ]:


df = df.drop(['Imports'], axis=1)


# In[ ]:


top_corrs = df.corr().nlargest(8, 'GDP').index
sns.pairplot(df[top_corrs])


# Once again, we can observe the clear relationship with our variables but diving a little bit deeper into how it actually looks like. all these variables are positively correlated and we can definetely speculate why that's the case, but that requires more than just a few words so let's focus on the statistics.
# Another thing to add, we can see that there is one outlier over there, but we'll handle it later as i've mentioned.
# 
# For now, let's further examine the Region variable

# In[ ]:


fig = plt.figure(figsize = (15,10))
plt.subplot(2,3,1)
sns.barplot(df['GDP'], df['Region'],palette='BrBG',ci = None )
plt.subplot(2,3,2)
sns.barplot(df['Exports'], df['Region'],palette='BrBG', ci = None)
plt.subplot(2, 3, 3)
sns.barplot(df['Net migration'], df['Region'], palette='BrBG', ci = None)
plt.subplot(2, 3, 4)
sns.barplot(df['Pop. Density'], df['Region'],palette='BrBG', ci = None)
plt.subplot(2, 3, 5)
sns.barplot(df['Deathrate'], df['Region'],palette='BrBG', ci = None)
plt.subplot(2, 3, 6)
sns.countplot(y = df['Region'],palette='BrBG')
plt.tight_layout()


# ### we can learn a lot from the plots above to be honest. 
# first, the deathrate in sub-saharan african countries is substantialy higher than any other country, and this region also leading the way in teems of number of countries, but the density of the population isn't big.
# second, and not surprising, western europe and north america have the highest GDP, what can explain the higher net migration if you ask me.
# 
# what about the religion aspect?

# In[ ]:


fig = plt.figure(figsize = (18,10))
plt.subplot(2,4,1)
sns.barplot(df['GDP'], df['Main Religion'],palette='BrBG',ci = None)
plt.subplot(2,4,2)
sns.barplot(df['Exports'], df['Main Religion'],palette='BrBG', ci = None)
plt.subplot(2, 4, 3)
sns.barplot(df['Net migration'], df['Main Religion'], palette='BrBG', ci = None)
plt.subplot(2, 4, 5)
sns.barplot(df['Pop. Density'], df['Main Religion'],palette='BrBG', ci = None)
plt.subplot(2, 4, 6)
sns.barplot(df['Deathrate'], df['Main Religion'],palette='BrBG', ci = None)
plt.subplot(2, 4, 7)
sns.countplot(y = df['Main Religion'],palette='BrBG')
plt.tight_layout()


# Not much to learn in my opinion as we got the majority of the countries christian and muslim, with a few other ones that make it difficult to have a clear insight.

# let's take a look at some outliers now

# In[ ]:


GDP_scaled = StandardScaler().fit_transform(df['GDP'][:,np.newaxis])
low_values = GDP_scaled[GDP_scaled[:,0].argsort()][:10]
high_values = GDP_scaled[GDP_scaled[:,0].argsort()][-10:]
print('the lower values of the distribution are:')
print(low_values)
print('the higher values of the distribution are:')
print(high_values)


# seems like we got on potential outlier in the right side of the distribution, let's check who that is and see what we can do with it

# In[ ]:


df[df['GDP'].values == df.GDP.max()]


# So, Luxembourg is our outlier, and rightly so. it's one of the richest and developed countries in the world, with a very small number of people. I don't think this one outlier is going to cause any problem going through, so Luxemburg is here to stay.

# ## a few more feature engineering before diving deep to the models

# In[ ]:


df_final = pd.get_dummies(df, columns=['Region', 'Main Religion', 'Climate'])


# One final thing to do - have a closer look at our star - the GDP
# in order to be able to fit the regressions properly, we got some assumptions to make:
# 
# * Normality - We'll have a look at the distribution of our variable and see if it's normal.
# 
# * Homoscedasticity - Basically it means that for every "point" in the dataset(X's) theres an equality of variance among the Y's. from the scatter plots we've plotted earlier it seems reasonable to assume that's the case.
# 
# * Linearity - as we've observed from the plots above, there's seem to be a linear correlation between our variables so we can check this assumption with a degree of confidence.
# 
# * Absence of correlated errors - basically thats why we got rid of 'Imports'.

# In[ ]:


df_final.GDP.describe()


# In[ ]:


plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sns.distplot(df_final['GDP'], fit=norm)
plt.subplot(1,2,2)
res = stats.probplot(df_final['GDP'], plot=plt)
plt.tight_layout()


# our target variable seems to be skewed to the right(remember the scaled values from earlier), which means it doesn't fit to the normality assumption, therefore we could have some problems in the future. the normality assumption is quite a strong one and it's important in order to statistically infer any insights for the data.
# with a simple transformation we could probably solve this problem. let's go for it.

# In[ ]:


y = df_final.GDP.values


# In[ ]:


y = np.log(y)


# In[ ]:


plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sns.distplot(y, fit=norm)
plt.subplot(1,2,2)
res = stats.probplot(y, plot=plt)


# ### ## looks much better now. let's start with our models:
# plan:
# split the data into training and validation
# train each set with a different method:
# 1. scaled
# 2. non-scaled
# 
# each method will be tested with specific features and all featuers.
# 
# accuracy will be evaluated by the mean square error value.

# In[ ]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, GridSearchCV , RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LinearRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.svm import SVR
import lightgbm as lgb


# In[ ]:


df_final1 = df_final.copy()
df_final1.columns


# In[ ]:


x_all = df_final1.drop(['Country','GDP'], axis=1)
x_features = df_final1[['Population', 'Pop. Density','Literacy','Infant mortality','Birthrate',
                      'ATMs per 100,000 adults 2013', 'Financial Freedom', 'Business Freedom',
                      'Phones','Net foreign invest', 'Service', 'Industry','Exports', 'Investment Freedom' ]]
x_all_scaled = StandardScaler().fit_transform(x_all)
x_features_scaled = StandardScaler().fit_transform(x_features)


# In[ ]:


x_all_train, x_all_test, y_train, y_test = train_test_split(x_all.values, y, train_size = 0.8)
x_features_train, x_features_test, y_train, y_test = train_test_split(x_features.values, y, train_size = 0.8)
x_all_scaled_train, x_all_scaled_test, y_train, y_test = train_test_split(x_all_scaled, y, train_size = 0.8)
x_features_scaled_train, x_features_scaled_test, y_train, y_test = train_test_split(x_features_scaled, y, train_size = 0.8)


# In[ ]:


models = []
models.append(('Lasso', Lasso()))
models.append(('RandomForest', RandomForestRegressor()))
models.append(('XGB', xgb.XGBRegressor(objective = 'reg:squarederror')))
models.append(('LR', LinearRegression()))
models.append(('SVR', SVR()))
models.append(('Enet',ElasticNet(tol=0.5)))
models.append(('LightGBM',lgb.LGBMRegressor()))


# ## All features, without scaling

# In[ ]:


names = []
results = []
n_fold = 6
for name, model in models:
    kfold = KFold(n_fold, random_state = None)
    cv_results = cross_val_score(model,x_all_train,y_train, scoring = 'neg_mean_squared_error' )
    results.append(cv_results)
    names.append(name)
    model.fit(x_all_train, y_train)
    predictions = model.predict(x_all_test)
    r2_msg = 'And the R2 score is: %f' %(r2_score(predictions, y_test))
    rmse_msg = 'For ' '%s' ' , the mean squared error is: ' '%f(%f)' % (name, cv_results.mean(), cv_results.std())
    print(rmse_msg)
    print(r2_msg)
fig = plt.figure()
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# ## All features, with scaling

# In[ ]:


names = []
results = []
n_fold = 6
for name, model in models:
    kfold = KFold(n_fold, random_state = None)
    cv_results = cross_val_score(model,x_all_scaled_train,y_train, scoring = 'neg_mean_squared_error' )
    results.append(cv_results)
    names.append(name)
    model.fit(x_all_scaled_train, y_train)
    predictions = model.predict(x_all_scaled_test)
    r2_msg = 'And the R2 score is: %f' %(r2_score(predictions, y_test))
    rmse_msg = 'For ' '%s' ' , the mean squared error is: ' '%f(%f)' % (name, cv_results.mean(), cv_results.std())
    print(rmse_msg)
    print(r2_msg)
fig = plt.figure()
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# ## Selected features, without scaling

# In[ ]:


names = []
results = []
n_fold = 6
for name, model in models:
    kfold = KFold(n_fold, random_state = None)
    cv_results = cross_val_score(model,x_features_train,y_train, scoring = 'neg_mean_squared_error' )
    results.append(cv_results)
    names.append(name)
    model.fit(x_features_train, y_train)
    predictions = model.predict(x_features_test)
    r2_msg = 'And the R2 score is: %f' %(r2_score(predictions, y_test))
    rmse_msg = 'For ' '%s' ' , the mean squared error is: ' '%f(%f)' % (name, cv_results.mean(), cv_results.std())
    print(rmse_msg)
    print(r2_msg)
fig = plt.figure()
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# ## Selected features, with scaling

# In[ ]:


names = []
results = []
n_fold = 6
for name, model in models:
    kfold = KFold(n_fold, random_state = None)
    cv_results = (cross_val_score(model,x_features_scaled_train,y_train, scoring = 'neg_mean_squared_error' ))
    results.append(cv_results)
    names.append(name)
    model.fit(x_features_scaled_train, y_train)
    predictions = model.predict(x_features_scaled_test)
    r2_msg = 'And the R2 score is: %f' %(r2_score(predictions, y_test))
    rmse_msg = 'For ' '%s' ' , the mean squared error is: ' '%f(%f)' % (name, cv_results.mean(), cv_results.std())
    print(rmse_msg)
    print(r2_msg)

fig = plt.figure()
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# It appers that the XGB and Random forest had the best results overall(LightGBM also did quite well)
# Let's examine these models further.
# 
# First, we're going to have each model fit to our data
# 
# Second, we'll generate predictions with the model by using our test data
# 
# Third, we'll take a look at the accuracy of the predictions by comparing them with the test data
# 
# finally, we'll perform a grid search and see if we improved the score.

# In[ ]:


model_xgb = xgb.XGBRegressor(objective = 'reg:squarederror')
model_randomforest = RandomForestRegressor()


# In[ ]:


model_randomforest.fit(x_features_scaled_train, y_train)
RF_predictions = model_randomforest.predict(x_features_scaled_test)
print(r2_score(RF_predictions, y_test))
print(mean_squared_error(RF_predictions, y_test))
graph = sns.regplot(RF_predictions, y_test).set(title='Random Forest model predictions', xlabel='Predicted GDP', ylabel='Actual GDP')


# ##### Those scores are pretty good indeed! let's check if we can have any improvement by implementing grid search

# In[ ]:


###performing Search grid search
parameters = { 
                      'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
                                                 }
gridRF = GridSearchCV(model_randomforest,param_grid= parameters, n_jobs = 4, cv=5)


# In[ ]:


gridRF.fit(x_features_scaled_train, y_train)


# In[ ]:


gridRF.best_estimator_


# In[ ]:


gridRF_preds = gridRF.predict(x_features_scaled_test)
print(mean_squared_error(gridRF_preds, y_test))
print(r2_score(gridRF_preds, y_test))
graph = sns.regplot(gridRF_preds, y_test).set(title='Random Forest model predictions', xlabel='Predicted GDP', ylabel='Actual GDP')


# well it looks like our improved model actually did worse. that's a question to another day though. let's move on the the XGB model.

# In[ ]:


model_xgb.fit(x_features_scaled_train, y_train)
xgb_predictions = model_xgb.predict(x_features_scaled_test)
print(mean_squared_error(xgb_predictions, y_test))
print(r2_score(xgb_predictions, y_test))
graph = sns.regplot(xgb_predictions, y_test).set(title='XGB model predictions', xlabel='Predicted GDP', ylabel='Actual GDP')


# Thats much better! i wonder if we can improve that high score. Let's see what our grid search has to say about it.

# In[ ]:


###performing Search grid search
n_estimators = [100,300, 400, 500]
learning_rate = [0.03,0.09, 0.1, 0.13, 0.2]
max_depth = [3,4,5,10]
min_child_weight = [1,2,3,4]
parameters = { 
                      'objective':['reg:linear'],
                      'learning_rate': learning_rate, 
                      'max_depth': max_depth,
                      'min_child_weight': min_child_weight,
                      'silent': [1],
                      'subsample': [0.5, 0.6, 0.8],
                      'n_estimators': n_estimators,
                      'booster': ['gbtree']
                                                 }
gridXGB = GridSearchCV(model_xgb,param_grid= parameters, n_jobs = 4, cv=5)


# In[ ]:


gridXGB.fit(x_features_scaled_train, y_train)


# In[ ]:


gridXGB_preds = gridXGB.predict(x_features_scaled_test)
print(mean_squared_error(gridXGB_preds, y_test))
print(r2_score(gridXGB_preds, y_test))
sns.regplot(gridXGB_preds, y_test)


# Well, this time we actually improved our model! Think we can do even better?
# 
# Lets tune it a bit more!

# In[ ]:


model_xgb_2 = xgb.XGBRegressor(booster = 'gbtree',
 learning_rate = 0.19,
 max_depth= 5,
 min_child_weight= 2,
 n_estimators= 90,
 objective= 'reg:linear',
 silent= 1,
 subsample= 0.6)


# In[ ]:


model_xgb_2.fit(x_features_scaled_train, y_train)
xgb_pred_2 = model_xgb_2.predict(x_features_scaled_test)
print(mean_squared_error(xgb_pred_2, y_test))
print(r2_score(xgb_pred_2, y_test))
graph = sns.regplot(xgb_pred_2, y_test).set(title='XGB model predictions', xlabel='Predicted GDP', ylabel='Actual GDP')


# Alright, seems like we can't get better than this and it's pretty powerful score there to be honest! 
# 
# last thing to do is checking the coefficients of our features, how they actually contribute to the GDP

# In[ ]:


featuers_coefficients = model_xgb_2.feature_importances_.tolist()
feature_names = x_features.columns
for i in range(len(feature_names)):
    coefs = 'The coefficient for the feature %s is: ' '%f' %(feature_names[i], featuers_coefficients[i])
    print(coefs)


# In[ ]:


feats = pd.DataFrame(pd.Series(featuers_coefficients, feature_names).sort_values(ascending=False),columns=['Coefficient'])
feats


# Ok, that seems to be all we needed.
# 
# Future questions will be why we got these results and is the relationships between the variables are correlational only or there's a causality there. but for now, that alone can give us some answers.
# 
# ## Thank you!
