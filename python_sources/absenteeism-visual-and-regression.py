#!/usr/bin/env python
# coding: utf-8

# In[102]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[103]:


df = pd.read_csv('../input/MFGEmployees4.csv')


# In[104]:


df.info()


# In[105]:


df.describe()


# In[106]:


df.sample(15)


# In[121]:


def mapper(field, new_field):
    keys = field.unique()
    dicts = dict(zip(keys, range(len(keys))))
    print(dicts)
    df[new_field] = field.map(dicts).astype(int)
    
mapper(df.Gender, 'MappedGender')
mapper(df.City, 'MappedCity')
mapper(df.JobTitle, 'MappedTitle')
mapper(df.DepartmentName, 'MappedDept')
mapper(df.StoreLocation, 'MappedStoreLoc')
mapper(df.Division, 'MappedDivision')


# ## Questions
# 
# what is the average absenteeism per DepartmentName, Division, StoreLocation? Do older employees have more Absent Hours recorded? Are men more often off work compared to women?
# 
# Let's try to find some answers
# 
# 

# In[132]:


# Average absenteeism per Department

plt.figure(figsize=(20, 9))
sns.boxplot("DepartmentName", "AbsentHours", hue="Gender", data=df)
plt.xticks(rotation = 45)
plt.title('Absenteeism per Department');


# In[131]:


# Average Absenteeism per Division

plt.figure(figsize=(20, 9))
sns.boxplot("Division", "AbsentHours", hue="Gender", data=df)
plt.xticks(rotation = 45)
plt.title('Absenteeism per Division');


# In[130]:


# And storeLocation

plt.figure(figsize=(20, 9))
sns.boxplot("StoreLocation", "AbsentHours", hue="Gender", data=df)
plt.title('Absenteeism per store location')
plt.xticks(rotation = 90);


# ## Conclusion
# Yes, there is some difference in absenteeism per division, location and department

# In[116]:


# Are older employees sick more often?

plt.scatter(x = df.Age, y = df.AbsentHours)


# In[117]:


sns.jointplot("Age", "AbsentHours", df, kind='kde');


# In[128]:


df_sample = df.sample(2000) # take a sample of the full data set

data = [
    {
        'y':  df_sample.Age,
        'x': df_sample.LengthService,
        'mode': 'markers',
        'marker': {
            'color': df_sample.MappedDivision,
            'size': df_sample.AbsentHours/10,
            'showscale': True
        },
        "text" :  df_sample.Division   
    }
]
layout = dict(title = 'Lenght of Service and Age. Size of bubbles indicate hours absent. Color indicates division',
              xaxis= dict(title= 'Length of Service',ticklen= 5,zeroline= False),
              yaxis = dict(title = 'Age', ticklen = 5, zeroline = False))
fig = dict(data = data, layout = layout)
py.iplot(fig)


# ## Conclusion
# 
# The question whether older people have more hours of absenteeism, the answer is Yes. There is even a significant Pearson correlation between age and absenthours. (.83). In the case of the stores, which has the most data entries, you see a clear size increase when the age goes up as well (bubble plot)

# In[135]:


# Is there a difference between men and women?
plt.figure(figsize=(15,9))
sns.kdeplot(df.AbsentHours[df.Gender=='M'], label='men', shade=True)
sns.kdeplot(df.AbsentHours[df.Gender=='F'], label='women', shade=True)
plt.xlabel('AbsentHours');


# In[137]:


average_male = df.AbsentHours[df.Gender == 'M'].mean()
average_female = df.AbsentHours[df.Gender == 'F'].mean()
print('Mean of absenthours for male: ', average_male)
print('Mean of absenthours for female: ', average_female)


# ## Conclusion
# males are on average 10 hours less absent compared to women.

# # More visuals before we train an estimator

# In[138]:


colormap = plt.cm.RdBu
plt.figure(figsize=(15,15))
plt.title('Pearson Correlation of Features', y=1.0, size=10)
sns.heatmap(df.corr(),linewidths=0.2,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# In[145]:


import plotly.figure_factory as ff

# prepare data
data = df.loc[:,["AbsentHours", "Age", "LengthService"]]
data["index"] = df.Gender
# scatter matrix
fig = ff.create_scatterplotmatrix(data, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
py.iplot(fig)


# In[140]:


sns.distplot(df.Age)


# In[83]:


sns.distplot(df.LengthService)


# In[84]:


sns.distplot(df.AbsentHours)


# In[85]:


df.Division.unique()


# In[86]:


plt.figure(figsize=(15, 9))
sns.violinplot("Division", "AbsentHours", hue="Gender", data=df, split=True, inner="quartile", 
               palette=["lightpink", "lightblue"])
plt.xticks(rotation = 45);


# In[87]:


plt.figure(figsize=(15,9))
plt.hist(df.Division);


# In[88]:


plt.figure(figsize=(15,9))
plt.hist(df.StoreLocation)
plt.xticks(rotation=90);


# # End of visuals, start of regression

# In[90]:


X = df.drop(['EmployeeNumber', 'Surname', 'GivenName', 'Gender', 'City', 'JobTitle', 'DepartmentName', 'StoreLocation', 'Division', 'BusinessUnit', 'AbsentHours'], axis=1)
y = df['AbsentHours'].values

from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)


# In[91]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[92]:


# Quick death match between a bunge of regressors
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet, Lars
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error


regressors = [DecisionTreeRegressor(), ExtraTreeRegressor(), #LogisticRegression(),
AdaBoostRegressor(), GradientBoostingRegressor(), ExtraTreesRegressor(), RandomForestRegressor(),
Ridge(), Lasso(), LinearRegression(), ElasticNet(), Lars()]

log_cols=["regressors", "MSE"]
log = pd.DataFrame(columns=log_cols)

for rgr in regressors:
    rgr.fit(X_train, y_train)
    name = rgr.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = rgr.predict(X_test)
    mse = np.sqrt(mean_squared_error(y_test, train_predictions))
    print("RMSE: {}".format(mse))
    
    
    log_entry = pd.DataFrame([[name, mse]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)


# In[93]:


sns.set_color_codes("muted")
sns.barplot(x='MSE', y='regressors', data=log, color="g")

plt.xlabel('RMSE')
plt.title('regressor RMSE')
plt.show()


# In[94]:


gb = GradientBoostingRegressor()
ad = AdaBoostRegressor()
et = ExtraTreesRegressor()
rf = RandomForestRegressor()

gb.fit(X_train, y_train)
gb_preds_train = gb.predict(X_train)
gb_preds_test = gb.predict(X_test)

ad.fit(X_train, y_train)
ad_preds_train = ad.predict(X_train)
ad_preds_test = ad.predict(X_test)

et.fit(X_train, y_train)
et_preds_train = et.predict(X_train)
et_preds_test = et.predict(X_test)

rf.fit(X_train, y_train)
rf_preds_train = rf.predict(X_train)
rf_preds_test = rf.predict(X_test)


# In[95]:


base_predictions_train = pd.DataFrame( {'GradientBoost': gb_preds_train.ravel(),
                                        'AdaBoost': ad_preds_train.ravel(),
                                        'ExtraTrees': et_preds_train.ravel(),
                                        'RandForest': rf_preds_train.ravel()
    })

base_predictions_test = pd.DataFrame( {'GradientBoost': gb_preds_test.ravel(),
                                        'AdaBoost': ad_preds_test.ravel(),
                                        'ExtraTrees': et_preds_test.ravel(),
                                        'RandForest': rf_preds_test.ravel()
    })
y_train, y_test = y_train.ravel(), y_test.ravel()
base_predictions_train.head()


# Above dataframe can be used to train a stacked regressor such as XGBoost. I tried it but didn't produce satisfying results.

# In[96]:


# Tune hyper parameters
from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'alpha': [0.1, 0.5, 0.9],
                     'learning_rate': [0.001, 0.01, 0.1, 1],
                     'n_estimators': [50, 100, 150]}]
                     #'kernel': ['rbf']}]

scores = ['neg_mean_squared_error', 'r2']

for score in scores:
    print(score)
    regr = GridSearchCV(GradientBoostingRegressor(), tuned_parameters, cv=5, verbose=1, n_jobs=-1,
                       scoring='%s' % score)
    regr.fit(X_train, y_train)
    print(regr.best_params_)

    # Not useful every time:
    means = regr.cv_results_['mean_test_score']
    stds = regr.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, regr.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))


# In[97]:


# Returns the coefficient of determination R^2 of the prediction.
regr.score(X_test, y_test)


# In[ ]:


yhat = regr.predict(X_test)

plt.figure(figsize=(15,9))
plt.plot(yhat[:25])
plt.plot(y_test[:25])


# # Final conclusion
# 
# This is generally an easy exercise but it is really interesting to see how some feats relate to absenteeism. We saw that women tend to have more absent hours and also age is an important measure for that. We were able to train a regressor with a reasonable R-squared. Overall, happy with the results
