#!/usr/bin/env python
# coding: utf-8

# # World Happiness Report 2019
# 
# - I am a noob in data science and some of the alogrithms that I applied in this dataset may be far from accurate. 
# - I've got curious of what factors shape countries' well-being and  thought this would be a fun project to learn and grow. 

# In[ ]:


import pandas as pd
import numpy as np
open_data = '../input/world-happiness-report-2019/world-happiness-report-2019.csv'
data_set = pd.read_csv(open_data)
data_set.head()


# - I wasn't sure what 'SD of Ladder' meant at first, but I assumed that it is an alternative term for ranking? So instead of 'SD of ladder,' I replaced it as 'ranking.'

# In[ ]:


data_set.rename(columns={'SD of Ladder':'Ranking'}, inplace = True)
data_set.head()


# In[ ]:


total_miss = data_set.isnull().sum()
perc_miss = total_miss/data_set.isnull().count()*100

missing_data = pd.DataFrame({'Total missing':total_miss,
                            '% missing':perc_miss})

missing_data.sort_values(by='Total missing',
                         ascending = False).head(2)


# In[ ]:


data_set.fillna(0, inplace = True)


# In[ ]:


order = data_set.sort_values('Ranking')
order.head()


# In[ ]:


data_set.sort_values('Ranking',ascending = False).head()


# - Netherlands, Turkmenistan, Luxembourg, Finland, and Singapore are reported to be one of the most happiest countries. 
# - Whereas, Congo, Sierra Leone, Mozambique, Dominican Republic, and Liberia ranked low in the world happiness report.

# # Correlation
# - I used the heatmap to see if any of these features are strongly correlated to another. 

# In[ ]:


from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
data_set.corr()


# In[ ]:


corr = data_set.corr()
fig, ax = plt.subplots(figsize=(10, 10))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")


# In[ ]:


data_set.describe()


# In[ ]:


sns.pairplot(data_set)


# # Ranking
# - I am aware of that Ranking does not have a strong correlation with one the attributes(or features) in the heatmap, but I still wanted to see if I could find any features that might affect countries ranking. 
# - I still applied some of the features into linear regression to see if they make major contributions to ranking. 

# # Model Performace 
# 
# - Linear Regression

# In[ ]:


from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# In[ ]:


attributes = ['Negative affect','Social support','Log of GDP\nper capita','Healthy life\nexpectancy','Positive affect',
         'Freedom','Corruption','Generosity']
X = data_set[attributes]
y = data_set.Ranking


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)
print(X_train.shape,y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:


lr =linear_model.LinearRegression()
lr= lr.fit(X_train,y_train)
predicted = cross_val_predict(lr, X_train, y_train, cv=10)


# In[ ]:


print('coefficient:', lr.coef_ )
print('Intercept',lr.intercept_)


# In[ ]:


fig, ax = plt.subplots()
ax.scatter(y_train, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Given')
ax.set_ylabel('Predicted')
plt.show()


# In[ ]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[ ]:



ranking_predict = lr.predict(X_test)
print('R Squared:',r2_score(y_test,ranking_predict))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test,ranking_predict)))
print('score:', lr.score(X_train, y_train)*100) 


# - Feature Selection and Random Forest Classifier.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


# In[ ]:


attributes = ['Negative affect','Social support','Log of GDP\nper capita','Healthy life\nexpectancy','Positive affect',
         'Freedom','Corruption','Generosity']
X = data_set[attributes]
y = data_set.Ranking 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)


# In[ ]:


rfc= RandomForestClassifier(n_estimators=4, random_state=0, n_jobs=-1)
rfc = rfc.fit(X_train, y_train)
for feature in zip(attributes, rfc.feature_importances_):
    print(feature)


# In[ ]:


sfm = SelectFromModel(rfc, threshold=0.1)
sfm.fit(X_train, y_train)


# In[ ]:


for feature_list_index in sfm.get_support(indices=True):
    print(attributes[feature_list_index])


# In[ ]:


selected = data_set[['Healthy life\nexpectancy','Generosity','Social support',
                    'Freedom','Corruption','Negative affect','Log of GDP\nper capita']]
ranking= data_set['Ranking']
selected_train, selected_test, ranking_train, ranking_test = train_test_split(selected, ranking,test_size = 0.3, random_state = 1)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression


# In[ ]:


selected,ranking = make_regression(n_features=6, n_informative=2, random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100)
regr.fit(selected_train, ranking_train)  


# In[ ]:


print(regr.feature_importances_)


# In[ ]:


target_predict = regr.predict(selected_test)
data_set.rename(columns={'Country (region)':'COUNTRY'}, inplace = True)
new_data = data_set
chart = pd.DataFrame(new_data, columns = ['COUNTRY','Ranking'])
chart.head()


# In[ ]:


results = pd.merge(new_data.COUNTRY, new_data.Ranking,left_index= True, right_index=True, how='right')
results = results.reset_index(drop=True)
results = pd.concat([results, pd.Series(target_predict)], axis=1)
results.columns = ['COUNTRY', 'Ranking', 'Ranking Predict']
results = results.set_index('COUNTRY')
results.head()


# In[ ]:


print('Random Forest Results:')
print('R Squared:',r2_score(ranking_test, target_predict))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(ranking_test,target_predict)))
meanabsolute = mean_absolute_error(ranking_test,target_predict)
print('Mean Absolute Error:',meanabsolute )


# # Log of GDP per Capita 
# - Log of GDP seems to be strongly correlated with Healthy life expectancy.

# In[ ]:


health = data_set['Healthy life\nexpectancy']
target = data_set['Log of GDP\nper capita']


# In[ ]:


health = health.values.reshape(-1,1)
predict = cross_val_predict(lr,health,target, cv=10)
fig, ax = plt.subplots()
ax.scatter(target, predict, edgecolors=(0, 0, 0))
ax.plot([target.min(), target.max()], [target.min(), target.max()], 'k--', lw=4)
ax.set_xlabel('Log of GDP per capita')
ax.set_ylabel('Healthy Life Expectancy')
plt.show()


# In[ ]:


health_train, health_test, target_train, target_test = train_test_split(health,target,test_size = 0.2, random_state = 1)


# In[ ]:


linear = lr.fit(health_train,target_train)
target_predict = linear.predict(health_test)
meanabsolute = mean_absolute_error(target_test,target_predict)
print('R Squared:',r2_score(target_test, target_predict))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(target_test,target_predict)))
print('Mean Absolute Error:',meanabsolute )
print('score:',lr.score(health_train,target_train)*100)


# # Conclusion and Map Visualization
# - The most challenging task done in this project was to find the right algorithm for my ranking variables. I wasn't sure if I had to  apply ranking as categorical variables or numerical variables, but I still applied it as classifier and then regressor in hopes to find the right predictive model.
# - Unfortunately, some countries are missing in the report. But you can still check out other countries ranking in the map visualization.
# - I am open to suggestions and comments. Thank you. 
# 

# In[ ]:


import plotly
from plotly.offline import plot
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[ ]:


cia_map = pd.read_csv('../input/2014_world_gdp_with_codes.csv/2014_world_gdp_with_codes.csv')
cia_map.head()


# In[ ]:


new_data = pd.merge(data_set,cia_map)


# In[ ]:


fig = go.Figure(data=go.Choropleth(
    locations = new_data['CODE'],
    z = new_data['Ranking'],
    text = new_data['COUNTRY'],
    colorscale = 'Blues',
    autocolorscale=False,
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_tickprefix = '',
    colorbar_title = 'Ranking',
))

fig.update_layout(
    title_text='World Happiness Report 2019',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ),
    annotations = [dict(
        x=0.55,
        y=0.1,
        xref='paper',
        yref='paper',
        text='Source: <a href="https://www.cia.gov/library/publications/the-world-factbook/fields/2195.html">\
            CIA World Factbook</a>',
        showarrow = False
    )]
)

fig.show()

