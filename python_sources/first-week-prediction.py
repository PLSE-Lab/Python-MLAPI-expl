#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
# from pandas_profiling import ProfileReport


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Import Data

# In[ ]:


train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# ## Exploratory Data Analysis

# In[ ]:


# train_profile = ProfileReport(train, title='Pandas Profiling Report', html={'style':{'full_width':True}})
# train_profile


# In[ ]:


# test_profile = ProfileReport(test, title='Pandas Profiling Report', html={'style':{'full_width':True}})
# test_profile


# In[ ]:


from plotly.offline import iplot
from plotly import tools
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as py
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = "plotly_dark"
py.init_notebook_mode(connected=True)


# ### Disease spread over the countries

# In[ ]:


temp = train.groupby(['Date', 'Country/Region'])['ConfirmedCases'].sum().reset_index()
temp['Date'] = pd.to_datetime(temp['Date']).dt.strftime('%m/%d/%Y')
temp['size'] = temp['ConfirmedCases'].pow(0.3) * 3.5

fig = px.scatter_geo(temp, locations="Country/Region", locationmode='country names', 
                     color="ConfirmedCases", size='size', hover_name="Country/Region", 
                     range_color=[1,100],
                     projection="natural earth", animation_frame="Date", 
                     title='COVID-19: Cases Over Time', color_continuous_scale="greens")
fig.show()


# ### Confirmed cases over time

# In[ ]:


grouped = train.groupby('Date')['Date', 'ConfirmedCases', 'Fatalities'].sum().reset_index()

fig = px.line(grouped, x="Date", y="ConfirmedCases", 
              title="Worldwide Confirmed Cases Over Time")
fig.show()

fig = px.line(grouped, x="Date", y="ConfirmedCases", 
              title="Worldwide Confirmed Cases (Logarithmic Scale) Over Time", 
              log_y=True)
fig.show()


# In[ ]:


latest_grouped = train.groupby('Country/Region')['ConfirmedCases', 'Fatalities'].sum().reset_index()


# In[ ]:


fig = px.bar(latest_grouped.sort_values('ConfirmedCases', ascending=False)[:20][::-1], 
             x='ConfirmedCases', y='Country/Region',
             title='Confirmed Cases Worldwide', text='ConfirmedCases', height=1000, orientation='h')
fig.show()


# ### Take a look at Europe

# In[ ]:


europe = list(['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland',
               'Italy', 'Latvia','Luxembourg','Lithuania','Malta','Norway','Netherlands','Poland','Portugal','Romania','Slovakia','Slovenia',
               'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus',
               'Albania', 'Bosnia and Herzegovina', 'Kosovo', 'Moldova', 'Montenegro', 'North Macedonia'])
europe_grouped_latest = latest_grouped[latest_grouped['Country/Region'].isin(europe)]


# In[ ]:


temp = train[train['Country/Region'].isin(europe)]
temp = temp.groupby(['Date', 'Country/Region'])['ConfirmedCases'].sum().reset_index()
temp['Date'] = pd.to_datetime(temp['Date']).dt.strftime('%m/%d/%Y')
temp['size'] = temp['ConfirmedCases'].pow(0.3) * 3.5

fig = px.scatter_geo(temp, locations="Country/Region", locationmode='country names', 
                     color="ConfirmedCases", size='size', hover_name="Country/Region", 
                     range_color=[1,100],scope='europe',
                     projection="natural earth", animation_frame="Date", 
                     title='COVID-19: Cases Over Time', color_continuous_scale='Cividis_r')
fig.show()


# In[ ]:


fig = px.bar(europe_grouped_latest.sort_values('ConfirmedCases', ascending=False)[:10][::-1], 
             x='ConfirmedCases', y='Country/Region', color_discrete_sequence=['#84DCC6'],
             title='Confirmed Cases in Europe', text='ConfirmedCases', orientation='h')
fig.show()


# ### take a look at US

# In[ ]:


usa = train[train['Country/Region'] == "US"]
usa_latest = usa[usa['Date'] == max(usa['Date'])]
usa_latest = usa_latest.groupby('Province/State')['ConfirmedCases', 'Fatalities'].max().reset_index()
fig = px.bar(usa_latest.sort_values('ConfirmedCases', ascending=False)[:10][::-1], 
             x='ConfirmedCases', y='Province/State', color_discrete_sequence=['#D63230'],
             title='Confirmed Cases in USA', text='ConfirmedCases', orientation='h')
fig.show()


# In[ ]:


usa = train[train['Country/Region'] == "US"]
usa_latest = usa[usa['Date'] == max(usa['Date'])]
usa_latest = usa_latest.groupby(['Province/State','Lat','Long'])['ConfirmedCases', 'Fatalities'].max().reset_index()
fig = go.Figure()
limits = [(0,10),(10,50),(50,100),(100,500),(500,1000)]
colors = ["royalblue","crimson","lightseagreen","orange","lightgrey"]

for i in range(len(limits)):
    lim = limits[i]
    range_usa = usa_latest[usa_latest['ConfirmedCases'].between(lim[0], lim[1])]
    fig.add_trace(go.Scattergeo(
        locationmode = 'USA-states',
        lon = range_usa['Long'],
        lat = range_usa['Lat'],
        text = range_usa['Province/State'],
        marker = dict(
            size = range_usa['ConfirmedCases'],
            color = colors[i],
            line_color='rgb(40,40,40)',
            line_width=0.5,
            sizemode='area'
        ),
        name = '{0} - {1}'.format(lim[0],lim[1])))

fig.update_layout(
        title_text = 'COVID19 in the US',
        showlegend = True,
        geo = dict(
            scope = 'usa',
            landcolor = 'rgb(217, 217, 217)',
        )
    )

fig.show()


# ### take a look at China

# In[ ]:


usa = train[train['Country/Region'] == "China"]
usa_latest = usa[usa['Date'] == max(usa['Date'])]
usa_latest = usa_latest.groupby('Province/State')['ConfirmedCases', 'Fatalities'].max().reset_index()
fig = px.bar(usa_latest.sort_values('ConfirmedCases', ascending=False)[:10][::-1], 
             x='ConfirmedCases', y='Province/State', color_discrete_sequence=['#D63230'],
             title='Confirmed Cases in USA', text='ConfirmedCases', orientation='h')
fig.show()


# In[ ]:





# ## Encoding Categorical Data
# 
# 1. Province Encoding
# 2. Country Encoding
# 3. Date Encoding
# 4. Extra Dataset
# 5. Missing Value Imputation

# ### Province Encoding
# Province is a string-type object in the dataset. To take advantage of them, we convert Province to a numeric index as shown below. `province_encoded` collects all states in the training data. Specially, `nan` cells indicate to index `0` avoiding missing data.

# In[ ]:


province_encoded = {state:index for index, state in enumerate(train['Province/State'].unique())}


# In[ ]:


train['province_encoded'] = train['Province/State'].apply(lambda x: province_encoded[x])
train.head()


# ### Country Encoding

# In[ ]:


country_encoded = dict(enumerate(train['Country/Region'].unique()))
country_encoded = dict(map(reversed, country_encoded.items()))


# In[ ]:


train['country_encoded'] = train['Country/Region'].apply(lambda x: country_encoded[x])
train.head()


# ### Date Encoding: sequential timestamp (poor design)

# In[ ]:


from datetime import datetime
import time


# In[ ]:


# date_encoded = {}
# for s in train['Date'].unique():
#     date_encoded[s] = time.mktime(datetime.strptime(s, "%Y-%m-%d").timetuple())


# In[ ]:


# train['date_encoded'] = train['Date'].apply(lambda x: date_encoded[x])
# train['date_encoded'] = (train['date_encoded'] - train['date_encoded'].mean()) / train['date_encoded'].std()
# train.head()


# ### Date encoding: convert `y-m-d`  to Month.and Day.

# In[ ]:


train['Mon'] = train['Date'].apply(lambda x: int(x.split('-')[1]))
train['Day'] = train['Date'].apply(lambda x: int(x.split('-')[2]))


# ### Date encoding: enhance by serial fetures (poor design)

# In[ ]:


train['serial'] = train['Mon'] * 30 + train['Day']
train.head()


# In[ ]:


train['serial'] = train['serial'] - train['serial'].min()


# In[ ]:


train.describe()


# ### Extra Dataset

# In[ ]:


gdp2020 = pd.read_csv('/kaggle/input/gdp2020/GDP2020.csv')
population2020 = pd.read_csv('/kaggle/input/population2020/population2020.csv')


# In[ ]:


gdp2020 = gdp2020.rename(columns={"rank":"rank_gdp"})
gdp2020_numeric_list = [list(gdp2020)[0]] + list(gdp2020)[2:-1]
gdp2020.head()


# #### Redefine all mismatch Country 

# In[ ]:


map_state = {'US':'United States', 
             'Korea, South':'South Korea',
             'Cote d\'Ivoire':'Ivory Coast',
             'Czechia':'Czech Republic',
             'Eswatini':'Swaziland',
             'Holy See':'Vatican City',
             'Jersey':'United Kingdom',
             'North Macedonia':'Macedonia',
             'Taiwan*':'Taiwan',
             'occupied Palestinian territory':'Palestine'
            }
map_state_rev = {v: k for k, v in map_state.items()}


# In[ ]:


population2020['name'] = population2020['name'].apply(lambda x: map_state_rev[x] if x in map_state_rev else x)
gdp2020['country'] = gdp2020['country'].apply(lambda x: map_state_rev[x] if x in map_state_rev else x)


# #### Losing Country in Population

# In[ ]:


set(train['Country/Region']) - set(population2020['name'])


# #### Losing Country in GDP2020

# In[ ]:


set(train['Country/Region']) - set(gdp2020['country'])


# In[ ]:


population2020 = population2020.rename(columns={"rank":"rank_pop"})
population2020_numeric_list = [list(population2020)[0]] + list(gdp2020)[2:]
population2020.head()


# In[ ]:


train = pd.merge(train, population2020, how='left', left_on = 'Country/Region', right_on = 'name')
train = pd.merge(train, gdp2020, how='left', left_on = 'Country/Region', right_on = 'country')


# ### Drop Nan cells or repalce them to more suitable values

# In[ ]:


train.isnull().sum()


# #### Set extra attributes to zero

# In[ ]:


train = train.fillna(0)


# Which country has `nan` coordinate ?

# In[ ]:


train['Country/Region'][train.isnull()['Lat'] | train.isnull()['Long']].unique()


# Find out coordinate in Aruba from extra information

# In[ ]:


# train.loc[:,'Lat'][train['Country/Region']=='Aruba'] = -69.9683
# train.loc[:,'Long'][train['Country/Region']=='Aruba'] = 12.5211


# ## Generate the numeric input for training

# In[ ]:


# numeric_features_X = ['Lat','Long', 'province_encoded' ,'country_encoded','Mon','Day']
numeric_features_X = ['Lat','Long', 'province_encoded' ,'country_encoded','Mon','Day'] + population2020_numeric_list + gdp2020_numeric_list
numeric_features_Y = ['ConfirmedCases', 'Fatalities']
train_numeric_X = train[numeric_features_X]
train_numeric_Y = train[numeric_features_Y]


# ## Generate the numeric input for testing 

# In[ ]:


test['province_encoded'] = test['Province/State'].apply(lambda x: province_encoded[x] if x in province_encoded else max(province_encoded.values())+1)


# In[ ]:


test['country_encoded'] = test['Country/Region'].apply(lambda x: country_encoded[x] if x in country_encoded else max(country_encoded.values())+1)


# In[ ]:


test['Mon'] = test['Date'].apply(lambda x: int(x.split('-')[1]))
test['Day'] = test['Date'].apply(lambda x: int(x.split('-')[2]))


# In[ ]:


test['serial'] = test['Mon'] * 30 + test['Day']
test['serial'] = test['serial'] - test['serial'].min()


# In[ ]:


test = pd.merge(test, population2020, how='left', left_on = 'Country/Region', right_on = 'name')
test = pd.merge(test, gdp2020, how='left', left_on = 'Country/Region', right_on = 'country')


# In[ ]:


# date_encoded = {}
# for s in test['Date'].unique():
#     date_encoded[s] = time.mktime(datetime.strptime(s, "%Y-%m-%d").timetuple())
# test['date_encoded'] = test['Date'].apply(lambda x: date_encoded[x])
# test['date_encoded'] = (test['date_encoded'] - test['date_encoded'].mean()) / test['date_encoded'].std()
# test.head()


# In[ ]:


# test.loc[:,'Lat'][test['Country/Region']=='Aruba'] = -69.9683
# test.loc[:,'Long'][test['Country/Region']=='Aruba'] = 12.5211


# In[ ]:


test_numeric_X = test[numeric_features_X]
test_numeric_X.isnull().sum()


# In[ ]:


test_numeric_X = test_numeric_X.fillna(-1)


# ## Model
# #### Single Model 
# 1. Linear Regression
# 2. SVM Regression
# 3. KNN 
# 
# #### Ensemble 
# 1. Random Forest
# 2. Adaboost 
# 
# #### SIR Model

# ### Linear Regression

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# In[ ]:


pipeline = Pipeline([('scaler', StandardScaler()), ('estimator', LinearRegression())])
pipeline.fit(train_numeric_X, train_numeric_Y)


# In[ ]:


predicted = pipeline.predict(test_numeric_X)


# In[ ]:


# submission = np.vstack((test['ForecastId'], predicted[:,0],predicted[:,1])).T
# submission.astype(np.int32)
# df = pd.DataFrame(data=submission, columns=['ForecastId','ConfirmedCases','Fatalities'])
# df.to_csv('LR_submission.csv', index=False)


# In[ ]:


predicted_x = pipeline.predict(train_numeric_X)
plt.scatter(train_numeric_Y['ConfirmedCases'], train_numeric_Y['Fatalities'],  color='gray', label='sample')
plt.plot(predicted_x[:,0], predicted_x[:,1], color='red', linewidth=2, label='pred')
plt.title('Regression Model Result',fontsize=20)
plt.xlabel('ConfirmedCases',fontsize=15)
plt.ylabel('Fatalities',fontsize=15)
plt.legend()
plt.show()


# ### SVR

# In[ ]:


from sklearn.svm import SVR


# In[ ]:


pipeline = Pipeline([('scaler', StandardScaler()), ('estimator', SVR())])
pipeline.fit(train_numeric_X, train_numeric_Y.values[:,0])
pipeline2 = Pipeline([('scaler', StandardScaler()), ('estimator', SVR())])
pipeline2.fit(train_numeric_X, train_numeric_Y.values[:,1])
discovered, fatal = pipeline.predict(test_numeric_X), pipeline2.predict(test_numeric_X)


# In[ ]:


# submission = np.vstack((test['ForecastId'], discovered, fatal)).T
# submission = submission.astype(np.int32)
# df = pd.DataFrame(data=submission, columns=['ForecastId','ConfirmedCases','Fatalities'])
# df.to_csv('SVR_submission.csv', index=False)
# df.to_csv('submission.csv', index=False)


# In[ ]:


predicted_x1 = pipeline.predict(train_numeric_X)
predicted_x2 = pipeline2.predict(train_numeric_X)

plt.scatter(train_numeric_Y['ConfirmedCases'], train_numeric_Y['Fatalities'],  color='gray', label='sample')
plt.plot(predicted_x1, predicted_x2, color='red', linewidth=2, label='pred')
plt.title('SVR Model Result',fontsize=20)
plt.xlabel('ConfirmedCases',fontsize=15)
plt.ylabel('Fatalities',fontsize=15)
plt.legend()
plt.show()


# ### KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


pipeline = Pipeline([('scaler', StandardScaler()), ('estimator', KNeighborsClassifier(n_jobs=4))])
pipeline.fit(train_numeric_X, train_numeric_Y)


# In[ ]:


predicted_x = pipeline.predict(train_numeric_X)
plt.scatter(train_numeric_Y['ConfirmedCases'], train_numeric_Y['Fatalities'],  color='gray', label='sample')
plt.scatter(predicted_x[:,0], predicted_x[:,1], color='red', linewidth=2, label='pred')
plt.title('KNN Model Result',fontsize=20)
plt.xlabel('ConfirmedCases',fontsize=15)
plt.ylabel('Fatalities',fontsize=15)
plt.legend()
plt.show()


# ## Ensemble

# ### Bagging: Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


RF_model = RandomForestClassifier(n_estimators=50,n_jobs=4,verbose=True)
RF_model.fit(train_numeric_X, train_numeric_Y)


# In[ ]:


# predicted = RF_model.predict(test_numeric_X)
# submission = np.vstack((test['ForecastId'], predicted[:,0],predicted[:,1])).T
# submission = submission.astype(np.int32)
# df = pd.DataFrame(data=submission, columns=['ForecastId','ConfirmedCases','Fatalities'])
# df.to_csv('RF_submission.csv', index=False)
# df.to_csv('submission.csv', index=False)


# In[ ]:


predicted_x = RF_model.predict(train_numeric_X)
plt.scatter(train_numeric_Y['ConfirmedCases'], train_numeric_Y['Fatalities'],  color='gray', label='sample',s=25)
plt.scatter(predicted_x[:,0], predicted_x[:,1], color='red', label='pred',alpha=.4, s=10)
plt.title('KNN Model Result',fontsize=20)
plt.xlabel('ConfirmedCases',fontsize=15)
plt.ylabel('Fatalities',fontsize=15)
plt.legend()
plt.show()


# ### Boosting: Adaboost

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


adaboost_model_for_ConfirmedCases = AdaBoostClassifier(n_estimators=15)
adaboost_model_for_ConfirmedCases.fit(train_numeric_X, train_numeric_Y[numeric_features_Y[0]])
adaboost_model_for_Fatalities = AdaBoostClassifier(n_estimators=15)
adaboost_model_for_Fatalities.fit(train_numeric_X, train_numeric_Y[numeric_features_Y[1]])


# In[ ]:


# predicted = adaboost_model_for_ConfirmedCases.predict(test_numeric_X)
# predicted2 = adaboost_model_for_Fatalities.predict(test_numeric_X)
# submission = np.vstack((test['ForecastId'], predicted,predicted2)).T
# submission = submission.astype(np.int32)
# df = pd.DataFrame(data=submission, columns=['ForecastId','ConfirmedCases','Fatalities'])
# df.to_csv('Adaboost_submission.csv', index=False)
# df.to_csv('submission.csv', index=False)


# In[ ]:


predicted_x1 = adaboost_model_for_ConfirmedCases.predict(train_numeric_X)
predicted_x2 = adaboost_model_for_Fatalities.predict(train_numeric_X)

plt.scatter(train_numeric_Y['ConfirmedCases'], train_numeric_Y['Fatalities'],  color='gray', label='sample',s=25)
plt.scatter(predicted_x1,predicted_x2, color='red', label='pred',alpha=.4, s=10)
plt.title('Adaboost Model Result',fontsize=20)
plt.xlabel('ConfirmedCases',fontsize=15)
plt.ylabel('Fatalities',fontsize=15)
plt.legend()
plt.show()


# ### Stacking

# In[ ]:


# from sklearn.ensemble import StackingClassifier


# In[ ]:


# estimators = [('rf',RF_model ), ('ada', adaboost_model_for_ConfirmedCases)]
# stacking_model_for_ConfirmedCases = StackingClassifier(estimators=estimators, n_jobs=4)
# stacking_model_for_ConfirmedCases.fit(train_numeric_X, train_numeric_Y[numeric_features_Y[0]])


# In[ ]:


# stacking_model_for_Fatalities = StackingClassifier(estimators=estimators, n_jobs=4)
# stacking_model_for_Fatalities.fit(train_numeric_X, train_numeric_Y[numeric_features_Y[1]])


# In[ ]:


# predicted = stacking_model_for_ConfirmedCases.predict(test_numeric_X)
# predicted2 = stacking_model_for_Fatalities.predict(test_numeric_X)

# submission = np.vstack((test['ForecastId'], predicted,predicted2)).T
# submission = submission.astype(np.int32)

# df = pd.DataFrame(data=submission, columns=['ForecastId','ConfirmedCases','Fatalities'])
# df.to_csv('stacking_submission.csv', index=False)
# df.to_csv('submission.csv', index=False)


# ### Basic Model Comparasion

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from mlxtend.classifier import StackingCVClassifier


# In[ ]:


clf1 = KNeighborsClassifier(n_neighbors=100)
clf2 = RandomForestClassifier(n_estimators=10)
clf3 = GaussianNB()
# Logit will be used for stacking
lr = LogisticRegression(solver='lbfgs')
# sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr, use_probas=True, cv=3)
sclf = StackingCVClassifier(classifiers=[clf1, clf2], meta_classifier=lr, use_probas=True, cv=3)


# Do CV
for clf, label in zip([clf1, clf2, clf3, sclf], 
                      ['KNN', 
                       'Random Forest', 
                       'Naive Bayes',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, train_numeric_X.values, train_numeric_Y[numeric_features_Y[0]].values, cv=3, scoring='neg_mean_squared_log_error')
    print("Avg_rmse: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


# - KNN: -6.54
# - Random Forest: -6.90
# - Naive Bayes: -25.89
# - StackingClassifier: -4.76

# ### After Model Comparing, here provide an optimal result 

# - KNN attains the better performance than others w.r.t. Fatalities
# - RF attains the better performance than others w.r.t. ConfirmedCases

# In[ ]:


# clf1 = KNeighborsClassifier(n_neighbors=100)
# clf1.fit(train_numeric_X.values, train_numeric_Y[numeric_features_Y[1]])
# predicted2 = clf1.predict(test_numeric_X)

# clf2 = RandomForestClassifier(n_estimators=10)
# clf2.fit(train_numeric_X.values, train_numeric_Y[numeric_features_Y[0]])
# predicted = clf2.predict(test_numeric_X)

# submission = np.vstack((test['ForecastId'], predicted,predicted2)).T
# submission = submission.astype(np.int32)
# df = pd.DataFrame(data=submission, columns=['ForecastId','ConfirmedCases','Fatalities'])
# df.to_csv('opt_submission.csv', index=False)
# df.to_csv('submission.csv', index=False)


# ### SIR Model (Not yet)

# ## Evaluation

# In[ ]:


train_y_pred = RF_model.predict(train_numeric_X)


# In[ ]:


# train_y_pred2 = clf2.predict(train_numeric_X)
# train_y_pred =  np.stack((train_y_pred, train_y_pred2), axis=-1)


# #### Actual Value v.s. Predicted Results

# In[ ]:


plt.figure(figsize=(12,8))
plt.hist([train_numeric_Y['ConfirmedCases'],train_y_pred[:,0]],bins=100, range=(1,100), label=['ConfirmedCases_actual','ConfirmedCases_pred'],alpha=0.75)
plt.title('ConfirmedCases Comparison',fontsize=20)
plt.xlabel('sample',fontsize=20)
plt.ylabel('match',fontsize=20)
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
plt.hist([train_numeric_Y['Fatalities'],train_y_pred[:,1]],bins=100, range=(1,100), label=['Fatalities_actual','Fatalities_pred'],alpha=0.75)
plt.title('Fatalities Comparison',fontsize=20)
plt.xlabel('sample',fontsize=20)
plt.ylabel('match',fontsize=20)
plt.legend()
plt.show()


# #### Root Mean Square Error
# 
# > Submissions are evaluated using the column-wise root mean squared logarithmic error.

# In[ ]:


error = np.sqrt((train_y_pred - train_numeric_Y)**2)
error = error.cumsum()


# In[ ]:


fig,ax = plt.subplots()
 
plt.xlabel('sample')
plt.ylabel('error')
plt.subplot(2, 1, 1)
plt.plot(range(len(error)), error['ConfirmedCases'], "x-",label="ConfirmedCases",color='orange')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(range(len(error)), error['Fatalities'], "+-", label="Fatalities")
plt.legend()

plt.show()


# In[ ]:


from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(train_numeric_Y, train_y_pred , squared=False)
rmse


# ### Correlation Visualization

# #### Pearson

# In[ ]:


corr = train[numeric_features_X+numeric_features_Y].corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
with sns.axes_style("white"):
    # Draw the heatmap with the mask and correct aspect ratio
    f, ax = plt.subplots(figsize=(15, 12))
    ax = sns.heatmap(corr, mask=mask,annot=True,cmap="YlGnBu",vmax=.3, square=True, linewidths=.4)
plt.show()


# #### Spearman

# In[ ]:


corr = train[numeric_features_X+numeric_features_Y].corr(method='spearman')
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
with sns.axes_style("white"):
    # Draw the heatmap with the mask and correct aspect ratio
    f, ax = plt.subplots(figsize=(15, 12))
    ax = sns.heatmap(corr, mask=mask,annot=True,cmap="YlGnBu",vmax=.3, square=True, linewidths=.4)
plt.show()


# #### Kendall

# In[ ]:


corr = train[numeric_features_X+numeric_features_Y].corr(method='kendall')
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
with sns.axes_style("white"):
    # Draw the heatmap with the mask and correct aspect ratio
    f, ax = plt.subplots(figsize=(15, 12))
    ax = sns.heatmap(corr, mask=mask,annot=True,cmap="YlGnBu",vmax=.3, square=True, linewidths=.4)
plt.show()


# #### Weights

# Parameter weights corresponding to `'Lat','Long', 'province_encoded' ,'country_encoded','Mon','Day'`

# In[ ]:


RF_model.feature_importances_


# In[ ]:


plt.bar(range(len(numeric_features_X)), RF_model.feature_importances_, tick_label=numeric_features_X)
plt.xlabel('feature')
plt.ylabel('weight')
plt.xticks(rotation=90)
plt.show()


# #### Scatter Data points 

# In[ ]:


f,ax = plt.subplots()
ax.scatter(train_numeric_Y['ConfirmedCases'], train_y_pred[:,0])
ax.scatter(train_numeric_Y['Fatalities'], train_y_pred[:,1])

plt.show()


# ### Look into the number of decision tree composed of RF

# In[ ]:


clf1 = RandomForestClassifier(n_estimators=1,n_jobs=4)
clf3 = RandomForestClassifier(n_estimators=3,n_jobs=4)
clf5 = RandomForestClassifier(n_estimators=5,n_jobs=4)
clf10 = RandomForestClassifier(n_estimators=10,n_jobs=4)
clf50 = RandomForestClassifier(n_estimators=50,n_jobs=4)


# In[ ]:


clf1.fit(train_numeric_X, train_numeric_Y)
clf3.fit(train_numeric_X, train_numeric_Y)
clf5.fit(train_numeric_X, train_numeric_Y)
clf10.fit(train_numeric_X, train_numeric_Y)
clf50.fit(train_numeric_X, train_numeric_Y)


# In[ ]:


predicted1 = clf1.predict(train_numeric_X)
predicted3 = clf3.predict(train_numeric_X)
predicted5 = clf5.predict(train_numeric_X)
predicted10 = clf10.predict(train_numeric_X)
predicted50 = clf50.predict(train_numeric_X)


# In[ ]:


a = np.sum((predicted1) - (train_numeric_Y))**2 / len(predicted1)
b = np.sum((predicted3) - (train_numeric_Y))**2 / len(predicted3)
c = np.sum((predicted5) - (train_numeric_Y))**2 / len(predicted5)
d = np.sum((predicted10) - (train_numeric_Y))**2 / len(predicted10)
e = np.sum((predicted50) - (train_numeric_Y))**2 / len(predicted50)


# In[ ]:


dt_nums = [1,3,5,10,50]
plt.figure(figsize=(15,10))

plt.subplot(221)
plt.title('Decision Tree Number & MSE \n of ConfirmedCases',fontsize=20)
plt.plot(range(len(dt_nums)), [a['ConfirmedCases'],b['ConfirmedCases'],c['ConfirmedCases'],d['ConfirmedCases'],e['ConfirmedCases']],
         label='ConfirmedCases')
plt.xlabel('decision tree numbers')
plt.ylabel('mse')
plt.xticks(range(len(dt_nums)),dt_nums)
plt.legend()

plt.subplot(222)
plt.title('Decision Tree Number & MSE \n of Fatalities',fontsize=20)
plt.plot(range(len(dt_nums)), [a['Fatalities'],b['Fatalities'],c['Fatalities'],d['Fatalities'],e['Fatalities']],
         label='Fatalities',color='y')
plt.xlabel('decision tree numbers')
plt.ylabel('mse')
plt.xticks(range(len(dt_nums)),dt_nums)
plt.legend()

plt.show()


# Above diagram demostrated that with about 5 decision tree, RF had been enough good to fit in our dataset 

# ### Look into the depth of decision tree composed of RF - Avoiding Overfitting

# In[ ]:


clf1 = RandomForestClassifier(n_estimators=10,n_jobs=4,max_depth=1)
clf2 = RandomForestClassifier(n_estimators=10,n_jobs=4,max_depth=2)
clf3 = RandomForestClassifier(n_estimators=10,n_jobs=4,max_depth=3)
clf4 = RandomForestClassifier(n_estimators=10,n_jobs=4,max_depth=4)
clf5 = RandomForestClassifier(n_estimators=10,n_jobs=4,max_depth=5)
clf10 = RandomForestClassifier(n_estimators=10,n_jobs=4,max_depth=10)


# In[ ]:


clf1.fit(train_numeric_X, train_numeric_Y)
clf2.fit(train_numeric_X, train_numeric_Y)
clf3.fit(train_numeric_X, train_numeric_Y)
clf4.fit(train_numeric_X, train_numeric_Y)
clf5.fit(train_numeric_X, train_numeric_Y)
clf10.fit(train_numeric_X, train_numeric_Y)


# In[ ]:


predicted1 = clf1.predict(train_numeric_X)
predicted2 = clf2.predict(train_numeric_X)
predicted3 = clf3.predict(train_numeric_X)
predicted4 = clf4.predict(train_numeric_X)
predicted5 = clf5.predict(train_numeric_X)
predicted10 = clf10.predict(train_numeric_X)


# In[ ]:


a = np.sum((predicted1) - (train_numeric_Y))**2 / len(predicted1)
b = np.sum((predicted2) - (train_numeric_Y))**2 / len(predicted2)
c = np.sum((predicted3) - (train_numeric_Y))**2 / len(predicted3)
d = np.sum((predicted4) - (train_numeric_Y))**2 / len(predicted4)
e = np.sum((predicted5) - (train_numeric_Y))**2 / len(predicted5)
f = np.sum((predicted10) - (train_numeric_Y))**2 / len(predicted10)


# In[ ]:


dt_nums = [1,2,3,4,5,10]
plt.figure(figsize=(15,10))

plt.subplot(221)
plt.title('Decision Tree Depth & MSE \n of ConfirmedCases',fontsize=20)
plt.plot(range(len(dt_nums)), [a['ConfirmedCases'],b['ConfirmedCases'],c['ConfirmedCases'],d['ConfirmedCases'],e['ConfirmedCases'],f['ConfirmedCases']],
         label='ConfirmedCases')
plt.xlabel('decision tree depth')
plt.ylabel('mse')
plt.xticks(range(len(dt_nums)),dt_nums)
plt.legend()

plt.subplot(222)
plt.title('Decision Tree Depth & MSE \n of Fatalities',fontsize=20)
plt.plot(range(len(dt_nums)), [a['Fatalities'],b['Fatalities'],c['Fatalities'],d['Fatalities'],e['Fatalities'],f['ConfirmedCases']],
         label='Fatalities',color='y')
plt.xlabel('decision tree depth')
plt.ylabel('mse')
plt.xticks(range(len(dt_nums)),dt_nums)
plt.legend()

plt.show()


# The deeper depth gets the lower mse in confirmedCases but the higher fatalities

# #### PCA to 2-D space

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB


# In[ ]:


pipeline = Pipeline([('scale', StandardScaler()), ('pca', PCA(n_components=2))])
pca = pipeline.fit(train_numeric_X)
pca_X = pca.transform(train_numeric_X)


# In[ ]:


fig = px.scatter(x=pca_X[:,0], y=pca_X[:,1])
fig.show()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(
    x=pca_X[:,0],
    y=pca_X[:,1],
    marker=dict(color="crimson", size=np.log(1 + train_numeric_Y['ConfirmedCases'])),
    mode="markers",
    name="training data",
))
fig.update_layout(title="PCA reduction scatter diagram",
                  xaxis_title="x",
                  yaxis_title="y")

fig.show()


# #### TSNE

# In[ ]:


from sklearn.manifold import TSNE


# In[ ]:


X_embedded = TSNE(n_components=3).fit_transform(train_numeric_X)


# In[ ]:


fig = go.Figure(data=[go.Scatter3d(
    x=X_embedded[:,0],
    y=X_embedded[:,1],
    z=X_embedded[:,2],
    mode='markers',
    marker=dict(
        size=12,
        color=train_numeric_Y['ConfirmedCases'],
        colorscale='Viridis',
        opacity=0.8
    )
)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()


# In[ ]:




