#!/usr/bin/env python
# coding: utf-8

# # **House Price Prediction by 5 models + EDA & FE**

# ![](https://images.pexels.com/photos/106399/pexels-photo-106399.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940)
# ###### Thanks for the photo Binyamin Mellish: https://www.pexels.com/photo/home-real-estate-106399/

# ### **Below screenshot explains what all headers mean.**

# ![](https://storage.googleapis.com/kaggle-forum-message-attachments/479761/11440/Screenshot%202019-02-27%20at%205.26.24%20PM.png)
# ###### Thanks for the screenshot: https://www.kaggle.com/harlfoxem/housesalesprediction/discussion/82135

# ## **Acknowledgements**
# #### This kernel uses such good kernels:
#    - https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
#    - https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models
#    - https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg\
#    - https://www.kaggle.com/shanroy1999/house-price-prediction-using-linear-regression
#    - https://www.kaggle.com/fulrose/copy-fix-house-price-prediction-1-2
#    - https://www.kaggle.com/sid321axn/house-price-prediction-gboosting-adaboost-etc
#    - https://www.kaggle.com/dnzcihan/house-sales-prediction-and-eda
#    - https://www.kaggle.com/darkcore/house-sales-visualization

# <a class="anchor" id="0.1"></a>
# ## **Table of Contents**
# 1. [Import libraries](#1)
# 2. [Download datasets](#2)
# 3. [EDA](#3)
# 4. [FE: building the feature importance diagrams](#4)
#   -  [LGBM](#4.1)
#   -  [XGB](#4.2)
#   -  [Logistic Regression](#4.3)
#   -  [Linear Reagression](#4.4)
# 5. [Comparison of the all feature importance diagrams](#5)
# 6. [Dada for modeling](#6)
# 7. [Preparing to modeling](#7)
# 8. [Tuning models](#8)
#   -  [Random Forest](#8.1)
#   -  [XGB](#8.2)
#   -  [LGBM](#8.3)
#   -  [BaggingRegressor](#8.4)
#   -  [ExtraTreesRegressor](#8.5)
# 9. [Models comparison](#9)
# 10. [Prediction](#10)

# <a class="anchor" id="1"></a>
# ## 1. Import libraries 
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
from mpl_toolkits import mplot3d
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')

# WordCloud
from wordcloud import WordCloud

# map visualization
import folium 
from folium import plugins

# preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
import pandas_profiling as pp

# models
from sklearn.linear_model import LinearRegression,LogisticRegression, SGDRegressor, RidgeCV
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor 
from sklearn.ensemble import BaggingRegressor
import sklearn.model_selection
from sklearn.model_selection import cross_val_predict as cvp
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import pearsonr

import xgboost as xgb
import lightgbm as lgb

# model tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

import warnings
warnings.filterwarnings("ignore")


# <a class="anchor" id="2"></a>
# ## 2. Download datasets
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


features = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view',
            'condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated',
            'zipcode','lat','long','sqft_living15','sqft_lot15']


# In[ ]:


valid_part = 0.3
pd.set_option('max_columns',100)


# In[ ]:


train0 = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
train0 = train0[features]
train0.head(5)


# In[ ]:


train0.info()


# In[ ]:


train0 = train0.dropna()
train0.head(5)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
# Determination categorical features
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = train0.columns.values.tolist()
for col in features:
    if train0[col].dtype in numerics: continue
    categorical_columns.append(col)
# Encoding categorical features
for col in categorical_columns:
    if col in train0.columns:
        le = LabelEncoder()
        le.fit(list(train0[col].astype(str).values))
        train0[col] = le.transform(list(train0[col].astype(str).values))


# In[ ]:


train0['price'] = (train0['price']).astype(int)
train0['floors'] = (train0['floors']).astype(int)
train0['bedrooms'] = (train0['bedrooms']).astype(int)


# <a class="anchor" id="3"></a>
# ## 3. EDA
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


train0.head(10)


# In[ ]:


train0.info()


# In[ ]:


train0['price'].value_counts()


# In[ ]:


train0.corr()


# In[ ]:


#Thanks to: https://www.kaggle.com/arthurtok/feature-ranking-rfe-random-forest-linear-models
str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in train0.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
# Get to the numeric columns by inversion            
num_list = train0.columns.difference(str_list) 
# Create Dataframe containing only numerical features
house_num = train0[num_list]
f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation of features')
# Draw the heatmap using seaborn
sns.heatmap(house_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="cubehelix", linecolor='k', annot=True)


# In[ ]:


# FINDING CORRELATION
# Thanks to: https://www.kaggle.com/sid321axn/house-price-prediction-gboosting-adaboost-etc
# As id and date columns are not important to predict price so we are discarding it for finding correlation
featuress = train0.iloc[:,3:].columns.tolist()
target = train0.iloc[:,0].name


# In[ ]:


# Finding Correlation of price woth other variables to see how many variables are strongly correlated with price
correlations = {}
for f in featuress:
    data_temp = train0[[f,target]]
    x1 = data_temp[f].values
    x2 = data_temp[target].values
    key = f + ' vs ' + target
    correlations[key] = pearsonr(x1,x2)[0]


# In[ ]:


# Printing all the correlated features value with respect to price which is target variable
data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index]


# In[ ]:


sns.distplot(train0['yr_built'])


# In[ ]:


train0.describe()


# In[ ]:


# As zipcode is negatively correlated with sales price , so we can discard it for sales price prediction.
drop_columns = ['zipcode', 'view', 'waterfront', 'yr_renovated']
train0 = train0.drop(columns = drop_columns)


# In[ ]:


train0.describe(percentiles=[.01, .05, .1, .5, .9, .92, .93, .94, .96, .97, .99])


# In[ ]:


(train0['condition'] > 2.5).value_counts()


# In[ ]:


(train0['grade'] == 4).value_counts()


# In[ ]:


train0 = train0[(
                (train0['price'] <= 1000000) & 
                (train0['price'] > 170000) & 
                (train0['bathrooms'] <= 4) & 
                (train0['condition'] > 2.5) & 
                (train0['grade'] != 4) &
                (train0['sqft_lot15'] > 1300) &
                (train0['sqft_lot15'] < 44000) &
                (train0['sqft_lot'] > 1500) &
                (train0['sqft_lot'] < 70000) &
                (train0['sqft_living'] > 700) & 
                (train0['yr_built'] > 1925) & 
                (train0['bedrooms'] > 0) & 
                (train0['bedrooms'] < 7) 
                )]


# In[ ]:


train0.info()


# In[ ]:


pp.ProfileReport(train0)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
def plotting_3_chart(df, feature):
    ## Importing seaborn, matplotlab and scipy modules. 
    style.use('fivethirtyeight')

    ## Creating a customized chart. and giving in figsize and everything. 
    fig = plt.figure(constrained_layout=True, figsize=(15,10))
    ## creating a grid of 3 cols and 3 rows. 
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    
    ## Customizing the histogram grid. 
    ax1 = fig.add_subplot(grid[0, :2])
    ## Set the title. 
    ax1.set_title('Histogram')
    ## plot the histogram. 
    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)

    # customizing the QQ_plot. 
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title. 
    ax2.set_title('QQ_plot')
    ## Plotting the QQ_Plot. 
    stats.probplot(df.loc[:,feature], plot = ax2)

    ## Customizing the Box Plot. 
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set title. 
    ax3.set_title('Box Plot')
    ## Plotting the box plot. 
    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );
    
plotting_3_chart(train0, 'price')


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
y = np.array(train0.price)
plt.subplot(131)
plt.plot(range(len(y)),y,'.');plt.ylabel('price');plt.xlabel('index');
plt.subplot(132)
sns.boxplot(y=train0.price)


# In[ ]:


#Thanks to https://towardsdatascience.com/an-easy-introduction-to-3d-plotting-with-matplotlib-801561999725
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection="3d")

z_points = train0['price']
x_points = train0['condition']
y_points = train0['yr_built']
ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv');

ax.set_xlabel('condition')
ax.set_ylabel('yr_built')
ax.set_zlabel('price')

plt.show()


# In[ ]:


# Thanks to: https://www.kaggle.com/shanroy1999/house-price-prediction-using-linear-regression
fig=plt.figure(figsize=(19,12.5))
ax=fig.add_subplot(2,2,2, projection="3d")
ax.scatter(train0['floors'],train0['bedrooms'],train0['sqft_living'],c="darkgreen",alpha=.5)
ax.set(xlabel='\nFloors',ylabel='\nBedrooms',zlabel='\nsqft Living')
ax.set(ylim=[0,12])
plt.show()


# In[ ]:


# Thanks to: https://www.kaggle.com/shanroy1999/house-price-prediction-using-linear-regression
grpby_bedrooms_df = train0[["price", "bedrooms"]].groupby(by = "bedrooms", as_index = False)
grpby_bedrooms_df = grpby_bedrooms_df.mean().astype(int)
grpby_bedrooms_df.head()


# In[ ]:


# Thanks to: https://www.kaggle.com/shanroy1999/house-price-prediction-using-linear-regression
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
ax1.set(yscale = "log")
sns.stripplot(x = "bedrooms", y = "price", data = train0, ax = ax1, jitter=True, palette="Blues_d")
sns.barplot(x = "bedrooms", y = "price", data = grpby_bedrooms_df, ax = ax2, palette="Blues_d")
plt.show()


# In[ ]:


# Thanks to: https://www.kaggle.com/fulrose/copy-fix-house-price-prediction-1-2
houses_map = folium.Map(location = [train0['lat'].mean(), train0['long'].mean()], zoom_start = 10)
lat_long_data = train0[['lat', 'long']].values.tolist()
h_cluster = folium.plugins.FastMarkerCluster(lat_long_data).add_to(houses_map)
houses_map


# In[ ]:


# Thanks to: https://www.kaggle.com/darkcore/house-sales-visualization
#Create Grade Frame
gradeframe = pd.DataFrame({"Grades":train0.grade.value_counts().index,"House_Grade":train0.grade.value_counts().values})
gradeframe["Grades"] = gradeframe["Grades"].apply(lambda x : "Grade " + str(x))
gradeframe.set_index("Grades",inplace=True)
#gradeframe


# In[ ]:


# Thanks to: https://www.kaggle.com/darkcore/house-sales-visualization
p1 = [go.Pie(labels = gradeframe.index,values = gradeframe.House_Grade,hoverinfo="percent+label+value",hole=0.1,marker=dict(line=dict(color="#000000",width=2)))]
layout4 = go.Layout(title="Grade Pie Chart")
fig4 = go.Figure(data=p1,layout=layout4)
iplot(fig4)


# In[ ]:


# Thanks to: https://www.kaggle.com/darkcore/house-sales-visualization
#Create Bedrooms Frame
bedroomsframe = pd.DataFrame({"Bedrooms":train0.bedrooms.value_counts().index,"House_bedrooms":train0.bedrooms.value_counts().values})
bedroomsframe["Bedrooms"] = bedroomsframe["Bedrooms"].apply(lambda x : "Bedrooms " + str(x))
bedroomsframe.set_index("Bedrooms",inplace=True)
#bedroomsframe


# In[ ]:


# Thanks to: https://www.kaggle.com/darkcore/house-sales-visualization
p1 = [go.Pie(labels = bedroomsframe.index,values = bedroomsframe.House_bedrooms,hoverinfo="percent+label+value",hole=0.1,marker=dict(line=dict(color="#000000",width=2)))]
layout4 = go.Layout(title="Bedrooms Pie Chart")
fig4 = go.Figure(data=p1,layout=layout4)
iplot(fig4)


# In[ ]:


# Thanks to: https://www.kaggle.com/darkcore/house-sales-visualization
#Create Floors Frame
floorsframe = pd.DataFrame({"Floors":train0.floors.value_counts().index,"House_floors":train0.floors.value_counts().values})
floorsframe["Floors"] = floorsframe["Floors"].apply(lambda x : "Floors " + str(x))
floorsframe.set_index("Floors",inplace=True)
#floorsframe


# In[ ]:


# Thanks to: https://www.kaggle.com/darkcore/house-sales-visualization
p1 = [go.Pie(labels = floorsframe.index,values = floorsframe.House_floors,hoverinfo="percent+label+value",hole=0.1,marker=dict(line=dict(color="#000000",width=2)))]
layout4 = go.Layout(title="Floors Pie Chart")
fig4 = go.Figure(data=p1,layout=layout4)
iplot(fig4)


# In[ ]:


# Thanks to: https://www.kaggle.com/darkcore/house-sales-visualization
#Create Condition Frame
conditionframe = pd.DataFrame({"Condition":train0.condition.value_counts().index,"House_condition":train0.condition.value_counts().values})
conditionframe["Condition"] = conditionframe["Condition"].apply(lambda x : "Condition " + str(x))
conditionframe.set_index("Condition",inplace=True)
#conditionframe


# In[ ]:


# Thanks to: https://www.kaggle.com/darkcore/house-sales-visualization
p1 = [go.Pie(labels = conditionframe.index,values = conditionframe.House_condition,hoverinfo="percent+label+value",hole=0.1,marker=dict(line=dict(color="#000000",width=2)))]
layout4 = go.Layout(title="Condition Pie Chart")
fig4 = go.Figure(data=p1,layout=layout4)
iplot(fig4)


# In[ ]:


# Thanks to: https://www.kaggle.com/darkcore/house-sales-visualization
builtyear = pd.DataFrame({"Years":train0.yr_built})
builtyear["Years"] = builtyear["Years"].apply(lambda x: "y" + str(x)) #I can't use wordcloud with integers so I put y on head
#builtyear["Years"].head()


# In[ ]:


# Thanks to: https://www.kaggle.com/darkcore/house-sales-visualization
plt.subplots(figsize=(8,8))
wcloud  = WordCloud(background_color="white",width=500,height=500).generate(",".join(builtyear["Years"]))
plt.imshow(wcloud)
plt.title("Years for Most Built Homes",fontsize=40)
plt.axis("off")
plt.show()


# <a class="anchor" id="4"></a>
# ## 4. FE: building the feature importance diagrams
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Clone data for FE 
train_fe = copy.deepcopy(train0)
target_fe = train_fe['price']
del train_fe['price']


# <a class="anchor" id="4.1"></a>
# ### 4.1 LGBM 

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
X = train_fe
z = target_fe


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
#%% split training set to validation set
Xtrain, Xval, Ztrain, Zval = train_test_split(X, z, test_size=0.2, random_state=0)
train_set = lgb.Dataset(Xtrain, Ztrain, silent=False)
valid_set = lgb.Dataset(Xval, Zval, silent=False)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': True,
        'seed':0,        
    }

modelL = lgb.train(params, train_set = train_set, num_boost_round=1000,
                   early_stopping_rounds=50,verbose_eval=10, valid_sets=valid_set)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
lgb.plot_importance(modelL,ax = axes,height = 0.5)
plt.show();plt.close()


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
feature_score = pd.DataFrame(train_fe.columns, columns = ['feature']) 
feature_score['score_lgb'] = modelL.feature_importance()


# <a class="anchor" id="4.2"></a>
# ### 4.2 XGB
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
#%% split training set to validation set 
data_tr  = xgb.DMatrix(Xtrain, label=Ztrain)
data_cv  = xgb.DMatrix(Xval   , label=Zval)
evallist = [(data_tr, 'train'), (data_cv, 'valid')]


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
parms = {'max_depth':8, #maximum depth of a tree
         'objective':'reg:squarederror',
         'eta'      :0.3,
         'subsample':0.8,#SGD will use this percentage of data
         'lambda '  :4, #L2 regularization term,>1 more conservative 
         'colsample_bytree ':0.9,
         'colsample_bylevel':1,
         'min_child_weight': 10}
modelx = xgb.train(parms, data_tr, num_boost_round=200, evals = evallist,
                  early_stopping_rounds=30, maximize=False, 
                  verbose_eval=10)

print('score = %1.5f, n_boost_round =%d.'%(modelx.best_score,modelx.best_iteration))


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
xgb.plot_importance(modelx,ax = axes,height = 0.5)
plt.show();plt.close()


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
feature_score['score_xgb'] = feature_score['feature'].map(modelx.get_score(importance_type='weight'))
feature_score


# <a class="anchor" id="4.3"></a>
# ### 4.3 Logistic Regression
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
# Standardization for regression model
train_fe = pd.DataFrame(
    preprocessing.MinMaxScaler().fit_transform(train_fe),
    columns=train_fe.columns,
    index=train_fe.index
)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(train_fe, target_fe)
coeff_logreg = pd.DataFrame(train_fe.columns.delete(0))
coeff_logreg.columns = ['feature']
coeff_logreg["score_logreg"] = pd.Series(logreg.coef_[0])
coeff_logreg.sort_values(by='score_logreg', ascending=False)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
# the level of importance of features is not associated with the sign
coeff_logreg["score_logreg"] = coeff_logreg["score_logreg"].abs()
feature_score = pd.merge(feature_score, coeff_logreg, on='feature')


# <a class="anchor" id="4.4"></a>
# ### 4.4 Linear Regression
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
# Linear Regression

linreg = LinearRegression()
linreg.fit(train_fe, target_fe)
coeff_linreg = pd.DataFrame(train_fe.columns.delete(0))
coeff_linreg.columns = ['feature']
coeff_linreg["score_linreg"] = pd.Series(linreg.coef_)
coeff_linreg.sort_values(by='score_linreg', ascending=False)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-fe-eda-with-3d
coeff_linreg["score_linreg"] = coeff_linreg["score_linreg"].abs()
feature_score = pd.merge(feature_score, coeff_linreg, on='feature')
feature_score = feature_score.fillna(0)
feature_score = feature_score.set_index('feature')
feature_score


# <a class="anchor" id="5"></a>
# ## 5. Comparison of the all feature importance diagrams 
# ##### [Back to Table of Contents](#0.1)

# ###### These wonderful charts are taken in [vbmokin](https://www.kaggle.com/vbmokin)
# ###### Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
# Thanks to: https://www.kaggle.com/nanomathias/feature-engineering-importance-testing
# MinMax scale all importances
feature_score = pd.DataFrame(
    preprocessing.MinMaxScaler().fit_transform(feature_score),
    columns=feature_score.columns,
    index=feature_score.index
)

# Create mean column
feature_score['mean'] = feature_score.mean(axis=1)

# Plot the feature importances
feature_score.sort_values('mean', ascending=False).plot(kind='bar', figsize=(20, 10))


# In[ ]:


feature_score.sort_values('mean', ascending=False)


# In[ ]:


# Thanks to: Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
# Create total column with different weights
feature_score['total'] = 0.5*feature_score['score_lgb'] + 0.3*feature_score['score_xgb']                        + 0.1*feature_score['score_logreg'] + 0.1*feature_score['score_linreg']

# Plot the feature importances
feature_score.sort_values('total', ascending=False).plot(kind='bar', figsize=(20, 10))


# In[ ]:


feature_score.sort_values('total', ascending=False)


# <a class="anchor" id="6"></a>
# ## 6. Dada for modeling
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models
target_name = 'price'
train_target0 = train0[target_name]
train0 = train0.drop([target_name], axis=1)


# In[ ]:


# Synthesis test0 from train0
train0, test0, train_target0, test_target0 = train_test_split(train0, train_target0, test_size=0.2, random_state=0)


# In[ ]:


# For boosting model
train0b = train0
train_target0b = train_target0
# Synthesis valid as test for selection models
trainb, testb, targetb, target_testb = train_test_split(train0b, train_target0b, test_size=valid_part, random_state=0)


# In[ ]:


#For models from Sklearn
scaler = StandardScaler()
train0 = pd.DataFrame(scaler.fit_transform(train0), columns = train0.columns)


# In[ ]:


train0.head(3)


# In[ ]:


len(train0)


# In[ ]:


# Synthesis valid as test for selection models
train, test, target, target_test = train_test_split(train0, train_target0, test_size=valid_part, random_state=0)


# In[ ]:


train.head(3)


# In[ ]:


test.head(3)


# In[ ]:


train.info()


# In[ ]:


test.info()


# <a class="anchor" id="7"></a>
# ## 7. Preparing to modeling
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models
acc_train_r2 = []
acc_test_r2 = []
acc_train_d = []
acc_test_d = []
acc_train_rmse = []
acc_test_rmse = []


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models
def acc_d(y_meas, y_pred):
    # Relative error between predicted y_pred and measured y_meas values
    return mean_absolute_error(y_meas, y_pred)*len(y_meas)/sum(abs(y_meas))

def acc_rmse(y_meas, y_pred):
    # RMSE between predicted y_pred and measured y_meas values
    return (mean_squared_error(y_meas, y_pred))**0.5


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models
def acc_boosting_model(num,model,train,test,num_iteration=0):
    # Calculation of accuracy of boosting model by different metrics
    
    global acc_train_r2, acc_test_r2, acc_train_d, acc_test_d, acc_train_rmse, acc_test_rmse
    
    if num_iteration > 0:
        ytrain = model.predict(train, num_iteration = num_iteration)  
        ytest = model.predict(test, num_iteration = num_iteration)
    else:
        ytrain = model.predict(train)  
        ytest = model.predict(test)

    print('target = ', targetb[:5].values)
    print('ytrain = ', ytrain[:5])

    acc_train_r2_num = round(r2_score(targetb, ytrain) * 100, 2)
    print('acc(r2_score) for train =', acc_train_r2_num)   
    acc_train_r2.insert(num, acc_train_r2_num)

    acc_train_d_num = round(acc_d(targetb, ytrain) * 100, 2)
    print('acc(relative error) for train =', acc_train_d_num)   
    acc_train_d.insert(num, acc_train_d_num)

    acc_train_rmse_num = round(acc_rmse(targetb, ytrain) * 100, 2)
    print('acc(rmse) for train =', acc_train_rmse_num)   
    acc_train_rmse.insert(num, acc_train_rmse_num)

    print('target_test =', target_testb[:5].values)
    print('ytest =', ytest[:5])
    
    acc_test_r2_num = round(r2_score(target_testb, ytest) * 100, 2)
    print('acc(r2_score) for test =', acc_test_r2_num)
    acc_test_r2.insert(num, acc_test_r2_num)
    
    acc_test_d_num = round(acc_d(target_testb, ytest) * 100, 2)
    print('acc(relative error) for test =', acc_test_d_num)
    acc_test_d.insert(num, acc_test_d_num)
    
    acc_test_rmse_num = round(acc_rmse(target_testb, ytest) * 100, 2)
    print('acc(rmse) for test =', acc_test_rmse_num)
    acc_test_rmse.insert(num, acc_test_rmse_num)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models
def acc_model(num,model,train,test):
    # Calculation of accuracy of model from Sklearn by different metrics   
  
    global acc_train_r2, acc_test_r2, acc_train_d, acc_test_d, acc_train_rmse, acc_test_rmse
    
    ytrain = model.predict(train)  
    ytest = model.predict(test)

    print('target = ', target[:5].values)
    print('ytrain = ', ytrain[:5])

    acc_train_r2_num = round(r2_score(target, ytrain) * 100, 2)
    print('acc(r2_score) for train =', acc_train_r2_num)   
    acc_train_r2.insert(num, acc_train_r2_num)

    acc_train_d_num = round(acc_d(target, ytrain) * 100, 2)
    print('acc(relative error) for train =', acc_train_d_num)   
    acc_train_d.insert(num, acc_train_d_num)

    acc_train_rmse_num = round(acc_rmse(target, ytrain) * 100, 2)
    print('acc(rmse) for train =', acc_train_rmse_num)   
    acc_train_rmse.insert(num, acc_train_rmse_num)

    print('target_test =', target_test[:5].values)
    print('ytest =', ytest[:5])
    
    acc_test_r2_num = round(r2_score(target_test, ytest) * 100, 2)
    print('acc(r2_score) for test =', acc_test_r2_num)
    acc_test_r2.insert(num, acc_test_r2_num)
    
    acc_test_d_num = round(acc_d(target_test, ytest) * 100, 2)
    print('acc(relative error) for test =', acc_test_d_num)
    acc_test_d.insert(num, acc_test_d_num)
    
    acc_test_rmse_num = round(acc_rmse(target_test, ytest) * 100, 2)
    print('acc(rmse) for test =', acc_test_rmse_num)
    acc_test_rmse.insert(num, acc_test_rmse_num)


# <a class="anchor" id="8"></a>
# ## 8. Tuning models
# ##### [Back to Table of Contents](#0.1)

# <a class="anchor" id="8.1"></a>
# ### 8.1 Random Forest
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models
# Random Forest

#random_forest = GridSearchCV(estimator=RandomForestRegressor(), param_grid={'n_estimators': [100, 1000]}, cv=5)
random_forest = RandomForestRegressor()
random_forest.fit(train, target)
acc_model(1,random_forest,train,test)


# <a class="anchor" id="8.2"></a>
# ### 8.2 XGB
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models
xgb_clf = xgb.XGBRegressor({'objective': 'reg:squarederror'}) 
parameters = {'n_estimators': [60, 100, 120, 140], 
              'learning_rate': [0.01, 0.1],
              'max_depth': [5, 7],
              'reg_lambda': [0.5]}
xgb_reg = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=5, n_jobs=-1).fit(trainb, targetb)
print("Best score: %0.3f" % xgb_reg.best_score_)
print("Best parameters set:", xgb_reg.best_params_)
acc_boosting_model(2,xgb_reg,trainb,testb)


# <a class="anchor" id="8.3"></a>
# ### 8.3 LGBM
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models
#%% split training set to validation set
Xtrain, Xval, Ztrain, Zval = train_test_split(trainb, targetb, test_size=0.2, random_state=0)
train_set = lgb.Dataset(Xtrain, Ztrain, silent=False)
valid_set = lgb.Dataset(Xval, Zval, silent=False)


# In[ ]:


params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': False,
        'seed':0,        
    }
modelL = lgb.train(params, train_set = train_set, num_boost_round=10000,
                   early_stopping_rounds=8000,verbose_eval=500, valid_sets=valid_set)


# In[ ]:


acc_boosting_model(3,modelL,trainb,testb,modelL.best_iteration)


# In[ ]:


fig =  plt.figure(figsize = (5,5))
axes = fig.add_subplot(111)
lgb.plot_importance(modelL,ax = axes,height = 0.5)
plt.show();
plt.close()


# <a class="anchor" id="8.4"></a>
# ### 8.4 Bagging Regressor
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models
# Bagging Regressor

bagging = BaggingRegressor()
bagging.fit(train, target)
acc_model(4,bagging,train,test)


# <a class="anchor" id="8.5"></a>
# ### 8.5 Extra Trees Regressor
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models
# Extra Trees Regressor

etr = ExtraTreesRegressor()
etr.fit(train, target)
acc_model(5,etr,train,test)


# <a class="anchor" id="9"></a>
# ### 9 Models comparison
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models
models = pd.DataFrame({
    'Model': ['Random Forest', 'XGB', 'LGBM', 'BaggingRegressor', 'ExtraTreesRegressor'],
    
    'r2_train': acc_train_r2,
    'r2_test': acc_test_r2,
    'd_train': acc_train_d,
    'd_test': acc_test_d,
    'rmse_train': acc_train_rmse,
    'rmse_test': acc_test_rmse
                     })


# In[ ]:


pd.options.display.float_format = '{:,.2f}'.format


# In[ ]:


print('Prediction accuracy for models by R2 criterion - r2_test')
models.sort_values(by=['r2_test', 'r2_train'], ascending=False)


# In[ ]:


print('Prediction accuracy for models by relative error - d_test')
models.sort_values(by=['d_test', 'd_train'], ascending=True)


# In[ ]:


print('Prediction accuracy for models by RMSE - rmse_test')
models.sort_values(by=['rmse_test', 'rmse_train'], ascending=True)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models
# Plot
plt.figure(figsize=[15,6])
xx = models['Model']
plt.tick_params(labelsize=14)
plt.plot(xx, models['r2_train'], label = 'r2_train')
plt.plot(xx, models['r2_test'], label = 'r2_test')
plt.legend()
plt.title('R2-criterion for 5 popular models for train and test datasets')
plt.xlabel('Models')
plt.ylabel('R2-criterion, %')
plt.xticks(xx, rotation='vertical')
plt.savefig('graph.png')
plt.show()


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models
# Plot
plt.figure(figsize=[15,6])
xx = models['Model']
plt.tick_params(labelsize=14)
plt.plot(xx, models['d_train'], label = 'd_train')
plt.plot(xx, models['d_test'], label = 'd_test')
plt.legend()
plt.title('Relative errors for 5 popular models for train and test datasets')
plt.xlabel('Models')
plt.ylabel('Relative error, %')
plt.xticks(xx, rotation='vertical')
plt.savefig('graph.png')
plt.show()


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models
# Plot
plt.figure(figsize=[15,6])
xx = models['Model']
plt.tick_params(labelsize=14)
plt.plot(xx, models['rmse_train'], label = 'rmse_train')
plt.plot(xx, models['rmse_test'], label = 'rmse_test')
plt.legend()
plt.title('RMSE for 5 popular models for train and test datasets')
plt.xlabel('Models')
plt.ylabel('RMSE, %')
plt.xticks(xx, rotation='vertical')
plt.savefig('graph.png')
plt.show()


# <a class="anchor" id="10"></a>
# ### 10 Prediction
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


test0.info()


# In[ ]:


test0.head(3)


# In[ ]:


#For models from Sklearn
testn = pd.DataFrame(scaler.transform(test0), columns = test0.columns)


# In[ ]:


#Linear Regression model for basic train
linreg.fit(train0, train_target0)
linreg_predict = linreg.predict(testn)
linreg_predict[:3]


# In[ ]:


#Random Forest model for basic train
random_forest.fit(train0, train_target0)
rf_predict = random_forest.predict(testn)
rf_predict[:3]


# In[ ]:


#Bagging Regressor model for basic train
bagging.fit(train0, train_target0)
bg_predict = bagging.predict(testn)
bg_predict[:3]


# In[ ]:


# XGB Regression model for basic train
xgb_reg.fit(train0, train_target0)
xgb_predict = xgb_reg.predict(testn)
xgb_predict[:3]


# In[ ]:


# LGB Regression model for basic train
lgb_predict = modelL.predict(test0)
lgb_predict[:3]


# In[ ]:


# Extra Trees Regressor model for basic train
etr.fit(train0, train_target0)
etr_predict = etr.predict(testn)
etr_predict[:3]


# In[ ]:


# Thanks to: https://www.kaggle.com/dnzcihan/house-sales-prediction-and-eda
final_df = test_target0.values
final_df = pd.DataFrame(final_df,columns=['Real_price'])
final_df['predicted_prices'] = lgb_predict.astype(int)
final_df['difference'] = abs(final_df['Real_price'] - final_df['predicted_prices']).astype(int)
final_df.head(20)


# In[ ]:


mean_real_price = round(final_df['Real_price'].mean(), 0)
mean_predicted_prices = round(final_df['predicted_prices'].mean(), 0)
mean_difference = round(final_df['difference'].mean(), 0)
# Create and append mean values to DataFrame 
mean_val = []
mean_val.append(('real_price', mean_real_price))
mean_val.append(('predicted_prices', mean_predicted_prices))
mean_val.append(('difference', mean_difference))
pd.DataFrame(mean_val, columns = ('Name', 'Average'))

