#!/usr/bin/env python
# coding: utf-8

# # New York City Airbnb Open Data
# ## Dataset Description
# >Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present more unique, personalized way of experiencing the world. This dataset describes the listing activity and metrics in NYC, NY for 2019

# In[ ]:


from IPython.display import Image
Image(url= "https://img4.cityrealty.com/neo/i/p/mig/airbnb_guide.jpg")


# In[ ]:


import numpy as np                 # Linear Algebra
import pandas as pd                # Data Processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib                  # 2D Plotting Library
import matplotlib.pyplot as plt
import seaborn as sns              # Python Data Visualization Library based on matplotlib
import geopandas as gpd            # Python Geospatial Data Library
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly as plotly                # Interactive Graphing Library for Python
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)

import folium
import folium.plugins
# featuretools for automated feature engineering
import featuretools as ft

# ignore warnings from pandas
import warnings
warnings.filterwarnings('ignore')

import wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import sklearn
from sklearn.linear_model import LinearRegression,RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error,make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler,MinMaxScaler
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedShuffleSplit, KFold, GridSearchCV, RandomizedSearchCV,train_test_split,cross_val_score

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import ExtraTreeRegressor,DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor


# In[ ]:


data=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


data.isna().sum()


# In[ ]:


plt.figure(figsize=(12,4))
sns.heatmap(data.isnull(),cbar=False,cmap='viridis',yticklabels=False)
plt.title('Missing value in the dataset');


# In[ ]:


df_airbnb = data.fillna({'reviews_per_month':0})


# In[ ]:


df_airbnb = df_airbnb.dropna()
data_viz = data.dropna()


# ## Exploratory Data Analysis
# #### Data Visualizations and More..............

# In[ ]:


col = "room_type"
grouped = data_viz[col].value_counts().reset_index()
grouped = grouped.rename(columns = {col : "count", "index" : col})

## plot
trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0], marker=dict(colors=["#6ad49b", "#a678de"]))
layout = go.Layout(title="", height=400, legend=dict(x=0.1, y=1.1))
fig = go.Figure(data = [trace], layout = layout)
iplot(fig)


# In[ ]:


col = "neighbourhood_group"
grouped = data_viz[col].value_counts().reset_index()
grouped = grouped.rename(columns = {col : "count", "index" : col})

## plot
trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0], marker=dict(colors=["#6ad49b", "#a678de"]))
layout = go.Layout(title="", height=400, legend=dict(x=0.1, y=1.1))
fig = go.Figure(data = [trace], layout = layout)
iplot(fig)


# In[ ]:


plt.figure(figsize=(10,6))
sns.set_style("darkgrid")
plt.title('Distribution of Airbnb price')
sns.kdeplot(data=data_viz['price'], shade=True).set(xlim=(0))


# In[ ]:


ent_home = data_viz[data_viz.room_type == 'Entire home/apt']
proom = data_viz[data_viz.room_type == 'Private room']
sh_room = data_viz[data_viz.room_type == 'Shared room']

plt.figure(figsize=(10,6))
sns.set_style("darkgrid")
sns.kdeplot(data=ent_home['price'],label='Entire home/apt', shade=True)
sns.kdeplot(data=proom['price'],label='Private room', shade=True)
sns.kdeplot(data=sh_room['price'],label='Shared room', shade=True)


# In[ ]:


sns.lmplot(x='price',y='number_of_reviews',data=data_viz,aspect=2,height=6)
plt.xlabel('Price')
plt.ylabel('No of Reviews')
plt.title('Price vs Reviews');


# In[ ]:


plt.figure(figsize=(14,6))
sns.boxplot(x='neighbourhood_group', y='availability_365',data=data_viz,palette='rainbow')
plt.title('Box plot of neighbourhood_group vs Availability');


# In[ ]:


name =  data_viz['name']
name  = name.dropna()


# In[ ]:


from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in name)
wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="white").generate(text)
plt.figure(figsize=(10,6))
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.show()


# ### Experimentation with Automated Feature Enginnering
# #### [Feature Tools](https://www.featuretools.com/)

# >The general problem of feature engineering is taking disparate data, often distributed across multiple tables, and combining it into a single table that can be used for training a machine learning model. Featuretools has the ability to do this for us, creating many new candidate features with minimal effort. These features are combined into a single table that can then be passed on to our model.
# 
# >Now that we know what we are trying to avoid (tedious manual feature engineering), let's figure out how to automate this process. Featuretools operates on an idea known as Deep Feature Synthesis. You can read the original paper here, and although it's quite readable, it's not necessary to understand the details to do automated feature engineering. The concept of Deep Feature Synthesis is to use basic building blocks known as feature primitives (like the transformations and aggregations done above) that can be stacked on top of each other to form new features. The depth of a "deep feature" is equal to the number of stacked primitives.
# 
# > Featuretools builds on simple ideas to create a powerful method, and we will build up our understanding in much the same way.
# 
# ###### The first part of Featuretools to understand is an entity. This is simply a table, or in pandas, a DataFrame. We corral multiple entities into a single object called an EntitySet. This is just a large data structure composed of many individual entities and the relationships between them.

# In[ ]:


es = ft.EntitySet(id = 'airbnb')


# ### Entities
# An entity is simply a table, which is represented in Pandas as a dataframe. Each entity must have a uniquely identifying column, known as an index.
# 
# When we create an entity in featuretools, we have to identify which column of the dataframe is the index. If the data does not have a unique index we can tell featuretools to make an index for the entity by passing in make_index = True and specifying a name for the index. If the data also has a uniquely identifying time index, we can pass that in as the time_index parameter.
# 
# Featuretools will automatically infer the variable types (numeric, categorical, datetime) of the columns in our data, but we can also pass in specific datatypes to override this behavior. As an example, even though the repaid column in the loans dataframe is represented as an integer, we can tell featuretools that this is a categorical feature since it can only take on two discrete values. This is done using an integer with the variables as keys and the feature types as values.
# 
# In the code below we create the entities and add them to the EntitySet. The syntax is relatively straightforward.

# In[ ]:


df_airbnb


# In[ ]:


# Create an entity from the client dataframe
# This dataframe already has an index and a time index
es = es.entity_from_dataframe(entity_id = 'airbnb', dataframe = df_airbnb, 
                              index = 'id', time_index = 'last_review')


# In[ ]:


es


# ##### Automated Deep Feature Synthesis
# > In addition to manually specifying aggregation and transformation feature primitives, we can let featuretools automatically generate many new features. We do this by making the same ft.dfs function call, but without passing in any primitives. We just set the max_depth parameter and featuretools will automatically try many all combinations of feature primitives to the ordered depth.
# 
# When running on large datasets, this process can take quite a while, but for our example data, it will be relatively quick. For this call, we only need to specify the entityset, the target_entity, and the max_depth.

# In[ ]:


# Perform deep feature synthesis without specifying primitives
features, feature_names = ft.dfs(entityset=es, target_entity='airbnb', 
                                 max_depth = 2)


# In[ ]:


print(feature_names)
features.head()


# <b> Deep feature synthesis has created 5 new features out of the existing data! While we could have created all of these manually, I am glad to not have to write all that code by hand. </b>

# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
corr = features.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240,10,as_cmap=True),
            square=True, ax=ax)


# ### Predictive Modelling

# In[ ]:


features['price']=features['price'].replace(0,features['price'].mean())
features.drop(['name','host_name','host_id','latitude','longitude'], axis=1, inplace=True)


# In[ ]:


# Dummy variable
categorical_columns = ['neighbourhood_group','neighbourhood', 'room_type']
df_encode = pd.get_dummies(data = features, prefix = 'OHE', prefix_sep='_',
               columns = categorical_columns,
               drop_first =True,
              dtype='int8')
print(features.columns)


# In[ ]:


# Lets verify the dummay variable process
print('Columns in original data frame:\n',features.columns.values)
print('\nNumber of rows and columns in the dataset:',features.shape)
print('\nColumns in data frame after encoding dummy variable:\n',df_encode.columns.values)
print('\nNumber of rows and columns in the dataset:',df_encode.shape)


# In[ ]:


from scipy.stats import boxcox
y_bc,lam, ci= boxcox(df_encode['price'],alpha=0.05)
ci,lam


# In[ ]:


x = df_encode.drop(['price'], axis = 1)
y = df_encode.price
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=353)


# In[ ]:


clfs = []
seed = 3

clfs.append(("LinearRegression", 
             Pipeline([("Scaler", StandardScaler()),
                       ("LogReg", LinearRegression())])))

clfs.append(("XGB",
             Pipeline([("Scaler", StandardScaler()),
                       ("XGB", XGBRegressor())]))) 
clfs.append(("KNN", 
             Pipeline([("Scaler", StandardScaler()),
                       ("KNN", KNeighborsRegressor())]))) 

clfs.append(("DTR", 
             Pipeline([("Scaler", StandardScaler()),
                       ("DecisionTrees", DecisionTreeRegressor())]))) 

clfs.append(("RFRegressor", 
             Pipeline([("Scaler", StandardScaler()),
                       ("RandomForest", RandomForestRegressor())]))) 

clfs.append(("GBRegressor", 
             Pipeline([("Scaler", StandardScaler()),
                       ("GradientBoosting", GradientBoostingRegressor(max_features=15, 
                                                                       n_estimators=600))]))) 

clfs.append(("MLP", 
             Pipeline([("Scaler", StandardScaler()),
                       ("MLP Regressor", MLPRegressor())])))


clfs.append(("EXT Regressor",
             Pipeline([("Scaler", StandardScaler()),
                       ("ExtraTrees", ExtraTreeRegressor())])))
clfs.append(("SV Regressor",
             Pipeline([("Scaler", StandardScaler()),
                       ("ExtraTrees", SVR())])))

scoring = 'r2'
n_folds = 10
msgs = []
results, names  = [], [] 

for name, model  in clfs:
    kfold = KFold(n_splits=n_folds, random_state=seed)
    cv_results = cross_val_score(model, x_train, y_train, 
                                 cv=kfold, scoring=scoring, n_jobs=-1)    
    names.append(name)
    results.append(cv_results)    
    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  
                               cv_results.std())
    msgs.append(msg)
    print(msg)


# In[ ]:


lr = LinearRegression().fit(x_train,y_train)

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

print(lr.score(x_test,y_test))


# In[ ]:


forest = RandomForestRegressor(n_estimators = 100,
                              criterion = 'mse',
                              random_state = 1,
                              n_jobs = -1)
forest.fit(x_train,y_train)
forest_train_pred = forest.predict(x_train)
forest_test_pred = forest.predict(x_test)

print('MSE train data: %.3f, MSE test data: %.3f' % (
mean_squared_error(y_train,forest_train_pred),
mean_squared_error(y_test,forest_test_pred)))
print('R2 train data: %.3f, R2 test data: %.3f' % (
r2_score(y_train,forest_train_pred),
r2_score(y_test,forest_test_pred)))


# In[ ]:


plt.figure(figsize=(10,6))

plt.scatter(forest_train_pred,forest_train_pred - y_train,
          c = 'black', marker = 'o', s = 35, alpha = 0.5,
          label = 'Train data')
plt.scatter(forest_test_pred,forest_test_pred - y_test,
          c = 'c', marker = 'o', s = 35, alpha = 0.7,
          label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Tailings')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = 0, xmax = 60000, lw = 2, color = 'red')
plt.show()


# In[ ]:


## Applying L2 Regularization
scorer = make_scorer(mean_squared_error, greater_is_better = False)

def rmse_cv_train(model):
    rmse= np.sqrt(-cross_val_score(model, x_train, y_train, scoring = scorer, cv = 10))
    return(rmse)

def rmse_cv_test(model):
    rmse= np.sqrt(-cross_val_score(model, x_test, y_test, scoring = scorer, cv = 10))
    return(rmse)
ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
ridge.fit(x_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)
ridge.fit(x_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)

print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())
print("Ridge RMSE on Test set :", rmse_cv_test(ridge).mean())
y_train_rdg = ridge.predict(x_train)
y_test_rdg = ridge.predict(x_test)

# Plot residuals
plt.scatter(y_train_rdg, y_train_rdg - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_rdg, y_test_rdg - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with Ridge regularization")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# Plot predictions
plt.scatter(y_train_rdg, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_rdg, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with Ridge regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()


# ## If you find this Kernal Awesome & Useful too Please Hit the Upvote!!!
# ##### Cheers!!!!
