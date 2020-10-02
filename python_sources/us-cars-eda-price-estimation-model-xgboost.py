#!/usr/bin/env python
# coding: utf-8

# # This notebook contains exploratory data analysis and price prediction on US cars

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


def plot_bar_vertical(df,figsize=(10,15),xlabel='Count Number',color='tab:blue'):
    ax = df.plot.barh(figsize=figsize,color=color)
    plt.xlabel(xlabel)
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width(),p.get_y() + p.get_height()/2,f'{int(p.get_width())}')

        
def plot_donut_chart(df,figsize=(10,15),subplots=True,radius=0.7,pctdistance=0.8):
    df.plot.pie(figsize=figsize,subplots=subplots,pctdistance=pctdistance,explode=[0.1 for x in df.index])
    centre_circle = plt.Circle((0,0),radius,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)


# # Load Dataset

# In[ ]:


car_us_df = pd.read_csv("/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv",index_col=0)
car_us_df.head()


# In[ ]:


car_us_df.info()


# In[ ]:


car_us_df.describe()


# ---

# # Dataset Report

# In[ ]:


ProfileReport(car_us_df)


# ---

# # EDA Phase

# ## Feature -> `price`

# ## What's the distribution of the selling price?

# In[ ]:


ax = car_us_df.price.plot.hist(figsize=(12,10))
plt.xlabel('Price')
plt.xticks(np.arange(0,100000,10000))
for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/4,p.get_y() + p.get_height(),f'{int(p.get_height())}')


# In[ ]:


car_us_df.price.describe()


# - Most selling price fell in the range from \$0 to \$30000
# - The average price was \$18767
# - Let's take a closer look at special cases, cars worth \$0 and greater than \$30000

# ## Cars with price tag \$0

# In[ ]:


special_cases_price_zero = car_us_df[(car_us_df.price == 0)]
display(special_cases_price_zero.head(10))
special_cases_price_zero.shape


# - 43 cars had price tags \$0

# ### What were their brands and models?

# In[ ]:


plot_bar_vertical(special_cases_price_zero.groupby(['brand','model']).brand.count().sort_values())


# In[ ]:


ax = plot_donut_chart(special_cases_price_zero.groupby(['brand','model']).brand.count(),radius=0.8)


# - 13 cars from `ford`,`door`, next was 9 cars from `chevrolet`,`door`

# ### A lot of them had damaged records, which made sense that the prices were extremely low, how about those 'clean' vehicles but yet still for \$0?

# In[ ]:


special_cases_price_zero[special_cases_price_zero.title_status == 'clean vehicle']


# - Two cars sold for \$0 while had no damage records.
# - first case had a mileage of 76858 mi which was considerably old
# - second case was a 2009 white heartland sundance, no mileage on it at all, sitting in a place in pennsylvania covered in dirt for 11 years since it was manufactured, yet it was tagged for \$0

# ## Cars with tag more than \$ 30000

# In[ ]:


special_cases_price_great = car_us_df[car_us_df.price >= 30000]
special_cases_price_great.head()


# ### What car types were the most common among them?

# In[ ]:


plot_bar_vertical(special_cases_price_great.groupby(['brand','model']).brand.count().sort_values().tail(10))


# - `ford`,`f-150` was the most common car type among cars over \$30000

# ### The most expensive one?

# In[ ]:


car_us_df[car_us_df.price == car_us_df.price.max()]


# - Most expensive one was the `mercedes-benz`,`sl-class` sold for \$84900 in `florida`

# ---

# ## Feature -> `brand`,`model`

# ## What were most common manufacturer and thier models?

# In[ ]:


plot_bar_vertical(car_us_df.groupby(['brand','model']).brand.count().sort_values().tail(10),color='tab:green')


# - 363 cars from `ford`,`door` 
# - Most common manufacturers were `ford`,`dodge`,`nissan`,`chevrolet`
# - Most common models were `door`,`f-150`,`caravan`

# ## What was the average price for each `ford`,`model`?

# In[ ]:


plot_bar_vertical(car_us_df.groupby(['brand','model']).price.mean().sort_values().tail(10),(12,15),'price',color='tab:green')


# - `mercedes-benz`,`sl-class` was the most expensive car type with a price of \$84900 , \$29300 more than second place `lexus`,`gx`
# - Luxury brands like `mercedes-benz`,`lexus`,`bmw`,`harley-davidson` occupied the top of the chart

# ---

# ## Feature -> `title_status`

# In[ ]:


section_color = 'tab:brown'


# ### Clean vs Salvage, How many cars were under salvage insurance?

# In[ ]:


plot_bar_vertical(car_us_df.title_status.value_counts(),(10,6),color=section_color)


# In[ ]:


print(f'Salvage insuranced rate: {round((car_us_df.title_status.value_counts()[1] / car_us_df.title_status.value_counts()[0]) * 100,2)}%')


# - Most cars were in clean condition while only 163 were damaged in the past
# - Close to 7% of all cars were salvage insuranced

# ### What was the price for clean vehicles? and for salvage insuranced?

# In[ ]:


plot_bar_vertical(car_us_df.groupby('title_status').price.mean(),(12,6),'Price',color=section_color)


# - In average, salvage insuranced cars were much cheaper than clean with a gap amount of \$17581
# - Clean vehicles' average price was close to the mean value we seen in the beginning as expected

# In[ ]:


car_us_df


# ---

# ## Feature -> `mileage`

# In[ ]:


section_color ='tab:cyan'


# ### What was the relation between `price` and `mileage`?

# In[ ]:


ax = car_us_df.plot(kind='scatter',x='mileage',y='price',figsize=(10,8),color=section_color)


# - An exponentially decrase relationship could be obtained
# - As `mileage` increased, its `price` dropped exponentially

# ### What was the maximum mileage? and the car with that number?

# In[ ]:


print(f'{int(car_us_df.mileage.max())} miles')
car_us_df[car_us_df.mileage == car_us_df.mileage.max()]


# - the maximum mileage was 1,017,936 miles and that was a pretty large one
# - It was a `2010`,`peterbilt`,`truck` with unknown color located in georgia which was under salvage insurance for a price of \$1025

# ## Feature -> `color`

# In[ ]:


section_color = 'tab:orange'


# ### What colors were most common?

# In[ ]:


top_colors = [color if color != 'no_color' else '#FFFFFF' for color in car_us_df.color.value_counts().sort_values().tail(10).index]
ax = car_us_df.color.value_counts().sort_values().tail(10).plot.barh(color=top_colors,figsize=(12,10))
ax.set_facecolor('purple')
for p in ax.patches:
    ax.text(p.get_x() + p.get_width(),p.get_y() + p.get_height()/2,f'{int(p.get_width())}')


# - `White` color was the most common color
# - `no_color` at 7th place was filled with white color but actually it would be transparent instead
# - Common colors were `white`,`black`,`gray`,`silver`

# ## Feature -> `state`

# In[ ]:


section_color ='tab:purple'


# ## Which state they were mostly located?

# In[ ]:


plt.ylabel('US State')
plot_bar_vertical(car_us_df.state.value_counts().sort_values().tail(10),color=section_color)


# - 299 cars were located in pennsylvania, following by 246 cars in florida

# # Price Estimator

# ## A price estimator could give a reasonable price for a new coming car according to the market based on given conditions

# ### Make a copy of original dataframe and dropped features that had no correlation with the target feature `price`

# In[ ]:


price_estimator_df = car_us_df.copy()
features_to_drop = ['vin','lot','country','condition']
price_estimator_df.drop(features_to_drop,axis=1,inplace=True)
price_estimator_df


# ### Found all catgorical features

# In[ ]:


cat_features = [col for col in price_estimator_df.select_dtypes('object')]
cat_features


# ### Obersrving from dataframe.info(), no missing entries here.

# ### Separate target feature `price` and other training features

# In[ ]:


train = price_estimator_df.drop('price',axis=1)
target = price_estimator_df.price


# ### Import libraries

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_predict,GridSearchCV
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


# ### Cardinality for each categorical feature

# In[ ]:


[(col,train[col].nunique()) for col in cat_features]


# - Cardinality for each categorical feature where basically every feature except title_status had more than 10 unique elements

# ### I use Label Encoder here instead of One-Hot Encoder due to some features had large cardinality numbers which could generate too many features to the dataset if using One-Hot Encoding

# In[ ]:


cat_transformer = LabelEncoder()

for col in cat_features:
    train[col] = cat_transformer.fit_transform(train[col])
train


# ### Split the dataset into training set and validation set

# In[ ]:


X_train,X_val,y_train,y_val = train_test_split(train,target,test_size=0.2,shuffle=True,random_state=20)


# ### I chose XGBRegressor since it was a regression task, you could try other regression models like Lasso,ElasticNet etc.

# ### Baseline Model

# In[ ]:


model = XGBRegressor()
model.fit(X_train,y_train)
r2_score(y_pred=model.predict(X_val),y_true=y_val)


# - With default settings, model had an r2 score of 0.66

# ### Using grid search to find parameters that had best performance (valuation metric R-squared `r2`)

# In[ ]:


params_set1 = {'max_depth':[3,4,5],'gamma':[0,1,5]}

model = XGBRegressor()

clf = GridSearchCV(model,params_set1,cv=KFold(n_splits=5),scoring='r2',refit=True)
clf.fit(X_train,y_train)


display(clf.best_score_,clf.best_params_)


# - Grid search for `max_depth` and `gamma`
# - Best one had a r2 score of 0.676
# - Optimal parameters shown above

# In[ ]:


params_set2 = {'n_estimators':[50,100,500,1000],'learning_rate':[0.01,0.03,0.05]}

model = XGBRegressor(max_depth=4,gamma=0)

clf = GridSearchCV(model,params_set2,cv=KFold(n_splits=5),scoring='r2',refit=True)
clf.fit(X_train,y_train)


display(clf.best_score_,clf.best_params_)


# - Second parameter set on `n_estimators` and `learning_rate`
# - Score was improved to 0.68

# In[ ]:


model = XGBRegressor(gamma=0, max_depth= 4,learning_rate=0.03,n_estimators=1000)
model.fit(X_train,y_train)
r2_score(y_pred=model.predict(X_val),y_true=y_val)


# - Finally, I apply those optimal parameters to the model and retrain on the dataset
# - r2 score was `0.69`, an increase of 0.03 comparing to the baseline score `0.66`

# # Thanks for reading
