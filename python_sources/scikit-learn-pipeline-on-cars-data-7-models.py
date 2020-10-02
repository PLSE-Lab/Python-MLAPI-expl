#!/usr/bin/env python
# coding: utf-8

# # Cars Dataset: (EDA + Models pipelines)
# 
# Whenever you start working on any data related projects, the work is mostly divided into two parts.
# 
# ### 1. [Exploratory Data Analysis](#eda)
# In the initial stages of any project, it is crucial to grasp the data intuitively. This is a research phase, where you perform calculate various statistics related to the data, perform statistical tests and visualize the data. This phase should quickly help one understand the strengths and issues with data. This phase also contains a lot of data cleaning and data wrangling steps to make data amenable for modeling. Sometimes, this phase also involves formulating hypotheses and formulating questions that can be solved analytically or via building statistical models on data.
# 
# ### 2. [Model Pipelines](#pipeline)
# Once it's clear that the data needs to be modeled for further applications, the second phase involved building modeling pipelines that can quickly produce a few simple models that can become baseline models against which all the other complicated models need to be evaluated and compared. Scikit-learn is a wonderfully designed package that quickly lets one build a robust machine learning pipelines. It also lets one iterate over the established pipelines very easily. The present notebook demonstrated such a pipeline for cars dataset to show how pipelines can be built within scikit-learn package.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy.interpolate import make_interp_spline, BSpline
import warnings
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')


# In[ ]:


df = (pd.read_csv('/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv')
      .drop('Unnamed: 0', axis=1))
# Dropping 'Unnamed: 0' feature. (It's an artifact of not saving the csv file properly. One should save any csv with index=False argument.) 


# <a id="eda"></a>
# ## Exploratory Data Analysis
# 
# 
# The dataset is perfect for starting for a simple exploration demonstration. Let's see what the attributes are:

# In[ ]:


df.sample(10)


# Let's get another (equivalent) perspective on the data.

# In[ ]:


df.sample(10).T


# In[ ]:


df.drop(['vin', 'lot'], axis=1, inplace=True)


# In[ ]:


df.describe()


# So, only 3 attributes are numerical. But, perhaps, 'condition' can be converted to a numerical column, as well. Even the color could be converted to a wavelength. So, color can be converted to numeric as well. State can probably be converted to numerical columns as well, by providing the longitude and latitude. 

# In[ ]:


df.info()


# In[ ]:


for col in df.columns:
    print('{:15} : {:5} : {:}'.format(col, df[col].nunique(), df[col].dtype))


# A simple distinct count for the numerical columns and plot the graphs for numerical ones:

# In[ ]:


num_feat = df.select_dtypes(include=np.number).columns
cat_feat = df.select_dtypes(include=['object']).columns


# In[ ]:


fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(20, 5))
for i in range(3):
    ax[i].hist(df[num_feat[i]])
    ax[i].set_title(num_feat[i])
plt.tight_layout()
plt.show()


# This is as expected. Price probably follows a lognormal distribution, the data probably is biased towards newer cars, and the mileage also probably follows a lognormal distribution.

# Is there a difference in the price due to the 'Title Status'?

# In[ ]:


round(df.groupby('title_status').agg({'price':'mean'}))


# In[ ]:


df['state'] = df['state'].str.capitalize()
plt.figure(figsize=(20, 20))
sns.heatmap(pd.pivot_table(df, values='price', index=['state'], columns=['year'], aggfunc='mean').iloc[:, 20:], annot=True, fmt='g', cmap='Greens')
plt.show()


# In[ ]:


sns.pairplot(df)
plt.show()


# In[ ]:


fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20, 5))
df[df.year > 2010].plot.scatter(x='price', y='mileage', c='year', cmap='coolwarm', ax=ax[0], ylim=(0, 2*10**5))
df[df.year > 1990].plot.hexbin(x='price', y='mileage', gridsize=15, cmap='coolwarm', ax=ax[1], ylim=(0, 2*10**5))
plt.tight_layout()
plt.show()


# ## Most of the cars are priced between \$13k and \$20k and have mileages between 50k to 100k.

# # Spline Interpolation
# 
# Since, the size of the dataset is really small, to see the relationships between different variables, it might be interesting to interpolate with splines, before working on rigorous linear/non-linear models to fit the data.

# In[ ]:


year = df[df['year'] >= 1995].groupby('year').agg({'price':'mean'}).index.to_numpy()
price = df[df['year'] >= 1995].groupby('year').agg({'price':'mean'}).to_numpy()
mileage = df[df['year'] >= 1995].groupby('year').agg({'mileage':'mean'}).to_numpy()

xnew = np.linspace(year.min(), year.max(), 4) 
spl1 = make_interp_spline(year, price, k=3)
power_smooth1 = spl1(xnew)
spl2 = make_interp_spline(year, mileage, k=3)
power_smooth2 = spl2(xnew)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
df[df['year'] >= 1995].groupby('year').agg({'price':'mean'}).plot(kind='line', marker='o', linewidth=1, ax=ax[0])
df[df['year'] >= 1995].groupby('year').agg({'mileage':'mean'}).plot(kind='line', marker='o', linewidth=1, ax=ax[1])
ax[0].plot(xnew, power_smooth1, linewidth=2)
ax[1].plot(xnew, power_smooth2, linewidth=2)
ax[0].set_title('Average Car Price by Model year')
ax[1].set_title('Average Mileage by Model year')
ax[0].set_ylabel('Price ($)')
ax[1].set_ylabel('Mileage')
ax[0].legend(['Actual Price', 'Spline Fit'])
ax[1].legend(['Actual Mileage', 'Spline Fit'])
plt.show()


# Two trends emerge:
# 1. Average price of a car model decreases exponentially with the age of the car.
# 2. Average mileage of a car seems to decrease almost linearly with the age of the car.
# 
# Note, these two variables are not independent.

# <a id="pipeline"></a>
# ## Model Pipelines
# #### Let's build a general modeling pipeline using sklearn and test 7 different models

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


num_feat = df.drop('price', axis=1).select_dtypes(include=np.number).columns
cat_feat = df.drop('price', axis=1).select_dtypes(include=['object']).columns
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()) #('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_feat),
        ('cat', categorical_transformer, cat_feat)
    ])


# ## Let's test the pipeline with Linear Regression Model

# In[ ]:


from sklearn.linear_model import LinearRegression
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',  LinearRegression())
])

model = pipe.fit(X_train, y_train)
y_pred = model.predict(X_test)

model.score(X_test, y_test)


# In[ ]:


from time import time
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score


# In[ ]:


results = pd.DataFrame(columns=['Name', 'Scores', 'StdDev', 'Time(s)'])

for model in [
    DummyRegressor,
    LinearRegression, 
    KNeighborsRegressor,
    DecisionTreeRegressor,
    RandomForestRegressor, 
    GradientBoostingRegressor,
    XGBRegressor,
    LGBMRegressor,
#     MLPRegressor
]:
    pipe = make_pipeline(preprocessor, model())
    start_time = time()
    kfold = StratifiedKFold(n_splits=10, random_state=1)
    scores = cross_val_score(pipe, X_train, 
                             y_train, scoring='r2', cv=kfold)
    time_mod = time() - start_time
    results = results.append({
        'Name' : model.__name__, 
        'Scores' : round(scores.mean(), 2), 
        'StdDev' : round(scores.std(), 2), 
        'Time(s)': round(time_mod, 2)
    }, ignore_index=True)
    del pipe
    print('Analyzed {}.'.format(model.__name__))
print('Done!')
results = results.sort_values('Scores', ascending=False)


# In[ ]:


results


# In[ ]:


results.set_index('Name')['Scores'].plot(kind='barh')
plt.ylabel('Model Name')
plt.xlabel('Model Score ($r^2$)')
plt.xlim(0, 1)
plt.show()


# In[ ]:


plt.errorbar(results['Scores'], results['Name'], xerr=results['StdDev'], linestyle='None', marker='o')
plt.xticks(rotation=90)
plt.xlabel('Model Score ($r^2$) and $\sigma$')
plt.ylabel('Model Name')
plt.xlim(0, 1.1)
plt.show()


# ### To summarize, we've built robust model pipelines using sklearn, after performing exploratory data analysis. These pipelines can be iterated simply by changing a few parameters only at a single place (which avoids repetition).
