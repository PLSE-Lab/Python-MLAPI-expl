#!/usr/bin/env python
# coding: utf-8

# # USA Housing data analysis
# The goal of this analysis is to determine the main driving factors of USA Housing
# We will be analysing data across all states and regions in the US.
#  - For numerical factors, we will use a Linear Regression Model to predict the effect of changes in each factor
#  - For categorical factors, we will use data exploration and visualisation to contextualize the data.

# #### For data manipulation and analysis

# In[ ]:


import numpy as np
import pandas as pd


# #### For data visualisation (use the % magic method to see visualisations in jupyter notebook)

# In[ ]:


import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### For the machine learning process

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# #### read the data into a dataframe

# In[ ]:


df = pd.read_csv('../input/USA_Housing.csv', engine='python')


# # Data cleaning

# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# #### Looking at the data there are no empty values and the columns are the correct data type.
# #### A change that needs to be made however, is that we need to format the address in a way that we can use to analyse the data.
# #### To do this, we can extract the state and assign it to a region.
# #### From this information we can then plot the avg. house price per region.

# In[ ]:


def get_state(x):
    words = [i for i in x.split(' ')]
    return words[::-1][1]

regions = {'New England':['CT','MA','ME','NH','RI','VT'],'Mid Atlantic':['PA','NY','DE','NJ','MD'],
              'South':['AL','AR','KY','GA','MS','LA','SC','NC','VA','WV','TN'],'Texas':['TX'],'Florida':['FL'],
               'Midwest':['IA','IL','IN','MI','MN','WI','OH','MO'],
              'Great Plains':['SD','ND','KS','NE','OK'],'Rocky Mountains':['CO','MT','ID','WY'],
              'South West':['NV','AZ','NM','UT'],'California':['CA'],'Pacific Northwest':['WA','OR'],'Alaska':['AK'],
              'Hawaii':['HI'],'Commonwealth':['DC','MH'],'Military':['AE','AA','AP']}
   
def get_region(x):
    for i in list(regions.keys()):
        if x in regions[i]:
            return i


# In[ ]:


list(regions.keys())


# In[ ]:


df['State'] = df['Address'].apply(lambda x:get_state(x))
df['Region'] = df['State'].apply(lambda x:get_region(x))


# In[ ]:


df.head()


# # Categorical Factors
#  - Location (at both a regional and state level)

# In[ ]:


dfreg = df[['Region','State','Price']]

regionmean = dfreg.groupby('Region').mean()

AvPerRegion = pd.DataFrame(regionmean).sort_values('Price')

AvPerRegion.plot(kind='bar',figsize=(16,7),fontsize=20)
plt.xlabel('Region', fontsize=20)
plt.ylabel('Avg. house price', fontsize=20)
plt.suptitle('Avg. House Price by Region', fontsize=30)


# #### A breakdown of average house price per state supports the regional analysis

# In[ ]:


stateprice = dfreg.groupby('State').mean()
stateprice.columns = ['Average house price']

statecodes = [i for i in stateprice.index]

px.choropleth(data_frame = stateprice,
               locations = statecodes,
              locationmode = 'USA-states',
              scope='usa',
             color = 'Average house price',
             color_continuous_scale = 'Blues')


# # Numerical Factors
#  - Avg. area income
#  - Avg. area House Age
#  - Avg. area number of rooms
#  - Area population

# ## Linear Regression

# In[ ]:


df.head()


# In[ ]:


X = df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
y = df['Price']


# #### Using train_test_split to get my training and test values and then fitting the the linear regression model

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=100)


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)


# In[ ]:


coef = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficients'])
coef.index.name = 'Independent variables'
coef


# # Conclusions to be made from the Linear Regression Model:
#  - Firstly, 
#      the independent factors that have most effect on the price of a home in any given area are:
#      - Avg. Area House Age
#      - Avg. Area Number of Rooms
#      the independent factor that has a moderate effect on the price of a home in any given area is:
#      - Avg. Area Income 
#      the independent factors that have a negligable effect on the price of a home in any given area are:
#      - Avg. Area Number of Rooms
#      - Area Population
#  - More specifically,   
#      - From a dollar increase in Avg. Area Income, we can expect a 21.61 dollar increase in the average price of a home in that area.
#      - From a 1 year increase in Avg. Area House Age, we can expect a 165804.74 dollar increase in the price of a home in that area.
#      - If the average number of rooms across the area increases by one, we can expect an increase of 120921.53 dollars in the price of a home in that area.
#      - If the average number of bedrooms across the area increases by one, we can expect a decrease of 16.56 dollars in the price of a home in that area.
#      - If the area population increases by one, we can expect an increase of 15.21 dollars in the price of a home in that area.

# In[ ]:


predictions = lm.predict(X_test)
predictions


# #### Here we plot the actual price against the predicted price 

# In[ ]:


sns.set_style('darkgrid')
sns.scatterplot(y_test,predictions).set(xlabel='Price',ylabel='Predicted Price')


# # Assessing the performance of the Linear Regression Model

# #### Below we then calculate the error and plot the amount of house prices against each error as we can see, the majority of house prices experienced little error as the error is normally distributed, this means that the model worked well

# In[ ]:


sns.distplot(y_test-predictions,bins=30).set(xlabel='Price',ylabel='Error')


# #### Lastly I use metrics from the sklearn library in order to measure the effectiveness of the linear regression model

# In[ ]:


mae = metrics.mean_absolute_error(y_true=y_test,y_pred=predictions)
mse = metrics.mean_squared_error(y_true=y_test,y_pred=predictions)
rmse = np.sqrt(mse)


# In[ ]:


errors = pd.DataFrame([mae,mse,rmse],columns=['error measure'],index=['mae','mse','rmse'])
errors

