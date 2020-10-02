#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from plotly import tools
import plotly
plotly.offline.init_notebook_mode(connected=True) #to plot graph with offline mode in Jupyter notebook
import plotly.graph_objs as go

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


# In[ ]:


# Read CSV
df = pd.read_csv("../input/winemag-data-130k-v2.csv")

#remove unnecessary column
df_cleaned = df.drop('Unnamed: 0', 1)


# <font size="6">**Basic Exploratory Data Analysis**</font>

# In[ ]:


pd.set_option('display.max_columns', 500)
print(df_cleaned.shape)
#df_cleaned.info()
df_cleaned.head()


# <font size="6">**Wine Production Breadown By Countries**</font>
# <br>
# Which country has the most wine produce?

# In[ ]:


###########################
#Analyse based on Country#
#########################
countries = df_cleaned['country'].value_counts().sort_values(ascending=True)
trace_countries = [go.Pie(
                   labels = countries.index,
                   values = countries.values)]

trace_countries_layout = go.Layout(
                            title = 'Market Share of Wine By Countries')

trace_countries_fig = go.Figure(data = trace_countries, layout = trace_countries_layout)


plotly.offline.iplot(trace_countries_fig, filename = 'country_produce_wine')


# * US is the largest wine producer which taking almost half of the market share.
# * France and Italy is the second and third which provide wine.

# <font size="6">**Insight of Winery Rating/Points & Price**</font>
# <br>
# What is the average for a bottle of wine's point and its price?

# In[ ]:


#####################
#Bivariate Analysis#
###################
print(df_cleaned[['points','price']].describe())

points_price_col = ['points','price']
df_cleaned[points_price_col].plot(kind='box',subplots=True, title = 'Boxplots on Wine Points & Price')


# In[ ]:


##############################
#Distribution of Wine Rating#
############################
hist_points = [go.Histogram(
                x = df_cleaned['points']
)]

hist_points_layout = go.Layout(
                        title = 'Distribution of Wine Points/Rating',
                        xaxis = dict(title = 'Points'),
                        yaxis = dict(title = 'Number of rating'))

hist_points_fig = go.Figure(data = hist_points, layout = hist_points_layout)
    
plotly.offline.iplot(hist_points_fig, filename = 'distribution_wine_points')

#############################
#Distribution of Wine Price#
###########################
hist_price = [go.Histogram(
                x = df_cleaned['price'])]

hist_price_layout = go.Layout(
                        title = 'Distribution of Wine Price',
                        xaxis = dict(title = 'Price'),
                        yaxis = dict(title = 'Frequency'))


hist_price_fig = go.Figure(data = hist_price, layout = hist_price_layout)

plotly.offline.iplot(hist_price_fig, filename='distribution_wine_price')


# * The wine point is distributed normally which its average is 88.447138.
# * The wine price from the datasets shows that it is on right skewed (with some wine price that goes up USD$3300.00); thus, the median of wine price in overall is USD$25.00.
# * The reason that mean is being used in wine point, and median is used in wine price is due to wine point data is closed to normally distributed while wine price data is right skewed. 

# <font size="6">**Points Vs Price**</font>
# <br>
# Does wine's point affect its price?

# In[ ]:


#Impute NaN field in price with its median price
df_cleaned[['price']] = df_cleaned[['price']].fillna(value=25)
df_cleaned.corr()


# In[ ]:


#split data into x-array and y-array
X_var = df_cleaned[['points']]
y_var = df_cleaned['price']


# In[ ]:


df_cleaned.plot(x='points',y='price',style='o')


# * The correlation between points and price is merely 0.399231 which indicate both variable have weak relationship.
# * From the plot graph, the data point is not linear. Thus, the points of wines have no direct influence to its price. However, higher price of wine tends to have high rating points.

# <font size="6">**Conclusion**</font>
# * The US is the largest producing wine and having almost half of the market share.
# * The average cost of wines on market are normally USD$25.00; However, some wine can go up to USD$3300.00.
# * Typically, wine rated at 88.447138 points and above are normally accepted quality wine.
# * The points of wine rate by testers does not have significant influence towards its price.
