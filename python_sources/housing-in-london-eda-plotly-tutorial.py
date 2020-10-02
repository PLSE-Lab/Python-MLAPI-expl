#!/usr/bin/env python
# coding: utf-8

# # Recently I discovered a good library called 'Plotly' which allowed me to implement fansy highly interactive graphs with only few lines, This notebook would be a tutorial on it

# # Some useful documentation to look at first:

# ## Plotly Library Documentation: [https://plotly.com/python/](https://plotly.com/python/) (official doc)
# ## Plotly.Express Module Documentation : [https://plotly.com/python/plotly-express/](https://plotly.com/python/plotly-express/) (fast plotting)
# ## Cufflinks Repo: [https://github.com/santosjorge/cufflinks](https://github.com/santosjorge/cufflinks) (allow direct plot from pd.DataFrame using .iplot())

# ---

# # Libraries

# In[ ]:


import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import plotly
import plotly.express as px
from plotly.offline import iplot
import cufflinks as cf


# ---

# # Dataset Examine

# In[ ]:


df_month = pd.read_csv("/kaggle/input/housing-in-london/housing_in_london_monthly_variables.csv")
df_month.date = pd.to_datetime(df_month.date)


# ## Samples from tail of dataframe

# In[ ]:


df_month.tail()


# ## Fraction of NaN entries for each feature

# In[ ]:


df_month.isnull().sum() / df_month.shape[0]


# - 45% of entries were NaN for `number of crime` feature. That was nearly a half
# - 0.69% of entries were NaN for `houses_sold` feature

# ## Brief report on dataset

# In[ ]:


# ProfileReport(df_month)


# ---

# # EDA Phase

# # Line Charts

# ## Remember, you can always get help using help() function when you don't know how to start
# ## This time, I want to examine the change in hoursing price over time against other features to find some insights

# In[ ]:


#help(px.line)


# ## What was the trend of average housing prices for each area from 1995 to 2019?

# In[ ]:


px.line(df_month,x='date',y='average_price',color='area',title='Average Price by Area Trend')


# - `kensington and chelsea` area had the greatest trend
# - but how much though specifically?

# In[ ]:


kc_2019 = df_month[(df_month.area == 'kensington and chelsea') & (df_month.date.dt.year == 2019)].average_price.mean()
kc_1995 = df_month[(df_month.area == 'kensington and chelsea') & (df_month.date.dt.year == 1995)].average_price.mean()
display(round(kc_2019 - kc_1995,2), str(round((kc_2019 / kc_1995 * 100),2)) + '%')


# - In 24 years, the average housing price in `kensington and chelsea` area was increased 1,054,267.33 GBP
# - Which was 646.66% more comparing to the price in 1995

# ## What was the trend of average housing price between boroughs and not boroughs?

# In[ ]:


px.line(df_month.groupby(['date','borough_flag']).average_price.mean().to_frame().reset_index(),x='date',y='average_price',color='borough_flag',title='Average Housing Price by Borough_Flag Trend')


# ## What was the trend of number of houses sold for each area from 1995 to 2019?

# In[ ]:


px.line(df_month,x='date',y='houses_sold',color='area',title='Number of Houses Sold Trend')


# ## What was the trend of crimes for each area from 1995 to 2019?

# In[ ]:


px.line(df_month,x='date',y='no_of_crimes',color='area',title='Crime')


# - Crime records started from Jan,2001
# - `westminster` had a high crime rate
# - `city of london` had 0 crime at all time?

# # Bar Chart

# In[ ]:


df_month


# ## How many areas were boroughs?

# In[ ]:


px.bar(df_month.groupby("borough_flag").area.nunique().reset_index(),x="borough_flag",y="area")


# ## What's the number of houses sold in each area in 2019?

# In[ ]:


houses_2019 = df_month[df_month.date.dt.year == 2019].groupby("area").houses_sold.sum().reset_index()
px.bar(houses_2019,x="area",y="houses_sold")


# # Pie Charts

# ## What was the total value (average_price * houses_sold) in 2019? and by each area?

# In[ ]:


df_month['total'] = df_month.average_price * df_month.houses_sold
total_2019 = df_month[df_month.date.dt.year == 2019].groupby('area').total.sum().reset_index()
print("Total value was %s GBP" % '{:,.2f}'.format(total_2019.total.sum()))


# In[ ]:


px.pie(total_2019,values='total',names='area')

