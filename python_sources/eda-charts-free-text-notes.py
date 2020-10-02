#!/usr/bin/env python
# coding: utf-8

# # Hello
# This notebook is followed by the Forecasting one <br>
# in which everything it more organized. <br>
# 
# This notebook was made within free creative, brain <br>
# stormer environment. The less the mind was concern <br>
# about formal aesthetics the more creative it is :)<br>
# 
# But don't worry, the lack of text and explanation <br>
# here is covered at Forecasting notebook.<br>
# However, the art and beauty on charts are here :)<br>
# 
# Enjoy!

# In[ ]:


import pandas as pd

import my_dao
import process
import pretties
import time_utils
import stats
import plotter

import warnings
from bokeh.plotting import show, output_notebook


# In[ ]:


pretties.max_data_frame_columns()
pretties.decimal_notation()
output_notebook()
warnings.filterwarnings('ignore')


# # <font color="darkred">Relations</font>
# with target variable

# # walmart-recruiting-store-sales-forecasting
# https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting

# In[ ]:


train = my_dao.load_dataset("train")
train = train.groupby("store_dept").apply(process.train_sales_semantic_enrichment)

feat = my_dao.load_features()
feat = process.features_semantic_enrichment(feat)

stores = my_dao.load_stores()


# In[ ]:


train = train.merge(feat, how="left", left_on=["Store", "Date"], right_on=["Store", "Date"], suffixes=["", "_y"])
del train["IsHoliday_y"]
del train["timestamp_y"]
train = train.merge(stores, how="left", left_on=["Store"], right_on=["Store"])


# In[ ]:


cols = ['Date', 'Store', 'Dept', 'Weekly_Sales', 'pre_holiday', 'IsHoliday', 'pos_holiday', 'Fuel_Price', 
        'CPI', 'Unemployment', 'celsius', 'datetime', 'Type', 'sales_diff', 'sales_diff_p',
        'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 
        'Size', 'Temperature', 'timestamp', 'store_dept', "day_n", "week_n", "month_n", "year", "wm_date", "up_diff", "celsius_diff"]

train = train[cols].sort_values("timestamp")
train.sample(6)


# # <font color="navy">Date</font>

# In[ ]:


train["Date"].head(1).append(train["Date"].tail(1))


# In[ ]:


grouped_sales = train.groupby("Date")["Weekly_Sales"].median()
p = plotter.plot_time_series_count(grouped_sales.index, grouped_sales, color="navy", title="Weekly_Sales median vs Datetime", legend="All depts sales median", relative_y_axis=True, height=300)
p = plotter.time_series_count_painted_holidays(train, p=p, color="cyan", alpha=0.9)
show(p)


# In[ ]:


p = plotter.time_series_count_painted(train, title="Weekly_Sales median vs Datetime - Diamonds repesents Holidays", height=300, width=900)
p = plotter.time_series_count_painted_holidays(train, p=p, color="cyan", alpha=0.9)
show(p)


# ## Notes
# The positive peaks up is always followed by one or more negative peaks. <br>
# Week_n is the order of the week within a month.

# In[ ]:


from statsmodels.tsa.filters.hp_filter import hpfilter

gdp_cycle, gdp_trend = hpfilter(grouped_sales, lamb=10)
p = plotter.plot_time_series_count(grouped_sales.index, grouped_sales, color="navy", title="Weekly_Sales median vs Datetime", legend="All depts sales median", relative_y_axis=True, height=300)

grouped_sales = train.groupby("Date")["Weekly_Sales"].median()
p = plotter.plot_time_series_count(grouped_sales.index, gdp_trend, color="magenta", title="Weekly_Sales vs Datetime", legend="Hodrick-Prescott filter", relative_y_axis=True, height=300, p=p)
show(p)


# ## Notes
# Hodrick-Prescott filter gave us an interesting smoothing.

# In[ ]:


p = plotter.plot_error_values(train, "week_n", "sales_diff_p", drop_quantile=0.15, 
                           title="Weekly_Sales errors grouped by week_n")
show(p)


# ## Notes
# These error bars are seem to represent few, but interesting, <br>
# differences in Sales between ordered month weeks.

# In[ ]:


p = plotter.plot_error_values(train, "wm_date", "Weekly_Sales", drop_quantile=0.25, 
                           title="Weekly_Sales errors grouped by wm_date", width=1200)
show(p)


# ## Notes
# Not interesting to say... A peak in Thanksgiving day and Crhistmas

# In[ ]:


p = plotter.plot_error_values(train, "wm_date", "sales_diff_p", drop_quantile=0.25, 
                           title="sales_diff_p errors grouped by wm_date", width=1400)
show(p)


# ## Notes
# Weekly_Sales difference seems to show that people use to make bigger <br>
# shoppings at the first week of month. <br>
# People are the same no matter where in the world  :)

# # <font color="navy">Store</font>

# In[ ]:


train.groupby("Store")["Weekly_Sales"].mean().sort_values().plot.bar(title="Sales amout per store", figsize=(10, 3))


# ## Notes
# Store sizes dispersion

# # <font color="navy">IsHoliday</font>
# holiday evaluation weight = 5 <br>
# not holiday evaluation weight = 1

# In[ ]:


train.drop_duplicates(["Store", "Date"]).groupby("Store")["IsHoliday"].value_counts().plot.bar(title="Holidays count by store",figsize=(18,3))


# In[ ]:


stats.freq(train.drop_duplicates(["Store", "Date"])["IsHoliday"])


# In[ ]:


train.groupby("IsHoliday")["Weekly_Sales"].median().plot.bar(title="Weekly_Sales grouped by Holidays")


# In[ ]:


p = plotter.plot_error_values(train, "IsHoliday", "Weekly_Sales", drop_quantile=0.25, 
                           title="Weekly_Sales errors grouped by IsHoliday")
show(p)


# #### Before Holiday

# In[ ]:


train.groupby("pre_holiday")["Weekly_Sales"].median().plot.bar(title="Weekly_Sales BEFORE Holidays", figsize=(5,2))


# In[ ]:


p = plotter.plot_error_values(train, "pre_holiday", "Weekly_Sales", drop_quantile=0.25, 
                           title="Weekly_Sales errors grouped by pre_holiday", width=350, height=200)
show(p)


# #### After Holiday

# In[ ]:


train.groupby("pos_holiday")["Weekly_Sales"].median().plot.bar(title="Weekly_Sales AFTER Holidays", figsize=(5,2))


# In[ ]:


p = plotter.plot_error_values(train, "pos_holiday", "Weekly_Sales", drop_quantile=0.25, 
                           title="Weekly_Sales errors grouped by pos_holiday", width=350, height=200)
show(p)


# # <font color="navy">Fuel_Price</font>

# In[ ]:


train.groupby(["Store", "wm_date"]).apply(lambda g : g["Fuel_Price"].corr(g["Weekly_Sales"])).hist(bins=20)


# ## Notes
# It seems that Fuel_Price variation for the same date over years is not strong related to Weekly_Sales variation

# # <font color="navy">CPI</font>

# In[ ]:


train.groupby(["Store", "wm_date"]).apply(lambda g : g["CPI"].corr(g["Weekly_Sales"])).hist(bins=20)


# ## Notes
# It seems that Fuel_Price variation for the same date over years is not strong related to Weekly_Sales variation

# # <font color="navy">Unemployment</font>

# In[ ]:


train.groupby(["Store", "wm_date"]).apply(lambda g : g["Unemployment"].corr(g["Weekly_Sales"])).hist(bins=20)


# ## Notes
# It seems that Fuel_Price variation for the same date over years is not strong related to Weekly_Sales variation

# # <font color="navy">Temperature</font>

# In[ ]:


train.plot.scatter("celsius", "Weekly_Sales")


# In[ ]:


grouped_sales = train.groupby("Date")["celsius"].median()
p = plotter.plot_time_series_count(grouped_sales.index, grouped_sales, color="magenta", title="Temperature vs Datetime", legend="celsius", relative_y_axis=True, height=200)
p.legend.location = 'bottom_center'
show(p)


# In[ ]:


grouped_sales = train.groupby("Date")["Weekly_Sales"].median()
p = plotter.plot_time_series_count(grouped_sales.index, grouped_sales, color="navy", title="Weekly_Sales vs Datetime", legend="overall median", relative_y_axis=True, height=200)
show(p)


# In[ ]:


train["celsius"].corr(train["Weekly_Sales"])


# In[ ]:


train["celsius_diff"].corr(train["Weekly_Sales"])


# # <font color="navy">Size</font>

# In[ ]:


size_sales = train.groupby("Size")["Weekly_Sales"].median().reset_index()
print(size_sales["Size"].corr(size_sales["Weekly_Sales"]))
size_sales.plot.scatter("Size", "Weekly_Sales", title="Weekly_Sales median vs Size")


# ## Notes
# Interesting spread :) <br>
# Also a good correlation

# # <font color="navy">Markdown</font>

# In[ ]:


mds = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]


# In[ ]:


for md in mds:
    print(md+":", round(train[md].corr(train["Weekly_Sales"]), 4))


# ## Notes
# :(

# # Exploring Transformations

# Overlaying year Store-Dept Sales Plot

# In[ ]:


from statsmodels.tsa.filters.hp_filter import hpfilter


# In[ ]:


store_dept = train["store_dept"].sample().iloc[0]
store_dept


# In[ ]:


store_dept_sales = train[train["store_dept"] == store_dept].set_index("Date")
years = store_dept_sales["year"].drop_duplicates().to_list()[0:2]
years


# In[ ]:


dy1 = store_dept_sales[store_dept_sales["year"] == years[0]]
dy2 = store_dept_sales[store_dept_sales["year"] == years[1]].reset_index()
dy2["Date"] = dy2["Date"].str.slice(4,10).apply(lambda dt : str(years[0]) + str(dt))
dy2 = dy2.set_index("Date")


# In[ ]:


pretties.display_md("#### Store-Dept: {}".format(store_dept))


# In[ ]:


p = plotter.plot_time_series_count(dy1.index, dy1["Weekly_Sales"], color="navy", title="Weekly_Sales vs Datetime for store_dept {}".format(store_dept), 
                                relative_y_axis=True, height=300, legend=str(years[0]), p=None)
p = plotter.plot_time_series_count(dy2.index, dy2["Weekly_Sales"], color="magenta", title="Weekly_Sales vs Datetime for store_dept {}".format(store_dept), 
                                relative_y_axis=True, height=300, legend=str(years[1]), p=p)

show(p)


# ## Notes
# Despite these both time series have different years, <br>
# they were placed together in order to check if there <br>
# evidence of Week_n and Month_n explaining shopping behavior.

# In[ ]:


cycle1, trend1 = hpfilter(dy1["Weekly_Sales"], lamb=0.5)
cycle2, trend2 = hpfilter(dy2["Weekly_Sales"], lamb=0.5)

p = plotter.plot_time_series_count(dy1.index, trend1, color="cyan", title="Weekly_Sales vs Datetime", 
                                relative_y_axis=True, height=300, line_width=3, legend="hp " + str(years[0]), p=p)
p = plotter.plot_time_series_count(dy2.index, trend2, color="#FFC0C8", title="Weekly_Sales vs Datetime", 
                                relative_y_axis=True, height=300, line_width=3, legend="hp " + str(years[1]), p=p)

p.legend.location = 'top_center'
show(p)


# ## Notes
# Plotting two years data with Hodrick-Prescott filter each one <br>
# HP filter seem to stand between them.
