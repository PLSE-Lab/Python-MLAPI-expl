#!/usr/bin/env python
# coding: utf-8

# # Initial EDA & insights
# 
# A good starting point for an analysis is always reading thoroughly the "Overview" and "Data" sections. The organizers also provide comprehensive M5 Participants Guide [1] which describes the dataset in details. However, naturally this notebook will touch upon the most important aspects.

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Imports" data-toc-modified-id="Imports-1">Imports</a></span></li><li><span><a href="#Total-sales" data-toc-modified-id="Total-sales-2">Total sales</a></span></li><li><span><a href="#Seasonal-decomposition" data-toc-modified-id="Seasonal-decomposition-3">Seasonal decomposition</a></span></li><li><span><a href="#Categories" data-toc-modified-id="Categories-4">Categories</a></span></li><li><span><a href="#Sales-by-states" data-toc-modified-id="Sales-by-states-5">Sales by states</a></span></li><li><span><a href="#Sales-by-stores" data-toc-modified-id="Sales-by-stores-6">Sales by stores</a></span></li><li><span><a href="#SNAP-(food-stamps)" data-toc-modified-id="SNAP-(food-stamps)-7">SNAP (food stamps)</a></span></li><li><span><a href="#Products" data-toc-modified-id="Products-8">Products</a></span></li><li><span><a href="#Holidays" data-toc-modified-id="Holidays-9">Holidays</a></span></li><li><span><a href="#Missing-variables" data-toc-modified-id="Missing-variables-10">Missing variables</a></span></li><li><span><a href="#Encoding-categorical-variables" data-toc-modified-id="Encoding-categorical-variables-11">Encoding categorical variables</a></span></li><li><span><a href="#Feature-selection" data-toc-modified-id="Feature-selection-12">Feature selection</a></span></li><li><span><a href="#References" data-toc-modified-id="References-13">References</a></span></li></ul></div>

# ## Imports

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import plotly.graph_objects as go
from scipy import stats 

try:
    import calmap
except:
    get_ipython().system(' pip install calmap')
    import calmap

plt.style.use('ggplot')
mpl.rcParams['figure.dpi'] = 100
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# don't use scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)
DATA_DIR="../input/m5-forecasting-accuracy/"


# In[ ]:


calendar = pd.read_csv(f"{DATA_DIR}calendar.csv")
sales = pd.read_csv(f"{DATA_DIR}sales_train_validation.csv")
sub = pd.read_csv(f"{DATA_DIR}sample_submission.csv")
prices = pd.read_csv(f"{DATA_DIR}sell_prices.csv")


# In[ ]:


sales.head()


# In[ ]:


calendar.head()


# In[ ]:


sub.head()


# "The dataset involves the unit sales of 3,049 products, classified in 3 product categories (Hobbies, Foods, and Household) and 7 product departments, in which the above-mentioned categories are disaggregated.  The products are sold across ten stores, located in three States (CA, TX, and WI). " [1]

# In[ ]:


sales.shape


# In[ ]:


sales.dept_id.unique()


# In[ ]:


sales.store_id.unique()


# There are 2 hobbies deparments, 2 household departments and 3 food departments. **Question:** How products from different department but the same category differ from each other?

# In[ ]:


sales.head()


# ## Total sales

# In[ ]:


days = [col for col in sales.columns if "d_" in col]


# Let's transform the data to get total daily sales. 

# In[ ]:


total_per_day = pd.DataFrame()
total_per_day['sales'] = sales[days].sum()
total_per_day['date'] = calendar.date[:1913].values
total_per_day['date_short'] =  total_per_day['date'].str[5:]
total_per_day['date'] = pd.to_datetime(total_per_day['date'],format='%Y-%m-%d')
total_per_day['day'] = total_per_day.date.dt.day
total_per_day['month'] = total_per_day.date.dt.month_name()
total_per_day['weekday'] = total_per_day.date.dt.weekday_name
total_per_day['year'] = total_per_day.date.dt.year
# to have dates as x-axis labels in decomposition plots
total_per_day = total_per_day.set_index("date")
total_per_day.head()


# In[ ]:


calendar['date'] = pd.to_datetime(calendar['date'],format='%Y-%m-%d')


# In[ ]:


import numpy as np                                                              
import seaborn as sns                                                           
from scipy import stats                                                         
import matplotlib.pyplot as plt                                                 

ax = sns.distplot(total_per_day.sales)                                    

mu, std = stats.norm.fit(total_per_day.sales)

# Plot the histogram.
# plt.hist(data, bins=25, density=True, alpha=0.6, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
ax.plot(x, p, 'g')
plt.title("Histogram of total daily sales")
plt.show()


# Data seems to be roughly normally distributed, let's conduct a statistical test to confirm that.

# In[ ]:


k2, p = stats.normaltest(total_per_day.sales)
alpha = 1e-3
print("Normality test:")
if p < alpha:  # null hypothesis: x comes from a normal distribution
    print("Data is normally distributed")
else:
    print("Data is NOT normally distributed")


# The next plot shows how sales were changing over years.

# In[ ]:


fig = go.Figure(layout={"title":{
    'text': "Daily total unit sales","x":0.5}})
fig.add_trace(go.Scatter(x=total_per_day.index, y=total_per_day['sales'],
                         mode='lines',marker_color='green',hovertext=total_per_day.index))
    
fig.show()


# There are 5 days with sudden drops. Plotly creates interactive plots, so after hovering on these drops it becomes clear that all of them happen on Christmas Day. Apparently, it's the only day when Walmart stores are closed. Those are extreme outliers which should be removed in the modelling phase. Moreover, the second lowest value each year occurs around 28rd of November, Thanksigiving Day, when people stay home and celebrate with their families instead of shopping.

# ## Seasonal decomposition

# Time series decomposition is helpful in time series analysis [2][3]. Every time series can be decomposed into three components:
# * trend - long-term increase or decrease in the data
# * seasonality - repeating short-term pattern with fixed frequency (e.g. days, weeks, months)
# * residual - random noise

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose

# daily measurements, season repeats every year
decompfreq = 366
model = 'multiplicative'
mpl.rcParams['figure.figsize'] = (15, 10)
decomposition = seasonal_decompose(
    total_per_day['sales'],
    freq=decompfreq)
fig = decomposition.plot()


# Thanks to decomposition we can clearly see a moderately increasing trend - Walmart's sales are constantly rising. There is also strong seasonality - each month follows a similar pattern over years.

# In[ ]:


def values_on_bars():
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., p.get_height(), str(int(p.get_height())), 
        fontsize=12, ha='center', va='bottom')


# In[ ]:


mpl.rcParams['figure.figsize'] = (15, 5)
ax = sns.barplot(x='month',y='sales',data=total_per_day,ci=None)
values_on_bars()
plt.title("Average monthly sales",fontsize=15)
plt.show()


# On average, sales are the lowest in May (32503) and the highest in August (35946). Overall, the differences are negligible.

# In[ ]:


sns.barplot(x='weekday',y='sales',data=total_per_day)
plt.title("Average sales by weekday",fontsize=15)
plt.show()


# Not surprisingly, sales jump during weekends.

# ## Categories

# To have a clear comparison we calculate sales distribution between categories in percentages.

# In[ ]:


total_cats = sales.groupby("cat_id")[days].sum().T
total_cats['date'] = list(calendar.date[:1913])
total_cats['year'] = pd.to_datetime(total_cats['date'],format='%Y-%m-%d').dt.year
total_cats = total_cats.groupby("year").sum().apply(lambda x:x*100/x.sum(),axis=1)
total_cats


# In[ ]:


total_cats.plot(kind='bar', stacked=True, title="Category distribution over years (in %)")
plt.legend(title="Category",bbox_to_anchor=(1,1))
plt.show()


# Categories distribution is roughly similar over years. "Foods" consitutes a vast majority (~70%) of overall sales, followed by "Household".

# In[ ]:


cats_per_store = sales.groupby(["cat_id","store_id"])[days].sum().sum(axis=1).unstack(level=0).apply(lambda x:100*x/x.sum(),axis=1)
cats_per_store


# In[ ]:


cats_per_store.plot(kind='bar', stacked=True,title="Category distribution per stores (in %)")
plt.legend(title="Category",bbox_to_anchor=(1,1))
plt.show()


# ## Sales by states

# In[ ]:


days = [col for col in sales.columns if "d_" in col]
total_states = sales.groupby("state_id")[days].sum().T
total_states.index = pd.to_datetime(calendar.date[:1913])
total_states.head()


# In[ ]:


total_states.describe()


# In[ ]:


fig = go.Figure(layout={"title":{
    'text': "Daily unit sales per state","x":0.5}})
for state in total_states.columns:
    fig.add_trace(go.Scatter(x=total_states.index, y=total_states[state],
                             mode='lines',hovertext=total_states.index,name=state))
    
fig.show()


# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose
trends_states = total_states.copy()
# daily measurements, season repeats every year
decompfreq = 366
model = 'multiplicative'
mpl.rcParams['figure.figsize'] = (15, 5)
fig = go.Figure(layout={"title":{
    'text': "Trends in sales per state","x":0.5}})
for state in total_states.columns:
    decomposition = seasonal_decompose(
        total_states[state],
        freq=decompfreq)
    fig.add_trace(go.Scatter(x=total_states.index, y=decomposition.trend,
                             mode='lines',hovertext=total_states.index,name=state))
    
fig.show()


# Clearly, California obtains the best sales by a large margin. Texas was superior to Wisconsin up until mid-2014, but since than the latter has a steeper upwards trend. However, it is important to remember here that CA has 4 stores while TX and WI both have 3, so for better comparison we should average sales by number of stores.

# ## Sales by stores

# In[ ]:


total_stores = sales.groupby(["store_id"])[days].sum().T
total_stores.index = calendar.date[:1913]
total_stores.head()


# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose
trends_stores = total_stores.copy()
# daily measurements, season repeats every year
decompfreq = 366
model = 'multiplicative'
mpl.rcParams['figure.figsize'] = (15, 5)
fig = go.Figure(layout={"title":{
    'text': "Trends in sales per state","x":0.5}})
for state in total_stores.columns:
    decomposition = seasonal_decompose(
        total_stores[state],
        freq=decompfreq)
    fig.add_trace(go.Scatter(x=total_stores.index, y=decomposition.trend,
                             mode='lines',hovertext=total_stores.index,name=state))
    
fig.show()


# CA_3 obtains much larger sales then other stores in that state. CA_2 has a significant drop for most of 2014 and equally significant jump in 2015. TX_2 is calmly decreasing since mid-2013. WI_3 after initial peak in 2012, also dropped in later years.

# We can observe that if we discard CA_3, the plots look pretty similar for all states. Therefore for the prediction it's much more important if the sales are from CA_3 or other stores than from which state the sales are.

# In[ ]:


total_states = sales[sales.store_id!='CA_3'].groupby("state_id")[days].sum().T
total_states.index = calendar.date[:1913]
from statsmodels.tsa.seasonal import seasonal_decompose
trends_states = total_states.copy()
# daily measurements, season repeats every year
decompfreq = 366
model = 'multiplicative'
mpl.rcParams['figure.figsize'] = (15, 5)
fig = go.Figure(layout={"title":{
    'text': "Trends in sales per state","x":0.5}})
for state in total_states.columns:
    decomposition = seasonal_decompose(
        total_states[state],
        freq=decompfreq)
    fig.add_trace(go.Scatter(x=total_states.index, y=decomposition.trend,
                             mode='lines',hovertext=total_states.index,name=state))
    
fig.show()


# ## SNAP (food stamps)

# US government provides food stamps for low-income households that allow buying some **selected** products for a discounted price. So probably on stamp days sales of those products might increase while sales of products that can't be bought by stamps stays similar to usual.

# In[ ]:


for state in ["CA","WI","TX"]:
    print("State",state)
#     snaps_state = total_states[state]
    snaps_state = calendar.set_index("date")["snap_"+state]
    plt.figure(figsize=(15,5))
    calmap.yearplot(snaps_state,year=2015)
    plt.show()


# In every state there are different designated stamp days, but in all of them there are no stamp days in the second half of the month. <br>
# Stamp days per state:
# * CA - 1 - 10
# * WI - 2,3,5,6,8,9,11,12,14,15 - pattern: a cycle of one day without stamps and two days with stamps
# * TX - 1,3,5,6,7,9,11,12,13,15

# In[ ]:


total_states_with_calendar = total_states.T.reset_index().melt(id_vars='state_id',var_name='date',value_name='sales')
total_states_with_calendar = pd.merge(total_states_with_calendar,calendar,on='date')
total_states_with_calendar['date'] = pd.to_datetime(total_states_with_calendar['date'],format='%Y-%m-%d')
total_states_with_calendar['snap'] = total_states_with_calendar.apply(lambda x:int(x["snap_"+x['state_id']]==1),axis=1)
total_states_with_calendar['isHoliday'] = total_states_with_calendar['event_name_1'].notna().astype(int)
total_states_with_calendar = total_states_with_calendar.set_index("date")


# In[ ]:


from scipy.stats import pearsonr     
pearsonr(total_states_with_calendar['sales'],total_states_with_calendar['snap'])


# In[ ]:


total_states_with_calendar.groupby("snap")['sales'].describe()


# Days with stamps have higher mean and median - people buy more on days with stamps. There is a moderate linear relationship between sales and whether for the given day and state food stamps are allowed.

# In[ ]:


sns.distplot(total_states_with_calendar.loc[total_states_with_calendar['snap']==0,'sales'])
plt.title("Distribution of sales on days without food stamps")
plt.xlim([0,30000])
plt.show()


# In[ ]:


sns.distplot(total_states_with_calendar.loc[total_states_with_calendar['snap']==1,'sales'])
plt.title('Distribution of sales on days with food stamps')
plt.xlim([0,30000])
plt.show()


# ## Products

# In[ ]:


most_sold_products = sales.groupby("item_id")[days].sum().stack().sum(level=0).sort_values(ascending=False)


# In[ ]:


sales.groupby("item_id")[days].sum().loc[['FOODS_3_090','FOODS_3_586','FOODS_3_252'],:]


# In[ ]:


most_sold_products.head()


# In[ ]:


prices_desc = prices.groupby("item_id")['sell_price'].describe().fillna(0)
prices_desc['max_diff'] = prices_desc['max']-prices_desc['min']
prices_desc.sort_values(by="max_diff",ascending=False).head()


# In[ ]:


prices_desc[prices_desc['max_diff']==0].shape[0]/prices_desc.shape[0]


# 10% of products have always the same price

# ## Holidays

# There were a few holidays that I didn't know before:
# * Chanukah - is a Jewish festival commemorating the rededication of the Second Temple in Jerusalem at the time of the Maccabean Revolt against the Seleucid Empire. It is also known as the Festival of Lights.
# * EidAlAdha - also called the "Festival of the Sacrifice", is the second of two Islamic holidays celebrated worldwide each year, and considered the holier of the two. It honours the willingness of Ibrahim to sacrifice his son as an act of obedience to God's command.
# * Purim - a Jewish holiday which commemorates the saving of the Jewish people from Haman, an Achaemenid Persian Empire official who was planning to kill all the Jews, as recounted in the Book of Esther.
# * Cinco de Mayo - an annual celebration held on May 5. The date is observed to commemorate the Mexican Army's victory over the French Empire at the Battle of Puebla, on May 5, 1862.
# * Eid al-Fitr - also called the "Festival of Breaking the Fast", is a religious holiday celebrated by Muslims worldwide that marks the end of the month-long dawn-to-sunset fasting of Ramadan. This religious Eid is the first and only day in the month of Shawwal during which Muslims are not permitted to fast.

# In total there are 157 dates with a one event happening that day and 5 dates with two events happening on the same day.

# The following are the dates with 2 events on the same day:

# In[ ]:


double_holidays = calendar[calendar.event_name_2.notnull()]


# In[ ]:


double_holidays.head()


# In[ ]:


total_per_day[total_per_day.index.isin(double_holidays.date)]


# The average daily sales on non-holidays are 34489 with standard deviation equal to 7133. Double holidays are clearly above that average.

# In[ ]:


total_per_day.head()


# In[ ]:


total_per_day_with_calendar = pd.DataFrame()
total_per_day_with_calendar['sales'] = sales[days].sum()
total_per_day_with_calendar['date'] = calendar.date[:1913].values
total_per_day_with_calendar = pd.merge(total_per_day_with_calendar,calendar,on='date')


# In[ ]:


holiday_sales_avg = total_per_day_with_calendar[total_per_day_with_calendar.event_name_1.notna()].groupby(['event_name_1'])['sales'].mean().sort_values(ascending=False)


# In[ ]:


calendar.event_name_1.unique()


# In[ ]:


holiday_sales_avg


# Add a feature indicating if it's a food stamps in a given state - for California it doesn't matter if food stamps are eligible on a given day in another states.

# ## Missing variables

# In the merged dataset, only columns related to events are missing and `sell_price`. In our case missing values are not just random errors in data collection. Conversely, they have particular meaning. Missing event means that there was no holiday that day and missing sell prices means that the given product was not sold on a given day. However, no products have gaps in sales in the test set, therefore in the modelling phase we need to decide what to do with that matter. For instance, we can remove all training examples when a product wasn't sold or try to interpolate the values based on its sales before and after the period when it wasn't sold. Both approaches should lead to better results compared to just leaving the values as they are.

# Products are out of stock for 20% of the time.

# ## Encoding categorical variables
# Most of the variables in the competition are categorical so encoding them efficiently should play quite an important role.

# Cardinalities (number of unique categories for a given variable) of all variables are pretty small, apart from item_id and date. 

# ## Feature selection

# Which features should we drop:
# * `date` - stores the same information as `d`
# * `wday` - stores the same information as `weekday`
# * after adding `is_snap`, remove `snap_CA`, `snap_TX` and	`snap_WI`

# I doubt if keeping `event_name_2` and `event_type_2`  makes sense because only 0.002 % of the dates have two events in one day.

# ## References

# 1. M5 Participants Guide - https://mofc.unic.ac.cy/m5-competition/
# 2. https://otexts.com/fpp3/tspatterns.html
