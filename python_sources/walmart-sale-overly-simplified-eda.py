#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import os


# In[ ]:


# set working directory to input folder
path = "/kaggle/input/m5-forecasting-accuracy/"
try:
    os.chdir(path)
    print("Directory changed to:", path)
except OSError:
    print("Can't change the Current Working Directory")
    
# import data
sell_prices = pd.read_csv("sell_prices.csv")
calendar = pd.read_csv('calendar.csv')
sales = pd.read_csv('sales_train_validation.csv')
sample_submission = pd.read_csv('sample_submission.csv')

print("Data loaded")


# ## Exploratory data analysis:
# 
# In order to capture my learning process in an organic manner, I would like to present the findings based on a series of questions and answers that arise from examining the data. Some questions may seem naive and some answers might seems contrived, but such is the nature of exploration. Therefore, this kernel is suitable for absolute newcomer, such as myself.
# 
# So just sit back and enjoy reading me fumbling through data. Let me know if you find something interests you.
# 
# ### Understand the Objective and Timelines
# 
# **What are we trying to predict?**
# 
# The competition wants 28 days ahead point forecasts. Here's what the submission looks like:

# In[ ]:


sample_submission.head()


# In the submission table, the `id` variable contains "validation" rows and "evaluation" rows. Validation corresponds to the Public leaderboard, while evaluation corresponds to the Private leaderboard (the prize pool). 

# In[ ]:


validation_prop = len(sample_submission[sample_submission['id'].str.contains('evaluation')]) / len(sample_submission)

print("Proportion of validation rows in sample_submission.csv:", validation_prop)


# ### Timelines

# In[ ]:


# gather a bunch of dates for timeline
sell_prices_cal = pd.merge(sell_prices, calendar, how = 'left', on = 'wm_yr_wk')


calendar_startdate = calendar.date.min()
calendar_enddate = calendar.date.max()
sales_train_validation_startdate = calendar[calendar.d == 'd_1'].date.item()
sales_train_validation_enddate = calendar[calendar.d == 'd_1913'].date.item()
submission_validation_startdate = calendar[calendar.d == 'd_1914'].date.item()
submission_validation_enddate = calendar[calendar.d == 'd_1941'].date.item()
submission_evaluation_startdate = calendar[calendar.d == 'd_1942'].date.item()
submission_evaluation_enddate = calendar[calendar.d == 'd_1969'].date.item()
sell_price_startdate = sell_prices_cal.date.min()
sell_price_enddate = sell_prices_cal.date.max()

del sell_prices_cal


# In[ ]:


import plotly.figure_factory as ff

df = [dict(Task="Sell Prices", Start = sell_price_startdate, Finish = sell_price_enddate),
      dict(Task="Calendar", Start = calendar_startdate, Finish = calendar_enddate),
      dict(Task="Sales train validation", Start = sales_train_validation_startdate, Finish = sales_train_validation_enddate),
      dict(Task="Submission validation", Start = submission_validation_startdate, Finish = submission_validation_enddate),
      dict(Task="Submission evaluation", Start = submission_evaluation_startdate, Finish = submission_evaluation_enddate)]

fig = ff.create_gantt(df)
fig.show()


# ### Sell_prices

# In[ ]:


sell_prices.head()


# **Observations**:
# 
# This table is fairly simple with only 4 variables
# 
# * `store_id` looks like it may contains state location, assuming CA stands for California
# 
# * `item_id` may contains category of the item
# 
# * `wm_yr_wk`'s meaning is unclear. So I looked it up using the [provided data descriptions](https://mofc.unic.ac.cy/wp-content/uploads/2020/03/M5-Competitors-Guide-Final-10-March-2020.docx). This variable denotes the week id. Not sure if the number is chronological, we will examine this later.
# 
# * `sell_price` This is the price is provided per week (average across seven days).

# ** Does store_id contain location info? **
# 
# We will use count to see the classes frequency:

# In[ ]:


sell_prices['store_id'].value_counts()


# **Intepretation**: It looks like the first 2 characters denote the state. We can create another variable called `state` to explore further:

# In[ ]:


# using vectorized str.split will be much faster than using apply here
sell_prices['state'] =  sell_prices['store_id'].str.split('_').str[0]


# ** Which states were included in the dataset? **

# In[ ]:


sell_prices['state'].unique().tolist()


# **Intepretation**: Only three states were included. Fortunately, these states were spreaded out in the US. Due to geographical differences, there may be differences in consumer behavior as well. We will need to keep location effect in mind.

# **Does item_id contain product categories info?**

# In[ ]:


sell_prices['product_cat'] =  sell_prices['item_id'].str.split('_').str[0]
sell_prices['product_cat'].unique().tolist()


# **Observation**: 3 product categories: Hobbies, Foods, and Household. Walmart is being very general with their data.

# **Do location affects the sale of different categories? (Consumer behavior: Location - Product)**
# 
# Since we are dealing with cat-cat variables, a mosaic plot would be appropriate here. A mosaic plot allows visualizing multivariate categorical data in a way that allow us to compare the proportion of different categories with one another:

# In[ ]:


from statsmodels.graphics.mosaicplot import mosaic

mosaic(sell_prices, ['state', 'product_cat'])
plt.show()


# **Interpretation**: Using the primitive power of my eye-balls, I can see that sale count of each product category does not differ across different locations. An average Walmart shopper in California would spend money on Hobbies, Foods, and Household roughly the same way a Texan does.
# 
# **Observation**: The area for CA is relatively larger, indicating higher number of items. Why is that?

# In[ ]:


sell_prices['state'].value_counts().plot(kind = 'bar')
plt.title('Number of rows by State')
plt.show()


# Something is not adding up here. There are way more CA rows than the other.

# In[ ]:


sell_prices['store_id'].value_counts()


# **Interpretation**: There it is! CA has 1 extra store "CA_4". One more compared to TX and WI.

# **How do sale prices differ across different states?**

# In[ ]:


sell_prices.groupby(['state'])['sell_price'].mean()


# Interpretation: TX Walmarts have the lowest sell price on average, while WI has the highest average sell price.

# **Which product category has the highest average sell price?**

# In[ ]:


sell_prices.groupby(['product_cat'])['sell_price'].mean()


# **Interpretation**: Food is cheaper than hobby and household item, makes sense.

# **Does item pricing fluctuate with time (weekly) - Grouped by product categories?**

# In[ ]:


plt.figure(figsize=(18,9)) # ah.. the sweet 18 by 9 ratio

sns.lineplot(x = 'wm_yr_wk', y = 'sell_price', hue = 'product_cat', data = sell_prices)
plt.title("Sell Price of Product Categories over time")
plt.show()


# **Observation**: Food price is on the rise yo, my guess: inflation. Meanwhile, household items are getting cheaper. I blame China for this (just kidding).
# 
# Notice the confident interval around the lines. This is because we have multiple data points for any given x. We can conclude that the pricing variation is greatest for hobby products.
# 
# There is a sudden jump in hobbies category. Perhaps a historic event? I might investigate this if week_id can be mapped to datetime. 

# **How about pricing trend grouped by states?**

# In[ ]:


plt.figure(figsize=(18,9))

sns.lineplot(x = 'wm_yr_wk', y = 'sell_price', hue = 'state', data = sell_prices)
plt.title("Sell Price of Product Categories over time")
plt.show()


# **Observation**: After some eye-balling, I would say Walmart pricing is fairly consistent across the states.

# ## Calendar

# In[ ]:


calendar.head()


# **Observations**:
# 
# * `wm_yr_wk` can be mapped with `sell_price` for price/event analysis
# 
# * `d`? What is d? 
# 
# * `event` variables contains lots of NaNs, which make sense since events are periodical. We will need to perform missing values analysis here.
# 
# * `snap`? SNAP stands for the [Supplemental Nutrition Assistance Program](https://www.feedingamerica.org/take-action/advocate/federal-hunger-relief-programs/snap). Which is food stamp program for the lower-income Americans. According to this website, "SNAP benefits cannot be used to buy any kind of alcohol or tobacco products or any nonfood items like household supplies and vitamins and medicines.". So this program only waives food purchases.
# 

# **So what is d?**

# In[ ]:


# duplicate check
calendar['d'].duplicated().sum()


# **Interpretation**: 0 duplicate found, suggesting this column may contains identification property, perhaps date?  

# In[ ]:


# convert date column (str) to date_time object
calendar['date'] = pd.to_datetime(calendar['date'])

# date range
display(max(calendar['date']) - min(calendar['date']))

# last d value
display(calendar['d'].tail(1))


# **Interpretation**: D is the number of day elapsed since 2011-01-28.

# **Handling missing data:**
# 

# In[ ]:


display(calendar['event_name_1'].value_counts())
display(calendar['event_name_2'].value_counts())


# **Observation**: Events are holidays, sports events, and religious significants. No special event hosted by Walmart.

# In[ ]:


plt.figure(figsize=(18,9))
sns.heatmap(calendar.isnull(), cbar = False)
plt.show()


# **Interpretation**: White represents the missing. `event_name` and `event_type` contain missing values in pair. We can fill NaNs with "Normal" date to avoid NaN values.

# In[ ]:


# before filling NaNs, better check if other columns contains any NaNs
calendar.isnull().sum(axis = 0)


# In[ ]:


calendar = calendar.fillna("Normal")
calendar.head()


# ## Relationship between Sell Price and Events
# 
# Let's visualize how events affect item pricing. We will need to join sell_prices with calendar, but...
# 
# * Pricing is set weekly, while events are daily. We will need to extract daily weeks that contain events before joining.
# 

# In[ ]:


events_weekly = calendar[calendar['event_type_1'] != 'Normal'][['wm_yr_wk', 'event_name_1', 'event_type_1']]
display(events_weekly)
print("Number of week duplicates:", events_weekly['wm_yr_wk'].duplicated().sum())


# **Observation**: Week duplicates, yikes! Some weeks may contain more than 1 event. This may become important later on (multiple events might affect sale number), but to make my life easier, I will remove the duplicates, keeping at 1 event/week. 

# In[ ]:


# "Some of You May Die, but that is a Sacrifice I'm Willing to Make"
events_weekly = events_weekly.drop_duplicates(subset = 'wm_yr_wk', keep = 'first')


# In[ ]:


# merge + fill NaNs
price_with_event = pd.merge(sell_prices, events_weekly, how = 'left', on = ['wm_yr_wk']).fillna('Normal')

# new column denotes event
price_with_event['event'] = np.where(price_with_event['event_type_1'] == 'Normal', 'Normal', 'Event')
price_with_event.head()


# Cool bean! Now let's draw:

# In[ ]:


plt.figure(figsize=(18,9))

sns.relplot(x = 'wm_yr_wk', y = 'sell_price', 
            hue = 'event_type_1', style = 'event_type_1', row = 'product_cat', 
            height = 4, # make the plot 4 units high
            aspect = 3, # height should be three times width
            kind = 'line',data = price_with_event, ci = None)  # remove confident interval for better clarity

plt.title("Sell Price of Product Categories over time")
plt.show()


# **Interpretation**: I can't really tell if price would decrease when events occured. We need to confirm this with another visualization:

# In[ ]:


sns.boxplot(x = 'event_type_1', y = 'sell_price', data = price_with_event)
plt.show()


# **Interpretation**: Boxplot also confirms that events don't directly influence the sell price. 
# 
# I guess lowering the price is a viable strategy to attract more customer to buy. But why would Walmart lower the price of ALL items on average during any specific event? I was foolish to even hope for any price action during events. 
# 
# But perhaps they would specifically lower a small number of products that Walmart knows would attract more sale during the event? 
# 
# We need to calculate the price difference between normal days and event days:

# In[ ]:


temp_df = price_with_event.groupby(['item_id', 'event'])['sell_price'].min().unstack()
temp_df.head()


# Notive that I used min() for group aggregation. This is because I want to target the items with most significant reduction in price during events. Now we can subtract min price of event by normal price to see how much item would decrease in price:

# In[ ]:


temp_df['event_delta'] = temp_df['Event'] - temp_df['Normal']
temp_df.head()


# **What is the proportion of items would go on discount during events?**

# In[ ]:


sum(temp_df['event_delta'] < 0) / len(temp_df)


# **Interpretation**: About 1 in every 5 products. However, from a glance, most price decrease is pretty small compared to the total cost of the item (cents to dollars). Therefore, we need to calculate the proportion of the discount relative to the price of the item:

# In[ ]:


temp_df['discount_prop'] = temp_df['event_delta'] / temp_df['Normal']
temp_df


# **Which items are most heavily discounted during events?**

# In[ ]:


temp_df.sort_values(by=['discount_prop']).head(10)


# **Interpretation**: Top 10 discounted items are food and household items. We can look at these item specifically to examine the discount effect on the sales.
# 
# We can slice the item_id string to get the product department. Let's check which department would offer most discounted items during events:
# 
# 

# In[ ]:


# reset the index to get the column item_id back
temp_df.reset_index(inplace = True)

# get item department from item_id
temp_df['department'] = temp_df['item_id'].str.split('_').str[:2].apply(lambda x: '_'.join(x))
temp_df.head()


# In[ ]:


temp_df[temp_df['discount_prop'] < 0].groupby(['department'])['discount_prop'].count().sort_values(ascending = False).plot(kind = 'bar')
plt.show()


# **Interpretation**: Food_3, Household_2 and Household_1 are departments that discount many items.

# ## Sale Train Validation

# In[ ]:


sales.head()


# **Observation**: Not much new information here expect the number of unit sold. `dept_id`, `cat_id`, and `state_id` has been conveniently separated for us.
# 
# * The last day is 1913, which is less than 1969 days worth of data.
# 
# * This table is in the wide format, which is not optimal for plotting. I will use the melt() function, check out [this guide on wide to long guide](http://www.datasciencemadesimple.com/reshape-wide-long-pandas-python-melt-function/). Let's reshape this data to long:
# 
# Warning: The reshaped data is pretty big (>3.5 Gb).

# In[ ]:


sales_long = pd.melt(sales, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'd', value_name = 'unit_sold')
sales_long.head()


# **Observation**: This is a format where we can create a graph using `day` as x-axis and `unit_sold` as y-axis. Let's do some aggregation:

# In[ ]:


sales_cat = sales_long.groupby(['d', 'cat_id'])['unit_sold'].sum().unstack()
sales_cat


# In[ ]:


# reset index
sales_cat.reset_index(inplace = True)

# melt the cat
sales_cat = pd.melt(sales_cat, id_vars =['d'], var_name = 'product_cat', value_name = 'unit_sold')

# merge with calendar to get datetime
sales_cat = pd.merge(sales_cat, calendar[['date', 'd']], how = 'left', on = 'd')

sales_cat


# ## Visualize the sale data:

# In[ ]:


plt.figure(figsize=(18,9))

sns.lineplot(x = 'date', y = 'unit_sold', hue = 'product_cat', data = sales_cat)
plt.title("Units Sold for each Product Category")
plt.show()


# **Observations:**
# 
# * Foods category accounted for most sales; hobbies accounted for least sales
# 
# * There are some troughs that are close to 0. Did they closed on those days?
# 
# * There might be seasonality and trend in the time-series plot. 
# 
# **Did Walmart close their door?**

# In[ ]:


sales_cat[sales_cat['unit_sold'] < 500]


# **Interpretation**: The sales fall off the cliff on December 25th. This suggests Wamart closed during Christmas day. After some googling, this website confirms indeed that ["Walmart will be closed on Christmas Day"](https://heavy.com/entertainment/2019/12/walmart-hours-open-closed-christmas-eve-day-2019/).
# 
# I wonder what are the units got sold during Christmas day? Employee's last minute purchases? Foods come to life and scan themselves out the door?
# 
# ## Time series analysis
# 
# ### Univariate time series
# 
# Before adding more variables into the predictive model, we would gain a lot of information from just a single time-dependent dependency: units sold over time.

# In[ ]:


sales_univariate = sales_long.groupby(['d'])['unit_sold'].sum().reset_index()

# merge with calendar to get datetime
sales_univariate = pd.merge(sales_univariate, calendar[['date', 'd', 'weekday', 'month', 'year']], how = 'left', on = 'd')

sales_univariate.head()


# **Which days of the week have the highest sales?**

# In[ ]:


sales_univariate.groupby(['weekday'])['unit_sold'].sum().plot(kind = 'bar')
plt.title("Number of Unit Sold by Weekday")
plt.show()


# **Answer**: Weekends are days with highest sales.

# **Which months have the highest sales?**

# In[ ]:


sales_univariate.groupby(['month'])['unit_sold'].sum().plot(kind = 'line')
plt.title("Number of Unit Sold by Month")
plt.show()


# **Answer**: The first quarter of the year has higher sales.

# **Is there a trend for sale throughout the years?**

# In[ ]:


sales_univariate.groupby(['year'])['unit_sold'].sum().plot(kind = 'line')
plt.title("Number of Unit Sold by Year")
plt.show()


# **Interpretation**: There is a increasing trend of sales from 2011 to 2015. The drop-off in 2016 is most likely due to the lack of data for this year.

# ### Time series decomposition:
# 
# There are two kind of decomposition model: additive and multiplicative. 
# 
# * Additive model is describe as:
# 
# y(t) = Level + Trend + Seasonality + Noise
# 
# * Multiplicative model is describe as:
# 
# y(t) = Level * Trend * Seasonality * Noise
# 
# In a multiplicative time series, the components multiply together to make the time series. If you have an increasing trend, the amplitude of seasonal activity increases. This is appropriate for our sale data since the increase in number of Walmart sales would also increase the seasonal sales.

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose

# set date as index
sales_univariate = sales_univariate.set_index('date')

# get the unit_sold series
unit_sold_series = sales_univariate['unit_sold'].sort_index()


# In[ ]:


plt.figure(figsize = (16,10))
# period is the number of observations in a series if you consider the natural time interval of measurement (weekly, monthly, yearly)
decomposition = seasonal_decompose(unit_sold_series, model='multiplicative',period= 365) 
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(unit_sold_series, label = 'Original')
plt.legend(loc = 'best')
plt.subplot(412)
plt.plot(trend, label = 'Trend')
plt.legend(loc = 'best')
plt.subplot(413)
plt.plot(seasonal, label = 'Seasonal')
plt.legend(loc = 'best')
plt.subplot(414)
plt.plot(residual, label = 'Residuals')
plt.legend(loc = 'best')
plt.tight_layout()


# **Interpretation**: The three components are shown separately in the bottom three panels of the figure. These components can be added together to reconstruct the data shown in the top panel (the original series).
# 
# Now that we have extracted the seasonal component from sales, we can examine how events might affect the seasonal behaviors of sale.
# 
# ## Relationship between Sales and Events

# In[ ]:


holiday_lists = calendar[(calendar['date'].dt.year == 2013) & (calendar['event_name_1'] != 'Normal')][['date', 'event_name_1']]
holiday_lists = pd.merge(holiday_lists, seasonal.reset_index())
holiday_lists['dayofyear'] = holiday_lists.date.dt.dayofyear

plt.figure(figsize = (16,10))
sns.lineplot(x = seasonal.index.dayofyear, y = seasonal)
p1 = sns.scatterplot(x = holiday_lists.date.dt.dayofyear, y = holiday_lists.seasonal, color = 'red')

# add holoday labels
for line in range(0,holiday_lists.shape[0]):
     p1.text(holiday_lists.dayofyear[line]+0.2, holiday_lists.seasonal[line], holiday_lists.event_name_1[line], horizontalalignment='left', size='medium', color='black')
plt.title('Average Impact of Seasonal and Events on Sales')


# **Explaination**: This plot is the condensation of how sesonality affect the units sold through the years (2011-2016). The blue line is the average of the total unit sold at any given day of the year. The blue shadow is the area that represent the 95% confident interval of sales since we have multiple y-values for any given x-value.
# 
# The y axis represent the multiplication of seasonality effect. For example, y = 1.2 suggests a 20% increase from the baseline sale (the "trend" in the decomposition plot).
# 
# **Observation**:
# We can see that certain events can strongly influence the sales. Some holidays such as Christmas, New Year,Thanksgiving, and Halloween associate with trough sales, while other holidays such as SuperBowl, Labor Day, Veterans Days, etc., associate with peak sales.
# 
# Let's see how each event influences the sale number:

# In[ ]:


# Prepare Data
x = holiday_lists.loc[:, ['seasonal']]
holiday_lists['z'] = x - 1 
holiday_lists['colors'] = ['red' if x < 0 else 'green' for x in holiday_lists['z']]
holiday_lists.sort_values('z', inplace=True)
holiday_lists.reset_index(inplace=True)

# Draw plot
plt.figure(figsize=(20,10), dpi= 80)
plt.hlines(y=holiday_lists.event_name_1, xmin=0, xmax=holiday_lists.z, color = holiday_lists.colors, alpha=0.4, linewidth=5)

for x, y, tex in zip(holiday_lists.z, holiday_lists.event_name_1, holiday_lists.z):
    t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if x < 0 else 'left', 
                 verticalalignment='center', fontdict={'color':'red' if x < 0 else 'green', 'size':14})

# Decorations
plt.gca().set(ylabel='Event', xlabel='Impact on Sale')
plt.title('Diverging Bars of Event Impact', fontdict={'size':20})
plt.grid(linestyle='--', alpha=0.5)


# **Explaination**: Plotting the seasonal components this way reveals the impact of event on sale. As we learn previously, the closure of Chrismas will result in reduction of ~100% in sale. However, Chismas event in this plot only suggest a 76% reduction in sale. This is because some of these losses in sale was accounted in the residuals of the decomposition. 
# 

# ## Relationship between Sell Price and Sales
# 
# As we learn previously that the item pricing is sensitive the type of item. To see pricing affect sale, I think it's best to look at a single item and observe the correlation between sell price and sales:

# In[ ]:


def item_visualizer(item_name):
    specific_item = sales_long[sales_long['item_id'] == item_name][['item_id', 'd', 'unit_sold']]

    # merge with calendar to get datetime
    specific_item = pd.merge(specific_item, calendar[['date', 'd', 'wm_yr_wk']], how = 'left', on = 'd')

    # merge with sell_prices to get pricing
    specific_item = pd.merge(specific_item, sell_prices[['item_id', 'sell_price', 'wm_yr_wk', ]], how = 'left', on = ['item_id', 'wm_yr_wk'])
    
#     # random sampling to reduce plot time
#     specific_item = specific_item.sample(frac = 0.1, random_state = 42)

    plt.figure(figsize = (14,7))
    ax1 = sns.lineplot(x = 'date', y = 'sell_price', data = specific_item, label = 'Sell Price', color = 'red', alpha = 0.8)
    ax1.legend(loc="upper right")

    ax2 = plt.twinx()
    sns.lineplot(x = 'date', y = 'unit_sold', data = specific_item, label = 'Units Sold', alpha = 0.5, ax=ax2)
    ax2.legend(loc="upper left")

    plt.title('Sell Price and Sales of item: ' + item_name)
    plt.show()


# In[ ]:


item_visualizer('FOODS_1_005')


# **Observations**: This is an example item that exhibit a strong association between units sold and discount actions.
# 
# For this particular item:
# 
# * Walmart drops the price when the sales is consistently low over a period of time. 
# 
# * As soon as the sales pick up, Walmart returns to normal or higher pricing.
# 
# * The confident intervals, especially during discount periods, indicate the differences in discount amount among different Walmart locations. This results in the high variation in discounted prices.
# 
# 

# ## Univariate Time Series Prediction
# 
# **Note**: This section my personal review on simple time series forcast with single variable (total units sold). Therefore, you can skip this section as it may not be relevant to the topic of this competition.

# In[ ]:


# split into train and test sets
unit_sold = unit_sold_series.reset_index()

n_train = round(365 * 4.5) # train on 4.5 year worth of data
train = unit_sold[:n_train]
test = unit_sold[n_train:]


# In[ ]:


plt.figure(figsize = (16,8))
sns.lineplot(x = train.date, y = train.unit_sold, label = 'Train')
sns.lineplot(x = test.date, y = test.unit_sold, label = 'Test')
plt.title('Total Units Sold from All Malmarts')
plt.show()


# ### Holt's Linear Trend Model

# In[ ]:


from statsmodels.tsa.api import ExponentialSmoothing, Holt

prediction_holt = test.copy()
linear_fit = Holt(np.asarray(train['unit_sold'])).fit()
prediction_holt['Holt_linear'] = linear_fit.forecast(len(test))

plt.figure(figsize = (16,8))
plt.plot(train.unit_sold, label = 'Train')
plt.plot(test.unit_sold, label = 'Test')
plt.plot(prediction_holt['Holt_linear'], label = 'Holt Linear Prediction')
plt.legend(loc = 'best')
plt.title('Holt Linear Trend Forcast')
plt.show()


# In[ ]:


# calculate RMSE to check the accuracy of the model:
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(test.unit_sold, prediction_holt.Holt_linear))

print(rms)


# ### Holt Winter's Model

# In[ ]:


prediction_holtwinter = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train['unit_sold']), seasonal_periods= 365, trend = 'add', seasonal= 'add', damped = True).fit()
prediction_holtwinter['Holt_Winter'] = fit1.forecast(len(test))

plt.figure(figsize = (16,8))
plt.plot(train['unit_sold'], label = 'Train')
plt.plot(test['unit_sold'], label = 'Test', alpha = 0.6)
plt.plot(prediction_holtwinter.Holt_Winter, label = 'Holt Winters Prediction', alpha = 0.6)
plt.legend(loc = 'best')
plt.show()


# In[ ]:


rms = sqrt(mean_squared_error(test.unit_sold, prediction_holtwinter.Holt_Winter))

print(rms)


# ### Seasonal ARIMA Model

# In[ ]:


import statsmodels.api as sm


train = unit_sold_series[:n_train]
test = unit_sold_series[n_train:]

y_hat_avg = test.copy()
fit1 = sm.tsa.statespace.SARIMAX(train, order=(2, 1, 4),seasonal_order=(0,1,1,12)).fit()

y_hat_avg['SARIMA'] = fit1.predict(start="2015-07-29", end="2016-04-24", dynamic=True)

plt.figure(figsize=(16,8))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.legend(loc='best')
plt.show()


# In[ ]:


rms = sqrt(mean_squared_error(test, y_hat_avg['SARIMA']))

print(rms)

