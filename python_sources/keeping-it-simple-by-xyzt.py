#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this notebook, I go through how I worked to find a decent solution for this challenge using simple uncomplicated techniques. No machine learning, no fancy black-box models. Throw away your ARIMAs and Gradient Boosts. Think simple.

# # Setup and Loading Data

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 12)

PATH = "../input"
train = pd.read_csv(f"{PATH}/train.csv", low_memory=False, 
                    parse_dates=['date'], index_col=['date'])
test = pd.read_csv(f"{PATH}/test.csv", low_memory=False, 
                   parse_dates=['date'], index_col=['date'])
sample_sub = pd.read_csv(f"{PATH}/sample_submission.csv")

# Make the sample submission (Score: 48.75440)
# sample_sub.to_csv("submission.csv", index=False)


# # A "Dumb" Prediction
# 
# Find the average of sales of an item at a store on the day and month of sales and use that as the prediction. This effectively gives us a sample size of 5 (since the training set is five years long) to find the mean. This is clearly a sub-optimal solution because almost no thought goes into it. But it is helpful to code such solutions to get acquianted with the "Getting Data -> Submitting Prediction" pipeline from start to finish, and generally getting a feel for the data. It also provides a helpful benchmark for future solutions. 
# 
# *Any method that scores worse than this prediction is probably doing something incredibly wrong.*
# 
# **Note: The following code block takes > 1 hour to run. It is extremely inefficient.**

# In[ ]:


def dumb_prediction(train, test, submission):
    for _, row in test.iterrows():
        item, store = row['item'], row['store']
        day, month = row.name.day, row.name.month
        itemandstore = (train.item == item) & (train.store == store)
        dayandmonth = (train.index.month == month) & (train.index.day == day)
        train_rows = train.loc[itemandstore & dayandmonth]
        pred_sales = int(round(train_rows.mean()['sales']))
        submission.at[row['id'], 'sales'] = pred_sales
    return submission

# dumb_pred = dumb_prediction(train, test, sample_sub.copy())
# dumb_pred.to_csv("dumb_submission.csv", index=False)


# ### This solution gets a score of 22.13108.
# Nothing impressive, but not completely terrible either. Pretty much the kind of error you can expect for such a silly model.

# # Slightly Better Prediction
# 
# The previous method simply took the historical average of an item (on the same date and at the same store) and used it to predict the sales on the test set. We can improve this by understanding the data better. Is the a difference between sales on different days? That is, Mondays vs. Fridays, Weekends vs Weekdays? Are there special days without sales? Is there a difference between these stores? Is there a difference between the items?
# 
# To understand these trend, we need to dive into the data!

# ## Exploring the data

# In[ ]:


# Expand dataframe with more useful columns
def expand_df(df):
    data = df.copy()
    data['day'] = data.index.day
    data['month'] = data.index.month
    data['year'] = data.index.year
    data['dayofweek'] = data.index.dayofweek
    return data

data = expand_df(train)
display(data)

grand_avg = data.sales.mean()
print(f"The grand average of sales in this dataset is {grand_avg:.4f}")


# ### Changes by year
# 
# All items and stores seem to enjoy a similar growth in sales over the years.

# In[ ]:


agg_year_item = pd.pivot_table(data, index='year', columns='item',
                               values='sales', aggfunc=np.mean).values
agg_year_store = pd.pivot_table(data, index='year', columns='store',
                                values='sales', aggfunc=np.mean).values

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(agg_year_item / agg_year_item.mean(0)[np.newaxis])
plt.title("Items")
plt.xlabel("Year")
plt.ylabel("Relative Sales")
plt.subplot(122)
plt.plot(agg_year_store / agg_year_store.mean(0)[np.newaxis])
plt.title("Stores")
plt.xlabel("Year")
plt.ylabel("Relative Sales")
plt.show()


# ### Changes by month
# 
# All items and stores seem to share a common pattern in sales over the months as well.
# 

# In[ ]:


agg_month_item = pd.pivot_table(data, index='month', columns='item',
                                values='sales', aggfunc=np.mean).values
agg_month_store = pd.pivot_table(data, index='month', columns='store',
                                 values='sales', aggfunc=np.mean).values

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(agg_month_item / agg_month_item.mean(0)[np.newaxis])
plt.title("Items")
plt.xlabel("Month")
plt.ylabel("Relative Sales")
plt.subplot(122)
plt.plot(agg_month_store / agg_month_store.mean(0)[np.newaxis])
plt.title("Stores")
plt.xlabel("Month")
plt.ylabel("Relative Sales")
plt.show()


# ### Changes by day of the week
# 
# All items and stores also seem to share a common pattern in sales over the days of the week as well.

# In[ ]:


agg_dow_item = pd.pivot_table(data, index='dayofweek', columns='item',
                              values='sales', aggfunc=np.mean).values
agg_dow_store = pd.pivot_table(data, index='dayofweek', columns='store',
                               values='sales', aggfunc=np.mean).values

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(agg_dow_item / agg_dow_item.mean(0)[np.newaxis])
plt.title("Items")
plt.xlabel("Day of Week")
plt.ylabel("Relative Sales")
plt.subplot(122)
plt.plot(agg_dow_store / agg_dow_store.mean(0)[np.newaxis])
plt.title("Stores")
plt.xlabel("Day of Week")
plt.ylabel("Relative Sales")
plt.show()


# ### Are these patterns degenerate?
# 
# This is an important question. Not checking for degeneracies in the data can lead to missing important trends in complex datasets. For example, when looking at the monthly patterns, we average over all days of the month, years and either items or stores. But what if sales have a multi-dimensional dependence on two of these parameters that isn't easily separable? So, always check for degeneracies in the data!

# In[ ]:


agg_dow_month = pd.pivot_table(data, index='dayofweek', columns='month',
                               values='sales', aggfunc=np.mean).values
agg_month_year = pd.pivot_table(data, index='month', columns='year',
                                values='sales', aggfunc=np.mean).values
agg_dow_year = pd.pivot_table(data, index='dayofweek', columns='year',
                              values='sales', aggfunc=np.mean).values

plt.figure(figsize=(18, 5))
plt.subplot(131)
plt.plot(agg_dow_month / agg_dow_month.mean(0)[np.newaxis])
plt.title("Months")
plt.xlabel("Day of Week")
plt.ylabel("Relative Sales")
plt.subplot(132)
plt.plot(agg_month_year / agg_month_year.mean(0)[np.newaxis])
plt.title("Years")
plt.xlabel("Months")
plt.ylabel("Relative Sales")
plt.subplot(133)
plt.plot(agg_dow_year / agg_dow_year.mean(0)[np.newaxis])
plt.title("Years")
plt.xlabel("Day of Week")
plt.ylabel("Relative Sales")
plt.show()


# In this case, however, there don't seem to be any sneaky degeneracies. We can effectively treat the "month", "year", "day of the week", "item" and "store" as completely independent modifiers to sales prediction. This leads to a *very very simple* prediction model.
# 
# "Relative sales" in the plots above are the sales relative to the average. Since there are very regular patterns in the "month", "day of week", and "year" trends. All we have to do is simply memorize these trends and apply them to our predictions by multiplying them to the expected average sales. We get the expected average sales for an item at a store from the historical numbers in the training set.

# ### What about the item-store relationship?

# In[ ]:


agg_store_item = pd.pivot_table(data, index='store', columns='item',
                                values='sales', aggfunc=np.mean).values

plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.plot(agg_store_item / agg_store_item.mean(0)[np.newaxis])
plt.title("Items")
plt.xlabel("Store")
plt.ylabel("Relative Sales")
plt.subplot(122)
plt.plot(agg_store_item.T / agg_store_item.T.mean(0)[np.newaxis])
plt.title("Stores")
plt.xlabel("Item")
plt.ylabel("Relative Sales")
plt.show()


# Same here. Just a constant pattern and no degeneracies. So, you just need a model for how items sell at different stores, which is easily captured by an average sales look-up table or yet another "relative sales" pattern model.
# 
# > *Aside: Based on the extremely regularity of the data, how neat it is, and how few degeneracies there are - I am fairly confident this is probably simulated data.*

# ## Writing the "slightly better predictor"
# 
# We just need an item-store average sale look-up table, and then the "day of week", "monthly", "yearly" models.

# In[ ]:


# Item-Store Look Up Table
store_item_table = pd.pivot_table(data, index='store', columns='item',
                                  values='sales', aggfunc=np.mean)
display(store_item_table)

# Monthly pattern
month_table = pd.pivot_table(data, index='month', values='sales', aggfunc=np.mean)
month_table.sales /= grand_avg

# Day of week pattern
dow_table = pd.pivot_table(data, index='dayofweek', values='sales', aggfunc=np.mean)
dow_table.sales /= grand_avg

# Yearly growth pattern
year_table = pd.pivot_table(data, index='year', values='sales', aggfunc=np.mean)
year_table /= grand_avg

years = np.arange(2013, 2019)
annual_sales_avg = year_table.values.squeeze()

p1 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 1))
p2 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 2))

plt.figure(figsize=(8,6))
plt.plot(years[:-1], annual_sales_avg, 'ko')
plt.plot(years, p1(years), 'C0-')
plt.plot(years, p2(years), 'C1-')
plt.xlim(2012.5, 2018.5)
plt.title("Relative Sales by Year")
plt.ylabel("Relative Sales")
plt.xlabel("Year")
plt.show()

print(f"2018 Relative Sales by Degree-1 (Linear) Fit = {p1(2018):.4f}")
print(f"2018 Relative Sales by Degree-2 (Quadratic) Fit = {p2(2018):.4f}")

# We pick the quadratic fit
annual_growth = p2


# We can do a simple linear regression on the yearly growth datapoints. But if you look carefully, you can tell that the growth is slowing down. The quadratic fit works better since it better captures the curvature in the growth curve. Since we only have 5 points, this is the highest degree polynomial fit you should do to avoid overfitting.

# Now, we write the predictor. It's quite simple! When we are asked to predict the sales of Item X at Store Y on, say, a Monday in February - all we have to do is to look up the historical average of the sales of Item X at Store Y and then multiply it by a factor corresponding to Monday and then a factor corresponding to February to account for the seasonal and weekly changes in item sales at the stores. Finally, we multiply by the annual growth factor for the year we are predicting for. And thus, we have a very simple forecast of the item's sales.
# 
# This predictor will run quite fast and should parse through the whole test dataset in less than 20 seconds. A significant improvement over the "dumb" prediction method both in accuracy and compute efficiency.

# In[ ]:


def slightly_better(test, submission):
    submission[['sales']] = submission[['sales']].astype(np.float64)
    for _, row in test.iterrows():
        dow, month, year = row.name.dayofweek, row.name.month, row.name.year
        item, store = row['item'], row['store']
        base_sales = store_item_table.at[store, item]
        mul = month_table.at[month, 'sales'] * dow_table.at[dow, 'sales']
        pred_sales = base_sales * mul * annual_growth(year)
        submission.at[row['id'], 'sales'] = pred_sales
    return submission

slightly_better_pred = slightly_better(test, sample_sub.copy())
slightly_better_pred.to_csv("sbp_float.csv", index=False)

# Round to nearest integer (if you want an integer submission)
sbp_round = slightly_better_pred.copy()
sbp_round['sales'] = np.round(sbp_round['sales']).astype(int)
sbp_round.to_csv("sbp_round.csv", index=False)


# ### This solution gets a public score of 13.88569, and 13.87573 when rounding to the nearest integer!
# **(A nice improvement especially given the simplicity of the solution)**
# 
# *Note: Rounding to the nearest integer likely gives a marginally better score because the ground truth values are integers and rounding on average gets you closer to the actual values if your model is good.*

# # How can we do better?
# 
# Now that we have a very simple and effective model, there are many different direction we can go in improving the model. Here are a few ideas:
# 
# * Try seeing how well the model does on the training set itself and what the SMAPE metric looks like. Does the noise properties make sense? Is there a trend in the SMAPE? Finding regions of high SMAPE in the training set can be a rough indicator of where accuracy is taking a hit on the test set!
# 
# * Is the sales data normally distributed around the trends we found? If not, that can distort our predictions. Correctly for the noise distribution can help lower the SMAPE (and ultimately, make a better predictor).
# 
# * Are there other trends we missed? ***Try not to depend on black-box algorithms!*** Use your domain knowledge of stores and think about what could affect item sales.
# 
# # Conclusion
# 
# While it is enticing to throw a complicated magical algorithm at any and all datasets blindly, it is usually easier and more meaningful to simply think about the data and come up with simpler models. This kernel was written to show how easy-to-understand methods such as finding averages and simple regressions used under the guidance of domain knowledge (i.e., thinking about how stores work) do equally as well, if not much better than overly-complicated algorithms.

# # Additional: Tweaking the predictor
# 
# One of the small tweaks we can make to the model is to weigh data by recency. So, we weigh older data less and much recent data more! One easy way to do this is to use an exponential decay function for your weight. We want the weights to get exponentially smaller the further back in the past we go.
# 
# *Since this is simulated data, if the simulation had some hidden variables that changed with time, perhaps this is a simple way to encode that into the model without knowing what it is.*
# 
# Here I use the following equation for the weights: $$\exp\left(\frac{year - 2018}{5}\right)$$
# 
# The factor of 5 is arbitrarily picked for simulated data. In real data, it might make sense since you would expect store sales to lose predictive power after a decade or so. 

# In[ ]:


years = np.arange(2013, 2019)
annual_sales_avg = year_table.values.squeeze()

weights = np.exp((years - 2018)/5)

annual_growth = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 2, w=weights[:-1]))
print(f"2018 Relative Sales by Weighted Fit = {annual_growth(2018)}")


# In[ ]:


def weighted_predictor(test, submission):
    submission[['sales']] = submission[['sales']].astype(np.float64)
    for _, row in test.iterrows():
        dow, month, year = row.name.dayofweek, row.name.month, row.name.year
        item, store = row['item'], row['store']
        base_sales = store_item_table.at[store, item]
        mul = month_table.at[month, 'sales'] * dow_table.at[dow, 'sales']
        pred_sales = base_sales * mul * annual_growth(year)
        submission.at[row['id'], 'sales'] = pred_sales
    return submission

weighted_pred = weighted_predictor(test, sample_sub.copy())

# Round to nearest integer
wp_round = weighted_pred.copy()
wp_round['sales'] = np.round(wp_round['sales']).astype(int)
wp_round.to_csv("weight_predictor.csv", index=False)


# ### This solution gets a public score of 13.85181!
# 
# **Which is quite nice, indeed!**

# # Additional II: Exploiting a noisy degeneracy
# 
# I noticed (quite late, I admit) that there is a small degeneracy I missed above. If you look at the plots above in the notebook, you notice that the store in which an item being sold has a *very stable* relative sales factor. However, the "day of the week" on which an item is being sold has a larger spread (or is more noisy to the eye).
# 
# This reveals a small error I made earlier: Making a store-item look up table. This should have been a "Day of week" - Item look up table. This would encode any built-in degeneracies over those dimensions and greatly improve the model.
# 
# Another change I decide to make is ignore all the data before 2015 (except for extrapolating the yearly trend, because we need more data points for that). The idea here is if the degeneracy evolves over time, you don't want the older data to bias you against it.

# In[ ]:


# Only data 2015 and after is used
cut_off_year = 2015
new_data = data.loc[data.year >= cut_off_year]
grand_avg = new_data.sales.mean()

# Day of week - Item Look up table
dow_item_table = pd.pivot_table(new_data, index='dayofweek', columns='item', values='sales', aggfunc=np.mean)
display(dow_item_table)

# Month pattern
month_table = pd.pivot_table(new_data, index='month', values='sales', aggfunc=np.mean)
month_table.sales /= grand_avg

# Store pattern
store_table = pd.pivot_table(new_data, index='store', values='sales', aggfunc=np.mean)
store_table.sales /= grand_avg


# For the annual trend, I use a longer exponential drop-off because it works better than before, honestly.

# In[ ]:


year_table = pd.pivot_table(data, index='year', values='sales', aggfunc=np.mean)
year_table /= grand_avg

years = np.arange(2013, 2019)
annual_sales_avg = year_table.values.squeeze()

weights = np.exp((years - 2018)/10)
annual_growth = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 2, w=weights[:-1]))
print(f"2018 Relative Sales by Weighted Fit = {annual_growth(2018)}")


# In[ ]:


def awesome_predictor(test, submission):
    submission[['sales']] = submission[['sales']].astype(np.float64)
    for _, row in test.iterrows():
        dow, month, year = row.name.dayofweek, row.name.month, row.name.year
        item, store = row['item'], row['store']
        base_sales = dow_item_table.at[dow, item]
        mul = month_table.at[month, 'sales'] * store_table.at[store, 'sales']
        pred_sales = base_sales * mul * annual_growth(year)
        submission.at[row['id'], 'sales'] = pred_sales
    return submission

pred = awesome_predictor(test, sample_sub.copy())
rounded = pred.copy()
rounded['sales'] = np.round(rounded['sales']).astype(int)
rounded.to_csv(f"awesome_prediction.csv", index=False)


# ### This solution gets a public score of 13.83850!
# 
# **Awesome!**
