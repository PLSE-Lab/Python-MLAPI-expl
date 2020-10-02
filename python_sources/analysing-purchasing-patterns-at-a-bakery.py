#!/usr/bin/env python
# coding: utf-8

# # Analysing Purchasing Patterns at a Bakery
# 
# ### Introduction
# 
# In this notebook (my first one, how exciting!) we will look at sales data from a little bakery. It's a pretty straightforward dataset containing each individual item sold at the bakery and the time the sale was made. Further, each sold item is assigned a transaction number, i.e. if you buy two muffins (everyone loves muffins), the dataset will contain two separate entries, both with the same transaction number and date/time of sale.
# 
# And that's it. Doesn't sound too exciting? Well, I'll try my best to squeeze as much insight out of this little dataset as possible, so let's get going!
# 
# ### Importing the Dataset and (light) cleaning
# 
# First, we import the usual packages:

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy as sp


# Reading the data and taking a first quick peek at it:

# In[ ]:


data = pd.read_csv("../input/BreadBasket_DMS.csv")
data.info()
data.head(5)


# Looks exactly like what we were expecting. No good reason to have the dates and times in separate columns, however, so we'll merge those and convert them to datetime format:

# In[ ]:


data["DateTime"] = pd.to_datetime((data.Date + " " + data.Time))
data.drop(columns=["Date", "Time"], inplace=True)


# Next, let's try to find out when this bakery is open for business:

# In[ ]:


data["DateTime"].dt.hour.value_counts().sort_index()


# It appears like this store usually opens during the 8 o'clock hour and closes around 5 in the afternoon. Anything outside those hours seems to be due to special events or the store opening slightly before 8. In this analysis I'll focus on the main business hours and drop anything outside this range.
# 
# Also, I'll create two extra variables by extracting the day of the week and the hour from the datetime!

# In[ ]:


data = (data.loc[(data["DateTime"].dt.hour.between(8, 17, inclusive=True))]
        .reset_index(drop=True))

data["Weekday"] = data["DateTime"].dt.day_name()
data["Hour"] = data["DateTime"].dt.hour


# Finally, let's take a look at the actual items sold:

# In[ ]:


np.sort(data.Item.unique())


# Looks fine and yummy to me! Except those NONE entries... I wonder how many there are...

# In[ ]:


data.loc[data["Item"]=="NONE"].shape[0]


# That is quite a lot (almost 4% of all observations). I'm not really sure what NONE in this context means, but a fair assumption is that these were cancelled sales. In any case let's get rid of them:

# In[ ]:


data.drop(index=data[data["Item"] == "NONE"].index, inplace=True)
data.reset_index(drop=True, inplace=True)

data.head(5)


# ### Analysis of daily total sales
# 
# Now that our data is whipped into shape, let's get to analyzing. First we will study daily total sales volume. For that the data needs to be aggregated to a daily format. Hence, we will create a second dataframe with the following features:
# 
# - Number of items sold per day
# - Number of unique transactions per day
# - Number of items sold per transaction per day
# - Day of the week

# In[ ]:


totsales = data.groupby(data.DateTime.dt.date).Item.count().to_frame()
totsales.index = pd.to_datetime(totsales.index)

totsales.rename(columns={"Item":"ItemsSold"}, inplace=True)
totsales["UniqueTrans"] = data.groupby(data.DateTime.dt.date).Transaction.nunique()
totsales["ItemsperTrans"] = totsales.ItemsSold / totsales.UniqueTrans
totsales["Weekday"] = totsales.index.day_name()

totsales.head(5)


# Awesome! We now have three time series we can investigate. Obviously, the next step is to plot them to see if there are any interesting patterns. In particular, it would be instructive to know whether there are any time trends and/or seasonality present in the data.
# 
# The next block of code creates three separate line plots for the three time series:
# 
# 1. A function is defined to take care of setting some of the plot properties on each subplot
# 2. We take advantage of some of the seaborn style and color setting functions to make our plots easier on the eyes
# 3. The actual plots are drawn

# In[ ]:


def subplot_properties(titles):
    fig = plt.gcf() # get current figure
    for i, axis in enumerate(fig.axes): # iterate over the axes (subplots) stored in the figure
        axis.yaxis.grid(True)
        axis.set_xlabel("")
        axis.set_ylabel("")
        sns.despine(ax=axis, left=True, bottom=True)
        axis.set_title(titles[i], fontweight="bold")
        plt.sca(axis)
        plt.yticks(fontweight="light")
    sns.despine(ax=axis, left=True, bottom=False, offset=15)
    plt.xticks(fontweight="light")
    fig.subplots_adjust(hspace=0.25)

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
sns.set(font="sans-serif", palette="deep", style="white", context="notebook")

fig, ax = plt.subplots(3,1,figsize=(10,8), sharex=True)

ax[0].plot(totsales.index, totsales.ItemsSold)
ax[1].plot(totsales.index, totsales.UniqueTrans)
ax[2].plot(totsales.index, totsales.ItemsperTrans)

plottitles = ["Total Items Sold", "Total Transactions", "Items Sold per Transaction"]
subplot_properties(plottitles)

plt.show()


# What pops out immediately are the spikes in all three time series happening at regular intervals. This shouldn't be too surprising for a bakery (or most any retail business, really), and we would expect these spikes to ocurr during the weekend (when people generally have more time to shop and dine). We'll investigate this further in a bit!
# 
# (Note: There is also a pretty clear negative time trend during the month of November. Unfortunately, we won't be able to determine whether this is random or part of a longer-term (monthly or quarterly) seasonal trend since we lack the data.)
# 
# Second, total sales and total transactions are almost mirror images of each other, while the shape of the graph of items sold per transaction is quite different. Since we don't just want to believe our own lying eyes, let's take a quick glance at the **correlation coefficients**:

# In[ ]:


totsales.iloc[:,:3].corr()


# Our first intution proves correct: Sales and transactions are strongly correlated while sales per transaction are only moderately correlated (with total sales) or virtually uncorrelated (with transactions). What we could, very tentatively, conclude is that these sales spikes we are observing may be driven largely by a higher amount of customers coming to the store and only to a lesser degree by customers buying more items.
# 
# Finally, sales and transactions are much more variable (in relative terms) than daily items sold per transaction. Again, we'll try to confirm this by looking at the **coefficient of variation** (standard deviation/mean):

# In[ ]:


sp.stats.variation(a=totsales.iloc[:,:3])


# As we see the normalized standard deviations are almost identical for sales and transactions and significantly lower for sales per transaction.

# ### Analyzing Sales by Day of the Week
# 
# We saw some clear periodicity in the data, likely based on particular weekdays. In this section we will investigate further. (Also, this is the reason we included that weekday column in the aggregated data earlier!)
# 
# First, we'll take a look at total average sales per weekday:

# In[ ]:


totsales.groupby("Weekday").ItemsSold.mean().to_frame().sort_values(by="ItemsSold", ascending=False)


# That's a pretty large spread. On their strongest weekday this bakery sells about twice as many items as on their weakest day. Note further that their top three days are all on the weekend, with Saturday sporting their highest sales by far.
# 
# Since we are now dealing with data grouped by a category (the day of the week), we need a kind of plot that can properly visualize this data. Here we will opt for a **box plot**, which aside from showing us the three middle quantiles also shows us potential outliers in the data. Further, from the size of the box and the location of the median, we can get a rough idea of the shape of the distribution of the data.

# In[ ]:


weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

fig, ax = plt.subplots(3, 1, figsize=(10,8), sharex=True)

sns.boxplot(x="Weekday", y="ItemsSold", data=totsales, order=weekdays, ax=ax[0])
sns.boxplot(x="Weekday", y="UniqueTrans", data=totsales, order=weekdays, ax=ax[1])
sns.boxplot(x="Weekday", y="ItemsperTrans", data=totsales, order=weekdays, ax=ax[2])

plottitles2 = [i + " by Weekday" for i in plottitles]
subplot_properties(plottitles2)

plt.show()


# Weekends have the highest sales volume and number of transactions and also the highest variability in both measures. Items sold per transaction, on the other hand are only slightly higher on the weekend (interesting: Sundays perform best in this measure) and show rather little variability on a by-weekday basis confirming what we already learned from the time series plots. Overall, there are rather few outliers.
# 
# One thing of note is that Sundays see the 2nd highest sales on average, yet based on number of transactions (customers), Sunday performs quite low, almost like a typical workday week. The reason it still is the 2nd best selling weekday is more items per transaction are sold on Sundays for some reason. It would certainly be interesting to find out why this pattern arises, i.e. what does the typical transaction on a Sunday look like?
# 
# ### Conclusion
# 
# We now have gained a pretty decent understanding of this bakery's sales performance:
# * Sales on the weekend are significantly higher, particularly on Saturdays
# * These increases are probably mostly driven by larger customer volume and only to a lesser degree by larger sales volume per customer
# * Sundays pose a little bit of a mystery: Number of transactions are quite low, but items sold per transaction are quite high.
# 
# ### What's next?
# 
# * Analyzing sales performance on a per-item basis
# * Diving deeper into Sunday sales patterns
# * Answering the following question: If more coffee is bought on any given day, all else being equal, are tea sales going to go down?
