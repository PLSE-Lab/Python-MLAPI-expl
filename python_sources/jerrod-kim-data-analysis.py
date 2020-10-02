#!/usr/bin/env python
# coding: utf-8

# **EDA / Pandas Tutorial**
# # Yield curve analysis (1977 - 2019)

# Let's import data from the csv file and save it as 'x'

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
x =  pd.read_csv('/kaggle/input/data-yield-curve/Data Yield Curve.csv')


# X is now our dataframe. Let's take a look at the first five rows of data, as well as columns. 
# Head() is a good method to specify the number of data we want to show from top to bottom.

# In[ ]:


x.head(5)


# Seems like the "Date" column could serve as a good index.
# 
# Let's set the index to x["Date"] and also turn it into pandas datetime format.

# In[ ]:


x.set_index('Date')
x.index = pd.to_datetime(x['Date'])
x.head(5)


# We have monthly data at the moment. Let's turn it into yearly values using groupby and Grouper.
# 
# Groupby helps us group data by columns, and Grouper helps Groupby group data by designated elements. Here, we are gonna specify that the frequency is "A". "A" stands for annual. This is possible because we have our indices in the pandas datetime format.
# 
# **And do not forget to get the average of the monthly values through "mean()".**

# In[ ]:


x = x.groupby(pd.Grouper(freq='A')).mean()
x.head(5)


# Perfect!

# Now, a yield curve is constituted of the differences between long-term treasury bonds and short-term treasury bonds. The interest rates of long-term treasury bonds usually exceed those of short-term treasury bonds in a normal economy as long-term bonds offer more coupon payments. However, when investors are fearful of an upcoming recession, short-term yields are higher than long-term yields.
# 
# For us, the "SPREAD" column shows the difference between the two bonds, and it could serve as a signal for potential recessions coming up.
# 
# **Now, let's make a new column that labels each layer as either good or bad based on the value of the spread.
# And Count the number for each to see how many years boded ill**
# 
# We are gonna use the "loc" function. It's a function that helps us locate rows, but this time, we are gonna use it to create a new column called "Status". To do so, we put a conditional in the loc[] function and also declared the column we'd like to be made. Lastly, we'll assign values (Good / Bad). 

# In[ ]:


x.loc[x['SPREAD'] <= 0, 'Status'] = 'Bad' 
x.loc[x['SPREAD'] > 0, 'Status'] = 'Good'
x.head(5)


# Now, let's see in how many years investors were worrying about a recession.
# We are gonna count all the entries for the column, "Status", and then locate the ones with "Bad" in it.

# In[ ]:


x.Status.value_counts().loc['Bad']


# Now that we know there were 6 of them. Why don't we list the 6 "BAD", or recession-fearing, years?
# Here, we are using the sort_values method to sort the values of the SPREAD column. And we are setting the ascending element of the function to be true because we are looking for the lowest values in the column.

# In[ ]:


y = x.SPREAD.sort_values(ascending=True)
y.head(6)


# This is nice to see the top six lowest yield differences. But if you want to pinpoint the year when investors most feared a recession. You can simply you the idmxmin() function. It locates the row with the minimum value and presents us with the index, giving us the date.

# In[ ]:


x.SPREAD.idxmin()


# Surprisingly, the most severe investor reaction to a potential recession took place in 1979. It actually makes sense considering that the worst recession in the past few decades, which happened in 2008, was not expected at all by the bankers who were busy selling sub-prime mortgages.
# 
# Let's end our discussion today with a line plot.
# 
# The plot function's default form is a line plot, so don't worry about specifying it.
# However, I do want to make the minimum y value at 0. Let's do so using ylim and setting it to 0.

# In[ ]:


x.SPREAD.plot(ylim = 0)


# Economists say that the parts of the yield curve that hover around or below the value of 0 serve as a signal for an upcoming recession. Will we be able to accurately predict the next one with such a method? We shall find out next time.
