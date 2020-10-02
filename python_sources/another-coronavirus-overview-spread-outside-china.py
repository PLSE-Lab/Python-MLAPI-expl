#!/usr/bin/env python
# coding: utf-8

# # COVID-19: overview of spread outside China (March 10 data)
# 
# Let's tabulate the total number of confirmed cases per country vs date.

# In[ ]:


import pandas as pd
import numpy as np

DIR = "../input/novel-corona-virus-2019-dataset/"
lastDate = "3/10/20"
droppedDates = ["3/11/20"] # Partial data, ignore it

class Data:
    pass

data = {"confirmed": Data(), "deaths": Data(), "recovered": Data()}
for k in data:
    # Load data
    data[k].filename = "{0}time_series_covid_19_{1}.csv".format(DIR, k)
    data[k].df = pd.read_csv(data[k].filename)
    
    # Daily cases
    data[k].dates = data[k].df.groupby(["Country/Region"]).sum()
    del data[k].dates["Lat"]
    del data[k].dates["Long"]
    data[k].dates = data[k].dates.drop(droppedDates, axis=1)
    data[k].dates = data[k].dates.drop("Mainland China", axis=0)    
    data[k].dates = data[k].dates.sort_values(lastDate, axis=0, ascending=False)
    data[k].dates = data[k].dates[data[k].dates[lastDate] > 0]
    
    data[k].datesTotal = data[k].dates.sum()

pd.options.display.max_rows = 10000
pd.options.display.max_columns = 10000
data["confirmed"].dates.iloc[:14, -12:]


# ## Plotting confirmed cases (outside China)
# 
# This is your typical coronavirus infection plot, where most countries (without extensive prevention methods in place) exhibit near-exponential growth, where each day the number of infections multiplies by a roughly constant factor.

# In[ ]:


data["confirmed"].dates.iloc[:20, :].transpose().plot(figsize=(15, 8))


# Lets take a closer look at higher profile Countries. A logarithmic plot makes it easier to verify exponentials, as these become linear. 

# In[ ]:


data["confirmed"].dates.iloc[:20, :].transpose().plot(logy=True, figsize=(20, 8))


# Most of the countries have an exponential growth in March. However, there is something strange going on around February 23 +/- ~8 days. Suddenly, a lot of countries find a large number of cases, usually within a single or few days. Such a bump happened in China on 11/12-th of February too: https://www.worldometers.info/coronavirus/country/china/.
# 
# The uptrend in China is the result of changes in diagnosing the virus (https://www.worldometers.info/coronavirus/how-to-interpret-feb-12-case-surge/, original source: https://twitter.com/WHO/status/1227980048952463360?s=20). I guess the same goes to other countries, but they started 1-2 weeks later.

# ## How fast does the virus spread?
# 
# Let's compute the growth rate per day.

# In[ ]:


sampleLength = 3 # days

current = data["confirmed"].dates - data["deaths"].dates - data["recovered"].dates
current = current.sort_values(lastDate, axis=0, ascending=False)
current = current.iloc[:11, :]
filtered = current.iloc[:11, :]
filtered = filtered.drop("Others", axis=0) # Drop others, as this data is suspicious

diff = pd.DataFrame()
for i in range(10 - sampleLength, 0, -1):
    caption = "{0}-{1}".format(filtered.columns[-i - sampleLength], filtered.columns[-i])
    first = filtered.iloc[:, -i - sampleLength]
    last = filtered.iloc[:, -i]
    diff[caption] = last / first
    diff[caption] = diff[caption]**(1./sampleLength)
    
diffWAvg = pd.DataFrame()
diffWAvg["mean"] = diff.mean(axis=1)
diffWAvg["std"] = diff.std(axis=1)
diffAvg = pd.concat([diff, diffWAvg], axis=1)

diffWAvg2 = pd.DataFrame()
diffWAvg2["mean"] = diffAvg.mean(axis=0)
diffWAvg2["std"] = diffAvg.std(axis=0)
diffWAvg2 = diffWAvg2.transpose()
diffWAvg2.loc["std", "std"] = (diffWAvg2.loc["mean", "std"]**2 + diffWAvg2.loc["std", "mean"]**2)**0.5

diffAvg = pd.concat([diffAvg, diffWAvg2], axis=0)

# Display

def chooseColor(col, row):
    if row == col == "mean" or row == col == "std":
        return "background: red"
    if col in ("mean", "std"):
        return "background: pink"
    if row in ("mean", "std"):
        return "background: pink"
    return ""

diffAvg.style.apply(lambda x: [chooseColor(x.name, i) for i,_ in x.iteritems()])


# The table above shows the average growth rate (for number of total cases minus deaths and recoveries) per day on a given 3 day time period. As you can probably see, South-Korea has nearly managed to level off and Japan is also not too far.
# 
# For the rest of us, however, the situation is rather severe. Every day the number of coronavirus infections increase by a factor of 1.28 +/- 0.15, at single standard deviation. In other words, the average growth rate during the last 1-2 weeks is roughly 1.3x per day, 6x per week, and 2000x per month. That's scary, but will likely continue for some time until people totally change their behaviour or a significant proportion of the population is infected.

# ## A word on data discrepancies
# 
# Apparently, there are slight differences between published datasets. For example, in Italy, the Kaggle dataset and worldometers.info data don't match between 20 Feb and 23 Feb. The differentiating period tends to overlap extremely well with the time diagnosis methods were changed, and are present only for a short time period. This analysis uses datapoints from later time, when data matches exactly.
