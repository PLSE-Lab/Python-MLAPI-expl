#!/usr/bin/env python
# coding: utf-8

# While I was looking for data of confirmed cases in China, I found this article published on February 24 by WHO
# 
# https://www.who.int/docs/default-source/coronaviruse/who-china-joint-mission-on-covid-19-final-report.pdf
# 
# in which, on page 7, they present a very interesting plot comparing symptom onset vs date of diagnosis data of "cases for all of China" (as written in fig.3 caption) from January 1 to February 21. Since there is a lag time between the start of illness and diagnosis, this plot can give an idea of the true daily coronavirus cases.
# 
# Since I was not able to find the data used in the article, I estimated the values from the plot of the article (see image below).
# The column 'CN_Onset_WHO' contains the estimated daily symptom onset, while the column 'CN_Diag_WHO' contains the estimated daily diagnosis.
# 
# ![a](https://i.imgur.com/sy5kf1e.png)
# 
# The sum of all the estimated orange bars is 56860, which is very near to the value of the article which is 56848 (fig.3 bottom panel).
# 
# In the following I tried to use the above Chinese symptom onset data to estimate the Italian symptom onset time series, though with a very rudimental and naive approach (I'm new to this field). I hope some more experienced user could run a more accurate analysis.
# 
# My initial intent was to find an ARIMA model, train it on article data and fit it to Italian data.
# Running the following R code
# 
# * library(forecast) library(ggplot2) library(tseries)
# * it_cases <- c(1,12,42,74,97,93,78,250,238,240,566,342,466,587,769,778,1247,1492,1797,1577,1713,2651,2547,3497,3590,3233)
# * adf.test(it_cases, alternative = "stationary") # it_cases is not stationary with p-value = 0.978
# * d1 = diff(it_cases, differences = 1)
# * adf.test(d1, alternative = "stationary")       # differenced series is stationary with p-value = 0.1985
# * Acf(d1, main='ACF for Differenced Series')     # there aren't significant values
# * Pacf(d1, main='PACF for Differenced Series')   # there aren't significant values
# * auto.arima(it_cases, seasonal=FALSE, stepwise=FALSE, approximation=FALSE)
# 
# last command suggests ARIMA(0,1,0) with drift. But I don't yet how to train and fit.
# 
# P.S. There is another article with similar data (https://jamanetwork.com/journals/jama/fullarticle/2762130) but even if the plots of the two articles have the same shape, the heights of the bars are proportionally different, for example the highest blue bars on February 1 are 3700 vs 3100).

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
data = pd.read_csv('../input/dataset/dataset.csv');

dates        = data['Date']          # dates
cn_cases_wom = data['CN_Diag_WOM']   # Chinese daily confirmed cases (Jan23-Mar15) provided by worldometers.info
cn_cases     = data['CN_Diag_CSSE']  # Chinese daily confirmed cases (Jan23-Mar15) provided by Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE)
art_onset    = data['CN_Onset_WHO']  # Chinese daily onset symptom cases (Jan1-Feb21) provided by the WHO article (link above)
art_cases    = data['CN_Diag_WHO']   # Chinese daily cases (Jan21-Feb21) provided by the WHO article (link above)
it_cases     = data['IT_Diag']       # Italian daily confirmed cases


# In[ ]:


# Firstly, I plotted Chinese daily cases by CSSE vs Chinese daily cases by WOM
# to check if they were equal

# plot data from Jan 23 to Mar 16
fig, ax = plt.subplots(figsize=(15, 5))
line1, = ax.plot(dates[22:76], cn_cases_wom[22:76], label='WOM')
line2, = ax.plot(dates[22:76], cn_cases[22:76], label='CSSE')
ax.legend()
ax.set_title('CHINA: daily cases by WOM vs. daily cases by CSSE')
ax.tick_params(axis='x',labelrotation=90)
plt.show()

# The two time series are not equal...why? Did somebody already noticed this fact?
# In the following I will use the one by CSSE since it is the one used here on Kaggle by other users


# In[ ]:


# Secondly, I plotted Real Chinese data by CSSE vs. estimated Diagnosis time series of the article

# plot data from Jan 21 to Feb 21
fig, ax = plt.subplots(figsize=(15, 5))
line1, = ax.plot(dates[20:52], cn_cases[20:52], label='Real Diagnosis')
line2, = ax.plot(dates[20:52], art_cases[20:52], label='Article Diagnosis')
ax.legend()
ax.set_title('CHINA: Real daily cases vs Article daily cases')
ax.tick_params(axis='x',labelrotation=90)
plt.show()

print('real data: total number of cases up to Feb 21',sum(cn_cases[20:52]))
print('article data: total number of cases',sum(art_cases[20:52]))

#
# We immediately see that the Diagnosis series of the article is very different from the real one:
# 
# - The values are different.
# - The cumulative confirmed cases number is different: the article says 56848, while the total cases in china up to February 21 (last date in the plot contained in the article) are 75000.
# - The real data presents a big spike on February 13 of about 15000 cases, while the article plot presents only 2000 new cases on that day.
#   (about the spike, around February 12 or 13, there was a change in how China confirmed new cases: they included criteria they didn't have before)
#
# It seems that the series of the article is just a rude approximation of the real data.
# The article was published on February 24, so maybe their data were incomplete?
# Did somebody already notice this?
#


# In[ ]:


# Rates of growth for the Onset series (not considering zero values)
# it will be used to estimate the Italian onset series
rg = []
for i in range(0,len(art_onset[0:51])):
    rg.append( art_onset[i+1]/art_onset[i] )
    
# Ratio of onset over diagnosis for each day (not considering zero values)
# it will be used to predict / forecast the Italian diagnosis series
ood = []
for i in range(20,52):
    ood.append( art_onset[i]/art_cases[i] )


# In[ ]:


# Estimating the Italian onset series

it_onset = [0] * 99
# To generate the Italian onset series I thought about the following procedure
# - the first value to be generated is the one corresponding to the Wuhan shut down / closing, which happened on January 23,
#   and it is generated by multypling the Italian diagnosis of March 10, on which Italy shut down all the country, by the ratio
#   of Chinese onset over diagnosis for January 23, i.e.
it_onset[69] = ood[2] * it_cases[69]

# - the previous (down to February 17) values are computed dividing by the rates of growth
for i in list(reversed(range(22))):
    it_onset[48+i-1] = it_onset[48+i] / rg[i] # 48 = 69-21

# - the next (up to April 8) values are computed multiplying by the rates of growth
for i in range(22,51):
    it_onset[48+i] = it_onset[48+i-1] * rg[i]


# In[ ]:


# Predicting / forecasting the Italian diagnosis series

# To forecast the Italian cases of the next day, divide the Italian onset of the same day by the corresponding Chinese ratio onset over diagnosis

for i in range(0,22):  # we forecast 22 values
    it_cases[77+i] = it_onset[77+i] / ood[10+i]  # it_cases has 78 values in the dataset


# In[ ]:


# Plot Italian time series from Feb 17 to Apr 8

labels = dates[47:100]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(15, 5))
rects1 = ax.bar(x - width/2, it_onset[47:100], width, label='Symptom onset')
rects2 = ax.bar(x + width/2, it_cases[47:100], width, label='Diagnosis')

ax.set_title('ITALY: Daily symptom onset (estimated) vs diagnosis (real up to 3/17, estimated from 3/18)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.tick_params(axis='x',labelrotation=90)
ax.legend()

fig.tight_layout()
plt.show()

# The big jump between 3/17 and 3/18 denotes a bad estimate
# How to properly estimate symptom onset and forecasted diagnosis?

