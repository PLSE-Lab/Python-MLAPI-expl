# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import beta
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

data = pd.read_csv('../input/burritodata_092616.csv')
data.columns
data['Date'] = pd.to_datetime(data['Date'])
data['month'] = data['Date'].apply(lambda x: x.month)
data['day'] = data['Date'].apply(lambda x: x.weekday())
day_data = data.groupby('day')
mo_data = data.groupby('month')

#mean_data_by_mo = mo_data.aggregate(lambda x: np.mean(x))

mo = []
monthly_cost = []
monthly_yelp = []
monthly_goog = []
for month in mo_data:
    month = month[1]
    mo.append(month['month'].unique())
    monthly_cost.append(month['Cost'].mean())
    monthly_yelp.append(month['Yelp'].mean())
    monthly_goog.append(month['Google'].mean())


avg_rate_mo = [x*y/5 for x,y in zip(monthly_yelp,monthly_goog)]
month_labs = ['January','February','March','April','May','June','July','August','September']
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(range(len(mo)),avg_rate_mo)
ax3.plot(mo,monthly_cost)
#ax2.set_xticks(mo)
ax3.set_xticklabels(month_labs)
ax3.legend(['Overall Rating','Average Burrito Cost'], loc = 'center right')
plt.title('Monthly Rating vs Cost')
fileName3 = 'mothly_rating_vs_cost'
plt.savefig(fileName3, type = 'png')
#monthly_cost = mean_data_by_mo['Cost']
#monthly_yelp = mean_data_by_mo['Yelp']
#monthly_goog = mean_data_by_mo['Google']
    

#mean_data_by_day = day_data.aggregate(lambda x: np.mean(x))

#daily_cost = mean_data_by_day['Cost']
#daily_yelp = mean_data_by_day['Yelp']
#daily_goog = mean_data_by_day['Google']

daily_cost = []
daily_yelp = []
daily_goog = []
yelp_4_5 = [0] * 2
goog_4_5 = [0] * 2
yelp_daily = []
goog_daily = []
tot_daily = []
for d in day_data:
    d = d[1]
    daily_cost.append(d['Cost'].mean())
    daily_yelp.append(d['Yelp'].mean())
    daily_goog.append(d['Google'].mean())
    yelp_4 = d[(d['Yelp']>= 4 )& (d['Yelp']<5)]['Yelp'].count()
    yelp_5 = d[d['Yelp'] == 5]['Yelp'].count()
    yelp_daily.append((yelp_4,yelp_5))
    yelp_4_5[0] = yelp_4_5[0] + yelp_4
    yelp_4_5[1] = yelp_4_5[1] + yelp_5

    
    goog_4 = d[(d['Google']>= 4 )& (d['Google']<5)]['Google'].count()
    goog_5 = d[d['Google'] == 5]['Google'].count()
    goog_daily.append((goog_4,goog_5))
    goog_4_5[0] = goog_4_5[0] + goog_4
    goog_4_5[1] = goog_4_5[1] + goog_5
    tot_daily.append(d['Yelp'].count() + d['Google'].count())

#not enough points to effectively do day by day distributions
#turns out no ratings of 5
tot_4 = yelp_4_5[0] + goog_4_5[0]
tot = sum(tot_daily)
tot_4
tot
#Assume that the probability of rating 1-5 is uniformly distributed (each has prob 1/5), or also beta(1,1)
#The number of 4 or above ratings is Y and is distributed binomial(110, p),
#Using the beta-binomial conjucacy, 
#we know that the posterior probability of observing a rating of 4 or more is distributed 
#is beta(84,28)
fig, ax = plt.subplots(1, 1)
a = 84
b = 28
x = np.linspace(0,1, 1000)
ax.plot(x, beta.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='beta pdf')
plt.title("Density plot of Probability of 4 or Above Rating")
fileName = 'density_plot'
plt.savefig(fileName, type = 'png')

#Look at the cost vs rating new rating metric day by day
day = range(0,7)
day_labs = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(day,daily_yelp)
ax1.plot(day,daily_goog)
ax1.plot(day,daily_cost)
ax1.legend(['Yelp Rating', 'Google Rating', 'Average Burrito Cost'], loc = 'center right')
ax1.set_xticks(day)
ax1.set_xticklabels(day_labs)
fileName1 = 'raw_rating_vs_cost'
plt.savefig(fileName1, type = 'png')

#new metric for the rating that is based on product of yelp and google ratings, but still out of 5
avg_rate = [x*y/5 for x,y in zip(daily_yelp,daily_goog)]

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(day,avg_rate)
ax2.plot(day,daily_cost)
ax2.set_xticks(day)
ax2.set_xticklabels(day_labs)
ax2.legend(['Overall Rating','Average Burrito Cost'], loc = 'center right')
fileName2 = 'rating_vs_cost'
plt.savefig(fileName2, type = 'png')
#We see that under the new metric, there is a clear negative correlation between the overall rating and the average burrito cost

