#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # # **HOW TO INCREASE WEEKEND ONLINE SALES**
# 
# # # # # *Introduction*
# 
# As COVID-19 continues to spread across the globe and most people of the world are practicing 'Stay Home, Save Life', ONLIE SHOPPING is not only an option anymore, but an essential daily life solution.
# 
# We have reasons to believe that online sales will play a more important role in our business. It's also believed that now is the right time to improved our online performance.
# 
# Based on the 'Online Sales Data' we collected as follows, let's take a close look to find facts and inspiration hepls business stratege.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


E_seller = pd.read_csv('../input/online-shoppers-intention/online_shoppers_intention.csv')
E_seller.head()


# In[ ]:


print(E_seller.info())


# ### Data Content

# To better understand the dataset, here's the descriptions of each column:
# 
# 1. Administrative: Number of Administrative Pages viewed by an online visitor;
# 2. Administrative_Duration: Seconds of Time of Administrative Pages viewed by an online visitor;
# 3. Informational: Number of Informational Pages viewed by an online visitor;
# 4. Informational_Duration: Seconds of Time of Informational Pages viewed by an online visitor;
# 5. ProductRelated: Number of Product Related Pages viewed by an online visitor;
# 6. ProductRelated_Duration: Seconds of Time of Product Related Pages viewed by an online visitor;
# 7. BounceRates: Percentage of the visitor exiting pages without triggering any additional tasks;
# 8. ExitRates: Percentage of pageviews the visitor ended at;
# 9. PageValues: Average Page Value calculated by the visitor purchase transaction value divide all pageviews;
# 10. SpecialDay: Closeness of the pageviewing day to a special day or holiday;
# 11. Month: Month of the pageviewing;
# 12. OperatingSystems: Operating system which an online visitor uses for pageviewing;
# 13. Browser: Browser which an online visitor uses for pageviewing;
# 14. Region: Geographic region where an online visitor locates;
# 15. TrafficType: Traffic category of an online visitor belongs to;
# 16. VisitorType: Visitor categorized in New Visitor, Returning Visitor, or Other;
# 17. Weekend: Whether or not an online visitor visits on weekend;
# 18. Revenue: Whether or not an online visitor completes the purchase.

# ### Data Cleansing

# In[ ]:


E_seller[['OperatingSystems', 'Browser', 'Region', 'TrafficType']]=E_seller[['OperatingSystems', 'Browser', 'Region', 'TrafficType']].astype(str)
E_seller=E_seller.fillna(0)
print(E_seller.info())


# In[ ]:


E_seller.describe()


# ### Data Anylysis

# ### 1. How's the Transaction performance?

# In[ ]:


# Transaction Not Completed VS. Transaction Completed

reve_count=E_seller['Revenue'].value_counts()

transaction=reve_count.plot.bar(width=0.8, color=['gray', 'orange']) 
transaction.set_title('Whether or not the transaction happened?', color='red', fontsize=16)
transaction.set_xlabel('Transaction Happened', color='black', fontsize=14)
transaction.set_ylabel('Visits Count', color='black', fontsize=14)
transaction.set_xticklabels(('False', 'True'), rotation='horizontal')
transaction.text(-0.17, 9200, str((~E_seller['Revenue']).sum()), fontsize=15, color='white', fontweight='bold')
transaction.text(0.88, 800, str(E_seller['Revenue'].sum()), fontsize=15, color='white', fontweight='bold')

plt.show()


# #### INSIGHT:  `Only 20% Transaction Completed, Big room for improvement`

# ### 2. Whether more average visit on weekday or weekend?

# In[ ]:


# Sum Visit Weekday vs. Sum Visit Weekend

weekend_count=E_seller['Weekend'].value_counts()

weekend=weekend_count.plot.bar(width=0.8, color=['gray', 'orange']) 
weekend.set_title('Whether or not visits on weekend?', color='red', fontsize=16)
weekend.set_xlabel('On Weekend', color='black', fontsize=14)
weekend.set_ylabel('Visits Count', color='black', fontsize=14)
weekend.set_xticklabels(('False', 'True'), rotation='horizontal')
weekend.text(-0.12, 8500, str((~E_seller['Weekend']).sum()), fontsize=15, color='white', fontweight='bold')
weekend.text(0.88, 1800, str(E_seller['Weekend'].sum()), fontsize=15, color='white', fontweight='bold')

plt.show()


# Averave Daily Visit Weekday vs. Average Daily Visit Weekend

x=[i for i in range(1,3)]
y=[(~E_seller['Weekend']).sum()/5, E_seller['Weekend'].sum()/2]

bars = plt.bar(x, height=y, width=.7, color=['gray', 'orange'])

xlocs, xlabs = plt.xticks()
xlocs=[i for i in x]
xlabs=[i for i in x]

plt.title('Whether more daily average visits on weekday or weekend?', color='red', fontsize=16)
plt.xlabel('Visit On', color='black', fontsize=14)
plt.ylabel('Average Daily Visit', color='black', fontsize=14)
plt.xticks(xlocs, ('Weekday', 'Weekend'))

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x()+0.18, yval-200, yval, fontsize=15, color='white', fontweight='bold')
    
plt.show()


# ### INSIGHT:  `~24% Less Visits on Weekend, Potential Needs Attention`

# ### 3. Whether more transaction on weekday or weedend?

# In[ ]:


# Average Daily transaction on Weekdays VS. on Weekends

weekend_revenue_avg = E_seller.query('Weekend == True & Revenue == True')['Revenue'].count()/2
weekday_revenue_avg = E_seller.query('Weekend == False & Revenue == True')['Revenue'].count()/5

b=[weekday_revenue_avg, weekend_revenue_avg]
a=[j for j in range(len(b))]

bars = plt.bar(a, height=b, width=.7, color=['gray', 'orange'])

xlocs, xlabs = plt.xticks()

xlocs=[j for j in a]
xlabs=[j for j in b]

plt.title('Whether more daily transaction on weekdays or weedends?', color='red', fontsize=16)
plt.xlabel('Transaction On', color='black', fontsize=14)
plt.ylabel('Average Transaction', color='black', fontsize=14)
plt.xticks(xlocs, ('Weekday', 'Weekend'))

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x()+0.22, yval-30, yval, fontsize=15, color='white', fontweight='bold')
    
plt.show()


# Average Daily Transaction Rate on Weekdays VS. on Weekends

weekend_revenue_pct = round(E_seller.query('Weekend == True & Revenue == True')['Revenue'].count()/E_seller.query('Weekend == True')['Revenue'].count()*100, 2)
weekday_revenue_pct = round(E_seller.query('Weekend == False & Revenue == True')['Revenue'].count()/E_seller.query('Weekend == False')['Revenue'].count()*100, 2)

d=[weekday_revenue_pct, weekend_revenue_pct]
c=[k for k in range(len(d))]

bars = plt.bar(c, height=d, width=.7, color=['gray', 'orange'])

xlocs, xlabs = plt.xticks()

xlocs=[k for k in c]
xlabs=[k for k in d]

plt.title('Whether higher daily transaction rate on weekdays or weedends?', color='red', fontsize=16)
plt.xlabel('Transaction On', color='black', fontsize=14)
plt.ylabel('Average Transaction', color='black', fontsize=14)
plt.xticks(xlocs, ('Weekday', 'Weekend'))

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x()+0.22, yval-2, str(yval)+'%', fontsize=15, color='white', fontweight='bold')


# ### INSIGHT:  `Less Transaction BUT Higher Transaction Rate, Weekend is More Valuable Time`

# ### 4. Which regions are import?
# 

# In[ ]:


# Regions Online Visitors In

region_count=E_seller['Region'].value_counts()
region=region_count.plot.barh(width=0.6) 
region.set_title('Which regions visitors online shopping in?', color='red', fontsize=16)
region.set_xlabel('Visitor Count', color='black', fontsize=14)
region.set_ylabel('Region', color='black', fontsize=14)

for i, v in enumerate(region_count):
    region.text(v+3, i-0.2, str(v), color='black', fontsize=10, fontweight='bold')

plt.show()


# Important Regions with Most Online Visitors

region1 = E_seller.query('Region == "1"')['Region'].count()
region2 = E_seller.query('Region == "2"')['Region'].count()
region3 = E_seller.query('Region == "3"')['Region'].count()
region4 = E_seller.query('Region == "4"')['Region'].count()
region_rest = E_seller['Region'].count() - region1 - region2 - region3 -region4

region_value = [region1, region3, region2, region4, region_rest]
region_name = ['Region 1', 'Region 3', 'Region 2', 'Region 4', 'Region 5~9']
region_colors = ['orange', 'orange', 'gray', 'gray', 'gray']
plt.pie(region_value, labels=region_name, colors=region_colors, startangle=60, explode=(0.1, 0.06, 0.03, 0.03, 0.03), autopct='%.1f%%', textprops={'fontsize': 12})
plt.title('Which regions are more import?', color='red', fontsize=16)

plt.show()


# ### INSIGHT:  `~60% Visitors are In 2 Regions, Region 1 & Region 3`

# ### 5. Which regions are more valuable?

# In[ ]:


# Important Regions with Most Online Transaction

region1 = E_seller.query('Region == "1" & Revenue == True')['Region'].count()
region2 = E_seller.query('Region == "2" & Revenue == True')['Region'].count()
region3 = E_seller.query('Region == "3" & Revenue == True')['Region'].count()
region4 = E_seller.query('Region == "4" & Revenue == True')['Region'].count()
region_rest = E_seller.query('Revenue == True')['Region'].count() - region1 - region2 - region3 -region4

region_value = [region1, region3, region2, region4, region_rest]
region_name = ['Region 1', 'Region 3', 'Region 2', 'Region 4', 'Region 5~9']
region_colors = ['orange', 'orange', 'gray', 'gray', 'gray']
plt.pie(region_value, labels=region_name, colors=region_colors, startangle=60, explode=(0.1, 0.06, 0.03, 0.03, 0.03), autopct='%.1f%%', textprops={'fontsize': 12})
plt.title('Which regions are more valuable?', color='red', fontsize=16)

plt.show()


# ### INSIGHT:  `Region 1 Brings In the Most Translation`

# ### 7. Which operating systems & browsers visitors prefer?
# 

# In[ ]:


# Operating systems visitors use

system_count=E_seller['OperatingSystems'].value_counts()
system=system_count.plot.barh(width=0.6) 
system.set_title('Which operating systems visitors prefer?', color='red', fontsize=18)
system.set_xlabel('Visitor Count', color='black', fontsize=14)
system.set_ylabel('Operating System', color='black', fontsize=14)

for i, v in enumerate(system_count):
    system.text(v+100, i-0.1, str(v), color='black', fontsize=10, fontweight='bold')

plt.show()


# Browsers visitors use

browser_count=E_seller['Browser'].value_counts()
browser=browser_count.plot.barh(width=0.6) 
browser.set_title('Which browser visitors prefer?', color='red', fontsize=18)
browser.set_xlabel('Visitor Count', color='black', fontsize=14)
browser.set_ylabel('Browser', color='black', fontsize=14)

for i, v in enumerate(browser_count):
    browser.text(v+100, i-0.2, str(v), color='black', fontsize=10, fontweight='bold')

plt.show()


# ### INSIGHT:  `Operating System #2 #1 #3 & Browser #2 #1 are Most Favorites`

# ### 8. What helps transaction completion?

# In[ ]:


# Valuables related to completed transaction

corr_df = E_seller.query('Revenue == True & Weekend == True')
E_seller_corr = corr_df.select_dtypes(include=['float64'])
corr = E_seller_corr.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(E_seller_corr.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(E_seller_corr.columns)
ax.set_yticklabels(E_seller_corr.columns)
plt.show()

corr.style.background_gradient(cmap='coolwarm')


# ### INSIGHT:  `First-sight Attractiveness & Longer Page Viewing Morelikely Leads to Revenue`
# 

# ### STRATEGY INSPIRATION:  
# `Invest more efforts to improve Weekend Online Sales`
# 
# `A. Target Region 1 and Region 3 as Key Areas of Promotion`
# 
# `B. Optimize Shopping Experience on Operating System #2 #1 #3 & Browser #2 #1`
# 
# `C. Increase Content Quality for higher Conversion Rate`
