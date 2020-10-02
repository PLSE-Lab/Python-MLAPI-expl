#!/usr/bin/env python
# coding: utf-8

# This notebook explores sales/transfers made by the Defense Logistics Agency (DLA) to state and local Law Enforcement Agencies (LEAs). From what I understand, state and local agencies might not pay the "acquisition value" for equipment. In some cases they may get the items for practically nothing under the provisions of a federal program. I use the terms purchase, transfer, and acquire interchangeably here.
# 
# 
# Disclaimer: I am not advocating for or against equipment purchases and/or militarization of police. I am curious to see what equipment LEAs have acquired and trends over time.
# 
# 

# In[ ]:


import pandas as pd
import skmem


# In[ ]:


sales = pd.read_csv('../input/police-violence-in-the-us/dod_equipment_purchases.csv', parse_dates= ['Ship Date'])
sales.columns = sales.columns.str.translate(str.maketrans({" ": "_", "(": None, ")": None})).str.lower()

sales = sales.fillna(-1)              .assign(demil_ic = lambda x: x.demil_ic.astype(int))              .iloc[:, :-1]

mr = skmem.MemReducer()
sales = mr.fit_transform(sales, float_cols=['acquisition_value'])


# The table below shows the most common items transferred since 2010. It's no surprise to me that rifles and rifle accessories are among the most popular items. In many parts of the US, police with handguns are often at a disadvantage to better armed criminals.
# 
# It appears that medical supplies and equipment are also popular.

# In[ ]:


sales.query('ship_date.dt.year > 2010')      .groupby('item_name')['quantity'].sum()      .sort_values(ascending=False)      .head(20).astype(int)


# This table shows the expensive items. The NSN for Aircraft, Fixed Wing is a generic one which means we can't see what airplane is purchased. Some research might uncover what sort of aircraft go for \\$17M-$22M used.

# In[ ]:


sales['cost_per_item'] = sales.acquisition_value / sales.quantity
sales.sort_values('cost_per_item', ascending=False).head(20)


# The table also shows that Payne County Oklahoma took receipt of a Mine Resistant Vehicle. They have a meth epidemic in that area so maybe that's why - just speculating.
# 
# ![mrap](http://www.military-today.com/apc/maxxpro_mrap.jpg)
# 

# This table shows who acquired the most equipment in the last 10 years. You may recall seeing the Arizona Dept of Public Safety earlier as a purchaser of many aircraft. According to their web site they use aircraft for transport, fire fighting, surveillance, and search and rescue. 

# In[ ]:


sales.query('ship_date.dt.year > 2010')      .groupby('station_name_lea')['acquisition_value'].sum()      .sort_values(ascending=False).head(20).astype(int)


# Hocking County Ohio, population 30K, spent \\$6M dollars, or about $500K per year. Here's what they have:

# In[ ]:


sales_hc = sales.query('ship_date.dt.year>2010 & '
                       'station_name_lea=="HOCKING CTY SHERIFF DEPT"') 
sales_hc_agg = sales_hc.groupby('item_name').agg(
            count=('quantity', 'sum'),
            spend=('acquisition_value', 'sum')
            )

sales_hc_agg.query('count>0').astype(int)             .sort_values('spend', ascending=False)
                   
 


# \\$5M of the $6M went to a SOMS-B mobile communications center, which broadcasts on radio and television frequencies. They also received a MRAP vehicle, rifles, and protective equipment.
Lastly, I want to jump back up and look at the trend of spending/transfers over time. Here is total value acquired year by year. Sloppy visuals aside, I find the chart very interesting. There's a huge spike in 2012-13 and then a steep reduction in the following years. I would bet this is due to US troops leaving Afghanistan and the military drawdown announced in 2012. The chart suggests that purchases/transfers are driven more by supply-side factors as opposed to demand, economy, or secular trends.
# In[ ]:


sales['year'] = sales.ship_date.dt.year
sales.groupby('year')['acquisition_value'].sum().plot.line()

