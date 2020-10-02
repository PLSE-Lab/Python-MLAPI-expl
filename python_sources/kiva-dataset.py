#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize


# In[ ]:


kiva_loans = pd.read_csv("../input/kiva_loans.csv")
loan_theme_ids = pd.read_csv("../input/loan_theme_ids.csv")
kiva_region_location = pd.read_csv("../input/kiva_mpi_region_locations.csv")
loan_themes_by_region = pd.read_csv("../input/loan_themes_by_region.csv")


# In[ ]:


kiva_loans.head(5)


# In[ ]:


kiva_loans['date'] = pd.to_datetime(kiva_loans['date'])
kiva_loans['posted_time'] = pd.to_datetime(kiva_loans['posted_time'])
kiva_loans['disbursed_time'] = pd.to_datetime(kiva_loans['disbursed_time'])
kiva_loans['funded_time'] = pd.to_datetime(kiva_loans['funded_time'])


# In[ ]:


borrower_genders = kiva_loans['borrower_genders']
borrower_genders = borrower_genders.str.split(', ')
#.apply(lambda x:x.count("female"))
borrower_genders.fillna("", inplace=True)
# print(borrower_genders.value_counts())
kiva_loans['female_borrowers'] = borrower_genders.apply(lambda x:x.count('female'))
kiva_loans['male_borrowers'] = borrower_genders.apply(lambda x:x.count('male'))
kiva_loans['borrowers'] = kiva_loans['female_borrowers'] + kiva_loans['male_borrowers']


# In[ ]:


sector = kiva_loans['sector'].value_counts()
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
ax = sns.barplot(x=sector.index, y=sector.values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


kiva_loans_agri_secator = kiva_loans[kiva_loans.sector == "Agriculture"]
kiva_loans_food_secator = kiva_loans[kiva_loans.sector == "Food"]
kiva_loans_retail_secator = kiva_loans[kiva_loans.sector == "Retail"]


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
agri_activity_freq = kiva_loans_agri_secator.activity.value_counts()
ax = sns.barplot(x=agri_activity_freq.index, y=agri_activity_freq.values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
food_activity_freq = kiva_loans_food_secator.activity.value_counts()[:12]
ax = sns.barplot(x=food_activity_freq.index, y=food_activity_freq.values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
retail_activity_freq = kiva_loans_retail_secator.activity.value_counts()[:12]
ax = sns.barplot(x=retail_activity_freq.index, y=retail_activity_freq.values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
loan_amount_by_country = kiva_loans.groupby('country').sum()['loan_amount'].sort_values(ascending=False)[:10]
plt.figure()
ax = sns.barplot(x=loan_amount_by_country.index, y=loan_amount_by_country.values, color="salmon")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
sns.distplot(kiva_loans['loan_amount'])


# In[ ]:


country = kiva_loans['country']
country_counts = country.value_counts()
other = country_counts[10:].sum()
country_counts = country_counts[:10]
country_counts.set_value("others", other)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
sns.set()
ax = sns.barplot(x=country_counts.index, y=country_counts.values, palette="Blues_d")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # explode 1st slic
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
country_counts.plot(kind="pie",explode=explode, autopct='%1.1f%%')


# In[ ]:


irregular = kiva_loans[kiva_loans.repayment_interval == "irregular"].country.value_counts()
plot_irregular = irregular.sort_values(ascending=False).head(10)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
ax = sns.barplot(x=plot_irregular.index, y=plot_irregular.values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


counting_irregular = kiva_loans.country.value_counts()[
    irregular.index
]
counting_irregular.sort_values(ascending=False).head(10)


# In[ ]:


irregular_payments_poor = (irregular/counting_irregular).sort_values(ascending=False).head(10) # poor payment
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
print(irregular_payments_poor)
ax = sns.barplot(x=irregular_payments_poor.index, y=irregular_payments_poor.values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


good_payment_irregular = (irregular/counting_irregular).sort_values().head(10) # good payment
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
print(good_payment_irregular)
ax = sns.barplot(x=good_payment_irregular.index, y=good_payment_irregular.values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
sns.regplot(x="borrowers", y="loan_amount", data=kiva_loans, x_estimator=np.mean)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
sns.pointplot(x="repayment_interval", y="loan_amount", data=kiva_loans)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
sns.pointplot(x="repayment_interval", y="funded_amount", data=kiva_loans)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
sns.pointplot(x="repayment_interval", y="borrowers", data=kiva_loans)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
sns.pointplot(x="repayment_interval", y="male_borrowers", data=kiva_loans)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
sns.pointplot(x="repayment_interval", y="female_borrowers", data=kiva_loans)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
sns.pointplot(x="repayment_interval", y="male_borrowers", data=kiva_loans[
    (kiva_loans['female_borrowers'] > 0) & (kiva_loans['male_borrowers'] > 0) ]
             )


# In[ ]:


loan_theme_type = loan_theme_ids['Loan Theme Type'].value_counts()[:10]


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
ax = sns.barplot(x=loan_theme_type.index,y=loan_theme_type.values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


loan_theme_by_region_vc_sec = loan_themes_by_region.sector.value_counts()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
ax = sns.barplot(x=loan_theme_by_region_vc_sec.index, y= loan_theme_by_region_vc_sec.values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


loan_themes_by_region_fpn_vc = loan_themes_by_region['Field Partner Name'].value_counts()[:10]
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
ax = sns.barplot(x=loan_themes_by_region_fpn_vc.index, y=loan_themes_by_region_fpn_vc.values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


kiva_region_location.head(5)


# In[ ]:


regions = kiva_region_location[['lat', 'lon', 'MPI', 'world_region']].dropna()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax = plt.subplots(figsize=(10,10))
m = Basemap(resolution='c', # c, l, i, h, f or None
            projection='merc',
           # lat_0=36, lon_0=68,
            llcrnrlon=-25.4, llcrnrlat=-47.1, urcrnrlon=63.8, urcrnrlat=37.5)
# westlimit=-25.4; southlimit=-47.1; eastlimit=63.8; northlimit=37.5
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.drawcountries()
m.drawcoastlines()
m.drawrivers(color='#0000ff')
def maplonlat(pos):
    x, y = m(pos[1], pos[0])
    size = pos[2]*25
    m.plot(x, y, 'o', markersize=size, color='#444444', alpha=0.8)
output = regions[regions.world_region == "Sub-Saharan Africa"].apply(maplonlat, axis=1)
# m.readshapefile('maps/States/Admin2', 'areas')


# In[ ]:


kiva_region_location.world_region.value_counts()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax = plt.subplots(figsize=(10,10))
m = Basemap(resolution='c', # c, l, i, h, f or None
            projection='merc',
           # lat_0=36, lon_0=68,
            llcrnrlon=-118.7, llcrnrlat=-56.1, urcrnrlon=-28.7, urcrnrlat=32.7)
# westlimit=-118.7; southlimit=-56.1; eastlimit=-28.7; northlimit=32.7
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.drawcountries()
m.drawcoastlines()
m.drawrivers(color='#0000ff')
def maplonlat(pos):
    x, y = m(pos[1], pos[0])
    size = pos[2]*25
    m.plot(x, y, 'o', markersize=size, color='#444444', alpha=0.8)
output = regions[regions.world_region == "Latin America and Caribbean"].apply(maplonlat, axis=1)
# m.readshapefile('maps/States/Admin2', 'areas')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax = plt.subplots(figsize=(10,10))
m = Basemap(resolution='c', # c, l, i, h, f or None
            projection='merc',
           # lat_0=36, lon_0=68,
            llcrnrlon=24.7, llcrnrlat=11.8, urcrnrlon=63.3, urcrnrlat=42.4)
# westlimit=24.7; southlimit=11.8; eastlimit=63.3; northlimit=42.4
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.drawcountries()
m.drawcoastlines()
m.drawrivers(color='#0000ff')
def maplonlat(pos):
    x, y = m(pos[1], pos[0])
    size = pos[2]*25
    m.plot(x, y, 'o', markersize=size, color='#444444', alpha=0.8)
output = regions[regions.world_region == "Arab States"].apply(maplonlat, axis=1)
# m.readshapefile('maps/States/Admin2', 'areas')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax = plt.subplots(figsize=(10,10))
m = Basemap(resolution='i', # c, l, i, h, f or None
            projection='merc',
           # lat_0=36, lon_0=68,
            llcrnrlon=60.5, llcrnrlat=-7.5, urcrnrlon=97.4, urcrnrlat=38.5)
# westlimit=60.5; southlimit=-7.5; eastlimit=97.4; northlimit=38.5
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.drawcountries()
m.drawcoastlines()
def maplonlat(pos):
    x, y = m(pos[1], pos[0])
    size = pos[2]*25
    m.plot(x, y, 'o', markersize=size, color='#444444', alpha=0.8)
output = regions[regions.world_region == "South Asia"].apply(maplonlat, axis=1)
# m.readshapefile('maps/States/Admin2', 'areas')


# In[ ]:


#westlimit=92.2; southlimit=-11.1; eastlimit=141.0; northlimit=28.5
get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax = plt.subplots(figsize=(10,10))
m = Basemap(resolution='i', # c, l, i, h, f or None
            projection='merc',
           # lat_0=36, lon_0=68,
            llcrnrlon=92.2, llcrnrlat=-11.1, urcrnrlon=141.0, urcrnrlat=28.5)
# westlimit=60.5; southlimit=-7.5; eastlimit=97.4; northlimit=38.5
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.drawcountries()
m.drawcoastlines()
def maplonlat(pos):
    x, y = m(pos[1], pos[0])
    size = pos[2]*25
    m.plot(x, y, 'o', markersize=size, color='#444444', alpha=0.8)
output = regions[regions.world_region == "East Asia and the Pacific"].apply(maplonlat, axis=1)
# m.readshapefile('maps/States/Admin2', 'areas')


# In[ ]:




