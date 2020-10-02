#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


#load the data
get_ipython().system('cp -r ../input/sgp-buildings/* ./')


# In[ ]:


tables=  {'total':{'csv_name':'singapore_building_gfa_energy.csv'},'commercial':{'csv_name':'energy_performance_data_2016.csv'}}
for t in tables:
    tables[t]['tbl'] = pd.read_csv(tables[t]['csv_name'])

column_key = {'buildingtype':'property type','grossfloorarea':'GFA m2','2016energyusintensity':'kWh/m2/yr'}
fields = ['property type','GFA m2','kWh/m2/yr']
tables['commercial']['tbl'].rename(columns=column_key,inplace=True)
comm = tables['commercial']['tbl'][fields].copy()

for f in ['GFA m2','kWh/m2/yr']:
    comm[f] = comm[f].str.replace(',','').astype(float)

comm['kWh/yr'] = comm['GFA m2']*comm['kWh/m2/yr']


# In[ ]:


comm['property type'].unique()


# In[ ]:


pct_OK = len(comm.dropna())/len(comm)
pct_OK


# In[ ]:


comm.dropna(inplace=True)


# In[ ]:


comm.head()


# In[ ]:


pvt = pd.pivot_table(comm,index='property type',values=['GFA m2','kWh/yr'],aggfunc='sum')
pvt['kWh/m2/yr'] = pvt['kWh/yr']/pvt['GFA m2']


# In[ ]:


total = tables['total']['tbl'].copy()
total.rename(columns={'kWh/yr/m2':'kWh/m2/yr'},inplace=True)
total['GWh/yr'] = total['mil m2']*total['kWh/m2/yr']
comm_totals = total[total['property type']=='commercial'].iloc[0]

pvt_scaled = pvt.copy()
pvt_scaled['mil m2'] = pvt['GFA m2']*comm_totals['mil m2']/pvt['GFA m2'].sum()
pvt_scaled['GWh/yr'] = pvt['kWh/yr']*comm_totals['GWh/yr']/pvt['kWh/yr'].sum()

pvt_scaled


# In[ ]:


fields = ['mil m2','kWh/m2/yr','GWh/yr']

ex_com = total.drop(total[total['property type']=='commercial'].index)
total = pd.concat([ex_com,pvt_scaled[fields].reset_index()],axis=0)

total


# In[ ]:


pop_mil = 5.6

pca = total.copy()
pca['m2/ca'] = total['mil m2']/pop_mil
pca['kWh/ca/yr'] = total['GWh/yr']/pop_mil

del pca['mil m2'], pca['GWh/yr']


# In[ ]:


pca.sort_values('kWh/ca/yr',ascending=False)


# In[ ]:


import seaborn as sb
sb.barplot(x='kWh/ca/yr',y='property type',data=pca.sort_values('kWh/ca/yr',ascending=False))


# In[ ]:


pca['kWh/ca/yr'].sum()


# In context, avg electricity consumption is 9,000 kWh/ca/yr of which building energy presented here is only ~ 28% and 500 kWh/ca/yr is from transport so the balance 6000 kWh/ca/yr is mostly from manufacturing.  The figure of 9,000 kWh/ca/yr is electricity only and does not include energy for transport using personal vehicles, taxis and public buses nor does it include holiday travel flights.  As a benchmark comparison from the HDI vs energy usage, typical energy usage for a country with HDI > 0.8 is in the range 2,000 - 4,000 kWh/ca/yr inclusive of industrial and transport energy.  
# 
# A quick scan of the building energy use highlights that home energy is the highest contributor at 1,000 kWh/ca/yr, so for buildings this is a reasonable starting point for efficiency improvement.   
# 
# Another question to consider is to look at the split of where a typical resident's energy usage is coming from, is it while they are at home, work, errands or leisure, considering for some simplifying assumptions about their time usage at each of these locations.

# In[ ]:


weekly_hrs = 168
m2_factors = dict(zip(list(pca['property type']),list(pca['m2/ca'])))
usage_factors = {
    'residential':(12*7+4*2)/weekly_hrs,
    'Office':0.35*40/weekly_hrs,
    'Retail':0.15*40/weekly_hrs,
    'Mixed Development':0.05*40/weekly_hrs,
    'education':0.25*40*9/12/weekly_hrs,
    'Hotel':0.05*40/weekly_hrs,
    'healthcare':0.05*40/weekly_hrs,
    'transport':0.02*40/weekly_hrs,
    'sports rec':0,
    'civil / community':0,
}
out_of_home_hrs = 16/weekly_hrs #excludes time in transit
out_of_home = ['Retail','Mixed Development','healthcare','transport',
          'sports rec','civil / community']
out_of_home_m2 = dict(zip(out_of_home,[m2_factors[x] for x in out_of_home]))
total_ooh_m2 = sum(out_of_home_m2.values())

for f in out_of_home:
    usage_factors[f] = usage_factors[f] + out_of_home_hrs*out_of_home_m2[f]/total_ooh_m2
usage_factors


# In[ ]:


hours_per_year = 24*365
usg = pd.DataFrame({'hrs/ca':usage_factors})
usg.index.name = 'property type'
df = pd.merge(pca,usg.reset_index(),on='property type')
df['W/ca'] = df['kWh/ca/yr']/(df['hrs/ca']*hours_per_year)*1000
df.sort_values('W/ca',ascending=False)


# In[ ]:


sb.barplot(x='W/ca',y='property type',
           data=df.drop(df[df['property type']=='Hotel'].index).sort_values('W/ca',ascending=False))


# Based on the weights input provided, after adjusting for the fact that more time is spent in the home other out of home energy uses appear more intense on a time-weighted basis.  The actual distribution between these sectors would require a more detailed validation of the usage weights.
