#!/usr/bin/env python
# coding: utf-8

# In this Kernel, EDA was performed to analyse and build simple visualizations to analyse and study the trade deficit change over the years from 2010-2018. This has also helped in determining the commodities that have been contibuting heavily to the trade deficit.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.graph_objs as pltgo
import plotly.express as px

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# The datasets are studied here to understand each column's type in the datasets and the count of data

# In[ ]:


export_2010_2018 = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_export.csv')
import_2010_2018 = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_import.csv')

print('Export Data:')
print(export_2010_2018.describe())

print('######################################')

print('Import Data:')
print(import_2010_2018.describe())


# The count of missing values were found to cleanse the data.

# In[ ]:


export_2010_2018.isnull().sum()


# The zero value rows were removed from both the datasets to remove extra rows.

# In[ ]:


export_2010_2018 = export_2010_2018[export_2010_2018.value!=0]
import_2010_2018 = import_2010_2018[import_2010_2018.value!=0]
export_2010_2018.year = pd.Categorical(export_2010_2018.year)
import_2010_2018.year = pd.Categorical(import_2010_2018.year)
print(export_2010_2018.describe())
print(import_2010_2018.describe())


# The total imports and exports value are grouped and summed year wise. The trade deficit is also calculated for each year. Now a new data frame is created to segregate imports, exports and trade deficit by year.

# In[ ]:


year = export_2010_2018['year'].unique()
import_data = import_2010_2018.groupby('year').value.sum()
export_data = export_2010_2018.groupby('year').value.sum()
trade_deficit = import_data-export_data
data={'imports':import_data,'exports':export_data,'trade_deficit':export_data-import_data}
trade_data = pd.DataFrame(data)
print(trade_data)


# A bar plot is created to compare imports, exports and trade deficit across years. This shows the pattern of how the trend has been across the years. This graph explains us how the imports are affecting the trade deficit. With increase in imports, the trade deficit is also increasing proptionately.

# In[ ]:


fig_trade_data = pltgo.Figure()
fig_trade_data.add_trace(pltgo.Scatter(x=trade_data.index, y=trade_data.imports,
                    mode='lines+markers',
                    name='imports'))
fig_trade_data.add_trace(pltgo.Scatter(x=trade_data.index, y=trade_data.exports,
                    mode='lines+markers',
                    name='exports'))
fig_trade_data.add_trace(pltgo.Scatter(x=trade_data.index, y=trade_data.trade_deficit,
                    mode='lines+markers', name='Trade Deficit'))
fig_trade_data.show()


# The growth percentage for the imports from 2010 to 2018 have been created across commodities. This done to find the commodities whose imports have increased exponentially across years.
# 
# The top 5 and the bottom 5 commodities that have shown drastic percentage change in imports have been determined.

# In[ ]:


imports_2010 = import_2010_2018[import_2010_2018.year==2010]
imports_2018 = import_2010_2018[import_2010_2018.year==2018]
import_commodity_2010 = imports_2010.groupby(['Commodity']).value.sum()
import_commodity_2018 = imports_2018.groupby(['Commodity']).value.sum()
imports_growth_percentage = (import_commodity_2018-import_commodity_2010)*100/import_commodity_2018

top5_import_commodity=imports_growth_percentage.nlargest(5)
bottom5_import_commodity = imports_growth_percentage.nsmallest(5)

print('Top 5 commodities with high percentage change from 2010-2018')
print(top5_import_commodity)
print('#################################################################################')
print('Top 5 commodities with low percentage change from 2010-2018')
print(bottom5_import_commodity)


# The top 5 commodities with increase in import growth % and the bottom 5 commodities with dip in growth % are segregated by year and commodity to find their value and growth over the years.

# In[ ]:


largest_import_commodities = pd.DataFrame()
for commodity in top5_import_commodity.index:
    largest_import_commodities=largest_import_commodities.append(import_2010_2018[import_2010_2018.Commodity==commodity])
largest_commodities_grouped = largest_import_commodities.groupby(['year','Commodity']).value.sum()
largest_commodities_grouped_data = {'Year':largest_commodities_grouped.index.get_level_values(0),'Commodity':largest_commodities_grouped.index.get_level_values(1),'Values':largest_commodities_grouped.values}
largest_commodities_grouped_df = pd.DataFrame(largest_commodities_grouped_data)

smallest_import_commodities = pd.DataFrame()
for commodity in bottom5_import_commodity.index:
    smallest_import_commodities=smallest_import_commodities.append(import_2010_2018[import_2010_2018.Commodity==commodity])
smallest_commodities_grouped = smallest_import_commodities.groupby(['year','Commodity']).value.sum()
smallest_commodities_grouped_data = {'Year':smallest_commodities_grouped.index.get_level_values(0),'Commodity':smallest_commodities_grouped.index.get_level_values(1),'Values':smallest_commodities_grouped.values}
smallest_commodities_grouped_df = pd.DataFrame(smallest_commodities_grouped_data)


# The top 5 commodities with the highest change in imports have been identified and the growth range over the years can be visualised in the below graph.

# In[ ]:


fig_largest_imports = px.line(largest_commodities_grouped_df, x=largest_commodities_grouped_df.Year, y=largest_commodities_grouped_df.Values, color='Commodity')
fig_largest_imports.update_layout(legend=dict(x=-.1, y=1.3))
fig_largest_imports.show()


# The bottom 5 commodities with the lowest change in imports have been identified and the growth range over the years can be visualised in the below graph.

# In[ ]:


fig_smallest_imports = px.line(smallest_commodities_grouped_df, x=smallest_commodities_grouped_df.Year, y=smallest_commodities_grouped_df.Values, color='Commodity')
fig_smallest_imports.update_layout(legend=dict(x=-.1, y=1.3))
fig_smallest_imports.show()


# 

# 
