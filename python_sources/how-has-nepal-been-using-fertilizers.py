#!/usr/bin/env python
# coding: utf-8

# # Analyzing the use of fertilizer over the years in Nepal
# 
# 
# In this notebook i am going to analyze the fertilizer uses in My country. 
# 
# 
# ## Problem Definition
# 
# How is the import and export relationship of each type of fertilizer in My country Nepal over the years?
# 
# 
# ## Data
# 
# The dataset contains information on product amounts for the Production, Trade, Agriculture Use and Other Uses of chemical and mineral fertilizers products, over the time series 2002-present.
# The fertilizer statistics data are validated separately for a set of over thirty individual products. Both straight and compound fertilizers are included.
# 
# The data has 11 columns:
# 1. Area Code
# 2. Area
# 3. Item Code
# 4. Item
# 5. Element Code
# 6. Element
# 7. Year Code
# 8. Year
# 9. Unit
# 10. Value
# 11. Flag
# 
# ### Flags Column
# Flag column has eight different values which means the followings:
# 
# * A - Aggregate; may include official; semi-official; estimated or calculated data;
# * E - Expert sources from FAO (including other divisions);
# * Fb - Data obtained as a balance;
# * Fm - Manual Estimation;
# * P - Provisional official data;
# * Qm - Official data from questionnaires and/or national sources and/or COMTRADE (reporters);
# * R - Estimated data using trading partners database;
# * W - Data reported on country official publications or web sites (Official) or trade country files;
# 
# <img src='https://www4.gep.com/sites/default/files/blog-images/outlook-for-the-global-fertilizer-market.jpg'/>
# 
# 
# ## Preparing Tools
# 
# Study and understanding data properly is the challenging task in any data science problem. So external tools are the most to understand the data properly. Let's import some useful libraries.

# In[ ]:


# EDA and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ### Load data
# Loading the data using pandas.

# In[ ]:


df = pd.read_csv('/kaggle/input/fertilizers-by-product-fao/FertilizersProduct.csv',encoding='ISO-8859-1')
df.head()


# From the total country extracting only the data of my country.

# In[ ]:


data_nepal = df[(df['Area']=='Nepal')].reset_index(drop=True)
data_nepal.head()


# let's group the value so that we can clearly visualize in a single frame.

# In[ ]:


fertilizer_by_item = df.groupby(["Item"])["Value"].sum().reset_index().sort_values("Value",ascending=False).reset_index(drop=True)


# In[ ]:


fig = px.pie(fertilizer_by_item, values=fertilizer_by_item['Value'], 
             names=fertilizer_by_item['Item'],
             title='Amount of Fertilizer used in Nepal',
            )
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(
    template='plotly_dark'
)
fig.show()


# Now we can see that urea stood in the most used position in Nepal.
# 
# Since urea is the most imported fertilizer in Nepal. Lets explore more of it.

# In[ ]:


imported_urea = data_nepal.loc[(data_nepal['Item'] == 'Urea') & (data_nepal['Element'] == 'Import Value')]
imported_urea


# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=imported_urea['Year'], y=imported_urea['Value'],
                    mode='lines',
                    name='Urea',marker_color='green'))

fig.update_layout(
    title='Import of urea over the years in Nepal (US$1000)',
        template='plotly_dark'

)

fig.show()


# Yeah, the use of urea increased from 2009 after that it is fluctuate over the time.
# 
# The second most used fertilizer in Nepal is Potassium chloride (muriate of potash) (MOP). Lets compare them.

# In[ ]:


cost_of_imported_urea = imported_urea['Value'].sum()
print('Total amount of money spend in Urea since 2002: Rs.{:.2f}'.format(cost_of_imported_urea*121.27))


# In[ ]:


imported_potassium = data_nepal.loc[(data_nepal['Item'] == 'Potassium chloride (muriate of potash) (MOP)') & (data_nepal['Element'] == 'Import Value')]
imported_potassium


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=imported_urea['Year'], y=imported_urea['Value'],
                    mode='lines',
                    name='Urea'))

fig.add_trace(go.Scatter(x=imported_potassium['Year'], y=imported_potassium['Value'],
                    mode='lines',
                    name='Potassium Chloride',line=dict(dash='dot')))


fig.update_layout(
    title='Comparison between Urea and Potassium Chloride',
    template='plotly_dark',

)

fig.show()


# In[ ]:


fig = go.Figure(data=[go.Bar(
            x=fertilizer_by_item['Item'][0:10], y=fertilizer_by_item['Value'][0:10],
            text=fertilizer_by_item['Value'][0:10],
            textposition='auto',
            marker_color='red',
 
        )])
fig.update_layout(
    title='10 Most Used Fertilizer since 2002 in Nepal',
    xaxis_title="Items",
    yaxis_title="Value",
    template='plotly_dark'
)
fig.show()


# These are the top 10 most used fertilizers in Nepal:
#     
# 1. Urea
# 2. Potassium Chloride
# 3. Phosphate rock
# 4. NPK fertilizes
# 5. Diammonium Phosphate(DAP)
# 6. Ammonia,anhydrous
# 7. Ammonium sulphate
# 8. Ammonium Nitrate
# 9. Urea and Ammonium nitrate solutions
# 10. Monoammonium phosphate
#     

# Evolution of importing and exporting of fertilizer over nepal from 2003 to 2017.

# In[ ]:


plt.figure(figsize=(20,10))
sns.set_style('dark')
sns.countplot(x='Year',data=data_nepal);
plt.title('Import/Export of fertilizer over the years')


# The maximum import export has happenned in 2011 where as the lowest in 2006. The import-export seems to be decreased in the period of 2003 to 2008 where as it increased from 2009 and slightly decreasing from 2011.
# From 2009 it is seem that around 50 tonnfertilizer are imported-exported in every year in Nepal.

# In[ ]:


fig = px.pie(data_nepal,values=data_nepal['Element'].value_counts().values,
             names=df['Element'].unique(),
             title='Import and Export amount of fertilizer in Nepal',
            )
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(
    template='plotly_dark'
)
fig.show()


# From the pie we can clearly say that is believes in importing the fertilizer rather than exporting. This is the reason why we are fuckin poor.
# 
# Lets see what we are producing.

# In[ ]:


nepal_production = data_nepal.loc[data_nepal['Element'] == 'Production']
nepal_production.sort_values(by=['Value'], ascending=False)


# we dont't believe in production rather we prefer to import.

# In[ ]:


fig = px.area(nepal_production, x="Year", y="Value", color="Item", line_group="Item", title='Production of fertilizers in Nepal')
fig.show()


# Look how reach we are hahaha!!!!
# 
# 
