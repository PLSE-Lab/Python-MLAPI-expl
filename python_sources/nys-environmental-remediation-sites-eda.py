#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# from this kernel: https://www.kaggle.com/jakevdp/altair-kaggle-renderer-test
get_ipython().system('pip install --ignore-installed --no-deps --target=. git+http://github.com/altair-viz/altair')


# In[ ]:


get_ipython().system('pip install mapclassify')


# In[ ]:


get_ipython().system('pip install chart-studio')


# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
import plotly.tools as tls
import chart_studio.plotly as py
import folium
import altair as alt
alt.renderers.enable('kaggle')
import geopandas as gpd
import mapclassify
import mplleaflet as mpll
init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
p = sns.diverging_palette(145, 280, s=85, l=25, n=10)


# In[ ]:


df = pd.read_csv('../input/nys-environmental-remediation-sites/environmental-remediation-sites.csv')
map_df = gpd.read_file('../input/nys-county-boundaries/tl_2016_36_cousub/tl_2016_36_cousub.shp')


# In[ ]:


df.head()


# In[ ]:


df.info()


# ### Process Date

# In[ ]:


df["Project Completion Date"] = pd.to_datetime(df["Project Completion Date"])
df["Completion Year"] = df["Project Completion Date"].dt.year
df.index = pd.DatetimeIndex(df["Project Completion Date"])


# In[ ]:


len(df[df["Completion Year"]>2019])


# > - A lot of projects to complete in the future!

# ## Projects completed by Year

# In[ ]:


plt.figure(figsize=(10,6))
df[df["Completion Year"]<2020].resample('Y').size().plot()


# > There are three spikes:
# > - 1995
# > - 2003
# > - 2012

# ## Most and least common contaminants

# In[ ]:


f, axes = plt.subplots(1,2)
f.set_figheight(8)
f.set_figwidth(15)
plt.subplots_adjust(wspace=.7)
freq = df.Contaminants.value_counts()
common = freq.iloc[:10]
rare = freq.iloc[-10:]
axes[0].set_title("Most common contaminants")
sns.countplot(y="Contaminants", data=df, order=common.index, palette=p, ax=axes[0])
axes[1].set_title("Least common contaminants")
sns.countplot(y="Contaminants", data=df, order=rare.index, palette=p, ax=axes[1])


# ## TCE
# > The chemical compound trichloroethylene is an effective solvent for a variety of organic materials. It is a clear non-flammable liquid with a sweet smell. Because of its widespread use, TCE has become a common environmental contaminant. Contamination results from:
# - discharge to surface waters and groundwater by industry commerce, and individual consumers
# - evaporative losses during use
# - incidental addition of TCE during food production
# - leaching from hazardous waste landfills leaching into groundwater  
# <img src="https://sc02.alicdn.com/kf/HTB1HhLowxuTBuNkHFNRq6A9qpXaI/Liquid-99-3-min-TCE-C2HCl3-CAS.jpg_350x350.jpg">
# ### TCE Health Effects
# > Relatively short-term exposure of animals to trichloroethylene resulted in harmful effects on the nervous system, liver, respiratory system, kidneys, blood, immune system, heart, and body weight.
# Exposure to trichloroethylene in the workplace may cause scleroderma (a systemic autoimmune disease) in some people. Some men occupationally-exposed to trichloroethylene and other chemicals showed decreases in sex drive, sperm quality, and reproductive hormone levels.
# #### Citations:  
# *Centers for Disease Control and Prevention* [Environmental Health and Medicine Education Accessed](https://www.atsdr.cdc.gov/csem/csem.asp?csem=15&po=5) Accessed July 14, 2019  
# *Wikipedia* [Trichloroethylene](https://en.wikipedia.org/wiki/Trichloroethylene). May 23, 2019. Accessed July 14, 2019  
# *Centers for Disease Control and Prevention* [Toxic Substances Portal - Trichloroethylene (TCE)](https://www.atsdr.cdc.gov/phs/phs.asp?id=171&tid=30). January 21, 2015. Accessed July 14, 2019

# In[ ]:


sc_freq = pd.DataFrame(df["Site Class"].value_counts()).reset_index()
sc_freq.columns = ["site_class", "count"]
alt.Chart(sc_freq).mark_bar().encode(
    x='site_class',
    y='count',
    color=alt.condition(
        alt.datum.site_class == "02",
        alt.value('purple'),
        alt.value('green')
    )
).properties(width=400)


# > - 02 - The disposal of hazardous waste represents a significant threat to the environment or to health

# ## Count of Remediation Sites by County (Overall)

# In[ ]:


f, axes = plt.subplots(1,2)
f.set_figheight(8)
f.set_figwidth(15)
plt.subplots_adjust(wspace=.7)
freq = df.County.value_counts()
common = freq.iloc[:10]
rare = freq.iloc[-10:]
axes[0].set_title("Counties with the most sites")
sns.countplot(y="County", data=df, order=common.index, palette=p, ax=axes[0])
axes[1].set_title("Counties with the least sites")
sns.countplot(y="County", data=df, order=rare.index, palette=p, ax=axes[1])


# In[ ]:


county_df = pd.DataFrame(df.County.value_counts()).reset_index()
county_df.columns = ['county', 'count']
county_df = map_df.set_index('NAME').join(county_df.set_index('county'))
vmin, vmax = 120, 220
fig, ax = plt.subplots(1, figsize=(18, 12))
county_df.plot(column='count', cmap='Purples', ax=ax, linewidth=0.8, scheme='userdefined', edgecolor='grey', classification_kwds={'bins':[10,100,500,1000,3000,4000,5000]}, legend=True)


# > - As we can see Nassau has way more remediation sites than other counties, lets take a look at Nassau specifically

# ## Dangerous Contamination Sites in Nassau today

# In[ ]:


df_now = df[df["Completion Year"] > 2018]


# In[ ]:


nas_df = df_now[(df_now.County == "Nassau") & (df_now["Site Class"]=="02")]
f, ax = plt.subplots(1, figsize=(14, 8))
nas_df.plot(kind='scatter', x='Longitude', y='Latitude', s=40, color='purple', ax=ax)
mpll.display(fig=f)


# In[ ]:


nas_df_contaminants = nas_df.Contaminants.value_counts()
common = nas_df_contaminants.iloc[:8]
plt.figure(figsize=(8, 6))
sns.countplot(y="Contaminants", data=nas_df, order=common.index, palette=p)


# ## What happened in 1995

# In[ ]:


f, axes = plt.subplots(2,2)
f.set_figheight(12)
f.set_figwidth(18)
f.suptitle("1995", fontsize=32)
plt.subplots_adjust(wspace=.7)
df_1995 = df[df["Completion Year"]==1995]
df_1995_nas = df_1995[df_1995.County=="Nassau"]
counties = df_1995.County.value_counts()
common = counties.iloc[:8]
axes[0,0].set_title("Counties with most sites")
sns.countplot(y="County", data=df_1995, order=common.index, palette=p, ax=axes[0,0])
contaminants = df_1995_nas.Contaminants.value_counts()
common = contaminants.iloc[:8]
axes[0,1].set_title("Top Contaminants of Nassau")
sns.countplot(y="Contaminants", data=df_1995_nas, order=common.index, palette=p, ax=axes[0,1])
freq = df_1995_nas["Project Name"].value_counts()
axes[1,0].set_title("Project Types in Nassau")
sns.countplot(y="Project Name", data=df_1995_nas, order=freq.index, palette=p, ax=axes[1,0])
freq = df_1995_nas["Site Class"].value_counts()
axes[1,1].set_title("Site Classes in Nassau")
sns.countplot(y="Site Class", data=df_1995_nas, order=freq.index, palette=p, ax=axes[1,1])


# In[ ]:


df_1995["Project Name"].value_counts()


# > - most of the contaminants are not recorded thus we can't make conclusions yet

# ## What happened in 2003

# In[ ]:


f, axes = plt.subplots(2,2)
f.set_figheight(12)
f.set_figwidth(18)
f.suptitle("2003", fontsize=32)
plt.subplots_adjust(wspace=.7)
df_2003 = df[df["Completion Year"]==2003]
df_2003_nas = df_2003[df_2003.County=="Nassau"]
counties = df_2003.County.value_counts()
common = counties.iloc[:8]
axes[0,0].set_title("Counties with most sites")
sns.countplot(y="County", data=df_2003, order=common.index, palette=p, ax=axes[0,0])
contaminants = df_2003_nas.Contaminants.value_counts()
common = contaminants.iloc[:8]
axes[0,1].set_title("Top Contaminants of Nassau")
sns.countplot(y="Contaminants", data=df_2003_nas, order=common.index, palette=p, ax=axes[0,1])
freq = df_2003_nas["Project Name"].value_counts()
axes[1,0].set_title("Project Types in Nassau")
sns.countplot(y="Project Name", data=df_2003_nas, order=freq.index, palette=p, ax=axes[1,0])
freq = df_2003_nas["Site Class"].value_counts()
axes[1,1].set_title("Site Classes in Nassau")
sns.countplot(y="Site Class", data=df_2003_nas, order=freq.index, palette=p, ax=axes[1,1])


# ## The Glen Cove Cleanup
# > - Mounds of excavated dirt laced with radioactive industrial waste and heavy metals have been piled along a bank of Glen Cove Creek.
# > - The government paid [20 million](https://www.nytimes.com/2003/10/12/nyregion/us-to-pay-20-million-in-glen-cove-cleanup.html) for the cleanup
# 
# <img src="https://cdn.newsday.com/polopoly_fs/1.12896495.1483964328!/httpImage/image.jpeg_gen/derivatives/landscape_768/image.jpeg">

# ## What happened in 2012?

# In[ ]:


f, axes = plt.subplots(2,2)
f.set_figheight(12)
f.set_figwidth(18)
f.suptitle("2012", fontsize=32)
plt.subplots_adjust(wspace=.7)
df_2012 = df[df["Completion Year"]==2012]
df_2012_nas = df_2012[df_2012.County=="Nassau"]
counties = df_2012.County.value_counts()
common = counties.iloc[:8]
axes[0,0].set_title("Counties with most sites")
sns.countplot(y="County", data=df_2012, order=common.index, palette=p, ax=axes[0,0])
contaminants = df_2012_nas.Contaminants.value_counts()
common = contaminants.iloc[:8]
axes[0,1].set_title("Top Contaminants of Nassau")
sns.countplot(y="Contaminants", data=df_2012_nas, order=common.index, palette=p, ax=axes[0,1])
freq = df_2012_nas["Project Name"].value_counts()
axes[1,0].set_title("Project Types in Nassau")
sns.countplot(y="Project Name", data=df_2012_nas, order=freq.index, palette=p, ax=axes[1,0])
freq = df_2012_nas["Site Class"].value_counts()
axes[1,1].set_title("Site Classes in Nassau")
sns.countplot(y="Site Class", data=df_2012_nas, order=freq.index, palette=p, ax=axes[1,1])

