#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt


# # Task : Which populations assessed should stay home and which should see an HCP

# In this notebook, we use Google Mobility Report and a database on health indicator by country to compare the mobility (wether people stayed at home) with the global health and covid-19 report for these countries. In particular, we use Google Mobility Report "Residential" percentage which shows the increase (in proportion) of people staying at home compared to the baseline of when there was no corona virus. The version of Google Mobility Report we use is [this one](https://www.kaggle.com/chaibapat/google-mobility) constructed the 3/04/20.

# In[ ]:


# We load the two datasets 
df_mobility = pd.read_csv('/kaggle/input/google-mobility/mobility_google.csv')
df_health = pd.read_csv('/kaggle/input/country-health-indicators/country_health_indicators_v3.csv')
df_covid = pd.read_csv('/kaggle/input/uncover/UNCOVER/worldometer/worldometer/worldometer-confirmed-cases-and-deaths-by-country-territory-or-conveyance.csv')
df_mobility= df_mobility.set_index('Country').sort_index()
df_health = df_health.set_index('Country_Region').sort_index()
df_covid = df_covid.set_index('country').sort_index()


# In[ ]:


# Select the features and merge the two datasets and 
df=pd.DataFrame()
df['Residential']=df_mobility['Residential']
df['total_deaths_per_1m_pop']=df_covid['total_deaths_per_1m_pop'] 

# change the features for proportions
health_features=['Cancers (%)','Diabetes, blood, & endocrine diseases (%)','Respiratory diseases (%)','Liver disease (%)',
                 'Diarrhea & common infectious diseases (%)','HIV/AIDS and tuberculosis (%)','Nutritional deficiencies (%)',
                'Share of deaths from smoking (%)','alcoholic_beverages']
df[health_features]=df_health[health_features]

# Parse the data from mobility database
def to_percent(x):
    try:
        return int(re.findall(r'\b\d+\b',x)[0])/100
    except :
        return np.nan
df['Residential']=df['Residential'].apply(to_percent)
df = df.dropna()


# In[ ]:


# Print the first few rows
df.head()


# ## First question : did people stay at home a lot more because of covid ?

# In[ ]:


sns.distplot(df['Residential'])
plt.title('Median proportion of increase in Residential : '+str(np.median(df['Residential'])))
plt.show()


# For most countries, the proportion of time in Residential area increased by 15%.

# ## Second question : is there a correlation between change in Residential feature and 

# ### Study on the whole dataset

# In[ ]:


# Visualization of the correlations
df.corr().style.background_gradient()


# The is indeed some correlation between Residential and the ratio deaths/recovered, the value is 0.14. We can also plot the joint density of ratio deaths/recovered and Residential (we use a logscale for total_deaths_per_1m_pop).

# In[ ]:


g = sns.jointplot(df["Residential"], np.log(df["total_deaths_per_1m_pop"]), kind="kde").set_axis_labels("Residential", "log(total_deaths_per_1m_pop)")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")


# There seems to be two mode of high density, both for Residential relatively low. 
# One mode give a low log(total_deaths_per_1m_pop) and a low Residential. Most probably this is for countris where covid is not yet virulent. The second mode (the one we are interested in) is for low Residential and low log(total_deaths_per_1m_pop).
# 
# Next, we study the same dataset but we select only the countries with a log(total_deaths_per_1m_pop)>=0 (more than 1 deaths per 1m pop). This can be interpreted by saying that we only study countries where corona-virus is virulent.

# ### Study only on countries with high death rate.

# In[ ]:


df = df[df['total_deaths_per_1m_pop']>=1]


# In[ ]:


# Visualization of the correlations
df.corr().style.background_gradient()


# we get a higher correlation between Residential and the proportion of deaths, and also a much higher correlation between Residential and the different conditions (diabetes, common infectious deseases, nutritional deficiencies).

# In[ ]:


g = sns.jointplot(df["Residential"], np.log(df["total_deaths_per_1m_pop"]), kind="kde").set_axis_labels("Residential", "log(total_deaths_per_1m_pop)")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")


# ## beginning of an answer for the Task question
# Finally, we plot the couple (death ratio, Residential) for all the different health conditions.

# In[ ]:


f, axes = plt.subplots(3, 3, figsize=(18, 11), sharex=True, sharey=True)
cmap = sns.cubehelix_palette(dark=.2, light=.7, as_cmap=True)

for f in range(9):
    sns.scatterplot(df["Residential"], np.log(df["total_deaths_per_1m_pop"]),size=df.columns[f+2],
                    hue=df.columns[f+2], data=df,ax=axes.ravel()[f],sizes=(50, 200),palette=cmap, legend=False)
    axes.ravel()[f].set_title('plot for '+df.columns[f+2])
plt.tight_layout()


# ## Conclusion
# 
# Primary findings seems to indicate that Cancer patient, People with repiratory desease and smokers should most certainly be in hospitals if the have corona virus
# 
# The results are less conclusif but it seems that diabetes, liver desease and alcohol beverage are also factors. The other plots are not conclusive.
# 

# This is the first draft of this notebook, it may be interesting to study different illnesses in conjunction with google mobility dataset to have further results.

# In[ ]:




