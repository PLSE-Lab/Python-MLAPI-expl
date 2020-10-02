#!/usr/bin/env python
# coding: utf-8

# ## ** Grade 4 Geography!!!**
# 
# Soooooo, the creater/owner**** of the this dataset had this ask - Can you predict the crop production in India, which is vital for so many things? 
# 
# Umm, Can I? 
# 
# ![YupUrl](https://media.giphy.com/media/l0MYwvNiMaWJhQZjy/giphy.gif "Yup")
# 
# Am I going to do that in this notebook?
# 
# ![NoUrl](https://media.giphy.com/media/zdq4DT1gHlxny/giphy.gif "No")
# 
# What I will be doing is exploring this dataset because it some features remind me of the Geography lessons that Mrs D'Souza taught me in school! And also because Greta Thunberg has a point!
# 
# ![GretaUrl](https://media.giphy.com/media/KH2sN0iICPCdimCwhk/giphy.gif "Greta")
# 
# 
# ### Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
sns.set_context("paper")


# ### Reading Data

# In[ ]:


crop_df = pd.read_csv("../input/crop-production-in-india/crop_production.csv")
crop_df.head()


# In[ ]:


crop_df.info()


# So the dataset has crop production details state-wise and then goes a level deep with district details. It has data from 1997-2014. Some other intersting features that are available to us are the crop season and area of the field. 
# 
# My approach here is going with the flow. I'm also going to back it up with some research information (along with links) to see if the data and analysis meet.
# 
# ##### Note: This data is only about India. And so the scope of research is also going to be limited to just India.
# 
# Let's begin by looking at the state-wise production
# 
# ### State-Wise Production - 1997-2014

# In[ ]:


fig, ax = plt.subplots(figsize=(25,65), sharex='col')
count = 1

for state in crop_df.State_Name.unique():
    plt.subplot(len(crop_df.State_Name.unique()),1,count)
    sns.lineplot(crop_df[crop_df.State_Name==state]['Crop_Year'],crop_df[crop_df.State_Name==state]['Production'], ci=None)
    plt.subplots_adjust(hspace=2.2)
    plt.title(state)
    count+=1


# Though informative and granular, to see patterns, we'll need to aggregate data. 
# 
# Let's get the zonal details to see if we can see any pattern. Using https://www.mapsofindia.com/zonal/ the states are divided into Zones. 
# 
# ### Zone-Wise Production - 1997-2014

# In[ ]:


north_india = ['Jammu and Kashmir', 'Punjab', 'Himachal Pradesh', 'Haryana', 'Uttarakhand', 'Uttar Pradesh', 'Chandigarh']
east_india = ['Bihar', 'Odisha', 'Jharkhand', 'West Bengal']
south_india = ['Andhra Pradesh', 'Karnataka', 'Kerala' ,'Tamil Nadu', 'Telangana']
west_india = ['Rajasthan' , 'Gujarat', 'Goa','Maharashtra','Goa']
central_india = ['Madhya Pradesh', 'Chhattisgarh']
north_east_india = ['Assam', 'Sikkim', 'Nagaland', 'Meghalaya', 'Manipur', 'Mizoram', 'Tripura', 'Arunachal Pradesh']
ut_india = ['Andaman and Nicobar Islands', 'Dadra and Nagar Haveli', 'Puducherry']


# In[ ]:


def get_zonal_names(row):
    if row['State_Name'].strip() in north_india:
        val = 'North Zone'
    elif row['State_Name'].strip()  in south_india:
        val = 'South Zone'
    elif row['State_Name'].strip()  in east_india:
        val = 'East Zone'
    elif row['State_Name'].strip()  in west_india:
        val = 'West Zone'
    elif row['State_Name'].strip()  in central_india:
        val = 'Central Zone'
    elif row['State_Name'].strip()  in north_east_india:
        val = 'NE Zone'
    elif row['State_Name'].strip()  in ut_india:
        val = 'Union Terr'
    else:
        val = 'No Value'
    return val

crop_df['Zones'] = crop_df.apply(get_zonal_names, axis=1)
crop_df['Zones'].unique()


# In[ ]:


fig, ax = plt.subplots(figsize=(25,30), sharex='col')
count = 1

for zone in crop_df.Zones.unique():
    plt.subplot(len(crop_df.Zones.unique()),1,count)
    sns.lineplot(crop_df[crop_df.Zones==zone]['Crop_Year'],crop_df[crop_df.Zones==zone]['Production'], ci=None)
    plt.subplots_adjust(hspace=0.6)
    plt.title(zone)
    count+=1


# We see 2005 to be of some significance as there's activity happening before and after 2005. 
# Similarly we see slight changes around 2011 for Central,West and North Zone which differ for East and North East Zone. 
# 
# Let's see what happened in 2005 and 2011. Excerpts from the reports published by the IMD for those respective years. 
# 
# #### 2005
# During the season, rainfall was not well distributed in time. Rainfall over the country was below normal (12% below LPA) in June. However, monsoon was active in July with excess rainfall (14% above LPA). Monsoon was subdued in August with a large deficiency of 28% of LPA. In September (rainfall +17% above LPA), monsoon became active again helping a timely revival and improving the seasonal rainfall situation over the country.
# 
# #### 2011
# Out of the total 36 meteorological subdivisions, 33 subdivisions constituting 92% of the total area of the country received excess/normal season rainfall and the remaining 3 subdivisions (Arunachal Pradesh, Assam & Meghalaya, and NMMT constituting 8% of the total area of the country) received deficient season rainfall.
# 
# Ref:
# 
# https://reliefweb.int/report/india/india-meteorological-department-southwest-monsoon-2005-end-season-report
# 
# https://reliefweb.int/report/india/southwest-monsoon-2011-end-season-report
# 
# ### Zone-Wise Production - Total

# In[ ]:


zone_df = crop_df.groupby(by='Zones')['Production'].sum().reset_index().sort_values(by='Production', ascending=False)
zone_df.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(zone_df.Zones, zone_df.Production)
plt.yscale('log')
plt.title('Zone-Wise Production: Total')


# Clearly the South Zone leads in terms of overall production. Lets delve into the southern zone and get more details.
# 
# ### South Zone Production

# In[ ]:


south_zone =  crop_df[(crop_df["Zones"] == 'South Zone')]
fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(south_zone.State_Name, south_zone.Production,errwidth=0)
plt.yscale('log')
plt.title('Southern-Zone Production')

south_zone.groupby(by='State_Name')['Production'].sum().reset_index().sort_values(by='Production', ascending=False)


# Kerala leads the production in southern zone by a huge margin. This is interesting! Let's see what makes Kerala have such a huge margin when it comes to crop production. What crop is it? No brownie points for guessing! 
# 
# ### South Zone - Highest Produced Crops

# In[ ]:


df = south_zone.groupby(by='Crop')['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(df.Crop, df.Production,errwidth=0)
plt.yscale('log')
plt.title('South Zone Crops vs Production')


# ##### **Coconut** 
# 
# So, now what all do I know? That South leads total production. And Kerala dominates in South Zone with its major contribution being Coconuts. Roughly looking at this, I would like to think that Coconut dominates all the crops in terms of production but let's just be sure about this.
# 
# ### Overall Crop Production

# In[ ]:


crop = crop_df.groupby(by='Crop')['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head(10)
crop 
fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(crop.Crop, crop.Production,errwidth=0)
plt.yscale('log')
plt.title('Overall Crops vs Production')


# I wasn't wrong about eh?.
# 
# The numbers spoke for crops. Also, Coconut can be harvested year long depending on its use. More here http://vikaspedia.in/agriculture/crop-production/package-of-practices/plantation-crops/coconut/coconut-cultivation-practices
# 
# But look at the other values. I see something fishy. If I remember correctly, Sugarcane, Rice, Maize are Kharif crops. Hmm... 
# Let's check this...

# In[ ]:


set(crop_df[(crop_df['Season'] == 'Whole Year ')].Crop.unique()) & set(crop_df[(crop_df['Season'] == 'Kharif     ')].Crop.unique()) 


# So, we have some fishy data here. I am going to clean this so that I can classify some of the most important crops into the either Kharif or Rabi, and remaining clubbed as 'Others'
# 
# Ref: http://www.arthapedia.in/index.php%3Ftitle%3DCropping_seasons_of_India-_Kharif_%2526_Rabi
# 

# In[ ]:


Kharif = ['Bajra','Jowar','Maize','Millet','Rice','Soybean','Fruits','Muskmelon','Sugarcane','Watermelon','Orange','Arhar/Tur,'
'Urad','Cotton(lint)','Cowpea(Lobia)','Moong(Green Gram)','Guar seed','Moth','Tomato','Turmeric', 'Ragi']
Rabi = ['Barley', 'Gram', 'Rapeseed &Mustard', 'Masoor', 'Coriander', 'Sunflower', 'Tobacco', 'Brinjal', 'Cabbage',
       'Onion','Sweet potato','Potato','Peas & beans (Pulses)', 'Oilseeds total', 'other oilseeds', 'Banana', 'Groundnut', 'Niger seed',
       'Sesamum','Safflower', 'Castor seed', 'Linseed', 'Soyabean']

def change_crop_seasons(row):
    if row['Crop'].strip() in Kharif:
        val = 'Kharif'
    elif row['Crop'].strip()  in Rabi:
        val = 'Rabi'
    else:
        val = 'Others'
    return val

crop_df['Updated_Crop_Season'] = crop_df.apply(change_crop_seasons, axis=1)
crop_df['Updated_Crop_Season'].unique()


# Using the new column, let's explore the some more..
# 
# 
# ### Season vs Production
# 
# Note: Class Imbalance Warning!

# In[ ]:


season = crop_df.groupby(by='Updated_Crop_Season')['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head(10)
season
fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(season.Updated_Crop_Season, season.Production,errwidth=0)
plt.yscale('log')
plt.title('Seasonal Crops vs Production')


# Since The Indian cropping season is classified into two main seasons-(i) Kharif and (ii) Rabi based on the monsoon, Let's look at them..
# 
# ### (i) Kharif
# 
# Let's look at the top ten most produced Kharif crops.

# In[ ]:


kharif_df = crop_df[(crop_df['Updated_Crop_Season'] == 'Kharif')]
df = kharif_df.groupby(by='Crop')['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head(10)
fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(df.Crop, df.Production,errwidth=0)
plt.yscale('log')
plt.xticks(rotation=40)
plt.title('Kharif Crops Production')


# Sugarcane, Rice and Cotton lead in terms of Kharif crop production. Let's look at the zonal distribution for Sugarcane

# In[ ]:


sugarcane_df = kharif_df[(kharif_df['Crop'] == 'Sugarcane')]
sugarcane_df.head()

fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(sugarcane_df.Zones, sugarcane_df.Production,errwidth=0)
plt.yscale('log')
plt.xticks(rotation=45)
plt.title('Sugarcane Zone-Wise Production')


# This completely makes sense because this finding is confirmed with the link - https://farmer.gov.in/cropstaticssugarcane.aspx.
# Here's another link talking about Sugarcane cultivation. 
# https://www.mapsofindia.com/answers/india/state-biggest-sugarcane-producer/
# Let's see if our data reflects the same.

# In[ ]:


df = sugarcane_df.groupby(by='State_Name')['Production'].sum().reset_index().sort_values(by='Production', ascending=False)
fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(df.State_Name.head(4), df.Production.head(4),errwidth=0)
plt.yscale('log')
plt.title('Sugarcane State-Wise Production')


# Very cool! Let's go a step deeper now. Since Uttar Pradesh leads in this, lets look at the agricultural area distribution. 

# In[ ]:


uttarpr_df = sugarcane_df[(sugarcane_df['State_Name'] == 'Uttar Pradesh')]
df = uttarpr_df.groupby(by=['District_Name', 'Crop'])['Area'].sum().reset_index().sort_values(by='Area', ascending=False)
fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(df.District_Name.head(5), df.Area.head(5),errwidth=0)
plt.title('Uttar Pradesh - Sugarcane Production')
df.head(5)


# The top 3 districts in Uttar Pradesh -  Kheri, Muzzaffnagar and Bijnor districts combined itself have a cultivation area of 11,640,115 units (1.1 crore units) only for Sugarcane. 
# 
# Let's repeat this for Rabi now. 
# 
# ### (ii) Rabi

# In[ ]:


rabi_df = crop_df[(crop_df['Updated_Crop_Season'] == 'Rabi')]
df = rabi_df.groupby(by='Crop')['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head(10)
fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(df.Crop, df.Production,errwidth=0)
plt.yscale('log')
plt.xticks(rotation=45)
plt.title('Rabi Crops Production')


# Potato, Banana and Soyabean lead in terms of Rabi crop production. Let's look at the zonal distribution.

# In[ ]:


potato_df = rabi_df[(rabi_df['Crop'] == 'Potato')]
potato_df.head()

fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(potato_df.Zones, potato_df.Production,errwidth=0)
plt.yscale('log')
plt.xticks(rotation=45)
plt.title('Potato Zone-Wise Production')


# East leads in Potato production followed closely by North. Let's take a look at the state-wise distribution and see if we have data that confirms the stats provided in this link - https://www.mapsofindia.com/top-ten/india-crops/potato.html

# In[ ]:


df = potato_df.groupby(by='State_Name')['Production'].sum().reset_index().sort_values(by='Production', ascending=False)
fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(df.State_Name.head(4), df.Production.head(4),errwidth=0)
plt.yscale('log')
plt.title('Potato State-Wise Production')


# And Indeed we do. Lets look at the agricultural area distribution within Uttar Pradesh

# In[ ]:


uttarpr_df = potato_df[(potato_df['State_Name'] == 'Uttar Pradesh')]
df = uttarpr_df.groupby(by=['District_Name', 'Crop'])['Area'].sum().reset_index().sort_values(by='Area', ascending=False)
fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(df.District_Name.head(5), df.Area.head(5), errwidth=0)
plt.title('Uttar Pradesh - Potato Production')
df.head(5)


# Agra, Kannauj and Firozabad have a combined area of 2,043,041 units for Potato. 
# 
# Now that we've explored the Seasons and Crops based on their production, the last thing I want to look at is agricultural area distribution all over India. From the above analysis we know, Uttar Pradesh is surely is going to top the list. Let's just check that out. 
# 
# ### Agricultural Area

# In[ ]:


df = crop_df.groupby(by='State_Name')['Area'].sum().reset_index().sort_values(by='Area', ascending=False)
df.head()

fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(df.State_Name.head(5), df.Area.head(5), errwidth=0)
plt.title('Agricultural Area Distribution - India')
df.head(5)


# And we proved it!
# 
# Let's look at the production of these five states over the years.

# In[ ]:


df = crop_df.groupby(by='State_Name')['Area'].sum().reset_index().sort_values(by='Area', ascending=False)
df = df.head(5)

fig, ax = plt.subplots(figsize=(25,30), sharey='col')
count = 1

for state in df.State_Name.unique():
    plt.subplot(len(df.State_Name.unique()),1,count)
    sns.lineplot(crop_df[crop_df.State_Name==state]['Crop_Year'],crop_df[crop_df.State_Name==state]['Production'], ci=None)
    plt.subplots_adjust(hspace=0.6)
    plt.title(state)
    count+=1


# And we see how the top 5 states w.r.t Production faltered around 2005. The biggest hit was felt by Maharashtra because of the 2005 Maharashtra floods.
# Ref - https://en.wikipedia.org/wiki/Maharashtra_floods_of_2005. 
#     
# So we can now safely conclude that climate is one of the main determinants of agricultural production. 
# 
# **Climate change** is any change in climate over time that is attributed directly or indirectly to human activity that alters the composition of global atmosphere in addition to natural climate variability observed over comparable time periods 
# 
# And since climatic factors serve as direct inputs to agriculture, any change in climatic factors is bound to have a significant impact on crop yields and production. 
# Ref - http://www.hpccc.gov.in/documents/Impact%20of%20Rainfall%20on%20Agriculture%20in%20H.P.pdf
# 
# **WE HAVE TO DO MORE BECAUSE...** 
# 
# ![GretaUrl](https://media.giphy.com/media/cNMUT2a23oobq3z8gr/giphy.gif "Greta")
# 
