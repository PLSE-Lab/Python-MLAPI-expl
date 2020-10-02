#!/usr/bin/env python
# coding: utf-8

# **Terrorism Attacks Worldwide**
# 
# For my first Kernel, I am choosing topic which is close to my heart i.e. geopolitics. 
# In this dataset, we will be exploring the terror attacks over the world from 1970-2016, finding interesting patterns.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
plt.style.use('fivethirtyeight')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as g
import plotly.tools as tls


# **Let's load and clean the data**

# In[ ]:


terror=pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1',low_memory=False)
terror.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
terror=terror[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]
terror['casualities']=terror['Killed']+terror['Wounded']
terror.head(3)


# **Let's do some basic analysis. This code is taken from I, coder's kernel**

# In[ ]:


print('Country with Highest Terrorist Attacks:',terror['Country'].value_counts().index[0])
print('Regions with Highest Terrorist Attacks:',terror['Region'].value_counts().index[0])
print('Month with Highest Terrorist Attacks:',terror['Month'].value_counts().index[0])
print('Most common attack type:',terror['AttackType'].value_counts().index[0])
print('Maximum people killed in an attack are:',terror['Killed'].max(),'that took place in',terror.loc[terror['Killed'].idxmax()].Country)


# ** Analysis of terror activities through the years **
# 
# 
# While terrorism has been around since long, it has grown exponentially in recent years

# In[ ]:


plt.subplots(figsize=(15,6))
sns.countplot('Year',data=terror,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number of Terror Activities')
plt.show()


# Now let's find out, most affected countries 

# In[ ]:


plt.subplots(figsize=(18,6))
sns.barplot(terror['Country'].value_counts()[:10].index,terror['Country'].value_counts()[:10].values,palette='inferno')
plt.title('Top Affected Countries - Number of Attacks')
plt.show()


# Again, we get expected result. Middle east and South Asian countries features in the top 4. Surprising to see India in the list. 
# 
# Now let's see, what is ratio between attack vs killed. This will help us understand, which countries have been most severely impacted by terrorism. 

# In[ ]:


coun_terror=terror['Country'].value_counts()[:15].to_frame()
coun_terror.columns=['Attacks']
coun_kill=terror.groupby('Country')['Killed'].sum().to_frame()
coun_terror.merge(coun_kill,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.show()


# In[ ]:


coun_terrorregion=terror['Region'].value_counts()[:15].to_frame()
coun_terrorregion.columns=['Attacks']
coun_killregion=terror.groupby('Region')['Killed'].sum().to_frame()
coun_terrorregion.merge(coun_killregion,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.show()


# Visulatisations above provide insights but do not provide any groundbbreaking information. 
# 
# As I mentioned earlier, while Middle East and Northen Africa seem most affected by terrorism, has this always been the case?
# 
# Let us find out. For this, I am breaking data into two subsets. Terror in Year 2000 or before and Terror from 2001 to 2016. 
# 
# Reason for this break up?
# 
# **9/11 and War on Terror**
# 
# While terrorism was a menance, it only captured world's attention after 9/11 and subsequent "War on Terror". Has war on terror heled in establishing world peace or has it made world riskier? let's find out

# In[ ]:


terror_inbefore2000 = terror[terror.Year <= 2000]
terror_after2000 = terror[terror.Year > 2000]


# Now it would be great to compare top 10 countries, top regions before 2001 and after 2001. 

# In[ ]:


coun_terror1=terror_inbefore2000['Country'].value_counts()[:15].to_frame()
coun_terror1.columns=['Attacks']
coun_kill1=terror_inbefore2000.groupby('Country')['Killed'].sum().to_frame()
coun_terror1.merge(coun_kill1,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.ylim(0, 80000)
fig.suptitle('Terrorism Activities (1970-2000)', fontsize=16)
plt.tight_layout()
plt.show()

coun_terror2=terror_after2000['Country'].value_counts()[:15].to_frame()
coun_terror2.columns=['Attacks']
coun_kill2=terror_after2000.groupby('Country')['Killed'].sum().to_frame()
coun_terror2.merge(coun_kill2,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.ylim(0, 80000)
fig.suptitle('Terrorism Activities (2001-2016)', fontsize=16)
plt.tight_layout()
plt.show()


# In[ ]:


coun_terrorregion1=terror_inbefore2000['Region'].value_counts()[:15].to_frame()
coun_terrorregion1.columns=['Attacks']
coun_killregion1=terror_inbefore2000.groupby('Region')['Killed'].sum().to_frame()
coun_terrorregion1.merge(coun_killregion1,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.ylim(0, 80000)
fig.suptitle('Terrorism Activities by Region (1970-2000)', fontsize=16)
plt.tight_layout()
plt.show()

coun_terrorregion2=terror_after2000['Region'].value_counts()[:15].to_frame()
coun_terrorregion2.columns=['Attacks']
coun_killregion2=terror_after2000.groupby('Region')['Killed'].sum().to_frame()
coun_terrorregion2.merge(coun_killregion2,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.ylim(0, 80000)
fig.suptitle('Terrorism Activities by Region (2001-2016)', fontsize=16)
plt.tight_layout()
plt.show()


# This clearly shows, after the start of **"War on Terror"**, terrorism has just increased and has become much more deadly confirming my fears. In fact, due to this, terror epicenter has shifted to Middle East, South Asia and Northen Africa from South America. 

# In[ ]:


plt.subplots(figsize=(18,6))
sns.barplot(terror['Group'].value_counts()[1:15].index,terror['Group'].value_counts()[1:15].values,palette='inferno')
plt.xticks(rotation=90)
plt.title('Top Terror Group - By Number of Attacks')

plt.show()


# Along with the usual suspects like Taliban, some lesser known group feature. Until today, I had never heard of "Shining Path" yet they are one of the most active terros group of last 50 years. Now let's see which are the most dealiest groups 

# In[ ]:


coun_terrorgroup=terror['Group'].value_counts()[1:15].to_frame()
coun_terrorgroup.columns=['Attacks']
coun_killgroup=terror.groupby('Group')['Killed'].sum().to_frame()
coun_terrorgroup.merge(coun_killgroup,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(12,12)
fig.suptitle('Terrorism Activities by Group', fontsize=16)
plt.tight_layout()
plt.show()


# As graph above show, Islamic groups are extremly deadly. Only LTTE comes close in attack vs death ration. Now let us see, which are the most common attack types and how deadly they are

# In[ ]:


coun_terrorattack=terror['AttackType'].value_counts()[:15].to_frame()
coun_terrorattack.columns=['Attacks']
coun_killattack=terror.groupby('AttackType')['Killed'].sum().to_frame()
coun_terrorattack.merge(coun_killattack,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(12,12)
fig.suptitle('Terrorism Activities by Attack Type', fontsize=16)
plt.tight_layout()
plt.show()


# Bombing is most common however armed assault seems to be most dangeorus. 
# 
# **End of Analysis**
# 
# **Summary**
# 
# I have been reading about negative impact of "War on terror" on world peace for years. However I became aware of its magnitude only recently after looking at this. 
# 
# As graphs above show, epicenter of terrorism has made definite shift to Middle East and South Asia. Top 15 countries by number of terror attacks for the period of year 1970-2000 feature South American countries where as for the period of 2001-2016, it is dominated by Middle Eastern, Sub Saharan and South Asian countries. In addition, number of terror activities for 2001-2016 period far exceed 1970-2000 period. In Iraq alone, more people were killed by terrorism in this period than top 5 countries of 1970-2000 combined. 
# 
# India features in both the periods which is sad but not surprising. 
# 
# There are many other insights which can be drawn from this data set but this shocked me most.
# 
# Off-course, there are other geopolitical factors influencing this but still this serves as shocking reminder of how everything we do has long term unintended consequences.
# 
# **This is my first Kernal and hence I chose a topic which I am passionate about i.e. geopolitics. Your suggestions and comments are welcome.**
# 
