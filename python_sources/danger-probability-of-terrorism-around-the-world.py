#!/usr/bin/env python
# coding: utf-8

# *Research Questions: *
# 
# 1.) Which countries would a civilian most likely die to terrorism?
# 
# 2.) Where is the most dangerous place to to stay in terms of terrorism?
# 
# 3.) Where would a civilian most likely survive a terrorist attack should he/she encounter one?
# 
# 
# 
# **Data Analysis: **

# In[ ]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')
data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','city':'City','attacktype1_txt':'AttackType','nkill':'Killed','nwound':'Wounded','targtype1_txt':'TargetType','success':'Successful'},inplace=True)
data = data[['Year','Month','Day','Country','Region','City','AttackType','Killed','Wounded','TargetType','Successful']]
data['Casualities']=data['Killed']+data['Wounded']
#data.head(10)

civilData = data[(data.TargetType == 'Private Citizens & Property')]

successattacks = data['Successful'].value_counts().tolist()
topcountries = data['Country'].value_counts()[:10].keys().tolist()

print('Total successful terrorist attacks: ',successattacks[0])
print('Total unsuccessful terrorist attacks: ',successattacks[1])
print('Total success rate: ',"{0:.2f}".format((successattacks[0]/(successattacks[0]+successattacks[1]))*100)+'%')

country=data['Country'].value_counts()[:10].to_frame()
country.columns=['Attacks']
successful=data.groupby('Country')['Successful'].sum().to_frame()
country.merge(successful,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.suptitle('Success Rate of Terrorist Attacks in Countries')
plt.show()

country=civilData['Country'].value_counts()[:10].to_frame()
country.columns=['Attacks']
successful=civilData.groupby('Country')['Successful'].sum().to_frame()
country.merge(successful,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.suptitle('Success Rate of Civilian-targeted Terrorist Attacks in Countries')
plt.show()


country=data['Country'].value_counts()[:10].to_frame()
country.columns=['Attacks']
killed=data.groupby('Country')['Killed'].sum().to_frame()
country.merge(killed,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.suptitle('Number of attacks and people killed in Countries')
plt.show()

country=civilData['Country'].value_counts()[:10].to_frame()
country.columns=['Attacks']
killed=civilData.groupby('Country')['Killed'].sum().to_frame()
country.merge(killed,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.suptitle('Number of Civilian-targeted attacks and Civilians killed in Countries')
plt.show()

countryattacks = data['Country'].value_counts().keys().tolist()
countryattacksnum = data['Country'].value_counts().tolist()
civilcountryattacks = civilData['Country'].value_counts().keys().tolist()
civilcountryattacksnum = civilData['Country'].value_counts().tolist()

successcountry = data.groupby('Country')['Successful'].sum()
civilsuccesscountry = civilData.groupby('Country')['Successful'].sum()
killedcountry = data.groupby('Country')['Killed'].sum()
civilkilledcountry = civilData.groupby('Country')['Killed'].sum()

countrykilled = killedcountry.keys().tolist()
countrykillednum = killedcountry.tolist()

print('Success Rate in the Top Ten Countries')
for x in range(0,10):
    print(countryattacks[x],' : ',"{0:.2f}".format(successcountry[countryattacks[x]]/countryattacksnum[x]*100),'%')

print('\nSuccess Rate (Civilian) in the Top Ten Countries')
for x in range(0,10):
    print(civilcountryattacks[x],' : ',"{0:.2f}".format(civilsuccesscountry[civilcountryattacks[x]]/civilcountryattacksnum[x]*100),'%')    
    
print('\nKilled to Attack Ratio in the Top Ten Countries')
for x in range(0,10):
    print(countryattacks[x],' : ',"{0:.2f}".format(killedcountry[countryattacks[x]]/countryattacksnum[x]))
 
print('\nKilled to Attack Ratio (Civilian) in the Top Ten Countries')
for x in range(0,10):
    print(civilcountryattacks[x],' : ',"{0:.2f}".format(civilkilledcountry[civilcountryattacks[x]]/civilcountryattacksnum[x]))


# As we can see above, Iraq is undoubtedly suffering the worst from terrorism. Not only does terrorism in Iraq have a high success rate, the amount of people dying is 3.21 people per attack, which is the highest amongst the top ten countries.
# Afghanistan can be considered second to Iraq, even while having a less amount of attacks than Pakistan, the killed to attack ratio of Afghanistan arguably makes it worse than Pakistan.
# 
# In terms of civilian-targeted attacks, Nigeria and Peru are amongst the worst, beating Iraq in terms of success rate and amount of people dying with each attack on average. Nigeria has an astonishing 8.97 amount of people dying in each civilian-targeted attack while Peru has a 4.75 ratio, suprassing Iraq which has a 3.71 ratio.
# 
# United Kingdom is arguably the best country amongst the top ten in handling terrorism. With a success rate of 80.56% and a killed to attack ratio of 0.66, only Israel (in terms of civilian-targeted attacks) comes close to matching the statistics of the United Kingdom.

# In[ ]:


country=data['Region'].value_counts()[:5].to_frame()
country.columns=['Attacks']
killed=data.groupby('Region')['Successful'].sum().to_frame()
country.merge(killed,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.suptitle('Success Rate of Terrorist Attacks in Regions')
plt.show()

country=data['Region'].value_counts()[:5].to_frame()
country.columns=['Attacks']
killed=data.groupby('Region')['Killed'].sum().to_frame()
country.merge(killed,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.suptitle('Number of attacks vs Number of killed in Regions')
plt.show()

regionattacks = data['Region'].value_counts().keys().tolist()
regionattacksnum = data['Region'].value_counts().tolist()

successregion = data.groupby('Region')['Successful'].sum()
killedregion = data.groupby('Region')['Killed'].sum()

print('Success Rate in the Top Five Regions')
for x in range(0,5):
    print(regionattacks[x],' : ',"{0:.2f}".format(successregion[regionattacks[x]]/regionattacksnum[x]*100),'%')

print('\nKilled to Attack Ratio in the Top Five Regions')
for x in range(0,5):
    print(regionattacks[x],' : ',"{0:.2f}".format(killedregion[regionattacks[x]]/regionattacksnum[x]))


# In the data visualization above, Middle East & North Africa are obviously the most dangerous regions in terms of terrorism. While South Asia has a lot more terrorist attacks than Sub-Saharan Africa, Sub-Saharan Africa has a noticeable 4.60 killed to attack ratio, which makes it potentially a more dangerous region than South Asia.

# In[ ]:


data = [ dict(
        type='choropleth',
        #autocolorscale = False,
        locations = countryattacks,
        z = countryattacksnum,
        text = countryattacks,
        locationmode = 'country names',
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Attacks")
        ) ]

layout = dict(
        title = 'Terrorism Around The World',
        geo = dict(
            scope='world',
            projection=dict( type='Mercator' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )

data = [ dict(
        type='choropleth',
        #autocolorscale = False,
        locations = countrykilled,
        z = countrykillednum,
        text = countrykilled,
        locationmode = 'country names',
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Killed")
        ) ]

layout = dict(
        title = 'People killed by Terrorism Around The World',
        geo = dict(
            scope='world',
            projection=dict( type='Mercator' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )


# **Conclusion:**
# 
# 1.) Which countries would a civilian most likely die to terrorism?
# > While Iraq may have a lot more terrorist attacks, the killed to attack ratio of both Nigeria and Peru are noticeably higher than Iraq, which makes those countries more susceptible to civilian deaths. The amount of people dying in each civilian-targeted terrorist attack in Nigeria is a suprising 8.97 deaths per attack, which is a large difference from other countries.
# > A civilian would most likely encounter a terrorist attack in Iraq rather than Nigeria, but has a better chance of surviving than encountering one in Nigeria.
# 
# 2.) Where is the most dangerous place to to stay in terms of terrorism?
# > In general, Middle East & North Africa are the top contenders in being the most dangerous place to stay in, while Sub-Saharan Africa may potentially be second because of its high killed to attack ratio.
# > Naturally, the most dangerous country to stay in would be in Middle East, which is Iraq. The sheer number of attacks and successful attacks alone make it the most dangerous country to stay in.
# 
# 3.) Where would a civilian most likely survive a terrorist attack should he/she encounter one?
# > This would undoubtedly be United Kingdom, with Israel being a close second. United Kingdom has an impressive 85.89% civilian-targeted success rate; with a 0.71 killed to attacked ratio, which is the only country besides Israel in which the amount of killed is less than the amount of attacks. 
# > When a civilian encounters a terrorist attack in United Kingdom, the chances of him/her surviving is considerably higher, as most attacks in United Kingdom end up with no people dying.
