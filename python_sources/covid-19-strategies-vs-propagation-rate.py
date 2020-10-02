#!/usr/bin/env python
# coding: utf-8

# # **Covid-19: Local strategies vs countries infection propagation rate **
# 
# As COVID-19 impact global economies despite the international boundaries, there is a key question that all goverments want to answer: How to flatten the curve?. So, this notebook tries to find if there is a co-relation between the local strategies implemented and how the infection propagate in the country. Also, it will be considered the fatalities caused by the complications of the infection as a measure that will help tp shed some light on it as well. It will be updated regulary.
# 
# The data is obtained from https://ourworldindata.org/grapher/covid-19-total-confirmed-cases-vs-total-confirmed-deaths. The original dataset has the total infections and the total deaths informed by date per country. Some articulation was done in order to get the daily cases and fatalities per country. The data was leveled to the first infection reported in the country, so rows with 0 cases were deleted. Finally, a counter attibute was added per country with the intention to be able to compare the slope of the cases & fatalities.
# 
# I also added a dataset which contains the measures taken by country, which however it is not compelte it is enough to get the idea of measures by country to find a relation with the infection control. https://hai.stanford.edu/news/treating-covid-19-how-researchers-are-using-ai-scale-care-find-cures-and-crowdsource-solutions?utm_source=Stanford+University&utm_campaign=e29354b7d0-EMAIL_CAMPAIGN_2020_04_17_08_12&utm_medium=email&utm_term=0_aaf04f4a4b-e29354b7d0-199833727
# 
# HBR has recently published an article which re-frame the scene and focus on tech adoption and the local strategies as key concept to fight against the pandemia: https://hbr.org/2020/04/how-digital-contact-tracing-slowed-covid-19-in-east-asia?utm_medium=email&utm_source=newsletter_weekly&utm_campaign=insider_activesubs&utm_content=signinnudge&referral=03551&deliveryName=DM77018#comment-section

# In[ ]:


#import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt #plotting, math, stats
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #plotting, regressions
import datetime
import os
import functools
from scipy import stats


# In[ ]:


#Import Data

#Confirmed Cases per country
df = pd.read_csv('/kaggle/input/cases-and-deaths-per-country/cases.csv', sep=';')
df = df.fillna(value=0)

measuresdf = pd.read_csv ('/kaggle/input/covid19-national-responses-dataset/COVID 19 Containment measures data.csv')
countermeasuresdf = pd.read_csv ('/kaggle/input/covid19-national-responses-dataset/countermeasures_db_johnshopkins_2020_03_30.csv')


# In[ ]:


#Organization of the entities to be compared
AR=df.loc[df['Country']== 'Argentina']
US=df.loc[df['Code']== 'USA']
BR=df.loc[df['Country']== 'Brazil']
IT=df.loc[df['Country']== 'Italy']
SP=df.loc[df['Country']== 'Spain']
CL=df.loc[df['Country']== 'Chile']
BO=df.loc[df['Country']== 'Bolivia']
PY=df.loc[df['Country']== 'Paraguay']
CH=df.loc[df['Country']== 'China']
CO=df.loc[df['Country']== 'Colombia']
CR=df.loc[df['Code']== 'CRI']
EC=df.loc[df['Country']== 'Ecuador']
ES=df.loc[df['Country']== 'Estonia']
FR=df.loc[df['Country']== 'France']
GU=df.loc[df['Country']== 'Guatemala']
MX=df.loc[df['Country']== 'Mexico']
PE=df.loc[df['Country']== 'Peru']
UK=df.loc[df['Country']== 'United Kingdom']
UY=df.loc[df['Country']== 'Uruguay']
SK=df.loc[df['Country']== 'South Korea']
World=df.loc[df['Country']== 'World']


# In[ ]:


data = [['South Korea', SK['Day'].max(),measuresdf.loc[measuresdf['Country']== 'South Korea'].Country.count(),SK['Cases'].max(), SK['Deaths'].max(),SK['Deaths'].max()/SK['Cases'].max()*100,SK['Daily_Cases'].mean(),SK['Daily_Cases'].median(),SK['Daily_Cases'].std(),SK['Daily_Fatalities'].mean(),SK['Daily_Fatalities'].median(),SK['Daily_Fatalities'].std()],
        ['Argentina', AR['Day'].max(),measuresdf.loc[measuresdf['Country']== 'Argentina'].Country.count(),AR['Cases'].max(), AR['Deaths'].max(),AR['Deaths'].max()/AR['Cases'].max()*100,AR['Daily_Cases'].mean(),AR['Daily_Cases'].median(),AR['Daily_Cases'].std(),AR['Daily_Fatalities'].mean(),AR['Daily_Fatalities'].median(),AR['Daily_Fatalities'].std()],
        ['United Kingdom', UK['Day'].max(),measuresdf.loc[measuresdf['Country']== 'United Kingdom'].Country.count(),UK['Cases'].max(), UK['Deaths'].max(),UK['Deaths'].max()/UK['Cases'].max()*100,UK['Daily_Cases'].mean(),UK['Daily_Cases'].median(),UK['Daily_Cases'].std(),UK['Daily_Fatalities'].mean(),UK['Daily_Fatalities'].median(),UK['Daily_Fatalities'].std()],
        ['Chile', CL['Day'].max(),measuresdf.loc[measuresdf['Country']== 'Chile'].Country.count(),CL['Cases'].max(), CL['Deaths'].max(),CL['Deaths'].max()/CL['Cases'].max()*100,CL['Daily_Cases'].mean(),CL['Daily_Cases'].median(),CL['Daily_Cases'].std(),CL['Daily_Fatalities'].mean(),CL['Daily_Fatalities'].median(),CL['Daily_Fatalities'].std()],
        ['Italy', IT['Day'].max(),measuresdf.loc[measuresdf['Country']== 'Italy'].Country.count(),IT['Cases'].max(), IT['Deaths'].max(),IT['Deaths'].max()/IT['Cases'].max()*100,IT['Daily_Cases'].mean(),IT['Daily_Cases'].median(),IT['Daily_Cases'].std(),IT['Daily_Fatalities'].mean(),IT['Daily_Fatalities'].median(),IT['Daily_Fatalities'].std()],
        ['Spain', SP['Day'].max(),measuresdf.loc[measuresdf['Country']== 'Spain'].Country.count(),SP['Cases'].max(), SP['Deaths'].max(),SP['Deaths'].max()/SP['Cases'].max()*100,SP['Daily_Cases'].mean(),SP['Daily_Cases'].median(),SP['Daily_Cases'].std(),SP['Daily_Fatalities'].mean(),SP['Daily_Fatalities'].median(),SP['Daily_Fatalities'].std()],
        ['China',CH['Day'].max(),measuresdf.loc[measuresdf['Country']== 'China'].Country.count(),CH['Cases'].max(), CH['Deaths'].max(),CH['Deaths'].max()/CH['Cases'].max()*100,CH['Daily_Cases'].mean(),CH['Daily_Cases'].median(),CH['Daily_Cases'].std(),CH['Daily_Fatalities'].mean(),CH['Daily_Fatalities'].median(),CH['Daily_Fatalities'].std()],
        ['Brazil', BR['Day'].max(),measuresdf.loc[measuresdf['Country']== 'Brazil'].Country.count(),BR['Cases'].max(), BR['Deaths'].max(),BR['Deaths'].max()/BR['Cases'].max()*100,BR['Daily_Cases'].mean(),BR['Daily_Cases'].median(),BR['Daily_Cases'].std(),BR['Daily_Fatalities'].mean(),BR['Daily_Fatalities'].median(),BR['Daily_Fatalities'].std()],
        ['World', World['Day'].max(),measuresdf.loc[measuresdf['Country']== 'World'].Country.count(),World['Cases'].max(), World['Deaths'].max(),World['Deaths'].max()/World['Cases'].max()*100,World['Daily_Cases'].mean(),World['Daily_Cases'].median(),World['Daily_Cases'].std(),World['Daily_Fatalities'].mean(),World['Daily_Fatalities'].median(),World['Daily_Fatalities'].std()],
        ['United States', US['Day'].max(),measuresdf.loc[measuresdf['Country']== 'United States'].Country.count(),US['Cases'].max(), US['Deaths'].max(),US['Deaths'].max()/US['Cases'].max()*100,US['Daily_Cases'].mean(),US['Daily_Cases'].median(),US['Daily_Cases'].std(),US['Daily_Fatalities'].mean(),US['Daily_Fatalities'].median(),US['Daily_Fatalities'].std()]]

df = pd.DataFrame(data, columns = ['Country', 'Days','Measures','Cases', 'Deaths', 'Death ratio','Cases Mean','Cases Median','Cases Std','Deaths Mean','Deaths Median','Deaths Std']) 

sns.set(style="whitegrid")
plt.figure(figsize=(25,15))
plt.title('Cases and fatalities per country') # Title
df = df.sort_values(by=['Cases'])

ax1 = sns.barplot(x='Country', y='Cases',label='Cases per country', data=df)

ax2 = ax1.twinx()
ax2.tick_params(axis='y')
ax2 = sns.lineplot(x='Country', y='Deaths',marker='*', label='Fatalities per country', data=df)

plt.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# If we consider the total cases in Argentina, Chile, South Korea, Brazil, China, Italy, Spain and United States, we can see that the United States is the country that has more cases reported and is facing with the biggest number of fatalities due to the virus. In the other hand, China shows low levels of infections and South Korea low number of fatalities.

# # Country measures, infection propagation and fatalities

# In[ ]:


df['Death ratio'] = df['Death ratio'].map('{:,.2f}%'.format)
df['Cases Mean'] = df['Cases Mean'].map('{:,.2f}'.format)
df['Deaths Mean'] = df['Deaths Mean'].map('{:,.2f}'.format)
df['Cases Median'] = df['Cases Median'].map('{:,.2f}'.format)
df['Deaths Median'] = df['Deaths Median'].map('{:,.2f}'.format)
df['Cases Std'] = df['Cases Std'].map('{:,.2f}'.format)
df['Deaths Std'] = df['Deaths Std'].map('{:,.2f}'.format)
df['Cases'] = df['Cases'].map('{:,}'.format)
df['Deaths'] = df['Deaths'].map('{:,}'.format)
df


# In[ ]:


x = AR
plt.figure(figsize=(25,15))
plt.title(x.iloc[0]['Country'] + ' daily cases distribution') # Title
sns.distplot(x['Daily_Cases'], kde=0, fit=stats.gamma);
sns.kdeplot(x['Daily_Cases'])
plt.legend();


# In[ ]:


x = SK
plt.figure(figsize=(25,15))
plt.title(x.iloc[0]['Country'] + ' daily cases distribution') # Title
sns.distplot(x['Daily_Cases'], kde=0, fit=stats.gamma);
sns.kdeplot(x['Daily_Cases'])
plt.legend();


# In[ ]:


x = CL
plt.figure(figsize=(25,15))
plt.title(x.iloc[0]['Country'] + ' daily cases distribution') # Title
sns.distplot(x['Daily_Cases'], kde=0, fit=stats.gamma);
sns.kdeplot(x['Daily_Cases'])
plt.legend();


# In[ ]:


x = BR
plt.figure(figsize=(25,15))
plt.title(x.iloc[0]['Country'] + ' daily cases distribution') # Title
sns.distplot(x['Daily_Cases'], kde=0, fit=stats.gamma);
sns.kdeplot(x['Daily_Cases'])
plt.legend();


# In[ ]:


x = CH
plt.figure(figsize=(25,15))
plt.title(x.iloc[0]['Country'] + ' daily cases distribution') # Title
sns.distplot(x['Daily_Cases'], kde=0, fit=stats.gamma);
sns.kdeplot(x['Daily_Cases'])
plt.legend();


# In[ ]:


x = UK
plt.figure(figsize=(25,15))
plt.title(x.iloc[0]['Country'] + ' daily cases distribution') # Title
sns.distplot(x['Daily_Cases'], kde=0, fit=stats.gamma);
sns.kdeplot(x['Daily_Cases'])
plt.legend();


# In[ ]:


x = SP
plt.figure(figsize=(25,15))
plt.title(x.iloc[0]['Country'] + ' daily cases distribution') # Title
sns.distplot(x['Daily_Cases'], kde=0, fit=stats.gamma);
sns.kdeplot(x['Daily_Cases'])
plt.legend();


# In[ ]:


x = IT
plt.figure(figsize=(25,15))
plt.title(x.iloc[0]['Country'] + ' daily cases distribution') # Title
sns.distplot(x['Daily_Cases'], kde=0, fit=stats.gamma);
sns.kdeplot(x['Daily_Cases'])
plt.legend();


# **South Korea**
# * First Case: Jan 20, 2020
# * First Measure: Feb 4, 2020
# 
# **China**
# * First Case: Dec 18, 2019 
# * First Measure: Dec 25, 2019
# 
# **Spain**
# * First Case: Feb 01, 2020
# * First Measure: Feb 12, 2020
# 
# **Italy**
# * First Case: Jan 31, 2020
# * First Measure: Mar 25, 2020
# 
# **United Kingdom**
# * First Case: Jan 31, 2020
# * First Measure: Mar 14, 2020
# 
# **United States**
# * First Case: Jan 21, 2020
# * First Measure: Mar 16, 2020
# 
# According to the data, seems that countries that implemented measures like social distancing among other strategies and were able to adopt technology to track and trace positive cases with also a clear effort in testing the population have been able to manage the infection propagation better than those who didn't manage that.

# # Daily evolution cases & fatalities

# In[ ]:


Countries=pd.concat([World]) 
Countries=Countries.sort_values(by=['Day','Date'], ascending=[True,True])

plt.figure(figsize=(25,15))
plt.title('World details') # Title
#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Fatalities", palette=['red'], dashes='true', hue="Country",data=Countries)


legend = plt.legend()
legend.texts[0].set_text("")
legend.texts[1].set_text("Daily Cases")
legend.texts[2].set_text("")
legend.texts[3].set_text("Daily Fatalities")


plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# The latest information showns that we are still in the area of exponential growth in terms of the infection propagation. At the global level the fatalities ratio is about 6% of the infected people.

# In[ ]:


Countries=pd.concat([SK]) 
Countries=Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('South Korea details') # Title
#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Fatalities", palette=['red'], dashes='true', hue="Country",data=Countries)


legend = plt.legend()
legend.texts[0].set_text("")
legend.texts[1].set_text("Daily Cases")
legend.texts[2].set_text("")
legend.texts[3].set_text("Daily Fatalities")


plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# On day 40 from the first infection reported in SK, the daily infection get the max. value. After that growth rate was negative. In SK the fatalities rate is about 2% of the infected people.

# In[ ]:


Countries=pd.concat([CH]) 
Countries=Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('China details') # Title
#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Fatalities", palette=['red'], dashes='true', hue="Country",data=Countries)


legend = plt.legend()
legend.texts[0].set_text("")
legend.texts[1].set_text("Daily Cases")
legend.texts[2].set_text("")
legend.texts[3].set_text("Daily Fatalities")


plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# China higher reported infected values was about in day 46. After that day the decrease was significantly. The fatalities rate for China is in the order of 5%

# In[ ]:


Countries=pd.concat([US]) 
Countries=Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('US details') # Title
#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Fatalities", palette=['red'], dashes='true', hue="Country",data=Countries)


legend = plt.legend()
legend.texts[0].set_text("")
legend.texts[1].set_text("Daily Cases")
legend.texts[2].set_text("")
legend.texts[3].set_text("Daily Fatalities")


plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# The US seems that is still in the exponential growth of the curve and seems that is shaping also the World curve. The fatalities ratio is in the order of 5%.

# In[ ]:


Countries=pd.concat([AR]) 
Countries=Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('Argentina details') # Title
#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Fatalities", palette=['red'], dashes='true', hue="Country",data=Countries)


legend = plt.legend()
legend.texts[0].set_text("")
legend.texts[1].set_text("Daily Cases")
legend.texts[2].set_text("")
legend.texts[3].set_text("Daily Fatalities")


plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# In[ ]:


Countries=pd.concat([CL]) 
Countries=Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('Chile details') # Title
#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Fatalities", palette=['red'], dashes='true', hue="Country",data=Countries)


legend = plt.legend()
legend.texts[0].set_text("")
legend.texts[1].set_text("Daily Cases")
legend.texts[2].set_text("")
legend.texts[3].set_text("Daily Fatalities")


plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# In[ ]:


Countries=pd.concat([UY]) 
Countries=Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('Uruguay details') # Title
#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Fatalities", palette=['red'], dashes='true', hue="Country",data=Countries)


legend = plt.legend()
legend.texts[0].set_text("")
legend.texts[1].set_text("Daily Cases")
legend.texts[2].set_text("")
legend.texts[3].set_text("Daily Fatalities")


plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# # Daily infection evolution & accumulative fatalities

# In[ ]:


Countries=pd.concat([World]) 
Countries=Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('World details') # Title
#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)
sns.barplot(x="Day", y="Deaths", palette=['red'], hue="Country",data=Countries)


plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# In[ ]:


Countries=pd.concat([SK]) 
Countries=Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('South Korea details') # Title
#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)
sns.barplot(x="Day", y="Deaths", palette=['red'], hue="Country",data=Countries)


plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# In[ ]:


Countries=pd.concat([CH]) 
Countries=Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('China details') # Title
#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)
sns.barplot(x="Day", y="Deaths", palette=['red'], hue="Country",data=Countries)


plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# In[ ]:


Countries=pd.concat([AR]) 
Countries=Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('Argentina details') # Title
#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)
sns.barplot(x="Day", y="Deaths", palette=['red'], hue="Country",data=Countries)


plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# In[ ]:


Countries=pd.concat([CL]) 
Countries=Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('Chile details') # Title
#sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
sns.lineplot(x="Day", y="Daily_Cases", palette=['green'], hue="Country",data=Countries)
sns.barplot(x="Day", y="Deaths", palette=['red'], hue="Country",data=Countries)


plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# # Accumulative evolution

# In[ ]:


# Concatenate dataframes 
Countries = pd.concat([World, SK, CH,US,UK, SP, IT]) 
Countries = Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('World, South Korea, China, UK and US reported COVID19 infections') # Title
sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# The graph above shows that US, Italy and Spain are outperforming the average infection rate of the world until day 76. Italy after day 76 is bellow the World curve. China and South Korea has shown that the curve is been flattening since day 45th. Is well known than China and SK have implemented some social distancing practices, but also have both adopted some technology to track and trace people infected and their "contacts".
# https://www.pharmaceutical-technology.com/features/coronavirus-affected-countries-south-korea-covid-19-outbreak-measures-impact/
# https://www.theguardian.com/commentisfree/2020/mar/20/south-korea-rapid-intrusive-measures-covid-19
# https://www.weforum.org/agenda/2020/03/south-korea-covid-19-containment-testing/

# In[ ]:


# Concatenate dataframes 
Countries=pd.concat([BR, US, SP, IT]) 
Countries = Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('Brazil, Spain, Italy and US reported COVID19 infections') # Title
sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# According to the dataset available, seems that the infection rate in BR is growing faster than compared with the US, just between Spain and Italy. At day 41 of the first report the number of infected people is bigger than in the States.
# 
# https://www.aljazeera.com/indepth/features/brazil-overcrowded-favelas-ripe-spread-coronavirus-200409113555680.html
# 
# https://www.bbc.com/news/world-latin-america-52137165
# 

# In[ ]:


Countries=pd.concat([AR,BR,CO,EC,GU,PE,CR,CL]) 
Countries = Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('Argentina, Brazil, Colombia, Ecuador, Guatemala, Costa Rica and Chile reported COVID19 infections') # Title
sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# In[ ]:


Countries=pd.concat([AR,BR,CO,EC,GU,PE,CR,CL]) 
Countries = Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('Argentina, Brazil, Colombia, Ecuador, Guatemala, Costa Rica and Chile reported COVID19 fatalities') # Title
sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# In this graph you can create basically 3 categories, countries with a very agressive slope like Brazil and Peru. Countries with a high growing rate such as Chile and Ecuador. And lastly countries with a very controlled growing curve like Peru, Argentina, Guatemala and Costa Rica

# In[ ]:


Countries=pd.concat([AR,BR,CO,EC,GU,PE,CR,CL]) 
Countries = Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('Argentina, Brazil, Colombia, Ecuador, Guatemala, Costa Rica and Chile reported COVID19 fatalities') # Title
sns.lineplot(x="Day", y="Deaths", hue="Country",data=Countries)
plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# If we use the same countries data set that in the graph below seems that the clusters are not following the same behaviour, all countries seems at this stage be in the same group except Brazil in where the fatalities curve is growing pretty fast.

# In[ ]:


Countries=pd.concat([SK,AR,IT, SP]) 
Countries = Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('South Korea, Argentina, Italia and Spain reported COVID19 infections') # Title
sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# Seems that argentina curve is following the SK case and the measures taken by the goverment helped to flatten the curve.

# In[ ]:


Countries=pd.concat([SK,AR,IT, SP]) 
Countries = Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('South Korea, Argentina, Italia and Spain reported COVID19 fatalities') # Title
sns.lineplot(x="Day", y="Deaths", hue="Country",data=Countries)
plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# In[ ]:


Countries=pd.concat([AR,CL, UY]) 
Countries=Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('Argentina, Chile and Uruguay reported COVID19 infection') # Title
sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
#sns.barplot(x="Day", y="Deaths", hue="Country",data=Countries)
plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# In[ ]:


Countries=pd.concat([AR,CL, UY]) 
Countries=Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('Argentina, Chile and Uruguay reported COVID19 fatalities') # Title
sns.lineplot(x="Day", y="Deaths", hue="Country",data=Countries)
#sns.barplot(x="Day", y="Deaths", hue="Country",data=Countries)
plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# Comparing infections and fatalities among Argentina, Chile and Uruguay, seems that the testing rate in AR is lower than in CL, considering that the # of fatalities in AR is bigger than in AR

# In[ ]:


Countries=pd.concat([CL,AR,BO, PY, UY]) 
Countries=Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('Chile, Argentina, Bolivia, Paraguay and Uruguay reported COVID19 infections') # Title
sns.lineplot(x="Day", y="Cases", hue="Country",data=Countries)
plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()


# In[ ]:


Countries=pd.concat([CL,AR,BO, PY, UY]) 
Countries=Countries.sort_values(by=['Day'], ascending=True)

plt.figure(figsize=(25,15))
plt.title('Chile, Argentina, Bolivia, Paraguay and Uruguay reported COVID19 deaths') # Title
sns.lineplot(x="Day", y="Deaths", hue="Country",data=Countries)
plt.xticks(Countries.Day.unique(), rotation=90)
plt.show()

