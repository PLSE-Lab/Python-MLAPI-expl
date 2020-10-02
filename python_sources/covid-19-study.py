#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import cufflinks as cf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly as ply
import datetime as dt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')
df.describe(include='all')


# In[ ]:


import datetime as dt

df['ReportDateStamp'] = pd.to_datetime(df['reporting date'])
df['ReportDate_F'] = df['ReportDateStamp'].apply(lambda x: x.date())
Sta = df['ReportDateStamp'].max().strftime("%d/%m/%Y")
asa = dt.datetime.today().strftime("%d/%m/%Y")
End = df['ReportDateStamp'].min().strftime("%d/%m/%Y")

print('Most Recent: ' + Sta)
print('Earliest: ' + End )
print('As At: ' + asa )


# ## Look at the line-by-line dataset, doesn't seem to be updated or very big but still worth a look

# In[ ]:


plt.figure(figsize=(15,5)) 
sns.set(palette='colorblind')
sns.distplot(df['age'],rug=True,hist=False)
plt.xlim(left=0)


# Seems have at an overall level, double peaks. Lets see how it varies by country.

# In[ ]:


country_list = list(df['country'].value_counts().head(7).reset_index()['index'])
popular_countries = df[df['country'].isin(country_list)]


fig = go.Figure()
fig.add_trace(go.Violin(x=popular_countries['country'][ popular_countries['gender'] == 'male' ],
                        y=popular_countries['age'][ popular_countries['gender'] == 'male' ],
                        legendgroup='M', scalegroup='M', name='Male',
                        line_color='blue')
             )
fig.add_trace(go.Violin(x=popular_countries['country'][ popular_countries['gender'] == 'female' ],
                        y=popular_countries['age'][ popular_countries['gender'] == 'female' ],
                        legendgroup='F', scalegroup='F', name='Female',
                        line_color='green')
             )

fig.update_traces(box_visible=True, meanline_visible=True)
fig.update_layout(violinmode='group')
fig.show()


# Conclusion from this plot shows that Thailand should focus on younger children since, the have the lowest concentration of ages. 
# Japan could be worse effected since they have a higher median age of all countries for both genders. 
# Looking at this plot, people across the world need to stop seeing their Grandmothers. Compared to males, females are skewed much higher on the infection for most countires.

# In[ ]:


fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(12,6))


else_countries = df[~df['country'].isin(country_list)]
sns.boxplot(y=else_countries['age'],x=else_countries['gender'],ax=ax[0])
sns.boxplot(y=popular_countries['age'],x=popular_countries['gender'],ax=ax[1])

# ax[2].table(cellText=else_countries['gender'].value_counts().reset_index()
#             , cellColours=None, cellLoc='right', colWidths=None, rowLabels=None, 
#       rowColours=None, rowLoc='left', colLabels=None, colColours=None, colLoc='center',
#       loc='right', bbox=None, edges='closed')

ax[0].title.set_text('All ex 7 countries')
ax[1].title.set_text('Top 7 countries')


plt.ylim((0,100))
print(df['gender'].value_counts().reset_index())


# Females seem to be much older when they contract virus when split down the different countires M/F, however when comparing the top 7 vs else, not so clear but *there is still a difference*. This shows that when countries get infected, they seem to be being driven by old(er) females. Could be noise but male distributions are consistent. Cannot draw any stable conclusions on 382 females overall

# ## Next Step to try and categorise somehow the groups infected
# 
# 

# Use the rolling feature to show recovery figures

# In[ ]:


import plotly.express as px
time_series = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv',
                          index_col='id', parse_dates=True)
time_series['recovered_value'] = np.where(time_series['recovered'] != '0',1,0)
time_series['case'] = 1

plot1 = time_series.groupby(['reporting date'])['recovered_value','case'].sum()
plot1['recovery_rate'] = plot1['recovered_value']/plot1['case'] 
# Compute the centered 7-day rolling mean
plt.figure(figsize=(15,5))
rolling=14
plot1_7d = plot1.rolling(rolling, center=True).mean().reset_index()
fig = px.line(plot1_7d, x="reporting date", y="recovered_value", title='Recovery figures  (rolling)')
fig.show()


# Looks like dataset hasn't been updated. Maybe continue once this has been refreshed?

# Could so classify the symptoms for dies vs not dies, might need more data though (?)

# In[ ]:


#sns.countplot(x='symptom',data=df)
df[['symptom_1','symptom_2','symptom_3']] = df['symptom'].str.split(pat=',', n=2, expand=True)

plot,ax = plt.subplots(nrows=2,ncols=1,figsize=(22,10))
sns.countplot(x='symptom_1',data=df,ax=ax[0])
sns.countplot(x='symptom_2',data=df,ax=ax[1])

ax[0].set_title('Symptom (1)')
ax[1].set_title('Symptom (2)')

for ax in plot.axes:
    plt.sca(ax)
    plt.xticks(rotation=45)
    plt.xlabel(' ')
plt.tight_layout()


# So there is no need to go out and buy toilet paper after all, we can sleep soundly. 
# Possibly need to stock up on Lemsip and cold packs.

# ## Switch datasets to the larger aggregate data, predict the cases X country

# In[ ]:


new_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=True,index_col='SNo')


# In[ ]:


new_df.describe(include='all')


# In[ ]:


new_df['ReportDateStamp'] = pd.to_datetime(new_df['Last Update'])
new_df['ReportDate_F'] = new_df['ReportDateStamp'].apply(lambda x: x.date())
Sta = new_df['ReportDateStamp'].max().strftime("%d/%m/%Y")
asa = dt.datetime.today().strftime("%d/%m/%Y")
End = new_df['ReportDateStamp'].min().strftime("%d/%m/%Y")

print('Most Recent: ' + Sta)
print('Earliest: ' + End )
print('As At: ' + asa )


# ## Analyse UK vs EU countries infection rates

# This df has been updated much more recently. 
# Here I am interested in EU countires cases to the UK. So try and group EU countries (exclude Italy, analyse seperately)
# I want to see if the UK governments' sliggish response has effected the cases.

# In[ ]:


total_df = new_df.groupby(['Country/Region','ObservationDate'])['Confirmed'].sum().reset_index()
countries_summary  = new_df.groupby(['Country/Region'])['Confirmed'].sum().reset_index()
countries_summary  = countries_summary.loc[countries_summary['Country/Region'] != 'Mainland China',:]
countries_summary  = countries_summary.sort_values(by=['Confirmed'],ascending=0).head(15)

plt.figure(figsize=(20,5))
sns.barplot(x='Country/Region',y='Confirmed',data=countries_summary)
plt.xticks(rotation=45)


# Simple countplot of all countries in df, excluding mainland China. Which clearly has the most cases. Next would be to group the EU countires excluding Italy to stop it skewing result.

# In[ ]:


EU_MAP = dict({"Austria" : 'EU',
"Belgium" : 'EU' ,
"Bulgaria" : 'EU' ,
"Croatia" : 'EU' ,
"Cyprus" : 'EU' ,
"Czechia" : 'EU' ,
"Denmark" : 'EU' ,
"Estonia" : 'EU' ,
"Finland" : 'EU' ,
"France" : 'EU' ,
"Germany" : 'EU' ,
"Greece" : 'EU' ,
"Hungary" : 'EU' ,
"Ireland" : 'EU' ,
#"Italy" : 'EU' ,
"Latvia" : 'EU' ,
"Lithuania" : 'EU' ,
"Luxembourg" : 'EU' ,
"Malta" : 'EU' ,
"Netherlands" : 'EU' ,
"Poland" : 'EU' ,
"Portugal" : 'EU' ,
"Romania" : 'EU' ,
"Slovakia" : 'EU' ,
"Slovenia" : 'EU' ,
"Spain" : 'EU' ,
"Sweden" : 'EU' ,
"UK" : 'UK'})

new_df['EU_member'] = new_df['Country/Region'].map(EU_MAP)
new_df.groupby(['EU_member'])['Confirmed'].sum().reset_index()


groupd_df = new_df.groupby(['EU_member']).agg({'Confirmed': 'sum'})
# Change: groupby state_office and divide by sum
pcts = groupd_df.apply(lambda x: 100 * x / float(x.sum()))
pcts


# It's not going to be a straight comparison since UK only represents 5% of cases. 

# In[ ]:


d1 = dt.datetime(2020,2,15) 
new_df['ObservationDate_F'] = pd.to_datetime(new_df['ObservationDate'])
new_df = new_df.loc[new_df['ObservationDate'] >= d1,:]

time_series = new_df.groupby(['EU_member','ObservationDate_F'])['Confirmed'].sum().reset_index()



plt.figure(figsize=(20,5))
sns.lineplot(x='ObservationDate_F', y='Confirmed', data=time_series, hue='EU_member')
plt.xticks(rotation=45)


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

model = Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', LinearRegression(fit_intercept=False))])

model = model.fit(x[:, np.newaxis], y)

