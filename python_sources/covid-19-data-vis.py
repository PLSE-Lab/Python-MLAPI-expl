#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly
import plotly.express as px
import plotly.graph_objects as go
plt.rcParams['figure.figsize']=13,7
import cufflinks as cf
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot
import folium 
import requests
from bs4 import BeautifulSoup


# In[ ]:


pyo.init_notebook_mode(connected=True)
cf.go_offline()


# data

# In[ ]:


column_names= ['Sr.No','Name of State / UT','Total_cases','Cured/Discharged/migrated','Death']
df=pd.DataFrame(columns=column_names)


# In[ ]:


page=requests.get('https://www.mohfw.gov.in/')
soup=BeautifulSoup(page.content,'html.parser')
Table1 =soup.find(class_ ='table table-striped')
print(Table1)


# In[ ]:


A=[]
B=[]
C=[]
D=[]
E=[]

for row in Table1.findAll("tr"):
    cells = row.findAll('td')
    if len(cells)==5: #Only extract table body not heading
        A.append(cells[0].find(text=True))
        B.append(cells[1].find(text=True))
        C.append(cells[2].find(text=True))
        D.append(cells[3].find(text=True))
        E.append(cells[4].find(text=True))
        
        
#Adding Data to our DataFrame
df['Sr.No']=A
df['Name of State / UT']=B
df['Total_cases']=C
df['Cured/Discharged/migrated']=D
df['Death']=E

df


# In[ ]:


df.drop('Sr.No',axis=1,inplace=True)
df.iloc[0:33,0]


# In[ ]:


df[['Total_cases','Cured/Discharged/migrated','Death']]=df[['Total_cases','Cured/Discharged/migrated','Death']].astype(int)
df['Name of State / UT']=df['Name of State / UT'].astype(str)


# In[ ]:


df.dtypes


# In[ ]:


df['Total active cases'] = df['Total_cases'] - (df['Cured/Discharged/migrated']+df['Death'])


# In[ ]:


df


# In[ ]:


#total cases overall@India
total_cases_overall= df['Total_cases'].sum()
total__active_cases=df['Total active cases'].sum()
total_cured_cases = df['Cured/Discharged/migrated'].sum()
print('The Total no of cases in India is',total_cases_overall)
print('The Total no of active cases in India is',total__active_cases)
print('The Total no of cured cases in India is',total_cured_cases)


# In[ ]:


df.style.background_gradient(cmap='Reds')


# In[ ]:


Covid19=df.groupby('Name of State / UT')['Total_cases'].sum().sort_values(ascending=False).to_frame()


# In[ ]:


Covid19.style.background_gradient(cmap='Reds')


# In[ ]:


df.iplot(kind='bar',x='Name of State / UT',y='Cured/Discharged/migrated',xTitle='States',
    yTitle='Cured/Dishcharged/Migrated',)


# In[ ]:


plt.barh(df['Name of State / UT'],df['Total active cases'])


# In[ ]:


df.iplot(kind='scatter',x = 'Name of State / UT',y='Total_cases',xTitle='Name of State / UT',yTitle='Total cases',title='Covid-19 India',mode='markers+lines')


# In[ ]:


px.bar(df,x = 'Name of State / UT',y='Death')


# In[ ]:


Indian_cord=pd.read_excel('../input/covid19-india-coord/Indian Coordinates copy.xlsx')
Indian_cord


# In[ ]:


df_full=pd.merge(Indian_cord,df,on='Name of State / UT')


# In[ ]:


df_full #.replace(to_replace='Uttarakhand',value='Uttaranchal'


# In[ ]:


map=folium.Map(location=[20,70],zoom_start=4)

for lat,long,value, name in zip(df_full['Latitude'],df_full['Longitude'],df_full['Total_cases'],df_full['Name of State / UT']):
    folium.Circle([lat,long],radius=value*13,popup=('<strong>State</strong>: '+str(name).capitalize()+'<br>''<strong>Total_cases</strong>: ' +  str(value)+ '<br>'),color='black',fill_color='red',fill_opacity=0.2).add_to(map)


# In[ ]:


map


# In[ ]:


m=folium.Map(location=[20,70],zoom_start=4)
state_data= "../input/covid19-india-coord/india_states.geojson"
folium.Choropleth(
    geo_data=state_data,
    name='choropleth',
    data=df_full,
    columns=['Name of State / UT', 'Total_cases'],
    key_on='feature.properties.NAME_1',
    fill_color='Reds',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='COVID-19',
    nan_fill_color='white'
).add_to(m)

for lat,long,value, name in zip(df_full['Latitude'],df_full['Longitude'],df_full['Total_cases'],df_full['Name of State / UT']):
    folium.CircleMarker([lat,long],
                  radius=value*0.0001,  
                  fill_color='red',
                  fill_opacity=0.000001,
                  color='red',
                  popup=('<strong>State</strong>: '+str(name).capitalize()+'<br>''<strong>Total_cases</strong>: ' +  str(value)+ '<br>')
                 ).add_to(m)



# In[ ]:


m

