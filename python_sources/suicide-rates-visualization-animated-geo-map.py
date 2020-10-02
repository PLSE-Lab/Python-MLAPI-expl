#!/usr/bin/env python
# coding: utf-8

# Hello to all! I'm a beginner and I'm open to all kinds of advice. 
# If I made any logical or other kinds mistakes, please warn me. I hope you find my kernel good :)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from wordcloud import WordCloud
from plotly import tools


# In[ ]:


df=pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")
worldmap=pd.read_csv("../input/world-capitals-gps/concap.csv")
df.head()


# In[ ]:


#There are some countries that do no fit with the world capital gps.
wmC=list(worldmap.CountryName)


for each in df.country.unique():
    if bool(each in wmC):
        continue
    else:
        print(each)


# In[ ]:


df.country=df.country.replace("Russian Federation","Russia")
df.country=df.country.replace("Republic of Korea","South Korea")
df.country=df.country.replace("Cabo Verde","Cape Verde")
df.country=df.country.replace("Saint Vincent and Grenadines","Saint Vincent and the Grenadines") 


# In[ ]:


#average suicides_no by countries(sorted)
country=df.country.unique()
suicide_numb=[]
for each in country:
    suicide_numb.append(df[df.country==each].suicides_no.mean())
country_suicide_no=pd.DataFrame({"country":country,"suicide_number":suicide_numb})
new_index=(country_suicide_no.suicide_number.sort_values(ascending=False)).index.values
country_suicide_no=country_suicide_no.reindex(new_index)
f,ax=plt.subplots(figsize=(10,15))
sns.barplot(x=country_suicide_no.suicide_number,y=country_suicide_no.country)
plt.xticks(rotation=90)
ax.legend(loc="lower right",frameon=True)
ax.set(xlabel="Average suicide number",ylabel="States",title="Average Suicide number by States")


# In[ ]:


#top 15 country most suicide
df.groupby(by=['country'])['suicides_no'].sum().reset_index().sort_values(['suicides_no'],ascending=True).tail(25).plot(x='country',y='suicides_no',kind='barh')
plt.title("Top 15 Suicides_no by Country")   
plt.show()


# In[ ]:


#SUICIDE COUNT BY COUNTRY AND SEX
d1=df[df.sex=="male"]
d2=df[df.sex=="female"]
list1=[]
list2=[]
for each in country:
  list1.append(d1[d1.country==each].suicides_no.sum())
  list2.append(d2[d2.country==each].suicides_no.sum())
list1=pd.DataFrame({"country":country,"suicides_no_male":list1,"suicides_no_female":list2})
list1=list1.sort_values(["suicides_no_male"],ascending=False).head(15)

plt.subplots(figsize=(20,20))
sns.barplot(x=list1.country,y=list1.suicides_no_male,color="green",alpha=0.5,label="male")
sns.barplot(x=list1.country,y=list1.suicides_no_female,color="blue",alpha=0.7,label="female")
plt.title("Top 15 Suicide Count by Country and Sex")
plt.show()


# In[ ]:


#SUICIDE COUNT BY YEAR
d1=df[df.sex=="male"]
d2=df[df.sex=="female"]
years=df.year.unique()
list1=[]
list2=[]
for each in years:
   list1.append(d1[d1.year==each].suicides_no.sum())
   list2.append(d2[d2.year==each].suicides_no.sum())
list1=pd.DataFrame({"year":years,"suicides_no_male":list1,"suicides_no_female":list2})
list1=list1.sort_values(["suicides_no_male"],ascending=False).head(15)

plt.subplots
sns.pointplot(x='year',y='suicides_no_male',data=list1,color='lime',alpha=0.8)
sns.pointplot(x='year',y='suicides_no_female',data=list1,color='red',alpha=0.8)
plt.text(40,0.6,'suicides_no_male',color='red',fontsize = 17,style = 'italic')
plt.text(40,0.55,'suicides_no_female',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('Year',fontsize = 15,color='blue')
plt.ylabel('Suicides_no',fontsize = 15,color='blue')
plt.title('Suicide count by year',fontsize = 20,color='blue')
plt.grid()


# In[ ]:


#Suicides_no by Age-group
age=df.age.unique()
list1=[]
for each in age:
    list1.append(df[df.age==each].suicides_no.sum())

list1=pd.DataFrame({"age":age,"suicides_no":list1})

pie1 = list1.suicides_no
labels = list1.age
fig = {
  "data": [
    {
      "values": pie1,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Suicide count by age-group",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Suicides_no by age-group",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Suicides_no",
                "x": 0.20,
                "y": 1
            },
        ]
    }
}
iplot(fig)


# In[ ]:


#AGE_GROUP and SEX
age=df.age.unique()
list1=[]
list2=[]
for each in age:
    list1.append(d1[d1.age==each].suicides_no.sum())#d1 is male(d1=df[df.sex=="male"])
    list2.append(d2[d2.age==each].suicides_no.sum())#d2 is female(d2=df[df.sex=="female"])
list1=pd.DataFrame({"age":age,"suicides_no_male":list1,"suicides_no_female":list2})

trace1 = go.Scatter(
    x=list1.age,
    y=list1.sort_values(["suicides_no_male"],ascending=False).suicides_no_male,
    mode='markers+text',
    text="Male",
    textposition='bottom center'
)
trace2 = go.Scatter(
    x=list1.age,
    y=list1.sort_values(["suicides_no_female"],ascending=False).suicides_no_female,
    mode='markers+text',
    text="female",
    textposition='bottom center'
)

fig = tools.make_subplots(rows=1, cols=2)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)

fig['layout'].update(height=600, width=800, title='Suicides_no by age-group')
iplot(fig)


# In[ ]:


#               AGE-GROUP AND COUNTRY
d1=df[df.sex=="male"]
d2=df[df.sex=="female"]
list1=[]
list3=[]
list2=[]

for each in country:
    for each1 in age:
        list1.append(df[(df.country==each) & (df.age==each1) & (df.sex=="male")].suicides_no.sum())
        list3.append(df[(df.country==each) & (df.age==each1) & (df.sex=="female")].suicides_no.sum())
        list2.append(each+"\n"+each1)

list1=pd.DataFrame({"country-age":list2,"suicides_no_male":list1,"suicides_no_female":list3})
list1=list1.sort_values(["suicides_no_male"],ascending=False).head(30)
women_bins = (list1.suicides_no_female)*-1
men_bins = list1.suicides_no_male


y = list1["country-age"]

layout = go.Layout(title="top 30 suicides_no by age-group and country",yaxis=go.layout.YAxis(side="right"),
                   xaxis=go.layout.XAxis(
                       range=[-100000, 300000],
#                       tickvals=[-100000, -50000, -10000, 0, 100000, 200000, 300000],
#                       ticktext=[300000, 200000, 100000, 0, 100000, 200000, 300000],
                      ),
                   barmode='overlay',
                   bargap=0.1)

data = [go.Bar(y=y,
               x=men_bins,
               orientation='h',
               name='Men',
               hoverinfo='x',
               marker=dict(color='powderblue')
               ),
        go.Bar(y=y,
               x=women_bins,
               orientation='h',
               name='Women',
               text=-1 * women_bins.astype('int'),
               hoverinfo='text',
               marker=dict(color='seagreen')
               )]
iplot(dict(data=data, layout=layout))


# In[ ]:


# wordcloud of countries that have more than 1000 suicides

list1=df[df.suicides_no>1000]

plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(list1.country))
plt.imshow(wordcloud)
plt.axis('off')

plt.show()


# In[ ]:


## swarm plot of suicides_no that is bigger than 2000 by generation and sex
list1=df[df.suicides_no>2000]
sns.swarmplot(x="sex", y="suicides_no",hue="generation", data=list1)
plt.show()


# In[ ]:


#           AGE-GROUP AND YEAR
list1=[]
list2=[]
years=df.year.unique()
age=df.age.unique()
for each in years:
    for each1 in age:
        list1.append(df[(df.year==each) & (df.age==each1)].suicides_no.sum())
        list2.append(str(each)+" ,"+each1)
list1=pd.DataFrame({"year-age":list2,"suicides_no":list1}).head(20)
list1=list1.sort_values(["suicides_no"],ascending=False)
f,ax=plt.subplots(figsize=(10,15))
sns.barplot(x=list1["suicides_no"],y=list1["year-age"])
plt.xticks(rotation=90)
ax.legend(loc="lower right",frameon=True)
ax.set(xlabel="suicide number",ylabel="year-age-group",title="Suicide count by Year and Age-group")
plt.show()


# In[ ]:


russia=df[df.country=='Russia']
russia=russia.reset_index()
male_russia=russia[russia["sex"]=="male"]
female_russia=russia[russia.sex=="female"]
russia_age=russia.age.unique()
#Russian males total suicide by age-group
russia_male_age=[]
for each in russia_age:
    russia_male_age.append(male_russia[male_russia.age==each].suicides_no.sum())
russia_male_age=pd.DataFrame({"age_group":russia_age,"total_suicide":russia_male_age})
#Russian females total suicide by age-group
russia_female_age=[]
for each in russia_age:
    russia_female_age.append(female_russia[female_russia.age==each].suicides_no.sum())
russia_female_age=pd.DataFrame({"age_group":russia_age,"total_suicide":russia_female_age})


f,ax=plt.subplots(figsize=(9,15))
sns.barplot(x=russia_male_age.age_group,y=russia_male_age.total_suicide,color="green",alpha=0.5,label="male")
sns.barplot(x=russia_female_age.age_group,y=russia_female_age.total_suicide,color="blue",alpha=0.7,label="female")
plt.title("suicides_no in russia by age-group and sex")
plt.show()


# In[ ]:


#Average suicide number in a country
country=df.country.unique()
suicide_numb=[]
for each in country:
    suicide_numb.append(df[df.country==each].suicides_no.mean())
country_suicide_no=pd.DataFrame({"country":country,"suicides_no":suicide_numb})
list1=[]
list2=[]
list3=[]
list4=[]

for each in country_suicide_no.country:
    x=worldmap[worldmap.CountryName==each].ContinentName.index.item()
    list1.append(worldmap.ContinentName[x])
    list2.append(worldmap.CapitalLatitude[x])
    list3.append(worldmap.CapitalLongitude[x])
country_suicide_no.head()
for each in country:
    x=country_suicide_no[country_suicide_no.country==each].suicides_no.index.item()
    list4.append("Country: " +each +" suicides_no:"+str("%.1f"%country_suicide_no.suicides_no[x]))
country_suicide_no["continent"]=list1
country_suicide_no["latitude"]=list2
country_suicide_no["longitude"]=list3
country_suicide_no["text"]=list4
country_suicide_no.head()
#CREATING MAP
country_suicide_no.continent.unique()
#array(['Europe', 'North America', 'South America', 'Australia', 'Asia',
#       'Central America', 'Africa'], dtype=object)
country_suicide_no["color"] = ""
country_suicide_no.color[country_suicide_no.continent == 'Europe'] = "rgb(0,116,217)"
country_suicide_no.color[country_suicide_no.continent == 'North America'] = "rgb(255,65,54)"
country_suicide_no.color[country_suicide_no.continent == 'Australia'] = "rgb(133,20,75)"
country_suicide_no.color[country_suicide_no.continent == 'Asia'] = "rgb(255,133,27)"
country_suicide_no.color[country_suicide_no.continent == 'Central America'] = "rgb(255,03,157)"
country_suicide_no.color[country_suicide_no.continent == 'Africa'] = "rgb(255,203,190)"
country_suicide_no.color[country_suicide_no.continent == 'South America'] = "rgb(50,21,86)"
country=country_suicide_no.country.unique()

data = [dict(
    type='scattergeo',
    lon = country_suicide_no["longitude"],
    lat = country_suicide_no["latitude"],
    hoverinfo = 'text',
    text = country_suicide_no["text"],
    mode = 'markers',
    marker=dict(
        sizemode = 'area',
        sizeref = 1,
        size= 10 ,
        line = dict(width=1,color = "white"),
        color = country_suicide_no["color"],
        opacity = 0.7),
)]
layout = dict(
    title = 'Suicide numbers by Country ',
    hovermode='closest',
    geo = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,
               countrywidth=1, projection=dict(type='mercator'),
              landcolor = 'rgb(217, 217, 217)',
              subunitwidth=1,
              showlakes = True,
              lakecolor = 'rgb(255, 255, 255)',
              countrycolor="rgb(5, 5, 5)")
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


#ANIMATED MAP
#shows total suicide number in a country by year
animation=df.copy()
country=df.country.unique()
year=df.year.unique()
list1=[]
list2=[]
list3=[]
list4=[]
list5=[]
list6=[]
list7=[]
for each in country:
    y=worldmap[worldmap.CountryName==each].index.item()
    for each1 in year:
        x=animation[(animation.country==each) & (animation.year==each1)].suicides_no.sum()
        list1.append(each)
        list2.append(each1)
        list3.append(x)
        list4.append(worldmap.ContinentName[y])
        list5.append(worldmap.CapitalLatitude[y])
        list6.append(worldmap.CapitalLongitude[y])
        list7.append(each+", Continent:"+worldmap.ContinentName[y]+", suicide number: "+str("%.1f"%x))
new=pd.DataFrame({"country":list1,"year":list2,"suicides_no":list3,"continent":list4
                  ,"Latitude":list5,"Longitude":list6,"text":list7}) 

dataset=new
new_index=(dataset.year.sort_values(ascending=True)).index.values
dataset=dataset.reindex(new_index)

years=[str(each) for each in dataset.year.unique()]
types=['Europe', 'North America', 'South America', 'Australia', 'Asia',
       'Central America', 'Africa']

custom_colors = {
    'Europe' : "rgb(0,116,217)",
    'North America' : "rgb(255,65,54)",
    'Australia' : "rgb(133,20,75)",
    'Asia' : "rgb(255,133,27)",
    'Central America' : "rgb(255,03,157)",
    'Africa' : "rgb(255,203,190)",
    'South America': "rgb(50,21,86)"
}

#types-->continent
figure = {
    'data': [],
    'layout': {},
    'frames': []
}

figure['layout']['geo'] = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,
               countrywidth=1, 
              landcolor = 'rgb(217, 217, 217)',
              subunitwidth=1,
              showlakes = True,
              lakecolor = 'rgb(255, 255, 255)',
              countrycolor="rgb(5, 5, 5)")
figure['layout']['hovermode'] = 'closest'
figure['layout']['sliders'] = {
    'args': [
        'transition', {
            'duration': 400,
            'easing': 'cubic-in-out'
        }
    ],
    'initialValue': '1985',
    'plotlycommand': 'animate',
    'values': years,
    'visible': True
}
figure['layout']['updatemenus'] = [
    {
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 500, 'redraw': False},
                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }
]

sliders_dict = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 20},
        'prefix': 'Year:',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': []
}
year = 1985
for ty in types:
    dataset_by_year = dataset[dataset['year'] == year]
    dataset_by_year_and_cont = dataset_by_year[dataset_by_year['continent'] == ty]
    
    data_dict = dict(
    type='scattergeo',
    lon = dataset['Longitude'],
    lat = dataset['Latitude'],
    hoverinfo = 'text',
    text = dataset.text,
    mode = 'markers',
    marker=dict(
        sizemode = 'area',
        sizeref = 1,
        size= 10 ,
        line = dict(width=1,color = "white"),
        color = custom_colors[ty],
        opacity = 0.7),
)
    figure['data'].append(data_dict)
    
# make frames
for year in years:
    frame = {'data': [], 'name': str(year)}
    for ty in types:
        dataset_by_year = dataset[dataset['year'] == int(year)]
        dataset_by_year_and_cont = dataset_by_year[dataset_by_year['continent'] == ty]

        data_dict = dict(
                type='scattergeo',
                lon = dataset_by_year_and_cont['Longitude'],
                lat = dataset_by_year_and_cont['Latitude'],
                hoverinfo = 'text',
                text = dataset_by_year_and_cont.text,
                mode = 'markers',
                marker=dict(
                    sizemode = 'area',
                    sizeref = 1,
                    size= 10 ,
                    line = dict(width=1,color = "white"),
                    color = custom_colors[ty],
                    opacity = 0.7),
                name = ty
            )
        frame['data'].append(data_dict)

    figure['frames'].append(frame)
    slider_step = {'args': [
        [year],
        {'frame': {'duration': 300, 'redraw': False},
         'mode': 'immediate',
       'transition': {'duration': 300}}
     ],
     'label': year,
     'method': 'animate'}
    sliders_dict['steps'].append(slider_step)


figure["layout"]["autosize"]= True
figure["layout"]["title"] = "Suicides_no by country and year"       

figure['layout']['sliders'] = [sliders_dict]

iplot(figure)


# In[1]:


#3D RIBBON CHART

trace1 = go.Scatter3d(
  x=new['year'],
  y=new['continent'],
  z=new['suicides_no'],
  text=new["text"],
  mode='markers',
  marker=dict(
      sizemode='diameter',
      sizeref=750,
      size=new['suicides_no'],
      color = new['suicides_no'],
      colorscale = 'Viridis',
      colorbar = dict(title = 'Suicides<br>No'),
      line=dict(color='rgb(140, 140, 170)')
  )
)

data=[trace1]

layout=go.Layout(height=800, width=800, title='Suicides_no by Year-Country')

fig=go.Figure(data=data, layout=layout)
iplot(fig)


# 
