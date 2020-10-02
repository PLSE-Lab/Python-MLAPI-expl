#!/usr/bin/env python
# coding: utf-8

# # Gun Violence Data Analysis
# 
# We will deal with data describing 260k gun incidents in the US. The data runs between January 2013 and March 2018.
# 
# The data was obtained from [gunviolencearchive.org](http://www.gunviolencearchive.org/). This database has been obtained from public records and news sources. While this has allowed to cover a large number of incidents, we have to understand that a considerable amount of data in the set will be missing. It will be a good exercise to work with other gun violence datasets and try to compare results. However, for this kernel we will stick to this data set.

# ## Initial Analysis
# 
# We will load required libraries and our data and get the basic outlook of our dataset.

# In[ ]:


# Import all required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from plotly import __version__
import plotly.graph_objs as go 
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 
get_ipython().run_line_magic('matplotlib', 'inline')
import re


# In[ ]:


# Read in the data
data = pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')

# Basic outlook of the data
data.head()


# In[ ]:


# Information about the dataset and datatypes
data.info()


# ## Visualizing missing values
# 
# While we have a wide range of data attributes to analyze, we need to understand how many of those attributes hold null values. Other than null, there are some columns with values such as, "Unknown". We will deal with such values later on.

# In[ ]:


plt.figure(figsize=(16,5))
sns.heatmap(data.isnull(),cbar=False,yticklabels=False)
plt.show()


# ### Multiple missing/useless values
# 
# With the above graph, we get a clear idea of all the data that is missing. This gives us a good direction in regards to which columns we will use to analyze the dataset. It should be intuitive to see that we should be careful while using data from fields like,  _participantRelationship_ or _gunType_ 

# ## Checking the news sources
# 
# Because all of this data was picked up from public data, we have ample data to analyze the data sources. We will be able to get an idea of the credibility of the sources.

# In[ ]:


from urllib.parse import urlparse

# Even though we have only a few null values, we still remove them
data_source = data.dropna(subset=['sources'],axis=0)

# We rebuild the index as it was broken by removing null values
data_source.reset_index(drop=True,inplace=True)

data_source['clean_source']= [urlparse(data_source.sources[i]).netloc for i in range(len(data_source.sources))]
data_source['clean_source'] = [data_source.clean_source[i].replace('www.','').replace('.com','') for i in range(len(data_source.sources))]


# In[ ]:


data_short = data_source['clean_source'].value_counts().head(15).to_dict()
x_list = list(data_short.keys())
y_list = list(data_short.values())
data1 = [go.Bar(x=x_list,y=y_list,marker=dict(color=['grey','grey','grey','grey','grey','grey','red','grey','grey','grey','grey','grey','grey','grey','grey']))]
iplot(data1)
#list(data_short.keys())


# ### Understanding the data sources
# 
# We see that most of the data has been picked up from respectable news sources but among those also lie sources such as **twitter** and **facebook**. In light of recent events, it is important to keep note of where all our data has been taken from. The source of the data holds credibility, so for now we can safely assume all data to be valid.

# ## Geographic distribution of shooting events
# 
# While the problem at hand is one that troubles the whole nation, it is not distributed equally across the country. Some places require extra care and attention to iron out their problems. The following graph should create an intuitive overview of the geographical distribution of the events.

# In[ ]:


data_geo = data['state'].value_counts()
x_geo = list(data_geo.keys())
y_geo = list(data_geo.values)
mean_count = data['state'].value_counts().mean()
geo_graph = [go.Bar(x=x_geo,y=y_geo)]
geo_layout = {'shapes':[{'type':'line','x0':0,'y0':mean_count,'x1':50,'y1':mean_count}]}
iplot({'data':geo_graph,'layout':geo_layout})


# ### Movement around the average
# 
# The horizontal line marks the overall national average of the number of events. We can clearly see that while some states lie markedly below the average, some are way past it. We need to keep in mind two things here. Firstly, not every state has equal population density. While this data does not allow us to do so, building a population densite graph against the number of violence events would have been another option. Secondly, this graph subtly complements the graph we built above with the news sources. Chicago newspapers were the biggest sources of information which falls in line with Illinois being the state with the most events. This can be taken as a minor indication that we might be heading in the right way with our analysis.

# ## Year wise deaths and injuries
# 
# We know that the problem has been troubling the republic for some time now. The question then stands that are we getting better or worse? Looking at the yearly death and injury rates can be a good overview 

# In[ ]:


## Here we change the date column in the date to individually work with years, months and days.
data['date']=pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month']=data['date'].dt.month
data['day_of_week'] = data['date'].dt.dayofweek
data.head()


# In[ ]:


year_data = data.groupby('year')['n_killed','n_injured'].sum()

nk_x = list(year_data['n_killed'].keys())
nk_y = list(year_data['n_killed'].values)
nk_graph = [go.Bar(x=nk_x,y=nk_y,marker=dict(color='crimson'))]
nk_layout = go.Layout(title='Year wise Deaths in Shootings')
iplot({'data':nk_graph,'layout':nk_layout})


# In[ ]:


ni_x = list(year_data['n_injured'].keys())
ni_y = list(year_data['n_injured'].values)
ni_graph = [go.Bar(x=ni_x,y=ni_y,marker=dict(color='lightcoral'))]
ni_layout = go.Layout(title='Year wise Injuries in Shootings')
iplot({'data':ni_graph,'layout':ni_layout})


# ### An increasing trend
# 
# In both the graphs, we can see a clear increasing trend of gun violence in the country. Later on in our analysis we will try to figure out what allowed this increasing trend. Were mass shootings a big effect in this growing trend or were there other events involved as well?
# 
# The outlier data of 2013 and 2018 should make intuitive sense. The dataset just didn't have enough data from these years to be able to represent them properly.
# 
# However, these graphs clearly show us that the problem is getting more serious and needs to be tackled on higher priority.

# ## Gun Trends
# 
# While we do see an increasing trend in the total number of deaths and injuries, we need to figure out what are the main sources of those injuries and fatalities. We can start by looking at the guns which were used. If we find that assault weapons contributed most to the damage, we know where to direct our efforts in order to solve the problem. Let's see what the graph tells us.

# In[ ]:


data['gun_type_parsed'] = data['gun_type'].fillna('0:Unknown')
gt = data.groupby(by=['gun_type_parsed']).agg({'n_killed': 'sum', 'n_injured' : 'sum', 'state' : 'count'}).reset_index().rename(columns={'state':'count'})

results = {}
for i, each in gt.iterrows():
    wrds = each['gun_type_parsed'].split("||")
    for wrd in wrds:
        if "Unknown" in wrd:
            continue
        wrd = wrd.replace("::",":").replace("|1","")
        gtype = wrd.split(":")[1]
        if gtype not in results: 
            results[gtype] = {'killed' : 0, 'injured' : 0, 'used' : 0}
        results[gtype]['killed'] += each['n_killed']
        results[gtype]['injured'] +=  each['n_injured']
        results[gtype]['used'] +=  each['count']
        
resultFrame = pd.DataFrame(results)

resultFrame.transpose().plot.bar(figsize=(12,3),title="Handgun Type Used")


# ### Changing Directions
# 
# This graph is probably the most visually intriguing in the whole lot. The difference between the elements is so stark that it's really hard to forget what this graph tells us.
# 
# Handguns, clearly have been used far more than any other gun. Not only that, bigger guns like AR-15s which were recently in the news actually are virtually non-existant on the graph when comparing to handguns.
# 
# This leads us into a new direction. We can't simply assume ideas, we need to check and verify everything. As a next step, we will try to see how many guns have been used in each incident. This is to check if the counts of handgun uses are higher because they are used both as an independent weapon and as a secondary one.

# In[ ]:


n_guns = data[data["n_guns_involved"].notnull()]
n_guns["n_guns_involved"] = n_guns["n_guns_involved"].astype(int)
n_guns = n_guns[["n_guns_involved"]]

def label(n_guns):
    if n_guns["n_guns_involved"] == 1 :
        return "ONE-GUN"
    elif n_guns["n_guns_involved"] > 1 :
        return "GREATER THAN ONE GUN"

n_guns["x"] = n_guns.apply(lambda n_guns:label(n_guns),axis=1)
n_guns["x"].value_counts().plot.pie(figsize=(7,7),autopct ="%1.0f%%",explode = [0,.2],shadow = True,colors=["indianred","grey"],startangle =25)
plt.title("NO OF GUNS INVOLVED")
plt.ylabel("")


# ### Moving focus
# 
# This graph clears the air a bit more. We can now safely and confidently say that most of the gun violence caused in the country is caused by handguns.
# 
# This difference now means we will have to look at our data in different ways. We are better off trying to analyze and work with the 91% of situations than working with the other 9%.
# 
# Before we move on to deeper analysis, we will try work with age groups of the participants.

# In[ ]:


age = data[data["participant_age"].notnull()][["participant_age"]]
age["participant_age"] = age["participant_age"].str.replace("::","-")
age["participant_age"] = age["participant_age"].str.replace(":","-")
age["participant_age"] = age["participant_age"].str.replace("[||]",",")
age = pd.DataFrame(age["participant_age"])
x1 = pd.DataFrame(age["participant_age"].str.split(",").str[0])
x2 = pd.DataFrame(age["participant_age"].str.split(",").str[1])
x3 = pd.DataFrame(age["participant_age"].str.split(",").str[2])
x4 = pd.DataFrame(age["participant_age"].str.split(",").str[3])
x5 = pd.DataFrame(age["participant_age"].str.split(",").str[4])
x6 = pd.DataFrame(age["participant_age"].str.split(",").str[5])
x7 = pd.DataFrame(age["participant_age"].str.split(",").str[6])
x1 = x1[x1["participant_age"].notnull()]
x2 = x2[x2["participant_age"].notnull()]
x3 = x3[x3["participant_age"].notnull()]
x4 = x4[x4["participant_age"].notnull()]
x5 = x5[x5["participant_age"].notnull()]
x6 = x6[x6["participant_age"].notnull()]
x7 = x7[x7["participant_age"].notnull()]

age_dec  = pd.concat([x1,x2,x3,x4,x5,x6,x7],axis = 0)
age_dec["lwr_lmt"] = age_dec["participant_age"].str.split("-").str[0]
age_dec["upr_lmt"] = age_dec["participant_age"].str.split("-").str[1]
age_dec.head()

age_dec= age_dec[age_dec["lwr_lmt"]!='']
age_dec["lwr_lmt"] = age_dec["lwr_lmt"].astype(int)
age_dec["upr_lmt"] = age_dec["upr_lmt"].astype(int)

age_dec["age_bins"] = pd.cut(age_dec["upr_lmt"],bins=[0,20,35,55,130],labels=["TEEN[0-20]","YOUNG[20-35]","MIDDLE-AGED[35-55]","OLD[>55]"])
plt.figure(figsize=(8,8))
age_dec["age_bins"].value_counts().plot.pie(autopct = "%1.0f%%",shadow =True,startangle = 0,colors = sns.color_palette("prism",5),
                                            wedgeprops = {"linewidth" :3,"edgecolor":"k"})
my_circ = plt.Circle((0,0),.7,color = "white")
plt.gca().add_artist(my_circ)
plt.ylabel("")
plt.title("Distribution of age groups of participants",fontsize=20)


# ## Going Deeper
# 
# This dataset has it's own flaws. We have a lot of data missing, which doesn't really allow us to completely rely on our analysis from those parameters.
# 
# One parameter though, that not only gives a lot of information about each incident but also has next to none null values is the "incident_characteristics" column. This is mostly text data but it is structured in a way that makes it easier to take data out and analyze it.

# In[ ]:


from collections import Counter

total_incidents = []
for i, each_inc in enumerate(data['incident_characteristics'].fillna('Not Available')):
    split_vals = [x for x in re.split('\|', each_inc) if len(x)>0]
    total_incidents.append(split_vals)
    if i == 0:
        unique_incidents = Counter(split_vals)
    else:
        for x in split_vals:
            unique_incidents[x] +=1

unique_incidents = pd.DataFrame.from_dict(unique_incidents, orient='index')
colvals = unique_incidents[0].sort_values(ascending=False).index.values
find_val = lambda searchList, elem: [[i for i, x in enumerate(searchList) if (x == e)][0] for e in elem]

a = np.zeros((data.shape[0], len(colvals)))
for i, incident in enumerate(total_incidents):
    aval = find_val(colvals, incident)
    a[i, np.array(aval)] = 1
incident = pd.DataFrame(a, index=data.index, columns=colvals)

prominent_incidents = incident.sum()[[4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                                      21, 23,22,24,45,51]]
fig = {
    'data': [
        {
            'labels': prominent_incidents.index,
            'values': prominent_incidents,
            'type': 'pie',
            'hoverinfo':'label+percent+name',
            "domain": {"x": [0, .45]},
        }
    ],
    'layout': {'title': 'Prominent Incidents of Gun Violence',
               'showlegend': False}
}
iplot(fig)


# ### Comprehending the characteristics
# 
# There are a total of around 50 characteristics used in the data in different permutations and combinations to desrcibe each event. The pie chart above displays the most interesting of those characteristics and how many times did they show up in the data.
# 
# There are a couple of interesting ideas that come from this graph. If we look at the blue segment ( Posession of gun(s) during other crimes ) we realize how big of a problem can that be. Generally it would be hard to assume that this would contribute so much to gun violence. Not only that, if we look at the red segment ( posession of gun by felon or prohibited person ) we see a probelm that everyone will accept to fight against, regardless of political affilication but it still eludes our discussions in the mainstream.
# 
# Following these, we have drug involvement and drive-bys as the biggest contributors but right after that we see that "Domestic Violence" turns out as a considerable contributor to the whole issue. We can work more with that because we also have data describing relationships between the participants.

# In[ ]:


relation = data['participant_relationship']
relation = relation[relation.notnull()]
relation = relation.str.replace("[:|0-9]"," ").str.upper()
relation1 = pd.DataFrame({"count":[len(relation[relation.str.contains("FAMILY")]),
               len(relation[relation.str.contains("ROBBERY")]),
               len(relation[relation.str.contains("FRIENDS")]),
               len(relation[relation.str.contains("AQUAINTANCE")]),
               len(relation[relation.str.contains("NEIGHBOR")]),
               len(relation[relation.str.contains("INVASION")]),
               len(relation[relation.str.contains("CO-WORKER")]),
               len(relation[relation.str.contains("GANG")]),
               len(relation[relation.str.contains("RANDOM")]),
               len(relation[relation.str.contains("MASS SHOOTING")])],
              "category":["FAMILY","ROBBERY","FRIENDS","AQUAINTANCE","NEIGHBOR","INVASION","CO-WORKER","GANG","RANDOM","MASS SHOOTING"]})
relation1
plt.figure(figsize=(14,5))
sns.barplot("category","count",data=relation1,palette="prism")
plt.title("COUNT PLOT FOR PARTICPANT RELATION TYPE IN VIOLENT EVENTS")


# ### A troubling majority
# 
# Now before we take away ideas from this graph, we should remind ourselves that a lot of the data about participant relationships is not available to us. So, whatever we do infer from this graph, will need to be checked further from different sources and different datasets.
# 
# However, against common intuition, we see that family, friends, aquaintance and neighbors contribute the most to the whole issue while mass shootings have the smallest count.
# 
# A different problem, indeed. As I said before, we are better off dealing with majority contributor of the problem rather than the other ones in the minority. This means that we actually need to shift our focus somewhere, because from where we started looking, mass shootings were the biggest problem but the data reveals that a lot more people are being hurt by things that we wouldn't even imagine by ourselves.

# ## Final Graphs
# 
# We are limited by our data because of missing values, but we can still juice out some more information from our dataset. The following graphs are the final ones for this document.

# In[ ]:


data['gun_stolen'] = data['gun_stolen'].fillna('Null')

data['gun_stolen'] = data['gun_stolen'].str.replace('::',',')
data['gun_stolen'] = data['gun_stolen'].str.replace('|',' ')
data['gun_stolen'] = data['gun_stolen'].str.replace(',',' ')
data['gun_stolen']= data['gun_stolen'].str.replace('\d+', '')


data['Stolenguns']=data['gun_stolen'].apply(lambda x: x.count('Stolen'))
data['stolenguns']=data['gun_stolen'].apply(lambda x: x.count('stolen'))
data['Stolengunstotal'] = data['Stolenguns'] + data['stolenguns']

df_year_stolenguns = data[['year','Stolengunstotal']].groupby(['year'], as_index = False).sum()


df_year_stolenguns[['year','Stolengunstotal']].set_index('year').plot(kind='bar')


# In[ ]:


from collections import Counter
big_text = "||".join(data['incident_characteristics'].dropna()).split("||")
incidents = Counter(big_text).most_common(30)
xx = [x[0] for x in incidents]
yy = [x[1] for x in incidents]

trace1 = go.Bar(
    x=yy[::-1],
    y=xx[::-1],
    name='Incident Characterisitcs',
    marker=dict(color='purple'),
    opacity=0.3,
    orientation="h"
)
data1 = [trace1]
layout = go.Layout(
    barmode='group',
    margin=dict(l=350),
    width=800,
    height=600,
    legend=dict(dict(x=-.1, y=1.2)),
    title = 'Key Incident Characteristics',
)

fig = go.Figure(data=data1, layout=layout)
iplot(fig, filename='grouped-bar')


# # Where to now?
# 
# Increasing death rate, handguns, family, posession by felons and increasing number of stolen guns. If I want anyone to take away anything from this data, then it would be the few words and phrases I have just mentioned.
# 
# When I started working on the data, I was only focussing on the events of mass shootings. Unfortunately, the description for mass shootings is different for the different data sets that are out there. In our case, any event involving 4+ people would be considered a mass shooting.
# 
# Moving away then, from the mass shootings a completely new image revealed itself. While people are debating wether or not to change gun laws in the country, the majority of the problem is being eluded by all discussions.
# 
# 
# ## Next steps
# 
# All these ideas and their in-depth analysis is not under the premise of this paper. We realized something new about the problem we are all facing - I urge you to go out and do your own research to come at your conclusion.
# 
# While, for this dataset - this is the most I was able to take from it. There are many more research papers available on the topic which have used more comprehensive data, these are generally one google search away and can be a good starting point for the curious ones.

# In[ ]:




