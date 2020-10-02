#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly


# In[ ]:


normal_summer=pd.read_csv('../input/olympic-games/summer.csv')
winter=pd.read_csv('../input/olympic-games/winter.csv')
dic=pd.read_csv('../input/olympic-games/dictionary.csv')


# In[ ]:


normal_summer.info()


# In[ ]:


dic.head()


# i will change the Country to code and i will add the country name from the dic data frame

# In[ ]:


normal_summer.rename(columns={'Country':'Code'},inplace=True)


# In[ ]:


normal_summer=pd.merge(normal_summer,dic,on='Code',how='outer')


# In[ ]:


normal_summer.head()


# In[ ]:


normal_summer.describe()


# <h2>1st impressions:</h2>
# *our dataset goes from 1896 to 2012
# (i don't see anything intresting other then this (at least for now))

# In[ ]:


normal_summer.describe(include=['O'])


# <h3>1st impressions:</h3>
# <ol>
#     <li>*USA has the majority of the medals </li>
#     <li>*Michael Pheleps has the majority of the medals</li>
#     <li>*the majority of the medals are taken in london </li>
# <li>*well when it comes to sport and disciplines it's kinda weird because the frequent sport is Aquatics but for the discipline we find athletics (hypothehis: the Aquatics is divided to many disciplines )</li>
# </ol>

# In[ ]:


normal_summer.head()


# In[ ]:


medals_map=normal_summer.groupby(['Country','Code'])['Medal'].count().reset_index()
medals_map=medals_map[medals_map['Medal']>0]

fig = px.choropleth(medals_map, locations="Code",
                    color="Medal", # lifeExp is a column of gapminder
                    hover_name="Country", # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Plasma)


# In[ ]:


fig.show()


# i will just column called 'useless' just to help with the counts for the rest of the EDA but consider as a count

# In[ ]:


normal_summer['useless']=1


# In[ ]:


medals_per_country=pd.pivot_table(index='Code',columns='Medal',values='useless',data=normal_summer,aggfunc=sum).fillna(0)


# In[ ]:


medals_per_country['Total']=medals_per_country['Gold']+medals_per_country['Silver']+medals_per_country['Bronze']


# In[ ]:


medals_per_country.sort_values(by='Total',ascending=False,inplace=True)


# In[ ]:


top=medals_per_country[:10]


# In[ ]:


top[['Bronze','Gold','Silver']].plot.barh(width=0.8,color=['#CD7F32','#FFDF00','#D3D3D3'])
fig=plt.gcf()
fig.set_size_inches(8,8)
plt.title('Medals Distribution Of Top 10 Countries (Winter Olympics)')
plt.show()


# well we just saw the middles per country let's look at the distribution of the medals per year (that looks intresting)

# In[ ]:


medals_per_year=pd.pivot_table(index='Year',columns='Medal',values='useless',data=normal_summer,aggfunc=sum)


# In[ ]:


medals_per_year.plot(color=['#CD7F32','#FFDF00','#D3D3D3'],figsize=(15,6))
plt.title('Medals per Year',fontsize=15)
plt.ylabel('Number of Medals',fontsize=15)
plt.xlabel('Year',fontsize=15)


# for some reason it is kinda weird because normally i thought the number that it will be the same since we always see in every sport there is (1 gold,1 silver,1 bronze) also the smthg happend around 1920 we should maybe dive a little bit in , also this huge increase by 2000 (it will be intresing to know the reason well i guess maybe there was another sports considred in the olympics)

# In[ ]:


disciplines_per_year=pd.pivot_table(index='Year',columns='Discipline',values='useless',data=normal_summer).fillna(0)


# In[ ]:


disciplines_per_year['disciplines']=0
for col in disciplines_per_year.columns:
    disciplines_per_year['disciplines']+=disciplines_per_year[col]


# In[ ]:


disciplines_per_year.head()


# <h2>now let's plot the disciplines per year so maybe that explains the number of medals</h2>

# In[ ]:


disciplines_per_year['disciplines'].plot(figsize=(15,6))
plt.ylabel('Disciplines',fontsize=15)
plt.xlabel('Year',fontsize=15)
plt.title('Count of discplines per year',fontsize=15)


# well we can see that our hypothesis is quiet correct because we see the peak around 1920 and the increase by 2000

# if we want we can dive even more and get insights why these discplines are just going up and down especailly in the beginning (why they are adding and and droppping disciplines 'some history' and what are these disciplines)

# In[ ]:


df=pd.merge(medals_per_country.reset_index(),normal_summer[['Code','Population','GDP per Capita']],on='Code').drop_duplicates().reset_index().drop('index',axis=1)


# In[ ]:


df.head()


# In[ ]:


df.corr()


# ### well we can notice a small correlation between the GDP and the Total medals 

# #### another good thing to discover is whether if a high achiver on average is high achiver in a lot of disciplines or he's just average in all that what make him on top 

# In[ ]:


medals_country_sport=pd.pivot_table(index='Code',columns='Sport',values='useless',data=normal_summer,aggfunc=sum).fillna(0)


# i guess the best way to see this is to grab the top 10  of every column and see it belongs to which country

# In[ ]:



for col in medals_country_sport.columns:
    plt.figure()
    data=medals_country_sport[col].sort_values(ascending=False)[:10]
    sns.barplot(x=data.index,y=data)


# #### well thanks to this we noticed a lot of stuff :
# <ol>
#     <li>thanks to the aquatics and the athletics sports the USA won around 2200 medals </li>
#     <li>USA is usually on top 3 but in some sports they have 0 medals also</li>
#     <li> there is some weird thing happening with that some sport there is only 1 or 2 countries that got medals like Rackets, Roque ... that's weird because the concept of always gold silver bronze assumme that in every sport we will at least 3 countries with medals so that makes us think that the concept of top 3 was not always there or that for some sports they don't follow this rule</li>
# </ol>
# i will just list down these sports :Basque pelota,cricket,Croquet,Ice hockey,jeu de Paume,Rackets,Roque,Water Motorsports

# ### let's go and discover these sports

# after some wikipedia search i noticed that all these sports except Ice hockey they were played only once or twice in the olympics and sometimes there is only 1 or nations in these games that's why we see only 1 or 2 countries sometimes ,well for the ice hockey it was transfered to the winter olympics that's why it only appears once in this data.
# (you can just so easily make some fast wiki reasearch)
# ##### maybe this can be explain the fact of the little zigzag in the number of disciplines from 1896 till 1940
# 

# In[ ]:


normal_summer.head()


# ##### still only few features we didn't see which are : the city , Gender and the athlete 

# In[ ]:


pd.pivot_table(index='Gender',columns='Medal',values='useless',aggfunc=sum,data=normal_summer)


# ##### we notice that the men have more medals but let's see why 

# In[ ]:


pd.pivot_table(index='Year',columns='Gender',values='useless',aggfunc=sum,data=normal_summer).fillna(0).plot(color=['#33D7FF','#FF33E6'],figsize=(10,6))


# over the years we see the men got always more medals then women but in the last years starting from 2000 diffrence is getting smaller this means more inclusion of the women in the olympics 
# 

# we can try to see if the medals per sport per gender

# In[ ]:


medals_per_sport_per_gender=pd.pivot_table(columns='Sport',index='Gender',values='useless',aggfunc=sum,data=normal_summer).fillna(0)


# In[ ]:


for col in medals_per_sport_per_gender.columns:
    plt.figure()
    sns.barplot(y=medals_per_sport_per_gender[col],x=medals_per_sport_per_gender.index,palette='coolwarm')


# we can notice that in all the sports men have more medals also we see that there is some sports just for men (only men participated ) and some others like (softball ) only women 

# ##### i think the years between 1916 and 1924 seem really intresting based on the plots that we made before i think we should dive in into that era 
# 

# In[ ]:


the_era=normal_summer[normal_summer['Year'].apply(lambda x: x in [1920])]


# In[ ]:


the_era.info()


# #### hypthesis : the peak is made because some sports were present in the olympics of 1920 that they were removed later 
# ##### let's test this hypothesis

# In[ ]:


d=dict()
for i in range(1900,2013,4):
    d[i]=normal_summer[normal_summer['Year']==i]['Sport'].nunique()
    


# In[ ]:


pd.Series(d).plot(figsize=(10,6))
plt.xlabel('Year',fontsize=15)
plt.ylabel('Count of sports',fontsize=15)
plt.title('Count of sports per year',fontsize=15)


# now since we confirmed our hypothesis it is easy we can just go and see the sports that were in and out of the olympics i'm just too lazy to do it
# 

# In[ ]:


normal_summer['City'].nunique()


# In[ ]:


cities=pd.pivot_table(index='City',columns='Year',values='useless',data=normal_summer).fillna(0)


# In[ ]:


cities['Total']=0
for col in cities.columns:
    cities['Total']+=cities[col]
    


# In[ ]:


cities[cities['Total']>1]


# In[ ]:


cities[cities['Total']==1]


# ###### well this  comes quiet shcocking to me turned out that all the cities host the olympics at least twice 

# ##### the excpections are :
# <ol>
#     <li>Athens,Paris,Los angeles:4</li>
#     <li>London:6</li>
#    </ol>

# # Conclusion of this EDA
# <ol>
#     <li>the USA is the country with the most of the medals in the majority of the disciplines </li>
#     <li>at the beginning there was some adding and dropping to the olympics</li>
#     <li>usually the men have more medals in the olympics then women but in the last years the gap shrinked so maybe they will be the same or they are already because the last date is 2012</li>
#     <li>we may assume that men are participating much more then women but we're seeing this just by the medals which is not really acurate but this causation seems logical to seem extent but to be sure we need othe data </li>
#     <li>there is some sports includes only men but one 'Soft Ball' include only women</li>
#     <li>Some sports have men and women have almost the same (you can check the graph up to see them)</li>
#     <li>The majority of the medals are from Aquatics and Athletics sports</li>
#     <li>If a country is good at certain sport they are just good at it We can see this base on the correlation between the number of the type of the medals </li>
#     <li>There is NO stable number of sports in olympics the number is usually changing</li>
#     <li>in the 90's 1920 they were a lot of Sports in the olympics that's why we see this peak in the medals around 1920 and this kinda expalin the zigzag around 1920 Because 1920 we have 22 sports while before 1916 there were 0 becasue of the war and after 1924 there was 17 sports Also 1940 and 1940 the olympics were canceled because of the war this may kinda explain more the zigzag shape in general</li>
#     <li>There is a little Corelation between the country's GDP per Capita and the Total number of medals won by that country but it is not huge correlation (I was expecting more tbh)</li>
#     </ol>

# #### PS: all of this is based on the summer olympics maybe by looking at the winter olympics we see smthg diffrent or more meaningful but we also should notice that the winter data Frame contains much more less than the summer data Frame which would make us think that in worst cases it won't change a lot but this is just hypothesis we should test it 

# In[ ]:




