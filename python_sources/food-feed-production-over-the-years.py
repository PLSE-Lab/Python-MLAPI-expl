#!/usr/bin/env python
# coding: utf-8

# ## CONTEXT
# This dataset provides an insight on our worldwide food production - focusing on a comparison between food produced for human consumption and feed produced for animals.

# ## CONTENT
# 
# The Food and Agriculture Organization of the United Nations provides free access to food and agriculture data for over 245 countries and territories, from the year 1961 to the most recent update (depends on the dataset). One dataset from the FAO's database is the Food Balance Sheets. It presents a comprehensive picture of the pattern of a country's food supply during a specified reference period, the last time an update was loaded to the FAO database was in 2013. The food balance sheet shows for each food item the sources of supply and its utilization. This chunk of the dataset is focused on two utilizations of each food item available:
# 
# Food - refers to the total amount of the food item available as human food during the reference period.
# Feed - refers to the quantity of the food item available for feeding to the livestock and poultry during the reference period.

# ## DATA SET
# 
# The Food Balance sheet's data was relatively complete. A few countries that do not exist anymore, such as Czechoslovakia, were deleted from the database. Countries which were formed lately such as South Sudan were kept, even though they do not have all full data going back to 1961. In addition, data aggregation for the 7 different continents was available as well, but was not added to the dataset.
# 
# Food and feed production by country and food item from 1961 to 2013, including geocoding.
# Y1961 - Y2011 are production years that show the amount of food item produced in 1000 tonnes

# | Area Abbreviation | Country name abbreviation       | String  |
# |-------------------|---------------------------------|---------|
# | Area Code         | Country code                    | Numeric |
# | Area              | Country name                    | String  |
# | Item Code         | Food item code                  | Numeric |
# | Item              | Food item                       | String  |
# | Element Code      | Food or Feed code               | Numeric |
# | Element           | Food or Feed                    | String  |
# | Unit              | Unit of measurement             | String  |
# | latitude          | Help us describe this column... | Numeric |
# | longitude         | Help us describe this column... | Numeric |
# | Y1961             | Help us describe this column... | Numeric |
# | Y1962             | Help us describe this column... | Numeric |
# | Y1963             | Help us describe this column... | Numeric |
# | Y1964             | Help us describe this column... | Numeric |
# | Y1965             | Help us describe this column... | Numeric |
# | Y1966             | Help us describe this column... | Numeric |
# | Y1967             | Help us describe this column... | Numeric |
# | Y1968             | Help us describe this column... | Numeric |
# | Y1969             | Help us describe this column... | Numeric |
# | Y1970             | Help us describe this column... | Numeric |
# | Y1971             | Help us describe this column... | Numeric |
# | Y1972             | Help us describe this column... | Numeric |
# | Y1973             | Help us describe this column... | Numeric |
# | Y1974             | Help us describe this column... | Numeric |
# | Y1975             | Help us describe this column... | Numeric |
# | Y1976             | Help us describe this column... | Numeric |
# | Y1977             | Help us describe this column... | Numeric |
# | Y1978             | Help us describe this column... | Numeric |
# | Y1979             | Help us describe this column... | Numeric |
# | Y1980             | Help us describe this column... | Numeric |
# | Y1981             | Help us describe this column... | Numeric |
# | Y1982             | Help us describe this column... | Numeric |
# | Y1983             | Help us describe this column... | Numeric |
# | Y1984             | Help us describe this column... | Numeric |
# | Y1985             | Help us describe this column... | Numeric |
# | Y1986             | Help us describe this column... | Numeric |
# | Y1987             | Help us describe this column... | Numeric |
# | Y1988             | Help us describe this column... | Numeric |
# | Y1989             | Help us describe this column... | Numeric |
# | Y1990             | Help us describe this column... | Numeric |
# | Y1991             | Help us describe this column... | Numeric |
# | Y1992             | Help us describe this column... | Numeric |
# | Y1993             | Help us describe this column... | Numeric |
# | Y1994             | Help us describe this column... | Numeric |
# | Y1995             | Help us describe this column... | Numeric |
# | Y1996             | Help us describe this column... | Numeric |
# | Y1997             | Help us describe this column... | Numeric |
# | Y1998             | Help us describe this column... | Numeric |
# | Y1999             | Help us describe this column... | Numeric |
# | Y2000             | Help us describe this column... | Numeric |
# | Y2001             | Help us describe this column... | Numeric |
# | Y2002             | Help us describe this column... | Numeric |
# | Y2003             | Help us describe this column... | Numeric |
# | Y2004             | Help us describe this column... | Numeric |
# | Y2005             | Help us describe this column... | Numeric |
# | Y2006             | Help us describe this column... | Numeric |
# | Y2007             | Help us describe this column... | Numeric |
# | Y2008             | Help us describe this column... | Numeric |
# | Y2009             | Help us describe this column... | Numeric |
# | Y2010             | Help us describe this column... | Numeric |
# | Y2011             | Help us describe this column... | Numeric |
# | Y2012             | Help us describe this column... | Numeric |
# | Y2013             | Help us describe this column... | Numeric |

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from mpl_toolkits.basemap import Basemap
import itertools


# # Loading the Data & Intial Inspection

# In[2]:


df=pd.read_csv('../input/FAO.csv',encoding='ISO-8859-1')


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# In[6]:


df.sample(10)


# In[7]:


df.dropna().shape


# In[8]:


df.shape


# In[9]:


df=df.fillna(0)


# In[10]:


df.sample(5)


# In[11]:


list(df.columns)[:10]


# In[12]:


Years=list(df.columns)[10:]


# In[13]:


Countries=list(df.Area.unique())


# In[14]:


df['latitude'][df.Area=='Japan'].unique()


# In[15]:


df_Element=pd.concat([df.loc[:,'Element'],df.loc[:,Years]],axis=1)


# In[16]:


Food_Type=df_Element.groupby('Element')


# In[17]:


list(Food_Type.Y2013)


# # Relationship of Food & Feed over the Years

# In[18]:


plt.figure(figsize=(20,15))
sns.heatmap(Food_Type.corr())


# The feed and food production seems to be highly correlated to the succeeding year.

# In[19]:


tot = Food_Type.sum()


# In[20]:


tot.apply(lambda x: x/x.sum()*100)


# The production of all feed and food seems to be broken into range of food being 74% to 80% whereas feed would be 26% to 20% throughout the years

# In[21]:


for name, group in Food_Type:
    print(name,group['Y1961'].sum())


# In[22]:


pd.DataFrame([Food_Type[year].sum() for year in Years]).plot(kind='bar',figsize=(20,10),color=('rg'),fontsize=14,width=.95,alpha=.5)
plt.yticks(np.arange(0,1.05*10**7,5*10**5))
plt.ylabel('Production in 1000 tonnes')
plt.title('Food & Feed vs Years',fontsize=14)
plt.show()


# The above graph shows the growth of food and feed over the years. Production of food and feed got a spurt of growth in 1992,where the production increased by about 6 x 10^5 tonnes for food and 5 x 10^5  compared to 1991. Growth of food production is much more drastic since 1992, compared to feed production.

# # Basemap Plots for Production of Food & Feed

# In[23]:


print('min longitude is',min(df.longitude))
print('max longitude is',max(df.longitude))
print('min latitude is',min(df.latitude))
print('max latitude is',max(df.latitude))


# In[24]:


q=df.groupby(['Element','Area']).sum().loc[:,Years]


# In[25]:


q=q.reset_index()


# In[26]:


q['latitude']=q.apply(lambda row: df['latitude'][df.Area==row['Area']].unique()[0],axis=1)
q['longitude']=q.apply(lambda row: df['longitude'][df.Area==row['Area']].unique()[0],axis=1)


# In[27]:


q


# In[28]:


def food_map(lon,lat,df,year):
    fig,ax = plt.subplots()
    fig.set_size_inches(12,20)
    plt.gca().set_color_cycle(['crimson','blue'])
    
    m = Basemap(projection='merc', llcrnrlat=-60, urcrnrlat=80,             llcrnrlon=-180, urcrnrlon=180,             lat_ts=20,             resolution='c')
    
    x,y=m(lon, lat)
        
    m.fillcontinents(color='white', alpha =.1) # add continent color
    m.drawcoastlines(color='black', linewidth=0.2)  # add coastlines
    m.drawmeridians(np.arange(-180,180,30),color='lightgrey',alpha=.6,labels=[0,0,0,1], fontsize=10) ## add latitudes
    m.drawparallels(np.arange(-60,100,20),color='lightgrey',alpha=.6,labels=[1,0,0,0], fontsize=10) # add longitude
    
    marker = itertools.cycle(('X','o'))

    for ele,m in zip(['Feed','Food'],marker):
        ax.scatter(x, y, marker =m, label=ele, s=q[year][q.Element==ele]/500 ,alpha=0.2)
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.,fontsize=10)
        plt.title(year,y=1.055)

    return plt.show()


# In[29]:


ax=[food_map(list(q.longitude),list(q.latitude),q,year) for year in ['Y1961','Y1971','Y1981','Y1991','Y2001','Y2011','Y2013']]


# In[ ]:





# The food and feed production has increased over the years.
# The major production happend between Tropic of Cancer and Tropic of Capricorn.
# 
# Production of food dominated by China,India and USA from 1961 onwards. Production of Feed has mainly been through China & USA , interesting to note that Brazils and India feed production has grown drastically from 1991.
# 
# African countries that are producing feed and food has increased over the years. Countries like Australia and Russia has not seem much growth of production of either feed or food since 2001. From 2001, onwards there seem to be more European countries involved in production of food and feed.
# 

# # Top Countries for Food & Feed Production

# In[30]:


largest_feed=pd.DataFrame([df[df.Element=='Feed'].groupby(['Area'])[year].sum().nlargest(5) for year in Years])


# In[31]:


largest_feed=largest_feed.fillna(0)


# In[32]:


largest_feed.head()


# In[33]:


plt.figure(figsize=(15,10))
plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow','purple','grey','orange','lightblue','violet','black'])
plt.plot(Years,largest_feed)
plt.xticks(Years, rotation=-300, fontsize=12)
plt.yticks(np.arange(0,710000,25000))
plt.xlabel('Years')
plt.ylabel('Production in 1000 tonnes')
plt.title('Countries Producing the Most Feed')
plt.legend(labels=largest_feed.columns, loc='best',fancybox=True,fontsize=14)
plt.show()


# There is high probability of feed production related previous years production.

# In[34]:


plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
cmap = sns.cubehelix_palette(25, start=.2, rot=-.5) ## colour maps
sns.heatmap(largest_feed,cmap=cmap,linecolor='w')


# China production has increased continously and produced over 60000. USA produce the second most but seems to produce a stable production throughout its years.

# In[35]:


largest_food=pd.DataFrame([df[df.Element=='Food'].groupby(['Area'])[year].sum().nlargest(5) for year in Years])
largest_food=largest_food.fillna(0)
largest_food.head()


# In[36]:


plt.figure(figsize=(18,12))
plt.rc('axes',prop_cycle=(cycler(color=['red', 'green', 'blue', 'pink','purple','maroon','orange','lightblue','violet','black'])))
plt.plot(Years,largest_food)
plt.xlabel('Years')
plt.ylabel('Production in 1000 tonnes')
plt.title('Countries Producing the Most Food')
plt.xticks(Years, rotation=-300,fontsize=12)
plt.yticks(np.arange(0,2500000,50000))
plt.legend(labels=largest_food.columns, loc='best',fancybox=True,fontsize=14)
plt.show()


# In[37]:


plt.figure(figsize=(10,10))
sns.set(font_scale=1.25)
cmap = sns.cubehelix_palette(25, start=.2, rot=-.5) ## colour maps
sns.heatmap(largest_food,cmap=cmap,linecolor='w')


# China production of food can be seen to have increased continously from 1961 to 2013, followed with similar pattern by india. USA production seems to be more stable output.

# # Growth of Items vs Years

# In[38]:


pd.DataFrame([df.groupby(['Item'])[year].sum().nlargest(10) for year in Years])


# In[39]:


df_Item=pd.DataFrame([df.groupby(['Item'])[year].sum().nlargest(10) for year in Years])
df_Item=df_Item.fillna(0)


# In[40]:


## cycle through colour and line styles
cycle=cycler('linestyle', ['-', '--', ':', '-.'])*cycler('color',['r', 'g', 'b', 'y', 'c', 'k'])


# In[41]:


plt.figure(figsize=(20,10))
plt.rc('axes',prop_cycle=cycle)
plt.plot(Years,df_Item)
plt.xticks(rotation=300,fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Years')
plt.ylabel('Production in 1000 tonnes')
plt.title('Top 10 Food & Feed Items produced vs Years')
plt.legend(labels=df_Item.columns, loc='best',fancybox=True,fontsize=14)


plt.show()


# The most produced food & feed item continously is Cereals and Milk.Till 1993, next most produced item was Startchy Roots but it was displaced by vegetables

# In[42]:


df_Item_Feed=pd.DataFrame([df[df.Element=='Feed'].groupby(['Item'])[year].sum().nlargest(10) for year in Years])
df_Item_Feed=df_Item_Feed.fillna(0)


# In[43]:


df_Item_Feed.head()


# In[44]:


cycle=cycler('linestyle', ['-', '--', ':', '-.'])*cycler('color',['r', 'g', 'b', 'y', 'c', 'k'])


# In[45]:


plt.figure(figsize=(20,15))
plt.rc('axes',prop_cycle=cycle)
plt.plot(Years,df_Item_Feed)
plt.xticks(rotation=300,fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Years')
plt.ylabel('Production in 1000 tonnes')
plt.title('Top 10 Feed Items produced vs Years')
plt.legend(labels=df_Item_Feed.columns, loc='best',fancybox=True,fontsize=14)

plt.show()


# The most produced feed item continously is Cereals and Maize. Milk and Starchy roots is at 3rd and 4th position over the years, interchanging positions continously.

# In[46]:


df_Item_Food=pd.DataFrame([df[df.Element=='Food'].groupby(['Item'])[year].sum().nlargest(10) for year in Years])
df_Item_Food=df_Item_Food.fillna(0)


# In[47]:


cycle=cycler('linestyle', ['-', '--', ':', '-.'])*cycler('color',['r', 'g', 'b', 'y', 'c', 'k'])


# In[48]:


plt.figure(figsize=(20,10))
plt.rc('axes',prop_cycle=cycle)
plt.plot(Years,df_Item_Food)
plt.xticks(rotation=300,fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Years')
plt.ylabel('Production in 1000 tonnes')
plt.title('Top 10 Food Items produced vs Years')
plt.legend(labels=df_Item_Food.columns, loc='best',fancybox=True,fontsize=14)

plt.show()


# The top 2 most produced food items are Milk and Cereal through out the years. Starchy roots were 3rd most produced item till 1978, after which point production of vegetables increased and takes over the position which was previously at 4th position.
# 
# For production of Wheat which had spike in production in 1992 but then seem to have stable production then on. Also to note fruit production show continously increasing production and in 2013 is the 4th most produced food in 2013

# In[49]:


print('No of items produced over the year are at',len(df.Item.unique()))


# # Percentage change of Production of Top Items in 2013 over the Years
# 
# The below graphs show how the top ten products of 2013 has varied as percentage over the decades compared to the production volume in 2013.

# In[50]:


#Attach a text label above each bar displaying its height
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%.2f' %float(height),
                ha='center', va='bottom')


# In[51]:


## create function to plot the percentage increase of items between 2 years (Year1, Year2). Element is either 'Food' or 'Feed'

def percentage_increase_plot(Element,Year1,Year2):
    if Element=='Food':
        a=[df_Item_Food.loc[Year1,i] for i in df_Item_Food.loc[Year2].index]
        b=[i for i in df_Item_Food.loc[Year2]]
        Percent_Growth=pd.DataFrame([df_Item_Food.loc[Year2].index,[(y-x)/y*100 for x,y in zip(a,b)]]).transpose()
        Percent_Growth.columns=['Item','Percentage_increase']
        Percent_Growth=Percent_Growth[~Percent_Growth.isin([np.inf,-np.inf])] ## returns inf & -inf as NaN with isin & ~ returns the df that satisfies the condition
    elif Element=='Feed':
        a=[df_Item_Feed.loc[Year1,i] for i in df_Item_Feed.loc[Year2].index]
        b=[i for i in df_Item_Feed.loc[Year2]]
        Percent_Growth=pd.DataFrame([df_Item_Feed.loc[Year2].index,[(y-x)/y*100 for x,y in zip(a,b)]]).transpose()
        Percent_Growth.columns=['Item','Percentage_increase']
        Percent_Growth=Percent_Growth[~Percent_Growth.isin([np.inf,-np.inf])]
    
    Percent_Growth=Percent_Growth.fillna(0)
    ## drop rows wirh 0% increase
    Percent_Growth=Percent_Growth.drop(Percent_Growth[Percent_Growth.Percentage_increase==0].index)
    x=Percent_Growth.Item
    y=Percent_Growth.Percentage_increase
    
    plt.figure(figsize=(15,5))
    autolabel(plt.bar(x,y,color=['darkblue','crimson']))
    plt.title(" ".join([Element,'Percentage Increase from',Year1[1:],'to',Year2[1:]]))
    plt.xlabel(' '.join(['Top 10 Items in',Year2[1:]]))
    plt.ylabel(' '.join(['Percentage Increase from',Year1[1:]]))
    plt.xticks(rotation=330,fontsize=10)
    plt.yticks(fontsize=14)
    


# In[52]:


for year in ['Y1961','Y1971','Y1981','Y1991','Y2001','Y2011','Y2012']:
    percentage_increase_plot('Food',year,'Y2013')
    
plt.show()


# All items have increased by minimum of 54% from 1961 to 2013. More interesting is that there is more than 20% increase in 6 of top ten item from 2001 and 3 items that has increased in volume by more than 10% compared to 2001.
# 

# In[53]:


for year in ['Y1961','Y1971','Y1981','Y1991','Y2001','Y2011','Y2012']:
    percentage_increase_plot('Feed',year,'Y2013')
    plt.show()


# Interesting Pototoes and it product has 15% drop in 2013 compared to 1961. Compared to 2001, Vegetable and Vegetable,Others items have increased by 100%, Cassava & Maize and their products have increased by more than 24% and 3 of remaining items have over 10%.
# 
# From 2012 to 2013, the most increased items of over 10%  is Barley and Maize. Except for Wheat and Startchy roots which has 1.22% and 3.08% increase rest have increased by minimum of 4.3%.

# In[ ]:




