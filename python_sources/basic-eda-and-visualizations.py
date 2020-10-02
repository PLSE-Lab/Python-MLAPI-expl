#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[ ]:


#Let us first understand the data


# In[ ]:


df = df = pd.read_csv("../input/co2-and-ghg-emission-data/emission data.csv")


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.sample(5)


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


#Let us now see if there are any missing values


# In[ ]:


df.isnull().sum().sum() #There are no missing values but seems like there are lot of zero values.


# In[ ]:


df[df['1751']!= 0]


# Seems like United Kingdom made most of the emissions of world and europe for a long time or the data is only available for UK.We will plot a graph to visualize this better later.

# In[ ]:


#I will try to create descendng wise list for emmiters for year 2017


# In[ ]:


year_17 = df.groupby('Country').agg({'2017':'sum'}).sort_values(by = '2017',ascending = False)


# In[ ]:


year_17 #country-wise emmisions arranged descendingly


# In[ ]:


#Now I am transposing the dataFrame and do some basic data operations for my ease of work


# In[ ]:


df2 = df.T


# In[ ]:


df2 = df2.reset_index() 


# In[ ]:


df2.columns = df2.iloc[0]


# In[ ]:


df2 = df2.rename(columns = {'Country':'Year'})


# In[ ]:


df2.drop(0,inplace = True)


# In[ ]:


df2.Year.dtype #Year Converted to int


# In[ ]:


df2 = df2.astype('float')


# In[ ]:


#I have just converted dataframe to float type even though year column should propbably be int type or DateTime column.But float should make no difference.


# In[ ]:


#Now let us plot some graphs


# In[ ]:


plt.figure(figsize = (16,8))
sns.lineplot(x = df2['Year'],y = df2['World']).set_title('World emissions')


# So this plot tells us that world emissions started increasing drastically after about 1900 and before that the increase was very gradual.

# In[ ]:


#So now lets plot europe vs world emissions.Afterall Europe is where the first Industrial Evolution started.


# In[ ]:


plt.figure(figsize = (16,8))
sns.lineplot(x = df2['Year'],y = df2['World'],label = 'World')
sns.lineplot(x = df2['Year'], y = df2['EU-28'],label = 'Europe').set_title('World emissions')
plt.plot()


# Now this is interesting.Seems like  28 major countries of europe made majority of world emissions till about 1920-1930.After that the world emissions increased at higher pace than europe.

# In[ ]:


plt.figure(figsize = (16,8))
sns.lineplot(x = df2['Year'],y = df2['World'],label = 'World')
sns.lineplot(x = df2['Year'], y = df2['EU-28'],label = 'Europe').set_title('World emissions')
sns.lineplot(x = df2['Year'], y = df2['United Kingdom'],label = 'UK').set_title('World emissions')
sns.lineplot(x = df2['Year'], y = df2['France'],label = 'Spain').set_title('World emissions')
plt.plot()


# This helps us viusalize that United Kingdom made most of emissions of Europe and world till about  1890 and after that other countries outpaced it.

# Now let us visualize rise of nations like United states, china and Russia in terms of year-wise CO2 and greenhouse emissions.These were the greatest emitter countries in the world for 2017 as we saw above.

# In[ ]:


plt.figure(figsize = (16,8))
sns.lineplot(x = df2['Year'],y = df2['United States'],label = 'US')
sns.lineplot(x = df2['Year'], y = df2['China'],label = 'China')
sns.lineplot(x = df2['Year'],y = df2['Russia'],label = 'Russia')
plt.plot()


# US is way ahead than other 2 nations.Russia has  a irregular graph and its rate of increase drops after 1990s.This may or may not be due to break up of Soviet Union.

# Lets compare United states,Europe and world emissions.

# In[ ]:


plt.figure(figsize = (16,8))
sns.lineplot(x = df2['Year'],y = df2['United States'],label = 'US')
sns.lineplot(x = df2['Year'], y = df2['EU-28'],label = 'Europe')
sns.lineplot(x = df2['Year'],y = df2['World'],label = 'World')
plt.plot()

US overtook Europe in emissions after 1990s.From 1950 till 1990 they kept pace with each other. 
# Now lets create a chloropeth map.

# In[ ]:



import pycountry
def do_fuzzy_search(country):
    try:
        result = pycountry.countries.search_fuzzy(country)
        return result[0].alpha_3
    except:
        return np.nan

df['country_code'] = df["Country"].apply(lambda country: do_fuzzy_search(country))
df.head()


# In[ ]:


plt.figure(figsize = (16,8))
plot_df = df.dropna()
fig = px.choropleth(plot_df, locations="country_code",
                    color="2017",
                    hover_name="Country",
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()


# There are some missing country codes which we can input explicitly.Lets correct them

# In[ ]:


df[df['country_code'].isnull() == True]['Country']

These are the codes missing from our data.Some of them are countries and others are not.So Now we will explicitly enter code for these countries.
# In[ ]:


correct_codes = {"Congo": "COD", "Democratic Republic of Congo": "COG", "Niger": "NER", "South Korea": "KOR"}


# In[ ]:


def update_wrong_country_codes(row):
    if row['Country'] in correct_codes.keys():
        row['country_code'] = correct_codes[row['Country']]
    return row

df = df.apply(lambda x: update_wrong_country_codes(x), axis=1)

We have input the missing values explicitly.So now let us plot again
# In[ ]:


plot_df = df.dropna()
fig = px.choropleth(plot_df, locations="country_code",
                    color="2017",
                    hover_name="Country",
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()


# In[ ]:





# In[ ]:





# ##Now let us try to consider commulative emmisions for some major countries  

# In[ ]:


cummulative_em = df2.set_index('Year')
cummulative_em = cummulative_em.cumsum()
cummulative_em = cummulative_em.reset_index()


# In[ ]:


sns.lineplot(x = cummulative_em['Year'], y = cummulative_em['World'],label = 'world')
sns.lineplot(x = cummulative_em['Year'], y = cummulative_em['United States'],label ="US")
sns.lineplot(x = cummulative_em['Year'], y = cummulative_em['EU-28'],label = "Europe")
plt.show()

Now this is interesting.US is behind in cummulative CO2 emmisions than Europe even though currently it is biggest emmiter for year 2017.This just shows Europe started in Industrial evolution much before US.Also keep in mindwe are only comparing 28 nations of Europe.
# ### Now let us compare these nations with nations like China,India and russia

# In[ ]:



sns.lineplot(x = cummulative_em['Year'], y = cummulative_em['United States'],label ="US")
sns.lineplot(x = cummulative_em['Year'], y = cummulative_em['EU-28'],label = "Europe")
sns.lineplot(x = cummulative_em['Year'], y = cummulative_em['China'],label = "China")
sns.lineplot(x = cummulative_em['Year'], y = cummulative_em['Russia'],label = "Russia")
sns.lineplot(x = cummulative_em['Year'], y = cummulative_em['India'],label = "India")
plt.show()

Seems like Developing countries have emmited nowhere near as much cummulatively as US or Europe.Now lets try one thing.lets add cummulative emmisions of all BRICS countries and compare with EU-28 and US
# In[ ]:


cummulative_em['BRICS'] = cummulative_em['Brazil']+cummulative_em["China"]+cummulative_em['India']+cummulative_em['Russia']+cummulative_em['South Africa']


# In[ ]:


sns.lineplot(x = cummulative_em['Year'], y = cummulative_em['United States'],label ="US")
sns.lineplot(x = cummulative_em['Year'], y = cummulative_em['EU-28'],label = "Europe")
sns.lineplot(x = cummulative_em['Year'], y = cummulative_em['BRICS'],label = "BRICS")


# In[ ]:





# In[ ]:




