#!/usr/bin/env python
# coding: utf-8

# **LITERACY IN CIRCLES**

# For this exploration of literacy, we will be exploring the influence literacy has on infant mortality rates, the birthrate, the deathrate, the number of phones per 1000 people, and the GDP per capita amongst different countries in various regions of the globe.  

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
print("Good to go")


# In[ ]:


#Loading the data ready for examination

file_pathway = "../input/countries-of-the-world/countries of the world.csv"
data = pd.read_csv(file_pathway, decimal=',')
data.columns = (["Country","Region","Population","Area","Density","Coastline","Migration","IMR","GDP","Literacy","Phones",
                 "Arable","Crops","Other","Climate","Birthrate","Deathrate","Agriculture","Industry","Service"])

data.Country = data.Country.astype('category')
data.Region = data.Region.astype('category')
data.Population = data.Population.astype('int')
data.IMR = data.IMR.astype(float)
data.Literacy = data.Literacy.astype(float)
data.Phones = data.Phones.astype(float)
data.Birthrate = data.Birthrate.astype(float)
data.Deathrate = data.Deathrate.astype(float)


# In[ ]:


# Selecting our attributes of interest

subset=data[['Country','Region','Population','IMR','GDP','Literacy','Phones','Birthrate','Deathrate']]


# In[ ]:


# All missing values for our attributes of interest

subset.isna().sum()


# There are 18 countries in the dataset with missing values for our independent attribute of interest: literacy.
# These countries will be excluded from our visualizations. 
# 
# *Note: The country of Western Sahara is missing all the data of interest for our exploration.*  

# In[ ]:


# All tuples in our database with missing values for literacy

unknown_literacy=pd.isnull(subset["Literacy"])
subset[unknown_literacy]


# Let's begin with examining the distribution of our primary attribute of interest.
# In plotting the **density of literacy**, we see that more countries in the database are literate than not, given the distribution is skewed to the right. 
# This illustrates that the majority of the world is indeed literate. Given this reality, what implication does a high literacy in the population have for a country? 

# In[ ]:


plt.figure(figsize=(10,5))
sns.kdeplot(data=subset['Literacy'], shade=True)


# **Countries with the Lowest Literacy Rates**
# 
# In terms of literacy, the bottom 10 ranking countries represents 4.41% of all 227 countries listed in the dataset. However of the 10 lowest literacy rankings, countries in the Sub-Saharan Africa region account for 80% of this list. 
# 
# Consider: what obstacles exist in nations of this region where literacy struggles to thrive to create upward postive social changes?   
# 
# In this dataset, the countries with the lowest rates of literacy (lower than 40%) are Niger, Burkina Faso, Sierra Leone, Guinea, Afghanistan and Somalia. All these countries reside in the Sub-Sahara Africa region with the exception of Afghanistan. 

# In[ ]:


low_lit=subset.sort_values(by='Literacy', ascending=True)
low_lit.head(10)


# **Countries with the Highest Literacy Rates**
# 
# On the other end of the spectrum, the region of Western Europe occupies 70% of of the top 10 list of countries with the hightest literacy rates, with 6 of these countries having a 100% literacy rate. 
# 
# Consider: what allows for such wonderful literacy rates in this region?

# In[ ]:


high_lit=subset.sort_values(by='Literacy', ascending=False)
high_lit.head(10)


# **Infant Mortality Rate over Literacy**
# 
# Plotting the IMR (per 1000 births) over literacy, there is an apparent negative correlation relationship. 
# Countries with higher IMR also seem to have lower literacy rates and conversely, countries with higher literacy rates experience lower rates of infant mortality. 
# The influence of literacy on a country's IMR is greatly impactful when we consider the population sizes of these nations and what influence improving literacy lends to the well-being of that country's future. For example, consider India, with the world's second largest population and approximately 60% literacy rate with an IMR of 56.29, in comparison to China with the world's largest population and literacy rate of 90.9 and IMR of 24.18.
# 
# Also to give thought to are countries with high IMRs and low literacy rates. What obstacles (conflict, famine, healthcare etc.) are present in the populations of these regions that are adversly affecting literacy and as a result perhaps influencing IMRs? 

# In[ ]:


fig = px.scatter(subset, x="Literacy", y="IMR", size="Population", color="Region",
           hover_name="Country", log_x=True, log_y=True, size_max=100, width=950, height=600)
fig.update_layout(showlegend=False,
    xaxis={'title':'Literacy Rate (%age)',},
    yaxis={'title':'Infant Moratality (per 1000 births)'})
fig.show()


# ****Highest IMR countries****
# 
# *Angola, Afghanistan, Sierra Leone, Mozambique and Liberia.*

# In[ ]:


subset.nlargest(5,'IMR') 


# **Lowest IMR countries**
# 
# *Singapore, Sweden, Hong Kong, Japan and Iceland.* 

# In[ ]:


subset.nsmallest(5,'IMR') 


# **Birthrate over Literacy**
# 
# Plotting birthrate over literacy, there exists a negative correlation relationship. It seems that as a country increases their literacy rate, the birthrate experiences a reduction. What does this mean? One suggestion is perhaps populations are able to be more literate being freed of parental responsibilities. Consider China with the largest population, literacy rate of 90.9% and birthrate of 13.25, which had been given a policy restricting birthrate. Did this policy lend to the improving the literacy rate in later years? In comparison, the United States which did not experience such implementations has a strong literacy rate of 97% with a low birthrate of 14.14.
# Literacy increases critical thinking and perhaps this results in delaying the creation of families due to work commitments. 
# 
# The top 5 countries in the dataset with the lowest birthrate all have literacy rates over 93.5% (Hong Kong rate). Interesting to think about is how the impact of affairs between Hong Kong and China may translate to the country's literacy rates in the years ahead, with or without birthrate experiencing change. 

# In[ ]:


fig = px.scatter(subset, x="Literacy", y="Birthrate", size="Population", color="Region",
                 hover_name="Country", log_x=True, log_y=True, size_max=100, width=950, height=600)
fig.update_layout(showlegend=False, 
    xaxis={'title':'Literacy Rate (%age)',},
    yaxis={'title':'Birthrate (per 1000)'})
fig.show()


# **Countries with the highest Birthrates**
# 
# *Niger, Mali, Uganda, Afghanistan and Sierra Leone.*

# In[ ]:


subset.nlargest(5,'Birthrate')


# **Countries with the lowest birthrates**
# 
# *Hong Kong, Germany, Macau, Andorra and Italy.*

# In[ ]:


subset.nsmallest(5,'Birthrate')


# **Deathrate over Literacy**
# 
# Plotting the deathrate over literacy it becomes evident that the region of Sub-Saharan Africa dominates with the highest deathrates globally. 
# This is the case even with countries of this region with higher literacy rates than India (59.5%) including Kenya, South Africa and Zimbabwe (with literacy rates of 85.1%, 86.4% and 90.7% respectively). 
# 
# Consider: what could be happening here that countries with relatively higher literacy rates are still experiencing higher death rates? Perhaps literacy in combination with a host of other factors (healthcare accessibility, sanitation and community resources etc.) can lower deathrates in a population. 

# In[ ]:


fig = px.scatter(subset, x="Literacy", y="Deathrate", size="Population", color="Region",
                 hover_name="Country", log_x=True, log_y=True, size_max=100, width=950, height=600)
fig.update_layout(showlegend=False,
    xaxis={'title':'Literacy Rate (%age)',},
    yaxis={'title':'Deathrate (per 1000) '})
fig.show()


# **Countries with the highest Deathrates**
# 
# *Swaziland, Botswana, Lesotho, Angola and Libera. 

# In[ ]:


subset.nlargest(5,'Deathrate')


# **Countries with the lowest Deathrates**
# 
# *Northern Mariana Islands, Kuwait, Saudia Arabia, Jordan and American Samoa.*

# In[ ]:


subset.nsmallest(5,'Deathrate')


# **Phones over Literacy**
# 
# Plotting phones (per 1000 people) over literacy, a positive relationship is evident as countries with higher rates of literacy also have more phones. 
# The region of Sub-Saharan Africa is most impacted with lowest literacy rates and least phones in the population. Niger which has the lowest literacy rate in the dataset of 17.6, also has 1.9 phones per 1000 people, or the third lowest of all countries in the dataset. Conversely, Monaco of Western Europe with a literacy rate of 99%, has 1035.6 phones per 1000 people, indicating that everyone owns a phone. 
# Does this suggest that countries with higher literacy rates cause more phone purchases in the population? No, we cannot infer such causation, however it is evident that a positive correlation coincides between these two attributes. We must consider a host of other factors such as GDP per capita, disposable income and telecommunications infrastructure also influencing the phone purchases of a country's population. 
# 
# Note: *metadata beyond phones per 1000 people is unavailable in the dataset, therefore we don't know if this is exclusively mobile phones or a combination of landlines and cellular.* 
# 
# 

# In[ ]:


fig = px.scatter(subset, x="Literacy", y="Phones", size="Population", color="Region",
                 hover_name="Country", log_x=True, log_y=True, size_max=100, width=950, height=600)
fig.update_layout(showlegend=False,
    xaxis={'title':'Literacy Rate (%age)',},
    yaxis={'title':'Phones (per 1000)'})
fig.show()


# **Countries with most phones (per 1000 people)**
# 
# *Monaco, United States, Gibraltar (literacy rate not available), Bermuda and Guernsey (literacy rate not available).*

# In[ ]:


subset.nlargest(5,'Phones')


# **Countries with least phones (per 1000 people)**
# 
# *Democratic Republic of Congo, Chad, Niger, Central African Republic and Liberia (also listed in top 5 IMR countries). *

# In[ ]:


subset.nsmallest(5,'Phones')


# **GDP over Literacy**
# 
# Lastly, plotting GDP per capita over Literacy illustrates a positive correlation relationship. Countries with higher rates of literacy benefit with a higher GDP per capita numbers. 
# A clear illustration of this is Luxembourg, which has the highest literacy rate achievable of 100%, while also capturing the highest GDP per capita in the dataset. 
# 
# In fact, the top 5 highest GDP per capita countries, all have literacy rates of 97% or more.

# In[ ]:


fig = px.scatter(subset, x="Literacy", y="GDP", size="Population", color="Region",
           hover_name="Country", log_x=True,log_y=True, size_max=100, width=950, height=600)
fig.update_layout(showlegend=False,
    xaxis={
        'title':'Literacy Rate (%age)',},
    yaxis={'title':'GDP per capita'})

fig.show()


# **Highest GDP per capita Countries**
# 
# *Luxembourg, Norway, United States, Bermuda and Cayman Islands*

# In[ ]:


subset.nlargest(5,'GDP')


# ***Lowest GDP per capita Countries****
# 
# *East Timor, Sierra Leone, Somalia, Burundi and Gaza Strip (literacy rate not available)*

# In[ ]:


subset.nsmallest(5,'GDP')

