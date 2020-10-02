#!/usr/bin/env python
# coding: utf-8

# In[7]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


file = '../input/Indicators.csv'
data = pd.read_csv(file,sep=',')


# # Growth of SADC countries in the midst of globalization and a services bound market (Part 1)
# 
# __Author: Keabetsoe Emmanuel Mosito__
# 
# ### Introduction
# 
# __Sub-Saharan Africa is plagued by a myriad of socio-economic problems and epidemics chief among them being unemployment. It is a region of the world that is still relatively less developed than other regions. But it is also considered the next frontier in investments (emerging markets). The more economically developed countries are shifting that economies from manufacturing economies to service providing economies. In this project a preliminary exploration of the progress of SADC countries in light of the continuously connecting and automating world is attempted. The general state of the economies and their trade capabilities are explored in an attempt to see where automation would leave these economies.__
# 
# Comments and suggestion are welcome...in fact encouraged. This is my first attempt.

# In[3]:


# List of all SADC countries for use in filtering
countries = ['Lesotho','South Africa','Zimbabwe','Botswana','Swaziland','Namibia','Zambia','Mozambique','Seychelles',
             'Tanzania','Angola','Congo','Malawi','Mauritius','Madagascar','SADC']

# A filtering mask to obtain a data frame of SADC countries
countries_mask = data['CountryName'].isin(countries)
sadc_countries = data[countries_mask] # The SADC countries data frame

# List for stylistic features of the graphs
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (230, 10, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 229, 141), (23, 190, 207), (158, 210, 229),
             (124,230,54), (34,100,85), (230,164,192), (12,33,122),(223,35,87),
             (111,233,68),(233,233,110),(110,110,43),(31,119,180),(31, 119, 180), (174, 199, 232),
            (255, 187, 120),(255,255,255),(245,245,245),(235,235,20),(225,225,40)]  

# change of the tuples above into RGB colours
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)

# Function to create a SADC wide average of the indicators to be explored
def sadc_stats(dataFrame):
    SADC = dataFrame.groupby('Year',as_index=False).mean()
    SADC['CountryName']='SADC'
    SADC['CountryCode']='SADC' # Both country name and country code are set as SADC
    SADC['IndicatorName']=dataFrame['IndicatorName'].iloc[0] # Add the indicator being explored to the SADC data frame 
    SADC['IndicatorCode']=dataFrame['IndicatorCode'].iloc[0]
    return SADC #return the new SADC data frame

# Function to print a line graph of the data for a list of countries
# The country names are passed in as a list
def print_data(dataFrame,alist):
    axis = plt.subplot(111)
    for country in alist:
        curr = dataFrame[dataFrame['CountryName']==country]
        if not curr.empty: #check to see if the country has data pertaining to the indicator being explored
            curr.plot(ax=axis,x='Year',y='Value',figsize=(8,6),marker='o',grid=True,label=country)
            axis.set_ylabel(dataFrame['IndicatorName'].iloc[0])
            axis.set_facecolor('k')
    plt.legend(loc='upper left',frameon=True,bbox_to_anchor=(1.05,1))
    plt.title(dataFrame['IndicatorName'].iloc[0])
    plt.show()

# Function to print an area graph of certain countries in a data frame
# The country names are passed in as a list
def print_dataA(dataFrame,alist):
    axis = plt.subplot(111)
    for country in alist:
        curr = dataFrame[dataFrame['CountryName']==country]
        if not curr.empty: #check to see if the country has data
            curr.plot.area(ax=axis,x='Year',y='Value',figsize=(8,6),grid=True,label=country)
            axis.set_ylabel(dataFrame['IndicatorName'].iloc[0])
            axis.set_facecolor('k')
    plt.legend(loc='upper left',frameon=True,bbox_to_anchor=(1.05,1))
    plt.title(dataFrame['IndicatorName'].iloc[0])
    plt.show()

# Function to print all the data of certain countries in a data frame
# The country names are passed in as a list
def printData(dataFrame,alist):
    ax = plt.subplot(111)
    i=0 # variable to help select colours for the line graph
    for country in alist:
        i+=2 # Incrementation of two so as to avoid similar colours
        curr = dataFrame[dataFrame['CountryName']==country]
        if not curr.empty: #check to see if the country has data
            curr.plot(ax=ax,figsize=(8,6),x='Year',y='Value',label=country,marker='o',color = tableau20[i],grid=True)
            ax.set_ylabel(dataFrame['IndicatorName'].iloc[0])
            ax.set_facecolor('k')


    plt.legend(loc='upper left',frameon=True,bbox_to_anchor=(1.05,1))
    plt.title(dataFrame['IndicatorName'].iloc[0])
    plt.show()


# In[5]:


#create a data frame of SADC countries filtered with the GDP (current LCU) data
indicator_mask = 'GDP (current LCU'
mask1 = sadc_countries['IndicatorName'].str.startswith(indicator_mask)
gdp_stage = sadc_countries[mask1]
SADC=sadc_stats(gdp_stage)
gdp_stage = gdp_stage.append(SADC)
gdp_stage = gdp_stage.sort_values('Year',ascending=True)


# ## __SADC DataFrame__
# __In the code above a new country was created call SADC and it was added to the GDP dataFrame.SADC represents the average of the whole region. the function sadc_stats is used tp create this average containig dataframe.__

# ## __GDP (LCU)__
# __Below the SADC country data is plotted with specific attention being given to their GDP at the Local currency unit (LCU).
# This is a helpful measure of the strength of the economy as it conveys how much the country's ouput is worth in its own currency.__

# In[8]:


#Print the GDP data
countries = gdp_stage['CountryName'].unique().tolist()
printData(gdp_stage,countries)


# __As can be seen in the figure above, Tanzania has the greatest GDP in its local currency. This may suggest the weakness of the Tanzanian currency or the quality of their output. Tanznia is followed by Madagascar. There is a very big disparity between the two countries. The table below shows the top 4 economies by LCU.__

# In[10]:


# Rank the countries in terms of GDP
high_gdp = gdp_stage.sort_values(['Year','Value'],ascending=False)
high_gdp.head()


# __In third place we have SADC. Thus in this case the average of the whole region ranks third out of all the countries. This mainly due to fact that Tanzania's GDP is extremely high compared to the other countries and thus drastically increases the region's average.__

# In[11]:


#print the data of the highest two countries by GDP (LCU) against the average of the region
h2countries = ['Tanzania','Madagascar','SADC']
print_dataA(gdp_stage,h2countries)
print_data(gdp_stage,h2countries)


# __The two best performing economies are plotted alongside the region average. The area plot gives us a better idea of the extent to which the Tanzania's GDP is growing over the years. As of 2014, its GDP measured in local currency was about 8 times the average of the region and more than two and a half times that of Madagascar. Next we turn our attentions to the two lowest performing economies. __

# In[12]:


high_gdp.head(15)


# In[13]:


#plot the data of the lowest two countries by GDP (LCU) against the region Average 
l2countries = ['Seychelles','Zimbabwe','SADC']
print_data(gdp_stage,l2countries)


# __As compared to the region average, it can be seen that Seychelles and Zimbabwe are seemingly practically stagnant. Mostly this illusion is created because of the scale of the GDP axis. each value is multiply by an order of magnitude of 13. Thus it would be more helpful the two country's datat was plotted separately.__

# In[14]:


# Plot the data of the lowest two ranking countries by GDP
l2countries = ['Seychelles','Zimbabwe']
print_dataA(gdp_stage,l2countries)
print_data(gdp_stage,l2countries)


# __From the graphs above we can see that the economies were truly not stagnant. It can also be seen that the Zimbabwean economy underwent some volatility from the 70's to 2000. This was mainly due to the sancations that were placed against the country as a result of The desire to force the President to step down. The graph also shows that the seychelles economy has been growing exponentially since the 60's. Which is remarkably considering it only has a population of about 95,000 people.__

# ## __GDP at market prices__
# __Next we explore the GDP's of these countries at market prices. This means that each country's GDP is converted to 2014 dollars. Again a SADC dataFrame is created to store the yearly averages.__

# In[16]:


# create a data frame of the country's GDP at market prices
indicator='GDP at market prices (curr'
mask = sadc_countries['IndicatorName'].str.startswith(indicator)
GDP_us = sadc_countries[mask]
SADC = sadc_stats(GDP_us)
GDP_us = GDP_us.append(SADC)
GDP_us = GDP_us.sort_values('Year',ascending=True)
#GDP_us.head(10)


# In[17]:


#plot the data
printData(GDP_us,countries)


# __The picture painted by this plot is somewhat astounding. In terms of US dollars The Tanzanian economy appears to be relatively flat. We however, have a new economic powerhouse (By SADC standards). South Africa's GDP is miles ahead of every other countries GDP. SInce this data is done in terms of an agreed upon standard (the US dollar), it is more reasonable to expect that South Africa actually has the biggest output in terms market value GDP and thus is the biggest economy. Below the values are sorted to Rank all the countries.__

# In[18]:


# Rank the countries by GDP at market price
max_gdp=GDP_us.sort_values(['Year','Value'],ascending=False)#.iloc[:2]
max_gdp.head(15)


# __In the table above South Africa is followed by Tanzania. However, close investigation of the matter will surely clarify the situation. Looking at the plot, one can see that before 2014 Angola was second. But it does not have a value for 2014. Also, it's GDP was a lot higher than Tanzania's GDP. Thus it would not be prudent to say that Tanzania has the second highest GDP in SADC. we could rather say that it has the third highest.__

# In[19]:


# Plot the data of the two highest countries by GDP against the region average
high_gdp_countries = ['South Africa','Angola','SADC']
print_dataA(GDP_us,high_gdp_countries)
print_data(GDP_us,high_gdp_countries)


# __The area plot above, helps put things into perspective. The South African economy is so massive it engulfs both the Angolan and SADC GDP.
# Below we plot the data of the two lowest countries by GDP. These are Lesotho and Seychelles.__

# In[20]:


# Plot the data of the lowest two countries against the region average
low_gdp_countries=['Lesotho','Seychelles','SADC']
print_data(GDP_us,low_gdp_countries)


# __When plotted with the SADC data, it is apparent that the average of the region dwarfs the output of this two countries. An area plot of such a disparity would take away from the plot above as we would'nt see the limits of the SADC plots (The axis would be green with orange and blue lines representing Lesotho and Seychelles as a proportion of the SADC data).__
# __Below we plot the two countries on their own. It can be seen that by 2014 Leostho's GDP had growth subtantially more then Seychelles GDP despite the fact that in the 60's they had almost the same GDP.__

# In[21]:


# Plot the data of the lowest two countries 
low_gdp_countries=['Lesotho','Seychelles']
print_dataA(GDP_us,low_gdp_countries)
print_data(GDP_us,low_gdp_countries)


# ## __Service imports and exports__
# __As mentioned earlier the more developed world is going towards a services market. Thus in order for the SADC countries to catch up with these countries they too would have to start exporting services that the bigger economies would need. Thus below we begin an exploartion of the service sectors in the SADC countries to see how substantial services are to their economies currently.__
# 
# __We begin firstly by looking at Trade in services as a percentage of GDP__

# In[23]:


# create a data frame of the countries using the Trade in Services indicator
indicator = 'Trade in services'
mask = sadc_countries['IndicatorName'].str.startswith(indicator)
trade_in_services = sadc_countries[mask]
SADC = sadc_stats(trade_in_services)
trade_in_services = trade_in_services.append(SADC)
trade_in_services = trade_in_services.sort_values('Year',ascending=True)
#trade_in_services.head()


# In[24]:


# Plot the data
printData(trade_in_services,countries)


# __From the graph above it is clear that Seychelles substantially relies on services for its output. Trade in services accounts for 94% of its GDP. It is followed by Mauritius at 47%__

# In[25]:


# Rank the countries by Trade in services
high_trade = trade_in_services.sort_values(['Year','Value'],ascending=False)
high_trade.head(20)


# In[26]:


# Plot the highest two country's data against the region average
high_trade = ['Seychelles','Mauritius','SADC']
print_dataA(trade_in_services,high_trade)
print_data(trade_in_services,high_trade)


# __It is interesting to note that services have continually increased in importance in the SADC region since 2005. The region average has gone from a little over 20% to 40% in the last 10 years.__

# In[27]:


# plot the lowest two country's data
low_trade = ['South Africa','Zambia','SADC']
print_data(trade_in_services,low_trade)


# __Zambia and South Africa are the two lowest countries using this metric. In their economies trade in services accounts for less than 10% of their output.__
# 
# __Next we see how the countries stack in terms of service imports. This measure is done monetarily.__

# In[28]:


# Create a data frame with the service imports indicator
indicator='Service imports (BoP'
service_imports = sadc_countries[sadc_countries['IndicatorName'].str.startswith(indicator)]
SADC = sadc_stats(service_imports)
service_imports = service_imports.append(SADC)
service_imports = service_imports.sort_values('Year',ascending=True)
#service_imports.head()


# In[29]:


# plot the data
printData(service_imports,countries)


# __Angola leads this measure. This is quite understandable because oil accounts for over 95% of its exports (More on this in a more indepth economic exploration of the SADC countries i.e in either Part 2 or 3). South Africa is second. This fact taken in conjunction with the fact the trade in services accounts for less than 10% of its GDP attests to the size of the South African economy relative to the other SADC nations.__

# In[30]:


# Rank the countries by service imports
high_imports = service_imports.sort_values(['Year','Value'],ascending=False)
high_imports.head(20)


# In[31]:


# plot the highest two countries against the region average
high_serv_imports = ['South Africa','Angola','SADC']
print_dataA(service_imports,high_serv_imports)
print_data(service_imports,high_serv_imports)


# __Lesotho and Seychelles are last again in this measure. However these are huge numbers taking into account that their GDP is also measured in 9 orders of magnitude (i.e x*10^9 where 0 < x < 10). Thus service imports would seem to be one of the reasons why their GDP's are so low.__

# In[32]:


# Plot the lowest two countries against the region average
low_serv_imports = ['Lesotho','Seychelles','SADC']
print_data(service_imports,low_serv_imports)


# In[33]:


# plot the lowest two countries
low_serv_imports=['Lesotho','Seychelles']
print_dataA(service_imports,low_serv_imports)
print_data(service_imports,low_serv_imports)


# ###  __Next we look at service exports.__

# In[34]:


# Create a data frame using the service exports indicator
indicator='Service exports (BoP'
mask = sadc_countries['IndicatorName'].str.startswith(indicator)
service_exports = sadc_countries[mask]
SADC = sadc_stats(service_exports)
service_exports = service_exports.append(SADC)
service_exports = service_exports.sort_values('Year',ascending=True)
#service_exports.head(10)


# In[35]:


# plot the data
printData(service_exports,countries)


# __South Africa yet again dominates in output with regard to this measure. All the other countries dwarf in comparison to the amount that South Africa makes in Service exports. This yet again emphasizes the size of the South African economy relative to the other SADC nations. although trade in services is less than 10% of the GDP the dollar amount of this trade in higher than most SADC countries. Mauritius or Tanzania would seem to take second place. But for simplicity we will plot all three.__

# In[36]:


# Rank the data in terms of the indicator
high_export = service_exports.sort_values(['Year','Value'],ascending=False)
high_export.head(20)


# In[37]:


# Plot the highest two countries against the region average
high_export = ['South Africa','Mauritius','Tanzania','SADC']
print_dataA(service_exports,high_export)
print_data(service_exports,high_export)


# __Lesotho yet again appears at the bottom of this measure. In this exploration, the country's economy would now appear to be the lowest in the region. This time it is joined by Swaziland.__

# In[38]:


# plot the lowest two countries against the region average
low_export=['Lesotho','Swaziland','SADC']
print_data(service_exports,low_export)


# __The area plot below shows that Lesotho's service exports dwarf in comparison to Swazilands. Thus it truly doesn't produce a lot in the way of service exports relative to its neighbours.__

# In[39]:


# plot the lowest two countries
low_export = ['Lesotho','Swaziland']
print_dataA(service_exports,low_export)
print_data(service_exports,low_export)


# ## Insurance and financial services
# __Some of the most prominent and valuable services provided the world over are Insurance and financial services. Thus naturally an exploration into the progress of the SADC countries in this paradigm would need to be made. This is the focus of this next section. Firstly, we look at Insurance and financial services as a percentage of service imports and then as a percentage of service exports.__

# In[41]:


# Create a data frame with the Insurance and financial services indicator
indicator= "Insurance and financial services (% of service imports, BoP)"
indmask= sadc_countries["IndicatorName"].str.startswith(indicator)
finance_stage = sadc_countries[indmask]
SADC = sadc_stats(finance_stage)
finance_stage = finance_stage.append(SADC)
finance_stage = finance_stage.sort_values('Year',ascending=True)
#finance_stage.head(10)


# In[92]:


# Plot the data
printData(finance_stage,countries)


# __The data in this metric is quite volatile. This is conveyed by the graph above which is quite messy. It is hard to discern exactly which countries are importing the most Insurance and financial services. However its is immediately apparent that Seychelles imported the most from 2006 to 2011. In 2005 Swaziland was leading the pack. A closer look is required. This need the use of a sorted table as has been done before.__

# In[43]:


# Rank the countries in terms of the indicator
high_fin = finance_stage.sort_values(['Year','Value'],ascending=False)
high_fin.head(20)


# __Considering the choppy nature of the data, although Botswana was the second highest importer in 2013, it does not have data for 2014. In this measure, it would be best to use the data we have for 2014. In this regard Zambia and Mauritius Are the two biggest importers.__

# In[44]:


# Plot the highest two countries against the region average
high_fin = ['Zambia','Mauritius','SADC']
print_dataA(finance_stage,high_fin)
print_data(finance_stage,high_fin)


# __The graphs show that even for these two countries no general trend can be discerned. There is a lot of uncertainty whether the fluctuations of these percentages are due to other services being imported or whether the countries are building their own capacity. This being particulary important as the economies of the SADC countries are growing.__

# In[45]:


# plot the lowest two countries against the region average
low_fin = ['Seychelles','Namibia','SADC']
print_dataA(finance_stage,low_fin)
print_data(finance_stage,low_fin)


# __during the first decade of the 21st century Seychelles has continually been the highest importer hovever by 2012 it had become the lowest importer. Again, at this stage it cannot be said with certainty why that is the case. We can just observe that it is the case. For Nambia on the other hand the percentage has remained relatively unchanged.__

# In[46]:


# Plot the lowest two countries
low_fin = ['Seychelles','Namibia']
print_dataA(finance_stage,low_fin)
print_data(finance_stage,low_fin)


# ### __Next we explore the exports of Insurance and financial services.__

# In[47]:


# Create the data frame using the Insurance and financial services exports indicator
indicator='Insurance and financial services (% of service exports'
mask = sadc_countries['IndicatorName'].str.startswith(indicator)
fin_exports = sadc_countries[mask]
SADC = sadc_stats(fin_exports)
fin_exports = fin_exports.append(SADC)
fin_exports = fin_exports.sort_values('Year',ascending=True)
#fin_exports.head()


# In[48]:


printData(fin_exports,countries)


# __As with the imports the data of the exports is also quite choppy. This again could be due to other services growing in prominence in the economies or conversely, services losing their prominence in favour of Insurance and financial services.__

# In[49]:


# Rank the countries in terms of Insuarance and financial services exports
high_exp = fin_exports.sort_values(['Year','Value'],ascending=False)
high_exp.head(20)


# In[50]:


high_exp=['South Africa','Botswana','Zambia','SADC']
print_dataA(fin_exports,high_exp)
print_data(fin_exports,high_exp)


# __As a result of the noise in the data it is quite hard to determine which economies would be in the top two. In recent years botwana has increased its insurance and financial services percentages but the country does not have data for 2014. If an extrapolation were to be made, botswana would probably have the highest percentage. However, because of the lack of data, South Africa, Botswana, and Zambia have been plotted. In the case of South Africa it can be seen that it also has substantial insurance and financial services in its economy.__

# In[51]:


low_exp = ['Mozambique','Seychelles','SADC']
#print_dataA(fin_exports,low_exp)
print_data(fin_exports,low_exp)


# __From 2006 to 2011 Seychelles increased the proportion of these services that were export relative to other services. Then in later years this proportion fell. This mirrors the behaviour of the imports of the services. This may lead one to speculate that other services have proved more prominent in that time period. Mozambique on the other hand has continuously had low export percentages.__

# In[52]:


low_exp = ['Mozambique','Seychelles']
print_dataA(fin_exports,low_exp)
print_data(fin_exports,low_exp)


# ## ICT service exports
# __One of the most rapidly growing services in the world today is to do with information and communications technology. This has lead to growth in automation, data analytics, machine learning, and artificial intelligence services. This is undoubtedly a sector that will continually grow in importance in the world for coming decades. Hence it is the focus of this section.__

# In[53]:


indicator = 'ICT service exports (%'
mask = sadc_countries['IndicatorName'].str.startswith(indicator)
ICT_exports = sadc_countries[mask]
SADC = sadc_stats(ICT_exports)
ICT_exports = ICT_exports.append(SADC)
ICT_exports = ICT_exports.sort_values('Year',ascending=True)
#ICT_exports.head()


# In[54]:


printData(ICT_exports,countries)


# __Yet again the data is choppy. This now suggests that the volatility is inherent in the services market in the SADC region. However, from the graph above, it seems that Swaziland has been continually increasing their ICT service exports in proportion of all its service exports. It unfortunately does not have 2014 data. In the past it has had large swings in these exports. it is uncertain whether the 2014 data would include an increase, a rapid increase, a decrease, or a rapid increase. But for the purposes of the exploration, it is assumed that it would have increased. Nonetheless the top three will be taken into account.__

# In[55]:


high_exp = ICT_exports.sort_values(['Year','Value'],ascending=False)
high_exp.head(20)


# In[56]:


high_exp = ['Namibia','Swaziland','Mauritius','SADC']
print_dataA(ICT_exports,high_exp)
print_data(ICT_exports,high_exp)


# __The area graph shows more accurately that ICT service exports are a more prominent export in Swaziland as a proprotion of the service exports in the country.
# Next we look at the lowest two countries in this metric.__

# In[57]:


low_exp = ['Angola','Zambia','SADC']
#print_dataA(ICT_exports,low_exp)
print_data(ICT_exports,low_exp)


# In[58]:


low_exp = ['Angola','Zambia']
print_dataA(ICT_exports,low_exp)
print_data(ICT_exports,low_exp)


# __Next we look at the monetary value of these ICT service exports.__

# In[59]:


indicator="ICT service exports (B"
indmask = sadc_countries["IndicatorName"].str.startswith(indicator)
ICT_stage = sadc_countries[indmask]
SADC = sadc_stats(ICT_stage)
ICT_stage = ICT_stage.append(SADC)
ICT_stage = ICT_stage.sort_values('Year',ascending=True)
#ICT_stage.head()


# In[60]:


printData(ICT_stage,countries)


# __Once again, when it comes to monetary value, South Africa has taken the reigns. It is followed by Mauritius but the gap is by no means miniscule. South African exports are just over two times those of Mauritius.__

# In[61]:


# Rank the countries in terms of ICT
high_exp = ICT_stage.sort_values(['Year','Value'],ascending=False)
high_exp.head(20)


# In[62]:


high_exp = ['South Africa','Mauritius','SADC']
print_dataA(ICT_stage,high_exp)
print_data(ICT_stage,high_exp)


# In[63]:


low_exp = ['Lesotho','Zambia','SADC']
#print_dataA(ICT_stage,low_exp)
print_data(ICT_stage,low_exp)


# __Lesotho is again last in this race. At this moment it is concerning as the world keep advancing at a faster rate. If something is not done to change course in the country, it will be unapologetically left behind. Zambia is second to last in this measure. The two countries have been woefully left behind as evidenced by the average SADC performance.__

# In[64]:


low_exp = ['Lesotho','Zambia']
print_dataA(ICT_stage,low_exp)
print_data(ICT_stage,low_exp)


# __Lastly we explore exports of all the equipment that is required in ICT. To this end we use the 'Communications, computer, etc. (% of service exports, BoP)' indicator.__

# In[65]:


indicator ='Communications, computer, etc. (% of service exports, BoP)'
mask= sadc_countries['IndicatorName'].str.startswith(indicator)
comp=sadc_countries[mask]
SADC = sadc_stats(comp)
comp = comp.append(SADC)
comp = comp.sort_values('Year',ascending=True)
#comp.head()


# In[66]:


printData(comp,countries)


# In[67]:


# Rank the countries
high_exp = comp.sort_values(['Year','Value'],ascending=False)
high_exp.head(20)


# In[68]:


high_exp = ['Botswana','Swaziland','SADC']
#print_dataA(comp,high_exp)
print_data(comp,high_exp)


# __Again Swaziland leads this metric. In 2013 its computer equipment exports were 73% of service exports. Botswana is sitting at 49%.
# Next we look at the lowest two countries.__

# In[69]:


low_exp = ['Angola','Zambia','SADC']
print_dataA(comp,low_exp)
print_data(comp,low_exp)


# In[70]:


low_exp = ['Angola','Zambia']
print_dataA(comp,low_exp)
print_data(comp,low_exp)


# ## Comparison with the rest of the world
# 
# __This exploration has revealed that in terms of monetary terms South Africa has the biggest economy and is the best positioned to take part and compete in the information age. In fact a few Data analytics and A.I companies are beginning to pop up. However the biggest threat to the success and security of the country is the volatile, unskilled, and unemployed populace. The country is plagued with violent strikes and demonstrations. They also have a high youth unemployment rate mainly due to low numbers of youths being able to afford a university education or get into university altogether. It is estimated that in the next 5 years 5 million jobs will be lost to automation and A.I. This is undoubetdly going to compound the tensions in the country. Nonetheless, It is the best chance SADC has to thrive in the coming decades. Therefore in the next section South Africa will be pitted against the top 10 biggest economies in the world.__

# In[71]:


indicator='GDP at market prices (curr'
mask = data['IndicatorName'].str.startswith(indicator)
global_gdp = data[mask]
global_gdp = global_gdp.sort_values(['Year','Value'],ascending=False)
#global_gdp.head(20)


# __Firstly The countries will be ranked in terms of market price GDP.__

# In[72]:


compare = ['United States','China','Japan','Germany','United Kingdom','France','Brazil','Italy','India','Canada']
mask = global_gdp['CountryName'].isin(compare)
comp_set = global_gdp[mask]
mask = sadc_countries['CountryName'].isin(['South Africa'])
sa = sadc_countries[mask]
mask = sa['IndicatorName'].str.startswith(indicator)
sa = sa[mask]
comp_set = comp_set.append(sa)
comp_set = comp_set.sort_values(['Year'],ascending=True)
#comp_set.head(11)


# In[73]:


alist=comp_set['CountryName'].unique().tolist()
printData(comp_set,alist)


# __At current market prices it can be seen that the general trend of the biggest economies is upwards. In comparison the South African GDP is realtively stagnant. Below are the descriptive statistics of 2014 for all the biggest economies.__

# In[74]:


counts = global_gdp[global_gdp['CountryName'].isin(compare)]
counts[counts['Year']==2014].describe()


# In[75]:


counts[counts['Year']==2014].mean()['Value']/sa[sa['Year']==2014]['Value'].iloc[0]


# __The calculation above that the South African GDP would need to increase 14.4 times in order to be average in this metric. The second last country by GDP is Canada Below is a graph showing South Africa's GDP plotted with Canada's. This is to give a sense of how far South Africa is from the bottom of the list.__

# In[76]:


print_dataA(comp_set,['South Africa','Canada'])
print_data(comp_set,['South Africa','Canada'])


# In[77]:


counts[(counts['Year']==2014) & (counts['CountryName'].str.startswith('Canada'))]['Value'].iloc[0]/sa[sa['Year']==2014]['Value'].iloc[0]


# __The above calculation shows that South Africa's output has to increase five times in order for it to have Canada's GDP. Next we look at Trade in Services.__

# In[78]:


indicator = 'Trade in services'
comp_set = data[data['CountryName'].isin(alist)]
trade_stage = comp_set[comp_set['IndicatorName'].str.startswith(indicator)]
trade_stage1 = trade_stage.sort_values(['Year','Value'],ascending=False)
trade_stage1.head(11)


# In[79]:


printData(trade_stage,alist)


# In[80]:


counts = data[data['CountryName'].isin(compare)]
counts_trade = counts[counts['IndicatorName'].str.startswith(indicator)]
counts_trade[counts_trade['Year']==2014].describe()


# In[81]:


sa = sadc_countries[sadc_countries['CountryName'].isin(['South Africa'])]
sa_trade = sa[sa['IndicatorName'].str.startswith(indicator)]
sa_trade[(sa_trade['Year']==2014)]['Value'].iloc[0]


# __This measure shows that as a percentage of GDP South Africa's trade in services is close to the median value of the world's biggest economies. The only difference being that South Africa's economy is 10 times smaller.
# Next we look at service exports in monetary terms.__

# In[82]:


indicator='Service exports (BoP'
serv_exp = comp_set[comp_set['IndicatorName'].str.startswith(indicator)]
serv_exp1 = serv_exp.sort_values(['Year','Value'],ascending=False)
serv_exp1.head(11)


# In[83]:


printData(serv_exp,alist)


# In[84]:


counts_serv = counts[counts['IndicatorName'].str.startswith(indicator)]
counts_serv[counts_serv['Year']==2014].describe()


# In[85]:


sa_serv = sa[sa['IndicatorName'].str.startswith(indicator)]
counts_serv[counts_serv['Year']==2014].mean()['Value']/sa_serv[sa_serv['Year']==2014]['Value'].iloc[0]


# __This calculation shows that South Africa's incomes is 14 times less than the average 10 ten countries income from service exports.
# Now we bring our attention to Insurance and financial service exports.__

# In[86]:


indicator='Insurance and financial services (% of service exports'
fin_exp = comp_set[comp_set['IndicatorName'].str.startswith(indicator)]
fin_exp1 = fin_exp.sort_values(['Year','Value'],ascending=False)
fin_exp1.head(11)


# __It can be seen from the table above that in terms of percentages South Africa is yet again in the middle of the distribution. The UK is an outlier in this data. It will undoubtedly skew the data as was the case with the US in the previous measure.__

# In[87]:


printData(fin_exp1,alist)


# __Next we focus our attention on the IT sector__

# In[88]:


indicator = 'ICT service exports (%'
ICT = comp_set[comp_set['IndicatorName'].str.startswith(indicator)]
ICT1 = ICT.sort_values(['Year','Value'],ascending=False)
ICT1.head(11)


# In[89]:


printData(ICT,alist)


# __As with GDP this race is not close for SA. As was the hypothesis of this exploration ICT is very important for developed countries. This graphs proves that for more developed countries ICT exports makes up a good portion of their service exports.
# Lastly we look at computer equipment.__

# In[90]:


indicator ='Communications, computer, etc. (% of service exports, BoP)'
comp = comp_set[comp_set['IndicatorName'].str.startswith(indicator)]
comp1 = comp.sort_values(['Year','Value'],ascending=False)
comp1.head(11)


# In[91]:


printData(comp,alist)


# __With this measure South Africa is even worse off. South Africa has computer equipment as 20% of its service exports. where as Italy, which is the second last, has equipment as 40% of its service exports.__

# # Conclusion
# 
# __We have seen through our exploration that the SADC country economies are growing. Chief amongst these being South Africa. IT has by far the biggest economy in the SADC region. It is also the most technologically advanced as evidenced by its ICT and services export sales. This puts South Africa in a unique position to take advantage of the technology that is being developed in this information age. However, this will not be with repercussions. As we will explore in the next project South Africa has other socio-economic issues that will stand in its way to development. This may slow the adoption of automation and A.I in the country leading to the region falling increasingly behind of the more developed world.__

# If you were able to make it this far, please provide some constructive criticism. Please don't just say shorten it, or it's too long. That was not intentional, I just got sucked in. 

# In[ ]:




