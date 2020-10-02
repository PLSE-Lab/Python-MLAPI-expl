#!/usr/bin/env python
# coding: utf-8

# ## Economic Freedom World Wide 2016
# #### Economic Freedom of the World measures the degree to which the policies and institutions of countries are supportive of economic freedom. The cornerstones of economic freedom are personal choice, voluntary exchange, freedom to enter markets and compete, and security of the person and privately owned property. Reference [Fraser Institude Economic Freedom Report] (https://www.fraserinstitute.org/economic-freedom)
# 
# This is considered my first work, and I want to define my country so that it will be in the field of my knowledge only. I am familiar with what happened at that time and I can link the numbers and the events well We will discuss in this kernel simple analyzes about
# * the Egyptian economic reality and the political 
# * economic situation of the government,
# * The policy of the state and the level of transparency of the law
# * January 25 Revolution and its causes and consequences
# 
# All that is mentioned in this kernel is based on analyzes of the previous data studied and linked to the real events, do not bias any government or any particular system, does not accuse any government or any system of corruption, but the figures will show that or you will notice
# This kernel was just a training on the above study can add some points that I can work on it or leave a comment to me about it, I will work later on the data of the world and continents one by one but at the moment I want a comment to encourage you to do more. Sorry if there were errors please re-review , Loaii abdalslam

# In[ ]:


#Import all required libraries for reading data, analysing and visualizing data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


efw = pd.read_csv('../input/efw_cc.csv')
efw.shape


# In[ ]:


efw.head()


# In[ ]:


efw.year.value_counts().sort_index().index


#  There is data for 162 countries ranging from 1970 till 2016. 1970 to 2000 has data once in for 5 years and there is data for every year from 2000.

# ### Describe Data Info

# In[ ]:


efw.info()


# ## What would affect Economic Freedom?
# 

# In[ ]:


efw_eff=efw[["ECONOMIC FREEDOM", "rank", "1_size_government", "2_property_rights", "3_sound_money", "4_trade", "5_regulation"]]
sns.set(font_scale=1.3)
x,ax=plt.subplots(figsize=(12,12))
sns.heatmap(efw_eff.corr(),cbar=True,annot=True,fmt='.2f',square=True)


# ## The average growth of global economic freedom
# 

# In[ ]:


efw_=efw[['ECONOMIC FREEDOM','year']]
#plt.plot(efw_['year'],efw_['ECONOMIC FREEDOM'])
efw_.groupby('year').mean().plot(figsize=[9,9],)


# In[ ]:


egypt=efw[ efw['countries']=='Egypt']
egypt.head(5)


# ## Let's Get More Information About Us !

# In[ ]:


egy_year=egypt.groupby('year')
def plot_stat(select,plot):
       egy_year[select].mean().plot(plot,figsize=[9,9],title=select)


# ###  Economic Freedom  at Egypt 1980 - 2016

# Egypt witnessed a period of economic freedom in terms of the rate of increase during the era of President Mohamed Hosni Mubarak, where growth indicators rose at the maximum capacity in the years from 1990 to 2000 and hence began economic fluctuations
# 
# * The government has been striving to provide long-term opportunities for the growth of the Egyptian economy, but a state of despair as a result of the inability of ordinary citizens to cope with the painful economic situation.
# 
# * A revolution broke out in the country in 2011, which stopped the development of government in the economic system and increased government spending to repair the gaps caused by the revolution and we continue to suffer from those gaps so far we note the linear decline from 2011 to 2016
# #### [wikipedia](https://ar.wikipedia.org/wiki/%D8%A7%D9%82%D8%AA%D8%B5%D8%A7%D8%AF_%D9%85%D8%B5%D8%B1)

# In[ ]:



plot_stat('ECONOMIC FREEDOM','line')


# ### Std Inflation at egypt (1985-2016)

# Inflation rates, as we see, are absolutely unstable throughout Egyptian history, but we note that with the beginning of the revolution in 2011 we noticed an increase in inflation compared to the current situation and the previous 10 years. This rate of increase is considered a problem and represents a threat to the Egyptian economy. In fact, we see a significant drop in inflation, but after ten years of work from our current history, we will get the inflation and economic rate that we enjoyed 20 years ago. In fact, we wasted a lot of time. We had a chance, but that's what happened.

# In[ ]:


plot_stat('3b_std_inflation','line')


# ## The Government activity in the economic sector in the last ten years
# **In this section, we will discuss the economic crisis and the government strategy. Were the solutions satisfactory or are they still trying? Will the floating of the pound later in 2016 be the best solution or are there other solutions?**

# In[ ]:


egy_year=egypt[egypt['year']>2010].groupby('year')
_ = egy_year.mean().plot(y=["1_size_government","1c_gov_enterprises",'1a_government_consumption','1d_top_marg_tax_rate','4a_tariffs',"3_sound_money"], figsize = (15,20), subplots=True)
_ = plt.xticks(rotation=360)
_x = egy_year.mean().plot(y=["1b_transfers","3a_money_growth",'4c_black_market','ECONOMIC FREEDOM'], figsize = (15,20), subplots=True)
_x = plt.xticks(rotation=360)


# We note from the previous chart that the decision to float the Egyptian pound was not a blatant mistake, as some thought it was due to large economic observations. I hope that I am not mistaken in this matter. The size of the government naturally affects the government consumption of the state money significantly. Over the past six years:
# - First, reducing the size of the Egyptian government and thus reducing the volume of government public consumption
# - In conjunction with the increase of government companies with a slight rise in the margin tax, and a very small contraction in customs value in conjunction with the imposition of sanctions and control of the black markets
# 
# This was the policy of the government in the economic sector in recent years, but tried as much as possible to defy the financial crisis and headed later in 2016 to float the Egyptian pound after the rates of trade achieved the worst level and the stagnation of local markets to investment, so the government has developed for that solution I do not have Any information about 2017 and 2018, but I would like to assure you that the situation is not good on perception and it was a failed strategy
# 
# Knowing that all these rates have not changed at all and that in light of all those taxes added to all the people did not increase their pricing much, all that happened is a decline in the value of the currency and this has created the gap between the people and the government and its policy

# ## The legal activity of the government to protect the rights of citizens and public property and organize business
# **In this section we will discuss the changes in the legal activities of the government and its adoption, the consequences of this, the integrity of the system of government and the transparency of the judiciary and the police in general**

# In[ ]:


egy_year=egypt[egypt['year']>2008].groupby('year')
_x = egy_year.mean().plot(kind='barh',y=["5a_credit_market_reg","5b_labor_market_reg",'5c_business_reg','5_regulation','2_property_rights','2c_protection_property_rights'], figsize = (15,20), subplots=True)


# i will tell it in the form of a short story:
# There was a revolution and destroyed everything, and it was the duty of the government to treat the situation in secret without disclosing the results of the revolution, because this threatens the stock market, trade, tourism and so on .. The Egyptian citizen thinks that the situation began to improve, and the government is working hard to repair what the people spoiled, The government is doing it, but in reality the government is not doing anything so far, it is trying to get what has already been corrupted, the figures are talking

# **I will discuss in this section the legal, diplomatic and judicial activity of the state and the government in general, what may affect the military intervention in the country and the extent of transparency and credibility of the judiciary and governance in recent years before and after the revolution and did the revolution for those reasons actually or not?**

# In[ ]:


egy_year=egypt[egypt['year']>1990].groupby('year')
_x = egy_year.mean().plot(kind='area',y=["2h_reliability_police",'2e_integrity_legal_system','2b_impartial_courts','2a_judicial_independence','2d_military_interference'], figsize = (15,20), subplots=True)
_x = plt.xticks(rotation=360)


# The former regime had gaps in close periods challenging the independence of the judiciary and the legal system in the country and therefore when the corruption of the judicial system, it results in corruption in the police and this is what we will notice in the last periods of the rule of Mohamed Hosni Mubarak
# 
# On the other hand, we will see that the current system has enjoyed transparency in the government and the independence of the judiciary and the courts in general and things have returned to normal away from the gap of the last ten years (2000-2010) of the rule of President Mohamed Hosni Mubarak as we are represented by graphic representation
# 
# **Indeed, they were demanding youth to live, freedom and social justice, and as we have seen from the rates of inflation already studied**
# > "They thought the country was about to collapse. What the numbers have shown me now is that it was not a collapse. It was just a fluctuation. What we live in now is the real collapse"

# In[ ]:




