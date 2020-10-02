#!/usr/bin/env python
# coding: utf-8

# <p style="font-family: Arial; font-size:3.5em;color:purple; font-style:bold">
# Wold Development Indicators </p><br><br>

# <h1 style="font-family: Arial; font-size:2.0em;color:blue; font-style:bold">
# Research Question</h1>
# <br>
# 
# ### Who is likely to be next super power after USA? 

# <h1 style="font-family: Arial; font-size:2.0em;color:blue; font-style:bold">
# Motivation</h1>
# <br>
# <p>Do you want to know, who is chasing USA to become a next super power?</p>
# <p>We all know that Russia or China can be the next super power. They both are continuously in race. However, we need to compare different factors which makes country a super power. A super power must be superior in almost all senses. </p>
# <p>What we can do?</p>
# <p>We have to identify, explore and compare these three countries on different sectors. Which will be helpful to determine that, who will be the next super power? </p>
# <p>Following are the sectors which decides growth and power of any country.</p>
# 
# 1. **Health**
# 2. **Employment**
# 3. **Economy**
# 4. **Energy**
# 5. **Demography**
# 6. **Trade**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import re
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/Indicators.csv")
df.shape


# In[ ]:


df.head()


# In[ ]:


indicators = df['IndicatorName'].unique().tolist()
indicators.sort()
#print(indicators)


# In[ ]:


indicators_list = df[['IndicatorName','IndicatorCode']].drop_duplicates().values


# In[ ]:


indicators_list


# In[ ]:


new_indicators =[]
indicators_code =[]

for ind in indicators_list:
    indicator = ind[0]
    code = ind[1].strip()
    if code not in indicators_code:
        #Delete ,() from indicators and convert all characters to lower case
        modified_indicator = re.sub('[,()]',"",indicator).lower()
        #Replace - with "to"
        modified_indicator = re.sub('-'," to ",modified_indicator).lower()
        new_indicators.append([modified_indicator,code])
        indicators_code.append(code)


# In[ ]:


new_indicators[:5]


# In[ ]:


indicators_code[:5]


# In[ ]:


Indicators = pd.DataFrame(new_indicators, columns=['IndicatorName','IndicatorCode'])
Indicators = Indicators.drop_duplicates()
print(Indicators.shape)


# In[ ]:


Indicators.head()


# Defining keyword dictionary to find the right indicators.
# In this way based on the indicators we choose, we can asses the countries.
# 
# For any country following indicators can be crucial to asses growth.
# 
# 1. Demography (Population, birth, death)
# 2. Trade (import, export, production)
# 3. Health (birth, mortality, health care, doctors)
# 4. Economy (GDP, GINI, income, debt)
# 5. Energy (Electricity, Fuel, Power Consuption, Production, Emission)
# 6. Education (literacy, youth)
# 7. Employment (Employed, Unemployed)

# In[ ]:


key_word_dict = {}

key_word_dict['Demography'] = ['population','birth','death','fertility','mortality','expectancy']
key_word_dict['Trade'] = ['trade','import','export','good','shipping','shipment']
key_word_dict['Health'] = ['health','desease','hospital','mortality','doctor']
key_word_dict['Economy'] = ['income','gdp','gini','deficit','budget','market','stock','bond','infrastructure','debt']
key_word_dict['Energy'] = ['fuel','energy','power','emission','electric','electricity']
key_word_dict['Education'] = ['education','literacy','youth']
key_word_dict['Employment'] =['employed','employment','umemployed','unemployment']


# In[ ]:


def pick_indicator(feature):
    
    for indicator_ele in Indicators.values:
        
        if feature not in key_word_dict.keys():
            print("Choose the right feature!")
            break
        
        for word in key_word_dict[feature]:
            
            word_list = indicator_ele[0].split() # it would split from ','
            
            if word in word_list or word+'s' in word_list:
                
                print(indicator_ele)
                
                break


# In[ ]:


pick_indicator('Health')


# Now that we are exploring China, Russia and USA lets filter out those countries.

# In[ ]:


df_CRU = df[(df['CountryCode'] == 'CHN') | (df['CountryCode'] == 'RUS') | (df['CountryCode'] == 'USA')]


# In[ ]:


df_CRU.head()


# We first have to decide the indicators for countries. 
# The problem you will face is that all of them may not have all the indicators, and years having value for those indicators.
# So first create a new data frame containing only our chosen indicators.

# In[ ]:


chosen_indicators = [#Health
                    'SP.DYN.IMRT.IN','SH.STA.BRTC.ZS','SH.XPD.TOTL.ZS','SL.TLF.0714.ZS', 
                    # Employment
                    'SL.UEM.TOTL.ZS','SL.UEM.TOTL.MA.ZS','SL.UEM.TOTL.FE.ZS','SL.GDP.PCAP.EM.KD',
                    'SL.EMP.1524.SP.NE.ZS','SL.UEM.1524.NE.ZS', 
                    # Economy
                    'NY.GDP.PCAP.CD','NY.GDP.PCAP.KD','NY.GDP.PCAP.KD.ZG','SL.GDP.PCAP.EM.KD',
                    'SI.POV.GINI','SI.DST.10TH.10','SI.DST.FRST.10','GC.DOD.TOTL.GD.ZS','SH.XPD.TOTL.ZS',
                    # Energy
                    'EN.ATM.CO2E.PC','EG.USE.COMM.CL.ZS','EG.IMP.CONS.ZS','EG.ELC.RNWX.KH',
                    'EG.USE.ELEC.KH.PC','EG.ELC.NUCL.ZS','EG.ELC.ACCS.ZS','EG.ELC.ACCS.RU.ZS',
                    'EG.ELC.ACCS.UR.ZS','EG.FEC.RNEW.ZS',
                    # Demography
                    'SP.DYN.CBRT.IN','SP.DYN.CDRT.IN','SP.DYN.LE00.IN','SP.POP.65UP.TO.ZS',
                    'SP.POP.1564.TO.ZS','SP.POP.TOTL.FE.ZS','SP.POP.TOTL','SH.DTH.IMRT','SH.DTH.MORT',
                    'SP.POP.GROW','SE.ADT.LITR.ZS','SI.POV.NAHC','SH.CON.1524.MA.ZS','SH.STA.DIAB.ZS', 
                    #Trade
                    'NE.IMP.GNFS.ZS','NE.EXP.GNFS.CD','NE.IMP.GNFS.CD','NE.TRD.GNFS.ZS']


# In[ ]:


df_CRU_subset = df_CRU[df_CRU['IndicatorCode'].isin(chosen_indicators)]


# In[ ]:


print(df_CRU_subset.shape)
df_CRU_subset.head()


# This function prepare the stage of data frame. This data frame will have countries with chosen indicator and all years of data. Here, this method will filter out any excessive data from data frame. 
# 
# For example, It will filter out all those indicators for which data is not available for any one of the three countries.
# It will trim data according to years of data availability. So, new data frame will have data for all three countries for a chosen indicator. 

# In[ ]:


def stage_prep(indicator):
    
        df_stage_china = df_CRU_subset[(df_CRU_subset['CountryCode'] == 'CHN') &
                                       (df_CRU_subset['IndicatorCode'] == indicator)]
        
        df_stage_russia = df_CRU_subset[(df_CRU_subset['CountryCode'] == 'RUS') &
                                        (df_CRU_subset['IndicatorCode'] == indicator)]
        
        df_stage_usa = df_CRU_subset[(df_CRU_subset['CountryCode'] == 'USA') &
                                     (df_CRU_subset['IndicatorCode'] == indicator)]
        
        if((df_stage_china.empty) | (df_stage_russia.empty) | (df_stage_usa.empty)):

            print("This indicator is not present in all three countries. Please choose another indicator.")
        
        else:
            
            min_year_c = df_stage_china.Year.min()
            max_year_c = df_stage_china.Year.max()
            
            min_year_r = df_stage_russia.Year.min()
            max_year_r = df_stage_russia.Year.max()
            
            min_year_us = df_stage_usa.Year.min()
            max_year_us = df_stage_usa.Year.max()
            
            min_list = [min_year_c, min_year_r,min_year_us]
            max_among_all_min_years = max(min_list)
            
            max_list = [max_year_c,max_year_r,max_year_us]
            min_among_all_max_years = min(max_list)
        
            year_and_indicator_filter = ((df_CRU_subset['Year'] >= max_among_all_min_years) & 
                                         (df_CRU_subset['Year'] <= min_among_all_max_years) &
                                         (df_CRU_subset['IndicatorCode'] == indicator))
                        
            df_stage = df_CRU_subset[year_and_indicator_filter] 
                
            return df_stage


# <h1 style="font-family: Arial; font-size:2.0em;color:blue; font-style:bold">
# Visualization:</h1><br>
# Lets create a function to visualize our data. Here, I have used grouped bar chart to plot the data. Remember that bar chart is use to track changes over time.

# In[ ]:


import plotly 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

init_notebook_mode(connected=True)


# In[ ]:


def plot_barchart(df_stages):
    
    figure ={
    'data':[],
    'layout':{}
    }
    
    # Creating stage for each country
    df_stages_china = df_stages[df_stages['CountryCode'] == 'CHN']
    
    trace_1 = go.Bar({
        'y': list(df_stages_china['Year']),
        'x': list(df_stages_china['Value']),
        'text': list(df_stages_china['Value']),
        'name': 'China',
        'orientation': 'h'
    })
    
    figure['data'].append(trace_1)
    
    df_stages_russia = df_stages[df_stages['CountryCode'] == 'RUS']
   
    trace_2 = go.Bar({
        'y': list(df_stages_russia['Year']),
        'x': list(df_stages_russia['Value']),
        'text': list(df_stages_russia['Value']),
        'name': 'Russia',
        'orientation': 'h'
    })
    
    figure['data'].append(trace_2)
    
    df_stages_usa = df_stages[df_stages['CountryCode'] == 'USA']
          
    trace_3= go.Bar({
        'y': list(df_stages_usa['Year']),
        'x': list(df_stages_usa['Value']),
        'text': list(df_stages_usa['Value']),
        'name': 'USA',
        'orientation': 'h'
    })
    
    figure['data'].append(trace_3)
    
    title = df_stages['IndicatorName'].iloc[0]
    
    figure['layout']['title'] = title
    figure['layout']['xaxis'] = {'title': 'Value'}
    figure['layout']['yaxis'] = {'title': 'Years'}
    figure['layout']['hovermode'] = 'compare'
    
    iplot(figure)
    


# We have data for inidcators and countries over several years. Which means continuous data. This kind of data can be greatly visualize using line graph. Line graph is mainly use to show data that changes over same period of time. This dataset is perfect example of that.
# 
# Here, I have used Plotly's connected scatter plot to show data points and changes over time.

# In[ ]:


def plot_line(df_stages):
    
    # Initializing figure. If we initialize it outside of function then new data will get appended with old data and 
    # plot will show all the data including new and old. To avoid that repetation of data in figure, we initialize it inside.
    figure ={
    'data':[],
    'layout':{}
    }
    
    # Creating stage for each country
    df_stages_china = df_stages[df_stages['CountryCode'] == 'CHN']
    
    trace_1 = go.Scatter({
        'x': list(df_stages_china['Year']),
        'y': list(df_stages_china['Value']),
        'connectgaps': True,
        'text': list(df_stages_china['Value']),
        'name': 'China'
    })
    
    figure['data'].append(trace_1)
    
    df_stages_russia = df_stages[df_stages['CountryCode'] == 'RUS']
   
    trace_2 = go.Scatter({
        'x': list(df_stages_russia['Year']),
        'y': list(df_stages_russia['Value']),
        'connectgaps': True,
        'text': list(df_stages_russia['Value']),
        'name': 'Russia'
    })
    
    figure['data'].append(trace_2)
    
    df_stages_usa = df_stages[df_stages['CountryCode'] == 'USA']
          
    trace_3= go.Scatter({
        'x': list(df_stages_usa['Year']),
        'y': list(df_stages_usa['Value']),
        'connectgaps': True,
        'text': list(df_stages_usa['Value']),
        'name': 'USA'
    })
    
    figure['data'].append(trace_3)
    
    title = df_stages['IndicatorName'].iloc[0]
    
    figure['layout']['title'] = title
    figure['layout']['xaxis'] = {'title': 'Years'}
    figure['layout']['yaxis'] = {'title': 'Value'}
    figure['layout']['hovermode'] = 'compare'
    
    iplot(figure, validate =False)


# ## It's time to explore countries with particular indicator and see how Russia and China are chasing USA.

# Lets go by one sector after another starting with Health sector.

# ## 1) Health

# #### Overall Analysis:
# <p>From findings below we can see that Russia and China are spending 4 - 6 percentage of GDP behind Health Expenditure while USA is spending 13 - 17 percent. From this data we can infere that this can be the reason behind low infant mortality rate in USA then China and Russia. However, more important thing to notice over here is, in 2015 all three countries have almost same infant mortality rate. Which suggest that there are other factors too effecting infant mortality for example, Birth attended by skilled health staff.</p>

# In[ ]:


df_stage_health_1 = stage_prep(chosen_indicators[0])


# In[ ]:


print(df_stage_health_1.shape)
print("Min year: ",df_stage_health_1.Year.min()," Max year: ", df_stage_health_1.Year.max())
df_stage_health_1.head()


# In[ ]:


plot_barchart(df_stage_health_1)


# In[ ]:


plot_line(df_stage_health_1)


# In[ ]:


df_stage_health_2 = stage_prep(chosen_indicators[1])


# In[ ]:


print(df_stage_health_2.shape)
print("Min year: ",df_stage_health_2.Year.min()," Max year: ", df_stage_health_2.Year.max())
df_stage_health_2.head()


# In[ ]:


plot_line(df_stage_health_2)


# In[ ]:


df_stage_health_3 = stage_prep(chosen_indicators[18])


# In[ ]:


print(df_stage_health_3.shape)
print("Min year: ",df_stage_health_3.Year.min()," Max year: ", df_stage_health_3.Year.max())
df_stage_health_3.head()


# In[ ]:


plot_barchart(df_stage_health_3)


# In[ ]:


plot_line(df_stage_health_3)


# ## 2) Employment

# #### Overall Analysis:
# <p>Here, Very good thing to make a note that China has provided Employment to the most of the population. It's unemployment rate is never goes up then 5%. If you look at carefully the bar chart shows that, Russia has higher unemployment rate before year 2005 and lower afterwards. Where as USA had lower unemployment before year 2005 and higher afterwards. The data shows that unemployed labors are more in USA then Russia and China.</p>

# In[ ]:


df_stage_emp_1 = stage_prep(chosen_indicators[4])


# In[ ]:


print(df_stage_emp_1.shape)
print("Min year: ",df_stage_emp_1.Year.min()," Max year: ", df_stage_emp_1.Year.max())
df_stage_emp_1.head()


# In[ ]:


plot_barchart(df_stage_emp_1)


# In[ ]:


plot_line(df_stage_emp_1)


# ## 3) Economy

# #### Overall Analysis (Part 1):
# <p>We know that GDP measures an economy's output and therefore it is measure of size of an economy.
# The below bar chart shows that USA has been always higher GDP then Russia and China or we can say it is duble size. But, the anohter important point to notice that annual GDP growth of China has always been double then USA. Also, Russia too lags behind China in this factor. </p>

# In[ ]:


df_stage_ec_1 = stage_prep(chosen_indicators[10])


# In[ ]:


print(df_stage_ec_1.shape)
print("Min year: ",df_stage_ec_1.Year.min()," Max year: ", df_stage_ec_1.Year.max())
df_stage_ec_1.head()


# In[ ]:


plot_barchart(df_stage_ec_1)


# In[ ]:


plot_line(df_stage_ec_1)


# In[ ]:


df_stage_ec_2 = stage_prep(chosen_indicators[12])


# In[ ]:


print(df_stage_ec_2.shape)
print("Min year: ",df_stage_ec_2.Year.min()," Max year: ", df_stage_ec_2.Year.max())
df_stage_ec_2.head()


# In[ ]:


plot_barchart(df_stage_ec_2)


# In[ ]:


plot_line(df_stage_ec_2)


# In[ ]:


df_stage_ec_3 = stage_prep(chosen_indicators[14])


# In[ ]:


print(df_stage_ec_3.shape)
print("Min year: ",df_stage_ec_3.Year.min()," Max year: ", df_stage_ec_3.Year.max())
df_stage_ec_3.head()


# #### Overall Analysis (Part 2):
# <p>From the graphs below we can say that income distribution and income share held by highest 10% shows almost same value in year 2010. This shows that China and Russia both chasing USA.</p>

# In[ ]:


plot_line(df_stage_ec_3)


# In[ ]:


df_stage_ec_4 = stage_prep(chosen_indicators[15])


# In[ ]:


print(df_stage_ec_4.shape)
print("Min year: ",df_stage_ec_4.Year.min()," Max year: ", df_stage_ec_4.Year.max())
df_stage_ec_4.head()


# In[ ]:


plot_line(df_stage_ec_4)


# ## 4) Energy

# #### Overall Analysis (Part 1):
# 
# <p>Here you will notice that CO2 Emissions per capita in Russia and USA has higher then China. But, one important factor come into play over here is <b>"Population".</b> China has the largest population in the world. So, when you count CO2 emission per capita, you may need to normalize the population factor. Because of that only we can not say which country has higher emissions rate. </p>

# In[ ]:


df_stage_energy_1 = stage_prep(chosen_indicators[19])


# In[ ]:


print(df_stage_energy_1.shape)
print("Min year: ",df_stage_energy_1.Year.min()," Max year: ", df_stage_energy_1.Year.max())
df_stage_energy_1.head()


# In[ ]:


plot_barchart(df_stage_energy_1)


# In[ ]:


plot_line(df_stage_energy_1)


# #### Overall Analysi (part 2):
# <p>We know that in world wars, Russia and USA both had nuclear power. China has later become nuclear state. So, energy production from nuclear sources can be low for China then Russia. However, China's electricity production from renewable sources is higher then Russia and very fast chasing production of USA. Also, the line chart of <b>"Access to Electricity (% of population)"</b> shows that China achieved 100% electricity in country by the end of 2012. So, In that sense also China competes both countries.</p>

# In[ ]:


df_stage_energy_2 = stage_prep(chosen_indicators[20])


# In[ ]:


print(df_stage_energy_2.shape)
print("Min year: ",df_stage_energy_2.Year.min()," Max year: ", df_stage_energy_2.Year.max())
df_stage_energy_2.head()


# In[ ]:


plot_barchart(df_stage_energy_2)


# In[ ]:


plot_line(df_stage_energy_2)


# In[ ]:


df_stage_energy_3 = stage_prep(chosen_indicators[22])


# In[ ]:


print(df_stage_energy_3.shape)
print("Min year: ",df_stage_energy_3.Year.min()," Max year: ", df_stage_energy_3.Year.max())
df_stage_energy_3.tail()


# In[ ]:


plot_barchart(df_stage_energy_3)


# In[ ]:


plot_line(df_stage_energy_3)


# In[ ]:


df_stage_energy_4 = stage_prep(chosen_indicators[25])


# In[ ]:


print(df_stage_energy_4.shape)
print("Min year: ",df_stage_energy_4.Year.min()," Max year: ", df_stage_energy_4.Year.max())
df_stage_energy_4.head()


# In[ ]:


plot_line(df_stage_energy_4)


# ## 5) Demography

# #### Overall Analysis:
# <p>In terms of Life Expectancy, all of the three has same numbers by the end of 2013. Also, notice the steep increase in China's life expectancy after year 1962. </p>

# In[ ]:


df_stage_dg_1 = stage_prep(chosen_indicators[31])


# In[ ]:


print(df_stage_dg_1.shape)
print("Min year: ",df_stage_dg_1.Year.min()," Max year: ", df_stage_dg_1.Year.max())
df_stage_dg_1.head()


# In[ ]:


plot_line(df_stage_dg_1)


# ## 6) Trade

# #### Overall Analysis:
# <p>Trade is also very important factor to consider while evaluating any country. WE can see that Import and Export of goods and services in all three countries shows same curve in line graph. More important thing to notice that imports of China is less then USA and exports are same by the end of the year 2013. Which shows that China is highly likely to overcome USA in Trade sector in near future.</p>

# In[ ]:


df_stage_trd_1 = stage_prep(chosen_indicators[45])


# In[ ]:


print(df_stage_trd_1.shape)
print("Min year: ",df_stage_trd_1.Year.min()," Max year: ", df_stage_trd_1.Year.max())
df_stage_trd_1.head()


# In[ ]:


plot_line(df_stage_trd_1)


# In[ ]:


plot_barchart(df_stage_trd_1)


# In[ ]:


df_stage_trd_2 = stage_prep(chosen_indicators[44])


# In[ ]:


print(df_stage_trd_2.shape)
print("Min year: ",df_stage_trd_2.Year.min()," Max year: ", df_stage_trd_2.Year.max())
df_stage_trd_2.head()


# In[ ]:


plot_line(df_stage_trd_2)


# In[ ]:


plot_barchart(df_stage_trd_2)


# <h1 style="font-family: Arial; font-size:2.0em;color:blue; font-style:bold">
# Conclusion</h1>

# * After doing deep analysis of USA, China and Russia on different factors, we can say that, China is chasing or competing USA faster then Russia.
# *  From above analysis we can say that, Russia is behind China in providing employment, GDP annual growth, electricity production from renewable energy sources, life expectancy and imports and exports of goods and services. While China is either competing or equal to USA in most of this sectors. 
# * Also, all of the three countries are almost same on sectors like infant mortality rate, GINI index and access to electricity. 
# * Overall, China is ahead of Russia in the race of becoming next super power.

# In[ ]:




