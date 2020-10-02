#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


final_data=pd.read_csv("../input/imputed-data/final_imputed_data.csv",encoding = 'unicode_escape')
final_data.head()
final_data.shape


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


var1=final_data.iloc[:,0:4]
var1.head()
var2=final_data.iloc[:,[0,1,2,4]]
var2.head()
var3=final_data.iloc[:,[0,1,2,5]]
var3.head()
var4=final_data.iloc[:,[0,1,2,6]]
var4.head()
var5=final_data.iloc[:,[0,1,2,7]]
var5.head()
var6=final_data.iloc[:,[0,1,2,8]]
var6.head()


# In[ ]:


var1_avg=pd.DataFrame(var1.groupby(["year","label"],as_index=False).mean())
#var1_avg.set_index("year",inplace=True)
var1_avg.label.unique()
plt.figure(figsize=(10,10))
sns.lineplot(y="access_to_electricity",x="year",data=var1_avg,hue="label",palette="Set1", legend = False)
plt.legend(title = "Country categories", labels = ["Developed", "In transition", "Developing"])
plt.title("Plot of Access to Electricity against Years 2010-2014")
plt.ylabel("Access to Electricity")
plt.xlabel("Year")


# In[ ]:


var2_avg=pd.DataFrame(var2.groupby(["year","label"],as_index=False).mean())
#var1_avg.set_index("year",inplace=True)
var2_avg
plt.figure(figsize=(10,10))
sns.lineplot(y="co2_emissions",x="year",data=var2_avg,hue="label",palette="Set1", legend = False)
plt.legend(title = "Country categories", labels = ["Developed", "In transition", "Developing"])
plt.title("Plot of CO2 Emissions against Years 2010-2014")
plt.ylabel("CO2 Emissions")
plt.xlabel("Year")


# In[ ]:


var3_avg=pd.DataFrame(var3.groupby(["year","label"],as_index=False).mean())
#var1_avg.set_index("year",inplace=True)
var3_avg
plt.figure(figsize=(10,10))
sns.lineplot(y="electric_power_consumption",x="year",data=var3_avg,hue="label",palette="Set1", legend = False)
plt.legend(title = "Country categories", labels = ["Developed", "In transition", "Developing"])
plt.title("Plot of Electric Power Consumption against Years 2010-2014")
plt.ylabel("Electric Power Consumption")
plt.xlabel("Year")


# In[ ]:


var4_avg=pd.DataFrame(var4.groupby(["year","label"],as_index=False).mean())
#var1_avg.set_index("year",inplace=True)
var4_avg
plt.figure(figsize=(10,10))
sns.lineplot(y="electricity_from_renewable_sources",x="year",data=var4_avg,hue="label",palette="Set1",legend = False)
plt.legend(title = "Country categories", labels = ["Developed", "In transition", "Developing"])
plt.title("Plot of Electricity from Renewable Sources(excluding Hydel) against Years 2010-2014")
plt.ylabel("Electricity from Renewable Sources")
plt.xlabel("Year")


# In[ ]:


var5_avg=pd.DataFrame(var5.groupby(["year","label"],as_index=False).mean())
#var1_avg.set_index("year",inplace=True)
var5_avg
plt.figure(figsize=(10,10))
sns.lineplot(y="renewable_electricity_output",x="year",data=var5_avg,hue="label",palette="Set1",legend = False)
plt.legend(title = "Country categories", labels = ["Developed", "In transition", "Developing"])
plt.title("Plot of Renewable Electricity Output against Years 2010-2014")
plt.ylabel("Renewable Electricity Output")
plt.xlabel("Year")


# In[ ]:


var6_avg=pd.DataFrame(var6.groupby(["year","label"],as_index=False).mean())
#var1_avg.set_index("year",inplace=True)
var6_avg
plt.figure(figsize=(10,10))
sns.lineplot(y="gdp_per_capita",x="year",data=var6_avg,hue="label",palette="Set1", legend = False)
plt.legend(title = "Country categories", labels = ["Developed", "In transition", "Developing"])
plt.title("Plot of GDP per Capita against Years 2010-2014")
plt.ylabel("GDP per Capita")
plt.xlabel("Year")


# ## Question 5)

# In[ ]:


final_data.head()
india=final_data[final_data.country=="India"]
india.reset_index(inplace=True,drop=True)
india["Hyd_Consumption"]=india["renewable_electricity_output"]-india["electricity_from_renewable_sources"]
india


# What could be the major findings for Indian economy with respect to CO2 mitigation, renewable energy adoption and economic development.
# 

# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot(x="year",y="co2_emissions",data=india,label="CO2 emissions")
sns.lineplot(x="year",y="electricity_from_renewable_sources",data=india,label="Electricity from renewable sources")
plt.title("Plot of CO2 Emissions and Electricity from Renewable Sources(excluding Hydel) against Years 2010-2014 in India")
plt.ylabel("CO2 Emissions and Electricity from Renewable Sources")
plt.xlabel("Year")


# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot(x="year",y="renewable_electricity_output",data=india,label="Renewable Output")
sns.lineplot(x="year",y="access_to_electricity",data=india,label="Electricity Access")
sns.lineplot(x="year",y="Hyd_Consumption",data=india,label="Hydel Production")
plt.title("Plot of Renewable Electricity Output,Access to Electricity,Hydel Consumption against Years 2010-2014 in India")
plt.ylabel("Renewable Electricity Output,Access to Electricity,Hydel Consumption")
plt.xlabel("Year")


# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot(x="year",y="gdp_per_capita",data=india,label="GDP per capita")
sns.lineplot(x="year",y="electric_power_consumption",data=india,label="Elecctric Power Consumption")
plt.title("Plot of GDP per Capita, Electric Power Consumption against Years 2010-2014 in India")
plt.ylabel("GDP per Capita, Electric Power Consumption")
plt.xlabel("Year")


# In[ ]:


india.info()
india_num=india.iloc[:,[3,4,5,6,7,8,11]]
india_num.head()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(india_num.corr(),cmap='coolwarm',annot=True)
plt.title("Correlations between all the variables of India",fontsize=18,fontweight="bold")


# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot("access_to_electricity","co2_emissions",data=india)
plt.title("Plot of CO2 emissions against Access To Electricity in India from 2010-14",fontsize=18,fontweight="bold")
plt.xlabel("Access to Electricity")
plt.ylabel("CO2 Emissions")
#clearly, there has been an increase in CO2 emisions with increasing access to electriccity in recejnt years


# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot("electric_power_consumption","co2_emissions",data=india)
plt.title("Plot of CO2 emissions against Electric Power Consumption in India from 2010-14",fontsize=18,fontweight="bold")
plt.xlabel("Electric Power Consumption")
plt.ylabel("CO2 Emissions")
#clear increase in CO2 emissions with increasing power consumption


# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot("renewable_electricity_output","co2_emissions",data=india)
plt.title("Plot of CO2 emissions against Renewable energy Output in India from 2010-14",fontsize=18,fontweight="bold")
plt.xlabel("Renewable energy Output")
plt.ylabel("CO2 Emissions")
#there has been a decresing trend in CO2 emissions with increase in Renewable energy output in recent years


# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot("gdp_per_capita","co2_emissions",data=india)
plt.title("Plot of CO2 emissions against GDP per Capita in India from 2010-14",fontsize=18,fontweight="bold")
plt.xlabel("GDP per Capita")
plt.ylabel("CO2 Emissions")
#there has been a increasing trend in CO2 emissions with increase in GDP in recent years


# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot("electricity_from_renewable_sources","co2_emissions",data=india)
plt.title("Plot of CO2 emissions against Electricity from renewable resources exccpe Hyd in India from 2010-14",fontsize=18,fontweight="bold")


# there has been a increasing trend in CO2 emissions with increase Renewable sources in recent years
# this just means that even though there is an increase in use of Electricity from renewable sources(exclusing Hyd), it does not cause a decrease in CO2 emissions in India. It could be that the Electricity generated from these renewable sources is not enough to meet the energy needs of the country, which is why there is still an increasing dependence on fossil fuels to meet the needs.
# also, CO2 emissions coes  from various sources, not just electricity producction. there may be vaariouss other sources which are using fossil fuels and are causing an increase in co2 emissions

# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot(y="Hyd_Consumption",x="co2_emissions",data=india)
plt.title("Plot of CO2 emissions against Electricity from hydroelectric resources exccpe Hyd in India from 2010-14",fontsize=18,fontweight="bold")


# We see that over the years, the electricity production from Hydroelectricity decreses, and CO2 emision increases. Here, does the correlation imply causation?
# also, in india, there are alredy several hydroelectric power plants, so maybe there is no new room for improvement. the figures here are % in the total renewable electriccity power production, which are not indicative of the absolute values. the relative vallues maybe decreasing over the years, but the absolute numbers may still be high. it could be that there are no new dams to build because of the abndance of many, while the oher renewable sectors are still untapped, which are being explored now. 
# from 2008-2018, Hydel power production has decreased a lot due to environmental and social impact. and production of other renewable sources of energy has more than doubled. In this graph, correlationn odes not imply causation. here, we simply notice that over the years, hydel power production has decreased while CO2 emissions haave increased over the years

# renewable energy adoption and economic development.

# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot(y="gdp_per_capita",x="renewable_electricity_output",data=india)
plt.title("Plot of GDP per Capita vs Renewable Energy Production in India from 2010-14",fontsize=18,fontweight="bold")
#there has been a increasing trend in CO2 emissions with increase in GDP in recent years


# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot(y="gdp_per_capita",x="electricity_from_renewable_sources",data=india)
plt.title("Plot of GDP per Capita vs Non Hydel Electric Production in India from 2010-14",fontsize=18,fontweight="bold")
#there has been a increasing trend in CO2 emissions with increase in GDP in recent years


# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot(y="gdp_per_capita",x="electricity_from_renewable_sources",data=india)
plt.title("Plot of GDP per Capita vs Non Hydel Electric Production in India from 2010-14",fontsize=18,fontweight="bold")
#there has been a increasing trend in CO2 emissions with increase in GDP in recent years


# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot(y="access_to_electricity",x="electricity_from_renewable_sources",data=india)
plt.title("Plot of Access to Electricity against Non Hydel Electric Production in India from 2010-14",fontsize=18,fontweight="bold")


# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot(y="access_to_electricity",x="renewable_electricity_output",data=india)
plt.title("Plot of Access to Electricity against Renewable Energy Production in India from 2010-14",fontsize=18,fontweight="bold")


# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot(y="electric_power_consumption",x="electricity_from_renewable_sources",data=india)
plt.title("Plot of Electricity Power consumption against Non Hydel Electric Production in India from 2010-14",fontsize=18,fontweight="bold")


# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot(y="electric_power_consumption",x="renewable_electricity_output",data=india)
plt.title("Plot of Electricity Power consumption against Renewable Energy Production in India from 2010-14",fontsize=18,fontweight="bold")


# In[ ]:


pip install linearmodels
from linearmodels import PanelOLS
final_data.head()
panel=final_data(["country","year"]).to_panel()


# In[ ]:


from linearmodels.panel import PanelOLS
exog_vars = ['expersq', 'union', 'married', 'year']
exog = sm.add_constant(data[exog_vars])
mod = PanelOLS(data.lwage, exog, entity_effects=True)
fe_res = mod.fit()
print(fe_res)

