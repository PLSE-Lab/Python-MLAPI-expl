#!/usr/bin/env python
# coding: utf-8

# **Iniatilizing the input directories**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **Constants and Imports**
# 
# We would be using pandas, matplotlib and numpy for our dashboard

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ## **India**
# 
# Getting the data for India

# In[ ]:


covidIndiaDF=pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")
covidIndiaDF


# In[ ]:


covidIndiaDF["Date"]=pd.to_datetime(covidIndiaDF["Date"],format="%d/%m/%y")
covidIndiaDF=covidIndiaDF.groupby(["Date","State/UnionTerritory"]).agg({
    "Cured" : "max",
    "Deaths" : "max",
    "Confirmed" : "max"
}).reset_index()

covidIndiaDF[covidIndiaDF["State/UnionTerritory"]=="West Bengal"].sort_values("Date", ascending=False)
covidIndiaDF["Death_Ratio"]=covidIndiaDF["Deaths"]/covidIndiaDF["Confirmed"]*100
covidIndiaDF["Treated"]=covidIndiaDF["Confirmed"]-covidIndiaDF["Deaths"]-covidIndiaDF["Cured"]
covidIndiaDF


# ### Visualizations

# **Day wise figures**

# In[ ]:



from datetime import timedelta
dailyFigDF=covidIndiaDF.copy().groupby(["State/UnionTerritory","Date"]).agg({"Confirmed":"max", "Deaths":"max","Cured":"max", "Treated":"max"}).reset_index()

#prevDateFigDF=dailyFigDF[["Date", "Confirmed", "Deaths", "Cured","Treated"]]
dailyFigDF["Prev_Date"]=dailyFigDF["Date"]-timedelta(days=1)

covidIndiaDFMerged=pd.merge(dailyFigDF.copy(), dailyFigDF[["State/UnionTerritory","Date", "Prev_Date", "Confirmed", "Deaths", "Cured","Treated"]].copy(),                             left_on=["State/UnionTerritory","Prev_Date"], right_on=["State/UnionTerritory","Date"], suffixes=["_today","_yesterday"], how="inner")
covidIndiaDFMerged["New_Cases"]=covidIndiaDFMerged["Confirmed_today"]-covidIndiaDFMerged["Confirmed_yesterday"]
covidIndiaDFMerged["New_Deaths"]=covidIndiaDFMerged["Deaths_today"]-covidIndiaDFMerged["Deaths_yesterday"]
covidIndiaDFMerged=covidIndiaDFMerged.sort_values(by="Date_today", ascending=True)[covidIndiaDFMerged["Date_today"]>="2020-03-15"]

fig, axs=plt.subplots(1,2,figsize=(22,6))
axs[0].set_xlabel("Dates")
axs[0].set_ylabel("Number")
axs[0].set_title("No. of New Cases")
axs[0].plot(covidIndiaDFMerged["Date_today"].drop_duplicates(), covidIndiaDFMerged.groupby("Date_today").agg({"New_Cases":"sum"}).reset_index()["New_Cases"])    
axs[0].legend()
axs[1].set_xlabel("Dates")
axs[1].set_ylabel("Number")
axs[1].set_title("No. of New Deaths")
axs[1].plot(covidIndiaDFMerged["Date_today"].drop_duplicates(), covidIndiaDFMerged.groupby("Date_today").agg({"New_Deaths":"sum"}).reset_index()["New_Deaths"])    
axs[1].legend()

# states=covidIndiaDFMerged["State/UnionTerritory"].drop_duplicates()
states=covidIndiaDFMerged[covidIndiaDFMerged["State/UnionTerritory"]!="Cases being reassigned to states"].        groupby("State/UnionTerritory").agg({"Confirmed_today" : "max"}).reset_index().        sort_values(by="Confirmed_today", ascending=False).head(10)["State/UnionTerritory"]

fig, axs=plt.subplots(1,2,figsize=(22,6))
axs[0].set_xlabel("Dates")
axs[0].set_ylabel("Number")
axs[0].set_title("No. of New Cases Statewise")
for state in states:
    axs[0].plot(covidIndiaDFMerged[covidIndiaDFMerged["State/UnionTerritory"]==state]["Date_today"],            covidIndiaDFMerged[covidIndiaDFMerged["State/UnionTerritory"]==state]["New_Cases"], label=state)    
axs[0].legend()
    
axs[1].set_xlabel("Dates")
axs[1].set_ylabel("Number")    
axs[1].set_title("No. of Deaths Statewise")    
for state in states:
    axs[1].plot(covidIndiaDFMerged[covidIndiaDFMerged["State/UnionTerritory"]==state]["Date_today"],            covidIndiaDFMerged[covidIndiaDFMerged["State/UnionTerritory"]==state]["New_Deaths"], label=state)    
axs[1].legend()


# #### West Bengal

# In[ ]:


# fig, axs=plt.subplots(1,2,figsize=(22,6))
# axs[0].set_xlabel("Dates")
# axs[0].set_ylabel("Number")
# axs[0].set_title("No. of New Cases")
# axs[0].bar(covidIndiaDFMerged["Date_today"].drop_duplicates(), covidIndiaDFMerged.groupby("Date_today").agg({"New_Cases":"sum"}).reset_index()["New_Cases"])    
# axs[0].legend()
# axs[1].set_xlabel("Dates")
# axs[1].set_ylabel("Number")
# axs[1].set_title("No. of New Deaths")
# axs[1].bar(covidIndiaDFMerged["Date_today"].drop_duplicates(), covidIndiaDFMerged.groupby("Date_today").agg({"New_Deaths":"sum"}).reset_index()["New_Deaths"])    
# axs[1].legend()

# # states=covidIndiaDFMerged["State/UnionTerritory"].drop_duplicates()
# states=covidIndiaDFMerged[covidIndiaDFMerged["State/UnionTerritory"]!="Cases being reassigned to states"].\
#         groupby("State/UnionTerritory").agg({"Confirmed_today" : "max"}).reset_index().\
#         sort_values(by="Confirmed_today", ascending=False).head(10)["State/UnionTerritory"]

fig, axs=plt.subplots(1,2,figsize=(22,6))
axs[0].set_xlabel("Dates")
axs[0].set_ylabel("Number")
axs[0].set_title("No. of New Cases West Bengal")
axs[0].plot(covidIndiaDFMerged[covidIndiaDFMerged["State/UnionTerritory"]=="West Bengal"]["Date_today"],            covidIndiaDFMerged[covidIndiaDFMerged["State/UnionTerritory"]=="West Bengal"]["New_Cases"])    
axs[0].legend()
    
axs[1].set_xlabel("Dates")
axs[1].set_ylabel("Number")    
axs[1].set_title("No. of Deaths in West Bengal")    
axs[1].plot(covidIndiaDFMerged[covidIndiaDFMerged["State/UnionTerritory"]=="West Bengal"]["Date_today"],        covidIndiaDFMerged[covidIndiaDFMerged["State/UnionTerritory"]=="West Bengal"]["New_Deaths"])    
axs[1].legend()


# **Coutry and Statewise Tally**

# In[ ]:


fig, axs=plt.subplots(1,2,figsize=(22,6))

#Country Wise Tally
countryWise=covidIndiaDF.groupby("Date").agg({
    "Cured" : "sum",
    "Deaths" : "sum",
    "Confirmed" : "sum",
    "Treated" : "sum"
}).sort_values(by="Date", ascending=True).reset_index()

axs[0].set_title("Country Wise Tally")
axs[0].plot(countryWise["Date"], countryWise["Confirmed"], label="Confirmed Cases", color="blue")
axs[0].plot(countryWise["Date"], countryWise["Cured"], label="Cured Cases", color="green")
axs[0].plot(countryWise["Date"], countryWise["Deaths"], label="Deaths", color="red")
axs[0].plot(countryWise["Date"], countryWise["Treated"], label="Active", color="yellow")
axs[0].legend()
axs[0].set_xlabel("Dates")
axs[0].set_ylabel("Counts")

# State wise tally
stateWise=covidIndiaDF[covidIndiaDF["State/UnionTerritory"]!="Cases being reassigned to states"].groupby("State/UnionTerritory").agg({
    "Cured" : "max",
    "Deaths" : "max",
    "Confirmed" : "max"
}).sort_values(by="Confirmed", ascending=False).reset_index().head(10)
axs[1].set_title("State Wise Tally")
axs[1].barh(stateWise["State/UnionTerritory"], stateWise["Confirmed"], label="Confirmed Cases", color="blue")
axs[1].barh(stateWise["State/UnionTerritory"], stateWise["Cured"], label="Cured", color="green")
axs[1].barh(stateWise["State/UnionTerritory"], stateWise["Deaths"], label="Deaths", color="red")
# axs[1].barh(stateWise["State/UnionTerritory"], stateWise["Treated"], label="Treated", color="yellow")
axs[1].legend()
axs[1].set_xlabel("Counts")
axs[1].set_ylabel("State / UnionTerritory")
axs[1].invert_yaxis()


# **India Stats**

# In[ ]:


lastDate=covidIndiaDF.agg({"Date" : "max"})

print("*** Country Wise Stats ***")
tmpDF=covidIndiaDF[covidIndiaDF["Date"]==lastDate[0]].agg({
    "Cured" : "sum",
    "Deaths" : "sum",
    "Confirmed" : "sum",
    "Treated" : "sum"
})
tmpDF["Death_Percentage"]=tmpDF["Deaths"]/tmpDF["Confirmed"]*100
print(tmpDF)
print("*** State Wise Stats. For highest death rates ***")
stateWise["Death_Percentage"]=stateWise["Deaths"]/stateWise["Confirmed"]*100
print(stateWise.sort_values(by=["Death_Percentage","Confirmed"], ascending=False).head(10))


# **Death Ratio From 15th March**

# In[ ]:


covidDeathRatioDF=covidIndiaDF.groupby("Date").agg({"Confirmed" : "sum", "Deaths" : "sum"}).reset_index()
covidDeathRatioDF["Death_Ratio"]=covidDeathRatioDF["Deaths"]/covidDeathRatioDF["Confirmed"]*100
covidDeathRatioDF=covidDeathRatioDF[covidDeathRatioDF["Death_Ratio"]!=np.inf].sort_values(by="Date", ascending=True)
covidDeathRatioDF=covidDeathRatioDF[covidDeathRatioDF["Date"]>="2020-03-15"]

#Country Wise
fig, axs=plt.subplots(1,2,figsize=(22,6))
axs[0].set_title("Country Wise Death Percentage")
axs[0].bar(covidDeathRatioDF["Date"], covidDeathRatioDF["Death_Ratio"], label="Country Wise Death Ratio")
axs[0].set_xlabel("Date")
axs[0].set_ylabel("Death Percentage")

#State Wise
axs[1].set_title("State Wise Death Percentage")
# Top 10 States
top10=stateWise["State/UnionTerritory"].head(8)
order=1
for state in top10:
    #print("Plotting : ",state)
    axs[1].bar(covidIndiaDF[(covidIndiaDF["State/UnionTerritory"]==state) & (covidIndiaDF["Date"]>="2020-03-15")].sort_values("Date")["Date"],              covidIndiaDF[(covidIndiaDF["State/UnionTerritory"]==state) & (covidIndiaDF["Date"]>="2020-03-15")].sort_values("Date")["Death_Ratio"],              label=state)
#     order=order+1
#     covidIndiaDF[(covidIndiaDF["State/UnionTerritory"]==state) & (covidIndiaDF["Date"]>="2020-03-15")].\
#         plot.bar(x=covidIndiaDF.sort_values("Date")["Date"],\
#                  y=covidIndiaDF.sort_values("Date")["Death_Ratio"], \
#                  ax=axs[1], \
#                  stacked=True,\
#                  label=state)
axs[1].legend()
axs[1].set_xlabel("Dates")
axs[1].set_ylabel("Death Percentage")
plt.show()


# ## ** World Data**
# 
# Getting the world data

# In[ ]:


worldDataMaster=pd.read_csv("https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv")
worldDataMaster.head(10)


# **Preprocessing**

# In[ ]:


worldDF=worldDataMaster[worldDataMaster["location"]!="World"].copy()
worldDF["Country"]=worldDF["location"]
worldDF["Confirmed"]=worldDF["total_cases"]
worldDF["Deaths"]=worldDF["total_deaths"]
worldDF["Cured_Or_Being_Treated"]=worldDF["Confirmed"]-worldDF["Deaths"]
worldDF["Date"]=pd.to_datetime(worldDF["date"], format="%Y-%m-%d")

worldDF=worldDF.groupby(["Date","Country"]).agg({
#     "Cured" : "max",
    "Deaths" : "max",
    "Confirmed" : "max"
}).reset_index()

worldDF["Death_Ratio"]=worldDF["Deaths"]/worldDF["Confirmed"]*100
# worldDF["Treated"]=worldDF["Confirmed"]-worldDF["Deaths"]-worldDF["Cured"]
worldDF=worldDF.replace(np.NaN, 0)
worldDF.head(10)


# ### **World Wide Visualizations**

# In[ ]:



#Country Wise Tally
worldWise=worldDF.groupby("Date").agg({
#     "Cured" : "sum",
    "Deaths" : "sum",
    "Confirmed" : "sum",
#     "Treated" : "sum"
}).sort_values(by="Date", ascending=True).reset_index()

plt.figure(figsize=(22,7))
plt.title("World Tally")
plt.plot(worldWise["Date"], worldWise["Confirmed"], label="Confirmed Cases", color="blue")
# plt.plot(worldWise["Date"], worldWise["Cured"], label="Cured Cases", color="green")
plt.plot(worldWise["Date"], worldWise["Deaths"], label="Deaths", color="red")
# plt.plot(worldWise["Date"], worldWise["Treated"], label="Treated", color="yellow")
plt.legend()
plt.xlabel("Dates")
plt.ylabel("Counts")
plt.show()

plt.figure(figsize=(22,7))
# Country wise tally top 20
countryWise=worldDF.groupby("Country").agg({
#     "Cured" : "max",
    "Deaths" : "max",
    "Confirmed" : "max"
}).sort_values(by="Confirmed", ascending=False).reset_index().head(20)
plt.title("Country Wise Tally")
plt.barh(countryWise["Country"], countryWise["Confirmed"], label="Confirmed Cases", color="blue")
# plt.barh(countryWise["Country"], countryWise["Cured"], label="Cured", color="green")
plt.barh(countryWise["Country"], countryWise["Deaths"], label="Deaths", color="red")
plt.legend()
plt.xlabel("Counts")
plt.ylabel("Country / Region")
plt.gca().invert_yaxis()
# countryWise


# **World Stats**

# In[ ]:


lastDate=worldDF.agg({"Date" : "max"})

print("*** World Wide Stats ***")
tmpDF=worldDF[worldDF["Date"]==lastDate[0]].agg({
    "Deaths" : "sum",
    "Confirmed" : "sum",
})
tmpDF["Death_Percentage"]=tmpDF["Deaths"]/tmpDF["Confirmed"]*100
print(tmpDF)
print("*** Country Wise Stats. For highest death rates ***")
countryWise["Death_Percentage"]=countryWise["Deaths"]/countryWise["Confirmed"]*100
print(countryWise.sort_values(by=["Death_Percentage","Confirmed"], ascending=False).head(20))


# **Death Ratio**

# In[ ]:


worldCovidDeathRatioDF=worldDF.copy().groupby("Date").agg({"Confirmed" : "sum", "Deaths" : "sum"}).reset_index()
worldCovidDeathRatioDF["Death_Ratio"]=worldCovidDeathRatioDF["Deaths"]/worldCovidDeathRatioDF["Confirmed"]*100
worldCovidDeathRatioDF=worldCovidDeathRatioDF[worldCovidDeathRatioDF["Death_Ratio"]!=np.inf].sort_values(by="Date", ascending=True)
worldCovidDeathRatioDF=worldCovidDeathRatioDF.replace(np.NaN, 0)
#worldCovidDeathRatioDF=worldDF[worldDF["Date"]>="2020-03-15"]

#World Wise
plt.figure(figsize=(22,7))
plt.title("World Wide Death Percentage")
plt.bar(worldCovidDeathRatioDF["Date"], worldCovidDeathRatioDF["Death_Ratio"], label="World Wide Death Ratio")
plt.xlabel("Date")
plt.ylabel("Death Percentage")

#Country Wise
plt.figure(figsize=(22,7))
plt.title("Country Wise Death Percentage")
# Top 20 Countries
top20Countries=countryWise["Country"]
for country in top20Countries:
    #print("Plotting : ",state)
    plt.bar(worldDF[(worldDF["Country"]==country) & (worldDF["Date"]>="2020-03-15")].sort_values("Date")["Date"],              worldDF[(worldDF["Country"]==country) & (worldDF["Date"]>="2020-03-15")].sort_values("Date")["Death_Ratio"], label=country)
plt.legend()
plt.xlabel("Dates")
plt.ylabel("Death Percentage")
plt.show()


# 
