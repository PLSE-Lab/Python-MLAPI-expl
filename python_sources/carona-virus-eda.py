#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")
df[:5]


# In[ ]:


df1 = df.drop("Sno",axis=1)


# In[ ]:


for i in df1.columns:
    print("Name of Column :",i)
    print("Total No. of unique value:",len(df1[i].unique()))
    print(df1[i].unique(),"\n")


# In[ ]:


#df["Province/State"].value_counts()
"1/22/2020 12:00".split()[0]
df1['Date'] = df1['Last Update'].apply(lambda x: x.split()[0])
df1 = df1.drop("Last Update",axis=1)
df1[:3]


# In[ ]:


from datetime import datetime
date_format = "%m/%d/%Y"


# In[ ]:


df1["Date"] = df1["Date"].replace("1/23/20","1/23/2020")
df1["days interval"] = df1["Date"].apply(lambda x:(datetime.strptime(x, date_format).day)
                                         -(datetime.strptime(df1["Date"].min(), date_format).day))

#df1["percent of Recovered"]= (df1["Recovered"]/df1["Confirmed"])*100
#df1["percent of Deaths"]= (df1["Deaths"]/df1["Confirmed"])*100
df1[:5]


# In[ ]:


df1.info()


# # Data Visualization

# # On Time Period

# In[ ]:


intreval = df1.groupby("days interval").sum().reset_index()[["days interval","Confirmed","Deaths","Recovered"]]
intreval["percent of Recovered"]= (intreval["Recovered"]/intreval["Confirmed"])*100
intreval["percent of Deaths"]= (intreval["Deaths"]/intreval["Confirmed"])*100
intreval


# In[ ]:


sns.lineplot("days interval", "Confirmed",data=intreval)


# In[ ]:


intreval.plot("days interval",["percent of Recovered","percent of Deaths"])


# From this plot, at initial stage recovery rate is high than death rate. i.e.. if we diagoniza & treat as soon as possible recovery rate is high.
# &
# Disease is go on spreading at 

# In[ ]:


intreval["increasing Rate in Confirm"] = round(intreval["Confirmed"].diff(+1)/intreval["Confirmed"]*100,2)
intreval["increasing Rate in Death"] = round(intreval["Deaths"].diff(+1)/intreval["Deaths"]*100,2)
intreval["increasing Rate in Recovered"] = round(intreval["Recovered"].diff(+1)/intreval["Recovered"]*100,2)
intreval


# In[ ]:


sns.lineplot("days interval","increasing Rate in Confirm",data=intreval)


# #### Rate of Confirmed paitents is gradually reducing. It shows that spread rate is controlled.

# In[ ]:


intreval.plot("days interval",["increasing Rate in Recovered","increasing Rate in Death"])


# #### Rate of death & Rate of recovery 
# 
# * Death Rate is controlled
# * Recovered rate is too fluctuating

# In[ ]:


plt.figure(figsize=(15,8))
sns.countplot(df1["Country"])
plt.xticks(rotation=90)


# # Country Insights

# In[ ]:


cou = df1.groupby("Country").sum().reset_index()[["Country","Confirmed","Deaths","Recovered"]]
cou["percent of Recovered"]= (cou["Recovered"]/cou["Confirmed"])*100
cou["percent of Deaths"]= (cou["Deaths"]/cou["Confirmed"])*100
cou[:5]


# plt.figure(figsize=(15,8))
# sns.barplot("Country","percent of Recovered",data=cou)
# plt.xticks(rotation=90)

# In[ ]:


cou.plot("Country",["percent of Recovered","percent of Deaths"],kind='bar',figsize=(15,8))
plt.figure(figsize=(15,8))


# #### Death is high only at Mainland China. Patients here must be treated carefully.
# * Recover Percent is high in Thailand & it is about 25%.
# * Recover in Japan is above 10%.
# * Recover in Australia is above 5%.

# In[ ]:


cou.sort_values("Deaths",ascending=False)[:3]


# In[ ]:


print("Death On Mainland China is",round((802/33986)*100,2),"%")


# Hence, death is only high in Mainland China on other Country There is no death.

# # Mainlnad China

# In[ ]:


mc = df1[df1["Country"]=="Mainland China"]
mc[:5]


# # On Time Period

# In[ ]:


intreval = mc.groupby("days interval").sum().reset_index()[["days interval","Confirmed","Deaths","Recovered"]]
intreval["percent of Recovered"]= (intreval["Recovered"]/intreval["Confirmed"])*100
intreval["percent of Deaths"]= (intreval["Deaths"]/intreval["Confirmed"])*100
intreval


# In[ ]:


intreval.plot("days interval",["percent of Recovered","percent of Deaths"])


# From this plot, at initial stage recovery rate is high than death rate. i.e.. if we diagoniza & treat as soon as possible recovery rate is high.
# &
# Disease is go on spreading at 

# In[ ]:


intreval["increasing Rate in Confirm"] = round(intreval["Confirmed"].diff(+1)/intreval["Confirmed"]*100,2)
intreval["increasing Rate in Death"] = round(intreval["Deaths"].diff(+1)/intreval["Deaths"]*100,2)
intreval["increasing Rate in Recovered"] = round(intreval["Recovered"].diff(+1)/intreval["Recovered"]*100,2)
intreval


# In[ ]:


sns.lineplot("days interval","increasing Rate in Confirm",data=intreval)


# #### Rate of Confirmed paitents is gradually reducing. It shows that spread rate is controlled.

# In[ ]:


intreval.plot("days interval",["increasing Rate in Recovered","increasing Rate in Death"])


# In[ ]:


confirm = mc["Confirmed"].sum()
death = mc["Deaths"].sum()
rec = mc["Recovered"].sum()
print("Total No. of Confirmed :",confirm)
print("Total No. of Recovered :",rec)
print("Total No. of Deaths    :",death)
print("Percent of Recovered :",(round((rec/confirm)*100,2)),"%")
print("Percent of Deaths    :",(round((death/confirm)*100,2)),"%")


# Total Death is high in MC is 2.36% from affected person

# #### Rate of death & Rate of recovery 
# 
# * Death Rate is controlled
# * Recovered rate is too fluctuating

# # Thailand

# In[ ]:


th = df1[df1["Country"]=="Thailand"]


# In[ ]:


confirm = th["Confirmed"].sum()
death = th["Deaths"].sum()
rec = th["Recovered"].sum()
print("Total No. of Confirmed :",confirm)
print("Total No. of Recovered :",rec)
print("Total No. of Deaths    :",death)
print("Percent of Recovered :",(round((rec/confirm)*100,2)),"%")
print("Percent of Deaths    :",(round((death/confirm)*100,2)),"%")


# Hence, There is no death in Thailand & Recover rate is also good.

# # Given data

# In[ ]:


confirm = df1["Confirmed"].sum()
death = df1["Deaths"].sum()
rec = df1["Recovered"].sum()
print("Total No. of Confirmed :",confirm)
print("Total No. of Recovered :",rec)
print("Total No. of Deaths    :",death)
print("Percent of Recovered :",(round((rec/confirm)*100,2)),"%")
print("Percent of Deaths    :",(round((death/confirm)*100,2)),"%")


# # Insight from Dataset

# * Sperading of disease is effectively controlled over period of time
# * Death rate Reducing & Recover rate is fluctuating
# * death rate is high at starting only after it is reducing. so, immediate diagonize & treatment will help to recover
