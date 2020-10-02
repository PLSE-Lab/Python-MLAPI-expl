#!/usr/bin/env python
# coding: utf-8

# I evaluated the Los Angeles International Airport data according to weight of cargo that moved in and out through the Airport. This analysis includes date, type of cargo, type of flights.
# 
# **Content:**
# 1. [Data Evaluation](#1)
# 1. [Freight Cargo:](#2)
#      1. [International or Domestic Freight](#3)
#      1. [Arrival or Departure Freight](#4)
#      1. [Freight Cargo with all Features](#5)
#  
# 1. [Mail Cargo:](#6)
#      1. [International or Domestic Mail](#7)
#      1. [Arrival or Departure Mail](#8)
#      1. [Mail Cargo with all Features](#9)
# 
# 1. [Conclusion](#10)

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math  
import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


data=pd.read_csv("../input/los-angeles-international-airport-air-cargo-volume.csv")


# <a id="1"></a> <br>
# **1. DATA EVALUATION**

# In[ ]:


data.info()


# Firstly we have 6 features in total. Date Extract Date, Report Period, Arrival or Departure, Domestic or International, Cargo Type and Air Cargo Tons, Every feature contains 1240 rows.

# In[ ]:


data.head()


# In[ ]:


data.dtypes


# In this data set, only the data type of the air cargo tons feature is integer and the rest of them are object.

# In[ ]:


data.describe()


# When we look at the table we see that the average cargo value is 21150 tons. Minimum cargo is 454 tons and maximum cargo is 82352 tons. We can see the histogram chart below.

# In[ ]:


data.plot(kind="hist",y="AirCargoTons",figsize=(15,5),bins=30,normed=True)
plt.xlabel("Tons")
plt.ylabel('Probability')
plt.plot()


# The following operations were performed to better analyze the data set. Firstly we drop Data Extract Date column. Because we want to evaluate the data according to their report period time and then we change the data types Report Period as datetime, arrival_departure, domestic_international and cargotype as category. 

# In[ ]:


data=data.drop(["DataExtractDate"],axis=1)


# In[ ]:


data.ReportPeriod=pd.to_datetime(data.ReportPeriod)
data["Arrival_Departure"]=data["Arrival_Departure"].astype("category")
data["Domestic_International"]=data["Domestic_International"].astype("category")
data["CargoType"]=data["CargoType"].astype("category")


# In[ ]:


data.dtypes


# Now we see the data types are changed.

# In[ ]:


print(data.index.name)


# Data index does not have any name. We used to following method to assign date to data index.

# In[ ]:


data1=data.set_index("ReportPeriod")


# In[ ]:


data1.head()


# We take the average all air cargo tones from 2006 to 2018 and showed them in figure.

# In[ ]:


data1.resample("A").mean()


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(data1.resample("A").mean(),color="b",label="Average Air Cargo Tons")
plt.grid()
plt.legend()
plt.xlabel("Year")
plt.ylabel("Ton")
plt.title("Average Air Cargo Tons According to Years")


# As seen in the graph, the minimum average cargo tons was moved in or out through the airport in 2010. And after 2015 this amount has always increased.

# When we resample the data by month, we can also see which month the minimum and maximum weight of cargo were moved.

# In[ ]:


month= data1.resample("M").mean()
print(month.min())
print(month.idxmin())


# In[ ]:


print(month.max())
print(month.idxmax())


# The minimum average aircargo is moved on February 2009 and the weight is 14150.125 tons. Also the maximum average is 27055.375 and moved on  October 2018.And a line chart is shown below.

# In[ ]:


plt.figure(figsize=(15,8))
plt.plot(data1.resample("M").mean(),color="g",label="Average Air Cargo Tons",linewidth=2)
plt.grid()
plt.legend()
plt.xlabel("Year")
plt.ylabel("Ton")
plt.title("Average Air Cargo Tons According To Months")
plt.show()


# However, we have seen that the lowest weight in 2010 is in the chart. This is because weights are counted at the end of dates. And also we said that after 2015 cargo weights are increased. In the month chart we see that, there are ups and downs. The month-by-month analysis gives more detailed and more accurate information. Of course, these  weights are shown regardless of the any features (domestic or international, cargo type etc.). In order to better examine the data, we grouped according to all the features below.

# In[ ]:


data2=data.set_index(["ReportPeriod","Arrival_Departure","Domestic_International","CargoType"])


# In[ ]:


data2.head(20)


# As can be seen from the table, the weight of freight type cargoes is much more than mail type cargo. In the next analysis, we will examine two types of cargo separately. Following table shows maximum and minimum cargo weights according to their cargo types and shows their dates.

# In[ ]:


cargotype=pd.concat([data1.groupby("CargoType")["AirCargoTons"].idxmin(),data1.groupby("CargoType")["AirCargoTons"].min(),data1.groupby("CargoType")["AirCargoTons"].idxmax(),data1.groupby("CargoType")["AirCargoTons"].max()],axis=1)
cargotype.columns.values[0:4]="Minimum Weight Date","Minimum Weight","Maximum Weight Date","Maximum Weight"
cargotype


# In[ ]:


freight=data1[data1.CargoType=="Freight"]
mail=data1[data1.CargoType=="Mail"]


# <a id="2"></a> <br>
#  **FREIGHT CARGO TYPE**

# In[ ]:


freight.describe()


# There are 620 freight type cargoes in the data set. Average weight is 40558 ton and minimum weight 23020 tons and maximum weight is 82352 tons. The histogram of the probability of the weights of freight type cargos moved is shown in below.

# In[ ]:


fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(15,5))
freight.plot(kind="hist",bins=30,color="tomato",normed=True,ax=axes[0],title="Freight Cargo Probability")
freight.plot(kind="hist",bins=30,color="tomato",normed=True,ax=axes[1],cumulative=True,title="Freight Cargo Cumulative Probability",label="Tons")
plt.show()


# And now lets see weights of cargos according to years and months.

# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(freight.resample("A").mean(),color="b",label="Average Air Cargo Tons")
plt.grid()
plt.legend()
plt.xlabel("Year")
plt.ylabel("Ton")
plt.title("Average Freight Type Air Cargo Weights According To Years")
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
plt.plot(freight.resample("M").mean(),color="r",label="Average Air Cargo Tons")
plt.grid()
plt.legend()
plt.xlabel("Year")
plt.ylabel("Ton")
plt.title("Average Freight Type Air Cargo Weights According To Months")
plt.show()


# We can see that there is no much difference between the graphs drawn without considering the cargo type and the freight type cargo graph.

# In[ ]:


month_freight= freight.resample("M").mean()
print(month_freight.max())
print(month_freight.idxmax())


# In[ ]:


print(month_freight.min())
print(month_freight.idxmin())


# However minimum average cargo weight was moved on January 2009 and the weight is 26936.75 tons. And maximum weight was moved on October 2018 and weight is 52012 tons. 

# <a id="3"></a> <br>
# > *** International or Domestic Freight***

# Now lets examine the freight cargo types according to domestic or international.

# In[ ]:


freight.groupby(["Domestic_International"]).describe()


# Average domestic freight cargo weight is 33018 tons. Minimum is 23020 tons and maximum is 44584 tons. 
# 
# Average international freight cargo weight is 48098 tons. Minimum is 23569 tons and maximum is 82352 tons. 
# 
# Lets see the date these cargos were moved.

# In[ ]:


print("maximum=",freight.groupby(["Domestic_International"]).idxmax())
print("minimum=",freight.groupby(["Domestic_International"]).idxmin())


# Minimum weights of domestic freight cargo is moved on 1st February 2009 and maximum weight is moved on 1st March 2006. Minimum weights of international freight cargo is moved on 1st January 2009 and maximum weight is moved on 1st September 2018. Now lets shown the domestic and international freight cargo weights according to resampling by months.

# In[ ]:


di=pd.DataFrame(dict(list(freight.groupby("Domestic_International")['AirCargoTons'])))
di.head()


# In[ ]:


di.resample("M").mean().plot(figsize=(15,8),title="Freight Cargo Tons According to Domestic or International",grid=True,linestyle="-.",marker="8",markersize=3,color=["rebeccapurple","crimson"])
plt.ylabel("Tons")
plt.show()


# We see that the weight of international freight cargoes moved is higher than domestic freight cargoes and there are more up and downs compared to domestic.

# Finally let's use scatter diagram to visualize whether there is a connection between them.

# In[ ]:


di.plot(kind="scatter",x="Domestic",y="International",figsize=(10,8),color="red",alpha=0.5,grid=True,title="Scatter Diagram of International and Domestic Freight Cargo",marker="*",s=120)
plt.show()


# In[ ]:


di.corr()


# Looks like there's not much of correlation between them.

# <a id="4"></a> <br>
# > ***Arrival or Departure Freight***

# Now lets examine the freight cargo types according to arrival or departure.

# In[ ]:


freight.groupby("Arrival_Departure").describe()


# Average arrival freight cargo weight is 43344.48 tons. Minimum is 23020 tons and maximum is 82352 tons.
# 
# Average departurel freight cargo weight is 37772.27 tons. Minimum is 23569 tons and maximum is 59002 tons. 
# 
# Lets see the date these cargos were moved.

# In[ ]:


print("maximum",freight.groupby("Arrival_Departure").idxmax())
print("minimum",freight.groupby("Arrival_Departure").idxmin())


# Minimum weights of arrival freight cargo is moved on 1st February 2009 and minimum weights of departure freight cargo is on 1st January 2009 and for maximum weight of arrival freight cargo is moved on 1st September 2018 and for departure, it is on 1st August 2018.

# In[ ]:


ad=pd.DataFrame(dict(list(freight.groupby('Arrival_Departure')['AirCargoTons'])))
ad.head()


# The arrival and departure freight cargo weights according to resampling by months.

# In[ ]:


ad.resample("M").mean().plot(figsize=(15,8),title="Freight Cargo Weights According to Arrival or Departure",grid=True,marker='o', linestyle='dashed',markersize=5)
plt.ylabel("Tons")


# 

# We can say that arrival freight weights of cargoes is much more than departures. Scatter diagram is shown below.

# In[ ]:


ad.plot(kind="scatter",x="Arrival",y="Departure",figsize=(10,8),color="maroon",alpha=0.5,grid=True,title="Scatter Diagram of Arrival and Departure Freight Cargo",marker="d",s=120)
plt.show()


# In[ ]:


ad.corr()


# We can say when we look at the correlation and scatter diagrams there is a highly positive correlation between arrival and departure freight cargo weights.

# <a id="5"></a> <br>
# > ***Freight Cargo with all Features***

# Lets examine these arrival or departure cargoes with domestic and international

# In[ ]:


freight.drop(["CargoType"],axis=1,inplace=True)


# In[ ]:


data3=pd.DataFrame(dict(list(freight.groupby(['Arrival_Departure', 'Domestic_International'])["AirCargoTons"])))
data3.head()


# In[ ]:


table=pd.concat([data3.idxmax(),data3.max(),data3.idxmin(),data3.min()],axis=1)
table.columns=table.columns.astype("str")
table.rename(columns={"0":"Maximum Date","1":"Maximum Weight","2":"Minimum Date","3":"Minimum Weight"},inplace=True)
table


# We have seen that the maximum cargo weight previously moved at the airport was 82352. As can be seen from the table above, the heaviest cargo transported at Los Angeles Airport is international and arrival cargo. It also moved in September 2018. 
# 
# Minimum freight cargo weight is 23020 and its moved on February 2009 and its domestic and arrival.

# In the line chart below, we can see the arrival and departure freight cargo with international or domestic according to resampling by months.

# In[ ]:


fig,ax=plt.subplots(nrows=2,ncols=1, figsize=(15,15))
data3.Arrival.resample("M").mean().plot(ax=ax[0],title="Arrival Freight Cargo",color=["purple","orange"],linewidth=3,grid=True)
plt.ylabel("Tons")
data3.Departure.resample("M").mean().plot(ax=ax[1],title="Departure Freight Cargo",color=["deeppink","darkcyan"],linewidth=3,grid=True)
plt.ylabel("Tons")
plt.show()


# In both departure and arrival cargoes we can see international cargo weights are generally more than domestic. Especially for arrivals.

# In the following scatter diagram and table, we can see how the properties are related to each other.

# In[ ]:


fig,ax=plt.subplots(nrows=2,ncols=1, figsize=(15,15))
data3.Arrival.plot(kind="scatter",x="Domestic",y="International",ax=ax[0],title="Arrival Freight Cargo",color="purple",marker='P', s=80,grid=True)
data3.Departure.plot(kind="scatter",x="Domestic",y="International",ax=ax[1],title="Departure Freight Cargo",color="deeppink",marker='P',s=80,grid=True)
plt.show()


# In[ ]:


data3.corr()


# <a id="6"></a> <br>
# > **MAIL CARGO**

# In[ ]:


mail.describe()


# 620 mail cargoes were moved between 2006 and 2019 at this airport. The maximum, minimum and weight of these cargoes are 6755, 454, 1742 tons respectively. The histogram of the probability of the weights of mail type cargos moved is shown in below.

# In[ ]:


fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(15,5))
mail.plot(kind="hist",bins=30,color="yellowgreen",normed=True,ax=axes[0],title="Mail Cargo Probability")
mail.plot(kind="hist",bins=30,color="yellowgreen",normed=True,ax=axes[1],cumulative=True,title="Mail Cargo Cumulative Probability",label="Tons")
plt.xlabel("Tons")
plt.show()


# In the figure below, weights of mail cargoes is shown according to years and months.

# In[ ]:


mail.resample("A").mean().plot(color="g",figsize=(15,5),title="Average Mail Type Air Cargo Weights According To Years",grid=True,linewidth=3)
plt.xlabel("Year")
plt.ylabel("Ton")
plt.show()


# The weight of transported mail cargoes decreased in 2007 and 2009. After 2014, weights increased gradually. And it seems in 2017 maximum weight is moved. 

# In[ ]:


mail.resample("M").mean().plot(color="r",figsize=(18,8),title="Average Mail Type Air Cargo Weights According To Months",grid=True,linewidth=3)
plt.xlabel("Year")
plt.ylabel("Ton")
plt.show()


# When we look at the mail cargoes weights there are some up and downs afer 2015. 

# In[ ]:


month_mail= mail.resample("M").mean()
print(month_mail.min())
print(month_mail.idxmin())


# Minimum cargo weight was moved on July 2007 and the weight is 1029 tons by months. 

# In[ ]:


print(month_mail.max())
print(month_mail.idxmin())


# And maximum weight was moved on July 2007 and weight is 3723.25 tons. 

# <a id="7"></a> <br>
# > ***International or Domestic Mail***

# In[ ]:


mail.groupby(["Domestic_International"]).describe()


# Average domestic mail cargo weight is 2286.76 tons. Minimum is 912 tons and maximum is 6755 tons. 
# 
# Average international mail cargo weight is 1197.45 tons. Minimum is 454 tons and maximum is 4497 tons. 
# 
# Lets see the date these cargos were moved.

# In[ ]:


print("maximum=",mail.groupby(["Domestic_International"]).idxmax())
print("minimum=",mail.groupby(["Domestic_International"]).idxmin())


# Minimum weights of domestic mail cargo is moved on 1st August 2007 and maximum weight is moved on 1st December 2017. Minimum weights of international mail cargo is moved on 1st November 2010 and maximum weight is moved on 1st March 2013. Now lets shown the domestic and international mail cargo weights according to resampling by months.

# In[ ]:


di_mail=pd.DataFrame(dict(list(mail.groupby("Domestic_International")['AirCargoTons'])))
di_mail.head()


# In[ ]:


di_mail.resample("M").mean().plot(figsize=(18,10),title="Mail Cargo Weights According to Domestic or International",linewidth=3,linestyle=":",marker="D",markersize=5,grid=True,color=["firebrick","steelblue"])
plt.ylabel("Tons")
plt.show()


# It seems domestic mail cargo weights are much more than international. Also in scatter diagram and correlation table we can see international and domestic relations.

# In[ ]:


di_mail.plot(kind="scatter",x="Domestic",y="International",figsize=(10,8),color="dodgerblue",alpha=0.5,grid=True,title="Scatter Diagram of International and Domestic Mail Cargo",marker="^",s=80)
plt.show()


# In[ ]:


di_mail.corr()


# <a id="8"></a> <br>
# > ***Arrival or Departure Mail***

# Now lets examine the mail cargo according to arrival or departure.

# In[ ]:


mail.groupby("Arrival_Departure").describe()


# Average arrival mail cargo weight is 1482.57 tons. Minimum is 454 tons and maximum is 4485 tons.
# 
# Average departurel mail cargo weight is 2001.63 tons. Minimum is 849 tons and maximum is 6755 tons. 
# 
# Lets see the date these cargos were moved.

# In[ ]:


print("maximum",mail.groupby("Arrival_Departure").idxmax())
print("minimum",mail.groupby("Arrival_Departure").idxmin())


# Minimum weights of arrival mail cargo is moved on 1st November 2010 and maximum weight is moved on 1st December 2017. Minimum weights of departure mail cargo is moved on 1st March 2017 and maximum weight is moved on 1st December 2017. Now lets shown the arrival and departure mail cargo weights according to resampling by months.

# In[ ]:


ad_mail=pd.DataFrame(dict(list(mail.groupby('Arrival_Departure')['AirCargoTons'])))
ad_mail.head()


# In[ ]:


ad_mail.resample("M").mean().plot(figsize=(15,8),title="Mail Cargo Weights According to Arrival or Departure",grid=True,marker='p',linestyle="--",markersize=6,colors=["coral","olivedrab"])
plt.ylabel("Tons")
plt.show()


# Departure cargo weights look like much more than arrival weights and scatter diagram is shown below.

# In[ ]:


ad_mail.plot(kind="scatter",x="Arrival",y="Departure",figsize=(10,8),color="peru",alpha=0.5,grid=True,title="Scatter Diagram of Arrival and Departure Mail Cargo",marker="s",s=80)
plt.show()


# In[ ]:


ad_mail.corr()


# As with freight type cargoes, mail type cargo also shows a high correlation between arrival and departure.

# <a id="9"></a> <br>
# ***Mail Cargo with all Features***

# Lets examine mail cargo type as arrival or departure cargoes with domestic and international

# In[ ]:


mail.drop(["CargoType"],axis=1,inplace=True)


# In[ ]:


data4=pd.DataFrame(dict(list(mail.groupby(['Arrival_Departure', 'Domestic_International'])["AirCargoTons"])))
data4.head()


# In[ ]:


table_mail=pd.concat([data4.idxmax(),data4.max(),data4.idxmin(),data4.min()],axis=1)
table_mail.columns=table_mail.columns.astype("str")
table_mail.rename(columns={"0":"Maximum Date","1":"Maximum Weight","2":"Minimum Date","3":"Minimum Weight"},inplace=True)
table_mail


# We have seen that the minimum cargo weight previously moved at the airport was 454. As can be seen from the table above, the lightest cargo transported at Los Angeles Airport is international and arrival cargo. It also moved in November 2010. 
# 
# Also maximum mail cargo is moved in December 2017. Its weight is 6755 tons and it's domestic and departure.

#  The arrival and departure freight cargo with international or domestic according to resampling by months are shown below with line charts.

# In[ ]:


fig,ax=plt.subplots(nrows=2,ncols=1, figsize=(18,18))
data4.Arrival.resample("M").mean().plot(ax=ax[0],title="Arrival Mail Cargo",color=["cornflowerblue","navy"], linewidth=3,grid=True)
plt.ylabel("Tons")
data4.Departure.resample("M").mean().plot(ax=ax[1],title="Departure Mail Cargo",color=["hotpink","purple"],linewidth=3,grid=True)
plt.ylabel("Tons")


# In the figures, generally domestic mail cargo weights more than international cargo weigts both for arrival and departure. But for arrivals in 2008 and 2013 international cargo weights exceed domestic weights. And also for departures at the beginning of 2013 international cargo weight reach the maximum point. 

# In the following scatter diagram and table, we can see how the properties are related to each other.

# In[ ]:


fig,ax=plt.subplots(nrows=2,ncols=1, figsize=(15,15))
data4.Arrival.plot(kind="scatter",x="Domestic",y="International",ax=ax[0],title="Arrival Mail Cargo",color="mediumvioletred",marker=7,alpha=0.5, s=120,grid=True)
data4.Departure.plot(kind="scatter",x="Domestic",y="International",ax=ax[1],title="Departure Mail Cargo",color="darkslategrey",alpha=0.5,marker=10,s=120,grid=True)
plt.show()


# In[ ]:


data4.corr()


# <a id="10"></a> <br>
# **4. CONCLUSION**

# As a result,
# 
# Cargo weights received and send at the begining of 2009 were very low after 2015 they increased gradually.
# 
# The heaviest cargo was moved with domestic and arrival lines in September 2018. It is 82353 tons and its cargo type is freight.
# 
# The lightest cargo was moved with international and arrival lines in November 2010. It is 450 tons and its cargo type is mail.
# 
# Also there is a highly positive correlation between arrival and departure cargo weights for both freight and mail cargo types.
# 
# We can say that from the analysis, generally people prefer to send or receive freight cargoes with international lines(The reason may be due to import or export and especially for arrivals domestic cargoes never exceed international freight cargoes and for departure before 2010 international and domestic cargoes were close to each other.) and for mail cargoes they generally send or receive them with domestic lines (The reason may be mail cargoes are more personal)(However for arrivals its exceed domestic in 2008 and 2013  and also same thing can be for departures in 2013)
