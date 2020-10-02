#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory




# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/PakistanSuicideAttacks Ver 6 (10-October-2017).csv", encoding="latin1")


# In[ ]:


data.head()


# In[ ]:


data.drop("S#",inplace=True,axis=1)


# In[ ]:


data.describe()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.columns


# In[ ]:


total=data.isnull().sum().sort_values(ascending=False)
percentage=(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total,percentage],axis=1,keys=["Total","Percentage"])
missing_data


# In[ ]:


data["Date"]=data["Date"].str.replace("-"," ")


# In[ ]:


data.head()


# In[ ]:


day_name=data.Date.str.split(' ').str[0]
month=data.Date.str.split(' ').str[1]
day=data.Date.str.split(' ').str[2]
year=data.Date.str.split(" ").str[3]
data["Day_name"]=day_name
data["Month"]=month
data["Day"]=day
data["Year"]=year
data.head()


# In[ ]:


data["Islamic Date"]=data["Islamic Date"].str.replace(" al","-al")
data.head()


# In[ ]:


day=data["Islamic Date"].str.split(' ').str[0]
month=data["Islamic Date"].str.split(' ').str[1]
year=data["Islamic Date"].str.split(' ').str[2]
data["Islamic_Day"]=day
data["Islamic_Month"]=month
data["Islamic_Year"]=year
data.drop("Islamic Date",axis=1,inplace=True)
data.head()


# In[ ]:


cleaned_data = data[pd.notnull(data['Killed Max'])]
cleaned_data["Killed Max"].isnull().sum()
sns.distplot(cleaned_data["Killed Max"])
cleaned_data["Killed Max"].mean()


# In[ ]:


data["Killed Max"]=data["Killed Max"].fillna(data["Killed Max"].mean())
data["Killed Max"].isnull().sum()


# In[ ]:


cleaned_data = data[pd.notnull(data['Killed Min'])]
cleaned_data["Killed Min"].isnull().sum()
sns.distplot(cleaned_data["Killed Min"])
cleaned_data["Killed Min"].mean()


# In[ ]:


data["Killed Min"]=data["Killed Min"].fillna(data["Killed Min"].mean())
data["Killed Min"].isnull().sum()


# In[ ]:


data["Injured Max"].dtypes


# In[ ]:


data["Injured Max"]=data["Injured Max"].convert_objects(convert_numeric=True)
data["Injured Max"].dtypes


# In[ ]:


cleaned_data = data[pd.notnull(data['Injured Max'])]
cleaned_data["Injured Max"].isnull().sum()
sns.distplot(cleaned_data["Injured Max"])
cleaned_data["Injured Max"].mean()


# In[ ]:


data["Injured Max"]=data["Injured Max"].fillna(data["Injured Max"].mean())
data["Injured Max"].isnull().sum()


# In[ ]:


cleaned_data = data[pd.notnull(data['Injured Min'])]
cleaned_data["Injured Min"].isnull().sum()
sns.distplot(cleaned_data["Injured Min"])
cleaned_data["Injured Max"].mean()


# In[ ]:


data["Injured Min"]=data["Injured Min"].fillna(data["Injured Min"].mean())
data["Injured Min"].isnull().sum()


# In[ ]:


data["Killed_avg"]=(data["Killed Max"]+data["Killed Min"])/2


# In[ ]:


data["Injured_avg"]=(data["Injured Max"]+data["Injured Min"])/2


# In[ ]:


#How many people got killed and injured per year?
sns.barplot("Year","Killed_avg",data=data)
#plt.rcParams['figure.figsize']=(25,25)
#a4_dims = (100,100)
#fig, ax = plt.subplots(figsize=a4_dims)


# In[ ]:


#Find out any correlation with suicide bombing attacks with influencing events given in the dataset
cleand_influencing=data[pd.notnull(data["Influencing Event/Event"])]


# In[ ]:


sns.countplot("Influencing Event/Event",data=cleand_influencing)


# In[ ]:


data["Explosive Weight (max)"]=data["Explosive Weight (max)"].str.replace("kg"," kg")
data["Explosive Weight (max)"]=data["Explosive Weight (max)"].str.replace("KG"," kg")


# In[ ]:


#correlation between blast/explosive weight and number of people killed and injured
data["Explosive Weight (max)"]=data["Explosive Weight (max)"].str.split(" ").str[0]
data["Explosive Weight (max)"]=data["Explosive Weight (max)"].str.split("-").str[0]


# In[ ]:


data["Explosive Weight (max)"]=data["Explosive Weight (max)"].convert_objects(convert_numeric=True)
cleaned_explosive=data[np.isfinite(data["Explosive Weight (max)"])]
#cleaned_data = data[np.isfinite(data['Killed Max'])]
#plt.scatter(data["Explosive Weight (max)"],data["Killed_avg"])
cleaned_explosive


# In[ ]:


cleaned_explosive.describe()


# In[ ]:


plt.scatter(cleaned_explosive["Explosive Weight (max)"],cleaned_explosive["Killed_avg"])
plt.xlim(1,200)
plt.rcParams['figure.figsize']=(8,8)
plt.xlabel("Explosive Weight(KG)")
plt.ylabel("Average Killed")


# In[ ]:


plt.scatter(cleaned_explosive["Explosive Weight (max)"],cleaned_explosive["Injured_avg"])
plt.xlim(1,200)
plt.rcParams['figure.figsize']=(8,8)
plt.xlabel("Explosive Weight(KG)")
plt.ylabel("Average Injured")


# In[ ]:


#What is the impact of Holiday Type on number of blast victims
cleaned_holiday=data[pd.notnull(data["Holiday Type"])]


# In[ ]:


cleaned_holiday["Holiday Type"].isnull().sum()


# In[ ]:



sns.boxplot(cleaned_holiday["Holiday Type"],cleaned_holiday["Killed_avg"])
plt.rcParams['figure.figsize']=(20,8)


# In[ ]:



sns.boxplot(cleaned_holiday["Holiday Type"],cleaned_holiday["Injured_avg"])
plt.rcParams['figure.figsize']=(20,8)


# In[ ]:


#correlation between Islamic date and blast day/time/size/number of victims
cleaned_Islamic_Date=data[pd.notnull(data["Islamic_Month"])]


# In[ ]:


sns.barplot("Islamic_Month","No. of Suicide Blasts",data=cleaned_Islamic_Date)


# In[ ]:


sns.boxplot("Islamic_Month","Killed_avg",data=cleaned_Islamic_Date)


# In[ ]:


sns.boxplot("Islamic_Month","Injured_avg",data=cleaned_Islamic_Date)


# In[ ]:


sns.barplot("Day_name","No. of Suicide Blasts",data=cleaned_Islamic_Date)


# In[ ]:


sns.barplot("Day_name","Killed_avg",data=cleaned_Islamic_Date)


# In[ ]:


sns.barplot("Day_name","Injured_avg",data=cleaned_Islamic_Date)


# In[ ]:


#Top 10 locations of Blasts
cleaned_location=data[pd.notnull(data["Location"])]


# In[ ]:


data["Location"].value_


# In[ ]:


#Find the names of hospitals sorted by number of victims
cleaned_hospitals=data[pd.notnull(data["Hospital Names"])]


# In[ ]:


cleaned_hospitals["Hospital Names"].unique()


# In[ ]:


sns.countplot(cleaned_holiday["Holiday Type"])


# In[ ]:


data = data[np.isfinite(data['No. of Suicide Blasts'])]


# In[ ]:


categorical=data.select_dtypes(include=["object"])
non_categorical=data.select_dtypes(exclude=["object"])
categorical.drop("Injured Max",inplace=True,axis=1)
non_categorical["Injured Max"]=data.iloc[:,20]


# In[ ]:


corr=non_categorical.corr()
sns.heatmap(corr,vmax=0.8,annot=True)


# In[ ]:


sns.boxplot("No. of Suicide Blasts","Killed Min",data=data)


# In[ ]:


sns.boxplot("No. of Suicide Blasts","Killed Max",data=data)


# In[ ]:


sns.boxplot("No. of Suicide Blasts","Injured Min",data=data)


# In[ ]:


sns.boxplot("No. of Suicide Blasts","Killed Max",data=data)


# In[ ]:


data = data[np.isfinite(data['Killed Max'])]


# In[ ]:


sns.boxplot("No. of Suicide Blasts","Killed Max",data=data)


# In[ ]:


sns.barplot("Province","No. of Suicide Blasts",data=data)


# In[ ]:


sns.barplot("Blast Day Type","No. of Suicide Blasts",data=data)


# In[ ]:


sns.barplot("Holiday Type","No. of Suicide Blasts",data=data)


# In[ ]:


data = data[np.isfinite(data['Injured Min'])]


# In[ ]:


data


# In[ ]:


sns.boxplot("No. of Suicide Blasts","Injured Min",data=data)


# In[ ]:


plt.hist(data["No. of Suicide Blasts"])


# In[ ]:


word_list=list(data["City"])


# In[ ]:


from collections import Counter


# In[ ]:


cities = ["Peshawar","Quetta","Swat","Bannu","Karachi","Rawalpindi","Islamabad","Hangu","khyber Agency"]


# In[ ]:


frequencies = [71,32,25,22,21,19,17,17,14]
pos = np.arange(len(cities))
width = 0.7
ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(cities)
plt.bar(pos, frequencies, width, color='g')
plt.ylabel("Count")
plt.show()


# In[ ]:


P=data.City=="Peshawar"
Q=data.City=="Quetta"
S=data.City=="Swat"
B=data.City=="Bannu"
K=data.City=="Karachi"
R=data.City=="Rawalpindi"
I=data.City=="Islamabad"
H=data.City=="Hangu"
Ky=data.City=="Khyber Agency"


# In[ ]:


Peshawar=data[P]
Quetta=data[Q]
Swat=data[S]
Bannu=data[B]
Karachi=data[K]
Rawalpindi=data[R]
Islamabad=data[I]
Hangu=data[H]
Khyber_Agency=data[Ky]


# In[ ]:


small_df=Peshawar.append([Quetta,Swat,Bannu,Karachi,Rawalpindi,Islamabad,Hangu,Khyber_Agency])
small_df["Injured Max"]=small_df["Injured Max"].convert_objects(convert_numeric=True).dropna()


# In[ ]:


sns.barplot("City","No. of Suicide Blasts",data=small_df)


# In[ ]:


sns.barplot("City","Killed Max",data=small_df)


# In[ ]:


sns.barplot("City","Killed Min",data=small_df)


# In[ ]:


sns.barplot("City","Injured Max",data=small_df)


# In[ ]:


sns.barplot("City","Injured Min",data=small_df)


# In[ ]:


sns.barplot("Location Sensitivity","No. of Suicide Blasts",data=small_df)


# In[ ]:


from pandas.tools.plotting import scatter_matrix

scatter_matrix(data, alpha=0.6, figsize=(16,9), diagonal='kde')


# In[ ]:


sns.distplot(data["Killed Max"])
data["Killed Max"].mean()


# In[ ]:


data["Killed Max"]=data["Killed Max"].fillna(data["Killed Max"].mean())
data["Killed Max"].isnull().sum()


# In[ ]:


sns.distplot(data["Killed Min"])
data["Killed Min"].mean()


# In[ ]:


data["Killed Min"]=data["Killed Min"].fillna(data["Killed Min"].mean())
data["Killed Min"].isnull().sum()


# In[ ]:


sns.distplot(data["Injured Min"])
data["Injured Min"].mean()


# In[ ]:


data["Injured Min"]=data["Injured Min"].fillna(data["Injured Min"].mean())
data["Injured Min"].isnull().sum()


# In[ ]:


data["Injured Max"]=data["Injured Max"].convert_objects(convert_numeric=True)


# In[ ]:


sns.distplot(data["Injured Min"])
data["Injured Min"].mean()


# In[ ]:


data["Injured Max"]=data["Injured Max"].fillna(data["Injured Max"].mean())
data["Injured Max"].isnull().sum()


# In[ ]:




