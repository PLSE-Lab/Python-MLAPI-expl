#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing Essential Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


#Importing CSV File
df = pd.read_csv("../input/crop-production-in-india/crop_production.csv")
df[:5]


# # Data Exploration

# In[ ]:


df.isnull().sum()


# In[ ]:


# Droping Nan Values
data = df.dropna()
print(data.shape)
test = df[~df["Production"].notna()].drop("Production",axis=1)
print(test.shape)


# In[ ]:


for i in data.columns:
    print("column name :",i)
    print("No. of column :",len(data[i].unique()))
    print(data[i].unique())


# In[ ]:


sum_maxp = data["Production"].sum()
data["percent_of_production"] = data["Production"].map(lambda x:(x/sum_maxp)*100)


# In[ ]:


data[:5]


# # Data Visulization

# In[ ]:


sns.lineplot(data["Crop_Year"],data["Production"])


# In[ ]:


plt.figure(figsize=(25,10))
sns.barplot(data["State_Name"],data["Production"])
plt.xticks(rotation=90)


# In[ ]:


sns.jointplot(data["Area"],data["Production"],kind='reg')


# In[ ]:


sns.barplot(data["Season"],data["Production"])


# In[ ]:


data.groupby("Season",axis=0).agg({"Production":np.sum})


# In[ ]:


data["Crop"].value_counts()[:5]


# In[ ]:


top_crop_pro = data.groupby("Crop")["Production"].sum().reset_index().sort_values(by='Production',ascending=False)
top_crop_pro[:5]


# ## Each type of crops required various area & various season. so, I'm going to pic top crop from this data

# ### 1.Rice

# In[ ]:


rice_df = data[data["Crop"]=="Rice"]
print(rice_df.shape)
rice_df[:3]


# In[ ]:


sns.barplot("Season","Production",data=rice_df)


# In[ ]:


plt.figure(figsize=(13,10))
sns.barplot("State_Name","Production",data=rice_df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


top_rice_pro_dis = rice_df.groupby("District_Name")["Production"].sum().reset_index().sort_values(
    by='Production',ascending=False)
top_rice_pro_dis[:5]
sum_max = top_rice_pro_dis["Production"].sum()
top_rice_pro_dis["precent_of_pro"] = top_rice_pro_dis["Production"].map(lambda x:(x/sum_max)*100)
top_rice_pro_dis[:5]


# In[ ]:


plt.figure(figsize=(18,12))
sns.barplot("District_Name","Production",data=top_rice_pro_dis)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot("Crop_Year","Production",data=rice_df)
plt.xticks(rotation=45)
#plt.legend(rice_df['State_Name'].unique())
plt.show()


# In[ ]:


sns.jointplot("Area","Production",data=rice_df,kind="reg")


# # Insights:
# From Data Visualization:
# Rice production is mostly depends on Season, Area, State(place).

# # 2. Coconut

# In[ ]:


coc_df = data[data["Crop"]=="Coconut "]
print(coc_df.shape)
coc_df[:3]


# In[ ]:


sns.barplot("Season","Production",data=coc_df)


# In[ ]:


plt.figure(figsize=(13,10))
sns.barplot("State_Name","Production",data=coc_df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


top_coc_pro_dis = coc_df.groupby("District_Name")["Production"].sum().reset_index().sort_values(
    by='Production',ascending=False)
top_coc_pro_dis[:5]
sum_max = top_coc_pro_dis["Production"].sum()
top_coc_pro_dis["precent_of_pro"] = top_coc_pro_dis["Production"].map(lambda x:(x/sum_max)*100)
top_coc_pro_dis[:5]


# In[ ]:


plt.figure(figsize=(18,12))
sns.barplot("District_Name","Production",data=top_coc_pro_dis)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot("Crop_Year","Production",data=coc_df)
plt.xticks(rotation=45)
#plt.legend(rice_df['State_Name'].unique())
plt.show()


# In[ ]:


sns.jointplot("Area","Production",data=coc_df,kind="reg")


# # Insight from Cocunut Production

# * cocunut production is directly proportional to area
# * its production is also gradually increasing over a time of period
# * production is highin kerala state
# * it does not depends on season

# # 3. Sugarcane

# In[ ]:


sug_df = data[data["Crop"]=="Sugarcane"]
print(sug_df.shape)
sug_df[:3]


# In[ ]:


sns.barplot("Season","Production",data=sug_df)


# In[ ]:


plt.figure(figsize=(13,8))
sns.barplot("State_Name","Production",data=sug_df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


top_sug_pro_dis = sug_df.groupby("District_Name")["Production"].sum().reset_index().sort_values(
    by='Production',ascending=False)
top_sug_pro_dis[:5]
sum_max = top_sug_pro_dis["Production"].sum()
top_sug_pro_dis["precent_of_pro"] = top_sug_pro_dis["Production"].map(lambda x:(x/sum_max)*100)
top_sug_pro_dis[:5]


# In[ ]:


plt.figure(figsize=(18,8))
sns.barplot("District_Name","Production",data=top_sug_pro_dis)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot("Crop_Year","Production",data=sug_df)
plt.xticks(rotation=45)
#plt.legend(rice_df['State_Name'].unique())
plt.show()


# In[ ]:


sns.jointplot("Area","Production",data=sug_df,kind="reg")


# # Insighits:
# * Sugarecane production is directly proportional to area
# * And the production is high in some state only.
