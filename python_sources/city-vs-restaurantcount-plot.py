#!/usr/bin/env python
# coding: utf-8

# # CITY VS RESTAURANTS COUNT PLOT FROM ZOMATO DATASET FOR INDIA

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from glob import glob as gb


# In[ ]:


#list all the directories
dirs=os.listdir("../input/zomato_data/")
dirs


# In[ ]:


len(dirs)


# In[ ]:


#storing all the files from every directory
li=[]
for dir1 in dirs:
    files=os.listdir(r"../input/zomato_data/"+dir1)
    #reading each file from list of files from previous step and creating pandas data fame    
    for file in files:
        
        df_file=pd.read_csv("../input/zomato_data/"+dir1+"/"+file,quotechar='"',delimiter="|")
#appending the dataframe into a list
        li.append(df_file.values)
    
    


# In[ ]:


len(li)


# In[ ]:


#numpys vstack method to append all the datafames to stack the sequence of input vertically to make a single array
df_np=np.vstack(li)


# In[ ]:


#no of rows is represents the total no restaurants ,now of coloumns(12) is columns for the dataframe
df_np.shape


# In[ ]:


#creating final dataframe from the numpy array
df_final=pd.DataFrame(df_np)


# In[ ]:


#adding the header columns
df_final=pd.DataFrame(df_final.values, columns =["NAME","PRICE","CUSINE_CATEGORY","CITY","REGION","URL","PAGE NO","CUSINE TYPE","TIMING","RATING_TYPE","RATING","VOTES"])


# In[ ]:


#displaying the dataframe
df_final


# In[ ]:


#header column "PAGE NO" is not required ,i used it while scraping the data from zomato to do some sort of validation,lets remove the column
df_final.drop(columns=["PAGE NO"],axis=1,inplace=True)


# In[ ]:


#display the dataframe again
df_final


# In[ ]:


#lets count how many unique cities are there 

df_final["CITY"].unique()


# In[ ]:


len(df_final["CITY"].unique())


# In[ ]:


#lets check city wise restaurant counts and save it in ascending order

city_vs_count=df_final["CITY"].value_counts().sort_values(ascending=True)


# In[ ]:


city_vs_count


# In[ ]:


#lets check max count
count_max=max(city_vs_count)


# In[ ]:


#lets find for city count is max

for x,y in city_vs_count.items():
    if(y==count_max):
        print(x)
    


# In[ ]:


#lets find for city count is min

min_count=min(city_vs_count)

for x,y in city_vs_count.items():
    if(y==min_count):
        print(x)


# 

# In[ ]:


#lets plot citywise restaurant count in barh form

fig=plt.figure(figsize=(20,40))
city_vs_count.plot(kind="barh",fontsize=30)
plt.grid(b=True, which='both', color='Black',linestyle='-')
plt.ylabel("city names",fontsize=50,color="red",fontweight='bold')
plt.title("CITY VS RESTAURANT COUNT GRAPH",fontsize=50,color="BLUE",fontweight='bold')


# In[ ]:


#lets plot citywise restaurant count in barh form,and each bar should display the count of the corresponding restuants for that city

fig=plt.figure(figsize=(20,40))
city_vs_count.plot(kind="barh",fontsize=30)
plt.grid(b=True, which='both', color='Black',linestyle='-')
plt.ylabel("city names",fontsize=50,color="red",fontweight='bold')
plt.title("CITY VS RESTAURANT COUNT GRAPH",fontsize=50,color="BLUE",fontweight='bold')
for v in range(len(city_vs_count)):
    #plt.text(x axis location ,y axis location ,text value ,other parameters......)
    plt.text(v+city_vs_count[v],v,city_vs_count[v],fontsize=20,color="BLUE",fontweight='bold')


# In[ ]:


#THATS ALL GUYS SEE YOU IN THE NEXT KERNEL

