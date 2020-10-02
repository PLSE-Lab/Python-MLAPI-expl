#!/usr/bin/env python
# coding: utf-8

# **1. Problem Statment**
# 
# In the field of data analysis data of specific domains plays an important role to make a decision. The data of a domain like the temperature of local territory and temperature of global for a specific period of time carry a few valuable insights. The project Explore Weather Trends provide such type of data and ask to find out that insight and depict meaningful figures, observations accordingly.
# 
# **2. Project Outline**
# 
# Project outline means overall steps were taken to accomplish the job by following scientific ways are given below:
# 
# * To process and visualize the data few technologies, tools and libraries have been used. Those are python 3.5, SQL, pandas, matplotlib and, spyder3.
# 
# * Collection of data from a given workspace using SQL queries. SQL queries retrieve the data of the nearest big city, i.e. Berlin, and the whole world from the year 1750 to 2013.
# * Read data from the CSV file using pandas.
# * Calculate the Moving Average of the nearest big city and global temperature using the formula :
#  
#  MA = (ai + ai+1 + ai+2 + ai+3 + ai+4 + ai+5 + ai+6) / m
# 
#  where a = temperature; i = 0; 1; 2.........n; m = number of examples taken
#  
# * Draw two line charts, one for the city, another one for the world, in a single figure to give more insight using the tool matplotlib.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

   
df_city = pd.read_csv("/kaggle/input/berlin_temp_avg.csv") 

df_global = pd.read_csv("/kaggle/input/world_temp_avg.csv") 


# In[ ]:


total_elements = len(df_city)
print(len(df_city))


# In[ ]:


ma_temp_city = []
ma_temp_year_city = []

ma_temp_global = []
ma_temp_year_global = []


# In[ ]:


for i in range(len(df_city)-6):
    average = (df_city['avg_temp'][i]+df_city['avg_temp'][i+1]+df_city['avg_temp'][i+2]+df_city['avg_temp'][i+3]+df_city['avg_temp'][i+4]+df_city['avg_temp'][i+5]+df_city['avg_temp'][i+6])
    ma_temp_city.append(round(average/7,2))
    ma_temp_year_city.append(df_city['year'][i+6])
   
#print(ma_temp_city)


# In[ ]:


for i in range(len(df_global)-6):
    average = (df_global['avg_temp'][i]+df_global['avg_temp'][i+1]+df_global['avg_temp'][i+2]+df_global['avg_temp'][i+3]+df_global['avg_temp'][i+4]+df_global['avg_temp'][i+5]+df_global['avg_temp'][i+6])
    ma_temp_global.append(round(average/7,2))
    ma_temp_year_global.append(df_global['year'][i+6])
    
#print(ma_temp_year,ma_temp_global)


# In[ ]:


plt.plot(ma_temp_year_city,ma_temp_city,color='green')
plt.plot(ma_temp_year_global,ma_temp_global,color='orange')
plt.xlabel('Years')
plt.ylabel('Temperature $^\circ$C')
plt.title('Berlin vs Global Avg Temperature 1750 to 2013 ')
plt.legend(['Berlin','Global'])
plt.show()


# In[ ]:


list1, list2 = (list(t) for t in zip(*sorted(zip(ma_temp_global, ma_temp_year_city))))

plt.plot(ma_temp_year_city,ma_temp_city,color='green')
plt.plot(ma_temp_year_global,ma_temp_global,color='orange')
plt.plot(list2[0],list1[0],'b*')
plt.xlabel('Years')
plt.ylabel('Temperature $^\circ$C')
plt.title('Berlin vs Global Avg Temperature 1750 to 2013 ')
plt.legend(['Berlin','Global','Global lowest temp '+str(list1[0])+' in '+str(list2[0])])
plt.show()


# In[ ]:


list1, list2 = (list(t) for t in zip(*sorted(zip(ma_temp_global, ma_temp_year_city),reverse=True)))

plt.plot(ma_temp_year_city,ma_temp_city,color='green')
plt.plot(ma_temp_year_global,ma_temp_global,color='orange')
plt.plot(list2[0],list1[0],'r*')
plt.xlabel('Years')
plt.ylabel('Temperature $^\circ$C')
plt.title('Berlin vs Global Avg Temperature 1750 to 2013 ')
plt.legend(['Berlin','Global','Global highest Temp '+str(list1[0])+' in '+str(list2[0])])
plt.show()


# In[ ]:




