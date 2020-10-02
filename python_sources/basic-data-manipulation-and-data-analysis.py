#!/usr/bin/env python
# coding: utf-8

# ![](http://static-news.moneycontrol.com/static-mcnews/2017/04/shutterstock_489439225-770x433.jpg)

# **BASICS**

# *Done by* 
# >*PRADIP*   
# >*GIRIRAM*   
# >*KALYAN*   
# >*NITHISH*

# **DATA MANIPULATION**
# 
# *1. IMPORT NUMPY AND PANDAS*
# > *1. 1 ASSIGNING DATASETS TO FD*
# > > *1. 1. 1 DISPLAY FIRST 5 OF THE ENTIRE DATASET*
# 
# *2. TO VIEW EACH COLUMN NAME*
# > *2. 1 DELETE A SPECIFIC COLUMN*
# 
# *3. TO FIND TOTAL NUMBER OF ROWS*
# 
# *4. FETCH UNIQUE VALUES*
# > *4. 1*
# 
# > *4. 2 DATASTRUCTURE IN PYTHON*
# 
# *5. TO DISPLAY DATASET WITH SPECIFIC CONSTRAINTS*
# 
# *6. CONVERT STRING TO FLOAT*
# 
# *7. TO CACULATE MAX VALUE AND PRINT THE ROW WHICH HAS THE MAXIMUM VALUE*
# > *7.1 MINIMUM VALUE*
# 
# **DATA VISUALIZATION**
# 
# *SET 1 FOR VALUES CITYNAME.INDEX AND CITYNAME.VALUES*
# > *1. BAR GRAPH*
# > > *2. SCATTER PLOT*
# > > > *3. PIE CHART*
# 
# *SET 2 FOR VALUES INVESTNAME.INDEX AND INVESTNAME.VALUES*
# > *1. BAR GRAPH*
# > > *2. SCATTER PLOT*
# > > > *3. PIE CHART*

# ![](http://assets.datacamp.com/production/tracks/14/badges/original/Data_Manipulation_10x.png?1506959694)

# 1. IMPORT NUMPY AND PANDAS

# In[ ]:


import numpy as np

#adds support for large, multi-dimensional arrays and matrices
#along with a large collection of high-level mathematical functions to operate.


import pandas as pd 

#high-performance,easy-to-use data structures


# **1. 1 ASSIGNING DATASETS TO FD**
#  
# *1. 1. 1 DISPLAY FIRST 5 OF THE ENTIRE DATASET*

# In[ ]:


fd = pd.read_csv("../input/startup_funding.csv")  # data processing, CSV file I/O (e.g. pd.read_csv)

fd.head(5)  #shows the first 5 DATASET


# **2. TO VIEW EACH COLUMN NAME**

# In[ ]:


print(*fd.columns)


# **2. 1 DELETE A SPECIFIC COLUMN**

# In[ ]:


del fd['Remarks'] #del - command
fd.head(5)


# **3. TO FIND TOTAL NUMBER OF ROWS**

# In[ ]:


leng=len(fd.index)+1  #find length

print(leng,"ROWS are present in this DATASET")


# **4. FETCH UNIQUE VALUES**

# In[ ]:


name = fd['InvestorsName'].unique()   #stores unique values of INVESTORS NAME
amount = fd['AmountInUSD'].unique()   #stores unique values of INVESTORS NAME


print(*amount[:5])
type(amount)


# **4. 1**

# In[ ]:


#to find unique names of listed cities

uni_city = [] #array declaration
uni_city = fd['CityLocation'].unique() #to find unique

print(uni_city[:5])
type(uni_city)


# ![](http://www.digitalvidya.com/wp-content/uploads/2018/08/data-structures-and-algorithms-in-python-1170x630.jpg)

# **DATASTRUCTURES**

# *1. LIST*

# In[ ]:


uni_city_list=np.array(uni_city).tolist()  #converts np.array to list

uni_city_list.append('R.m.d')   #appends RMD

print(uni_city_list[:5])
print(len(uni_city_list))   #prints len of list

type(uni_city_list)


# *2. DICTIONARY*

# In[ ]:


startup={}
type(startup)

for i in range(0,len(uni_city)):
    startup[uni_city[i]]=amount[i];
for i,j in startup.items():
    print(i,":",j)


# **5. TO DISPLAY DATASET WITH SPECIFIC CONSTRAINTS**

# In[ ]:


fd[fd.CityLocation == 'Chennai'].head(5)


# **6. CONVERT STRING TO FLOAT**

# In[ ]:


#to convert string to float

fd["AmountInUSD"] = fd["AmountInUSD"].apply(lambda x: float(str(x).replace(",",""))) #expression conversion is done using lambda
fd["AmountInUSD"] = pd.to_numeric(fd["AmountInUSD"]) #now those amount are converted to numeric format

fd.head(5)



#fd["Date"] = fd["Date"].apply(lambda x: float(str(x).replace("/",""))) #expression conversion is done using lambda
#fd["Date"] = pd.to_numeric(fd["Date"]) #now those amount are converted to numeric format

#fd.head(5)


# **6.1 CONVERSION OF NaN TO 0**

# In[ ]:


#to convert NaN (Not a NUMBER) to 0

fd.fillna(0).head(5)


# In[ ]:


#total

val = fd['AmountInUSD'].sum() #which sums up all the values in the row "AmountInUSD"
print("Total funding amount",val) #print total


# **7. TO CACULATE MAX VALUE AND PRINT THE ROW WHICH HAS THE MAXIMUM VALUE**

# In[ ]:


#to calculate max value and to print the max row

max_invest=max(fd['AmountInUSD'])  #find max
print("maximum amount invested",max_invest)


max_index=fd['AmountInUSD'].idxmax()   #to assign max amount's index value
fd.iloc[[max_index]]   #print the row



# **7.1 MINIMUM VALUE**

# In[ ]:


min_invest=min(fd['AmountInUSD'])  
print("minimum amount invested",min_invest)


min_index=fd['AmountInUSD'].idxmin()   #to assign max amount's index value
fd.iloc[[min_index]]   #print the row


# ![](https://www.tertiarycourses.com.sg/media/catalog/product/cache/1/image/512x/040ec09b1e35df139433887a97daa66f/d/a/data-visualization-python.jpg)

# **DATA VISUALIZATION**

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)

get_ipython().run_line_magic('matplotlib', 'inline')


# **SET 1 FOR VALUES CITYNAME.INDEX AND CITYNAME.VALUES**

# **1. BAR GRAPH**

# In[ ]:


cityname = fd['CityLocation'].value_counts().head(10)
plt.figure(figsize=(15,8))
sns.barplot(cityname.index, cityname.values)
plt.xticks(rotation='vertical')
plt.xlabel('Cities Name')
plt.ylabel('Number of STARTUPS in each cities')

plt.show()
#fd.loc[fd['CityLocation'] == "Mumbai", 'AmountInUSD']

#total = fd.loc[fd['CityLocation'] == "Bangalore", 'StartupName'].sum()
x=0
for i in cityname.index:
    print("Number of STARTUPS in",i, "are",cityname.values[x])
    x=x+1


# **2. SCATTER PLOT**

# In[ ]:


plt.scatter(cityname.index,cityname.values)
plt.xticks(rotation='vertical')
plt.xlabel('Cities Name')
plt.ylabel('Number of STARTUPS in each cities')

plt.show()


# **3. PIE CHART**

# In[ ]:


plt.pie(cityname.values, labels = cityname.index, autopct = "%.01f")
plt.show()


# **SET 2 FOR VALUES INVESTNAME.INDEX AND INVESTNAME.VALUES**

# **1. BAR GRAPH**

# In[ ]:


investname = fd['InvestorsName'].value_counts().head(5)
plt.figure(figsize=(15,8))
sns.barplot(investname.index, investname.values)
#plt.xticks(rotation='vertical')
plt.xlabel('Investors Name')
plt.ylabel('No. of Investments made')

plt.show()

x=0
for i in investname.index:
    print("Investments made by",i, "on",investname.values[x],"startups")
    x=x+1


# **2. SCATTER PLOT**

# In[ ]:


plt.scatter(investname.index,investname.values)
plt.xticks(rotation='vertical')
plt.xlabel('Investors Name')
plt.ylabel('No. of Investments made')

plt.show()


# **3. PIE CHART**

# In[ ]:


plt.pie(cityname.values, labels = cityname.index, autopct = "%.01f")
plt.show()

