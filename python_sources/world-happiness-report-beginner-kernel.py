#!/usr/bin/env python
# coding: utf-8

# # My First Data Analysis Report
# **This is my first Data Analysis Report**
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt#plot drawing
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Before starting analyzing data, data sets should be imported as dataframe from .csv file
# 
# Then, reading column information about those dataframes with .info()
# 
# Then looking which columns do have data frames with .column

# In[ ]:


data = pd.read_csv("../input/2017.csv")
print("Data Info")
data.info()
#Change . sepertor with _ 
data.columns = data.columns.str.replace(".", "_")



# In[ ]:


data.columns = data.columns.str.replace("Economy__GDP_per_Capita_", "Economy")
data.columns = data.columns.str.replace("Health__Life_Expectancy_","Health")
data.columns = data.columns.str.replace("Trust__Government_Corruption_","Government_trust")
data.columns


# Lets continue with CORRELATION MAP to understand relationships between features with corr() function.
# 
# If corr==1 btw 2 feature,those features has linear relation(positive)
# If corr ==0, features has no realtionship

# In[ ]:


print("2017 data set")
data.corr() 
#correlation map view
f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(data.corr(), annot = True, linewidths=.5, fmt = ".2f", ax=ax)
plt.show()
data.describe()


# <a id="2">MATPLOTLIB</a> <br>
# **Line Plot** 
# 
# This plot is useful when we need to analyze increasing/changing value on x-axis and impacts on other features y-axis
# In this part, I want to analyze more correlated values effects on happiness score.

# In[ ]:


plt.plot(data.Happiness_Score, data.Economy, color = "red", label = "Economy",alpha = 0.8)
plt.plot(data.Happiness_Score, data.Family, color = "yellow", label = "Family",alpha = 0.8)
plt.plot(data.Happiness_Score, data.Freedom, color = "blue", label = "Freedom",alpha = 0.8)
plt.plot(data.Happiness_Score, data.Health, color = "green", label = "Health",alpha = 0.8)

plt.legend()    
plt.xlabel('Happiness Score')           
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# According to those values, for happiness the most important factor is Economy, but also having a family has a big effect on it.

# **Scatter Plot **
# 
# When we need to analyze correlated values and their behaviour, scatter plot helps a lot. Therefore, I would like to analyze Economy values with other important values. Hence, Happiness score is more related with Economy, then lets analyze how the others are related with Economy...
# 
# 

# In[ ]:


#data.plot(kind="scatter", x = "Economy", y ="Family", color = "yellow", label = "Economy-Family")
#data.plot(kind="scatter", x = "Economy", y ="Health", color = "green", label = "Economy-Health")
#data.plot(kind="scatter", x = "Economy", y ="Freedom", color = "blue", label = "Economy- Freedom")
plt.scatter(data.Economy, data.Family, color = "yellow", label = "Family")
plt.scatter(data.Economy, data.Health, color = "green", label = "Health")
plt.scatter(data.Economy, data.Freedom, color = "blue", label = "Freedom")
plt.xlabel('Economy') 
plt.legend()
plt.title('Economy Relations')            
plt.show()


# As it seems from the figure above, Health and Freedom has more relations then having family. For example, Counties have the Economy scala between 0.75 and 1.0(averagly normal economical countires) has more common points and it is hard distinguish between them. 

# In[ ]:


plt.scatter(data.Happiness_Score, data.Health, color = "green", label = "Health")
plt.scatter(data.Happiness_Score, data.Freedom, color = "blue", label = "Freedom")
plt.xlabel('Happiness Score') 
plt.legend()
plt.title('Happiness Score Relations')            
plt.show()


# Additionally, when i analyzed health and freedom with respect to happiness score, it is hard to distinguish them, therefore to analyze those is a little bit hard.

# **Histogram Plot**
# 
# Histogram plot is useful to analyze the data with frequency differences. Also, in the beginning this graphs is very useful to understand mean values and distiributions.
# For histogram plot, I want to analyze "Happy and Unhappy countries behaviours".
# Note: Mean of the Happiness score is approximately 5.3 therefore I'll use that data as a threashold on dividing happy and unhappy dataset.
# 

# In[ ]:


happy = data[data["Happiness_Score"] > 5.3] 
unhappy = data[data["Happiness_Score"] < 5.3] 

happy.Economy.plot(kind = 'hist',bins = 50,figsize = (8,8), color = "blue", alpha = 0.7, label = "Happy")
unhappy.Economy.plot(kind = 'hist',bins = 50,figsize = (8,8), color = "red", alpha = 0.7, label = "Unhappy")

plt.xlabel('Economy')             
plt.title('Happiness and Economy relations') 
plt.legend()
plt.show()


# As it seems from the figure, in "Happy" counties Economical situation is better than "Unhappy" countries.
# Then I want focussing on "How Freedom has effect on Happy and Unhappy countries?"

# In[ ]:


happy = data[data["Happiness_Score"] > 5.3] 
unhappy = data[data["Happiness_Score"] < 5.3]

happy.Freedom.plot(kind = 'hist',bins = 50,figsize = (8,8), color = "blue", alpha = 0.7, label = "Happy")
unhappy.Freedom.plot(kind = 'hist',bins = 50,figsize = (8,8), color = "red", alpha = 0.7, label = "Unhappy")

plt.xlabel('Freedom')             
plt.title('Happiness and Freedom relations') 
plt.legend()
plt.show()


# For Freedom perspective, it is hard to distinguish values because both graphs mean values are very similar with each other. 
# 
# It means that Economical situation has more power than Freedom to distinguish countries with each other.

# <a id="3">PANDAS</a> <br>
# This part, I will seperate my datas to improve my basic skills. Therefore now I changed my dataset and now I'll work on 2016's dataset

# In[ ]:


data = pd.read_csv('../input/2016.csv')
data.columns
#I need to make some changes about column names to work more better


# In[ ]:


#Change . sepertor with _ 
data.columns = data.columns.str.replace(" ", "")
data.columns = data.columns.str.replace("(", "")
data.columns = data.columns.str.replace(")", "")

data.columns = data.columns.str.replace("EconomyGDPperCapita", "Economy")
data.columns = data.columns.str.replace("HealthLifeExpectancy","Health")
data.columns = data.columns.str.replace("TrustGovernmentCorruption","Government_trust")
data.columns


# In[ ]:


series = data['HappinessScore']
print(type(series))
#Creating DataFrames to work more quicky
#df is a column of happiness score --> new data frame creation
df = data[['HappinessScore']]
print(type(df))


# Now using logical expression, lets create happy and unhappy dataframes again

# In[ ]:


happy = data[data["HappinessScore"] > 5.3]
unhappy = data[data["HappinessScore"] < 5.3]
print("First 5 happy Countries scores")
happy.head()
happy.columns


# Lets analyze the first 5 happy and unhappy countries in Western Europe situation with creating new dataframes by using while and for loops. In western Europe there are 19 data.

# In[ ]:


print("TOP 10 HAPPY COUNTRIES")
west = happy[happy["Region"] == "Western Europe"]
cnt = 0
for index,row in west.iterrows():
    if(cnt < 10):
        print(happy.Country[index])
        cnt += 1

print("FIRST 10 UNHAPPY COUNTRIES")

westu = unhappy[unhappy["Region"] == "Western Europe"]
cnt = 0
for index,row in westu.iterrows():
    if(cnt < 10):
        print(unhappy.Country[index])
        cnt += 1



# As it can seen from output  there are only 2 unhappy countries are there in Western Europe.

# 
# <a id="7">FUNCTIONS</a> <br>
# Default and Flexiable functions examples:<br>
# Lets given a threashold find other happy countries...

# In[ ]:


data.info()


# In[ ]:


#Find the happiest countries above from threashold value
df_a = pd.DataFrame(columns=["Country","HappinessScore", "Economy"])
df_b = pd.DataFrame(columns=["Country","HappinessScore", "Economy"])


# In[ ]:


#Find the happiest countries above from threashold value
def above_threashold(thr,avg = 5.3):
    if(thr >= avg):
        for index, value in data.iterrows():
            if((data.HappinessScore[index] > thr) & (data.HappinessScore[index] > avg)):
                print("(",data.Country[index],"-", data.HappinessScore[index], ") above ", thr)
            elif((data.HappinessScore[index] < thr) & (data.HappinessScore[index] > avg)):
                print("(",data.Country[index],"-", data.HappinessScore[index], ") below ", thr)
            else:
                continue
    else:
        print("your threashold is lower than average happiness score")        

print(above_threashold(7))




# In[ ]:


#df_a.HappinessScore.plot(kind = 'line', figsize = (8,8), color = "blue", alpha = 0.7, label = "Above Countries")
#df_b.HappinessScore.plot(kind = 'line',figsize = (8,8), color = "red", alpha = 0.7, label = "Below Countries")
#plt.plot(df_a.Country, df_a.HappinessScore, color = "red", label = "Above Counries",alpha = 0.7)
#plt.plot(df_b.Country, df_b.HappinessScore, color = "blue", label = "Below Counries",alpha = 0.7)


# <a id="7">ITERATORS</a> <br>
# Inside of the list,String,dictionaries iterations instead of using for loop **iter() , next(), *iter_ed value** functions can be used
# <br>
# zip() function is also useful for combining list 
# 

# In[ ]:


west_countries = list(west.Country)
happy_score = list(west.HappinessScore)
c_list = []
s_list = []
for each in range(len(west_countries)):
    c_name = west_countries[each]
    h_score = int(happy_score[each])
    i_name = iter(c_name)
    c_list.append(next(i_name))
    s_list.append(h_score)
print("clist: ", c_list, "slist: ", s_list)
#ZIP 2 list to combine
z = zip(c_list, s_list)
print(z)
z_list = list(z)
print(z_list)


# <a id="8">LIST COMPREHENSION</a> <br>
# We use list comprehension for data analysis often. 
# <br> list comprehension: collapse for loops for building lists into a single line
# 

# In[ ]:


data.Economy.describe()


# In[ ]:


# Conditionals on iterable
threshold = sum(data.Economy) / len(data.Economy)
data["Economical_level"] = ["high" if i > threshold else "low" for i in data.Economy]


# In[ ]:


hec = data[data["Economical_level"] == "high"]
lec = data[data["Economical_level"] == "low"]

plt.plot(hec.Generosity, hec.HappinessScore, color = "red", label = "Economically High",alpha = 0.8 )
plt.plot(lec.Generosity, lec.HappinessScore, color = "blue", label = "Economically Low",alpha = 0.8 )

plt.xlabel('Generosity')             
plt.title('How Generosity is effect on Happiness in economically high and low countries') 
plt.legend()
plt.show()



# In[ ]:




