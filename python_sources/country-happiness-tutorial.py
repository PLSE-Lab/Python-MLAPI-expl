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
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data17 = pd.read_csv("../input/2017.csv") 
data15 = pd.read_csv("../input/2015.csv")
data16 = pd.read_csv("../input/2016.csv")


# In[ ]:


data17.columns 
data15.columns
data16.columns


# In[ ]:


data17.info()


# In[ ]:


data17.describe()


# In[ ]:


data17.head(10) 


# In[ ]:


data17.tail(10) 


# In[ ]:



data17.corr() 


# In[ ]:


f,ax = plt.subplots(figsize = (15,15)) 
sns.heatmap(data17.corr(),annot = True, linewidths = 5, fmt = ".2f", ax = ax)
plt.title("Heat Map")
plt.show()


# In[ ]:


dataa = data17.drop("Happiness.Rank", axis = 1) 
dataa.plot(figsize = (14,9), kind ="line", linewidth = 1, label = "Plot Line")
plt.legend()
plt.show()


# In[ ]:


data17["Happiness.Score"].plot(figsize = (14,9), kind="line", color = "r", alpha = 0.5, linewidth = 1,label = "Happiness score")
data17["Family"].plot (color ="green", alpha = 0.6, linewidth = 1, label = "Family")
plt.xlabel("Rank")
plt.gca().invert_xaxis() # For reverse to x-axis
plt.ylabel("Happiness")
plt.legend(loc = "upper left")
plt.title("Line Plot")
plt.show()


# In[ ]:


data17.plot(figsize = (12,7),kind = "scatter", x = "Happiness.Score", y = "Family", alpha = 0.5, linewidth = 1, color = "g",label = "Happiness vs Family")
plt.legend(loc ="upper left")
plt.xlabel("Happiness Score")
plt.ylabel("Family")
plt.title("Scatter Plot")
plt.show()


# In[ ]:


data17["Happiness.Score"].plot(kind = "hist", bins = 15, figsize = (10,10), label = "Happiness score distribution", range = (0,7), normed = True)
plt.xlabel("Happiness Score",FontSize = 20)
plt.ylabel("Frequency",FontSize = 20)
plt.legend()
plt.show()


# In[ ]:



data17["Happiness.Score"].plot(kind = "line", color = "blue", linewidth = 2,figsize = (10,8), label = "2017")
data16["Happiness Score"].plot(kind = "line", color = "green", linewidth = 2,figsize = (10,8), label = "2016")
data15["Happiness Score"].plot(kind = "line", color = "red", linewidth = 2,figsize = (10,8),label = "2015")
plt.title("2017 Happiness vs 2016 Happiness", FontSize = 15)
plt.legend()
plt.show()


# In[ ]:


dicti = { "Turkey"  : "Unhappy", "Germany" : "Unhappy"}
print(dicti.keys())
print(dicti.values())


# In[ ]:


dicti["India"] = "Happy" #Add
dicti["Turkey"] = "Happy" # Update
del(dicti["India"]) #delete specific entry
print("Turkey" in dicti) #check include or not
print(dicti)
dicti.clear() #Clear all data from dicti
print(dicti)


# In[ ]:


x = data17["Happiness.Score"] > 7
data17[x]


#  Less Generosity but more Happiness Data

# In[ ]:


data17[(data17["Happiness.Score"]> 7) & (data17["Generosity"] < 0.3)] ## datas which are Happiness score's more than 7 and Generosity's smaller than 0.3


# In[ ]:


data17[np.logical_and(data17["Happiness.Score"]> 7 , data17["Generosity"] < 0.3)]


# In[ ]:


liste = [1,2,3,4,5]
for index,value in enumerate(liste):
    print("Index: ", index)
    print("Value: ", value)
    
    
for key,items in dicti.items():
    print("Key: ", key)
    print("Value: ", items)
    
for index,value in data17[["Happiness.Score","Country"]].iterrows():
    print("Index: ", index)
    print("Score: ", value)
    
    


# In[ ]:


#Tuple
def tuple_ex():
    t = [1,2,3,5]
    return t

x = tuple_ex()
print(x)


# In[ ]:


x = 10
def xev():
    x = 3
    return x
print(x)
print(xev())


# In[ ]:


import builtins
dir(builtins)


# In[ ]:


def squr():
    def summo():
        x = 10
        y = 2
        z = 3
        return x+y+z
    return summo()**2

print(squr())


# In[ ]:


def func(x,y =1,z =3):
    return x+y+z
func(2,3)


# In[ ]:


def func(*args):
    x = sum(args)
    return x
func(2,4)

def func2(**kwargs):
    for key,values in kwargs.items():
        print(key, ": ", values)
        

func2(count = "2", b= "12" ,c = 3)        


# In[ ]:


#lambda
divide = lambda x: x/2
divide(20)


# In[ ]:


lis = [1,23,4,5]

y = map(lambda x:x**2,lis)
print(list(lis))


# In[ ]:


name = "sansal"
itera = iter(name)
next(itera)
next(itera)
next(itera)
print(next(itera))
print(*itera)


# In[ ]:


list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)


# In[ ]:


un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip) # unzip returns tuble
print(un_list1)
print(un_list2)
print(type(un_list2))


# In[ ]:


liste1 = [1,2,3,4]
liste2 = [i**2 for i in liste1]
liste3 = [i+2 if i < 2 else i+3 if i == 2 else i -1 for i in liste1]
liste3


# Separation of happy and unhappy countries

# In[ ]:


meanof = np.mean(data17["Happiness.Score"]) ## Creating new column on to my dataframe for separate countries to happy or unhappy
data17["Happy_Or_Not"] = ["Happy" if each > meanof  else "Unhappy" for each in data17["Happiness.Score"]]
meanof6 = np.mean(data16["Happiness Score"])
data16["Happy_Or_Not"] = ["Happy" if each > meanof6  else "Unhappy" for each in data16["Happiness Score"]]
meanof5 = np.mean(data15["Happiness Score"])
data15["Happy_Or_Not"] = ["Happy" if each > meanof5  else "Unhappy" for each in data15["Happiness Score"]]
data17.loc[:,["Country", "Happiness.Score", "Happy_Or_Not"]]


# In[ ]:


data17["Country"].value_counts(dropna = False)


# In[ ]:





# In[ ]:





# In[ ]:


data17.boxplot(column = "Happiness.Score",by = "Happy_Or_Not")
plt.show()


# In[ ]:


meltme = data17.head(10)


# In[ ]:


melted = pd.melt(frame= meltme, id_vars = "Country", value_vars = ["Happy_Or_Not","Family"])
melted


# In[ ]:


melted.pivot(index = "Country", columns = "variable", values = "value")


# In[ ]:


cdata17 = data17["Happiness.Score"].head()
cdata16 = data17["Happy_Or_Not"].head()
datavert = pd.concat([cdata16,cdata17], axis = 1)
datavert


# In[ ]:


cdata17 = data17.head()
cdata16 = data17.tail()
datavert = pd.concat([cdata16,cdata17], axis = 0, ignore_index = True)
datavert


# In[ ]:


data17.dtypes


# In[ ]:


data17["Happiness.Score"] = data17["Happiness.Score"].astype("int")
data17.dtypes
#data17["Happiness.Score"]


# In[ ]:


data15.info()


# In[ ]:


dataaa = data17
data17["Happy_Or_Not"].dropna(inplace = True)


# In[ ]:


assert data17["Happy_Or_Not"].notnull().all()


# In[ ]:


data17["Happy_Or_Not"].fillna("empty", inplace = True)


# In[ ]:


assert data17["Happy_Or_Not"].notnull().all()


# In[ ]:


assert data17.columns[0] == "Country"
assert data17["Family"].dtype == "float64"


# In[ ]:


country = ["turkey", "spain"]
city = ["Istanbul", "Madrid"]
listlabel = ["country", "city"]
listcol = [country, city]
zipped = list(zip(listlabel,listcol))

df = pd.DataFrame(dict(zipped))
df


# In[ ]:


df["population"] = [10,12]
df


# In[ ]:


df["income"] = 1
df


# In[ ]:


data1 = data17.loc[:,:]
data1.plot()


# In[ ]:


data1.plot(subplots = True)
plt.show()


# In[ ]:


data17["Happiness.Score"].plot(kind = "hist", bins = 15, figsize = (10,10), label = "Happiness score distribution", range = (0,7), normed = True, cumulative = True)
plt.xlabel("Happiness Score",FontSize = 20)
plt.ylabel("Frequency",FontSize = 20)
plt.legend()
plt.show()


# In[ ]:


datestr = ["1997-09-29"]
print(type(datestr[0]))
datetime_object = pd.to_datetime(datestr)
print(type(datetime_object))


# In[ ]:


data17.head()
data2 = data17.head()
timese = ["1997-09-29","1997-09-23","1997-10-29","1987-09-29","1997-02-19"]
datee = pd.to_datetime(timese)
data2["date"] = datee

data2 = data2.set_index("date")
data2


# In[ ]:


data2.loc["1997-09-29"]
data2.loc["1987-09-29":"1997-09-29"]
data2


# In[ ]:


data2.resample("A").mean().interpolate("linear")


# In[ ]:


data17["#"] = [i+1 for i in data17.index]
data17
data17 = data17.set_index(["#"])


# In[ ]:



#data17["Happiness.Score"][1]
data17


# In[ ]:


data17.loc[1,["Happiness.Score"]]


# In[ ]:


data17.Family[1]


# In[ ]:


data17[["Happiness.Rank", "Happiness.Score"]]


# In[ ]:


data17.loc[1:10,"Happiness.Rank":"Family"]


# In[ ]:


data17.loc[10:1:-1,"Happiness.Rank":"Family"]


# In[ ]:


data17.loc[1:10,"Happiness.Rank":]


# In[ ]:


boolean = data17["Happiness.Score"] > 7
boolean2 = data17["Freedom"] < 0.6
data17[boolean & boolean2]


# In[ ]:


data17.Family[data17["Happiness.Score"] > 7]


# In[ ]:


def sqr(n):
    return n**2
data5 = data17.Family.apply(sqr)


# In[ ]:


data5


# In[ ]:


data17["Happiness.Rank"].apply(lambda n : n*10)


# In[ ]:


data17["Happinessproportion"] = data17["Happiness.Rank"] + data17["Happiness.Score"]
data17


# In[ ]:


data17

