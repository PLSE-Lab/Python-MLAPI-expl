#!/usr/bin/env python
# coding: utf-8

# i'm Vamsi studying B-Tech 3rd year and this is my first DataScience project
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv("../input/googleplaystore.csv")
#data1=pd.read_csv("googleplaystore.csv")


# In[ ]:


data.dropna(inplace=True)
#data1.dropna(inplace=True)


# In[ ]:


data.info()


# In[ ]:


data.tail()


# In[ ]:


data["Genres"].value_counts().count()


# In[ ]:


data["Reviews"][10840]


# In[ ]:


'''data["Reviews"]=data["Reviews"].apply(lambda x :x.replace("M","") if "M" in str(x) else x)

data["Reviews"]=data["Reviews"].apply(lambda x :x.replace(",","") if "M" in str(x) else x)

data["Reviews"]=data["Reviews"].apply(lambda x :int(x))
'''


# In[ ]:





# In[ ]:


def filter(per):
    if "M" in str(per) and "," in str(per):
        per = str(per).replace("M","")
        per = per.replace(",","")
        return int(per)*1000000
    elif "M" in str(per):
        per = int(str(per).replace("M",""))
        return per*1000000
    elif "," in str(per):
        per = str(per).replace(",","")
        return int(per)
    
    else:  
        return int(per)


# In[ ]:


#data["Reviews"] = list(map(filter,data["Reviews"].values))


# In[ ]:


data["Reviews"] =data["Reviews"].apply(filter)


# In[ ]:


data.info()


# In[ ]:





# In[ ]:


data1.sample(5,random_state=55)


# In[ ]:


data1.sample(5,random_state=55)


# In[ ]:


data[data["Reviews"]==data.Reviews.max()]


#         invalid literal for int() with base 10: '19M'
#         invalid literal for int() with base 10: '8.7'
#             could not convert string to float: 'Varies with device
# ValueError: could not convert string to float: '201k'
# 

# In[ ]:


def filter1(per):
    per = str(per)
    if "M" in per:
        per = per.replace("M","")
        return float(per)
    elif per == "Varies with device":
        return np.NaN
    elif "k" in per:
        return float(per.replace("k",""))/1000
    else:
        return float(per)


# In[ ]:


#l1= list(map(filter1,data["Size"]))


# In[ ]:


data["Size"]=data["Size"].apply(filter1) 


# In[ ]:


data.info()


# In[ ]:


data["Size"].min()


# In[ ]:


data["Size"].max()


# In[ ]:


type(data.iloc[1567].Size)


# In[ ]:


data.sample(5)


# In[ ]:


def filter2(per):
    per = str(per)
    if "+" in per:
        per = per.replace("+","")
    if "," in per:
        per = per.replace(",","")
        
    return int(per)


# In[ ]:


#l2 = list(map(filter2,data["Installs"]))


# In[ ]:


#filter2('10,000+')


# In[ ]:


data["Installs"]=data["Installs"].apply(filter2)


# In[ ]:


data.info()


# In[ ]:


data[data["Installs"].isnull()]


# In[ ]:





# In[ ]:


data[data["App"]=="Home Pony 2"]


# In[ ]:


def filter3(per):
    per = str(per)
    if "$" in per:
        per='$4.99'.split("$")[1]
    return (float(per)*69.44)


# In[ ]:


float('$4.99'.split("$")[1])


# In[ ]:


data["Price"]=data["Price"].apply(filter3)


# In[ ]:


data.info()


# In[ ]:


data.to_csv("clean_data.csv",index=False)


# In[ ]:


cleandata=pd.read_csv("clean_data.csv")


# In[ ]:


cleandata.sample(5)


# In[ ]:


cleandata.info()==data.info()


# In[ ]:





# In[ ]:


sns.pairplot(cleandata,hue="Type")


# In[ ]:


temp=pd.DataFrame(cleandata["Content Rating"].value_counts()).reset_index()


# In[ ]:


temp.columns=['user', 'Content Rating']


# In[ ]:


temp


# In[ ]:





# In[ ]:


plt.figure(figsize=(12,6))
sns.barplot(data=temp,x="user",y="Content Rating")


# In[ ]:


sns.set_context('talk',font_scale=1)
plt.figure(figsize=(17,13))
sns.countplot(data=cleandata,y="Category",hue="Type")


# In[ ]:


plt.figure(figsize=(16,12))
sns.boxplot(data=cleandata,x="Size",y="Category",palette='rainbow')


# In[ ]:


plt.figure(figsize=(17,13))
sns.countplot(data=cleandata[cleandata["Reviews"]>1000000],y="Category",hue="Type")
plt.title("most popular apps with 1000000+ reviews")
plt.xlabel("no of apps")


# In[ ]:


plt.figure(figsize=(12,6))
sns.distplot(cleandata["Rating"],bins=10,color="red")


# In[ ]:


sns.countplot(x=cleandata["Type"])


# In[ ]:


sns.heatmap(cleandata.corr(),cmap='coolwarm')


# In[ ]:


sns.scatterplot(x="Installs",y="Reviews",data=cleandata,palette="rainbow")


# In[ ]:


plt.figure(figsize=(16,6))
sns.scatterplot(data=cleandata[cleandata["Reviews"]>100000],x="Size",y="Rating",hue="Type")
plt.title("apps with reviews graterthan 100000")


# In[ ]:


sns.kdeplot(data=cleandata["Size"])
plt.title("size vs count")
plt.xlabel("")


# In[ ]:


listcat = cleandata["Category"].unique()
i=0


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


sns.scatterplot(data=cleandata[cleandata["Category"]==listcat[i]],x="Size",y="Reviews",hue="Type")
plt.title(str(listcat[i]))
i+=1


# In[ ]:


cleandata.columns


# In[ ]:


cleandata.groupby('Category')["Rating"].mean().index


# In[ ]:


plt.figure(figsize=(12,6))

sns.scatterplot(x = cleandata.groupby('Category')['Rating'].mean().index, y = cleandata.groupby('Category')['Rating'].mean().values)
plt.ylabel('Category', fontsize=13)
plt.xlabel('Rating', fontsize=13)
plt.xticks(rotation=90)
plt.title("avg rating table based on category")


# In[ ]:


most_popular_apps = cleandata[(cleandata["Reviews"]>10000000) ][ (cleandata["Rating"]>=4.5)]


# In[ ]:


sns.countplot(most_popular_apps["Category"])
plt.xticks(rotation=90)


# In[ ]:


sns.pairplot(most_popular_apps,hue="Type")


# In[ ]:


sns.heatmap(most_popular_apps.corr())


#             Thank you  
# i'm a student and still learning, 
# is their anything wrong in my code..?
#             please let me know.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




