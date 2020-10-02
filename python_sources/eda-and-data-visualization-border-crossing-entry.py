#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap

import datetime

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv("../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv")


# In[ ]:


df.head()


# In[ ]:


df.drop(["Port Code"],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.columns=["port_name","state","border","date","measure","value","location"]


# In[ ]:


df.tail()


# In[ ]:


df.location=df.location.apply(lambda x: str(x).replace("POINT (","") if 'POINT (' in str(x) else str(x))
df.location=df.location.apply(lambda x: str(x).replace(")","") if ")" in str(x) else str(x))
#df.location=df.location.apply(lambda x: str(x).replace(" ",",") if " " in str(x) else str(x))
#df["location"]=df["location"].apply(lambda x: float(x))


# In[ ]:


df_loc = pd.DataFrame(df.location.str.split(' ',1).tolist(),columns = ['lat','lon'])
df["lat"]=df_loc.lat
df["lon"]=df_loc.lon


# In[ ]:


df["lat"]=df["lat"].apply(lambda x: float(x))
df["lon"]=df["lon"].apply(lambda x: float(x))


# In[ ]:


df.drop(["location"],axis=1,inplace=True)


# In[ ]:


df = df.sort_values(by=["value"], ascending=False)
df['rank']=tuple(zip(df.value))
df['rank']=df.groupby('value',sort=False)['rank'].apply(lambda x : pd.Series(pd.factorize(x)[0])).values
df.head()


# In[ ]:


df.drop(["rank"],axis=1,inplace=True)


# In[ ]:


df.reset_index(inplace=True,drop=True)


# In[ ]:


df.date = pd.to_datetime(df.date)


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dtypes


# In[ ]:


df.nunique()


# In[ ]:


plt.figure(figsize=(25,18))
sns.set(style='whitegrid')
ax=sns.barplot(x=df['port_name'].value_counts().index,y=df['port_name'].value_counts().values)
plt.legend(loc=8)
plt.xlabel('Port Name')
plt.xticks(rotation=90)
plt.ylabel('Frequency')
plt.title('Show of Port Name Bar Plot')
plt.show()


# In[ ]:


df.state.value_counts()


# In[ ]:


plt.figure(figsize=(18,5))
sns.set(style='whitegrid')
ax=sns.barplot(x=df['state'].value_counts().index,y=df['state'].value_counts().values,palette="Blues_d",hue=['North Dakota',
                                                                                                             'Washington',
                                                                                                              "Montana",
                                                                                                            "Maine",
                                                                                                            "Texas",
                                                                                                            "Minnesota",
                                                                                                            "New York",
                                                                                                            "Arizona",
                                                                                                            "California",
                                                                                                            "Vermont",
                                                                                                            "Alaska",
                                                                                                            "Michigan",
                                                                                                            "Idaho",
                                                                                                            "New Mexico",
                                                                                                            "Ohio"])

plt.xlabel('state')
plt.xticks(rotation=75)
plt.ylabel('Frequency')
plt.title('Show of state Bar Plot')
plt.legend(loc=10)
plt.show()


# In[ ]:


df.border.value_counts()


# In[ ]:


sns.set(style='whitegrid')
ax=sns.barplot(x=df['border'].value_counts().index,y=df['border'].value_counts().values,palette="Blues_d",hue=['US-Canada Border','US-Mexico Border'])
plt.legend(loc=8)
plt.xlabel('Border')
plt.ylabel('Frequency')
plt.title('Show of Border Bar Plot')
plt.show()


# In[ ]:


df.measure.value_counts()


# In[ ]:


plt.figure(figsize=(18,5))
sns.barplot(x=df['measure'].value_counts().index,y=df['measure'].value_counts().values)
plt.title('measure other rate')
plt.ylabel('Rates')
plt.xticks(rotation=75)
plt.legend(loc=0)
plt.show()


# In[ ]:


plt.figure(figsize=(18,7))
sns.barplot(x = "state", y = "value", hue = "measure", data = df)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


#plt.figure(figsize=(18,18))
#sns.barplot(x = "date", y = "value", hue = "measure", data = df)
#plt.xticks(rotation=45)
#plt.show()


# In[ ]:


plt.figure(figsize=(18,7))
sns.barplot(x = "state", y = "value", hue = "border", data = df)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


plt.figure(figsize=(18,5))
ax = sns.violinplot(x="state", y="value",
                    data=df[df.value < 30000],
                    scale="width", palette="Set3")


# In[ ]:


plt.figure(figsize=(6,12))
sns.catplot(y="border", x="value",
                 hue="measure",
                 data=df, kind="bar")
plt.show()


# In[ ]:


plt.figure(figsize=(12,12))
sns.catplot(y="border", x="value",
                 hue="state",
                 data=df, kind="bar")
plt.show()


# In[ ]:


labels=df['border'].value_counts().index
colors=['blue','red']
explode=[0,0.1]
values=df['border'].value_counts().values

#visualization
plt.figure(figsize=(7,7))
plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')
plt.title('Border According Analysis',color='black',fontsize=10)
plt.show()


# In[ ]:


labels=df['measure'].value_counts().index
colors=['blue','yellow']
explode=[0.3,0.3,0,0,0,0,0,0,0,0,0,0]
values=df['measure'].value_counts().values

#visualization
plt.figure(figsize=(7,7))
plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')
plt.title('Measure According Analysis',color='black',fontsize=10)
plt.show()


# In[ ]:


labels=df['state'].value_counts().index
colors=['blue','yellow']
explode=[0.3,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
values=df['state'].value_counts().values

#visualization
plt.figure(figsize=(7,7))
plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')
plt.title('State According Analysis',color='black',fontsize=10)
plt.show()


# In[ ]:


sns.kdeplot(df['value'])
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Value Kde Plot System Analysis')
plt.show()


# In[ ]:


sns.boxenplot(x="border", y="value",
              color="b",
              scale="linear", data=df)
plt.show()


# In[ ]:


sns.lineplot(x='date',y='value',hue="border",data=df)
plt.show()


# In[ ]:


sns.lineplot(x='date',y='value',hue="state",data=df)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(18, 6.5))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="date", y="value",
                hue="state", size="state",data=df)
plt.show()


# In[ ]:


from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in df.port_name)
# Create and generate a word cloud image:
wordcloud = WordCloud(max_words=200, background_color="white").generate(text)
plt.figure(figsize=(15,10))
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

