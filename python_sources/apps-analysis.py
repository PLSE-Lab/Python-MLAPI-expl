#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime, timedelta
#import plotly.offline as py
#py.init_notebook_mode(connected=True)
#import plotly.graph_objs as go

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph.

import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import warnings
warnings.filterwarnings('ignore')
from pylab import rcParams
# figure size in inches

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


## Read file

data = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")
user_r=pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv")
data


# In[ ]:


data['Installs'].unique()
data['Installs'].value_counts()
data['Installs'].replace()


# # "Installs" column

# data['Installs']=pd.to_numeric(data['Installs'])
# data.Installs.hist()
# plt.xlabel('Number of Installs')
# plt.ylabel('Frequency')

# In[ ]:


# Converting the Installs column to 
#data['Installs']=data['Installs'].str.replace(r",","").str.replace(r"+","")
#data.drop(data.index[10472], inplace=True)
#data['Installs']=pd.to_numeric(data['Installs'])

data.Installs=data.Installs.apply(lambda x: x.strip('+'))
data.Installs=data.Installs.apply(lambda x: x.replace(',',''))
data.Installs=data.Installs.replace('Free',np.nan)
data.Installs.value_counts()


# In[ ]:


data["Size"].value_counts()
data["Size"].unique()
data['Installs'].value_counts()


# In[ ]:


# Converting the Rating column to integer form
data["Rating"]=pd.to_numeric(data['Rating'])
data['Rating'].value_counts()
#data['Rating'].unique()



# In[ ]:


user_r


# In[ ]:


data.info()
data.describe()
data.columns


# In[ ]:


#Checking for any null values
data.isnull().info()


# # "Reviews" column

# Checking if all values in number of Reviews numeric

# In[ ]:


data.Reviews.str.isnumeric().sum()


# One value is non numeric out of 10841. Lets find its value and id.

# In[ ]:


#data[~data.Reviews.str.isnumeric()]


# In[ ]:


data=data.drop(data.index[10472])


# To check if row is deleted

# In[ ]:


data[10471:].head(2)


# In[ ]:


#data.iloc[10472, data.columns.get_loc('Reviews')] = "3000000"
#data['Reviews']=data['Reviews'][10472].replace("3.0M","3000000")
print(data["Reviews"].tail())
data['Reviews'].unique()
#Here we see that the data type is string for the "Number of reviews",so we need to change it to numeric type
data['Reviews']=pd.to_numeric(data['Reviews'])
print(data["Reviews"].tail())


# In[ ]:


# Which

a=data.groupby(["Category"])["App","Reviews"].mean().reset_index()
print(a)


# In[ ]:


# Which Category of apps have the best Ratings
a=data.groupby(["Category"])["App","Rating"].mean().reset_index()
print(a)


# In[ ]:


import seaborn as sns
# Number of apps per category

g=sns.countplot(x="Category",data=data,palette="Set1")


# In[ ]:


# rating distibution 
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 11.7,8.27
sns.distplot(data["Rating"])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 11.7,8.27
g=sns.kdeplot(data["Rating"],color='Red',shade=True)
g.set_xlabel("Rating")
g.set_ylabel("Frequency")


# Finding
# 
# Average of rating of application in store is around 4 which is very high

# In[ ]:


rcParams['figure.figsize']=20,15

# Let"s try to plot some categorical data using catplot function of the seaborn

z=sns.boxplot(x='Category',y='Rating',data=data)
z.set_xticklabels(z.get_xticklabels(),rotation=55)
y=sns.lineplot(x='Category',y='Rating',data=data)
z.set_xticklabels(z.get_xticklabels(),rotation=55)


# rcParams['figure.figsize']=20,15
# 
# # Let"s try to plot some categorical data using catplot function of the seaborn
# sns.kdeplot(x='Category',y='Installs',data=data)

# In[ ]:


# Number of Categories present
print("The number of categories present are :",len(data["Category"].unique()))


# In[ ]:


# Number of apps in each category
g1 = sns.countplot(x="Category",data=data, palette = "Set1")
g1.set_xticklabels(g1.get_xticklabels(),rotation=55)
plt.title('Count of app in each category',size = 20)


# Family category has the most number of apps 

# In[ ]:


# Review distribution
 
rcParams['figure.figsize'] = 11.7,8.27
g = sns.kdeplot(data.Reviews, color="Green", shade = True)
g.set_xlabel("Reviews")
g.set_ylabel("Frequency")
plt.title('Distribution of Reveiw',size = 20)


# In[ ]:


plt.figure(figsize = (10,10))
sns.regplot(x="Reviews", y="Rating", color = 'darkorange',data=data[data['Reviews']<1000000]);
plt.title('Rating VS Reveiws',size = 20)


# SIZE Column Analysis

# In[ ]:


# It is in object format so we need to deal with it

data['Size'].value_counts()

# we have "varies with device" a lot so we need to change it a bit
data['Size'].replace("Varies with device",np.nan,inplace=True)
data


# #data.Size.value_counts()
# data.Size = (data.Size.replace('k', 10**3))
# data.Size.unique()

# In[ ]:


#data.Size = (data.Size.replace(r'[kM]+$', '', regex=True).astype(float) * \data.Size.str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1)
#            .replace(['k','M'], [10**3, 10**6]).astype(int))


# Converting the Size column from object to float 
# As we see the the size has things like kilo(k) and Mega(M) at end so we first need to remove it 
# And then we convert from object to numeric

# In[ ]:


data.Size=data.Size.str.replace('k','e+3')
data.Size=data.Size.str.replace('M','e+6')
data.Size=pd.to_numeric(data["Size"])
data.Size.head()


# In[ ]:


data.Size.unique()


# In[ ]:


data.hist(column='Size')
plt.xlabel('Size')
plt.ylabel('Frequency')


# # Rating Column

# In[ ]:


data.Rating.value_counts()


# In[ ]:


print("Range:",data.Rating.min(),"-",data.Rating.max())


# In[ ]:


data.Rating.dtype


# In[ ]:


print(data.Rating.isna().sum(),"null values out of ",len(data.Rating))


# In[ ]:


data.Rating.replace(np.nan,data.Rating.mean())


# In[ ]:


data.Rating.unique()


# In[ ]:


data.Rating.hist()
plt.xlabel("Rating")
plt.ylabel("Frequecy")


# # Type column

# In[ ]:


data.Type.value_counts()


# In[ ]:


data.Type.isna().sum()


# In[ ]:


data[data.Type.isna()]


# In[ ]:


data=data.drop(data.index[9148])


# In[ ]:


#Check if the row is removed
print(data[9146:].head(4))


# data.Type.hist()
# plt.xlabel("Type")
# plt.ylabel("Frequency")

# # Price Column

# In[ ]:


data.Price.value_counts()


# In[ ]:


data.Price.unique()


# In[ ]:


data.Price.apply(lambda x: x.strip('$'))


# In[ ]:


#data.Price=pd.to_numeric(data.Price)
#data.Price.hist()
#plt.xlabel("Price")
#plt.ylabel("Frequency")


# In[ ]:


#temp=data[data.Price > 350]
#temp=data.Price.apply(lambda x:True if x >(350) else False)
#data[temp].head(3)


# # Category column

# In[ ]:


data.Category.unique()


# # Content Rating
Here as we have "content Rating" with a space in between and we know that this is not allowed so we need to rename the column
# In[ ]:


data=data.rename(columns={"Content Rating":"Content_Rating"})

data.columns


# In[ ]:


data.Content_Rating.unique()


# In[ ]:


data.Content_Rating.value_counts().plot(kind='bar')
plt.yscale('log')


# # Genres

# In[ ]:


data.Genres.unique()


# In[ ]:


data.Genres.value_counts().plot(kind='bar')


# In[ ]:


sep=";"
rest=data.Genres.apply(lambda x:x.split(sep)[0])
data['Pri_Genres']=rest
data.Pri_Genres.unique()


# In[ ]:


data['Pri_Genres'].head()


# In[ ]:


rest=data.Genres.apply(lambda x:x.split(sep)[-1])
rest.unique()
data['Sec_Genres']=rest
data.Sec_Genres.unique()


# In[ ]:


grouped = data.groupby(['Pri_Genres','Sec_Genres'])
grouped.size().head(15)


# # Last Updated

# In[ ]:


data=data.rename(columns={'Last Updated':'Last_Updated'})
data.columns


# In[ ]:


data.Last_Updated.unique()


# In[ ]:


from datetime import datetime,date
temp1=pd.to_datetime(data['Last_Updated'])
temp1.head()


# In[ ]:


data["Last_Updated_Days"]=temp1.apply(lambda x:date.today()-datetime.date(x))
data.Last_Updated_Days.head()


# # Android Version

# In[ ]:


data=data.rename(columns={'Android Ver':'Android_Version'})
data.columns


# In[ ]:


data.Android_Version.unique()

