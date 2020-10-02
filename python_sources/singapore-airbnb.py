#!/usr/bin/env python
# coding: utf-8

# ## Importing the python module

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
'''Seaborn and Matplotlib Visualization'''
import matplotlib                  # 2D Plotting Library
import matplotlib.pyplot as plt
import seaborn as sns              # Python Data Visualization Library based on matplotlib
import geopandas as gpd            # Python Geospatial Data Library
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

'''Plotly Visualizations'''
import plotly as plotly                # Interactive Graphing Library for Python
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)

'''Spatial Visualizations'''
import folium
import folium.plugins

'''NLP - WordCloud'''
import wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

'''Machine Learning'''
import sklearn
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor


# ![Skyline](https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/1_singapore_city_skyline_dusk_panorama_2011_%28cropped_for_Wikivoyage%29.jpg/1920px-1_singapore_city_skyline_dusk_panorama_2011_%28cropped_for_Wikivoyage%29.jpg)

# ![Singapore Airbnb](https://cdn.dnaindia.com/sites/default/files/styles/full/public/2019/06/17/837540-airbnb-logo.jpg)

# ![Map](https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Singapore_Districts.png/1920px-Singapore_Districts.png)

# In[ ]:


df = pd.read_csv('/kaggle/input/singapore-airbnb/listings.csv')


# In[ ]:


#by default head shows us top 5 records
df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


#finding count of total null values in each column
df.isna().sum()


# ** Dealing with missing values**

# In[ ]:


sns.set(rc={'figure.figsize':(19.7,8.27)})

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


#Fields not needed to our problem
to_drop = ["last_review","reviews_per_month"]
    
# Drop selected fields in place
df.drop(to_drop, inplace=True, axis=1)    


# In[ ]:


df.dropna(inplace=True)
df.shape


# ## Exploratory Data Analysis

# In[ ]:


sns.distplot(df["price"])


# In[ ]:


sns.scatterplot(x='price',y='minimum_nights',data=df)


# In[ ]:


sns.countplot(df["neighbourhood_group"])


# In[ ]:


plt.figure(figsize=(10,6))
sns.scatterplot(df.longitude,df.latitude,hue=df.neighbourhood_group)
plt.ioff()


# **Hotel Distribution through a Heat Map**

# In[ ]:


import folium
from folium.plugins import HeatMap
m=folium.Map([1.44255,103.79580],zoom_start=11)
HeatMap(df[['latitude','longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)
display(m)


# In[ ]:


ng = df[df.price <250]
plt.figure(figsize=(10,6))
sns.boxplot(y="price",x ='neighbourhood_group' ,data = ng)
plt.title("neighbourhood_group price distribution < 250")
plt.show()


# > >Central_Region has the most expensive accomodation followed by East_Region
#  

#  **Getting the room details**

# In[ ]:


import plotly.offline as pyo
import plotly.graph_objs as go
roomdf = df.groupby('room_type').size()/df['room_type'].count()*100
labels = roomdf.index
values = roomdf.values

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])
fig.show()


# > > We can see that Entirehome/apt is most prefered and Shared room is least preffered

# **Room types occupied by neighbourhood_group**

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x = 'room_type',hue = "neighbourhood_group",data = df)
plt.title("Room types occupied by the neighbourhood_group")
plt.show()


# In[ ]:


#catplot room type and price
plt.figure(figsize=(10,6))
sns.catplot(x="room_type", y="price", data=df);
plt.ioff()


# > >We can clearly see that most of all room types lie below 2000. 

# In[ ]:


#word cloud
from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in df.name)
# Create and generate a word cloud image:
wordcloud = WordCloud(max_words=200, background_color="white").generate(text)
plt.figure(figsize=(10,6))
plt.figure(figsize=(15,10))
# Display the generated image:
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.show()


# > > Our word cloud shows the words that are more often used in the Name of this listings

# In[ ]:


#let's comeback now to the 'name' column as it will require litte bit more coding and continue to analyze it!

#initializing empty list where we are going to put our name strings
_names_=[]
#getting name strings from the column and appending it to the list
for name in df.name:
    _names_.append(name)
#setting a function that will split those name strings into separate words   
def split_name(name):
    spl=str(name).split()
    return spl
#initializing empty list where we are going to have words counted
_names_for_count_=[]
#getting name string from our list and using split function, later appending to list above
for x in _names_:
    for word in split_name(x):
        word=word.lower()
        _names_for_count_.append(word)
        
#we are going to use counter
from collections import Counter
#let's see top 25 used words by host to name their listing
_top_20_w=Counter(_names_for_count_).most_common()
_top_20_w=_top_20_w[0:20]



#now let's put our findings in dataframe for further visualizations
sub_w=pd.DataFrame(_top_20_w)
sub_w.rename(columns={0:'Words', 1:'Count'}, inplace=True)


# In[ ]:


#we are going to use barplot for this visualization
plt.figure(figsize=(10,6))
viz_5=sns.barplot(x='Words', y='Count', data=sub_w)
viz_5.set_title('Counts of the top 20 used words for listing names')
viz_5.set_ylabel('Count of words')
viz_5.set_xlabel('Words')
viz_5.set_xticklabels(viz_5.get_xticklabels(), rotation=80)


# **Sorting rooms according to maximum number of reviews**

# In[ ]:


df1=df.sort_values(by=['number_of_reviews'],ascending=False).head(1000)
df1.head()


# **Map below showing rooms with Highest number of reviews**

# In[ ]:


import folium
from folium.plugins import MarkerCluster
from folium import plugins
print('Rooms with the most number of reviews')
Long=103.91492
Lat=1.32122
mapdf1=folium.Map([Lat,Long],zoom_start=10,)

mapdf1_rooms_map=plugins.MarkerCluster().add_to(mapdf1)

for lat,lon,label in zip(df1.latitude,df1.longitude,df1.name):
    folium.Marker(location=[lat,lon],icon=folium.Icon(icon='home'),popup=label).add_to(mapdf1_rooms_map)
mapdf1.add_child(mapdf1_rooms_map)

mapdf1


# **Room Availability**

# In[ ]:


plt.figure(figsize=(10,6))
plt.scatter(df.longitude, df.latitude, c=df.availability_365, cmap='spring', edgecolor='black', linewidth=1            , alpha=1)

cbar = plt.colorbar()
cbar.set_label('availability_365')


# In[ ]:


plt.figure(figsize=(10,6))
sub_6=df[df.price<500]
viz_4=sub_6.plot(kind='scatter', x='longitude',y='latitude',label='availability_365',c='price',cmap=plt.get_cmap('jet'),colorbar=True,alpha=0.4,figsize=(10,10))
viz_4.legend()
plt.ioff()


# > > Red color dots are the apartment or rooms with higher price.

# ## Predictions using machine learnng model 

# In[ ]:


#prepare data
df.drop(['name','id','host_name'],axis=1,inplace=True)


# **Converting categorical features to numeric features**

# In[ ]:


'''Encode labels with value between 0 and n_classes-1.'''
le = preprocessing.LabelEncoder()                                            # Fit label encoder
le.fit(df['neighbourhood_group'])
df['neighbourhood_group']=le.transform(df['neighbourhood_group'])    # Transform labels to normalized encoding.

le = preprocessing.LabelEncoder()
le.fit(df['neighbourhood'])
df['neighbourhood']=le.transform(df['neighbourhood'])

le = preprocessing.LabelEncoder()
le.fit(df['room_type'])
df['room_type']=le.transform(df['room_type'])

df.sort_values(by='price',ascending=True,inplace=True)

df.head()


# **Training a model**

# In[ ]:


#Train Linear Regression model

lm = LinearRegression()

X = df[['host_id','neighbourhood_group','neighbourhood','latitude','longitude','room_type','minimum_nights','number_of_reviews','calculated_host_listings_count','availability_365']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

lm.fit(X_train,y_train)


# **Getting predictions**

# In[ ]:


predicts = lm.predict(X_test)
error_airbnb = pd.DataFrame({
        'Actual Values': np.array(y_test).flatten(),
        'Predicted Values': predicts.flatten()})
error_airbnb.head()


# In[ ]:


title=['Pred vs Actual']
fig = go.Figure(data=[
    go.Bar(name='Predicted', x=error_airbnb.index, y=error_airbnb['Predicted Values']),
    go.Bar(name='Actual', x=error_airbnb.index, y=error_airbnb['Actual Values'])
])

fig.update_layout(barmode='group')
fig.show()

