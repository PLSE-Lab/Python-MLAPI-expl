#!/usr/bin/env python
# coding: utf-8

# # Crimes Data Analysis and Visualization (EDA)

# ### Used Libraries
# 1. NumPy (Numerical Python)
# 2. Pandas
# 3. Matplotlib
# 4. Seaborn
# 5. Plotly
# 6. Missingno
# 7. Folium

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualization tools
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium.plugins import HeatMap
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Content:
# 1. Missingo - Missing Data
# 2. Data Cleaning
# 3. Seaborn - Count Plot
# 4. Seaborn - Bar Plot
# 5. Plotly - 2D Histogram
# 6. Plotly - Donut Chart
# 7. Plotly - Map Box
# 8. Folium - Map
# 9. Word Cloud

# ### Reading Data

# In[ ]:


df=pd.read_csv("../input/crimes-in-boston/crime.csv",encoding = "ISO-8859-1")


# In[ ]:


df.sample(5)


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# # Missingno - Missing Data

# In[ ]:


import missingno as msno
msno.matrix(df)
plt.show()


# # Data Cleaning
# ### Removed unnecessary

# In[ ]:


df.columns


# In[ ]:


df.drop(columns=['INCIDENT_NUMBER','OFFENSE_CODE','SHOOTING'],inplace=True)


# # Seaborn - Count Plot

# In[ ]:


plt.figure(figsize=(25,15))
ax = sns.countplot(x="HOUR", data=df,
                   facecolor=(0, 0, 0, 0),
                   linewidth=5,
                   edgecolor=sns.color_palette("dark", 24))


# # Seaborn - Bar Plot

# In[ ]:


df2 = pd.DataFrame(columns = ['Offenses'])
df2["Offenses"]=[each for each in df.OFFENSE_CODE_GROUP.unique()]
df2["Count"]=[len(df[df.OFFENSE_CODE_GROUP==each]) for each in df2.Offenses]
df2=df2.sort_values(by=['Count'],ascending=False)

plt.figure(figsize=(25,15))
sns.barplot(x=df2.Offenses.head(50), y=df2.Count.head(50))
plt.xticks(rotation= 90)
plt.xlabel('Offenses')
plt.ylabel('Count')
plt.show()


# # Plotly - 2D Histogram ( Interactive )

# In[ ]:


x = df.DAY_OF_WEEK
y = df.HOUR

fig = go.Figure(go.Histogram2d(
        x=x,
        y=y
    ))
fig.show()


# # Plotly - Donut Chart ( Interactive )

# In[ ]:


labels = df.DAY_OF_WEEK.unique()
values=[]
for each in labels:
    values.append(len(df[df.DAY_OF_WEEK==each]))

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.show()


# # Plotly - Map Box ( Interactive )

# In[ ]:


fig = px.scatter_mapbox(df[df["OFFENSE_CODE_GROUP"]=="Service"], lat="Lat", lon="Long", hover_name="HOUR", hover_data=["YEAR", "HOUR"],
                        color_discrete_sequence=["fuchsia"], zoom=10, height=600)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# # Folium - Map

# In[ ]:


vand=df[df["OFFENSE_CODE_GROUP"]=="Service"].iloc[:,11:13]
vand.rename(columns={'Lat':'latitude','Long':'longitude'}, inplace=True)
vand.latitude.fillna(0, inplace = True)
vand.longitude.fillna(0, inplace = True) 

BostonMap=folium.Map(location=[42.5,-71],zoom_start=10)
HeatMap(data=vand, radius=16).add_to(BostonMap)

BostonMap


# # Word Cloud

# In[ ]:


plt.figure(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='black',
                          width=1920,
                          height=1080
                         ).generate(" ".join(df.OFFENSE_CODE_GROUP))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')
plt.show()


# # Thank You
# 
# If you have any suggestion or advice or feedback, I will be very appreciated to hear them.
# ### Also there are other kernels
# * [FIFA 19 Player Data Analysis and Visualization EDA](https://www.kaggle.com/ismailsefa/f-fa-19-player-data-analysis-and-visualization-eda)
# * [Crimes Data Analysis and Visualzation (EDA)](https://www.kaggle.com/ismailsefa/crimes-data-analysis-and-visualzation-eda)
# * [Google Play Store Apps Data Analysis (EDA)](https://www.kaggle.com/ismailsefa/google-play-store-apps-data-analysis-eda)
# * [World Happiness Data Analysis and Visualization](https://www.kaggle.com/ismailsefa/world-happiness-data-analysis-and-visualization)
# * [Used Cars Data Analysis and Visualization (EDA)](https://www.kaggle.com/ismailsefa/used-cars-data-analysis-and-visualization-eda)
# * [Gender Recognition by Voice Machine Learning SVM](https://www.kaggle.com/ismailsefa/gender-recognition-by-voice-machine-learning-svm)
# * [Iris Species Classify Machine Learning KNN](https://www.kaggle.com/ismailsefa/iris-species-classify-machine-learning-knn)
# * [Breast Cancer Diagnostic Machine Learning R-Forest](https://www.kaggle.com/ismailsefa/breast-cancer-diagnostic-machine-learning-r-forest)
# * [Heart Disease Predic Machine Learning Naive Bayes](https://www.kaggle.com/ismailsefa/heart-disease-predic-machine-learning-naive-bayes)
# * [Mushroom Classify Machine Learning Decision Tree](https://www.kaggle.com/ismailsefa/mushroom-classify-machine-learning-decision-tree)
