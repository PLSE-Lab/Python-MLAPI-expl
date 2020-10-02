#!/usr/bin/env python
# coding: utf-8

# # Used Cars Data Analysis and Visualization (EDA)

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
# 1. Missingno - Missing Data
# 2. Data Cleaning
# 3. Seaborn - Bar Plot
# 4. Plotly - Donut Chart
# 5. Plotly - 2D Histogram
# 6. Plotly - Map Box
# 7. Folium - Map

# ### Reading Data

# In[ ]:


df=pd.read_csv("../input/dataset/craigslistVehicles.csv")


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


df.drop(columns=['url','image_url','VIN'],inplace=True)


# # Seaborn - Bar Plot

# In[ ]:


df=df.sort_values(by=['odometer'],ascending=False)
plt.figure(figsize=(25,15))
sns.barplot(x=df.manufacturer, y=df.odometer)
plt.xticks(rotation= 90)
plt.xlabel('Manufacturer')
plt.ylabel('Odometer')
plt.show()


# # Plotly - Donut Chart ( Interactive )

# In[ ]:


gasLabels = df[df["fuel"]=="gas"].paint_color.value_counts().head(10).index
gasValues = df[df["fuel"]=="gas"].paint_color.value_counts().head(10).values
dieselLabels = df[df["fuel"]=="diesel"].paint_color.value_counts().head(10).index
dieselValues = df[df["fuel"]=="diesel"].paint_color.value_counts().head(10).values
electricLabels = df[df["fuel"]=="electric"].paint_color.value_counts().head(10).index
electricValues = df[df["fuel"]=="electric"].paint_color.value_counts().head(10).values

from plotly.subplots import make_subplots

# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=gasLabels, values=gasValues, name="Gas Car"),
              1, 1)
fig.add_trace(go.Pie(labels=dieselLabels, values=dieselValues, name="Diesel Car"),
              1, 2)
fig.add_trace(go.Pie(labels=electricLabels, values=electricValues, name="Electric Car"),
              1, 3)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.show()


# # Plotly - 2D Histogram ( Interactive )
# ### Effect of Type on paint color

# In[ ]:


x = df.type
y = df.paint_color

fig = go.Figure(go.Histogram2d(
        x=x,
        y=y
    ))
fig.show()


# # Plotly - Map Box ( Interactive )

# In[ ]:


fig = px.scatter_mapbox(df[df["type"]=="bus"], lat="lat", lon="long", hover_name="paint_color", hover_data=["paint_color", "price"],
                        color_discrete_sequence=["fuchsia"], zoom=4, height=600)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# # Folium - Map

# In[ ]:


cars=df[df["type"]=="bus"].iloc[:,17:19]
cars.rename(columns={'lat':'latitude','long':'longitude'}, inplace=True)
cars.latitude.fillna(0, inplace = True)
cars.longitude.fillna(0, inplace = True) 

CarMap=folium.Map(location=[42.5,-71],zoom_start=4)
HeatMap(data=cars, radius=16).add_to(CarMap)
CarMap.save('index.html')
CarMap


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
