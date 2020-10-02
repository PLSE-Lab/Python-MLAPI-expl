#!/usr/bin/env python
# coding: utf-8

# # Google Play Store Apps Data Analysis (EDA)

# ### Used Libraries
# 1. NumPy (Numerical Python)
# 2. Pandas
# 3. Matplotlib
# 4. Seaborn
# 5. Plotly
# 6. Missingno

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualization tools
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import pandas_profiling as pp
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Content:
# 1. Pandas Profiling Report
# 2. Missingno - Missing Data
# 3. Data Cleaning
# 4. Seaborn - Bar Plot
# 5. Plotly - Donut Chart
# 6. Plotly - Box Plot
# 7. Seaborn - Count Plot
# 8. Word Cloud

# ### Reading Data
# 

# In[ ]:


df=pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")


# In[ ]:


df.sample(5)


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# # Pandas Profiling Report

# In[ ]:


report = pp.ProfileReport(df)

report.to_file("report.html")

report


# # Missingno - Missing Data

# In[ ]:


import missingno as msno
msno.matrix(df)
plt.show()


# # Data Cleaning

# In[ ]:


df.columns=[each.replace(" ","_") for each in df.columns]


# In[ ]:


df.columns


# In[ ]:


df["Category"]=[each.replace("_"," ") for each in df.Category]
df["Price"]=[str(each.replace("$","")) for each in df.Price]


# In[ ]:



df.Reviews = pd.to_numeric(df.Reviews, errors='coerce')
df.Price = pd.to_numeric(df.Price, errors='coerce')
df.Rating = pd.to_numeric(df.Rating, errors='coerce')


# # Seaborn - Bar Plot

# In[ ]:


df2 = pd.DataFrame(columns = ['Category'])
df2["Category"]=[each for each in df.Category.unique()]
df2["Count"]=[len(df[df.Category==each]) for each in df2.Category]
df2=df2.sort_values(by=['Count'],ascending=False)

plt.figure(figsize=(25,15))
sns.barplot(x=df2.Category, y=df2.Count)
plt.xticks(rotation= 90)
plt.xlabel('Categorys')
plt.ylabel('Count')
plt.show()


# # Plotly - Donut Chart ( Interactive )
# ### Google Play Apps Android Versiyon Ratio

# In[ ]:


labels = df.Android_Ver.unique()
values=[]
for each in labels:
    values.append(len(df[df.Android_Ver==each]))

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.show()


# # Plotly - Box Plot ( Interactive )
# ### Game, Family and Medical Category (min,q1,median,q3,max value)

# In[ ]:


Category1 = df[df.Category=="GAME"].Rating
Category2 = df[df.Category=="FAMILY"].Rating
Category3 = df[df.Category=="MEDICAL"].Rating

fig = go.Figure()
# Use x instead of y argument for horizontal plot
fig.add_trace(go.Box(x=Category1, name='GAME'))
fig.add_trace(go.Box(x=Category2, name='FAMILY'))
fig.add_trace(go.Box(x=Category3, name='MEDICAL'))

fig.show()


# # Seaborn - Count Plot

# In[ ]:


plt.subplots(figsize=(25,15))
plt.xticks(rotation=90)
ax = sns.countplot(x="Installs", data=df, palette="Set3")


# # Word Cloud

# In[ ]:


plt.subplots(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='black',
                          width=1920,
                          height=1080
                         ).generate(" ".join(df.Category))
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
