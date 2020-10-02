#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


crime_rates=pd.read_csv("../input/USArrests.csv", index_col=0)


# [This helped](https://stackoverflow.com/questions/26098710/rename-unnamed-column-pandas-dataframe) to create a data frame ready to be normalized from the originally imported csv.

# In[3]:


crime_rates.head()


# This data set contains statistics, **in arrests per 100,000 residents** for assault, murder, and rape in each of the 50 US states in 1973. Also given is the percent of the population living in urban areas.

# In[4]:


crime_rates.describe()


# # Normalizing data for any type of clustering

# In[5]:


#standardize the data to normal distribution
from sklearn import preprocessing
crime_rates_standardized = preprocessing.scale(crime_rates)
print(crime_rates_standardized)
crime_rates_standardized = pd.DataFrame(crime_rates_standardized)


# # K-means Clustering

# The number of clusters has to be be decided when k-means clustering is used unlike hierarchical clustering. Let's start with creating scree plot. Scree plot is a plot between **WCSS** (Within cluster sum of squares) and number of clusters. Without sound domain knowledge or in the scenarios with unclear motives, the scree plots help us decide the number of clusters to specify.

# In[6]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(10, 8))
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(crime_rates_standardized)
    wcss.append(kmeans.inertia_) #criterion based on which K-means clustering works
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# The scree plot levels off at **k=4** and let's use it to determine the clusters.

# ### What is random_state?

# To set a seed and make the randomization more deterministic. Read [in detail](https://scikit-learn.org/stable/glossary.html#term-random-state).

# In[7]:


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(crime_rates_standardized)

y_kmeans


# In[8]:


#beginning of  the cluster numbering with 1 instead of 0
y_kmeans1=y_kmeans+1

# New list called cluster
cluster = list(y_kmeans1)
# Adding cluster to our data set
crime_rates['cluster'] = cluster


# In[9]:


#Mean of clusters 1 to 4
kmeans_mean_cluster = pd.DataFrame(round(crime_rates.groupby('cluster').mean(),1))
kmeans_mean_cluster


# In[14]:


import seaborn as sns

plt.figure(figsize=(12,6))
sns.scatterplot(x=crime_rates['Murder'], y = crime_rates['Assault'],hue=y_kmeans1)


# > The above scatter shows a distribution of how the states are scattered and clusters are visible based on `Murders` and `Assaults`. And there is a positive correlation between occurrence of `Murder` and `Assault` in different states.

# ### States in cluster 1

# In[11]:


crime_rates[crime_rates['cluster']==1]


# In[12]:


from IPython.display import HTML
import base64

df = crime_rates
def create_download_link( df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(df)


# ### A tableau viz  - all the states split into clusters

# In[13]:


get_ipython().run_cell_magic('HTML', '', "\n<div class='tableauPlaceholder' id='viz1558006161579' style='position: relative'>\n<noscript>\n<a href='#'>\n<img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;US&#47;USCrimeRatesClusters&#47;Dashboard&#47;1_rss.png' style='border: none' /></a>\n</noscript>\n\n<object class='tableauViz'  style='display:none;'>\n<param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />\n<param name='embed_code_version' value='3' />\n<param name='site_root' value='' />\n<param name='name' value='USCrimeRatesClusters&#47;Dashboard' />\n<param name='tabs' value='no' />\n<param name='toolbar' value='yes' />\n<param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;US&#47;USCrimeRatesClusters&#47;Dashboard&#47;1.png' /> <param name='animate_transition' value='yes' />\n<param name='display_static_image' value='yes' />\n<param name='display_spinner' value='yes' />\n<param name='display_overlay' value='yes' />\n<param name='display_count' value='yes' />\n</object>\n</div>               \n<script type='text/javascript'>                    var divElement = document.getElementById('viz1558006161579');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='700px';vizElement.style.height='527px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# # Insights

# * **Cluster 3** - south western states (however, definition changes for different sources) maxes out in three (`Assault`, `Urbanpop`, `Rape`) out of four crimes.
# * **Cluster 4** stands out for minimum crime rate as the means of 3 crimes are least relative to other crimes
# * Coincidentally, most of the southern region states belong to **cluster 2** where `Murder`s seems to be popular.
# * The southern states in **cluster 4** have more number of average arrests/ 100,000 persons in all the three crime categories than states (half of mid-western states) in **cluster 2** in spite of the average `Urbanpop` (%) being very close. For causality, more variables are needed apart from the number of arrests to understand the inferences like the above. For instance, it could be the density of people or number of people with a different socio-economic statuses in that state.

# # Cheers to these references

# * https://medium.com/@ikekramer/tableau-visual-in-jupyter-notebook-7b9faf60e8fd
# * Download csv - https://medium.com/ibm-data-science-experience/how-to-upload-download-files-to-from-notebook-in-my-local-machine-6a4e65a15767
# * https://medium.com/datadriveninvestor/unsupervised-learning-with-python-k-means-and-hierarchical-clustering-f36ceeec919c
