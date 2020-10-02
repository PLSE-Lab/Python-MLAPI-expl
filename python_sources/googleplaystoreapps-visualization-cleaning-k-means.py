#!/usr/bin/env python
# coding: utf-8

# # Exploring, Visualizing and Cleaning Google Play Store Apps DATA

# ## 1. Business understanding

# Google Play store, formerly Android Market, is a digital distribution service operated and developed by Google. It serves as the official app store for the Android operating system, allowing users to browse and download applications developed with the Android studio, unity 3D.... and published through Google. Google Play store also serves as a digital media store, offering music, books, movies, and television programs

# ## 2. Data exploring

# ### 2.1 Collect initial data

# https://www.kaggle.com/lava18/google-play-store-apps

# ### 2.2 Importing data & Libraries 

# In[ ]:


# Thanks @Lavanya Gupta whose kernel inspired me!
# Please check https://www.kaggle.com/lava18/all-that-you-need-to-know-about-the-android-market


# import required packages



from matplotlib import pyplot as plt
import plotly
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from scipy import stats
plt.style.use("ggplot")
color = sns.color_palette()
sns.set(rc={'figure.figsize':(25,15)})
import warnings
warnings.filterwarnings('ignore')
from pylab import rcParams


# In[ ]:


df = pd.read_csv('../input/googleplaystore.csv') # read the data


# ###  2.3 Data analysing 

# In[ ]:


df.shape
df.head()


# ### 2.4 Classification according to user ratings

# In[ ]:


df['Rating'].describe()


# In[ ]:


# rating distibution 
rcParams['figure.figsize'] = 16.7,8.27
g = sns.kdeplot(df.Rating, color="Red", shade = True)
g.set_xlabel("Rating")
g.set_ylabel("Frequency")
plt.title('Distribution of Rating',size = 20)


# Average of rating of application in store is around 4 which is very high

# ### 2.5 Classification of applications according to their categories

# In[ ]:


print( len(df['Category'].unique()) , "categories")

print("\n", df['Category'].unique())


# In[ ]:


g = sns.countplot(x="Category",data=df, palette = "Set1")
g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")
g 
plt.title('Count of app in each category',size = 20)


# Game and Family category are the most appearances for application in store
# 

# ### 2.6  Classification of android applications (Free / Paid):
# 

# In[ ]:


df['Type'].value_counts().plot.bar(width=0.6,rot=0) 
plt.xlabel('Applications')
plt.ylabel('nombre d"applications')
plt.show()


# Free application category are the most appearances for application in store

# ###  2.7 Classification of applications by age

# In[ ]:


BlueOrangeWapang = ['#fc910d','#ff0040','#239cd3','#1674b1','#8000ff']
plt.figure(figsize=(10,10))
label_names=df['Content Rating'].value_counts().sort_index().index
size = df['Content Rating'].value_counts().sort_index().tolist()
my_circle=plt.Circle( (0,0), 0.5, color='white')
plt.pie(size, labels=label_names, colors=BlueOrangeWapang)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


# ## 3. Data Cleaning 

# 
# 1. Convert app size to MB
# 2. Remove '+' from 'Number of Installs'
# 3. Remove 'NaN'
# 4. Remove '$' from 'Price'
# 5. Convert 'String' to 'int' or 'float'

# In[ ]:


# data cleanning

df["Size"] = df["Size"].apply(lambda x: str(x).replace('Varies with device', 'NaN') 
                              if "Varies with device" in str(x) else x)
df["Size"] = df["Size"].apply(lambda x: str(x).replace('M','') 
                              if 'M' in str(x) else x)
df["Size"] = df["Size"].apply(lambda x: str(x).replace(',','') 
                              if ',' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace('+', '') 
                              if '+' in str(x) else x)
df['Size'] = df["Size"].apply(lambda x: float(str(x).replace('k',''))/1024 
                              if 'k' in str(x) else x)


# In[ ]:


df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in x else x)
df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', '') if ',' in x else x)
df['Installs'] = df['Installs'].apply(lambda x: x.replace('Free', 'NaN') if 'Free' in x else x)
df['Installs'] = df['Installs'].apply(lambda x: x.replace('Paid', 'NaN') if 'Paid' in x else x)


# In[ ]:


df["Price"] = df["Price"].apply(lambda x: str(x).replace('$','') if '$' in x else x)


# In[ ]:


df = df.dropna()


# In[ ]:


df['Size'] = df['Size'].apply(lambda x:float(x))
df['Installs'] = df['Installs'].apply(lambda x:int(x))
df['Reviews'] = df['Reviews'].apply(lambda x:int(x))
df['Price'] = df['Price'].apply(lambda x:float(x))
df = df.dropna()


# ## 4. K-Means:
# 
# 

# k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells.

# 
# 1. Saving the attributs we want to use to 'data'
# 2. Using mean normilization to normilize the raw data
# 3. Use TensorFlow K-means to classify the apps into 3 groups (num_iterations = 20)

# In[ ]:


data = df[['Rating', 'Size', 'Installs', 'Reviews', 'Price']]


# In[ ]:


def mean_norm(x):
    x = np.array(x)
    x = (x - np.mean(x))/np.std(x)
    return x


# In[ ]:


data['Rating'] = mean_norm(data['Rating'])
data['Size'] = mean_norm(data['Size'])
data['Installs'] = mean_norm(data['Installs'])
data['Reviews'] = mean_norm(data['Reviews'])
data['Price'] = mean_norm(data['Price'])
result = data


# In[ ]:


data = np.array(data)
print(data[0:5])


# In[ ]:


def input_fn():
    return tf.data.Dataset.from_tensors(tf.convert_to_tensor(data, dtype=tf.float32)).repeat(1)


# In[ ]:


x = input_fn()
print(x)


# In[ ]:


num_clusters = 3
kmeans = tf.contrib.factorization.KMeansClustering(
    num_clusters=num_clusters, use_mini_batch=False)


# In[ ]:


num_iterations = 30
previous_centers = None
for _ in range(num_iterations):
    kmeans.train(input_fn)
    cluster_centers = kmeans.cluster_centers()
    if previous_centers is not None:
        dis = previous_centers - cluster_centers
        print('dleta' + str(dis))
    previous_centers = cluster_centers
    print("socre:" + str(kmeans.score(input_fn)))
print("the centers are: " + str(previous_centers))
center = previous_centers


# In[ ]:


cluster_indices = list(kmeans.predict_cluster_index(input_fn))


# In[ ]:


result['Group'] = cluster_indices


# In[ ]:


result['Group'] = result['Group'].apply(lambda x: str(x))
result['Group'] = result['Group'].apply(lambda x: x.replace('0', 'Group1') if '0' == x else x)
result['Group'] = result['Group'].apply(lambda x: x.replace('1', 'Group2') if '1' == x else x)
result['Group'] = result['Group'].apply(lambda x: x.replace('2', 'Group3') if '2' == x else x)


# ### 4.1 Data Visualization

# In[ ]:


ra = df['Rating']
s = df['Size']
i = df['Installs']
re = df['Reviews']
c = result['Group']
p = df['Price']

sns.pairplot(pd.DataFrame(list(zip(ra, s, np.log(i), np.log(re), c, p)), 
                        columns=['Rating','Size', 'Installs', 'Reviews', 'Group', 'Price']),hue='Group', palette="Set2")


# In[ ]:


num_of_app_in_group = result['Group'].value_counts().sort_values(ascending=True)
data1 = [go.Pie(
        labels = num_of_app_in_group.index,
        values = num_of_app_in_group.values
)]
plotly.offline.iplot(data1, filename='apps_per_group')


# # Conclusion

# Through the analysis and exploration of this data, we come to the obvious, in order to finally establish an effective strategy for the development of mobile applications.
