#!/usr/bin/env python
# coding: utf-8

# <h1 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">PROJECT CONTENT</h1>
#      
# > ### 1. EXPLORATORY DATA ANALYSIS
# 
# > ### 2. EXPLORATION OF THE SUPERMARKET: MEIJER
# 
# > ### 3. KETO DIET: SHOPPING GUIDE IN THE MEIJER SUPERMARKET
# 
# > ### 4. MACHINE LEARNING: CLUSTERING
# 
# > ### 5. COMPARATIVE ANALYSIS: AGGLOMERATIVE VS KMEANS
# 
# > ### 6. CONCLUSION

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from wordcloud import WordCloud, STOPWORDS 
plt.style.use('seaborn')
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
   for filename in filenames:
       print(os.path.join(dirname, filename))

pd.options.mode.chained_assignment = None # Warning for chained copies disabled


# In[ ]:


a = pd.read_csv("/kaggle/input/world-food-facts/en.openfoodfacts.org.products.tsv",
                       delimiter='\t',
                       encoding='utf-8')


# In[ ]:


#Use this code to show all the 163 columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# # 1- EXPLORATORY DATA ANALYSIS:
# ***
# > ### 1.1 Data cleaning: 
# 
# We First start by visualizing the missing values in our dataset. The features with more than 70% missing values will be dropped.

# In[ ]:


def msv1(data, thresh=20, color='black', edgecolor='black', width=15, height=3):
    """
    SOURCE: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking
    """
    
    plt.figure(figsize=(width,height))
    percentage=(data.isnull().mean())*100
    percentage.sort_values(ascending=False).plot.bar(color=color, edgecolor=edgecolor)
    plt.axhline(y=thresh, color='r', linestyle='-')
    plt.title('Missing values percentage per column', fontsize=20, weight='bold' )
    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh+12.5, 'Columns with more than %s%s missing values' %(thresh, '%'), fontsize=12,weight='bold', color='crimson',
         ha='left' ,va='top')
    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh - 5, 'Columns with less than %s%s missing values' %(thresh, '%'), fontsize=12,weight='bold', color='blue',
         ha='left' ,va='top')
    plt.xlabel('Columns', size=15, weight='bold')
    plt.ylabel('Missing values percentage', weight='bold')
    plt.yticks(weight ='bold')
    
    return plt.show()


# In[ ]:


msv1(a,30, color=('silver', 'gainsboro', 'lightgreen', 'white', 'lightpink'))


# We get rid of all the columns having more than 70% missing values to avoid ending up with misleading results. 118 columns are dropped. We go from 163 columns to 45 columns

# In[ ]:


ab=a.dropna(thresh=106800, axis=1)
print(f"Data shape before cleaning {a.shape}")
print(f"Data shape after cleaning {ab.shape}")
print(f"We dropped {a.shape[1]- ab.shape[1]} columns")


# Now, we start our exploration by checking the countries present in this dataset
# > ### 1.2 Data exploration:

# In[ ]:


countries=ab['countries_en'].value_counts().head(10).to_frame()
s = countries.style.background_gradient(cmap='Blues')
s


# USA and France are the countries with most products in this dataset, the next thing to do is check the brands represented in this dataset

# In[ ]:


brands= ab['brands'].value_counts().head(10).to_frame()
k = brands.style.background_gradient(cmap='Reds')
k


# Since France and USA are the most represented countries in this dataset, the most frequent brands are french and American supermarkets. 'Meijer' supermarket is chosen for further analysis in this study: What do they have in their shelves?

# # 2-Exploration of the supermarket: MEIJER
# ***
# > ### 2.1 Meijer supermarket filtering:

# In[ ]:


#Filter the data and keep just the Meijer brand products:
ac=ab[ab['brands']=='Meijer']


# Let's fill the remaining NaN values with 0

# In[ ]:


ac=ac.fillna(0, axis=1)


# > ### 2.2 The correlation between our features:

# In[ ]:


ac_corr=ac.corr()
f,ax=plt.subplots(figsize=(10,7))
sns.heatmap(ac_corr, cmap='viridis')
plt.title("Correlation between features", 
          weight='bold', 
          fontsize=18)
plt.xticks(weight='bold')
plt.yticks(weight='bold')

plt.show()


# ###### From the correlation heatmap we can see a strong correlation between many features:
# - Sugars and carbohydrates.
# - Fat and saturated fat and energy
# - Carbohydrates and energy
# - Vitamin C and calcium
# - Vitamin C and Vitamin a
# 
# And finally a strong correlation between the nutrition score (FR and UK) and sugars, fat, saturated fat and energy, which means that the scores are given based on the amounts of fat, calories and carbs in the product.
# Let's check the type of relation between the nutrition score and calories:

# In[ ]:


plt.figure(figsize=(15, 6))

plt.scatter(x=ac['nutrition-score-uk_100g'], y=ac['energy_100g'], color='deeppink', alpha=0.5)
plt.title("UK Nutrition score of Meijer's products based on calories ", 
          weight='bold', 
          fontsize=15)
plt.xlabel('Nutrition score UK', weight='bold', fontsize=14)
plt.ylabel('Calories', weight='bold', fontsize=14)
plt.xticks(fontsize=12, weight='bold')
plt.yticks(fontsize=12,weight='bold')


plt.show()


# * We can see a pattern in this scatter plot, the more energy the product has, the higher score it gets.
# Many high calories products have a 0 nutrition score.
# 
# * We already dropped all the columns with more than 70% missing values, and filled the gap with 0s. Now let's keep just the features we are interested in:
# > We bascially keep 7 features:

# In[ ]:


ad=ac[['product_name','energy_100g', 'fat_100g',
       'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g',
       'proteins_100g']]
print(f"we have {ad.shape[0]} products in Meijer supermarkets and {ad.shape[1]} features")


# # 3- Keto diet: Shopping guide in the Meijer supermarket
# ***
# > ### 3.1 Filtering keto products:
# 
# 1. Let's help people willing to start a keto diet with their grosseries: 
# Generally, popular ketogenic resources suggest an average of 70-80% fat from total daily calories, 5-10% carbohydrate, and 10-20% protein. For a 2000-calorie diet, this translates to about 165 grams fat, 40 grams carbohydrate, and 75 grams protein.
# It's just an overview of the products that might be suitable for a keto diet meal plan (since the nutrient values we are dealing with are in the 100g portion).
# We will start our analysis by filtering the rows and columns, discarding the elements that have higher energetic or nutritive values:

# In[ ]:


keto= ad[(ad['energy_100g']<2000)&(ad['carbohydrates_100g']<40)&(ad['fat_100g']<165)&(ad['proteins_100g']<75)]
print(f'We have {keto.shape[0]} keto products in Meijer supermarkets')


# By filtering the columns, we end up with 1057 out of 1995 products, which means:
# - " if you wanna go keto, you gotta give up half of the food available in the supermarket "

# **The distribution of the nutritive values in the filtered keto products:**

# In[ ]:


plt.style.use('seaborn')
sns.set_style('whitegrid')

fig= plt.figure(figsize=(15,10))
#2 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((2,2),(0,0))
plt.hist(keto.energy_100g, bins=3, color='orange', alpha=0.7)
plt.title('Calories',weight='bold', fontsize=18)
plt.yticks(weight='bold')
plt.xticks(weight='bold')
#first row sec col
ax1 = plt.subplot2grid((2,2), (0, 1))
plt.hist(keto.fat_100g, bins=3, alpha=0.7)
plt.title('Fat',weight='bold', fontsize=18)
plt.yticks(weight='bold')
plt.xticks(weight='bold')
#Second row first column
ax1 = plt.subplot2grid((2,2), (1, 0))
plt.hist(keto.proteins_100g, bins=3, color='red', alpha=0.7)
plt.title('Protein',weight='bold', fontsize=18)
plt.yticks(weight='bold')
plt.xticks(weight='bold')
#second row second column
ax1 = plt.subplot2grid((2,2), (1, 1))
plt.hist(keto.carbohydrates_100g, bins=3, color='green', alpha=0.7)
plt.title('Carbs',weight='bold', fontsize=18)
plt.yticks(weight='bold')
plt.xticks(weight='bold')

plt.show()


# ### The distribution analysis:
# ###### 1-Calories:
# The distribution of calories shows a higher frequency of calories in products falling within the range of 0-500 calories, which is a positive thing, since it gives many options for a meal plan.
# ###### 2-Fat:
# Good for the keto diet: Most of the products fall within the range 0-17g, which is good to have fat in several meals.
# ###### 3-Carbohydrates:
# Good for the keto diet: Most of the products have a lower amount of carbs (10g).
# ###### 4-Proteins:
# Good for the keto diet: Most of the products fall within the range 0-13g of proteins.
# 
# * **Now, let's have a look over some keto products:**

# In[ ]:


da=keto.sort_values(by=['energy_100g'],ascending=False).sample(5)
n = da.style.background_gradient(cmap='Purples')
n


# > ### 3.2 Feature engineering:
# 
# We will create new columns to label calories, fat, carbs and proteins, that will be used later in our meal plan choices.
# Let's create functions for our new label columns (calories, carbs, fat, calories) where we categorise them into: low, medium and high.

# In[ ]:


def label_cal (row):
   if row['energy_100g'] < 250  :
      return 'low'
   if row['energy_100g'] > 250 and row['energy_100g'] < 500 :
      return 'medium'
   if row['energy_100g'] > 500 :
      return 'high'
   
   return 'Other'


def label_fat (row):
   if row['fat_100g'] < 10 :
      return 'low'
   if row['fat_100g'] >= 10 and row['fat_100g'] < 20 :
      return 'medium'
   if row['fat_100g'] >= 20 :
      return 'high'
   
   return 'Other'


def label_pro (row):
   if row['proteins_100g'] < 10 :
      return 'low'
   if row['proteins_100g'] >= 10 and row['proteins_100g'] < 20 :
      return 'medium'
   if row['proteins_100g'] >= 20 :
      return 'high'
   
   return 'Other'


def label_carb (row):
   if row['carbohydrates_100g'] < 4 :
      return 'low'
   if row['carbohydrates_100g'] >= 4 and row['carbohydrates_100g'] < 12 :
      return 'medium'
   if row['carbohydrates_100g'] >= 12 :
      return 'high'
   
   return 'Other'

# we add those new columns to the existing keto dataset:
keto['calories'] = keto.apply (lambda row: label_cal(row), axis=1)

keto['fat'] = keto.apply (lambda row: label_fat(row), axis=1)

keto['protein'] = keto.apply (lambda row: label_pro(row), axis=1)

keto['carbs'] = keto.apply (lambda row: label_carb(row), axis=1)

#Create dataframe

db=keto.calories.value_counts().reset_index()
dd= keto.fat.value_counts().reset_index()
de=keto.protein.value_counts().reset_index()
dg=keto['carbs'].value_counts().reset_index()

#Merge them on the 'index' column:
merged=db.merge(dd,on='index').merge(de, on='index').merge(dg, on='index')
mergedstyle = merged.style.background_gradient(cmap='Greens')
mergedstyle


# > ### 3.3 Keto products categories:
# 
# 
# That would help us set our meal plan based on the calories, since a balanced diet requieres an average daily intake of 2000 calories, in the case of having 3 meals a day, it would be around 800 calories per meal, and in order to have a variated meal, there should be at least 3 components, each having a maximum of 250 calories.
# 
# 
# Those are just estimations for the ideal products available in this supermarket that can be useful in a keto meal plan.

# In[ ]:


label1=db['index']
label2=dd['index']
label3=de['index']
label4=dg['index']


fig = plt.figure(figsize=(15,10))
#2 rows 2 cols
#first row, first col
ax1 = plt.subplot2grid((2,2),(0,0))
plt.pie(db.calories,colors=("grey","r","orange"),labels=label1, autopct='%.2f',textprops={'fontsize': 14, 'weight':'bold'})
plt.title('calories',weight='bold', fontsize=18)
#first row sec col
ax1 = plt.subplot2grid((2,2), (0, 1))
plt.pie(dd.fat,colors=("grey","r","orange"),labels=label2, autopct='%.2f',textprops={'fontsize': 14, 'weight':'bold'})
plt.title('fat',weight='bold', fontsize=18)
#Second row first column
ax1 = plt.subplot2grid((2,2), (1, 0))
plt.pie(de.protein,colors=("grey","r","orange"),labels=label3, autopct='%.2f',textprops={'fontsize': 14, 'weight':'bold'})
plt.title('protein',weight='bold', fontsize=18)
#second row second column
ax1 = plt.subplot2grid((2,2), (1, 1))
plt.pie(dg.carbs,colors=("grey","r","orange"),labels=label4, autopct='%.2f',textprops={'fontsize': 14, 'weight':'bold'})
plt.title('carbs',weight='bold', fontsize=18)
plt.show()


# In[ ]:


ketocat=keto[['product_name', 'calories', 'protein','fat','carbs']]
keto_low=ketocat.loc[ketocat['calories']=='low']
keto_medium=ketocat.loc[ketocat['calories']=='medium']
keto_high=ketocat.loc[ketocat['calories']=='high']


# > ### 3.4 WordClouds of the keto products:

# In[ ]:


wordcloud1 = WordCloud(width=600, height=500, background_color='white').generate(' '.join(keto_low['product_name']))
WordCloud.generate_from_frequencies


wordcloud2 = WordCloud(width=600, height=500, background_color='white').generate(' '.join(keto_medium['product_name']))
WordCloud.generate_from_frequencies


wordcloud3 = WordCloud(width=600, height=500, background_color='white').generate(' '.join(keto_high['product_name']))
WordCloud.generate_from_frequencies


fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,8))

fig.suptitle('Low, medium and high calories products', weight='bold', fontsize=20)



ax1.set_title('Low calories products', weight='bold', fontsize=15, color='b')
# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)
im1 = ax1.imshow(wordcloud1, aspect='auto')


ax2.set_title('Medium calories products', weight='bold', fontsize=15, color='b')
im4 = ax2.imshow(wordcloud2, aspect='auto')

ax3.set_title('High calories products', weight='bold', fontsize=15, color='b')
im4 = ax3.imshow(wordcloud3, aspect='auto')

# Make space for title
plt.subplots_adjust(top=0.85)
plt.show()


#  ### Here are some notes:
# - Most of the products in this dataset are either French or American
# - 1057 out of 1995 products in Meijer supermarkets are suitable for a keto diet.
# - Meijer has its own food brand: True goodness
# - There are 385 low calories products in Meijer supermarkets. Main products: Vegetable, pureed baby, beens, water beverage, carrot, strawberry...
# - There are 276 medium calories products in Meijer supermarkets. Main products: Beans, nonfat yogurt, pasta sauce, banana, tomato ketchup...
# - There are  396 high calories products in Meijer supermarkets. Main products: Cheese,chicken, ice cream, cheddar, sausage....
# 
# 
# 

# # 4- Machine Learning: Clustering
# ***
# > ### 4.1 Preprocessing
# 
# We start preparing the data for machine learning, first thing to do is log transform skewed numeric features:
# 
# 
# 

# In[ ]:


ketoc=keto[['energy_100g','fat_100g', 'saturated-fat_100g','carbohydrates_100g', 'sugars_100g', 'proteins_100g']]


# In[ ]:


from scipy.stats import skew

numeric_feats = ketoc.dtypes[ketoc.dtypes != "object"].index

skewed_feats = ketoc[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

ketoc[skewed_feats] = np.log1p(ketoc[skewed_feats])


# Since we didn't detect and eliminate outliers, we will use the robust scaler because it's powerfull against outliers

# In[ ]:


from sklearn.preprocessing import RobustScaler

scaler=RobustScaler()
scaler.fit(ketoc)


# In[ ]:


from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.colors import rgb2hex, colorConverter
from scipy.cluster.hierarchy import set_link_color_palette
import pandas as pd
import scipy.cluster.hierarchy as sch
get_ipython().run_line_magic('pylab', 'inline')


# In[ ]:


#ketoo=keto.drop(['calories', 'protein','fat','carbs'], axis=1 )
#keton=ketoo.set_index('product_name')
#ketonn=keton.T
#ketom=ketonn.reset_index(drop=True)


# Before assigning clusters to our data, we start with a hierarchical cluster analysis. It is an algorithm that groups similar objects into groups called clusters. The endpoint is a set of clusters, where each cluster is distinct from each other cluster, and the objects within each cluster are broadly similar to each other.

# In[ ]:


from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(20, 7))
plt.title("Supermarket food products Dendograms")
plt.xticks(rotation='vertical')


dend = shc.dendrogram(shc.linkage(ketoc, method='ward'))


# This dendogram gives us an idea on how the algorithm clusters the data, it shows 4 different clusters of different sizes, the red cluster is very small in comparison to the green. For the sake of finding sub-clusters inside of big clusters, I will start my analysis with 8 clusters instead of 4. The idea is to split the green and cean clusters into 2 or 3 sub-clusters.
# 
# * The ideal method to find the right number of clusters would be to try the measurement: Within Cluster Sum of Squares (**WCSS**), which measures the squared average distance of all the points within a cluster to the cluster centroid. (The Euclidean distance between a given point and the centroid to which it is assigned)

# > ### 4.2 Agglomerative algorithm:
# 
# An agglomerative algorithm is a type of hierarchical clustering algorithm where each individual element to be clustered is in its own cluster. These clusters are merged iteratively until all the elements belong to one cluster. It assumes that a set of elements and the distances between them are given as input.
# * We set the number of clusters to 8 

# In[ ]:


agc = AgglomerativeClustering(n_clusters=8, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', pooling_func='deprecated')
pred_ag = agc.fit_predict(ketoc)


# In[ ]:


keto['ag_cluster']= agc.fit_predict(ketoc)


# In[ ]:


plt.figure(figsize=(15,5))
plt.style.use('seaborn')
sns.set_style('whitegrid')
keto['ag_cluster'].value_counts().plot(kind='bar', color=['tan', 'crimson', 'silver', 'darkcyan',
                                                          'deeppink', 'deepskyblue','lightgreen', 'orchid'])
plt.ylabel("Count",fontsize=14, weight='bold')
plt.xlabel(' Agglomerative Clusters', fontsize=14, weight='bold')
plt.show()


# Interesting! We have 8 clusters of different sizes: 3 small clusters, 3 big clusters and 2 medium size clusters.

# In[ ]:


#Clusters column
agcluster0=keto[keto['ag_cluster']==0]
agcluster1=keto[keto['ag_cluster']==1]
agcluster2=keto[keto['ag_cluster']==2]
agcluster3=keto[keto['ag_cluster']==3]
agcluster4=keto[keto['ag_cluster']==4]
agcluster5=keto[keto['ag_cluster']==5]
agcluster6=keto[keto['ag_cluster']==6]
agcluster7=keto[keto['ag_cluster']==7]

wordcloud20 = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(' '.join(agcluster0['product_name']))
WordCloud.generate_from_frequencies


wordcloud21 = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(' '.join(agcluster1['product_name']))
WordCloud.generate_from_frequencies


wordcloud22 = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(' '.join(agcluster2['product_name']))
WordCloud.generate_from_frequencies


wordcloud23 = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(' '.join(agcluster3['product_name']))
WordCloud.generate_from_frequencies


wordcloud24 = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(' '.join(agcluster4['product_name']))
WordCloud.generate_from_frequencies


wordcloud25 = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(' '.join(agcluster5['product_name']))
WordCloud.generate_from_frequencies

wordcloud26 = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(' '.join(agcluster6['product_name']))
WordCloud.generate_from_frequencies


wordcloud27 = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(' '.join(agcluster7['product_name']))
WordCloud.generate_from_frequencies

#Create color dictionary for clusters
col_dic = {0:'darkblue',1:'green',2:'darkorange',3:'yellow',4:'magenta',5:'black', 6:'cyan', 7:'lime', 8:'red', 9:'darkviolet', 10:'grey'}
colors3 = [col_dic[x] for x in pred_ag]

#Funciton to plot the clusters
def plot_ag_cluster(keto, color):
    fig, ax = plt.subplots(2, 2, figsize=(12,11)) # define plot area         
    x_cols = ['carbohydrates_100g', 'fat_100g', 'sugars_100g', 'proteins_100g']
    y_cols = ['energy_100g', 'proteins_100g', 'energy_100g', 'carbohydrates_100g']
    for x_col,y_col,i,j in zip(x_cols,y_cols,[0,0,1,1],[0,1,0,1]):
        for x,y,c in zip(ketoc[x_col], ketoc[y_col], colors3):
            ax[i,j].scatter(x,y, color = c)
        ax[i,j].set_title('Scatter plot of ' + y_col + ' vs. ' + x_col) # Give the plot a main title
        ax[i,j].set_xlabel(x_col) # Set text for the x axis
        ax[i,j].set_ylabel(y_col)# Set text for y axis
    plt.show()


# In[ ]:


plot_ag_cluster(ketoc, colors3)


# We can see clear clusters in the first figure. Calories is the sum of the other macromolecules, so having clear clusters that don't overlap much is essensial.
# However, we can't see clear clusters when plotting the products in terms of their proteins or fat values. This can be a good signal, because we don't want a "calories based clustering", otherwise we could just do it manually as we did in previous sections of this work.
# * We will evaluate those clusters further on, now, we try another clustering algorithm: **Kmeans**

# > ### 4.3 K-Means algorithm:
# 
# K-Means is chosen with the following parameters:
# 
# * 8 clusters (n_clusters);
# * Initial cluster centres chosen using K-Means++ (init);
# * The algorithm will be run 10 times using different centroid seeds, with the best chosen using inertia (n_init);
# * Convergence is declared when inertia falls below 1e-4 (tol);
# * We have overriden the default number of CPUs used (n_jobs) from no parallel computing (1) - eases debugging - to all CPUS (-1);
# * Seed for the random selection of initial centres set to 1 (random_state)

# In[ ]:


from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np

kmeans = KMeans(n_clusters=8
                       ,init = 'k-means++'
                       , n_init = 10
                       , tol = 0.0001
                       , n_jobs = -1
                       , random_state = 1).fit(ketoc)
labels2 = kmeans.labels_

centers=kmeans.cluster_centers_


# Kmeans cluster prediction

# In[ ]:


pred1 = kmeans.fit_predict(ketoc)


# In[ ]:


color_km = [col_dic[x] for x in pred1]


# In[ ]:


def plot_km_cluster(keto, color):
    fig, ax = plt.subplots(2, 2, figsize=(12,11)) # define plot area         
    x_cols = ['carbohydrates_100g', 'fat_100g', 'sugars_100g', 'proteins_100g']
    y_cols = ['energy_100g', 'proteins_100g', 'energy_100g', 'carbohydrates_100g']
    for x_col,y_col,i,j in zip(x_cols,y_cols,[0,0,1,1],[0,1,0,1]):
        for x,y,c in zip(ketoc[x_col], ketoc[y_col], color_km):
            ax[i,j].scatter(x,y, color = c)
        ax[i,j].set_title('Scatter plot of ' + y_col + ' vs. ' + x_col) # Give the plot a main title
        ax[i,j].set_xlabel(x_col) # Set text for the x axis
        ax[i,j].set_ylabel(y_col)# Set text for y axis
    plt.show()


# In[ ]:


plot_km_cluster(keto, color_km)


# Same as with agglomerative clustering: Clear clusters in the first figure. 

# In[ ]:


keto['km_cluster']= kmeans.fit_predict(ketoc)


# # 5- Comparative analysis: Kmeans and Agglomerative clustering
# ***
# > ### What is the difference between k-means and hierarchical clustering?
# 
# * In **k-means clustering**, we try to identify the best way to divide the data into k sets simultaneously. A good approach is to take k items from the data set as initial cluster representatives, assign all items to the cluster whose representative is closest, and then calculate the cluster mean as the new representative, until it converges (all clusters stay the same).
# 
# * In **agglomerative clustering** (bottom-up hierarchical clustering), we start with each data item having its own cluster. We then look for the two items that are most similar, and combine them in a larger cluster. We keep repeating until all the clusters we have left are too dissimilar to be gathered together, or until we reach a preset number of clusters.
# 
# Both algorithms have different approaches but are supposed to come up with the same results: **Accurate clusters**.
# > we check in this section how similar are the clusters created by both algorithms:
# 
# 
# 

# In[ ]:


plt.style.use('seaborn')
sns.set_style('whitegrid')


plt.subplots(0,0,figsize=(15,4))
plt.title("Comparison between Kmeans and agglomerative clusters", fontsize=20, weight='bold')

keto['ag_cluster'].value_counts().plot(kind='bar', color=['tan', 'crimson', 'silver', 'darkcyan',                                                          'deeppink', 'deepskyblue','lightgreen', 'orchid'])
plt.ylabel("Count",fontsize=14, weight='bold')
plt.xticks(weight='bold')
plt.xlabel('Agglomerative Clusters', fontsize=14, weight='bold')

plt.subplots(1,0,figsize=(15,4))
keto['km_cluster'].value_counts().plot(kind='bar', color=['tan', 'crimson', 'silver', 'darkcyan',                                                        'deeppink', 'deepskyblue','lightgreen', 'orchid'])
plt.ylabel("Count",fontsize=14, weight='bold')
plt.xticks(weight='bold')
plt.xlabel('Kmeans Clusters', fontsize=14, weight='bold')

plt.tight_layout()
plt.show()


# We already know from the plots that the clusters are not similar:
# * Although the clusters are not similar in terms of cluster numbers, we can see some similarities in the distributions. Both algorithms clustered the data to: 3 small size clusters, 2 medium clusters and 3 big clusters.
# 
# * The smallest clusters in both algorithms are:
# > * Cluster 7 in agglomerative clustering
# > * Cluster 0 in Kmeans clustering
# * Let's check what products do we have in those small clusters

# In[ ]:


#Create clusters column
cluster0=keto[keto['km_cluster']==0]
cluster1=keto[keto['km_cluster']==1]
cluster2=keto[keto['km_cluster']==2]
cluster3=keto[keto['km_cluster']==3]
cluster4=keto[keto['km_cluster']==4]
cluster5=keto[keto['km_cluster']==5]
cluster6=keto[keto['km_cluster']==6]
cluster7=keto[keto['km_cluster']==7]
cluster8=keto[keto['km_cluster']==8]

#Generate word clouds for each cluster
wordcloud10 = WordCloud(width=400, height=300, background_color='white').generate(' '.join(cluster0['product_name']))
WordCloud.generate_from_frequencies

wordcloud11 = WordCloud(width=400, height=300, background_color='white').generate(' '.join(cluster1['product_name']))
WordCloud.generate_from_frequencies

wordcloud12 = WordCloud(width=400, height=300, background_color='white').generate(' '.join(cluster2['product_name']))
WordCloud.generate_from_frequencies

wordcloud13 = WordCloud(width=400, height=300, background_color='white').generate(' '.join(cluster3['product_name']))
WordCloud.generate_from_frequencies

wordcloud14 = WordCloud(width=400, height=300, background_color='white').generate(' '.join(cluster4['product_name']))
WordCloud.generate_from_frequencies

wordcloud15 = WordCloud(width=400, height=300, background_color='white').generate(' '.join(cluster5['product_name']))
WordCloud.generate_from_frequencies

wordcloud16 = WordCloud(width=400, height=300, background_color='white').generate(' '.join(cluster6['product_name']))
WordCloud.generate_from_frequencies

wordcloud17 = WordCloud(width=400, height=300, background_color='white').generate(' '.join(cluster7['product_name']))
WordCloud.generate_from_frequencies



#Plot clusters
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))

fig.suptitle('Comparison between Kmeans and agglomerative clusters', weight='bold', fontsize=20)



ax1.set_title('Agglomerative cluster 7', weight='bold', fontsize=15, color='r')
# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)
im1 = ax1.imshow(wordcloud27, aspect='auto')
plt.xticks(weight='bold')
plt.yticks(weight='bold')


ax2.set_title('Kmeans cluster 0', weight='bold', fontsize=15, color='g')
im4 = ax2.imshow(wordcloud10, aspect='auto')


# Make space for title
plt.subplots_adjust(top=0.85)
plt.xticks(weight='bold')
plt.yticks(weight='bold')

plt.show()


# Effectively, the smallest clusters of Kmeans and agglomerative algorthims contain the same products (olives, salad, manzanilla, pimiento, vegetable...), which means that both clustering algorithms resulted in similar clusters.
# * We can easily categorize those products to: **VEGETABLES**

# # 6- Clustering evaluation
# ***
# We will evaluate here the **kmeans** clusters. 
# * We will try to find some clear clusters with speficic food categories. The main food categories in a supermarket are: 
# * Meat, fruits/vegetables, sweets, bread/pasta, beverages, milk/cheese/yogurt. 
# 
# Let's see if we can find clear food categories in the kmeans clusters

# In[ ]:


f, axarr = plt.subplots(3,2, figsize=(15,17))


fig.suptitle('Title of figure', fontsize=20)


# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)
axarr[0,0].set_title('Products: CLUSTER 0', weight='bold')
axarr[0,0].imshow(wordcloud10, aspect='auto')

axarr[0,1].set_title('Products: CLUSTER 1',weight='bold')
axarr[0,1].imshow(wordcloud11, aspect='auto')

# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)
axarr[1,0].set_title('Products: CLUSTER 3',weight='bold')
axarr[1,0].imshow(wordcloud13, aspect='auto')


# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)
axarr[1,1].set_title('Products: CLUSTER 4',weight='bold')
axarr[1,1].imshow(wordcloud14, aspect='auto')

# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)
axarr[2,0].set_title('Products: CLUSTER 5',weight='bold')
axarr[2,0].imshow(wordcloud15, aspect='auto')


# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)
axarr[2,1].set_title('Products: CLUSTER 7',weight='bold')
axarr[2,1].imshow(wordcloud17, aspect='auto')

# Make space for title
plt.subplots_adjust(top=0.85)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.show()


# **INTERESTING!!!** Some clusters are really clear and well separated:
# * **Cluster 0: Vegetables mainly for salad**
# * **Cluster 1: Beverages**
# * **Cluster 2: Cheese**
# * **Cluster 3: Fruits**
# * **Cluster 5: Meat/Chicken/Fish**
# * **Cluster 7: Vegetables and legumes**
# 
# The remaining 2 clusters contain mixed products, let's check them out!
# 

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))


ax1.set_title('Products: cluster 2', weight='bold', fontsize=15)
# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)
im1 = ax1.imshow(wordcloud12, aspect='auto')


ax2.set_title('Products: CLUSTER 6', weight='bold', fontsize=15)
im4 = ax2.imshow(wordcloud16, aspect='auto')

# Make space for title
plt.subplots_adjust(top=0.85)
plt.show()


# * **Cluster 2: Cheese, meat and vegetables**
# * **Cluster 6: Vegetables, meat, and beverages**
# 

# # 7- Conclusion:
# ***
# We started with a hierarchical cluster analysis. It is an algorithm that groups similar objects into groups, this helped us to guess an ideal cluster number to start our clustering (8 clusters), but it's not the best method. There is a popular method known as elbow method which is used to determine the optimal value of K to perform the K-Means Clustering Algorithm. The basic idea behind this method is that it plots the various values of cost with changing k. ... The lesser number of elements means closer to the centroid. 
# 
# Effectively, k=8 clusters was not the optimum value since we got 2 clusters that are not clear with mixed products, which means **there is still significant room for improvement.** 
# However, in the other clusters, we could see a clear dominance of certain type of food, for example, cluster 5 has mainly: boneless meat, chicken, turkey, ham, bacon, sausages...
# 
# **The clear clusters tend to be the smallest**, with fewer products in comparison to other clusters. **This invites us to increase the number of clusters** in order to decrease the variation between clusters. The variation within the clusters should be as large as possible, while the variation between clusters should be as small as possible.
# 
# 
# 
# 
