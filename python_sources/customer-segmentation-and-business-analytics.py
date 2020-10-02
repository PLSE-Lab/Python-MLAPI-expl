#!/usr/bin/env python
# coding: utf-8

# In this notebook, we try to segment customers into different categories based on their purchasing behaviour. Conclusions from customer segmentation can give us bussiness insights and strategies. Feel free to fork the complete project here: https://github.com/Hari365/customer-segmentation-python. Which has a segmentation based on customers rather than just orders. There's also a model which classifies the customers based on purchasing behaviour.

# ##### Topics explored in the notebook

# 1) Importing required libraries and data<br>
# 2) Simple visualization of the data and few  samples<br>
# 3) Visualisation via pca and pairplot<br>
# 4) Checking for dependent variables<br>
# 5) Outlier detection and handling<br>
# 6) Cluster analysis: Are there clusters, how many?<br>
# 7) Clustering and interpretation<br>
# 8) Deriving conclusions

# ### 1) Importing required libraries and data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
import matplotlib as mpl


# In[ ]:


import itertools


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data = pd.read_csv('../input/ulabox_orders_with_categories_partials_2017.csv')


# In[ ]:


data.head()


# ### 2) Simple visualization

# In[ ]:


data.describe()


# 1. customer is the unique customer id. <br>
# 2. order is the unique order id. <br>
# 3. total_items is the number of products bought in the order. <br>
# 4. discount% is the amount of discount provided during the purchase, the negative values in discount stands for extra amount the customer paid to ulabox as delivery charge or any other mode of fee. <br>
# 5. weekday is the day of the week in which the order was placed.<br>
# 6. hour is the time in which the order is placed. <br>
# 7. Food% is the amount of money spent on non fresh food in the purchase, it may include grocery products like sugar, coffee  powder, oats etc.<br>
# 8. Fresh% is the amount of money spent on fresh food like milk, fruits, vegetables etc.<br>
# 9. Drinks% is most probably the percentage of amount spent on alchohol like wine, vodka, scotch etc. There is a teeny tiny chance that these also include soft drinks.<br>
# 10. Home% is the percentage of money spent in home accessories.<br>
# 11. Beauty% is the percentage of amount spent in beauty products<br>
# 12. Health% is the percentage of amount spent in medicine or health products like protein supplement, carb supplement etc.<br>
# 13. Baby% is the percentage spent in baby products.<br>
# 14. Pets% is the percentage spent in pet products like pedigree.

# In[ ]:


data[data['discount%']<0].sort_values(by='discount%', ascending=True).head(10)


# -> drinks% and negative discount are highly correlated, may be the company imposed a lot of inconvenience and transport charges on drinks.

# #### Selecting samples

# In[ ]:


indices = [56,2459,908,23632,1803,218,592,349]
data.iloc[indices, :]


# -> 56 order seems to depend on ulabox for grocery, fresh food and drinks. <br>
# -> 2459 and 908 order seems to depend on ulabox for everything, probably most valuable customers. <br>
# -> 23632 order seems to buy a lot of drinks from ulabox in spite of the negative discount, which implies extra charges.<br>
# -> 1803 buys a lot of pet products, the order must be by a pet lover. <br>
# -> 218 buys a lot of home decoratory accessories. <br>
# -> 592 must be a woman, who would like her beauty products, to be delivered by ulabox at her door.<br>
# -> 349 seems to be parents, who newly had a baby.

# There may not be any relevant information in the hour in which the order was placed, but the weekday in which the order was placed may reveal some information about weekend buyers. Hece let's keep it.

# Let's remove customer, order and hour features from the data

# In[ ]:


df = data.drop(['customer', 'order', 'hour'], axis=1)
frame = data


# ### 3) Visualization via pca and pairplot

# Let's try and do pca of the features and see the explained variance and plots

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA(n_components=11)
pca.fit(df.values)


# This function accepts pca object and data frame as arguements and plots the scatter plot of first four principal components.

# In[ ]:


def pca_2d_plot(pca, df):
    fig = plt.figure(figsize=(10,10))
    transformed_data = pca.transform(df.values)
    data = pd.DataFrame(transformed_data, columns=['dim'+str(i) for i in range(1,12)])
    sns.lmplot(x='dim1', y='dim2', data=data, size=12, fit_reg=False, scatter_kws={'s':8});
    sns.lmplot(x='dim3', y='dim4', data=data, size=12, fit_reg=False, scatter_kws={'s':8});
    plt.show()


# In[ ]:


pca_2d_plot(pca, df)


# Now let's plot the pairplots and see the variations and distributions of features with respect to each other.

# In[ ]:


figure = plt.figure(figsize=(20,20))
sns.pairplot(df);
plt.show()


# -> total_items is skewed, applying a log transformation will help the clustering.<br>
# -> when discount% increases total_items icreases which makes sense, people will buy more on discount.<br>
# -> below the 0 discount line only Drinks% has non zero percentage entries. Food%, Fresh% etc. have only zero percentage entries in negative discount area.<br>
# -> it makes sense that the plots in the right bottom are bound by the line x+y = 100, as the data is actually in percentage x+y <= 100.<br>
# -> the distribution plots are more and more skewed as we move towards the right bottom, as pet products, baby products and health products are brought by very less people.

# ### 5) Outlier detection

# In[ ]:


fig = plt.figure(figsize=(16,12))
sns.distplot(df['total_items']);
plt.show()


# This distribution is skewed negatively, let's apply a log transformation.

# In[ ]:


df['total_items'] = np.log(df['total_items'])
fig = plt.figure(figsize=(16,12))
sns.distplot(df['total_items']);
plt.show()


# That's better

# #### Turkey Outlier Detection

# -> According to Turkey method a point is an outlier if it lies 1.5 times inter quartile distance to the right of third quartile or if it lies 1.5 times inter quartile distance to the left of first quartile.<br>
# -> For more info refer: https://en.wikipedia.org/wiki/Outlier

# This function takes df as an arguement and columns for which outlier detection has to be done, as an optional arguement. It returns a dictionary whose keys are column names and elements are indices of outlier points in the corresponding columns. It also prints the number of outliers in every column.

# In[ ]:


def turkey_outlier_detector(df, cols=None):
    if cols  is None:
        cols = [str(s) for s in df.describe().columns]
        
    q1 = {}
    q3 = {}
    iqd = {}
    r_limit = {}
    l_limit = {}
    outlier_count = {}
    outlier_indices = {}
    for col in cols:
        q1[col] = np.percentile(df[col].values, 25)
        q3[col] = np.percentile(df[col].values, 75)
        iqd[col] = q3[col] - q1[col]
        r_limit[col] = q3[col] + 1.5*iqd[col]
        l_limit[col] = q1[col] - 1.5*iqd[col]
        data_outlier = df[~((df[col]<r_limit[col]).multiply(df[col]>l_limit[col]))]
        outlier_count[col] = data_outlier.shape[0]
        outlier_indices[col] = data_outlier.index
        
    for col in cols:
        print('_'*25)
        print(col+'-'*8+'>'+str(outlier_count[col]))
        
    return outlier_indices


# In[ ]:


outlier_indices = turkey_outlier_detector(df)


# -> The outliers in Health% and Pets% are due to the fact that, lot people don't buy these products and the entries are mostly 0.<br>
# -> The outliers in Food%, Fresh% etc. are due to the 0% and 100% entries which is a completely natural phenomenon in this scenerio.<br>
# -> The outliers in discount% is also due to 0% and 100% entries.<br>
# -> For these features let's acknoledge the fact that there are outliers and leave it there.<br>
# -> Let's remove the outliers in total_items.

# In[ ]:


df.drop(outlier_indices['total_items'], inplace=True)


# In[ ]:


frame.drop(outlier_indices['total_items'], inplace=True)


# ## 6) Are there clusters in the data, how many clusters?

# #### Elbow Method

# -> The first method we are going to try is the elbow method.<br>
# -> In this method we plot the sum of distances of all the data points to the correspoding cluster centeroids vs number of clusters, for a range of number of clusters.<br>
# -> If there is a elbow in the plot the point at which elbow occured is the number of clusters present in the data.<br>
# -> We are lucky if we see an elbow in the plot, but in most cases the plot will just be smooth revealing no information about the number of clusters.<br>

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[ ]:


X = scaler.fit_transform(df.values)


# In[ ]:


clusters = range(3,31)
inertia = []
for n in clusters:
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(clusters, inertia);
plt.show()


# We got lucky! There is somewhere around 10 clusters in the data.

# #### Silhoutte Score

# -> a(i) is the sum of the sum of distances of the ith data point to the other data points in it's cluster.<br>
# -> Calculate the sum of distance of ith data point to the points in every other cluster.<br>
# -> b(i) is the sum of distances from ith data point to all points in a cluster, for which sum of distances is munimum.<br>
# -> silhoutte score, s(i) = 1-a(i)/b(i)<br>
# -> If a data point is more similar to it's own cluster and very much different from other clusters, then 
# a(i)<<b(i), greater will be the silhoutte score.<br>
# -> The silhoutte score we plot is the average of it over all the data points.<br>

# In[ ]:


def plot_silhoutte_score(X, max_clusters=20):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    num_clusters = range(2,max_clusters+1)
    sil_score = []
    for n in num_clusters:
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X)
        preds = kmeans.predict(X)
        sil_score.append(silhouette_score(X, preds))
        
    fig, ax = plt.subplots(figsize=(12,12))
    ax.plot(num_clusters, sil_score)
    plt.show()


# In[ ]:


plot_silhoutte_score(X,30)


# We plot number of clusters vs silhoutte score, the silhoutte score hits it's maximum at around 10 clusters.

# ### Validity Index

# -> When the number of clusters is less than the correct number of clusters then the data is under partitioned, if the number of clusters is more than the correct number of clusters then the data is over partitioned.<br>
# -> Sum of distances from the data points to the corresponding cluster centers is a measure of under partition.<br>
# -> Number of clusters devided by minimum distance between two clusters is a measure of over partition, it increases when the data is more over partitioned.<br>
# -> A normalized sum of these two can help finding the actual number of clusters.<br>
# -> This idea is published in this paper:http://armi.kaist.ac.kr/korean/files/_2001______________________a_novel_validity_index_for_determination_of_the_optimal_number_of_clutters_.pdf

# The same has been implemented in the following functions.

# In[ ]:


def under_partition_measure(X, k_max):
    from sklearn.cluster import KMeans
    ks = range(1,k_max+1)
    UPM = []
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        UPM.append(kmeans.inertia_)
    return UPM


# In[ ]:


def over_partition_measure(X, k_max):
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import  pairwise_distances
    ks = range(1,k_max+1)
    OPM = []
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        d_min = np.inf
        for pair in list(itertools.combinations(centers, 2)):
            d = pairwise_distances(pair[0].reshape(1,-1), pair[1].reshape(1,-1), metric='euclidean')
            if d<d_min:
                d_min = d
        OPM.append(k/d_min)
    return OPM


# In[ ]:


def validity_index(X, k_max):
    UPM = under_partition_measure(X, k_max)
    OPM = over_partition_measure(X, k_max)
    UPM_min = np.min(UPM)
    OPM_min = np.min(OPM)
    UPM_max = np.max(UPM)
    OPM_max = np.max(OPM)
    norm_UPM = []
    norm_OPM = []
    for i in range(k_max):
        norm_UPM.append((UPM[i]-UPM_min)/(UPM_max-UPM_min))
        norm_OPM.append((OPM[i]-OPM_min)/(OPM_max-OPM_min))
        
    validity_index = np.array(norm_UPM)+np.array(norm_OPM)
    fig, ax = plt.subplots(figsize=(12,12))
    ax.plot(range(1,k_max+1), validity_index)
    return validity_index


# In[ ]:


_ = validity_index(X, 30)


# This again gives us a surety that, there are around 10 clusters.

# All our analysis so far suggests there could be around 10 clusters in the data, let's now manually examine and try to interpret the meaning of these clusters.

# ### 7) Clustering and interpretation

# In[ ]:


kmeans_10 = KMeans(n_clusters=10, random_state=42)
kmeans_10.fit(X)
frame['labels'] = kmeans_10.predict(X)


# In[ ]:


frame[frame['labels']==0].head(10)


# In[ ]:


frame[frame['labels']==0].describe()


# -> These are the class of people who have ordered drinks a lot.<br>
# -> These people had to face a lot of extra charges for drink purchases.<br>
# -> They are potential customers, as we all know drinks can be addictive atleast in a teeny tiny level.

# In[ ]:


frame.loc[frame['labels']==0, 'class'] = 'drink_buyers'


# In[ ]:


frame[frame['labels']==1].head(10)


# In[ ]:


frame[frame['labels']==1].describe()


# -> This class of orders buy a lot of Food, Fresh and Drinks. With Fresh being more dominant.<br>
# -> These orders might be a little valuable, as these orders cover Food%, Fresh%, Drinnks%, Home%.

# In[ ]:


frame.loc[frame['labels']==1, 'class'] = 'loyals_fresh'


# In[ ]:


frame[frame['labels']==2].head(10)


# In[ ]:


frame[frame['labels']==2].describe()


# -> These are again very loyal customers who depend on ulabox for a lot of things.<br>
# -> They tend to buy grocery a little more, let's call them loyals grocery.

# In[ ]:


frame.loc[frame['labels']==2, 'class'] = 'loyals_grocery'


# In[ ]:


frame[frame['labels']==3].head(10)


# In[ ]:


frame[frame['labels']==3].describe()


# -> Theese customers buy a lot of beauty products.<br>
# -> They also buy considerable amount of grocery and drinks.<br>
# -> Probably woman who shop for home.

# In[ ]:


frame.loc[frame['labels']==3, 'class'] = 'beauty_concious'


# In[ ]:


frame[frame['labels']==4].head(10)


# In[ ]:


frame[frame['labels']==4].describe()


# -> This is the class of people who had bought a lot of health products, let's call them health concious people

# 

# In[ ]:


frame.loc[frame['labels']==4, 'class'] = 'health_concious'


# In[ ]:


frame[frame['labels']==5].head(10)


# In[ ]:


frame[frame['labels']==5].describe()


# -> These customers buy all kinds of products from ulabox, fresh, drinks and food dominnantly.<br>
# -> These are the loyal customers of ulabox who depends on ulabox for everything. <br>
# -> Let's call them loyals. <br>

# In[ ]:


frame.loc[frame['labels']==5, 'class'] = 'loyals'


# In[ ]:


frame[frame['labels']==6].head(10)


# In[ ]:


frame[frame['labels']==6].describe()


# -> This class of customers seem to buy non fresh food a lot, let's call them grocery shoppers.<br>
# -> These are probably monthly regular shopppers of the company.

# 

# In[ ]:


frame.loc[frame['labels']==6, 'class'] = 'grocery_shoppers'


# In[ ]:


frame[frame['labels']==7].head(10)


# In[ ]:


frame[frame['labels']==7].describe()


# -> This should be the class of orders that buy a lot of home utility products like floor cleaner, curtains, washing powder etc.<br>
# -> Let's call this people home decorators.

# In[ ]:


frame.loc[frame['labels']==7, 'class'] = 'home_decorators'


# In[ ]:


frame[frame['labels']==8].head(10)


# In[ ]:


frame[frame['labels']==8].describe()


# -> This is the class of pet lovers, that's obvious from the data we see

# In[ ]:


frame.loc[frame['labels']==8, 'class'] = 'pet_lovers'


# In[ ]:


frame[frame['labels']==9].head(10)


# In[ ]:


frame[frame['labels']==9].describe()


# -> These class of people have brought baby products a lot.<br>
# -> They must be couple with new babies, let's call them new parents.

# In[ ]:


frame.loc[frame['labels']==9, 'class'] = 'new_parents'


# That was nicely interpretable!

# ### 8) Deriving Conclusions

# In[ ]:


def pca_2d_plot_labels(pca, df, frame):
    plt.figure(figsize=(18,18));
    transformed_data = pca.transform(df.values)
    data = pd.DataFrame({'dim1':transformed_data[:,0], 'dim2':transformed_data[:,1], 'labels':frame['class'].values})
    sns.lmplot(x='dim1',y='dim2',hue='labels',data=data, fit_reg=False, size=16);
    data1 = pd.DataFrame({'dim2':transformed_data[:,1], 'dim3':transformed_data[:,2], 'labels':frame['class'].values})
    sns.lmplot(x='dim2',y='dim3',hue='labels',data=data1, fit_reg=False, size=16);
    plt.show()


# In[ ]:


pca_2d_plot_labels(pca, df, frame)


# -> From the 2d plot we see that clusters are nicely separated in space.

# In[ ]:


frame.groupby('class')['total_items'].describe()


# -> The variation of total_items with class is not very sound, all classes of orders have similar number of total item counts.

# In[ ]:


frame.groupby('class')['discount%'].describe()


# -> The customers who have placed orders on grocery have been seen to enjoy a lot of discount, may be there was a stock clearance sale or a promotional sale ulabox.

# In[ ]:


frame['class'].value_counts().sort_values(ascending=False)


# In[ ]:


plt.figure(figsize=(9,9))
frame['class'].value_counts().sort_values(ascending=False).plot.pie(autopct='%1.0f%%', labels=list(frame['class'].value_counts().sort_values(ascending=False).index))
plt.show()


# -> Our hypothesised loyal costomers are placed at the top when it comes to number of orders.<br>
# -> Our next hypothesis of drink buyers being potential customers is also subtantiated.<br>
# -> Pet lovers are very less in number, ulabox should buy less pet products accordingly.<br>
# -> When seeing the large discount enjoyed by grocery shoppers in the previous data frame and the less number of grocery shoppers here. They are supposedly customers who brought only on the discount sale.<br>
# -> ulabox can actually frame their buying strategies according to these numbers.<br>

# In[ ]:


plt.figure(figsize=(9,9))
frame[frame['discount%']<0]['class'].value_counts().sort_values(ascending=False).plot.pie(autopct='%1.0f%%', labels=frame[frame['discount%']<0]['class'].value_counts().sort_values(ascending=False).index)
plt.show()


# In[ ]:


frame[(frame['discount%']<0).multiply(frame['class']!='drink_buyers')].describe()


# -> From the table we can say, even the people in other clusters who had to pay a negative discount have brought a lot drinks.<br>

# In[ ]:


frame[frame['discount%']<0].shape[0]


# -> Only 124 among 30k had to pay an extra charge, that's not a pain killer problem.

# In[ ]:




