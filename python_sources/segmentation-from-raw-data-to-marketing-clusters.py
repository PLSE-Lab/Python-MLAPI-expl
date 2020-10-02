#!/usr/bin/env python
# coding: utf-8

# Segmentation is the practice of dividing a database into groups of observations that are similar in specific ways relevant to marketing.<br>
# Each groups contain individuals that are similar in-between themselves, and different from individuals from the other groups.<br>
# Segmentation is widely used as a marketing tool to create clusters of clients and adapt a relevant strategy for each of them.<br><br>
# In this Kernel, we'll create our own segmentation using the mall customers dataset.<br>
# We'll then see how we can use the segmentation marketing-wise.<br><br>
# Note that it gets particularly interesting for malls, as clusters will let us know which shops of the malls to promote to whom.<br>
# Let's first get a quick look at the dataset.

# In[ ]:


#IMPORTING REQUIRED MODULES
import pandas as pd
pd.options.display.max_columns = None
pd.set_option('display.float_format', lambda x: '%.6f' % x)
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from sklearn.cluster import KMeans
import seaborn as sns
import colorlover as cl
import plotly as py
import plotly.graph_objs as go
py.offline.init_notebook_mode(connected = True)


# In[ ]:


df = pd.read_csv('../input/Mall_Customers.csv', index_col=0)
#changing column names for better manipulability
df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Annual Income (k$)': 'annual_income', 'Spending Score (1-100)': 'spending_score'})
df['gender'].replace(['Female','Male'], [0,1],inplace=True)
df.head()


# Right before jumping in though, we need to scale the data.<br>
# Indeed, creating clusters imply calculating distances between points.<br>
# If a variable has a high scale (goes from 0 to 100 000), the distances between its points will be overweighted vs. a variable that goes from 1 to 10.<br>

# In[ ]:


#SCALING
#stocking mean and standard deviation in a dataframe (we will need them for unscaling)
dfsp = pd.concat([df.mean().to_frame(), df.std().to_frame()], axis=1).transpose()
dfsp.index = ['mean', 'std']
#new dataframe with scaled values
df_scaled = pd.DataFrame()
for c in df.columns:
    if(c=='gender'): df_scaled[c] = df[c]
    else: df_scaled[c] = (df[c] - dfsp.loc['mean', c]) / dfsp.loc['std', c]
df_scaled.head()


# Making clusters with a many variables is interesting in terms of precision.<br>
# However it becomes complicated to have a good vizualization of the clusters with more than 3 independant variables.<br>
# 
# But as all variables seem here interesting, we want use them all. <br>
# We'll create clusters with all 4 variables of the dataset, and still vizualize them.<br><br>
# To do this, we'll first create two clusters "manually" : male vs. female.<br>
# We wouldn't have done that with any kind of dataset, but most mall shops are "gender-oriented".<br> 
# They don't have the same communication and promotion strategies whether they are targeting male or female.<br>
# Therefore it is safe enough to consider that a male will never share a cluster with a female.<br><br>
# 

# In[ ]:


#the two "intuitive" clusters
dff = df_scaled.loc[df_scaled.gender==0].iloc[:, 1:] #no need of gender column anymore
dfm = df_scaled.loc[df_scaled.gender==1].iloc[:, 1:]


# Within these two "intuitevely" created clusters, we'll create another n clusters.<br>
# However this time we won't intervene in the process : we'll let unsupervised learning algorithms search for interesting structures itself.<br>
# The algorithm we'll use is called K-means clustering.<br>
# The algorithm creates the clusters as followed : it maximizes the distances <b>between</b> groups, and minimzes the distances  <b>within</b> groups.
# 
# First, we need to define the number of clusters we want to have within each group.<br>
# On the one hand, the more clusters we'll have, the smaller the distances between points within a cluster will be.<br>
# On the other hand, we don' want to find ourselves with 99 clusters that would be too hard understand for a human and unrelevant marketing-wise.<br><br>
# We need to find just the right number, and this is done using the elbow method :<br>
# When the decrease in distance between points within a cluster gets too small, we consider generating more clusters is unsignificant.<br>
# On the below graph, that would be when the line makes an elbow, also marked with a black dot.<br>

# In[ ]:


def number_of_clusters(df):

    wcss = []
    for i in range(1,20):
        km=KMeans(n_clusters=i, random_state=0)
        km.fit(df)
        wcss.append(km.inertia_)

    df_elbow = pd.DataFrame(wcss)
    df_elbow = df_elbow.reset_index()
    df_elbow.columns= ['n_clusters', 'within_cluster_sum_of_square']

    return df_elbow

dfm_elbow = number_of_clusters(dfm)
dff_elbow = number_of_clusters(dff)

fig, ax = plt.subplots(1, 2, figsize=(17,5))

sns.lineplot(data=dff_elbow, x='n_clusters', y='within_cluster_sum_of_square', ax=ax[0])
sns.scatterplot(data=dff_elbow[5:6], x='n_clusters', y='within_cluster_sum_of_square', color='black', ax=ax[0])
ax[0].set(xticks=dff_elbow.index)
ax[0].set_title('Female')

sns.lineplot(data=dfm_elbow, x='n_clusters', y='within_cluster_sum_of_square', ax=ax[1])
sns.scatterplot(data=dfm_elbow[5:6], x='n_clusters', y='within_cluster_sum_of_square', color='black', ax=ax[1])
ax[1].set(xticks=dfm_elbow.index)
ax[1].set_title('Male');


# For both male and female, we want create 5 clusters.<br>
# We'll extract their centroids (which is the gravity center of the cluster).<br>
# We'll plot only the centroids for cleaner visualization.

# In[ ]:


def k_means(n_clusters, df, gender):

    kmf = KMeans(n_clusters=n_clusters, random_state=0) #defining the algorithm
    kmf.fit_predict(df) #fitting and predicting
    centroids = kmf.cluster_centers_ #extracting the clusters' centroids
    cdf = pd.DataFrame(centroids, columns=df.columns) #stocking in dataframe
    cdf['gender'] = gender
    return cdf

df1 = k_means(5, dff, 'female')
df2 = k_means(5, dfm, 'male')
dfc_scaled = pd.concat([df1, df2])
dfc_scaled.head()


# Now that clusters are defined, we can "unscale" the data and the values of centroids.<br>
# This will let us know the real values of age, income and spending score and thus have a better representation of the clusters.

# In[ ]:


#UNSCALING
#using the mean and standard deviation of the original dataframe, stocked earlier
dfc = pd.DataFrame()
for c in dfc_scaled.columns:
    if(c=='gender'): dfc[c] = dfc_scaled[c]
    else: 
        dfc[c] = (dfc_scaled[c] * dfsp.loc['std', c] + dfsp.loc['mean', c])
        dfc[c] = dfc[c].astype(int)
        
dfc.head()


# Now that we've got our centroids for both our clusters, we can plot them on a 3D plot.<br>
# We apply different colors for female centroids vs. male centroids.

# In[ ]:



def plot(dfs, names, colors, title):

    data_to_plot = []
    
    for i, df in enumerate(dfs):

        x = df['spending_score']
        y = df['annual_income']
        z = df['age']
        data = go.Scatter3d(x=x , y=y , z=z , mode='markers', name=names[i], marker = colors[i])
        data_to_plot.append(data)


    layout = go.Layout(margin=dict(l=0,r=0,b=0,t=40),
        title= title, scene = dict(xaxis = dict(title  = x.name,), 
        yaxis = dict(title  = y.name), zaxis = dict(title = z.name)))

    fig = go.Figure(data=data_to_plot, layout=layout)
    py.offline.iplot(fig)
    

dfcf = dfc[dfc.gender=='female']
dfcm = dfc[dfc.gender=='male']
purple = dict(color=cl.scales['9']['seq']['RdPu'][3:8])
blue = dict(color=cl.scales['9']['seq']['Blues'][3:8])
plot([dfcf, dfcm], names=['male', 'female'], colors=[purple, blue], title = 'Clusters - All Targets')


# Here are all our clusters.<br>
# Now, because we are big capitalistic firm that only wants profit, we'll keep only the interesting clusters.<br>
# That means the clusters that either spend a lot, or have a high income that could allow them to spend a lot.<br>
# In other words, we set aside the clusters that have low income <b> and </b> low spending score.<br>
# We find ourselves with only the <b> primary targets </b>

# In[ ]:


dfc = dfc[(dfc.annual_income>40) & (dfc.spending_score>40)]
dfc = dfc.sort_values('age').reset_index(drop=True)
dfc


# In[ ]:


dfcf = dfc[dfc.gender=='female']
dfcm = dfc[dfc.gender=='male']
purple = dict(color=cl.scales['9']['seq']['RdPu'][3:8])
blue = dict(color=cl.scales['9']['seq']['Blues'][3:8])
plot([dfcf, dfcm], names=['male', 'female'], colors=[purple, blue], title = 'Clusters - Primary Targets')


# It's interesting to see how four of the clusters go two by two, while one cluster stands alone.<br>
# We therefore regroup the pairs, and find ourselves with three groups, which we describe, name and plot :
# 
# - Female in their 30's, average spendings and average income. <b>Younger women - moderated spenders</b>.
# - Male & female in their 30's, high spendings and high income. <b>Rich & independant young adults</b>.
# - Male & female in their 50's, average spendings and average income. <b>Parents - moderated spenders</b>.

# In[ ]:


df1 = dfc.iloc[[0], :]
df2 = dfc.iloc[[1,2], :]
df3 = dfc.iloc[[3,4], :]

names = ['younger women - moderated spenders', 'rich & independant young adults', 'parents - moderated spenders']

colors = []
for i in [1, 3, 5]: 
    colors.append(dict(color = cl.scales['11']['qual']['Paired'][i]))

plot([df1, df2, df3], names=names, colors=colors, title = 'Marketing Clusters - Primary Targets')


# In[ ]:




