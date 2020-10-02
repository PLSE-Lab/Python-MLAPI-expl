#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import cluster
import os


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


# In[4]:


data_for_clustering = pd.read_csv('../input/spotify-dataset-19212020-160k-tracks/data_by_genres.csv')
#this is the data to be used to initialize the centroids for the clusters of genres in the 160k+ tracks


# In[8]:


print(data_for_clustering.columns)


# In[9]:


data_for_clustering.drop(columns=['genres','duration_ms','popularity'])


# In[11]:


data_for_clustering = data_for_clustering[['acousticness','danceability','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','valence']]


# In[12]:


print(data_for_clustering)


# In[ ]:




# In[22]:


data_csv = pd.read_csv('../input/spotify-dataset-19212020-160k-tracks/data.csv')
#this is the data to be used to initialize the centroids for the clusters of genres in the 160k+ tracks


# In[23]:


print(data_csv.columns)


# In[24]:


data_to_cluster = data_csv
print(data_to_cluster)


# In[25]:


data_to_cluster = data_to_cluster.drop(columns=['Unnamed: 0','duration_ms','release_date','popularity','year','id','explicit'])
data_to_cluster = data_to_cluster[['name','artists','acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']]
#currently not considering release year (i.e, music eras) to affect people's preference of music, to be ensembled later


# In[26]:


print(data_for_clustering.iloc[:,:].values.shape)
print(data_to_cluster.iloc[:,:].values.shape)
print(data_for_clustering.columns)
print(data_to_cluster.columns)


# In[27]:


cluster_centers = data_for_clustering.iloc[:,:].values
data = data_to_cluster.iloc[:,:].values


# In[28]:


#APPROACH 1
#normalizing the data as a whole
data_to_scaleandnorm = np.append(data[:,2:],cluster_centers,axis=0)
for i in range(data_to_scaleandnorm.shape[1]):
    data_to_scaleandnorm[:,i] = (data_to_scaleandnorm[:,i] - data_to_scaleandnorm[:,i].mean())/data_to_scaleandnorm[:,i].std()

cluster_centers = data_to_scaleandnorm[:cluster_centers.shape[0],:]
print(cluster_centers.shape)
data[:,2:] = data_to_scaleandnorm[cluster_centers.shape[0]:,:]
print(data.shape)



'''
#APPROACH 2

for col in range(cluster_centers.shape[1]):
    mean = cluster_centers[:,col].mean()
    std = cluster_centers[:,col].std()
    cluster_centers[:,col] = (cluster_centers[:,col] - mean)/std
    data[:,2:] = (data[:,2:] - mean)/std    #mean and std of by_genre data is used in order to treat music as a whole 
    #if we normalized with respect to the 'data', it could have been biased, example: if the data contained a lot of pop, the preference would be towards pop

'''



# In[15]:


#not necessary to cluster, as genrewise cluster centers are known, now we have to calculate genrewise cluster index 

#cluster_model = cluster.KMeans(n_clusters=data_for_clustering.iloc[:,:].values.shape[0], init = cluster_init, n_init=1)



# In[ ]:


#clustering phase
#not required while recommending to user, as the data has been saved in 'data_with_cluster_labels.csv'
#however, 'data_with_cluster_labels' array is required to execute K-nearest neighbours to find recommendations, this is the same array

'''
cluster_column = np.zeros((168592, 1))
for i in range (data.shape[0]):
    dist = np.sum((cluster_centers - data[i,2:])**2,axis=1)
    print(100*i/data.shape[0],"percent done")
    cluster_column[i] = np.argmax(dist)
    print("\t",np.argmin(dist))

data_with_cluster_labels = np.append(data,cluster_column,axis=1)
print(data_with_cluster_labels.shape)

data_to_cluster['clustered_into_genre'] = list(cluster_column)
print(data_to_cluster)
data_to_cluster.to_csv('data_clustered.csv')

np.savetxt('cluster_labels_wrt_data_by_genre.csv',cluster_column,header='clustered_into_genre')
    '''


# In[ ]:


#data_clustered.csv = original 'data.csv' but with an additional column containing ~2600 genre labels from 'data_by_genre.csv'
#cluster_labbels_wrt_data_by_genre.csv = genre labels of all the songs in 'data.csv' as in code above


# In[ ]:


cluster_column = pd.read_csv('../output/kaggle/working/cluster_labels_wrt_data_by_genre.csv')


# In[ ]:


print(cluster_column.shape)
print(data.shape)
print(data[0])


# In[ ]:


#temp = pd.read_csv('../input/spotify-dataset-19212020-160k-tracks/data_by_genres.csv')
#print(temp.iloc[2300:2500,0].values)


# In[ ]:


l = np.random.randint(0,168392)
for i in range(l,l+100):
    plt.scatter(i,cluster_column[i])


# In[ ]:


plt.hist(cluster_column)


# In[ ]:


print(data_to_cluster.columns)


# In[ ]:


'''
CURRENTLY, THE RECOMMENDATIONS ARE POOR BECAUSE OF THE 'DISTANCE FORMULA' USED TO CLUSTER GENRE, TREATING EVERY FEATURE AS EQUAL(THEY HAVE BEEN NORMALIZED ALREADY, SO BY LAW OF LARGE NUMBERS, 
THEY TEND TO STANDARD NORMAL), EXAMPLE THE VOCALNESS AND TEMPO OF SONG HAVE DIFFERENT IMPACT ON SIMILARITY OF TWO SONGS
IN REALITY, SOME FEATURES HAVE MORE WEIGHT THAN OTHER WHILE DECIDING THEIR SIMILARITY, I NEED TO FIND THOSE WEIGHTS

----> CAN TRY TO TRAIN A BINARY-CLASS LOGISTIC REGRESSION MODEL TO DETERMINE WHETHER THEY ARE SIMILAR OR NOT, CAN USE THE 1000 TRAINING EXAMPLES IN 'DATA_W_GENRES.CSV' AS FOLLOWS:
            NOTE: THE INPUT OF THIS NETWORK WOULD BE VECTOR SUBTRACTION OF 2 SONGS' PARAMETERS, AFTER TRAINING THEM, THE SIGMOID UNIT SHALL BE REMOVED TO OBTAIN AN ENCODER, WHOSE OUTPUT WILL BE
            USED AS THE PARAMETER TO DETERMINE DISTANCE BETWEEN TWO SONGS (CURRENTLY USING L2 DISTANCE, WHICH IS NOT PRACTICAL AS EXPLAINED ABOVE)
            (I) GROUP THE SONGS BASED ON GENRES (NAIVE ASSUMPTION, BUT HAS TO GO WITH THIS FOR NOW)
            (II) ASSUME SONGS IN SAME GENRE TO BE SIMILAR, I.E, TRUE LABEL = 1, CAN TRAIN THE LOGISTIC NETWORK NOW
-----> HAVE TO GENERATE GENRE-CLUSTERS AGAIN, BUT THIS TIME, INSTEAD OF USING L2-DISTANCE, WE USE THIS ENCODER
'''


# In[ ]:


#APPROACH 1: SIMPLY SELECTING K NEAREST NEIGHBOURS OUT OF ALL 168592 SONGS

k = 5     #no. of recommendations


#TODO, use better strategy for song clustering
#TODO, deal with duplicates, or colliding names from different artists


song_index = -1
name_not_found = 0

name_list = list(data[:,0])
while(1):
    if(name_not_found == 0):
        print('Enter a song name:\n')
    name = str(input())
    if name in name_list:
        print('Song found, generating recommendations:\n')
        name_not_found = 0
        song_index = name_list.index(name)
    else:
        name_not_found = 1
        print('Please enter a valid song name:\n')
        continue
    dist = np.argsort(np.sum((data[:,2:] - data[song_index,2:])**2,axis=1))          #for calculating k-nearest neighbours
    print(dist.shape)
    for i in range(1,k+1):      #since the song itself will have minimum distance to itself
        index_in_data = list(dist).index(i)
        print("Recommendation ",i,":\n\t")
        print(data[index_in_data][0], " by", data[index_in_data][1],"\n")
    print("Do you want to type another song? (Y/N):\n")
    ans = str(input())
    if(ans=='N'):
        break
    else:
        continue


# In[ ]:


#APPROACH 2: RECOMMENDING SONGS ONLY OUT OF THE SAME CLUSTER


k = 5     #no. of recommendations


song_index = -1
name_not_found = 0

name_list = list(data[:,0])
while(1):
    if(name_not_found == 0):
        print('Enter a song name:\n')
    name = str(input())
    if name in name_list:
        print('Song found, generating recommendations:\n')
        name_not_found = 0
        song_index = name_list.index(name)
    else:
        name_not_found = 1
        print('Please enter a valid song name:\n')
        continue
    
    #returns a subarray of data with same genre, according to our clustering algoritm
    genre_ofinput = np.asarray([cluster_column[song_index]])
    relevant_songs = data[np.in1d(cluster_column, genre_ofinput)]
    
    dist = np.argsort(np.sum((relevant_songs[:,2:] - data[song_index,2:])**2,axis=1))          #for calculating k-nearest neighbours
    print(dist.shape)
    for i in range(1,k+1):      #since the song itself will have minimum distance to itself
        index_in_data = list(dist).index(i)
        print("Recommendation ",i,":\n\t")
        print(relevant_songs[index_in_data][0], " by", relevant_songs[index_in_data][1],"\n")
    print("Do you want to type another song? (Y/N):\n")
    ans = str(input())
    if(ans=='N'):
        break
    else:
        continue


# In[ ]:


#APPROACH 3: RECOMMENDING K SONGS OUT OF ANY OF THE L NEAREST CLUSTERS OF THE INPUT SONG
#DEBUG THIS!!!

k = 5     #no. of recommendations
L = 3     #no. of nearest clusters to consider

song_index = -1
name_not_found = 0

name_list = list(data[:,0])
while(1):
    if(name_not_found == 0):
        print('Enter a song name:\n')
    name = str(input())
    if name in name_list:
        print('Song found, generating recommendations:\n')
        name_not_found = 0
        song_index = name_list.index(name)
    else:
        name_not_found = 1
        print('Please enter a valid song name:\n')
        continue
    
    #returns a subarray of data with same genre, according to our clustering algoritm
    dist_from_cluster_centers = np.argsort(np.sum((cluster_centers - data[song_index,2:])**2,axis=1))  
    seed = np.random.randint(low=1,high=L+1)
    genre_seed = list(dist_from_cluster_centers).index(seed)
    print(genre_seed)
    
    #returns a subarray of data with matching genres, according to our clustering algoritm
    relevant_songs = data[np.in1d(cluster_column, np.asarray([genre_seed]))]
    
    dist = np.argsort(np.sum((relevant_songs[:,2:] - data[song_index,2:])**2,axis=1))          #for calculating k-nearest neighbours
    print(relevant_songs.shape)
    for i in range(1,k+1):      #since the song itself will have minimum distance to itself
        index_in_data = list(dist).index(i)
        print("Recommendation ",i,":\n\t")
        print(relevant_songs[index_in_data][0], " by", relevant_songs[index_in_data][1],"\n")
    print("Do you want to type another song? (Y/N):\n")
    ans = str(input())
    if(ans=='N'):
        break
    else:
        continue


# In[ ]:


x = np.array([1,1000,500,100])
print(np.argsort(x))
asdsa = np.argsort(x)
print(list(asdsa).index(3))
print(data[data[]])


# In[ ]:





# In[ ]:




