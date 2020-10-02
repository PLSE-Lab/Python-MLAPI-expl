#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# For this project you'll dig into a large amount of text and apply most of what you've covered in this unit and in the course so far.
# 
# First, pick a set of texts. This can be either a series of novels, chapters, or articles. Anything you'd like. It just has to have multiple entries of varying characteristics. At least 100 should be good. There should also be at least 10 different authors, but try to keep the texts related (either all on the same topic of from the same branch of literature - something to make classification a bit more difficult than obviously different subjects).
# 
# This capstone can be an extension of your NLP challenge if you wish to use the same corpus. If you found problems with that data set that limited your analysis, however, it may be worth using what you learned to choose a new corpus. Reserve 25% of your corpus as a test set.
# 
# The first technique is to create a series of clusters. Try several techniques and pick the one you think best represents your data. Make sure there is a narrative and reasoning around why you have chosen the given clusters. Are authors consistently grouped into the same cluster?
# 
# Next, perform some unsupervised feature generation and selection using the techniques covered in this unit and elsewhere in the course. Using those features then build models to attempt to classify your texts by author. Try different permutations of unsupervised and supervised techniques to see which combinations have the best performance.
# 
# Lastly return to your holdout group. Does your clustering on those members perform as you'd expect? Have your clusters remained stable or changed dramatically? What about your model? Is it's performance consistent?
# 
# If there is a divergence in the relative stability of your model and your clusters, delve into why.
# 
# Your end result should be a write up of how clustering and modeling compare for classifying your texts. What are the advantages of each? Why would you want to use one over the other? Approximately 3-5 pages is a good length for your write up, and remember to include visuals to help tell your story!
# 
# ### Outline
# - [Data](#data)
# - [Clustering](#clustering)
# - [Unsupervised feature selection](#unsupervised)
# - [Classification models](#classification)
# - [Holdout Group](#holdout)

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import stft
import librosa
import time
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFdr, chi2
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.neighbors import NearestCentroid,KNeighborsClassifier
from sklearn.cluster import KMeans,MeanShift, estimate_bandwidth, AffinityPropagation, SpectralClustering
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.neural_network import BernoulliRBM


# <a id = "data"></a>
# # Data
# The [dataset](https://github.com/karoldvl/ESC-50) is a collection of audio data labelled with the entity that made the sound. It is widely known as the ESC-50 because there are 50 labels in the set each with 40 samples. I am taking a selection of labels, namely the ESC-10. The paper [ESC Dataset for Environmental Sound Classification](http://karol.piczak.com/papers/Piczak2015-ESC-Dataset.pdf) describes a baseline score of 72.7% using a Random Forest Classifier on 10 classes after extracting frame-level features. 

# In[2]:


import os
os.listdir('../input/wav_concatenated_compressed/');


# In[3]:


# list of esc-10 sounds
esc_10_path = ['302-Sneezing.wav','101-Dog.wav','409-Clocktick.wav','301-Cryingbaby.wav','102-Rooster.wav','501-Helicopter.wav','201-Rain.wav','202-Seawaves.wav','502-Chainsaw.wav','203-Cracklingfire.wav']
path = '../input/wav_concatenated_compressed/'
esc=[]
clips=[]
for esc_path in esc_10_path:
    data, Fs = sf.read(path + esc_path)
    hop = 5*Fs # samples are concatenated in 5 second sections
    for n in range(40):
        clips.append(list(data[n*hop:(n+1)*hop]))
        esc.append(esc_path[4:-4])
df_time = pd.DataFrame(clips)
df_time['esc'] = pd.Categorical(esc)
print("Rows",df_time.shape[0],"\nColumns",df_time.shape[1])


# In[4]:


sample_clip_nums = [121,350,50,390]
fig,ax = plt.subplots(3,4,sharex=True,sharey=False,figsize=(15,15));
for x,sample_clip_num in zip(range(4),sample_clip_nums):
        # time series
    ax[0,x].plot(np.linspace(0,5,len(clips[sample_clip_num])),clips[sample_clip_num])
    ax[0,x].set_title("Time Series: " + esc[sample_clip_num])
    if x == 0: ax[0,x].set_ylabel("Amplitude")
        # spectrogram
    f, t, Zxx = stft(clips[sample_clip_num],Fs)
    ax[1,x].pcolormesh(t,f,np.abs(Zxx))
    ax[1,x].set_title("Spectrogram: " + esc[sample_clip_num])
    if x ==0: ax[1,x].set_ylabel("Frequency (Hz)")
        # mfccs
    mfccs = librosa.feature.mfcc(np.array(clips[sample_clip_num]),sr=Fs, n_mfcc=20, n_fft=1024, hop_length=256)
    mfccs_slice = mfccs[1:13,:]
    ax[2,x].pcolormesh(np.linspace(0,5,len(mfccs_slice[0])),range(1,len(mfccs_slice)+1,1),mfccs_slice)
    ax[2,x].set_title("MFCCs: " + esc[sample_clip_num])
    ax[2,x].set_xlabel("Time (s)")
    if x ==0: ax[2,x].set_ylabel("Frequency Bin")
plt.show()
#plt.savefig('Sample Sounds.png', bbox_inches='tight')
# size of each 'image'
print ("Time Series size:",len(clips[sample_clip_num]))
print ("Spectrogram size:",len(Zxx),"by",len(Zxx[0]),"=",(len(Zxx) * len(Zxx[0])),"pixels")
print ("MFCC size:",len(mfccs_slice),"by",len(mfccs_slice[0]),"=",(len(mfccs_slice) * len(mfccs_slice[0])),"pixels")


# In[5]:


start = time.time()
# Piczak features: for each mfcc frame calculate mean and standard deviation, include zero crossing rate
mfcc_means = []
mfcc_stds = []
for clip in clips:
    #get mfcc 
    mfcc_temp = librosa.feature.mfcc(np.array(clip),sr=Fs, n_mfcc=20, n_fft=1024, hop_length=256)
    mfccs_slice = mfcc_temp[1:13,:] # only use coefficients 1 to 12 as per Piczak
    #get mfcc mean and std
    mfcc_means.append(np.mean(mfccs_slice,axis=0))
    mfcc_stds.append(np.std(mfccs_slice,axis=0))
#piczak dataframe
def name_cols(base,length):
    return [base+str(x) for x in range(length)]
mfcc_mean_cols = name_cols("mfcc_mean_",len(mfcc_means[0]))
mfcc_std_cols = name_cols("mfcc_std_",len(mfcc_stds[0]))
df_piczak = pd.concat([pd.DataFrame(mfcc_means,columns=mfcc_mean_cols),
                       pd.DataFrame(mfcc_stds,columns=mfcc_std_cols)],axis=1)
df_piczak['esc'] = pd.Categorical(esc)
print("Shape of Piczak dataframe",df_piczak.shape)
print("Processing time {:0.1f} seconds".format(time.time()-start))


# <a id = "clustering"></a>
# # Clustering

# In[6]:


X = df_piczak.drop(['esc'],axis=1)
Y = df_piczak['esc']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=42)
n_clusters=len(esc_10_path)


# In[7]:


# piczak clustering kmeans
km_piczak = KMeans(n_clusters=n_clusters,max_iter=300,tol=1e-4,n_jobs=-1,random_state=42,).fit(X_train)
print(pd.crosstab(Y_train,km_piczak.labels_))
print("\nARI",adjusted_rand_score(Y_train,km_piczak.labels_))
print("Silhouette Score",silhouette_score(X_train,km_piczak.predict(X_train)))


# In[8]:


# mean shift
bandwidth = estimate_bandwidth(X_train, random_state=42,n_jobs=-1)
ms_piczak = MeanShift(bandwidth=bandwidth, bin_seeding=True,n_jobs=-1).fit(X_train)
print("\nARI",adjusted_rand_score(Y_train,ms_piczak.labels_))
#print("Silhouette Score",silhouette_score(X_train,ms_piczak.predict(X_train)))


# In[9]:


# spectral clustering
sc_piczak = SpectralClustering(n_clusters=n_clusters,gamma=1.0,random_state=42,n_jobs=-1)
sc_piczak_pred = sc_piczak.fit_predict(X_train)
print("\nARI",adjusted_rand_score(Y_train,sc_piczak.labels_))
print("Silhouette Score",silhouette_score(X_train,sc_piczak_pred))


# In[10]:


# affinity propogation
af_piczak = AffinityPropagation(damping = .7).fit(X_train)
print(pd.crosstab(Y_train,af_piczak.labels_))
print("\nARI",adjusted_rand_score(Y_train,af_piczak.labels_))
print("Silhouette Score",silhouette_score(X_train,af_piczak.predict(X_train)))


# <a id = "unsupervised"></a>
# # Unsupervised feature selection

# ### PCA

# In[11]:


piczak_pca_scree = PCA(whiten=True)
piczak_pca_scree.fit(X_train);


# In[12]:


plt.plot(piczak_pca_scree.explained_variance_ratio_[:100],label="Per Component")
plt.plot(piczak_pca_scree.explained_variance_ratio_[:100].cumsum(),label="Cumulative")
plt.title("Scree Plot: PCA")
plt.legend()
plt.xlabel("Number of Components")
plt.ylabel("Explained Variance Ratio")
plt.show()


# In[32]:


n_components = 35
start_time = time.time()
piczak_pca = PCA(n_components=n_components,whiten=True)
X_pca_train = piczak_pca.fit_transform(X_train)
X_pca_test = piczak_pca.transform(X_test)
print("Processing time: {:0.4f} seconds".format(time.time() - start_time))
print ("Percent variance explianed by {} components\n{:0.2f}%".format(n_components,piczak_pca.explained_variance_ratio_.sum()*100.0))


# ### Nearest Shrunken Centroids

# In[33]:


# Nearest Shrunken Centroids
X_nsc = X_train
print("Shrinkage\t Accuracy")
curr_val = 0
last_val = 0
start_time = time.time()
for shrinkage in np.arange(0.0,5.0,.1):
    nsc = NearestCentroid(shrink_threshold = shrinkage)
    last_val = curr_val
    curr_val = cross_val_score(nsc,X_nsc,Y_train).mean()
    if curr_val < last_val:
        print("Stopped early\tUse last shrinkage value to avoid loss in accuracy")
        break
    print(shrinkage,"\t\t",curr_val)
print("Processing time: {:0.4f} seconds".format(time.time() - start_time))


# ### Restricted Boltzmann Machine

# In[29]:


start_time = time.time()
rbm = BernoulliRBM(n_components=1000,learning_rate = 1e-6,random_state=42)
rbm.fit(X_train)
X_rbm_train = rbm.transform(X_train)
X_rbm_test = rbm.transform(X_test)
print("Processing time: {:0.4f} seconds".format(time.time() - start_time))
print("Shape of RBM transformed data\n{}".format(X_rbm_train.shape))


# <a id = "classificaiton"></a>
# # Classification models
# "MFCCs were computed using the librosa package with default settings resulting in frame length of 11.6 ms. Discarding the 0th coefficient, first 12 MFCCs and zero-crossing rate were summarized for each clip with their mean and standard deviation across frames...The ESC-10 dataset had an average classification accuracy [of] 72.7% for the random forest ensemble..."

# ### Models

# In[16]:


# ranfom forest on MFCC features
rfc_mfcc = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2,n_jobs=-1, random_state=42)
rfc_mfcc.fit(X_train,Y_train)
# random forest on PCA features
rfc_pca = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2,n_jobs=-1, random_state=42)
rfc_pca.fit(X_pca_train,Y_train)
# SVC on PCA features
svc_pca = SVC(C=1.0, kernel='rbf',gamma='auto',random_state=42)
svc_pca.fit(X_pca_train,Y_train)
# Lasso on PCA features
lasso_pca = LogisticRegression(penalty='l1',C=1.0,random_state=42)
lasso_pca.fit(X_pca_train,Y_train)
# Nearest Shrunken Centroid
# use shrinkage of 0.2 from above
nsc_mfcc = NearestCentroid(shrink_threshold = 0.2)
nsc_mfcc.fit(X_train,Y_train)
# SVC on RBM features
svc_rbm = SVC(C=1, kernel='linear',gamma='auto',random_state=42)
svc_rbm.fit(X_rbm_train,Y_train)
# Lasso on RBM features
lasso_rbm = LogisticRegression(penalty='l1',C=1,random_state=42);
lasso_rbm.fit(X_rbm_train,Y_train);


# In[34]:


print("Random Forest using MFCC features\nTrain Accuracy: {:0.4f}\nTest Accuracy: {:0.4f}\t\tOVERFITTING".format(rfc_mfcc.score(X_train,Y_train),rfc_mfcc.score(X_test,Y_test)))
print("\nRandom Forest using PCA features\nTrain Accuracy: {:0.4f}\nTest Accuracy: {:0.4f}\t\tOVERFITTING".format(rfc_pca.score(X_pca_train,Y_train),rfc_pca.score(X_pca_test,Y_test)))
print("\nSVC using PCA features\nTrain Accuracy: {:0.4f}\nTest Accuracy: {:0.4f}\t\tBEST".format(svc_pca.score(X_pca_train,Y_train),svc_pca.score(X_pca_test,Y_test)))
print("\nLasso Regression using PCA features\nTrain Accuracy: {:0.4f}\nTest Accuracy: {:0.4f}".format(lasso_pca.score(X_pca_train,Y_train),lasso_pca.score(X_pca_test,Y_test)))
print("\nNearest Shrunken Centroid\nTrain Accuracy: {:0.4f}\nTest Accuracy: {:0.4f}".format(nsc_mfcc.score(X_train,Y_train),nsc_mfcc.score(X_test,Y_test)))
print("\nSVC using RBM features\nTrain Accuracy: {:0.4f}\nTest Accuracy: {:0.4f}".format(svc_rbm.score(X_rbm_train,Y_train),svc_rbm.score(X_rbm_test,Y_test)))
print("\nLasso Regression using RBM features\nTrain Accuracy: {:0.4f}\nTest Accuracy: {:0.4f}".format(lasso_rbm.score(X_rbm_train,Y_train),lasso_rbm.score(X_rbm_test,Y_test)))


# <a id = "holdout"</a>
# # Holdout group

# ### K-means on Test Data

# In[18]:


# mfcc clustering kmeans
km_piczak = KMeans(n_clusters=n_clusters,max_iter=300,tol=1e-4,n_jobs=-1,random_state=42,).fit(X_test)
print("K-Means Clustering on MFCC Testing Data")
print(pd.crosstab(Y_test,km_piczak.labels_))
print("ARI\t\t\t{:0.4f}".format(adjusted_rand_score(Y_test,km_piczak.labels_)))
print("Silhouette Score\t{:0.4f}".format(silhouette_score(X_test,km_piczak.predict(X_test))))


# ### K-means on PCA and RBM Training Data

# In[19]:


# pca clustering kmeans
km_pca = KMeans(n_clusters=n_clusters,max_iter=300,tol=1e-4,n_jobs=-1,random_state=42,).fit(X_pca_train)
print("K-Means Clustering on PCA Training Data")
print("ARI\t\t\t{:0.4f}".format(adjusted_rand_score(Y_train,km_pca.labels_)))
print("Silhouette Score\t{:0.4f}".format(silhouette_score(X_pca_test,km_pca.predict(X_pca_test))))
# rbm clustering kmeans
km_rbm = KMeans(n_clusters=n_clusters,max_iter=300,tol=1e-4,n_jobs=-1,random_state=42,).fit(X_rbm_train)
print("\nK-Means Clustering on RBM Training Data")
print("ARI\t\t\t{:0.4f}".format(adjusted_rand_score(Y_train,km_rbm.labels_)))
print("Silhouette Score\t{:0.4f}".format(silhouette_score(X_rbm_test,km_rbm.predict(X_rbm_test))))


# In[ ]:




