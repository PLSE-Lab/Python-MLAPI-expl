#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Fraud detection is a common application of machine learning. There is a reason why the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) is the most popular dataset on kaggle.
# 
# But most of the methods addressing the problem of fraud detection are methods of supervised learning, where you have information about which data belongs to a fraud.<br>
# **But what if you don't have this information?**<br>
# 
# In this notebook I am going to present a method of outlier detection, which can be put to good use in the context of fraud detection.
# 
# *By the way: This is my first notebook, so some feedback would be greatly appreciated.
# *

# # Robust outlier detection using FSRMCD-MAC
# In this notebook I want to demonstrate the FSRMCD-MAC algorithm introduced in [Jobe, Pokojovy (2015)](https://www.researchgate.net/publication/268817944_A_Cluster-Based_Outlier_Detection_Scheme_for_Multivariate_Data). It is a cluster-based approach to outlier detection and is based on three main steps, which I will explain later on.<br>
# FSRMCD-MAC stands for **finite sample reweighted minimum covariance determinant mode association clustering**. So I will keep it short and refer to it as the FSRMCD-MAC algorithm.
# 
# As data we are going to use the [Swiss bank note dataset](https://www.kaggle.com/chrizzles/swiss-banknote-conterfeit-detection), which was also used in the original paper of the algorithm. It contains information on 200 swiss banknotes and wether they are a genuine or a conterfeit banknote.
# 
# ### What are we going to do?
# 1. **Introduction**: If you're reading this, we're already past that point.
# 2. **Short data exploration**: This notebook is not about data analysis, but we will still have a short look at the data.
# 3. **The FSRMCD-MAC algorithm**: Short overview over the actual algorithm, as simple as possible.
# 4. **Comparing different methods for Outlier detection**: Comparing the FSRMCD-MAC algorithm with other outlier detection methods.
# 5. **Conclusion**
# 6. **Further Information**: Here you can find the relevant papers.

# ## Import packages

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet
import scipy as sc
import plotly.graph_objects as go
from scipy.spatial.distance import mahalanobis, euclidean
from scipy.stats import chi2
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import scikitplot as skplt


# ## Import data

# In[ ]:


data_raw=pd.read_csv("/kaggle/input/swiss-banknote-conterfeit-detection/banknotes.csv")


# # Short data exploration
# Before we get started, we should have look at the data.  
# The data consists of the following Attributes:
# * **conterfeit**: Wether a banknote is conterfeit (1) or genuine (0)
# * **Length**: Length of bill (mm)
# * **Left**: Width of left edge (mm)
# * **Right**: Width of right edge (mm)
# * **Bottom**: Bottom margin width (mm)
# * **Top**: Top margin width (mm)
# * **Diagonal**: Length of diagonal (mm)
# 
# From the datasets description we know, that the feature 'conterfeit' is evenly distributed, **there are 100 genuine and 100 fake banknotes.**

# In[ ]:


data_raw.head()


# ### Drop label
# Since we will be using unsupervised learning, the label is now dropped and only used in the end for evaluation.

# In[ ]:


#Drop conterfeit information
data=data_raw.drop(["conterfeit"],axis=1).copy()


# ## Density plot
# We want to know what the distribution for each feature looks like. Is it normal distributed or not? Is it unimodal or multimodal?

# In[ ]:


f, ax = plt.subplots(2,3,figsize=(18, 8))
columns=data.columns.values
for index,name in enumerate(columns):
    x,y=(0,index) if index<3 else (1,len(columns)-index-2)
    sns.distplot(data[name],ax=ax[x][y])


# As you can see in the plot, most of the features seem to form some kind of normal distribution.
# Especially the features '**Length**' and '**Top**' have a really nice looking curve.
# The Features '**Bottom**' and '**Diagonal**' on the other hand, seem to come from some kind of multimodal distribution, since they have two peeks (especially 'Diagonal').

# ## Correlation heatmap
# We now want to take a look at the correlation of the different features. This might indicate wether some features have redundant information.

# In[ ]:


f, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(data.corr())
plt.show()


# The highest correlation is between the features 'left' and 'right', which makes sense, since both edges of a banknote should be somewhat equally sized.  The feature 'Diagonal' has a negative correlation with all features except 'length'. 

# # The FSRMCD-MAC Algorithm
# ## Basic idea
# I will try to explain the basic steps of the algorithm as simple as possible.<br>
# **Assumptions:**<br>
# The data is assumed to come from a finite mixture of multivariate normal densities, with the biggest component representing the main cluster of datapoints. All datapoints which are outside of this cluster are treated as potential outliers.<br>
# **Aim:**<br>
# The aim of the algorithm is to estimate the multivariate probability density function of the data. With this estimation different clusters can be identified and the largest cluster is extracted. The data from the largest cluster is then used to estimate the mean and the covariance of the data. All data which are "too far away" from the mean are now identified as outliers.
# 
# ### Steps of the algorithm
# The algorithm consists of three main steps, which I will describe a little further. <br>
# 1. **FSRMCD-step**<br>
# In this step robust estimators for the mean and covariance of the data are created. Why do we need robust estimators?<br> A robust method in a statistical sense, is a method that is less affected by outliers than "normal" methods.<br> So while the "usual" estimators of mean and covariance would be hugely distorted, due to the outliers in the dataset, the effect is not as big using the robust estimators.<br>
# The estimator used in this algorithm is the FSRMCD estimator. You can find more information about it at the end of the notebook.
# 2. **MAC-step**<br>
# The robust estimators are used to estimate the probability density function of the data using KDE (Kernel density estimation). This function is now maximized for each data point. All data points which correspond to the same maximum are put in the same cluster. This process is called modal association clustering (MAC). From these clusters, the largest one is extracted and its mean and covariance are calculated.
# 3. **Outlier detection step**<br>
# Using the mean and covariance from the largest cluster a distance measure, the mahalanobis distance, is calculated for each datapoint. All datapoints with a mahalanobis distance exceeding a specified level are now identified as outliers.
# 

# ## Implementation

# In[ ]:


def get_mcd_outliers(data,threshold,type="outlier"):
    #MCD Estimators and Mahalanobis distances of data
    mcd_estimator=MinCovDet().fit(data)
    mahalanobis=mcd_estimator.mahalanobis(data)
    #Calculate outliers based on threshold
    if type=="outlier":
        transformation = lambda x: 0 if x <= threshold else 1
    elif type=="weights":
        transformation = lambda x: 1 if x <= threshold else 0
    outliers = np.array([transformation(xi) for xi in mahalanobis])
    return outliers

def get_fsrmcd_estimators(data,threshold,n,p,type="estimator"):
    weights=get_mcd_outliers(data,threshold,"weights")
    #FSRMD Location
    fsrmcd_location = np.matrix(np.average(data, axis=0, weights=weights))

    #FSRMCD Covariance
    fsrmcd_covariance = 0
    for x in range(0, n):
        fsrmcd_covariance = fsrmcd_covariance + weights[x] * np.dot(
            np.transpose((data.loc[x].values - fsrmcd_location)),
            (data.loc[x].values - fsrmcd_location))

    k_constant=(np.floor(0.975*n)/n)/(chi2.cdf(threshold,p+2))

    fsrmcd_covariance=(k_constant/(sum(weights)-1))*fsrmcd_covariance
    #Returns estimators
    if type=="estimator":
        return (fsrmcd_location,fsrmcd_covariance)
    
    #Returns outliers for FSRMCD method only
    elif type=="outlier":
        #Calculate mahalanobis distances and determine the outliers
        mahalanobis_fsrmcd=np.array([])
        for x in range(0,n):    
            maha=np.power(sc.spatial.distance.mahalanobis(data.loc[x].values,fsrmcd_location,np.linalg.inv(fsrmcd_covariance)),2)
            mahalanobis_fsrmcd=np.append(mahalanobis_fsrmcd,maha) 
        transformation = lambda x: 0 if x <= threshold else 1
        outliers = np.array([transformation(xi) for xi in mahalanobis_fsrmcd])
        return outliers


def get_fsrmcdmac_outliers(data):
    #Data shape
    n=len(data)
    p=data.shape[1]
    
    #Threshold Chi-Square Distribution
    threshold = chi2.ppf(0.975, p)
    
    #Get FSRMCD Estimators
    fsrmcd_location,fsrmcd_covariance=get_fsrmcd_estimators(data,threshold,n,p)

    #Calculate bandwidth matrix
    #Split matrix in pre factor hb and matrix H
    hb_factor=np.power((4/(4+p)),(1/(p+6)))*np.power(n,(-1/(p+6)))
    #Factor is different for different values for p, see the paper for details
    cnp_factor=0.7532+37.51*np.power(n,-0.9834)
    H=sqrtm(cnp_factor*fsrmcd_covariance)
    H_inverse=np.linalg.inv(H)

    #Copy and standardize data
    data_standardized = data.copy()
    for x in range(0, n):
        data_standardized.loc[x] = np.reshape(
            np.dot(H_inverse, np.transpose((data_standardized.loc[x].values - fsrmcd_location))),p)

    #Mode Association Clustering Algorithm
    modes=np.zeros((p,n))
    for x in range(0,n):

        #Select each datapoint as starting point from where to find the local maximum
        x0=data_standardized.loc[x].values
        x0_old=x0+1

        #Define stopping-criteria
        cnt=0
        err=100

        #Iterative algorithm to find the maximum
        while ((err>0.0001) and (cnt<150)):
            diag=np.zeros((p,p),int)
            np.fill_diagonal(diag,1)
            kde=multivariate_normal.pdf(data_standardized,x0,hb_factor*diag)

            d=kde/sum(kde)
            x0=np.dot(np.transpose(d),data_standardized)

            err=np.linalg.norm(x0-x0_old,ord=2)/np.maximum(np.linalg.norm(x0,ord=2),1)

            x0_old=x0
            cnt=cnt+1

        modes[:,x]=x0

    #Put different modes into different clusters
    clusters=np.zeros((1,n))
    clust_cnt=0

    err=1/(2*n)

    for x in range(0,n):
        if clusters[:,x]==0:
            clust_cnt=clust_cnt+1
            clusters[:,x]=clust_cnt

            for y in range(0,n):
                if clusters[:,y]==0:
                    if np.linalg.norm((modes[:,x]-modes[:,y]),ord=2)<err:
                        clusters[:,y]=clust_cnt

    #Get largest cluster and corresponding mode
    clust_max=-1
    clust_s=0
    for x in range(1,clust_cnt+1):
        s=len(clusters[clusters==x])

        if s>clust_s:
            clust_max=x
            clust_s=s

            ind=min(clusters[clusters==x])
            mode=modes[:,int(ind)]

    bulk=np.where(clusters==clust_max)[1]

    #Get mode by reverting the standardization
    mode=np.dot(H,mode)+fsrmcd_location

    #Save final Cluster
    if len(data.loc[bulk])<(p+1):
        cluster_final=data
    else:
        cluster_final=data.loc[bulk]
        cluster_final.reset_index(inplace=True)
        cluster_final.drop("index",axis=1,inplace=True)

    #Get estimators from cluster
    mean_cluster = cluster_final.mean().values
    covariance_cluster = cluster_final.cov().values
    weights=np.array([])

    #Get Mahalanobis distance and outliers
    for x in range(0, len(cluster_final)):
        maha = np.power(sc.spatial.distance.mahalanobis(cluster_final.loc[x].values, mean_cluster,
                                               np.linalg.inv(covariance_cluster)),2)
        if maha <= threshold:
            weights=np.append(weights,1)
        else:
            weights=np.append(weights,0)


    #Get final robust estimators
    #Location
    robust_location = np.matrix(np.average(cluster_final, axis=0, weights=weights))

    #Covariance
    robust_covariance = 0
    for x in range(0, len(cluster_final)):
        robust_covariance = robust_covariance + weights[x] * np.dot(
            np.transpose((data.loc[x].values - robust_location)),
            (data.loc[x].values - robust_location))

    robust_covariance=(1/(sum(weights)-1))*robust_covariance

    #Calculate final mahalanobis distances and determine the final outliers
    mahalanobis_robust=np.array([])
    for x in range(0,n):    
        maha=np.power(sc.spatial.distance.mahalanobis(data.loc[x].values,robust_location,np.linalg.inv(robust_covariance)),2)
        mahalanobis_robust=np.append(mahalanobis_robust,maha)    

    #Outlier thresholds
    #L1 and L2 depend on the dimension of the dataset. See the paper for values for different dimensions
    L1=31.9250
    L2=16.9710
    if mahalanobis_robust.max()<L1:
        outliers=np.repeat(0,200)
    else:
        outliers=np.array([])
        for x in range(0,n):
            if mahalanobis_robust[x]>L2:
                outliers=np.append(outliers,1)
            else:
                outliers=np.append(outliers,0)
    return outliers


# ## Results and confusion matrix
# How did the method perform?

# In[ ]:


results=data_raw.copy()
results["Outlier_FSRMCDMAC"]=get_fsrmcdmac_outliers(data)
results["Outlier_FSRMCDMAC"]=results["Outlier_FSRMCDMAC"].astype(int)
print("Accuracy: "+str(accuracy_score(results["conterfeit"],results["Outlier_FSRMCDMAC"])))
skplt.metrics.plot_confusion_matrix(results["conterfeit"],results["Outlier_FSRMCDMAC"],normalize="true")
plt.show()


# As you can see, the FSRMCD-MAC algorithm was able to detect all fake banknotes and almost all of the genuine banknotes, without having any information about which is which.<br> The overall accuracy is **98%**, which is pretty amazing.<br>
# But to see if this is actually a special result, we should compare it to some other outlier detection methods.

# # Comparing different methods for Outlier detection

# Now we want to compare the FSRMCD-MAC algorithm to different methods for outlier detection.  I've decided to include three methods as comparison.
# * **Isolation Forest**: Density based outlier detection method implemented in sklearn.
# * **Local Outlier Factor**: Outlier detection based on random forests, implemented in sklearn.
# * **MCD-Method**: Robust covariance estimation using MCD and outlier detection using Mahalanobis distance measure, available in sklearn.
# 
# ### What makes a good method?
# Since we are looking at the case of fraud detection, it is more important to detect all frauds, than all non-frauds. Therefore a good method detects as many conterfeits as possible. Since we defined conterfeit as 1 and genuine banknote as 0, this is the same as having a **high TPR (True positive rate) or a high sensitivity.**<br>
# The TNR (True negative rate) shouldn't be ignored though, we don't want to accuse too many innocent people of faking a banknote.

# In[ ]:


#Isolation Forest
isf=IsolationForest()
isf_outliers=isf.fit_predict(data)
results["Outlier_ISF"]=isf_outliers
results["Outlier_ISF"]=results["Outlier_ISF"].map({1:0,-1:1})

#Local Outlier Factor
lof=make_pipeline(StandardScaler(),LocalOutlierFactor())
lof_outliers=lof.fit_predict(data)
results["Outlier_LOF"]=lof_outliers
results["Outlier_LOF"]=results["Outlier_LOF"].map({1:0,-1:1})

#Mahalanobis Distance with MCD Estimators
p=data.shape[1]
threshold = chi2.ppf(0.975, p)
mcd_outliers = get_mcd_outliers(data,threshold)
results["Outlier_MCD"]=mcd_outliers

#Compare different methods
methods=["Outlier_ISF","Outlier_LOF","Outlier_MCD","Outlier_FSRMCDMAC"]
comparison=pd.DataFrame(columns=["TN","FP","FN","TP"])
for method in methods:
    comparison.loc[method]=confusion_matrix(results["conterfeit"],results[method]).reshape(4)
comparison["Accuracy"]=(comparison["TP"]+comparison["TN"])/len(data)
comparison["Sensitivity"]=(comparison["TP"])/(len(data)/2)
comparison["Specificity"]=(comparison["TN"])/(len(data)/2)
comparison.sort_values(by="Sensitivity",inplace=True)
comparison


# We can already see how the different methods compare, but lets look at a visualization to make it a bit easier.

# In[ ]:


methods=["Local Outlier Factor","MCD","Isolation Forest","FSRMCD-MAC"]
fig=go.Figure()
fig.add_trace(go.Bar(y=methods,x=comparison["TN"]/100,orientation="h",name="Correctly identified genuine banknotes (TNR) (%)"))
fig.add_trace(go.Bar(y=methods,x=comparison["TP"]/100,orientation="h",name="Correctly identified fake banknotes (TPR) (%)"))


# ## FSRMCD-MAC beats them all
# As you can see in the plot, none of the other methods came close in detecting the fake banknotes.<br>
# While the true negative rate is somewhat similar for all methods, **the FSRMCD-MAC algorithm detected 5 times more conterfeits than all the other methods.**
# 
# 

# # Conlusion
# The FSRMCD-MAC algorithm is a powerful method for outlier detection and even works if half of the data is made of outliers. It performed way better than other comparable methods.
# 
# You should be a bit careful though, since the dataset is from the original paper, it is possible that the algorithm performs especially good on it.<br>
# What about data which is more distorted, categorical data, or data which is clearly not normal distributed?<br>
# **Feel free to copy this notebook and try out the algorithm on different datasets.**
# 
# I hope you liked the notebook and maybe you can use it for your next Data Science project.<br>
# **Since this was my first notebook, feedback would be greatly appreciated.**

# # Further information
# If you want to obtain further information on this algorithm and the ones its based on, here are the main resources.
# * **Jobe, J. M. und Pokojovy, M. (2015). A cluster-based outlier detection scheme for multivariate data.**
# * **Li, J., Ray, S., und Lindsay, B. (2007). A nonparametric statistical approach to clustering via mode identification.**
# * **Cerioli, A. (2010). Multivariate outlier detection with high-breakdown estimators.**
# * **Rousseeuw, P. und Driessen, K. (1999). A fast algorithm for the minimum covariance determinant estimator.**
# 

# In[ ]:




