#!/usr/bin/env python
# coding: utf-8

# <img src="http://phoenix-tec.com/wp-content/uploads/2015/01/Online-Advertising-8.png">

# #                                  Advertisements Images detection -U.C.I

# This dataset represents a set of possible advertisements on Internet pages. 
# <h3>The features encode :-</h3> 
# - the geometry of the image (if available) 
# - phrases occuring in the URL
# - the image's URL and alt text
# - the anchor text, 
# - words occuring near the anchor text
# <h4 style="color:red"> The task is to predict whether an image is an advertisement ("ad") or not ("nonad")<\h4>
# 
# 
# -The aim is to classify based on the given features given the features mentioned
# 
# 
# 

# #### Attribute Information:
# 
# (3 continous; others binary; this is the "STANDARD encoding" mentioned in the [Kushmerick, 99].) 
# 
# One or more of the three continous features are missing in 28% of the instances; missing values should be interpreted as "unknown".
# 
# 

# ### Importing resources and toolkits

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')
import os
os.listdir()


# ### Exploring the dataset

# In[ ]:


df=pd.read_csv('../input/add.csv',low_memory=False,header=0)


# In[ ]:


df=df.drop('Unnamed: 0',axis=1)


# In[ ]:


df.columns=df.columns.astype('int')


# In[ ]:


df.dtypes.head(3)


# 
# - First three colums are important as most of the variation between results are because of first three continuous varibles
# 
# * column1-> height of image
# * column2->widht of image
# * column30->aspect ratio

# #### Information regarding data as general
# 

# In[ ]:


df.info() # data set information


# ####  Information of First three continous variables
# - 28% data is missing for three continous attributes

# In[ ]:


df.iloc[:,0:3].info()


# Since we can see that missing values are there so first we have to replace missing values
# - yellow strips represent the missing data

# In[ ]:


df[0][10]


# In[ ]:


newdf=df.iloc[:,[0,1,2,3]]
newdf=newdf.applymap(lambda x:'?' in x)
plt.figure(figsize=(7,5))
sns.heatmap(newdf,cbar=False,yticklabels=False,cmap='viridis')


# ####  Frequency of missing values

# In[ ]:


for i in (newdf):
    print('column['+str(i)+'] has missing values -'+str(sum(newdf[i])))


# #### Filling missing data with mean of each attribute
# 

# In[ ]:


def replace_missing(df):
    for i in df:
        df[i]=df[i].replace('[?]',np.NAN,regex=True).astype('float')
        df[i]=df[i].fillna(df[i].mean())
    return df


# In[ ]:


df[[0,1,2,3]]=replace_missing(df.iloc[:,[0,1,2,3]].copy()).values


# In[ ]:


df[3]=df[3].apply(lambda x:round(x))


# ## Exporatory  Data Analysis(E.D.A)

# Statsitical approach
# 
# - data is right skewed

# In[ ]:


df[[0,1,2,3]].describe()


# In[ ]:



fig,ax=plt.subplots(nrows=1,ncols=3)
fig.set_figheight(5)
fig.set_figwidth(13)
sns.distplot(df[0],ax=ax[0])
sns.distplot(df[1],ax=ax[1])
sns.distplot(df[2],ax=ax[2])


# Relations between three continous variables

# In[ ]:


sns.pairplot(data=df.iloc[:,[0,1,2,3,1558]])


# > #### *Plotting discrete data

# In[ ]:


fig,ax=plt.subplots(nrows=3,ncols=1)
fig.set_figheight(15)
fig.set_figwidth(18)
sns.stripplot(y=1558,x=0,data=df,ax=ax[0])
sns.stripplot(y=1558,x=1,data=df,ax=ax[1])
sns.stripplot(y=1558,x=2,data=df,ax=ax[2])


# >    How classes are in frequency

# In[ ]:


sns.countplot(data=df,y=1558,palette='husl')


# - how data and central tendency is distributed 
# - Boxplot helps to see the difference in quartiles,mean and the outliers

# In[ ]:


plt.figure(figsize=(15,10))
sns.boxplot(x=1558,y=1,data=df)
plt.xlabel('label-add/non-ad')
plt.ylabel('width')


# 

# #### Feature Engineering

# Encoding last column(class variable)
# * '0' - non advertisement image
# * '1' - addvertisement image

# In[ ]:


df.iloc[:,-1]=df.iloc[:,-1].replace(['ad.','nonad.'],[1,0])


# #### Preparing features for model

# In[ ]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# > ### Scaleing data
# - Since the data is highly varied and hence mikowski mertric would have a large varition of distances given a small change
# - gradient decent will not work effective unless all input attributes are scaled to a same version

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaled=StandardScaler()
x=scaled.fit_transform(x)


# - we can see that EDA becomes more clear after scaling data 

# In[ ]:


sns.pairplot(data=df.iloc[:,[0,1,2,-1]],hue=1558)


# # Model Selection
# - Splitting data 
# - applying models
# - k fold cross vaidations
# - building classification confusion matrix

# In[ ]:


from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.30,random_state=8)


# In[ ]:


import collections
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def fit_models(classifiers,xtrain,ytrain):
    """This function fit multiple models by sklearn and return the dictionary with values as  objects of models"""
    models=collections.OrderedDict()
    for constructor in classifiers:
        obj=constructor()
        obj.fit(xtrain,ytrain)
        models[str(constructor).split(':')[0]]=obj
    return models

def classification_multi_report(ytest,models_array):
    """This function generate classification accuracy report for given input model objects"""
    for i in models_array:
        print('__________________________________________________')
        print('the model - '+str(i))
        print(classification_report(ytest,models_array[i].predict(xtest)))
def cross_Fucntion(models,cv):
    """This function return cross validated accuray and the variance of given input model obejects"""
    accuracy={}
    for model in models:
        cross_val_array=cross_val_score(models[model],xtrain,ytrain,scoring='accuracy',cv=cv)
        accuracy[model]=[np.mean(cross_val_array),np.std(cross_val_array)]
    return accuracy

def multi_grid_search(param_grid_array,estimator_list,x,y):
    """This function calculate the grid search parameters and accuracy  for given input modles and return dictionary with each tupple containing accuracy and best parameters"""
    d={}
    count=0
    for i in estimator_list:
        gc=GridSearchCV(estimator=estimator_list[i],param_grid=param_grid_array[count],scoring ='accuracy',cv=5).fit(x,y)
        d[i]=(gc.best_params_,gc.best_score_)
        count+=1
    return d


# **Fitting the train data **

# In[ ]:


classifiers=[GaussianNB,SVC,KNeighborsClassifier]

model_list=fit_models(classifiers,xtrain,ytrain)

classification_multi_report(ytest,model_list)


# ### applying kfold cross validation to check biase and variance in the model

# In[ ]:


obj=cross_Fucntion(model_list,cv=20)
for model in obj:
    print('the model -'+str(model)+'has \n || crosss validated accuracy as  -> '+str(obj[model][0])+' | variance - '+str(obj[model][1])+' ||' )
    print('______________________________________________________________________________________________________________')


# In[ ]:



param_grid_svm=[
    {
        'kernel':['linear'],'random_state':[0]
    },
     {
        'kernel':['rbf'],'random_state':[0]
     },
    
    {
        'kernel':['poly'],'degree':[1,2,3,4],'random_state':[0]
    }
]

param_grid_knn=[

    {   
        'n_neighbors':np.arange(1,50),
        'p':[2]
        
    }
]

param_grid_nb=[
    {}
]

param_grid_array=[param_grid_nb,param_grid_svm,param_grid_knn]
multi_grid_search(param_grid_array,model_list,xtrain,ytrain)


# Fitting model with best hyperparmater and cv score

# In[ ]:


classifier=SVC(kernel='poly',degree=1,random_state=0)


# In[ ]:


classifier.fit(xtrain,ytrain)


# ## Confusion matrix
# 

# In[ ]:


sns.heatmap(pd.crosstab(ytest,classifier.predict(xtest)),cmap='coolwarm')
plt.xlabel('predicted')
plt.ylabel('actual')


# In[ ]:


print(classification_report(ytest,classifier.predict(xtest)))


# ## Clustering-
# - we applied kmeans clustering to find out clusters based upon IMAGE   AND FEATURES

# In[ ]:


df.head()


# In[ ]:


x=df.iloc[:,:-1].values


# > Most optimal number of clusters 

# In[ ]:


from sklearn.cluster import KMeans
def best_knumber_cluster(x,iter_number):
    wwss=[]
    for i in range(1,iter_number+1):
        kmeans=KMeans(n_clusters=i)
        kmeans.fit(x)
        wwss.append(kmeans.inertia_)
    plt.figure(figsize=(20,10))
    c=plt.plot(np.arange(1,iter_number+1),wwss,marker='o',markersize=10,markerfacecolor='black')
    plt.xlabel('number of clusters')
    plt.ylabel('wwss')
    plt.title('Elbow Curve')
    
    return plt.show()


best_knumber_cluster(x,15)


# 3 clusters are most optimal

# In[ ]:


kmeans=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=100)
kmeans.fit(x)
newdf=df.copy()
newdf['Cluster']=kmeans.predict(x)


# #### we can see that dimetions,aspect ratio  form 3 different cluster for given image

# In[ ]:


fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(kmeans.cluster_centers_[0,0],kmeans.cluster_centers_[0,1],kmeans.cluster_centers_[0,2],c='yellow',marker='x',s=300)
ax.scatter(kmeans.cluster_centers_[1,0],kmeans.cluster_centers_[1,1],kmeans.cluster_centers_[1,2],c='yellow',marker='x',s=300,label='centroid')
ax.scatter(kmeans.cluster_centers_[2,0],kmeans.cluster_centers_[2,1],kmeans.cluster_centers_[2,2],c='yellow',marker='x',s=300)

ax.scatter(newdf[newdf['Cluster']==0][0],newdf[newdf['Cluster']==0][1],newdf[newdf['Cluster']==0][2],c='blue',s=newdf[2]*10,label='Cluster-1')
ax.scatter(newdf[newdf['Cluster']==1][0],newdf[newdf['Cluster']==1][1],newdf[newdf['Cluster']==1][2],c='red',s=newdf[2]*10,label='Cluster-2')
ax.scatter(newdf[newdf['Cluster']==2][0],newdf[newdf['Cluster']==2][1],newdf[newdf['Cluster']==2][2],c='green',s=newdf[2]*10,label='Cluster-3')
ax.legend()
plt.xlabel('height')
plt.ylabel('width')
ax.set_zlabel('aspect ratio')

