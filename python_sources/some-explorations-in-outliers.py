#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# hide warnings
import warnings
warnings.simplefilter('ignore')

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

# Graph
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import axes3d
import plotly.express as px

#Dimensional Reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# From this competition, I learned that CNN is extremely powerful technique to classify MNIST datasets. I guess someone who achieved above 98% definitely used CNN. I think more than 95% accuracy means model is well generalizing the situation but does not discriminate outliers(looking really strange or should not be there).
# To check my hypothsis, I look into what letters are misclassified and how it looks like.

# In[ ]:


train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')


# In[ ]:


#Original Label
train_label=train['label'].copy()
train_label=train_label.values

#Original
train_data = train.loc[:,'pixel0':].values
train_data = train_data/255.0
train_data=train_data.reshape(-1,28,28,1)


# I cited CNN model from [here](https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist) which performs above 99.7%, and I reduced 15  to 3 with 3 epochs since my laptop is not bearable to computing all, but it will achieve above 99% accuracy on the trainset! This will be enough to detect the outliers.

# In[ ]:


# BUILD CONVOLUTIONAL NEURAL NETWORKS
nets = 3
model = [0] *nets
for j in range(nets):
    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(128, kernel_size = 4, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Flatten())
    model[j].add(Dropout(0.4))
    model[j].add(Dense(10, activation='softmax'))

    #Compile
    model[j].compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


# TRAIN NETWORKS
history = [0] * nets
epochs = 3
for j in range(nets):
    history[j] = model[j].fit(train_data,train_label,
        epochs = epochs, batch_size = 64,  
         verbose=0)
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}".format(
        j+1,epochs,max(history[j].history['accuracy'])))


# In[ ]:


# ENSEMBLE PREDICTIONS
results = np.zeros( (train_data.shape[0],10) ) 
result=[0]*nets
for j in range(nets):
    output = model[j].predict(train_data)
    results = results + output
    result[j] = output


# In[ ]:


def labelConcat(results):
    choice = np.argmax(results,axis=1)
    choice = pd.DataFrame({'predict_label':choice})
    labels = pd.DataFrame({'true_label':train['label']})
    labels = pd.concat([labels,choice], axis=1)
    return labels
labels = labelConcat(results)


# In[ ]:


for i in len(result):
    
    if i ==0:
        choice = result.copy()
        choice = np.argmax(choice,axis=1)
        labels = pd.DataFrame({'label':train['label']})
        choice = pd.DataFrame({'predict_label':choice})
        labels = pd.concat([labels,choice], axis=1)
    else:
        
        choice = [choice,labelConcat(result[i])]
    
    return choice


# Took a long time to compute...
# Let's check the letters which deceive 10 stacked CNN.

# I aggregated all discrepancies between true label and predicted label from the models and visualized it.
# Accroding to the chart, some letters are confused with some specific letters or each other. 
# 
# 0, 1, and 2 are not distinguishable. 3, 6 and 7 are not distinguishable. We will see how these looks like.

# In[ ]:


def unmatchPlot(labels):
    idx = labels['true_label']!=labels['predict_label']
    unmatch = labels[idx]
    fig = sns.swarmplot(x="true_label", y="predict_label", hue="predict_label",
             alpha=.5
            , data=unmatch,size=7)
    plt.ylim(-1, 9)
    plt.legend(bbox_to_anchor=(1, 0.95))
    
    
unmatchPlot(labels)


# In[ ]:


def UnmatchImages(labels,imageset,num):
    
    if num>10:
        
        print('out of bounds!')
        
    else:
        
        img = [0]*num
        pred = [0]*num
        for i in range(num):
            idx = (labels['true_label']==i)&(labels['true_label']!=labels['predict_label'])
            img[i] = imageset.loc[idx]
            img[i] = img[i].loc[:,'pixel0':].values
            img[i]=img[i]#.reshape(-1,28,28,1)
            pred[i] = labels['predict_label'].loc[idx].values
            
    return img , pred


def Randomdiplayoutliers(Images,pred):
    
    width=10
    height=4
    
    fig, ax = plt.subplots(height
                           ,width
                           ,figsize=(10,5)
                           ,subplot_kw = {'xticks':[], 'yticks':[]})
    
    for i , ax in enumerate(ax.flat):
        
        num = i%width
        a = np.arange(len(Images[num]))
        np.random.shuffle(a)
        

        if len(a[:1])!=0:
            ax.imshow(Images[num][a[:1]].reshape((28,28))
                              ,cmap='binary'
                              ,clim = (0,16))
            ax.title.set_text('predict%d' % pred[num][a[:1]])
            Images[num] = Images[num][a[1:]]
            
        else:
            ax.title.set_text('DNE')
            ax.imshow(np.zeros((28,28))
                              ,cmap='binary'
                              ,clim = (0,16))
            
Images, prediction = UnmatchImages(labels,train,10)
Randomdiplayoutliers(Images,prediction)


# Please check that 0,1,2 group and 3,6,7 group. each groups are similar in it. 9 to 6 also. 
# It seems difficult to clarify if it is truly labeled in my opinion.

# In[ ]:


def labelExtract(datasets,labels,compare,comp_num,method):
    
    dataset = datasets.copy()
    
    k=0
    idx=[]
    idx1=[]
    idx2=[]
    for i in compare:
        
        if k==0:
            idx = labels['true_label'] == i
            dataset['label'][idx] = 'true_label{}'.format(i)
            idx1 = labels['true_label']!=labels['predict_label']
            dataset['label'][idx1] = 'outlier{}'.format(i)
        else:
            idx2 = labels['true_label'] == i
            dataset['label'][idx2] = 'label{}'.format(i)   
        k = k+1
    
    if len(idx2)!=0:
        idx3 = (idx2 | idx)
    else:
        idx3 = idx
        
    dataset = dataset.loc[idx3]
    dataset_val = dataset.loc[:,'pixel0':].values
    
    
    if method=='PCA':
        
        comp = ['PC1','PC2','PC3']
        
        scaler = StandardScaler()
        scaler.fit(dataset_val)
        dataset_val = StandardScaler().fit_transform(dataset_val)
        pca = PCA(n_components=comp_num)
        dataset_PC = pca.fit_transform(dataset_val)
        
        if comp_num ==2:
        
            dataset_PC = pd.DataFrame(data = dataset_PC
                                  , columns = comp[:2])
            dataset_PC['label'] = dataset['label'].values
            
        elif comp_num == 3:
            
            dataset_PC = pd.DataFrame(data = dataset_PC
                                  , columns = comp)
            dataset_PC['label'] = dataset['label'].values
        
        dataset = dataset_PC
            
    elif method == 'tSNE':
        
        comp = ['Axis1','Axis2','Axis3']
        
        tsne = TSNE(random_state = 42, n_components=comp_num
                , verbose=0, perplexity=40
                , n_iter=300).fit_transform(dataset_val)
        
            
        if comp_num == 2:
            
            dataset_tSNE = pd.DataFrame(data = tsne
                                  , columns = comp[:2])
            
            dataset_tSNE['label'] = dataset['label'].values
        
        elif comp_num == 3:
            
            dataset_tSNE = pd.DataFrame(data = tsne
                                  , columns = comp)
        
            dataset_tSNE['label'] = dataset['label'].values
            
        dataset = dataset_tSNE
            
    
    return dataset

def labelPCAplot(datasets):
    
    dim = len(datasets.T)-1
    
    if dim == 2:
        
        fig=px.scatter(datasets
                       , x ='PC1'
                       ,y ='PC2'
                       ,color='label')
        fig.show()
        
    
    elif dim == 3:
        
        
        fig = px.scatter_3d(datasets
                            , x='PC1'
                            , y='PC2'
                            , z='PC3'
                            ,color="label")
        
        fig.show()
        
    else:
        
        print('>3D dimensional plot???')

def labeltSNEplot(datasets):
    
    dim = len(datasets.T)-1
    
    if dim == 2:
        
        
        fig=px.scatter(datasets
                       ,x ='Axis1'
                       ,y ='Axis2'
                       ,color='label')
        fig.show()
        
    elif dim == 3:
        
        
        fig = px.scatter_3d(datasets
                            , x='Axis1'
                            , y='Axis2'
                            , z='Axis3'
                            ,color="label")
        fig.show()
        
    else:
        print('>3D dimensional plot???')
    


# In[ ]:


Reduced = labelExtract(train,labels,[9],3,'PCA')
labelPCAplot(Reduced)


# Some dimensional reduction techniques are good to check similarities among classes, so I will use it to compare them with the outliers detected.
# 
# PCA is the one of classical and famous dimensonal reduction techniques
# Intuitively, It finds new coordinates(basis) in which every points behave independently in each axis. and principal components are usually choosen from some axis in which points are broadely disributed in that axis.
# I draw 2D,3D scatter plots about 2 and 3 principal components using plotly and see where these outliers places in the cluster , and results were not so much distinctive as opposed to what I expected at first. One of the reasons might be it losses a lot of information from original one since it reduces the dimension from 28*28(784) dim to 2 or 3 dim. 
# You might check the proportion of 2 or 3 principal components from the total(It is not big enough). 
# 
# However, here something intresting. One can see that the outliers are clustered in some area, so this implies that outliers are simliar to each other. From this, I have suspicion that the area where outliers clustered might be overlapped with another cluster. 
# 
# Let's check it

# In[ ]:


Reduced = labelExtract(train,labels,[6,9],3,'PCA')
labelPCAplot(Reduced)


# In[ ]:


Reduced = labelExtract(train,labels,[3],2,'PCA')
labelPCAplot(Reduced)


# In[ ]:


Reduced = labelExtract(train,labels,[7,3],3,'PCA')
labelPCAplot(Reduced)


# In[ ]:


Reduced = labelExtract(train,labels,[7,3],2,'PCA')
labelPCAplot(Reduced)


# In[ ]:


Reduced = labelExtract(train,labels,[2,0],2,'PCA')
labelPCAplot(Reduced)


# In[ ]:


Reduced = labelExtract(train,labels,[2,0],3,'PCA')
labelPCAplot(Reduced)


# In[ ]:


Reduced = labelExtract(train,labels,[2],2,'PCA')
labelPCAplot(Reduced)


# Outliers from 6 are placed middle of two clusters as you can check

# In[ ]:


Reduced = labelExtract(train,labels,[9],2,'tSNE')
labeltSNEplot(Reduced)


# This is tSNE. I do not know this technique well, but it seems to try to reduce Kullerback-Leiber divergence between target dimension(reduced dimension) and original dimension by converting its metric distance to probability-looking formula. if these metric distance is similar, then KL-divergence also reduced. Am I right? :)

# In[ ]:


Reduced = labelExtract(train,labels,[6,9],2,'tSNE')
labeltSNEplot(Reduced)


# In[ ]:


Reduced = labelExtract(train,labels,[3],2,'tSNE')
labeltSNEplot(Reduced)


# In[ ]:


Reduced = labelExtract(train,labels,[7,3],2,'tSNE')
labeltSNEplot(Reduced)


# In[ ]:


Reduced = labelExtract(train,labels,[7,3],3,'tSNE')
labeltSNEplot(Reduced)


# In[ ]:


Reduced = labelExtract(train,labels,[3],3,'tSNE')
labeltSNEplot(Reduced)


# In[ ]:


Reduced = labelExtract(train,labels,[2,0],2,'tSNE')
labeltSNEplot(Reduced)


# In[ ]:


Reduced = labelExtract(train,labels,[2],2,'tSNE')
labeltSNEplot(Reduced)


# There are more interested things I want to check, but I do not have much time :)
# For examplem, I want to see average distributions drawn from the actual output(softmax values) and would like to check these outliers equally or unequally distributed and so on.
