#!/usr/bin/env python
# coding: utf-8

# Hey! I'm interested in joining a team. I'm currently working as a programmer at the Howard Hughes Medical Institute at Janelia, processing large-scale neural calcium imaging recordings https://www.janelia.org/lab/pachitariu-lab
# I have a master's degree in Applied Mathematics and I've worked as a data scientist in a team focused on big data. I've also done an internship in applying convolutional neural networks on satellite images. My github is here https://github.com/mariakesa If you would be interested in taking me into your team shoot me an email at maria.kesa@gmail.com

# # PCA for visualizing batch effects

# PCA is a linear dimensionality technique that is useful for visualizing and understanding the structure of high-dimensional data. It relies on the eigendecomposition of the covariance matrix of the data points. Here we plot the two principal components corresponding to the dimensions that explain the most variance of the data for different cell types by batches. We use 50 samples from each batch. The data pipeline developed for working with the data will be extended with more methods and functions in subsequent notebooks.

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.decomposition import PCA

from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F

import torchvision
from torchvision import transforms as T

from tqdm import tqdm

import matplotlib.colors as colors
from matplotlib import cm


# In[ ]:


data_path='../input'
csv_path='../'


# In[ ]:


#ImagesDS was taken from https://www.kaggle.com/leighplt/densenet121-pytorch
class ImagesDS(D.Dataset):
    def __init__(self, csv_file, img_dir, mode='train', site=1, channels=[1,2,3,4,5,6]):
        
        df = pd.read_csv(csv_file)
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        
    @staticmethod
    def _load_img_as_tensor(file_name):
        with Image.open(file_name) as img:
            return T.ToTensor()(img)

    def _get_img_path(self, index, channel):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return '/'.join([self.img_dir,self.mode,experiment,f'Plate{plate}',f'{well}_s{self.site}_w{channel}.png'])
        
    def __getitem__(self, index):
        paths = [self._get_img_path(index, ch) for ch in self.channels]
        img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])
        
        if self.mode == 'train':
            return img, self.records[index].sirna
        else:
            return img, self.records[index].id_code

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

class ProcessData():
    def __init__(self, data_path,csv_path):
        self.data_path=data_path
        self.csv_path=csv_path
        self.batch_names={'HEPG2':['HEPG2-01','HEPG2-02','HEPG2-03','HEPG2-04','HEPG2-05','HEPG2-06','HEPG2-07'],
                          'HUVEC':['HUVEC-01','HUVEC-02','HUVEC-03','HUVEC-04','HUVEC-05','HUVEC-06','HUVEC-07',
                                  'HUVEC-08','HUVEC-09','HUVEC-10','HUVEC-11','HUVEC-12','HUVEC-13','HUVEC-14',
                                  'HUVEC-15','HUVEC-16'],
                          'RPE':['RPE-01','RPE-02','RPE-03','RPE-04','RPE-05','RPE-06','RPE-07'],
                          'U2OS':['U2OS-01','U2OS-02','U2OS-03']                            
        }

    def create_random_loader(self, nr_of_samples):
        dataset= ImagesDS(data_path+'/train.csv', self.data_path)
        sampler = SubsetRandomSampler(np.arange(nr_of_samplers, dtype=np.int64))
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=2)
        return loader
    
    def create_loader_for_pca(self,nr_of_samples,cell_type):
        df=pd.read_csv(self.data_path+'/train.csv')
        if cell_type=='HEPG2':
            batches=self.batch_names['HEPG2']
        if cell_type=='HUVEC':
            batches=self.batch_names['HUVEC']
        if cell_type=='RPE':
            batches=self.batch_names['RPE']
        if cell_type=='U2OS':
            batches=self.batch_names['U2OS']
        df_lst=[]
        for batch in batches:
            ind=df['experiment']==batch
            sub=np.array(df[ind].index)
            sub_=list(df[ind].index)
            nr_of_samples_in_batch=len(sub_)
            generate_random_numbers=np.random.randint(0,nr_of_samples_in_batch,nr_of_samples)
            sub=sub[generate_random_numbers]
            df_lst=df_lst+list(sub)
        df_=df.loc[df_lst,:]
        df_.to_csv(csv_path+'train_pca.csv')
        dataset = ImagesDS(csv_path+'train_pca.csv', self.data_path)
        loader = D.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        return loader
        
    def flatten_data_for_PCA(self,cell_type,nr_of_samples):
        loader=self.create_loader_for_pca(nr_of_samples,cell_type)
        arr=torch.zeros((1572864,1))
        for x, y in tqdm(loader):
            #Flatten into 1D vector of features for PCA
            x=x.flatten().view(1572864,1)
            arr=torch.cat((arr,x),dim=1)
        return arr[:,1:].numpy()
    
    def PCA_for_batches(self,cell_type,n_components,nr_of_samples):
        '''
        nr of samples denotes how many samples to take from each batch. 
        '''
        arr=self.flatten_data_for_PCA(cell_type,nr_of_samples)
        pca=PCA(n_components=n_components)
        tr=pca.fit_transform(arr.T)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.title(cell_type)
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()
        labels=[]
        lab_dict={}
        colors_=range(0,len(self.batch_names[cell_type]))
        cmp = cm.get_cmap('viridis',len(self.batch_names[cell_type]))
        for j in range(0,len(self.batch_names[cell_type])):
            labels=labels+[j]*nr_of_samples
            lab_dict[self.batch_names[cell_type][j]]=j
        labels=np.array(labels)
        for g in np.unique(self.batch_names[cell_type]):
            ix=np.where(labels==lab_dict[g])
            plt.scatter(tr[:,0].flatten()[ix], tr[:,1].flatten()[ix], cmap=cmp, label=g)
        plt.legend()
        plt.title(cell_type)
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.show()


# In[ ]:


proc=ProcessData(data_path,csv_path)
proc.PCA_for_batches('HEPG2',100,50)
proc.PCA_for_batches('HUVEC',100,50)
proc.PCA_for_batches('RPE',100,50)
proc.PCA_for_batches('U2OS',100,50)


# In[ ]:




