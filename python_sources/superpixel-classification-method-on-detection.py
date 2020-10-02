# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 18:19:44 2019

@author: zhuyihao
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os  
import h5py
from skimage.measure import regionprops
import warnings
from warnings import warn
import _pickle as cPickle
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from sklearn.cluster import KMeans
import pickle
def make_sp_seg(seed_val):
    np.random.seed(seed_val)
    return slic(petct_vol, 
                  n_segments = 800, 
                  compactness = 0.1,
                 multichannel = True)
def scalar_attributes_list(im_props):
    """
    Makes list of all scalar, non-dunder, non-hidden
    attributes of skimage.measure.regionprops object
    """

    attributes_list = []

    for i, test_attribute in enumerate(dir(im_props[0])):

        # Attribute should not start with _ and cannot return an array
        # does not yet return tuples
        try:
            if test_attribute[:1] != '_' and not \
                    isinstance(getattr(im_props[0], test_attribute), np.ndarray):
                attributes_list += [test_attribute]
        except Exception as e:
            warn("Not implemented: {} - {}".format(test_attribute, e), RuntimeWarning)

    return attributes_list


def regionprops_to_df(im_props):
    """
    Read content of all attributes for every item in a list
    output by skimage.measure.regionprops
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        attributes_list = scalar_attributes_list(im_props)

    # Initialise list of lists for parsed data
    parsed_data = []
    # Put data from im_props into list of lists
    for i, _ in enumerate(im_props):
        parsed_data += [[]]

        for j in range(len(attributes_list)):
            parsed_data[i] += [getattr(im_props[i], attributes_list[j])]

    # Return as a Pandas DataFrame
    return pd.DataFrame(parsed_data, columns=attributes_list)

def select(label):
    idx=[]
    for i in range(label.shape[0]):
        if np.sum(label[i])>0:
            idx.append(i)
    idx=np.array(idx)
    return idx       

make_proj = lambda x: np.sum(x,1)[::-1]
make_mip = lambda x: np.max(x,1)[::-1]
CATE = "../input/dataset"
print(os.listdir(CATE)) 
#%%Established the Random Forest Model
choose=6
with h5py.File(os.path.join(CATE,'lab_petct_vox_5.00mm.h5'),'r') as p_data:
    aa = list(p_data['ct_data'].keys())
    aa[choose],aa[6] = aa[6],aa[choose]
    x_data = pd.DataFrame()
    y_data = pd.DataFrame()
    for num in aa[0:6]:
        ct_image = p_data['ct_data'][num].value
        pet_image = p_data['pet_data'][num].value
        label_image = (p_data['label_data'][num].value>0).astype(np.uint8)
        idx=select(label_image)
        idx_min=max(idx[0]-30,0)
        idx_max=min(idx[-1]+30,label_image.shape[0])
        ct_proj = make_proj(ct_image[idx_min:idx_max])
#        scipy.misc.imsave("%s/%d.png"%(outpath,index),ct_proj)
        suv_max = make_mip(pet_image[idx_min:idx_max])
#        scipy.misc.imsave("%s/%d.png"%(outpath,index+7),suv_max)
        lab_proj = make_proj(label_image[idx_min:idx_max])

        pet_weight = 5.0 # how strongly to weight the pet_signal (1.0 is the same as CT)
        petct_vol = np.stack([np.stack([(ct_slice+-200).clip(0,2048)/2048, 
                            pet_weight*(suv_slice).clip(0,5)/5.0
                           ],-1) for ct_slice, suv_slice in zip(ct_proj, suv_max)],0)

        out_df_list =[]
        for seeds in range(5):
            t_petct_segs = make_sp_seg(seeds)
            REG = regionprops(t_petct_segs, intensity_image=suv_max)
            REG_df = regionprops_to_df(REG)
            # add a malignancy score
            REG_df['malignancy'] = REG_df['label'].map(lambda sp_idx: np.mean(lab_proj[t_petct_segs==sp_idx]))
            # add the mean CT value
            REG_df['meanCT'] = REG_df['label'].map(lambda sp_idx: np.mean(petct_vol[:,:,0][t_petct_segs==sp_idx]))
            out_df_list += [REG_df]
            REG_df = pd.concat(out_df_list)

        reg_var = 'malignancy'
        reg_var1='label'
        reg_var2='mean_intensity'
# boost the malignancy count by 1e3
        boost_df = REG_df.sample(10000, weights=(1e10+REG_df[reg_var].values), replace = True)
        #boost_df = sp_rprop_df.sample(10000, replace = True)
# break into variables and outcomes
        numeric_df = boost_df.select_dtypes(include=[np.number])
        pp1 = numeric_df[[ccol for ccol in numeric_df.columns if ccol not in [reg_var]]]
        pp1 = pp1[[ccol for ccol in pp1.columns if ccol not in [reg_var1]]]
        pp2 = pd.DataFrame(boost_df[reg_var].values)
        x_data = pd.concat([x_data,pp1],axis=0)
        y_data = pd.concat([y_data,pp2],axis=0)
from sklearn.tree import DecisionTreeRegressor
malig_tree = DecisionTreeRegressor()
malig_tree.fit(x_data, y_data)

with h5py.File(os.path.join(CATE,'lab_petct_vox_5.00mm.h5'),'r') as p_data:
    print(list(p_data.keys()))
    #% Get the number of patient
    num_patient = len(list(p_data['label_data'].keys()))
    a = list(p_data['ct_data'].keys())
    index = -1
    for num in a[choose:choose+1]:
        
        ct_image = p_data['ct_data'][num].value
        pet_image = p_data['pet_data'][num].value
        label_image = (p_data['label_data'][num].value>0).astype(np.uint8)
        index = index + 1
        ct_proj = make_proj(ct_image)
#        scipy.misc.imsave("%s/%d.png"%(outpath,index),ct_proj)
        suv_max = make_mip(pet_image)
#        scipy.misc.imsave("%s/%d.png"%(outpath,index+7),suv_max)
        lab_proj = make_proj(label_image)
#        scipy.misc.imsave("%s/%d.png"%(outpath,index+14),lab_proj)
        #plt.imsave('D:\Rensselaer\Bioimage\project/data/'+str(index)+'.png',lab_proj)


        pet_weight = 5.0 # how strongly to weight the pet_signal (1.0 is the same as CT)
        petct_vol = np.stack([np.stack([(ct_slice+-200).clip(0,2048)/2048, 
                            pet_weight*(suv_slice).clip(0,5)/5.0
                           ],-1) for ct_slice, suv_slice in zip(ct_proj, suv_max)],0)

        out_df_list =[]
        for seeds in range(1):
            t_petct_segs = make_sp_seg(seeds)
            REG = regionprops(t_petct_segs, intensity_image=suv_max)
            REG_df = regionprops_to_df(REG)
            # add a malignancy score
            REG_df['malignancy'] = REG_df['label'].map(lambda sp_idx: np.mean(lab_proj[t_petct_segs==sp_idx]))
            # add the mean CT value
            REG_df['meanCT'] = REG_df['label'].map(lambda sp_idx: np.mean(petct_vol[:,:,0][t_petct_segs==sp_idx]))
            out_df_list += [REG_df]
            REG_df = pd.concat(out_df_list)
         
        reg_var = 'malignancy'
        reg_var1='label'
        reg_var2='mean_intensity'
# boost the malignancy count by 1e3
        #boost_df = sp_rprop_df.sample(3000, weights=(1e-3+sp_rprop_df[reg_var].values), replace = True)
        boost_df = REG_df.sample(REG_df.shape[0], replace = False)
        
     
        numeric_df = boost_df.select_dtypes(include=[np.number])
        p1 = numeric_df[[ccol for ccol in numeric_df.columns if ccol not in [reg_var]]]
        p1 = p1[[ccol for ccol in p1.columns if ccol not in [reg_var1]]]
        p2 = pd.DataFrame(boost_df[reg_var].values)
#        file = open('x_test.txt','wb')
#        cPickle.dump(p1,file)
#        file.close()
#        file1 = open('y_test.txt','wb')
#        cPickle.dump(p2,file1)
#        file1.close()

        boost_df['score'] = malig_tree.predict(p1)

#        data_file =open('nn_pred.txt', 'rb') 
#        boost_df = pickle.load(data_file,encoding='bytes')
#        data_file.close()


        n_img=np.zeros(t_petct_segs.shape,dtype=np.float32)
        for _, n_row in boost_df.iterrows():
            n_img[t_petct_segs==n_row['label']]=n_row['score']
            
        plt.imshow(n_img)
        plt.colorbar()
        plt.show()
        print('Region Analysis for ', len(REG), 'superpixels')
t=np.unique(n_img)[-4]

thre=n_img*np.uint(n_img>t)      
plt.imshow(thre)
plt.colorbar()
plt.show()

i=1*(n_img>0)
l=1*(lab_proj>0)
over_w=np.sum(i[i+l==1])/np.sum(i)
over_t=np.sum(l[i+l==2])/np.sum(l)
print('w{}\nt{}'.format(over_w,over_t))