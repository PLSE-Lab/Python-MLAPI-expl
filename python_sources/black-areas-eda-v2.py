#!/usr/bin/env python
# coding: utf-8

# ### In this kernel I will analyse images with black areas and hope to find some dependencies

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import cv2
import os

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


def rle2mask(rle, imgshape = (256,1600)):
    width = imgshape[0]
    height= imgshape[1]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )


# ### Read train data

# In[ ]:


path = '../input/severstal-steel-defect-detection/'


# In[ ]:


tr = pd.read_csv(path + 'train.csv')
print(tr.shape)
tr.head()


# In[ ]:


df = tr[tr['EncodedPixels'].notnull()]
df['ClassId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])
df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
print(len(df))
df.head()


# ### Show some images with masks

# In[ ]:


def ShowImgMask(df, sub = 'train',  columns = 1, rows = 4):
    fig = plt.figure(figsize=(20,columns*rows+6))
    for i in range(1,columns*rows+1):
        fn = df['ImageId_ClassId'].str[:-2].iloc[i]
        fig.add_subplot(rows, columns, i).set_title(fn)
        img = cv2.imread( path + sub + '_images/'+fn )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = rle2mask(df['EncodedPixels'].iloc[i])
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, 0, 2)
        plt.imshow(img)
    plt.show()


# In[ ]:


ShowImgMask(df)


# ### Fing images with black areas

# In[ ]:


def GetLftRgtBl(img_nm, sub = 'train'):
    img = plt.imread(path + sub + '_images/' + img_nm)[:,:,0][:1][0]
    
    bgn_lf = 0
    for i, x in enumerate(img):
        if x > 15:
            bgn_lf = i
            break
    
    bgn_rg = 0
    for i, x in reversed(list(enumerate(img))):
        if x > 15:
            bgn_rg = i
            break
    return bgn_lf, bgn_rg


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_cut2 = df.copy(deep=True).reset_index(drop=True)\ndf_cut2['BgnLeft'] = 0\ndf_cut2['BgnRigth'] = 0\ndf_cut2.head()\n\nfor i, row in df_cut2.iterrows():    \n    df_cut2.at[i,'BgnLeft'], df_cut2.at[i,'BgnRigth'] = GetLftRgtBl(row['ImageId'])")


# In[ ]:


df_cut2.head(10)


# ### Distribution of images with black areas and without

# In[ ]:


df_bl2 = df_cut2[(df_cut2['BgnLeft'] > 0) | (df_cut2['BgnRigth']  < 1599)]
print(len(df_bl2))                                             
df_bl2.head()


# In[ ]:


ShowImgMask(df_bl2)


# ### Show distribution of images that have and do not have black areas

# In[ ]:


def ShowDistBlc(df_black, df_all):
    lbls = ('has_black_area', 'no_black_area')
    y_pos = np.arange(len(lbls))
    cnt = [len(df_black),(len(df_all) - len(df_black))]
    print(cnt)

    plt.bar(y_pos, cnt, align='center', alpha=0.5)
    plt.xticks(y_pos, lbls)
    plt.ylabel('Count')
    plt.title('Distribution of images with black areas and without')

    plt.show()


# In[ ]:


ShowDistBlc(df_bl2, df)


# In[ ]:


def ShowDist(df, isblk = ''):
    lbls = ('1', '2', '3', '4')
    y_pos = np.arange(len(lbls))

    cnt = df.groupby('ClassId')['ImageId_ClassId'].count()
    print(cnt)

    plt.bar(y_pos, cnt, align='center', alpha=0.5)
    plt.xticks(y_pos, lbls)
    plt.ylabel('Count')
    plt.title('Distribution of deffect classes among images with ' + isblk +' black areas')

    plt.show()


# ### Distribution of deffect classes among images with black areas

# In[ ]:


ShowDist(df_bl2)


# ### Distribution of deffect classes among images with no black areas

# In[ ]:


# No black

df_no_bl = df_cut2[~df_cut2.index.isin(df_bl2.index)]
ShowDist(df_no_bl, 'no')


# ### As we can see, almost all images with defect 2 have black areas. This is a useful discovery! 

# ### Distribution of black areas sizes

# In[ ]:


def ShowPlt(side):                    
    fig, axs = plt.subplots(1, 4, figsize=(10, 5))
    axs = axs.ravel()

    for i in range(4):
        df1 = df_bl2[df_bl2['ClassId'] == str(i + 1)]

        if side == 0: # Rigth side area
            df_rgt =df1[df1['BgnLeft'] == 0]
            cnt1 = 1599 - df_rgt['BgnRigth']
        else: # Left side area
            df_rgt =df1[df1['BgnRigth'] == 1599]
            cnt1 = 0 + df_rgt['BgnLeft']

        axs[i].hist(cnt1)
        axs[i].set_title('id = ' + str(i + 1))

    axs[0].set_xlabel('Length of black area')
    axs[0].set_ylabel('Count')
    plt.show()


# #### Rigth side black area

# In[ ]:


ShowPlt(side=0)


# #### Left side black area

# In[ ]:


ShowPlt(side=1)


# #### At first glance, there are certain dependencies that a defect belongs to a certain class, depending on the properties of black areas.
# #### These dependencies, provided they are same in the test dataset, can be used to adjust model predictions.

# ## Analyse test images

# ### Read data

# In[ ]:


files = []
for file in os.listdir(path + 'test_images/'):
    files.append(file)

df_tst = pd.DataFrame(files, columns=['ImageId'])
print(len(df_tst))
df_tst.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_tst2 = df_tst.copy(deep=True).reset_index(drop=True)\ndf_tst2['BgnLeft'] = 0\ndf_tst2['BgnRigth'] = 0\ndf_tst2.head()\n\nfor i, row in df_tst2.iterrows():    \n    df_tst2.at[i,'BgnLeft'], df_tst2.at[i,'BgnRigth'] = GetLftRgtBl(row[0], 'test')")


# In[ ]:


df_tst2.head()


# ### Fing images with black areas

# In[ ]:


df_tst_bl2 = df_tst2[(df_tst2['BgnLeft'] > 0) | (df_tst2['BgnRigth']  < 1599)]
print(len(df_tst_bl2))                                             
df_tst_bl2.head()


# ### Show distribution of images that have and do not have black areas

# In[ ]:


ShowDistBlc(df_tst_bl2, df_tst)


# ### As we can see, in test dataset a lot of images have black areas

# #### Show some image

# In[ ]:


plt.imshow(plt.imread(path + 'test_images/' + df_tst_bl2.iloc[4]['ImageId']))


# ### Distribution of black areas sizes

# In[ ]:


def ShowLenDistTst(side):
    
    if side == 0: # Rigth side black area
        df_rgt = df_tst_bl2[df_tst_bl2['BgnLeft'] == 0]
        cnt = 1599 - df_rgt['BgnRigth']
    elif side == 1: # Left side black area
        df_rgt =df_tst_bl2[df_tst_bl2['BgnRigth'] == 1599]
        cnt = 0 + df_rgt['BgnLeft']    

    plt.hist(cnt)
    plt.xlabel('Length of black areas')
    plt.ylabel('Count')
    plt.title('Distribution of lengths of black areas')

    plt.show()


# #### Rigth side black area

# In[ ]:


ShowLenDistTst(0)


# #### Left side black area

# In[ ]:


ShowLenDistTst(1)


# ### Make some predictions

# #### Let's see again on train disribution of left side black areas

# In[ ]:


ShowPlt(1)


# #### As we see if image have left side black area and it lenght in range between 600 to 700, than with high probability that image will contain deffect 1.
# #### Let's check this

# In[ ]:


df_tst_bl2.head()


# In[ ]:


df_lft_67 = df_tst_bl2[(df_tst_bl2['BgnRigth'] == 1599) & 
                       (df_tst_bl2['BgnLeft'] > 590) &
                       (df_tst_bl2['BgnLeft'] < 710)]


print(len(df_lft_67))
df_lft_67.head()


# In[ ]:


plt.figure(figsize=(20, 2))
plt.imshow(plt.imread(path + 'test_images/' + df_lft_67.iloc[4]['ImageId']))
plt.show()


# #### Further research will continue... :)
# #### Upvote, if you think this kernel was useful.
