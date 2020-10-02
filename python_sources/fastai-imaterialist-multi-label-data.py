import numpy as np
import pandas as pd
import shutil
import zipfile  

from pathlib import Path
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *

from itertools import groupby
from progressbar import ProgressBar
import cv2
from multiprocessing import Pool
import os
import json
import torchvision
from datetime import datetime


path = Path("../input")
path_img = path/'train'
path_lbl = Path("../labels")
path_lbl_bw = Path("../labels_bw")

SIZE_LBL = 512
# only the 27 apparel items, plus 1 for background
# model image size 224x224
category_num = 46 + 1
size = 512
EXT_LBL = 'png'

# create a folder for the mask images
if  not os.path.isdir(path_lbl):
    os.makedirs(path_lbl)
    

if  not os.path.isdir(path_lbl_bw):
    os.makedirs(path_lbl_bw)
    
    
    
# train dataframe
df = pd.read_csv(path/'train.csv')
# get and show categories
with open(path/"label_descriptions.json") as f:
    label_descriptions = json.load(f)
    
    
masks = {}

def make_attr_img(segment_df):
#     import pdb; pdb.set_trace()
    seg_width = segment_df.Width
    seg_height = segment_df.Height

    seg_img = np.copy(masks.get((seg_width, seg_height)))
    try:
        if not seg_img:
            seg_img = np.full(seg_width*seg_height, category_num-1, dtype=np.int32)
            masks[(seg_width, seg_height)] = np.copy(seg_img)
    except:
        # seg_img exists
        pass
    
    encoded_pixels = segment_df["EncodedPixels"]
    class_id = segment_df["ClassId"].split("_")
    class_id, attr_id = class_id[0], class_id[1:]
#     new_label = ["{}:{}".format(class_id, a) for a in attr_id]
    new_label = " ".join(attr_id) or "BG"
    
    if not new_label: 
        return seg_img, class_id, new_label
    
    pixel_list = list(map(int, encoded_pixels.split(" ")))
    
    for i in range(0, len(pixel_list), 2):
        start_index = pixel_list[i] - 1
        index_len = pixel_list[i+1] - 1
        seg_img[start_index:start_index+index_len] = int(class_id)
            
    seg_img = seg_img.reshape((seg_height, seg_width), order='F')
    return seg_img, class_id, new_label
        

def create_attribute_label(df, images, path_lbl_bw):
    """
    img_name = "000aac3870ea7c59ca0333ffa5327323.jpg"
    tdf = create_attribute_label(tdf, [img_name], path_lbl_bw)
    """
    print("Start creating label")
    zipf = zipfile.ZipFile('multiclass-labels.zip', 'w')
    for idx,img in enumerate(images):
        img_df = df[df.ImageId == img]
        for i, row in img_df.iterrows():
            fpath = path_lbl_bw.joinpath(
                "{}_{}.{}".format(os.path.splitext(img)[0] ,row.ClassId.split('_')[0],EXT_LBL))
            fname = fpath.as_posix()
            mask, kid, lbl = make_attr_img(row)
            if not os.path.isfile(fname): # if not exist, create
                img_mask_3_chn = np.dstack((mask, mask, mask))
                cv2.imwrite(fname, img_mask_3_chn)
                zipf.write(fname)
                fpath.unlink()
            df.loc[i, "AttrId"] = lbl
            df.loc[i, "ClassId"] = kid
        if idx % 40 ==0 : print(idx, end=" ")
    print("Finish creating label")
    zipf.close()
    return df
            
            


# img_name = "000aac3870ea7c59ca0333ffa5327323.jpg"
# tdf = create_attribute_label(tdf, [img_name], path_lbl_bw)

t1df = df[df.apply(lambda x: len(x.ClassId.split('_'))>1, axis=1)]
t1df["AttrId"] = ""
t2df = df[~df.ImageId.isin((t1df.ImageId.unique()))]   
t2df = t2df.sample(10000)
t2df["AttrId"] = ""
tdf = pd.concat([t1df[:3000], t2df[:500]])                                #<---------- HERE
del t1df, t2df #, df

tdf = create_attribute_label(tdf, tdf.ImageId.unique(), path_lbl_bw)     #<---------- HERE
tdf = tdf[tdf.ImageId.isin(tdf.ImageId.unique()) ]                       #<---------- HERE

shutil.make_archive('multi-label', 'zip', path_lbl_bw)

##
tdf["ImageId"] = tdf.apply(lambda x: ( x.ImageId.split(".")[0]  + '_' + x["ClassId"] ), axis=1)
tdf = tdf[['ImageId','AttrId']]
tdf.to_csv('multilabel.csv',index=False)
tdf.head()

path_lbl_bw.unlink()