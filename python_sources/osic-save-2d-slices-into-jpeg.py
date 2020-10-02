#!/usr/bin/env python
# coding: utf-8

# Notebooks refered 
# * https://www.kaggle.com/vijaybj/merge-all-slices-into-one-image-per-patient (fixed some errors wrt dimensions)
# * https://www.kaggle.com/ulrich07/osic-keras-starter-with-custom-metrics
# 
# Attempt to merge patients 2d slices into a jpeg

# In[ ]:


import numpy as np
import pandas as pd
import pydicom
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


# In[ ]:


ROOT = "../input/osic-pulmonary-fibrosis-progression"
#DESIRED_SIZE = 256 # Memory issue
DESIRED_SIZE = 128


# In[ ]:


tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
tr['basescan'] = tr['Patient']
chunk = pd.read_csv(f"{ROOT}/test.csv")
chunk['basescan'] = chunk['Patient']

print("add infos")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub['basescan'] = sub['Patient']
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub =  sub[['Patient','Weeks','Confidence','Patient_Week', 'basescan']]
sub = sub.merge(chunk.drop('Weeks', axis=1), on=["Patient","basescan"])


# In[ ]:


tr.shape, chunk.shape, sub.shape


# In[ ]:


tr.head(10)


# In[ ]:


tr.tail(2)


# In[ ]:


tr['WHERE'] = 'train'
chunk['WHERE'] = 'val'
sub['WHERE'] = 'test'
data = tr.append([chunk, sub])


# In[ ]:


COLS = ['Sex','SmokingStatus']
FE = []
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)
#=================


# In[ ]:


#
data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )
data['week'] = (data['Weeks'] - data['Weeks'].min() ) / ( data['Weeks'].max() - data['Weeks'].min() )
data['percent'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min() )
FE += ['age','percent','week']


# In[ ]:


data.head()


# In[ ]:


data.shape, tr.shape


# In[ ]:


tr = data.loc[data.WHERE=='train']
chunk = data.loc[data.WHERE=='val']
sub = data.loc[data.WHERE=='test']
del data


# In[ ]:


tr.shape, chunk.shape, sub.shape


# In[ ]:


tr.head(2)


# In[ ]:


range(tr.shape[0])


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys


# In[ ]:


# Load the scans in given folder path
def load_scan(path):
    #print(os.listdir(path))
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    #print(slices[0].ImagePositionPatient)
    #print(slices[0].ImagePositionPatient[2])
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


# In[ ]:


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


# In[ ]:


get_ipython().system('pip install pillow')


# In[ ]:


from pydicom.pixel_data_handlers import gdcm_handler, pillow_handler

pydicom.config.image_handlers = ['pillow_handler']
def merge_2d_slices_into_one(slices):
    #print("slices len : %d " % (len(slices)))
    arr = np.zeros((512, 512), np.int16)
    count = 3
    for im in slices:
        #print(type(im))
        #print(im)
        im = Image.fromarray(im)
        im = im.resize((512,512)) 
        pixels = list(im.getdata())
        width, height = im.size
        pixels = [pixels[i * width:(i + 1) * width] for i in range(height)] 
        im = np.array(pixels)
        #print(type(im))
        #print(im.shape)
        smallest = np.amin(im)
        biggest = np.amax(im)
        #print(" biggest : %d , smallest : %d" % ( biggest,smallest ))
        #imarr = np.array(im, dtype=np.int16)        
        arr = arr + (1 - im)*(np.log(count)/(biggest - smallest))

        #print ((N * 14)/ np.log10(count))
        count = count + 1
        #arr = np.array(np.round(arr), dtype=np.uint8)
        arr = np.array(np.round(arr),dtype=np.uint8)
    
    return arr


# In[ ]:


import zipfile
def zip_and_remove(zipname, path):
    ziph = zipfile.ZipFile(f'{zipname}.zip', 'w', zipfile.ZIP_DEFLATED)
    
    for root, dirs, files in os.walk(path):
        print(files)
        for file in tqdm(files):
            file_path = os.path.join(root, file)
            #print(file_path)
            ziph.write(file_path)
            os.remove(file_path)
    
    ziph.close()


# In[ ]:


out_path = "/output/kaggle/working/"
import traceback
import sys

def get_images_3d(df, how="train"):
    xo = []
    p = []
    w  = []
    img_set = []
    for i in tqdm(range(df.shape[0])):
        patient = df.iloc[i,0]
        week = df.iloc[i,1]
        basescan = df.iloc[i,7]
        try:
            if basescan not in img_set:
                img_path = f"{ROOT}/{how}/{basescan}/"
                slices = load_scan(img_path)
                slices_pixel_array = get_pixels_hu(slices)
                pixel_array = merge_2d_slices_into_one(slices_pixel_array)
                #print(basescan)
                #print(pixel_array.shape)
                im = Image.fromarray(pixel_array)
                im = im.resize((DESIRED_SIZE,DESIRED_SIZE)) 
                out_path = f"../working/{how}/{basescan}.jpg"
                im.save(out_path)
                im = np.array(im)
                xo.append(im[np.newaxis,:,:])
                p.append(patient)
                w.append(week)
                img_set.append(basescan)
        except Exception as e:
            print(e)
            traceback.print_exc()
            pass
    data = pd.DataFrame({"Patient":p,"Weeks":w})
    return xo, data


# In[ ]:


train_path = "../working/train/" 
os.makedirs(train_path , exist_ok=True)
x, df_tr = get_images_3d(tr, how="train")


# In[ ]:


print(os.listdir("../working/train"))


# In[ ]:


zip_and_remove('train' ,train_path)


# In[ ]:


def list_zip_file_contents(zipfilename):
    # Create a ZipFile Object and load sample.zip in it
    with zipfile.ZipFile(zipfilename, 'r') as zipObj:
       # Get list of files names in zip
       listOfiles = zipObj.namelist()
       # Iterate over the list of file names in given list & print them
       for elem in listOfiles:
           print(elem)


# In[ ]:


list_zip_file_contents("train.zip")


# In[ ]:


test_path="../working/test/"
os.makedirs(test_path , exist_ok=True)
x_test, df_test = get_images_3d(sub, how="test")
zip_and_remove('test' ,test_path)


# In[ ]:


list_zip_file_contents("test.zip")


# In[ ]:


sub['Patient'].nunique()


# In[ ]:


tr['Patient'].head(100).nunique()


# In[ ]:




