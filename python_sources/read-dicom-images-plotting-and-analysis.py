#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train=pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')
test=pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')


# In[ ]:


train.head(1)


# In[ ]:


test.head()


# In[ ]:


from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *
from fastai2.medical.imaging import *

import pydicom
import seaborn as sns

import numpy as np
import pandas as pd
import os


# In[ ]:


pip install fastai2


# In[ ]:


source = Path("/kaggle/input/osic-pulmonary-fibrosis-progression/")
files = os.listdir(source)
print(files)


# In[ ]:


train = source/'train'
train_files = get_dicom_files(train)
train_files


# In[ ]:


patient1 = train_files[7]
dimg = dcmread(patient1)


# In[ ]:


dimg


# In[ ]:


def show_one_patient(file):
    """ function to view patient image and choosen tags within the head of the DICOM"""
    pat = dcmread(file)
    print(f'patient Name: {pat.PatientName}')
    print(f'Patient ID: {pat.PatientID}')
    #print(f'Patient age: {pat.PatientAge}')
    print(f'Patient Sex: {pat.PatientSex}')
    print(f'Body part: {pat.BodyPartExamined}')
    trans = Transform(Resize(256))
    dicom_create = PILDicom.create(file)
    dicom_transform = trans(dicom_create)
    return show_image(dicom_transform)


# In[ ]:


show_one_patient(patient1)


# In[ ]:


from pydicom.pixel_data_handlers.util import convert_color_space


# In[ ]:


arr = dimg.pixel_array
convert = convert_color_space( arr,'RGB', 'RGB')
show_image(convert)


# In[ ]:


px = dimg.pixels.flatten()
plt.hist(px, color='c')


# In[ ]:


df = pd.read_csv(source/'train.csv')
df.head()


# In[ ]:


test


# In[ ]:


#Plot 3 comparisons
def plot_comparison3(df, feature, feature1, feature2):
    "Plot 3 comparisons from a dataframe"
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (16, 4))
    s1 = sns.countplot(df[feature], ax=ax1)
    s1.set_title(feature)
    s2 = sns.countplot(df[feature1], ax=ax2)
    s2.set_title(feature1)
    s3 = sns.countplot(df[feature2], ax=ax3)
    s3.set_title(feature2)
    plt.show()


# In[ ]:


plot_comparison3(df, 'Sex', 'Age', 'SmokingStatus')


# In[ ]:


#Plot 1 comparisons
def plot_comparison1(df, feature):
    "Plot 1 comparisons from a dataframe"
    fig, (ax1) = plt.subplots(1,1, figsize = (16, 4))
    s1 = sns.countplot(df[feature], ax=ax1)
    s1.set_title(feature)
    plt.show()


# In[ ]:


plot_comparison1(df, 'Percent')


# In[ ]:


eda_df = df[['Sex','Age','SmokingStatus','Percent','FVC']]
eda_df.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
edaa_df = eda_df.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')
edaa_df.head()


# In[ ]:


sns.set(style="whitegrid")
sns.set_context("paper")


# In[ ]:


sns.pairplot(eda_df, hue="FVC", height=5, aspect=2, palette='gist_rainbow_r')


# In[ ]:


get_x = lambda x:source/'train'/f'{x[0]}.dcm'


# In[ ]:


get_y=ColReader('FVC')


# In[ ]:


batch_tfms = aug_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[ ]:


class PILDicom2(PILBase):
    _open_args,_tensor_cls,_show_args = {},TensorDicom,TensorDicom._show_args
    @classmethod
    def create(cls, fn:(Path,str,bytes), mode=None)->None:
        "Open a `DICOM file` from path `fn` or bytes `fn` and load it as a `PIL Image`"
        dimg = dcmread(fn)
        arr = dimg.pixel_array; convert = convert_color_space(arr,'YBR_FULL_422', 'RGB')
        im = Image.fromarray(convert)
        im.load()
        im = im._new(im.im)
        return cls(im.convert(mode) if mode else im)


# In[ ]:


blocks = (ImageBlock(cls=PILDicom2), CategoryBlock)


# In[ ]:


ct = DataBlock(blocks=blocks,
                   get_x=get_x,
                   splitter=RandomSplitter(),
                   item_tfms=Resize(128),
                   get_y=ColReader('target'),
                   batch_tfms=batch_tfms)


# In[ ]:


dls = ct.dataloaders(df.sample(100), bs=4)


# In[ ]:




