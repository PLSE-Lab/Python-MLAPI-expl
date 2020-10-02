#!/usr/bin/env python
# coding: utf-8

# # Fastai2 DICOM starter

# [Fastai2](https://github.com/fastai/fastai2) starter code using DICOMs.  DICOM(Digital Imaging and COmmunications in Medicine) is the de-facto standard that establishes rules that allow medical images(X-Ray, MRI, CT) and associated information to be exchanged between imaging equipment from different vendors, computers, and hospitals.
# 
# DICOM files typically have a `.dcm` extension and provides a means of storing data in separate 'tags' such as patient information as well as image/pixel data. A DICOM file consists of a header and image data sets packed into a single file. The information within the header is organized as a constant and standardized series of tags. 
# 
# By extracting data from these tags one can access important information regarding the patient demographics, study parameters, etc
# 
# ![Parts of a DICOM](https://asvcode.github.io/MedicalImaging/images/copied_from_nb/my_icons/dicom_.PNG)
# 
# You can find out more about medical imaging by viewing this [blog](https://asvcode.github.io/MedicalImaging/)

# In[ ]:


get_ipython().system('pip install fastai2 -q')


# ### Load the dependancies

# In order to load DICOMs in fastai2 we need to all load the `fastai2.medical.imaging` module.  However we will not be able to use the full functionality of the medical imaging module because these DICOM images are saved as `XC` format which stands for `External-camera Photography` hence these images are restricted to pixel values between `0` and `255`.  This is way limited to say 16 bit DICOM images that could have values ranging from `-32768` to `32768`.
# 
# `Pydicom` is a python package for parsing DICOM files and makes it easy to covert DICOM files into pythonic structures for easier manipulation. Files are opened using pydicom.dcmread
# 
# 

# In[ ]:


#Load the dependancies
from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *
from fastai2.medical.imaging import *

import pydicom
import seaborn as sns

import numpy as np
import pandas as pd
import os


# Specify the source

# In[ ]:


source = Path("../input/siim-isic-melanoma-classification")
files = os.listdir(source)
print(files)


# Specify the folder that contains the training images `train` and use `fastai2`s method of accessing the DICOM files by using `get_dicom_files`

# In[ ]:


train = source/'train'
train_files = get_dicom_files(train)
train_files


# Lets see what information is contained within each DICOM file

# In[ ]:


patient1 = train_files[7]
dimg = dcmread(patient1)


# You can now view all the information of the DICOM file. Explanation of each element is beyond the scope of this notebook but [this](http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html#sect_C.7.6.3.1.4) site has some excellent information about each of the entries. Information is listed by the DICOM tag (eg: 0008, 0005) or DICOM keyword (eg: Specific Character Set)

# In[ ]:


dimg


# Create a function that will display an image and show choosen tags within the head of the DICOM, in this case `PatientName`, `PatientID`, `PatientSex`, `BodyPartExamined` and we can use `Tranform` from `fastai2` that conveniently allows us to resize the image

# In[ ]:


def show_one_patient(file):
    """ function to view patient image and choosen tags within the head of the DICOM"""
    pat = dcmread(file)
    print(f'patient Name: {pat.PatientName}')
    print(f'Patient ID: {pat.PatientID}')
    print(f'Patient age: {pat.PatientAge}')
    print(f'Patient Sex: {pat.PatientSex}')
    print(f'Body part: {pat.BodyPartExamined}')
    trans = Transform(Resize(256))
    dicom_create = PILDicom.create(file)
    dicom_transform = trans(dicom_create)
    return show_image(dicom_transform)


# In[ ]:


show_one_patient(patient1)


# **But why does the image look so unnatural?**

# This is because these images are stored in `YBR_FULL_422` color space and this is stated in the following tag:
# 
# `(0028, 0004) Photometric Interpretation CS: 'YBR_FULL_422'`
# 
# To view the images as they are intended the color space needs to be converted from `YBR_FULL_422` to `RGB`.  `Pydicom` provides a means of converting from one color space to another by using `convert_color_space` where it takes the (pixel array, current color space, desired color space) as attributes.  This is done by acessing the `pixel_array` and then converting to the desired color space

# In[ ]:


from pydicom.pixel_data_handlers.util import convert_color_space


# In[ ]:


arr = dimg.pixel_array
convert = convert_color_space(arr, 'YBR_FULL_422', 'RGB')
show_image(convert)


# That looks better!

# ### Pixel Distribution

# We can also view the pixel distribution of the image.  For this competition this is not really that important but can be as shown in this kernel [Understanding Dicoms](https://www.kaggle.com/avirdee/understanding-dicoms)

# In[ ]:


px = dimg.pixels.flatten()
plt.hist(px, color='c')


# # EDA

# Load in the csv file

# In[ ]:


df = pd.read_csv(source/'train.csv')
df.head()


# We can now explore the distribution of the data

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


plot_comparison3(df, 'sex', 'age_approx', 'benign_malignant')


# In[ ]:


#Plot 1 comparisons
def plot_comparison1(df, feature):
    "Plot 1 comparisons from a dataframe"
    fig, (ax1) = plt.subplots(1,1, figsize = (16, 4))
    s1 = sns.countplot(df[feature], ax=ax1)
    s1.set_title(feature)
    plt.show()


# In[ ]:


plot_comparison1(df, 'diagnosis')


# In[ ]:


plot_comparison1(df, 'target')


# In[ ]:


plot_comparison1(df, 'anatom_site_general_challenge')


# Lets create a dataframe with a few features we want to explore more

# In[ ]:


eda_df = df[['sex','age_approx','anatom_site_general_challenge','diagnosis','target']]
eda_df.head()


# In[ ]:


len(eda_df)


# There are a number of `Nan` values within each column that we want to get rid off

# In[ ]:


sex_count = eda_df['sex'].isna().sum(); age_count = eda_df['age_approx'].isna().sum(); anatom_count = eda_df['anatom_site_general_challenge'].isna().sum()
print(f'Nan values in sex column: {sex_count}, age column: {age_count}, anatom count: {anatom_count}')


# In[ ]:


df_drop = eda_df.dropna()
len(df_drop)


# For some more EDA we need to convert the categorical features in the dataframe into numeric values. `LabelEncoder` encode labels with a value between 0 and n_classes-1 where `n` is the number of distinct labels

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
edaa_df = eda_df.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')
edaa_df.head()


# lets set some seaborn parameters

# In[ ]:


sns.set(style="whitegrid")
sns.set_context("paper")


# In[ ]:


sns.pairplot(eda_df, hue="target", height=5, aspect=2, palette='gist_rainbow_r')


# In[ ]:


sns.pairplot(eda_df, hue="age_approx", height=6, aspect=3, diag_kws={'bw':'0.05'})


# The plot above shows the distribution between age and target, here it is clear to see the differences between age in patients with target `0` and target `1`

# In[ ]:


sns.set(style="whitegrid")
sns.set_context("poster")
sns.pairplot(edaa_df, hue="target", height=6, palette='gist_rainbow', diag_kws={'bw':'0.05'})


# # Getting the data ready for training

# Lets first specify the `x` or input.  In this case we can create a `lambda` function that will get the image files from the `train` folder

# In[ ]:


get_x = lambda x:source/'train'/f'{x[0]}.dcm'


# We now specify the 'y' or output using `ColReader` and specify the `target` column in the csv file which in this case `0` denotes benign and `1` denotes malignant.  

# In[ ]:


get_y=ColReader('target')


# Getting some quick `batch_tfms`

# In[ ]:


batch_tfms = aug_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# `fastai2` provides a convenient way of using `blocks`, in this case because we are specifying an `x` and a `y` we can now specify that the `x` will be `PILDicom` image and the `y` will be a `CategoryBlock` because we want the target to be either benign, `0` or malignant `1`

# However before we can create the `DataBlock` so that the images look 'real' we need to create a new method so that `PILBase` takes into consideration the `Photometric Interpretation`.

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


# We can now easily collate all the data into a `DataBlock` and use `fastai2`s inbuilt `splitter` function that will split the data into `train` and `valid` sets.  `Resize` ensure all the images are the same size when we feed it to the model.

# In[ ]:


melanoma = DataBlock(blocks=blocks,
                   get_x=get_x,
                   splitter=RandomSplitter(),
                   item_tfms=Resize(128),
                   get_y=ColReader('target'),
                   batch_tfms=batch_tfms)


# As the dataset is huge, we can test the model by just training with `100` samples from the `train` dataset

# In[ ]:


dls = melanoma.dataloaders(df.sample(100), bs=2)


# In[ ]:


dls = dls.cuda()


# Viewing a batch

# In[ ]:


dls.show_batch(max_n=12, nrows=2, ncols=6)


# Specify the evaluation metric and check how many labels there are

# In[ ]:


roc = RocAuc()
dls.c


# Specify the architecure to be used.  In this case we ensure that the the output of the model is either `0` or `1` or 2 classes.  `dls.c` is a convenient way to specify that the output of the model will be 2.

# In[ ]:


model = xresnet18_deeper(n_out=dls.c)


# In[ ]:


set_seed(77)
learn = Learner(dls, model, 
                opt_func=ranger,
                loss_func=LabelSmoothingCrossEntropy(),
                metrics=[accuracy, roc],
                cbs = ShowGraphCallback())


# In[ ]:


learn.freeze()
learn.fit_one_cycle(1, 5e-2)


# In[ ]:


learn.save('xresnet18_stg1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_flat_cos(2,slice(1e-6,1e-4))


# In[ ]:


learn.save('xresnet18_stg2')


# In[ ]:


interp = Interpretation.from_learner(learn)


# We can look at the top losses

# In[ ]:


interp.plot_top_losses(12)


# # Loading the `test` set

# In[ ]:


tst = source/'test'
test_set = get_dicom_files(tst)
test_set


# For testing purposes we will only use the first 100 images in the test set

# In[ ]:


test_set = test_set[:100]
test_set


# Specify a test patient 

# In[ ]:


test_patient = test_set[1]
test_patient


# In[ ]:


learn.load('xresnet18_stg2')


# Lets look at a prediction for the `test_patient`

# In[ ]:


_ = learn.predict(test_patient)
_


# The `predict` function displays the predicted class, in this case `0`, the tensor class `tensor(0)` and the probabilites of the each of the classes.  In this dataset there are 2 classes `0` and `1` and the the probabilites are predicted for each class so in this case the probablility that the `test_patient` is `benign` or class `0` is `0.9923` and the probability that the `test_patient` is `malignant` or `1` is `0.0317`

# The evalution requirement in this competiton is that for each image_name in the test set, you must predict the probability (target) that the sample is malignant.  So we need to get the probability of class `1`
# 
# We can use the code below to get the probability of class `1`:

# In[ ]:


_ = learn.predict(test_patient)
print(_[2][1])


# Load the `sample_submisson`

# In[ ]:


sample_sub = pd.read_csv(source/'sample_submission.csv')
sample_sub = sample_sub[:100]
sample_sub


# We can delete the `target` column as we will be populating this with the probabilites

# In[ ]:


del sample_sub['target']


# Get the probabilites for the test set and create a list of the probabilites for each image in the test set and convert the probabilty to a float

# In[ ]:


sample_list = []
for i in test_set:
    pre = learn.predict(i)
    l = float(pre[2][1])
    sample_list.append(l)


# In[ ]:


sample_list


# In[ ]:


sub = sample_sub.assign(target=sample_list)
sub.to_csv('submission.csv', index=False)


# In[ ]:


sub = pd.read_csv('submission.csv')
sub


# ## Next steps:
# 
# - Experiment with various models and augmentations
# 
