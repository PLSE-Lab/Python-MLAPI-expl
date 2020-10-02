#!/usr/bin/env python
# coding: utf-8

# This tutorial is for helping you kickstart this competition using fast.ai V1 library. In case you are not aware of this library please have a look at it here. It is really awesome and you can get good results really fast using this library.   

# ## Import packages

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import pydicom # for reading .dcm images 
import os
import numpy
from matplotlib import pyplot, cm
from fastai.vision import *
import fastai
from IPython.display import FileLink


# ## Code for Reading DCM images

# Note that fast.ai library v1 does not natively support reading .dcm images. Therefore, I have made some modifications to the open image function to help us read .dcm images. Also some rescaling is required for these .dcm images, so I have incorporated that also in the code. You can read more about it here http://www.idlcoyote.com/fileio_tips/hounsfield.html

# In[ ]:


# function to open dcm images
def open_dcm_image(fn:PathOrStr, div:bool=True, convert_mode:str='L', cls:type=Image,
        after_open:Callable=None)->Image:
    "Return `Image` object created from image in file `fn`."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin
        #x= PIL.Image.open(fn).convert(convert_mode) # previous code
        # code added for opening dcm images
        dicom_file = pydicom.dcmread(str(fn))
        arr = dicom_file.pixel_array.copy() 
        arr = arr * int(dicom_file.RescaleSlope) + int(dicom_file.RescaleIntercept) 
        level = 40; window = 80
        arr = np.clip(arr, level - window // 2, level + window // 2)
        x = PIL.Image.fromarray(arr).convert(convert_mode)
    if after_open: x = after_open(x)
    x = pil2tensor(x,np.float32)
    if div: x.div_(255)
    return cls(x)


# In[ ]:


# modifying open_image function of fast.ai
fastai.vision.data.open_image = open_dcm_image


# ## Preparation of Input Data

# In[ ]:


dirpath = "../input/rsna-intracranial-hemorrhage-detection/"
get_ipython().system('ls ../input/rsna-intracranial-hemorrhage-detection')


# ### Read train Data

# In[ ]:


# read train csv that has filename and the corresponding labels 
df_train = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')
df_train['fn'] = df_train.ID.apply(lambda x: '_'.join(x.split('_')[:2]) + '.dcm')
df_train['label'] = df_train.ID.apply(lambda x: x.split('_')[-1])
df_train.head()
# it was pointed out in the discussion that there is one file that is corrupred
# removing the corrupted file
df_train = df_train[df_train.fn != 'ID_000039fa0.dcm']


# In[ ]:


# display images
open_dcm_image( dirpath+"stage_1_train_images/"+df_train.fn.values[5],convert_mode= 'L').show(cmap= 'gray')


# In[ ]:


print(df_train.shape)
df_train.drop_duplicates(inplace = True)
print(df_train.shape)


# This is a typical multilabel classification problem. In the train set csv, the data is arranged such that for each image and label (type of haemorrage) combination we have tag that particular type is present or not in image. I will rearrange this data to have all the labels presnt for that image in one row instead of multiple. This is particularly done so that we can load the data using fastai functions.

# In[ ]:


pivot = df_train.pivot(index='fn', columns='label', values='Label')
pivot.reset_index(inplace=True)
# chcek if there are only two types of values in any
assert pivot[pivot['any'] == 0].shape[0] + pivot[pivot['any'] == 1].shape[0] == pivot.shape[0] 
pivot['any'].value_counts()


# In[ ]:


mask = pivot['any'] == 0
pivot['None'] = ""
pivot.loc[mask, 'None'] = 'None'

label_cols = ['any', 'epidural', 'intraparenchymal', 'intraventricular','subarachnoid', 'subdural', 'None'] 
for col in label_cols:
    print(col, end= ", ")
    pivot[col] = pivot[col].replace({0:"", 1:col})
    
pivot['MultiLabel'] = pivot[label_cols].apply(lambda x: " ".join((' '.join(x)).split()), axis=1)


# ### Read Test data

# similarly we have read the test data.

# In[ ]:


df_test = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv')
df_test['fn'] = df_test.ID.apply(lambda x: '_'.join(x.split('_')[:2]) + '.dcm')
df_test['label'] = df_test.ID.apply(lambda x: x.split('_')[-1])


# In[ ]:


pivot_test = df_test.pivot(index='fn', columns='label', values='Label')
pivot_test.reset_index(inplace=True)
pivot_test['MultiLabel'] = " "


# # Create Image databunch

# Now we will load the data

# In[ ]:


# get_transforms is for data augmentation, however I am not using this as of now
#tfms = get_transforms(do_flip = False)


# In[ ]:


path = "../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/"
np.random.seed(42)
data_train = (ImageList
.from_df(path=path,df= pivot[['fn', 'MultiLabel']])
.split_by_rand_pct()
.label_from_df(cols=1, label_delim = " ")
.transform(size=(128,128))
.databunch()
.normalize(imagenet_stats))


# In[ ]:


path = "../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/"
data_test = (ImageList
.from_df(path=path,df= pivot_test[['fn', 'MultiLabel']])
.split_by_rand_pct(valid_pct= 0)
.label_from_df(cols=1, label_delim = " ")
.transform(size=(128,128))
.databunch()
.normalize(imagenet_stats))


# In[ ]:


assert len(data_test.train_ds.y) == pivot_test.shape[0]


# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# ## Train model

# In[ ]:


path = '/output/'
acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)


# In[ ]:


learn = cnn_learner( data_train, models.resnet34, path = path, metrics = [acc_02, f_score] )


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 3e-3
learn.freeze()
learn.fit_one_cycle(2, slice(lr))


# In[ ]:


learn.save('stage-1')
learn.load('stage-1')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(1, slice(3e-5, 3e-4))


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# ## Get Predictions

# In[ ]:


# load the classed from the learner class
# This will give us the order in which predictions were given by the model
data_classes = learn.data.classes


# In[ ]:


# loading test dataset 
learn.data = data_test
# get prediction from the model
y_predict, _ = learn.get_preds(ds_type=DatasetType.Fix)
# check we got predictions for all test files
assert len(y_predict) == pivot_test.shape[0]


# In[ ]:


learn.data.classes


# In[ ]:


pivot_test['MultiLabel']  = y_predict
# Now loading the prediction from multilabel column to repsective labels
for i, col in enumerate(data_classes):
    print(col, end= ", ")
    pivot_test[col] = pivot_test['MultiLabel'].apply(lambda x: x[i].numpy())


# In[ ]:


# Now rearranging the predictions to the competitons format
cols_to_consider = [col for col in pivot_test.columns if not col in ['None', 'MultiLabel']]
print(cols_to_consider)
df_temp = pd.melt(pivot_test[cols_to_consider], id_vars= ['fn'])


# In[ ]:


df_temp['ID'] = df_temp['fn'].apply(lambda x: x[:-4])
df_temp['ID'] = df_temp[['ID', 'label']].apply(lambda x: "_".join(x), axis = 1)


# In[ ]:


assert len(set(df_test.ID.unique()).intersection(set(df_temp.ID.unique()))) == df_test.shape[0]


# In[ ]:


df_temp.rename(columns={'value':'Label'}, inplace = True)
df_temp[['ID', 'Label']].to_csv("submission6.gz", compression = 'gzip' , index = False)


# In[ ]:


# this will generate a link using which you can download your predictions
FileLink("submission6.gz")

