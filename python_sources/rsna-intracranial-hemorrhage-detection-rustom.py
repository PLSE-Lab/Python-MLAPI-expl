#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import pydicom
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from matplotlib import pyplot, cm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


import fastai
from fastai.vision import *

from IPython.display import FileLink

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))


# First, let's take a look at the training data file. 

# In[ ]:


path = '../input/rsna-intracranial-hemorrhage-detection/'


# In[ ]:


train_raw = pd.read_csv(path + 'stage_1_train.csv')


# In[ ]:


train_raw.head()


# According to the dataset information, the ID column includes the patient's ID and the probability (0-1) of each of the 6 types of intracranial hemorrhages occuring. 

# In[ ]:


train_raw[0:20]


# We can see the types cycle every 6 rows. 

# In[ ]:


train_raw.shape


# Let's create a new dataframe that formats this training data more intuitively. The first step to this is splitting up the ID column using the underscores. 

# In[ ]:


train_raw['Sub_type'] = train_raw['ID'].str.split("_", n = 3, expand = True)[2]
train_raw['PatientID'] = train_raw['ID'].str.split("_", n = 3, expand = True)[1]
train_raw['PatientID'] = 'ID_' + train_raw['PatientID']
train_raw


# The new DataFrame will have a column of probabilities for each type of hemorrhage for each patient. 

# In[ ]:


train = pd.DataFrame()
train = train_raw.drop_duplicates()
train = train.pivot(index = 'PatientID', columns = 'Sub_type', values = 'Label')
train.reset_index(level=0, inplace=True)
train.columns.name = None
train


# In[ ]:


train['labels'] = ''
train.loc[train['any'] == 1,'labels'] += 'any'
train.loc[train['epidural'] == 1,'labels'] += ',epidural'
train.loc[train['intraparenchymal'] == 1,'labels'] += ',intraparenchymal'
train.loc[train['intraventricular'] == 1,'labels'] += ',intraventricular'
train.loc[train['subarachnoid'] == 1,'labels'] += ',subarachnoid'
train.loc[train['subdural'] == 1,'labels'] += ',subdural'
train


# Let's create a bar plot to see the frequency of different types of hemorrhages appearing. 

# In[ ]:


#nonzero = np.count_nonzero(train, axis=0)
g = sns.barplot(train.columns, np.count_nonzero(train, axis = 0))
g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")
plt.title('Frequncies of Hemorrhage Types')
plt.xlabel('Hemorrhage Type')
plt.ylabel('Nonzero Instances')


# Now, let's take a look at the images themselves. 

# In[ ]:


train_images_path = (path + 'stage_1_train_images/')
train_images = [f for f in listdir(train_images_path) if isfile(join(train_images_path, f))]
test_images_path = (path + 'stage_1_test_images/')
test_images = [f for f in listdir(test_images_path) if isfile(join(test_images_path, f))]
print('5 Training images', train_images[:5]) # Print the first 5


# In[ ]:


print('Number of train images:', len(train_images))
print('Number of test images:', len(test_images))


# In[ ]:


image = pydicom.dcmread(train_images_path + 'ID_ffff922b9.dcm')
plt.imshow(image.pixel_array)


# Now that we're able to plot one image, let's make a function to plot multiple images side by side (capped at 12). 

# In[ ]:


def plot_dcm(IDs):
    if (type(IDs) == str):
        IDs = [IDs]
    if len(IDs) > 12:
        raise Exception('Number of images should not exceed 12. The number of images was: {}'.format(len(IDs)))
    fig = plt.figure(figsize = (15, 10))
    columns = 4
    rows = 3
    index = 1
    for ID in IDs: 
        if (not ID.endswith('.dcm')):
            ID = ID + '.dcm'
        image = pydicom.dcmread(train_images_path + ID)
        fig.add_subplot(rows, columns, index)
        plt.imshow(image.pixel_array)
        fig.add_subplot
        index += 1
    plt.show()


# First, we'll just plot the random first ten images.

# In[ ]:


plot_dcm(train_images[12:24])


# Now, let's plot some of each kind of hemorrhage. 

# In[ ]:


certain_cases = train.sort_values('epidural', ascending = False)
plot_dcm(certain_cases.PatientID[0:12])


# In[ ]:


certain_cases = train.sort_values('subdural', ascending = False)
plot_dcm(certain_cases.PatientID[0:12])


# In[ ]:


certain_cases = train.sort_values('subarachnoid', ascending = False)
plot_dcm(certain_cases.PatientID[0:12])


# In[ ]:


certain_cases = train.sort_values('intraventricular', ascending = False)
plot_dcm(certain_cases.PatientID[0:12])


# In[ ]:


certain_cases = train.sort_values('intraparenchymal', ascending = False)
plot_dcm(certain_cases.PatientID[0:12])


# As we can see, the difference in hemorrhage types is difficult to discern for the untrained eye. This is because we're looking at the raw images instead of the Hounsfield Units which the dicom images are scaled with. 

# In[ ]:


#https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing

def window_image(img, window_center,window_width, intercept, slope):

    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    return img

def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def new_open_image(path, div=True, convert_mode=None, after_open=None):
    dcm = pydicom.dcmread(str(path))
    window_center, window_width, intercept, slope = get_windowing(dcm)
    im = window_image(dcm.pixel_array, window_center, window_width, intercept, slope)
    im = np.stack((im,)*3, axis=-1)
    im -= im.min()
    im_max = im.max()
    if im_max != 0: im = im / im.max()
    x = Image(pil2tensor(im, dtype=np.float32))
    #if div: x.div_(2048)  # ??
    return x


vision.data.open_image = new_open_image


# In[ ]:


certainly_affected = train.sort_values('any', ascending = False).filter(['PatientID', 'labels'])[:10000]
certainly_affected.reset_index(drop = True, inplace=True)

certainly_unaffected = train.sort_values('any', ascending = True).filter(['PatientID', 'labels'])[:10000]#.PatientID[:10000]
certainly_unaffected.reset_index(drop = True, inplace=True)

train_subset = pd.concat([certainly_affected, certainly_unaffected], axis = 0)
train_subset['PatientID'] += '.dcm'
train_subset


# In[ ]:


batch_size = 128

im_list = ImageList.from_df(train_subset, path = (path + "stage_1_train_images"))
test_fnames = pd.DataFrame("ID_" + pd.read_csv(path + "stage_1_sample_submission.csv")["ID"].str.split("_", n = 2, expand = True)[1].unique() + ".dcm")
test_im_list = ImageList.from_df(test_fnames, path = (path + "stage_1_test_images"))

tfms = get_transforms(do_flip = False)

data = (im_list.split_by_rand_pct(0.2)
               .label_from_df(label_delim=",")
               .transform(tfms, size = 512)
               .add_test(test_im_list)
               .databunch(bs = batch_size, num_workers = 0)
               .normalize())


# In[ ]:


acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)

learn = cnn_learner(data, models.resnet18, metrics = [acc_02, f_score, accuracy_thresh], model_dir = '/kaggle/working/models')

#models_path = Path("/kaggle/working/models")
#if not models_path.exists(): models_path.mkdir()
    
#learn.model_dir = models_path
#learn.metrics = [accuracy_thresh]


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 1e-3
learn.freeze()
learn.fit_one_cycle(3, slice(lr))


# In[ ]:





# In[ ]:


submission = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv')
submission['fn'] = submission.ID.apply(lambda x: '_'.join(x.split('_')[:2]) + '.dcm')
submission['label'] = submission.ID.apply(lambda x: x.split('_')[-1])

pivot_test = submission.pivot(index='fn', columns='label', values='Label')
pivot_test.reset_index(inplace=True)
pivot_test['MultiLabel'] = " "


# In[ ]:


#test_path = "../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/"
data_test = (ImageList
.from_df(path=test_images_path,df= pivot_test[['fn', 'MultiLabel']])
.split_by_rand_pct(valid_pct= 0)
.label_from_df(cols=1, label_delim = " ")
.transform(size=(128,128))
.databunch()
.normalize(imagenet_stats))


# In[ ]:


data_classes = learn.data.classes
learn.data = data_test


# In[ ]:


y_predict, _ = learn.get_preds(ds_type=DatasetType.Fix)


# In[ ]:


assert len(y_predict) == pivot_test.shape[0]


# In[ ]:


pivot_test['MultiLabel']  = y_predict


# In[ ]:


for i, col in enumerate(data_classes):
    print(col, end= ", ")
    pivot_test[col] = pivot_test['MultiLabel'].apply(lambda x: x[i].numpy())
    
cols_to_consider = [col for col in pivot_test.columns if not col in ['None', 'MultiLabel']]
print(cols_to_consider)
df_temp = pd.melt(pivot_test[cols_to_consider], id_vars= ['fn'])


# In[ ]:


df_temp['ID'] = df_temp['fn'].apply(lambda x: x[:-4])
df_temp['ID'] = df_temp[['ID', 'label']].apply(lambda x: "_".join(x), axis = 1)

assert len(set(submission.ID.unique()).intersection(set(df_temp.ID.unique()))) == submission.shape[0]

df_temp.rename(columns={'value':'Label'}, inplace = True)
df_temp[['ID', 'Label']].to_csv("submission1.gz", compression = 'gzip' , index = False)


# In[ ]:


FileLink('submission1.gz')


# In[ ]:




