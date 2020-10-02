#!/usr/bin/env python
# coding: utf-8

# # Using Deep Learning to detect COVID-19 presence from x-ray scans

# Disclaimer: I am not a doctor nor a medical researcher. This work is only intended as a source of inspiration for further studies.

# ## COVID-19-xray
# 

# The more the pandemic crisis progresses, the more it gets important that countries perform tests to help understand and stop the spread of COVID-19.
# Unfortunately, the capacity for COVID-19 testing is still low in many countries. 

# ### How are tests performed?

# The standard COVID-19 tests are called PCR (Polymerase chain reaction) tests. This family of tests looks for the existence of antibodies of a given infection. Two main issues with this test are:
# 
# 1. a shortage a tests available worldwide
# 2. a patient might be carring the virus without having symptoms. In this case the test fails to identify infected patients

# [Dr. Joseph Paul Cohen, Postdoctoral Fellow at University of Montreal](https://josephpcohen.com/w/), recently open sourced a [database](https://github.com/ieee8023/covid-chestxray-dataset) containing chest x-ray pictures of patients suffering from the COVID-19 disease. 
# 

# The database only contains pictures of patients suffering from COVID-19. In order to build a classifier for xray images we first need to find similar x-ray images of people who are not suffering from the disease.
# It turns out Kaggle has a database with chest x-ray images of patients suffering of pneumonia and healthy patients. Hence, we are going to use both sources images in our dataset.
# 

# The notebook is organized as follows:
# 
# 1. Data Preparation  
# 
# 2. Train Network using Fastai
# 
# 3. Optimize Network
# 
# 4. Test
# 
# 5. What's Next

# But first let's import necessary libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# # 1. Data Preparation

# Let's import Fastai, create useful paths and create covid_df

# In[ ]:


from fastai import *
from fastai.vision import *

# useful paths
input_path = Path('/kaggle/input') 
covid_xray_path = input_path/'xraycovid'
pneumonia_path = input_path/'chest-xray-pneumonia/chest_xray'

covid_df = pd.read_csv(covid_xray_path/'metadata.csv')
covid_df.head()


# We notice straight away that we have a large number of NaN, let's remove them and see what we are left with

# In[ ]:


covid_df.dropna(axis=1,inplace=True)
covid_df


# That looks better. We are mainly interested in two columns: ```finding``` and ```filename```. The former tells us wether or not a patient is suffering from the virus whereas the latter tells us the finename. The other interesting column is ```view```. It turns out the view is the angle used when the scan is taken and the most frequently used is PA. PA view stands for Posteroanterior view.

# In[ ]:


covid_df.groupby('view').count()


# In[ ]:


covid_df.groupby('finding').count()


# PA makes up the majority of the datapoints. Let's keep them and remove the rest.

# In[ ]:


covid_df = covid_df[lambda x: x['view'] == 'PA']
covid_df


# For simplicity, let's also rename the elements in column ```finding``` to be ```positive``` if the patient is suffering from COVID-19 and negative otherwise.

# In[ ]:


covid_df['finding'] = covid_df['finding'].apply(lambda x:'positive' if x == 'COVID-19' else 'negative')
covid_df


# Finally, let's replace the ```filename``` column by the entire system path and keep only the two columns we are more interested in

# In[ ]:


def makeFilename(x = ''):
    return covid_xray_path/f'images/{x}'

covid_df['filename'] = covid_df['filename'].apply(makeFilename)
covid_df = covid_df[['finding', 'filename']]
covid_df


# We now need to create a dataframe of the same format using the pictures from the other database. Once we have that dataframe, we can use the mighty [ImageDataBunch](https://docs.fast.ai/vision.data.html) methods to create a dataset that we can feed to our convolutional network.  
# 
# Since our second database is made up of pictures of both healthy patients and pneumonia suffering patients, we are going to take an equal mix of both. I tried using only images of healthy people from this database but I reflected that since COVID-19 and pneumonia are linked somehow then it might give our network an edge to also contain pneumonia x-rays.
# 
# This is what our ```pneumonia_df``` looks like:

# In[ ]:



pneumonia_df = pd.DataFrame([], columns=['finding', 'filename'])
folders = ['train/NORMAL', 'val/NORMAL', 'test/NORMAL']
for folder in folders:
    fnames = get_image_files(pneumonia_path/folder)
    fnames = map(lambda x: ['negative', x], fnames)
    df = pd.DataFrame(fnames, columns=['finding', 'filename'])
    pneumonia_df = pneumonia_df.append(df, ignore_index = True)



folders = ['train/PNEUMONIA', 'val/PNEUMONIA', 'test/PNEUMONIA']
for folder in folders:
    fnames = get_image_files(pneumonia_path/folder)
    fnames = map(lambda x: ['negative', x], fnames)
    df = pd.DataFrame(fnames, columns=['finding', 'filename'])
    pneumonia_df = pneumonia_df.append(df, ignore_index = True)

pneumonia_df.info()


# As you can see we have 5856 pictures which is about 60 times larger than our covid_df.  
# 
# Since we have 92 pictures in our covid_df, I decided to take an equal number of pictures of healthy patients and an equal number of picture of pneumonia patients. In other words, 92 covid_df images, 92 healthy patient images, and 92 pneumonia affected patients. As far as our analysis goes, we are really only interested in covid positive and covid negative. Therefore, both the healthy and pneumonia patients will be labeled as ```negative```

# NB: Following great suggestions, I received, I am gonna run the Convolutional Net on two dataBunch:
# 
# - The first will have covid_df and healthy_df
# - The second one will have covid_df and pneumonia_df
# 
# We will then compare the perfomances and, hopefully, we will get comparable results so that we can have more confidence in our results.
# 

# In[ ]:



healthy_df = pd.DataFrame([], columns=['finding', 'filename'])
folders = ['train/NORMAL', 'val/NORMAL', 'test/NORMAL']
for folder in folders:
    fnames = get_image_files(pneumonia_path/folder)
    fnames = map(lambda x: ['negative', x], fnames)
    df = pd.DataFrame(fnames, columns=['finding', 'filename'])
    healthy_df = healthy_df.append(df, ignore_index = True)
    
pneumonia_df = pd.DataFrame([], columns=['finding', 'filename'])
folders = ['train/PNEUMONIA', 'val/PNEUMONIA', 'test/PNEUMONIA']
for folder in folders:
    fnames = get_image_files(pneumonia_path/folder)
    fnames = map(lambda x: ['negative', x], fnames)
    df = pd.DataFrame(fnames, columns=['finding', 'filename'])
    pneumonia_df = pneumonia_df.append(df, ignore_index = True)

pneumonia_df = pneumonia_df.sample(covid_df.shape[0]).reset_index(drop=True)

healthy_df = healthy_df.sample(covid_df.shape[0]).reset_index(drop=True)

negative_df = healthy_df.append(pneumonia_df, ignore_index = True)

negative_df.head()


# Now, we can finally merge our dataframes to get the dataframe needed to build our [ImageDataBunch](https://docs.fast.ai/vision.data.html).

# In[ ]:


df = covid_df.append(healthy_df, ignore_index = True)
df = df.sample(frac=1).reset_index(drop=True)
df.sample(20)


# # 2. Train Network using Fastai

# I am going to run the Convolutional Net using two training sets.
# The first will have  covid_df and healthy_df
# The second one will have covid_df and pneumonia_df
# 
# We will then compare the perfomances and, hopefully, we will get comparable results so that we can have more confidence in our results.

# ## First case: COVID-19 patients and healthy patients

# We are now ready to create the ImageDataBunch.

# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_df(
        '/', 
        df, 
        fn_col='filename',
        label_col='finding',
        ds_tfms=get_transforms(), ## data augmentation: flip horizozntally
        size=224, 
        num_workers=4
    ).normalize(imagenet_stats)

data


# Let's take random images from ```data``` to see if they look consistent.

# In[ ]:


data.show_batch(rows=80, figsize=(21,21))


# To my untrained eyes, it looks like the images look consistent. We are going to use a resnet50 and leverage Kaggle free GPU Quota. Let's start training ten cycles.

# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate)


# Let's first fit 10 cycles and see how it improves

# In[ ]:


learn.fit_one_cycle(5)


# Looks like we can do better, let's run ten cycles more.

# In[ ]:


learn.fit_one_cycle(5)


# It looks good. We are going to keep the 5.5% error for now and try the next data frame
# 

# In[ ]:


learn.save('stage-1')


# ## Second case: COVID-19 patients and pneumonia patients

# df2 = covid_df.append(pneumonia_df, ignore_index = True)
# df2 = df2.sample(frac=1).reset_index(drop=True)
# np.random.seed(42)
# data2 = ImageDataBunch.from_df(
#         '/', 
#         df2, 
#         fn_col='filename',
#         label_col='finding',
#         ds_tfms=get_transforms(), ## data augmentation: flip horizozntally
#         size=224, 
#         num_workers=4
#     ).normalize(imagenet_stats)
# 
# learn2 = cnn_learner(data2, models.resnet50, metrics=error_rate)
# learn2.fit_one_cycle(5)

# Let's run some more cycles

# learn2.fit_one_cycle(10)

# learn2.fit_one_cycle(10)

# That's a nice error rate. Let's save ```learn2``` and starti optimizing

# learn2.save('learn2-stage-1')
# 

# # 3. Optmize

# Results for the first case were already pretty solid in **Part 2**. We are going to first optimize the results for the first case and then optimize results for the second case. Then, if the two cases accuracy do not differ too much, we will be confident in our result and try to predict random images online.

# In[ ]:


learn.load('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


#  The longest downward shape is found in the region around ```1e-4``` let's use that as our starting point

# In[ ]:


learn.fit_one_cycle(10, max_lr=slice(7e-5,2e-4))


# I obtained the 0% error rate after a updated my notebook on kaggle and used a balanced dataset. This error rate though, is probably due to the fact that I am still collecting data and would require much more images to have an more stable error rate.

# In[ ]:


learn.save('stage-2')


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(7e-5,2e-4))


# Looks like the error rate is not really moving. With 3.6% error rate we might be satisfied with this first results. We are going to save and plot the confusion matrix.

# In[ ]:


learn.load('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# Since the error rate is 0, the confusion matrix shows we have no errors.

# ## Second case: COVID-19 patients and pneumonia patients

# learn2.load('learn2-stage-1')
# 
# learn2.unfreeze()
# 
# learn2.lr_find()
# 
# learn2.recorder.plot()
# 

#  The longest downward shape is found in the region around ???????????```1e-4``` let's use that as our starting point

# learn2.load('learn2-stage-1')
# learn2.unfreeze()
# learn2.fit_one_cycle(4, max_lr=slice(7e-5,1e-4))

# learn2.save('learn2-stage-2')

# Both cases have an error rate < 3%. Given the scarsity of data, this is a promising first result. Since using both models, covid-19 prediction seems to be consistent, we can be confident enough in its predictions.

# # Test

# In[ ]:


# learn = cnn_learner(data, models.resnet50, metrics=error_rate)
# learn.load('stage-3')
# interp = ClassificationInterpretation.from_learner(learn)
img = open_image(input_path/'testimg/df1053d3e8896b53ef140773e10e26_gallery.jpeg')
learn.predict(img)


# That image is taken from https://radiopaedia.org/images/52197348 and it is an image of a positive patient.

# # What's next?

# First of all, I would like to incorporate scans from other sources and see if accuracy and generalization might increase.
# Today, while I was about to pusblish this article, I found out that [MIT](https://www.technologyreview.com/s/615399/coronavirus-neural-network-can-help-spot-covid-19-in-chest-x-ray-pneumonia/) has released a database containing xrays images of covid patients. Next, I am going to incorporate MIT's database and see where we get.

# In[ ]:




