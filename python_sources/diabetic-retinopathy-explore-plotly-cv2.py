#!/usr/bin/env python
# coding: utf-8

# ## Objective 
# Objective of this kernel is to quickly explore the dataset given and understand the dataset and have an idea about **Diabetic Retinopathy**.
# 
# What is Diabetic Retinopathy?  
# DR is a damage to the **Retina** caused by complications of diabetes mellitus. This condition can lead to blindness if left untreated.
# 
# What exactly is happening to the Retina?  
# DR is damage of blood vessels in the retina that happens due to diabetes.
# 
# What are the symptoms?  
# Common symptoms are blurred vision, color blindness, floaters and complete loss of vision.
# 
# How does a normal eye and an eye affected with DR looks?  
# This image from American Optmetric Association shows the difference between a normal eye and a DR eye.
# ![Normal Eye vs DR Eye](https://www.aoa.org/Images/public/Diabetic_Retinopathy.jpg)  
# 
# What other complications occur along with DR?  
# - **Vitreous hemorrhage** - Leak in new blood vessel
# - **Detached Retina** - Scar tissue that pulls the retina away from back of the eye
# - **Galucoma** - Blockage of the flow of fluid in the eye when new blood vessels form
# 
# Is this curable?  
# Yes to some extent but significant **side-effects** are anticipated in **Proliferative DR** condition.

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo('7cEd2ZrItNg')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.


# ## Prepare Dataset
# Let us prepare the dataset for us to easily navigate across the images.
# - Read the train and test files
# - Create the Diagnostic Label dataframe and merge to train data
# - Enumerate all images and add to the train and test dataset

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
print("Shape of train data: {0}".format(train_df.shape))
test_df = pd.read_csv("../input/test.csv")
print("Shape of test data: {0}".format(test_df.shape))

diagnosis_df = pd.DataFrame({
    'diagnosis': [0, 1, 2, 3, 4],
    'diagnosis_label': ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
})

train_df = train_df.merge(diagnosis_df, how="left", on="diagnosis")

train_image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("../input/train_images")) for f in fn]
train_images_df = pd.DataFrame({
    'files': train_image_files,
    'id_code': [file.split('/')[3].split('.')[0] for file in train_image_files],
})
train_df = train_df.merge(train_images_df, how="left", on="id_code")
del train_images_df
print("Shape of train data: {0}".format(train_df.shape))

test_image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("../input/test_images")) for f in fn]
test_images_df = pd.DataFrame({
    'files': test_image_files,
    'id_code': [file.split('/')[3].split('.')[0] for file in test_image_files],
})


test_df = test_df.merge(test_images_df, how="left", on="id_code")
del test_images_df
print("Shape of test data: {0}".format(test_df.shape))


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


print("Number of unique diagnosis: {0}".format(train_df.diagnosis.nunique()))
diagnosis_count = train_df.diagnosis.value_counts()


# In[ ]:


import matplotlib.pyplot as plt
from PIL import Image

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 9999
pd.options.display.float_format = '{:20, .2f}'.format


# ## Sample Distribution by Severity

# In[ ]:


def render_bar_chart(data_df, column_name, title, filename):
    series = data_df[column_name].value_counts()
    count = series.shape[0]
    
    trace = go.Bar(x = series.index, y=series.values, marker=dict(
        color=series.values,
        showscale=True
    ))
    layout = go.Layout(title=title)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename=filename)
    
    
render_bar_chart(train_df, 'diagnosis_label', 'Diabetic Retinopathy: Observation Distribution by Severity ', 'members')


# ## No DR
# Examine the dataset and identify whether we can figure out any significant differences across the various diagnosis of DR images.  
# Let us pick some images randomly and closely watch. 

# In[ ]:


SAMPLES_TO_EXAMINE = 5
import cv2
def render_images(files):
    plt.figure(figsize=(50, 50))
    row = 1
    for an_image in files:
        image = cv2.imread(an_image)[..., [2, 1, 0]]
        plt.subplot(6, 5, row)
        plt.imshow(image)
        row += 1
    plt.show()
    
no_dr_pics = train_df[train_df["diagnosis"] == 0].sample(SAMPLES_TO_EXAMINE)
render_images(no_dr_pics.files.values)


# ## Mild DR
# I couldn't distinguish anything between no DR and early stages of DR images with my naked eye.

# In[ ]:


mild_dr_pics = train_df[train_df["diagnosis"] == 1].sample(SAMPLES_TO_EXAMINE)
render_images(mild_dr_pics.files.values)


# ## Moderate DR
# Images where DR was diagnosed as moderate one, I could clearly see some patches and bright spots in almost all images. However the blood vessel appearance remains almost same.

# In[ ]:


moderate_dr_pics = train_df[train_df["diagnosis"] == 2].sample(SAMPLES_TO_EXAMINE)
render_images(moderate_dr_pics.files.values)


# ## Severe DR
# In sever DR diagnostic conditions, blood vesses are quite significant and seen densely. Bright patches are much more evident.

# In[ ]:


severe_dr_pics = train_df[train_df["diagnosis"] == 3].sample(SAMPLES_TO_EXAMINE)
render_images(severe_dr_pics.files.values)


# ## Proliferative DR
# In Proliferative DR images, the images are significantly different from the other 4 classes. There is a haziness and dull. In some of the images, it looks blood is oozing out from the vessels... bit scary.

# In[ ]:


preoliferative_dr_pics = train_df[train_df["diagnosis"] == 4].sample(SAMPLES_TO_EXAMINE)
render_images(preoliferative_dr_pics.files.values)


# In[ ]:





# In[ ]:




