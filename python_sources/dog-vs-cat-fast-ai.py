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


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import zipfile

# Will unzip the files so that you can see them..../input/dogs-vs-cats/test1.zip
with zipfile.ZipFile("../input/dogs-vs-cats/test1.zip","r") as z:
    z.extractall(".")

with zipfile.ZipFile("../input/dogs-vs-cats/train.zip","r") as z:
    z.extractall(".")


# # Understanding the data

# In[ ]:


import cv2
import matplotlib.pyplot as plt
path = "/kaggle/working/train"
image_name = os.listdir(path)
image_name[:10]


# In[ ]:


img = cv2.imread(os.path.join(path, image_name[7]))
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
print(img.shape)
plt.imshow(img)


# # Implementing Fast.Ai library

# In[ ]:


from fastai.vision import *


# In[ ]:


path = Path('/kaggle/working/train')
path


# In[ ]:


fn_paths = [path/name for name in image_name]; fn_paths[:2]
#str(fn_path)
#fn_path = [path/image_name]
category = str(fn_paths[17]).split('.')[0]
category[22:]
#if category == 'dog':
#    print(1)
#else:
#    print(0)


# In[ ]:


tfms = get_transforms(do_flip=True)
bs = 64


# # **The dataset requires that we classify all dogs to 1 and  cats to 0**

# In[ ]:


def get_labels(filename):
    category = str(filename).split('.')[0]
    if category[22:] == 'dog':
        return 1
    else:
        return 0


# In[ ]:


data = ImageDataBunch.from_name_func(path, fn_paths, label_func=get_labels,valid_pct=0.2, ds_tfms=tfms, size=224,bs=bs)
data = data.normalize(imagenet_stats)


# In[ ]:


data.show_batch(rs=3, figsize=(7,8))


# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# # Creating a pretrained Resnet-34 DNN that was trained on imagenet dataset

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# # Training the head network to identify dogs and cats

# In[ ]:


learn.fit_one_cycle(4)
learn.recorder.plot_losses()


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.load('stage-1');


# In[ ]:


learn.unfreeze()
learn.lr_find()


# In[ ]:


learn.recorder.plot()
learn.recorder.plot_lr()


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(1e-5,2e-4))


# In[ ]:


learn.save('stage-2')


# In[ ]:


learn.load('stage-2');


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# # Obsevering the Losses

# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.plot_top_losses(9, figsize = (15,11))


# In[ ]:


path = Path('/kaggle/working/train')
learn.export()
learn = load_learner(path)
#path = '/kaggle/working/test1'
#learn = load_learner(learn.load('stage-2'),test=ImageItemList.from_folder(path))
#preds = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


test_path = Path('/kaggle/working/test1')
learn = load_learner(path, test=ImageList.from_folder(test_path))


# In[ ]:


preds = learn.get_preds(ds_type=DatasetType.Test)
preds


# In[ ]:


y = np.argmax(to_np(preds[0]),axis=1)


# # Comparison of predicted output with the input of the test data

# In[ ]:


n= 0
y[n]


# In[ ]:


img = ImageList.from_folder(test_path)
#learn.predict(img)
img[n]


# In[ ]:


test_img_name = os.listdir(test_path)
test_img_name[:10]


# # Submission

# In[ ]:


ID = pd.Series(test_img_name)
label = pd.Series(y)
submit = pd.DataFrame({ 'id': ID, 'label': label })
submit.head()


# In[ ]:


submit.to_csv('submission.csv', index=False)

