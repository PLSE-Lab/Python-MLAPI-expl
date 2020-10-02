#!/usr/bin/env python
# coding: utf-8

# ### Load the libraries

# In[ ]:


from fastai import *
from fastai.vision import *
from torchvision.models import * 

import os
import matplotlib.pyplot as plt


# ### Data loading and exploration

# In[ ]:


path = Path("../input/stanford-dogs-dataset/")
path


# In[ ]:


path.ls()


# In[ ]:


# path_anno = path/'annotations/Annotations'
path_img = path/'images/Images/'

# path_anno
path_img


# In[ ]:


# , classes=data.classes[:20]


# In[ ]:


path_img.ls()


# In[ ]:


tfms = get_transforms()
# data = ImageDataBunch.from_folder(path_img ,train='.', valid_pct = 0.2,ds_tfms = tfms , size = 227)


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=227, num_workers=0).normalize(imagenet_stats)


# ### Classyfying 20 categories

# In[ ]:


data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=227, num_workers=0, classes=data.classes[:20]).normalize(imagenet_stats)


# In[ ]:


# data.normalize(imagenet_stats)


# ### Displaying the data

# In[ ]:


data.show_batch(rows = 3 ,figsize = (7,6))


# In[ ]:


print(data.classes)
len(data.classes), data.c # data.c = for classification problems its number of classes


# ### Train Resnet34 model

# In[ ]:


learn = create_cnn(data , models.resnet34, metrics = error_rate) 


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.model_dir = "/kaggle/working"
learn.save('stage-1')


# ### Improving the model by looking at the learning rate 

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4, max_lr=slice(3e-6,3e-5))


# In[ ]:


learn.model_dir = "/kaggle/working"
learn.save('stage-2')


# In[ ]:


learn.load('stage-2');


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(9 , figsize = (15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize = (12,12), dpi = 60)


# In[ ]:


interp.most_confused(min_val = 2) # useful tool


# In[ ]:


data.classes


# ### Predict the breed using the trained model

# In[ ]:


img = open_image('../input/test-cc/japanese_spaniel.jpg')
img


# In[ ]:


classes = data.classes
data2 = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = create_cnn(data2 , models.resnet34)
learn.load('/kaggle/working/stage-2')


# In[ ]:


pred_class,pred_idx,outputs = learn.predict(img)
pred_class


# In[ ]:


prediction = str(pred_class)
prediction[10:]
print("The predicted breed is " + prediction[10:] + '.')


# In[ ]:





# In[ ]:





# In[ ]:




