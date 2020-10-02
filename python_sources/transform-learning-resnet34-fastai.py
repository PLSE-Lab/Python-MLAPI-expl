#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         os.path.join(dirname, filename)
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import os


# In[ ]:


from fastai.vision import *


# In[ ]:


data_path = Path('/kaggle/input/intel-image-classification')
data_path.ls()


# In[ ]:


# get image data using fastai's data block api
data_sz64 = ImageList.from_folder(data_path).split_by_folder(train='seg_train', valid='seg_test').label_from_folder().add_test_folder(test_folder='seg_pred').transform(get_transforms(), size=64).databunch().normalize(imagenet_stats)
learner_sz64 = cnn_learner(data_sz64, models.resnet34, metrics=[accuracy, error_rate])
learner_sz64.fit_one_cycle(16)


# In[ ]:


# get image data using fastai's data block api
data = ImageList.from_folder(data_path).split_by_folder(train='seg_train', valid='seg_test').label_from_folder().add_test_folder(test_folder='seg_pred').transform(get_transforms(), size=150).databunch().normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,7))


# In[ ]:


print(data.c, data.classes)


# **Model with defaults**

# In[ ]:


# create CNN using resnet model & train the model with defaults
learner_default = cnn_learner(data, models.resnet34, metrics=[accuracy, error_rate])
learner_default.fit_one_cycle(16)


# **Note: With change in transforms 'size' attribute, accuracy improved by ~3%**
# 
# With 10 epochs, we got an accuracy of ~93.33%.
# 
# I think, from 11th cycle, the error rate is kind of increasing (varying upwards) & towards the end, we ran into overfitting.

# In[ ]:


learner_default.recorder.plot_losses()


# ***Default learning rate 1e-3 looks good - as the loss curve is quickly on downside***

# **Experimenting & fine tuning**

# In[ ]:


np.random.seed(4)
learner = cnn_learner(data, models.resnet34, metrics=[accuracy, error_rate])


# In[ ]:


# train only the weights of newly introduced layers, using default learning_rate
learner.freeze()
learner.fit_one_cycle(8)


# **Why freeze? What does it do?**
# 
# All thanks to Jeremy & the fastai course, for very nice explaination on this.
# 
# When CNN model is created, we are using resnet34 architecture pretrained for imagenet. The data & the classification targetted here need not be same as that of in imagenet. So, when creating the new model, the fastai library identifies the number of classes involved and introduces 2 new layers (towards the end), with new random weights.
# 
# freeze() ensures that only the weights of new layers are updated, but not those of existing resnet layers.

# Key things to note:
# 1. train_loss > valid_loss => underfitting
# 2. error_rate improving with each epoch => maybe, we can try running more epochs to make it better & closely monitor train_loss, valid_loss & error_rate

# In[ ]:


# save model
learner.model_dir = '/kaggle/working/'
learner.save('learner_v1')


# In[ ]:


learner.recorder.plot_losses()


# In[ ]:


# let us try experimenting more on our model, but from the initially training model, 'learner_v1' to save the computational time
# create new CNN model & load a previous model version
learner = cnn_learner(data, models.resnet34, metrics=[accuracy, error_rate])
learner.load('/kaggle/working/learner_v1')


# In[ ]:


# here, now that the newly added layers are having proper weights, we are trying to train all the layers in resnet model
learner.unfreeze()
learner.fit_one_cycle(5)


# **Note: Improved accuracy when compared to learner_v1**

# In[ ]:


learner.model_dir='/kaggle/working/'
learner.save('learner_v2')


# **Look into model results**

# In[ ]:


interp = ClassificationInterpretation.from_learner(learner)


# In[ ]:


interp.plot_top_losses(9, figsize=(10, 10))


# In[ ]:


interp.plot_confusion_matrix(figsize=(10,10), dpi=75)


# In[ ]:


interp.most_confused(min_val=2)


# **Predicting on seg_pred data using 'learner_v2**

# In[ ]:


learner.export('/kaggle/working/export.pkl')


# In[ ]:


pred_list = ImageList.from_folder(data_path/'seg_pred')
learner_inference = load_learner('/kaggle/working/', test=pred_list)


# In[ ]:


preds, _ = learner_inference.get_preds(ds_type=DatasetType.Test)


# In[ ]:


import random
rand_max = len(preds)
for i in range(10):
    j = random.randint(0, rand_max)
    pred_name = data.classes[np.argmax(np.array(preds[j]))]
    print(pred_name, '--', preds[j])
    show_image(pred_list[j])

