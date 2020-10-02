#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
print(os.listdir("../input/"))


# In[4]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


import numpy as np
import pandas as pd
from fastai.vision import *
from fastai.metrics import error_rate
from fastai import *


# In[6]:


print("PyTorch version - ", torch.__version__)
print("Cuda version - ", torch.version.cuda)
print("cuDNN version - ", torch.backends.cudnn.version())
print("Device - ", torch.device("cuda:0"))
print("python PIL version - ", PIL.PILLOW_VERSION)


# In[7]:


get_ipython().system('nvidia-smi')


# In[8]:


batch_size = 64


# In[9]:


data_path = "../input/"
data_path_train = data_path + "train/train/"
data_path_test = data_path + "test/test/"


# In[10]:


df_train = pd.read_csv(data_path + "train.csv")
df_test = pd.read_csv(data_path + "sample_submission.csv")


# In[11]:


df_train.head()


# In[12]:


data = ImageDataBunch.from_df(data_path_train, df_train, ds_tfms=get_transforms(), bs=batch_size).normalize(imagenet_stats)


# In[13]:


data.add_test(ImageList.from_df(df_test, path=data_path_test))


# In[14]:


data


# In[15]:


data.show_batch(rows = 3, figsize = (10,8))


# In[16]:


print(data.classes)


# In[17]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/")


# In[19]:


learn.model


# In[20]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# ### Predictions

# In[ ]:


predictions = learn.get_preds(ds_type=DatasetType.Test)[0]


# In[ ]:


predictions[0]


# In[ ]:


predictions[:10]


# In[ ]:


predicted_classes = np.argmax(predictions, axis=1)


# In[ ]:


predicted_classes[:10]


# In[ ]:


df_test['has_cactus'] = predicted_classes
df_test.head(10)


# ### Create Submission File

# In[ ]:


from datetime import datetime
time_format = "%Y%m%d-%H%M%S.%f"
time_stamp = datetime.now().strftime(time_format)
file_path = "{0}submission_{1}.csv".format(data_path, datetime.now().strftime(time_format))
                                        
print("Exporting Submission file with {0} rows at {1}".format(df_test.shape[0], file_path))

df_test.to_csv(file_path, index = False)


# In[ ]:





# In[ ]:




