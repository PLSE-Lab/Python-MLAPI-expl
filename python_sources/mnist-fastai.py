#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import fastai
from fastai.vision import *


# In[ ]:


path = Path('../input')


# In[ ]:


train = pd.read_csv(path/'train.csv', header='infer')
train.head()


# In[ ]:


test = pd.read_csv(path/'test.csv', header='infer')
test.head()


# ### The original methods have problems reading the dataframe, so we have to customize some functions.

# In[ ]:


class CustomImageList(ImageList):
    def open(self, fn):
        """Replace the original open method"""
        fn = fn.reshape(28,28)
        fn = np.stack((fn,)*3, axis=-1)
        return Image(pil2tensor(fn,dtype=np.float32))
    
    @classmethod
    def from_df_custom(cls, df, path:PathOrStr, **kwargs) ->'ItemList':
        res = super().from_df(df, path=path, cols=0, **kwargs)
        if 'label' in df.columns:
            df = df.drop('label', axis=1)
        df = np.array(df,dtype=np.float32)/255.
        df = df/255.
        mean = df.mean()
        std = df.std()
        res.items = (df-mean)/std
        return res


# ### Preprocess the data and then create DataBunch 

# In[ ]:


test_data = CustomImageList.from_df_custom(test,path=path)
test_data


# In[ ]:


tfms = get_transforms(do_flip=False)
data = (CustomImageList.from_df_custom(train, path=path)
        .split_by_rand_pct(.2, seed=2019)
        .label_from_df(cols='label')
        .add_test(test_data, label=0)
        .transform(tfms)
        .databunch(bs=128, num_workers=0)
        .normalize(imagenet_stats))


# In[ ]:


data.show_batch(2, figsize=(6,6))


# ### Choose an architecture for the model.

# In[ ]:


learner = cnn_learner(data, models.resnet50,metrics=accuracy,model_dir='/kaggle/working/models')


# In[ ]:


learner.summary()


# ### Plot the curve so that we can find the best learning rate for this model.

# In[ ]:


learner.lr_find()


# In[ ]:


learner.recorder.plot()


# In[ ]:


learner.fit_one_cycle(7, 1e-2)


# In[ ]:


learner.unfreeze()


# In[ ]:


learner.lr_find()
learner.recorder.plot()


# In[ ]:


learner.fit_one_cycle(8, slice(1e-5,1e-4))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learner)
interp.plot_top_losses(9, figsize=(7,7))


# In[ ]:


learner.save('stage1')


# In[ ]:


learner.recorder.plot_losses()


# In[ ]:


learner.lr_find()
learner.recorder.plot()


# ### Try different learning rates on sequencial layers

# In[ ]:


learner.unfreeze()
learner.fit_one_cycle(8, slice(1e-5,1e-4))


# In[ ]:


learner.recorder.plot_losses()


# In[ ]:


learner.save('MnistStage2')


# In[ ]:


learner.show_results()


# In[ ]:


pred, y, losses = learner.get_preds(ds_type=DatasetType.Test, with_loss=True)


# In[ ]:


labels = torch.argmax(pred, dim=1)


# In[ ]:


submission_df = pd.DataFrame({'ImageId': range(1, len(y) + 1), 'Label': labels}, columns=['ImageId', 'Label'])
submission_df.head()


# In[ ]:


submission_df.to_csv('MnistSubmission.csv', index=False)

