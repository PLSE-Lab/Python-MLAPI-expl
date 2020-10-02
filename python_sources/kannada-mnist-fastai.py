#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from fastai import *
from fastai.vision import *
import imageio


# In[ ]:


path = Path('../input/Kannada-MNIST')
train = pd.read_csv('../input/Kannada-MNIST/train.csv')
test  =pd.read_csv('../input/Kannada-MNIST/test.csv')


# In[ ]:


df=pd.read_csv('../input/Kannada-MNIST/train.csv')


# In[ ]:


df.head()


# In[ ]:


df_test=pd.read_csv('../input/Kannada-MNIST/test.csv')


# In[ ]:


class CustomImageItemList(ImageList):
    def open(self, fn):
        img = fn.reshape(28, 28)
        img = np.stack((img,)*3, axis=-1) # convert to 3 channels
        return Image(pil2tensor(img, dtype=np.float32))

    @classmethod
    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs) -> 'ItemList':
        df = pd.read_csv(Path(path)/csv_name, header=header)
        res = super().from_df(df, path=path, cols=0, **kwargs)
        # convert pixels to an ndarray
        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 783.0, axis=1).values
        return res


# In[ ]:


test = CustomImageItemList.from_csv_custom(path=path, csv_name='test.csv', imgIdx=1)
# DigMNIST = CustomImageItemList.from_csv_custom(path=path, csv_name='Dig-MNIST.csv')
data = (CustomImageItemList.from_csv_custom(path=path, csv_name='train.csv')
                       .split_by_rand_pct(.2)
#                       .split_by_idx(list(range(60000,70240)))
                       .label_from_df(cols='label')
                       .add_test(test, label=0)
                       .transform(get_transforms(do_flip = False))
#                        .transform(get_transforms(do_flip = False), size=49)
                       .databunch(bs=32, num_workers=16))
#                        .normalize(mnist_stats))
data


# In[ ]:


df2 = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')


# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=[error_rate, accuracy], model_dir="../output/kaggle/working/", pretrained=False)


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


#learn1.lr_find()
#learn1.recorder.plot(suggestion=True)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix(figsize=(10,10), dpi=60)


# In[ ]:


test


# In[ ]:


test.drop('id',axis = 'columns',inplace = True)


# In[ ]:


tmp_df = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
tmp_df.head()


# In[ ]:


for i in range(5000):
    img = learn.data.test_ds[i][0]
    tmp_df.loc[i]=[i,int(learn.predict(img)[1])]
tmp_df


# In[ ]:


tmp_df.to_csv('submission.csv',index=False)


# In[ ]:




