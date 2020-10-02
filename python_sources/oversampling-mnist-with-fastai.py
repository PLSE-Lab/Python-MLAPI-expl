#!/usr/bin/env python
# coding: utf-8

# # Oversampling MNIST with Fastai
# 
# This kernel highlights the usefulness of oversampling for imbalanced datasets. I use fastai callbacks to oversample data during training. I will train on the full dataset, then an imbalanced dataset, and then an oversampled version of the imbalanced dataset.
# 
# We will see that the oversampled version will get improved performance (both on training set and on public leaderboard)

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.vision import *
from fastai.metrics import *

import os
path = '../input'
print(os.listdir(path))


# ## Load data
# 
# Since the data are represented as rows in a csv file, a custom `ImageList` is necessary to be able to properly open the data (adapted from [this](https://www.kaggle.com/steventesta/digit-recognizer-fast-ai-custom-databunch) kernel):

# In[ ]:


class CustomImageList(ImageList):
    def open(self, fn):
        img = fn.reshape(28,28)
        img = np.stack((img,)*3, axis=-1)
        return Image(pil2tensor(img, dtype=np.float32))
    
    @classmethod
    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList': 
        df = pd.read_csv(Path(path)/csv_name, header=header)
        res = super().from_df(df, path=path, cols=0, **kwargs)
        
        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values
        
        return res
    
    @classmethod
    def from_df_custom(cls, path:PathOrStr, df:DataFrame, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList': 
        res = super().from_df(df, path=path, cols=0, **kwargs)
        
        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values
        
        return res


# Let's create our DataBunch.

# In[ ]:


test = CustomImageList.from_csv_custom(path=path, csv_name='test.csv', imgIdx=0)


# In[ ]:


data = (CustomImageList.from_csv_custom(path=path, csv_name='train.csv', imgIdx=1)
                .split_by_rand_pct(.2)
                .label_from_df(cols='label')
                .add_test(test, label=0)
                .transform(get_transforms(do_flip=False))
                .databunch(bs=128, num_workers=0)
                .normalize(imagenet_stats))


# In[ ]:


data.show_batch(rows=3, figsize=(5,5))


# ## Train Original Model

# In[ ]:


learn = cnn_learner(data, models.resnet18, metrics=[accuracy], model_dir='/kaggle/working/models')


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(4,max_lr=1e-2)


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(10,max_lr = slice(1e-6,1e-4))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# As we can see, we are able to get effectively perfect accuracy on the validation set. Let's create a submission of our model:

# In[ ]:


# get the predictions
predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)
# output to a file
submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})
submission_df.to_csv(f'submission_orig.csv', index=False)


# # Creating imbalanced dataset
# 
# Now let's create an imbalanced version of the MNIST dataset. The training dataset will be imbalanced and the validation dataset will be the same.

# In[ ]:


train_df = pd.read_csv(path+'/train.csv')
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train_df,test_size=0.2) # Here we will perform an 80%/20% split of the dataset, with stratification to keep similar distribution in validation set


# Current distribution:

# In[ ]:


train_df['label'].hist(figsize = (10, 5))


# In[ ]:


proportions = pd.DataFrame({0: [0.5],
                            1: [0.05],
                            2: [0.1],
                            3: [0.03],
                            4: [0.03],
                            5: [0.03],
                            6: [0.03],
                            7: [0.5],
                            8: [0.5],
                            9: [0.5],
                           })


# In[ ]:


imbalanced_train_df = train_df.groupby('label').apply(lambda x: x.sample(frac=proportions[x.name]))


# New distribution:

# In[ ]:


imbalanced_train_df['label'].hist(figsize = (10, 5))


# In[ ]:


df = pd.concat([imbalanced_train_df,val_df])


# Let's create our DataBunch:

# In[ ]:


data = (CustomImageList.from_df_custom(df=df,path=path, imgIdx=1)
                .split_by_idx(range(len(imbalanced_train_df)-1,len(df)))
                .label_from_df(cols='label')
                .add_test(test, label=0)
                .transform(get_transforms(do_flip=False))
                .databunch(bs=128, num_workers=0)
                .normalize(imagenet_stats))


# In[ ]:


data.show_batch(rows=3, figsize=(5,5))


# As you can see, the images are predominately zeroes, sevens, eights and nines.

# ## Train model on imbalanced data

# In[ ]:


learn = cnn_learner(data, models.resnet18, metrics=[accuracy], model_dir='/kaggle/working/models')


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(4,max_lr=1e-2)


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(10,max_lr = slice(1e-6,5e-4))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# There is significantly less accuracy for the same set-up. Let's create a submission.

# In[ ]:


# get the predictions
predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)
# output to a file
submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})
submission_df.to_csv(f'submission_imbalanced.csv', index=False)


# ## Train on imbalanced dataset with oversampling
# 
# I will first show how we can use the Weighted Random Sampler in PyTorch to implement oversampling. We will then implement a callback for fastai that will perform oversampling of the dataset.

# Currently the sampler is a random sampler:

# In[ ]:


data.train_dl.dl.sampler


# In[ ]:


labels = []
for img,target in data.train_dl.dl:
    labels.append(target)
labels = torch.cat(labels)
plt.hist(labels)


# If we instead use a weighted random sampler with weights that are inverse of the counts of the labels, we can get a relatively balanced distribution.

# In[ ]:


np.bincount([data.train_dl.dataset.y[i].data for i in range(len(data.train_dl.dataset))])


# In[ ]:


type(np.max(np.bincount([data.train_dl.dataset.y[i].data for i in range(len(data.train_dl.dataset))])))


# In[ ]:


from torch.utils.data.sampler import WeightedRandomSampler

train_labels = data.train_dl.dataset.y.items
_, counts = np.unique(train_labels,return_counts=True)
class_weights = 1./counts
weights = class_weights[train_labels]
label_counts = np.bincount([learn.data.train_dl.dataset.y[i].data for i in range(len(learn.data.train_dl.dataset))])
total_len_oversample = int(learn.data.c*np.max(label_counts))
data.train_dl.dl.batch_sampler = BatchSampler(WeightedRandomSampler(weights,total_len_oversample), data.train_dl.batch_size,False)


# In[ ]:


labels = []
for img,target in data.train_dl:
    labels.append(target)
labels = torch.cat(labels)
plt.hist(labels)


# We can now create a callback which can be passed to the `Learner`.

# In[ ]:


class OverSamplingCallback(LearnerCallback):
    def __init__(self,learn:Learner):
        super().__init__(learn)
        self.labels = self.learn.data.train_dl.dataset.y.items
        _, counts = np.unique(self.labels,return_counts=True)
        self.weights = torch.DoubleTensor((1/counts)[self.labels])
        self.label_counts = np.bincount([self.learn.data.train_dl.dataset.y[i].data for i in range(len(self.learn.data.train_dl.dataset))])
        self.total_len_oversample = int(self.learn.data.c*np.max(self.label_counts))
        
    def on_train_begin(self, **kwargs):
        self.learn.data.train_dl.dl.batch_sampler = BatchSampler(WeightedRandomSampler(weights,self.total_len_oversample), self.learn.data.train_dl.batch_size,False)


# In[ ]:


learn = cnn_learner(data, models.resnet18, metrics=[accuracy], callback_fns = [partial(OverSamplingCallback)], model_dir='/kaggle/working/models')


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(4,1e-2)


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(10,5e-4)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# This is much better than the severely imbalanced case but still not as good as training on the original dataset. Let's create a submission.

# In[ ]:


# get the predictions
predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)
# output to a file
submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})
submission_df.to_csv(f'submission_oversampled.csv', index=False)


# If we look at the submissions to the leaderboard, a previous run obtained these results:
# * Original - 0.99142
# * Imbalanced - 0.94700
# * Oversampled - 0.98557
# 
# This again demonstrates the usefulness of oversampling for imbalanced datasets.

# I hope this kernel demonstrated how to perform oversampling in the fastai library. This feature may be added to the `fastai` library if this is helpful to others.
