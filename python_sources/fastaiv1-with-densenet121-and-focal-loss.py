#!/usr/bin/env python
# coding: utf-8

# ## **Intro**
# fastai is a deep-learning framework built on top of Pytorchv1. Recently it got update to [v1](https://github.com/fastai/fastai) which is a huge update from previous version and contains preety amazing and cool apis to load data, create models and train them. You can get the overview on how to use that from their [MOOC](https://course.fast.ai/) and can get started with deep learning.
# 
# This kernel uses **fastai.vison** for image classification task. We will use transfer learning using densenet121 model from [torchvision](https://pytorch.org/docs/stable/torchvision/index.html) along with [focal loss](https://arxiv.org/pdf/1708.02002.pdf). Following techniques we will be using are already implemented in fastai.
#  
# 1. Learning rate finder 
# 2. Cyclic learning rate ([paper](https://arxiv.org/pdf/1506.01186.pdf))
# 3. Discriminative learning rates
# 4. Data augmentation
# 5. Transfer learning with pretrained model
# 6. Test time augmentation

# ## Let's get started
# Load the vision package which also loads other required packages <br>
# Load required metric function <br>

# In[ ]:


from fastai.vision import *
from sklearn.metrics import roc_auc_score

import warnings
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=UserWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# Define the path using [pathlib](https://docs.python.org/3/library/pathlib.html).

# In[ ]:


PATH = Path('../input')
PATH.ls()


# ## **Data loading**

# In[ ]:


train_df = pd.read_csv(PATH/'train_labels.csv')
print(train_df.shape)
train_df.head()


# In[ ]:


train_df['label'].value_counts(normalize=True)


# So the ratio is around 3:2

# In[ ]:


src = (ImageItemList.from_csv(PATH, folder='train', csv_name='train_labels.csv', suffix='.tif')
      .random_split_by_pct(0.1, seed=77)
      .label_from_df()
      .add_test_folder())


# **get_transforms()** defines all the required data augmentation we need such as vertical flip,  horizontal flip, zoom, rotation etc. You can read them [here](https://docs.fast.ai/vision.transform.html#get_transforms).

# In[ ]:


tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=0, max_zoom=1., max_lighting=0.05, max_warp=0)

data = (src.transform(tfms, size=96, resize_method=ResizeMethod.SQUISH)
       .databunch(bs=64, path='.'))

data.normalize(imagenet_stats);


# In[ ]:


data.show_batch(rows=5, figsize=(15, 15))


# ## **Model and training**
# We will define focal loss and roc metric

# In[ ]:


class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=1.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, **kwargs):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * ((1-pt)**self.gamma) * CE_loss
        return F_loss.mean()

    
def roc_score(inp, target):
    _, indices = inp.max(1)
    return torch.Tensor([roc_auc_score(target, indices)])[0]


# Let's create a learner which will contain our data, model and metrics. <br>
# Also we need to define the loss function we want to use. By default fastai uses softmax or cross-entropy loss for classification task but we want to use focal loss.

# In[ ]:


loss_func = FocalLoss(gamma=1.)
learn = create_cnn(data, models.densenet121, metrics=[accuracy, roc_score], loss_func=loss_func)


# Find learning rate and plot it

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# To choose the optimal learning rate find the point where the loss is minimum and then either find a point before that where the plot has a steepest slope or just divide that point by 10, both works fine.

# Fit the plot for 2 cycle

# In[ ]:


learn.fit_one_cycle(2, 1e-3)


# See how the learning rate and momentum varies with the training

# In[ ]:


learn.recorder.plot_lr(show_moms=True)


# Let's see the losses.

# In[ ]:


learn.recorder.plot_losses()


# Here you will see the loss gets a little bump after the initial drop. This is due to the increase in learning rate in the first half cycle and it will drive the model out of the local minima. In the second cycle the learning rate will decrease gradually which will help obtain the global minima.

# Save and load your model

# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.load('stage-1');


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(title='Confusion matrix')


# Unfreeze the other two layers and find an optimal learning rate again

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# Now here we define the learning rates using slice(). This function will create different learning rates for different groups to use discriminative learning. In short the main idea is to fine-tune the layers with different learning rates in which the initial layers will be optimized with lower learning rate and final layers with higher learning rate. <br>
# Train more...

# In[ ]:


max_lr = 1e-4
learn.fit_one_cycle(4, slice(1e-6, max_lr))


# In[ ]:


learn.save('stage-2')


# In[ ]:


learn.load('stage-2');


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(title='Confusion matrix')


# In[ ]:


auc_val = learn.validate()[2].item()


# ## **Testing**
# TTA is basically test time augmentation. You can read more about it in the fastai [docs](https://docs.fast.ai/basic_train.html#TTA
# ). <br>
# Here we need to apply sigmoid since we have to submit the probability. TTA will give logits not the probabilities because we have used FocalLoss here which is not defined in fastai yet but it's not needed for the submission.

# In[ ]:


preds, y = learn.TTA(beta=0.4, ds_type=DatasetType.Test)
preds = torch.softmax(preds, dim=1)[:, 1].numpy()


# Take care of the sequence of the ids to be submitted in **sample_submission.csv**

# In[ ]:


test_ids = [f.stem for f in learn.data.test_ds.items]
subm = pd.read_csv(PATH/'sample_submission.csv')
orig_ids = list(subm['id'])


# In[ ]:


def create_submission(orig_ids, test_ids, preds):
    preds_dict = dict((k, v) for k, v in zip(test_ids, preds))
    pred_cor = [preds_dict[id] for id in orig_ids]
    df = pd.DataFrame({'id':orig_ids,'label':pred_cor})
    df.to_csv(f'submission_{auc_val}.csv', header=True, index=False)
    
    return df


# In[ ]:


test_df = create_submission(orig_ids, test_ids, preds)
test_df.head()


# ## **Useful links**
# * [How fastai data block api works](https://blog.usejournal.com/finding-data-block-nirvana-a-journey-through-the-fastai-data-block-api-c38210537fe4)
# * [One cycle scheduler](https://sgugger.github.io/the-1cycle-policy.html)
# * [Motivation](https://www.fast.ai/2019/01/02/one-year-of-deep-learning)
# * [What can you do with deep learning?](https://www.fast.ai/2019/02/21/dl-projects/)
# * [Humpback Whale Identification Competition Starter Pack](https://github.com/radekosmulski/whale)
# 
# 

# In[ ]:




