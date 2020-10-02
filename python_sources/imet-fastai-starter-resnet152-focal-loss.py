#!/usr/bin/env python
# coding: utf-8

# # iMet Collection 2019 - FGVC6
# **Simple baseline for iMet Collection 2019 competition using fastai v1**
# * Model: resnet152
# * Loss: Focal loss
# * Metric: $F_{2}$ score
# 
# **What to try next?**
# * Different models
# * Optimize hyperparameter choice
# * Few-shot learning to improve score on classes with very few samples
# 
# Refereed from original kernel [here](https://www.kaggle.com/mnpinto/imet-fastai-starter)

# In[1]:


import fastai
from fastai.vision import *
fastai.__version__


# In[2]:


BATCH  = 32
SIZE   = 250


# In[3]:


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


# # Initial setup

# In[4]:


path = Path('../input/imet-2019-fgvc6/') # iMet data path


# In[5]:


import os
os.listdir('../input/')


# In[6]:


# Making pretrained weights work without needing to find the default filename
from torch.utils import model_zoo
Path('models').mkdir(exist_ok=True)
get_ipython().system("cp '../input/resnet15216epochs/stage-1.pth' 'models/resnet152.pth'")
def load_url(*args, **kwargs):
    model_dir = Path('models')
    filename  = 'resnet152.pth'
    if not (model_dir/filename).is_file(): raise FileNotFoundError
    return torch.load(model_dir/filename)
model_zoo.load_url = load_url


# In[7]:


# Load train dataframe
train_df = pd.read_csv(path/'train.csv')
train_df.head()


# In[8]:


# Load labels dataframe
labels_df = pd.read_csv(path/'labels.csv')
labels_df.head()


# In[9]:


# Load sample submission
test_df = pd.read_csv(path/'sample_submission.csv')
test_df.head()


# # Create data object using datablock API

# In[10]:


tfms = get_transforms(do_flip=True, flip_vert=False, max_rotate=0.10, max_zoom=1.5, max_warp=0.2, max_lighting=0.2,
                     xtra_tfms=[(symmetric_warp(magnitude=(-0,0), p=0)),])

train, test = [ImageList.from_df(df, path=path, cols='id', folder=folder, suffix='.png') 
               for df, folder in zip([train_df, test_df], ['train', 'test'])]
data = (train.split_by_rand_pct(0.1, seed=42)
        .label_from_df(cols='attribute_ids', label_delim=' ')
        .add_test(test)
        .transform(tfms, size=SIZE, resize_method=ResizeMethod.PAD, padding_mode='border',)
        .databunch(path=Path('.'), bs=BATCH).normalize(imagenet_stats))


# # Create learner with resnet152 and FocalLoss
# For problems with high class imbalance Focal Loss is usually a better choice than the usual Cross Entropy Loss.

# In[11]:


# Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val +                ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()


# In[12]:


learn = cnn_learner(data, base_arch=models.resnet152, loss_func=FocalLoss(), metrics=fbeta, pretrained=False)
learn.load('resnet152')


# # Train the model

# In[13]:


# Find a good learning rate
learn.lr_find()
learn.recorder.plot()


# In[14]:


learn.unfreeze()


# In[ ]:


lr = 1e-2
learn.fit_one_cycle(10, slice(1e-5, 1e-3))


# In[ ]:


learn.save('stage-1', return_path=True)
learn.export()


# # Get predictions

# In[ ]:


def find_best_fixed_threshold(preds, targs, do_plot=True):
    score = []
    thrs = np.arange(0, 0.5, 0.01)
    for thr in progress_bar(thrs):
        score.append(fbeta(valid_preds[0],valid_preds[1], thresh=thr))
    score = np.array(score)
    pm = score.argmax()
    best_thr, best_score = thrs[pm], score[pm].item()
    print(f'thr={best_thr:.3f}', f'F2={best_score:.3f}')
    if do_plot:
        plt.plot(thrs, score)
        plt.vlines(x=best_thr, ymin=score.min(), ymax=score.max())
        plt.text(best_thr+0.03, best_score-0.01, f'$F_{2}=${best_score:.3f}', fontsize=14);
        plt.show()
    return best_thr

i2c = np.array([[i, c] for c, i in learn.data.train_ds.y.c2i.items()]).astype(int) # indices to class number correspondence

def join_preds(preds, thr):
    return [' '.join(i2c[np.where(t==1)[0],1].astype(str)) for t in (preds[0].sigmoid()>thr).long()]


# In[ ]:


# Validation predictions
valid_preds = learn.get_preds(DatasetType.Valid)
best_thr = find_best_fixed_threshold(*valid_preds)


# In[ ]:


# Test predictions
test_preds = learn.TTA(ds_type=DatasetType.Test)
test_df.attribute_ids = join_preds(test_preds, best_thr)
test_df.head()


# In[ ]:


test_df.to_csv('submission.csv', index=False)

