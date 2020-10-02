#!/usr/bin/env python
# coding: utf-8

# # iMet Collection 2019 - FGVC6
# **Simple baseline for iMet Collection 2019 competition using fastai v1**
# * Model: densenet201
# * Loss: Focal loss
# * Metric: $F_{2}$ score
# 
# **What to try next?**
# * Different models
# * Optimize hyperparameter choice
# * Few-shot learning to improve score on classes with very few samples

# In[ ]:


import fastai
from fastai.vision import *
fastai.__version__


# # Initial setup

# In[ ]:


path = Path('../input/imet-2019-fgvc6/') # iMet data path


# In[ ]:


# Making pretrained weights work without needing to find the default filename
from torch.utils import model_zoo
Path('models').mkdir(exist_ok=True)
get_ipython().system("cp '../input/densenet201/densenet201.pth' 'models/'")
def load_url(*args, **kwargs):
    model_dir = Path('models')
    filename  = 'densenet201.pth'
    if not (model_dir/filename).is_file(): raise FileNotFoundError
    return torch.load(model_dir/filename)
model_zoo.load_url = load_url


# In[ ]:


# Load train dataframe
train_df = pd.read_csv(path/'train.csv')
train_df.head()


# In[ ]:


# Load labels dataframe
labels_df = pd.read_csv(path/'labels.csv')
labels_df.head()


# In[ ]:


# Load sample submission
test_df = pd.read_csv(path/'sample_submission.csv')
test_df.head()


# # Create data object using datablock API

# In[ ]:


train, test = [ImageList.from_df(df, path=path, cols='id', folder=folder, suffix='.png') 
               for df, folder in zip([train_df, test_df], ['train', 'test'])]
data = (train.split_by_rand_pct(0.2, seed=42)
        .label_from_df(cols='attribute_ids', label_delim=' ')
        .add_test(test)
        .transform(get_transforms(), size=128)
        .databunch(path=Path('.'), bs=64).normalize())


# In[ ]:


data.show_batch()


# # Create learner with densenet121 and FocalLoss
# For problems with high class imbalance Focal Loss is usually a better choice than the usual Cross Entropy Loss.

# In[ ]:


# Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val +                ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()
    
class FbetaLoss(nn.Module):
    def __init__(self, beta=1):
        super(FbetaLoss, self).__init__()
        self.small_value = 1e-6
        self.beta = beta

    def forward(self, logits, labels):
        beta = self.beta
        batch_size = logits.size()[0]
        p = F.sigmoid(logits)
        l = labels
        num_pos = torch.sum(p, 1) + self.small_value
        num_pos_hat = torch.sum(l, 1) + self.small_value
        tp = torch.sum(l * p, 1)
        precise = tp / num_pos
        recall = tp / num_pos_hat
        fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + self.small_value)
        loss = fs.sum() / batch_size
        return 1 - loss

class CombineLoss(nn.Module):
    def __init__(self):
        super(CombineLoss, self).__init__()
        self.fbeta_loss = FbetaLoss(beta=2)
        self.focal_loss = FocalLoss()
        
    def forward(self, logits, labels):
        loss_beta = self.fbeta_loss(logits, labels)
        loss_focal = self.focal_loss(logits, labels)
        return 0.5 * loss_beta + 0.5 * loss_focal


# In[ ]:


learn = cnn_learner(data, base_arch=models.densenet201, loss_func=CombineLoss(), metrics=fbeta)


# # Train the model

# In[ ]:


# Find a good learning rate
learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 3e-2
learn.fit_one_cycle(3, slice(lr))


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 1e-3
learn.fit_one_cycle(21, slice(lr/10, lr))


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
test_preds = learn.get_preds(DatasetType.Test)
test_df.attribute_ids = join_preds(test_preds, best_thr)
test_df.head()


# In[ ]:


test_df.to_csv('submission.csv', index=False)

