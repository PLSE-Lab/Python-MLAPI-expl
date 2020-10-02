#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import fastai
from fastai.vision import *


# In[ ]:


path = '../input/game-of-deep-learning-ship-datasets/train/'


# In[ ]:


src = ImageList.from_csv(path,'train.csv', folder='images',
                   suffix='').split_by_rand_pct(0.2).\
                    label_from_df()


# In[ ]:


tfms = get_transforms(flip_vert=False, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[ ]:


data = (src.transform(tfms, size=128)
        .databunch().normalize(imagenet_stats))


# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# In[ ]:


arch = models.resnet101


# In[ ]:


from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction='elementwise_mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction is None:
            return F_loss
        else:
            return torch.mean(F_loss)


# In[ ]:


# fbeta_binary = fbeta_binary_me(1)
learn = cnn_learner(data, arch,model_dir="/kaggle/working/")
learn.loss_fn = FocalLoss()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = 0.01


# In[ ]:


learn.fit_one_cycle(20, slice(lr))


# In[ ]:


learn.save('stage-1-rn50')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(title='Confusion matrix')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(20, slice(1e-5, lr/5))


# In[ ]:


data = (src.transform(tfms, size=256)
        .databunch().normalize(imagenet_stats))

learn.data = data
data.train_ds[0][0].shape


# In[ ]:


learn.freeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr=1e-2/2


# In[ ]:


learn.fit_one_cycle(20, slice(lr))


# In[ ]:


learn.save('stage-1-256-rn50')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(title='Confusion matrix')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(30, slice(1e-5, lr/5))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage-2-256-rn50')


# In[ ]:


learn.export('/kaggle/working/fastai_resnet.pkl')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(title='Confusion matrix')


# In[ ]:


test = ImageList.from_folder('../input/avgameofdltestjpg/test-jpg/')
len(test)


# In[ ]:



learn = load_learner('/kaggle/working/','fastai_resnet.pkl', test=test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


preds[:5]


# In[ ]:


labelled_preds = torch.argmax(preds,1)+1


# In[ ]:


# thresh = 0.2
# labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]


# In[ ]:


fnames = [f.name for f in learn.data.test_ds.items]


# In[ ]:


df = pd.DataFrame({'image':fnames, 'category':labelled_preds}, columns=['image', 'category'])


# In[ ]:


df.to_csv('submission.csv', index=False)


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df['class1_prob'] = preds[:,0].numpy()
df['class2_prob'] = preds[:,1].numpy()
df['class3_prob'] = preds[:,2].numpy()
df['class4_prob'] = preds[:,3].numpy()
df['class5_prob'] = preds[:,4].numpy()
df.to_csv('raw_prob.csv', index=False)

