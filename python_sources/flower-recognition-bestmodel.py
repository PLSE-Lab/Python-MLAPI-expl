#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate
from fastai.callbacks import SaveModelCallback
from sklearn.metrics import roc_auc_score


# In[ ]:


bs = 48
img_size=320 #actual size of all pics is 500,500


# In[ ]:


PATH = Path('../input/flower-recognition-he/he_challenge_data/data/')


# In[ ]:


train_df=pd.read_csv(PATH/'train.csv')
train_df.head()


# valid_names=trn_df.sample(n=350, random_state=1) 
# validList = ImageList.from_df(df=valid_names,path='../input/aptos2019-blindness-detection/train_images/',cols='id_code',suffix='.png') 
# validList
# 
# valid_idx=valid_names.index 
# np.save('../working/valid_idx',valid_idx) 
# np.savetxt('../working/val_idx1',valid_idx)
# 
# trn_df.drop(index=valid_names.index,axis=0,inplace=True) 
# trn_df.shape
# 
# trn_idx=trn_df.index 
# np.save('../working/trn_idx',trn_idx) 
# np.savetxt('../working/trn_idx1',trn_idx)
# 
# base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/') 
# train_dir = os.path.join(base_image_dir,'train_images/') 
# trn_df['path'] = trn_df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x))) 
# trn_df = trn_df.drop(columns=['id_code']) 
# trn_df.shape

# In[ ]:


tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=360.0, max_zoom=1.35, max_warp=0.2, max_lighting=.4,)


# In[ ]:


src = (
    ImageList.from_df(train_df,PATH,folder='train',suffix='.jpg')
        .split_by_rand_pct(0.2, seed=42)
        .label_from_df(cols='category')    
    )
data = (
    src.transform(tfms=tfms,size=img_size)
    .databunch(bs=bs)
    .normalize(imagenet_stats)
)


# In[ ]:


#data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


#print(data.classes)
#len(data.classes),data.c


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

'''

def roc_score(inp, target): #defined for binary classification only not multiclass
    _, indices = inp.max(1)
    return torch.Tensor([roc_auc_score(target, indices)])[0]


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
'''


# In[ ]:


#Making pretrained weights work without needing to find the default filename
if not os.path.exists('/tmp/.cache/torch/checkpoints/'):
       os.makedirs('/tmp/.cache/torch/checkpoints/')
get_ipython().system("cp '../input/resnet50/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'")
#!cp '../input/resnet34/resnet34.pth' '/tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth'


# In[ ]:


#loss_func = FocalLoss(gamma=1.)
learn = (cnn_learner(data,
                     models.resnet50,
                     metrics=[accuracy, error_rate],
                     loss_func= LabelSmoothingCrossEntropy(),
                     model_dir= '../../../../working/'))


# In[ ]:


#learn.lr_find() 
#learn.recorder.plot()


# In[ ]:


learn.load('../input/flower-weights2/bestmodel');


# In[ ]:


lr=1e-02


# In[ ]:


learn.fit_one_cycle(15, lr,callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy')])


# In[ ]:


learn.save('stage-1')
get_ipython().system('cp /kaggle/working/bestmodel.pth /kaggle/working/bestmodel_stg1.pth ')


# In[ ]:


learn.load('bestmodel');
learn.fit_one_cycle(10, lr,callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy')])


# In[ ]:


get_ipython().system('cp /kaggle/working/bestmodel.pth /kaggle/working/bestmodel_stg1_2.pth ')


# Since our model is working as we expect it to, we will *unfreeze* our model and train some more.

# In[ ]:


learn.unfreeze()


# learn.lr_find() 
# learn.recorder.plot()

# In[ ]:


lr=1e-05
learn.fit_one_cycle(15, slice(lr/100,lr/10),callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy')])


# In[ ]:


learn.save('stage-2')
get_ipython().system('cp /kaggle/working/bestmodel.pth /kaggle/working/bestmodel_stg2.pth')


# In[ ]:


bs = 24
img_size=399 #actual size of all pics is 500,500
data = (
    src.transform(tfms=get_transforms(),size=img_size)
    .databunch(bs=bs)
    .normalize(imagenet_stats)
)


# In[ ]:


#learn.data=data


# In[ ]:


#learn.freeze()


# In[ ]:


learn.load('bestmodel');


# In[ ]:


#learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-4),callbacks=[SaveModelCallback(learn, every='improvement', monitor='valid_loss')])


# In[ ]:


learn.save('stage-1-299');


# In[ ]:


#learn.unfreeze()
learn.load('bestmodel');
#learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-4),callbacks=[SaveModelCallback(learn, every='improvement', monitor='valid_loss')])


# That's a pretty accurate model!

# In[ ]:


learn.save('stage-2-299')
learn.load('bestmodel');


# ## Predictions

# In[ ]:


sample_df=pd.read_csv(PATH/'sample_submission.csv')


# In[ ]:


learn.data.add_test(ImageList.from_df(sample_df,PATH,folder='test',suffix='.jpg'))


# In[ ]:


preds,_ = learn.TTA(ds_type=DatasetType.Test)


# In[ ]:


preds=np.argmax(preds,axis=1)
preds[:5]


# In[ ]:


fnames = [f.split('.')[2].split('/')[-1] for f in learn.data.test_ds.items]
fnames[:3]


# In[ ]:


labelled_preds = [learn.data.classes[pred] for pred in preds]
labelled_preds[:10]


# In[ ]:


df = pd.DataFrame({'image_id':fnames,'category':labelled_preds}, columns=['image_id','category'])
df.head()


# In[ ]:


df = df.sort_values(by = ['image_id'], ascending = [True])
df.to_csv('resnet34.csv', index=False)


# In[ ]:


#without comminting download sub file :)
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "resnet50.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(df)

