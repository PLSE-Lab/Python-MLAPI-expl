#!/usr/bin/env python
# coding: utf-8

# ## Regressor for the Blindness Detection challenge.
# 
# This kernel uses an **EfficientNetB0** regression model (initially **pre-trained** as a [classifier](https://www.kaggle.com/hmendonca/efficientnet-pytorch-ignite-amp-aptos19)),
# minimising the mean squared error (**MSE**) of the predicted blindness score \[0-4\] against the training labels.
# 
# Later, it maximises the **Quadratic Weighted Kappa** by optimising the boundary coefficients as borrowed from [abhishek](https://www.kaggle.com/abhishek/optimizer-for-quadratic-weighted-kappa)'s kernel with a slightly cleaner code base, I hope ;)
# 
# It also uses a BatchSampler with weighted **OverSampling** to balance out the classes distribution during training, and test time augmentation (**TTA**) to improve the submission score.
# 
# 
# If you please, upvote and leave questions or any constructive feedback below (for me and other kagglers learning).
# Cheers!
# 
# 
# EfficientNet architecture and pre-trained weights from: https://www.kaggle.com/hmendonca/efficientnet-pytorch-ignite-aptos19 and [Ignite samples](https://github.com/pytorch/ignite/tree/master/examples):
# > Recently new ConvNets architectures have been proposed in "[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf)" paper. According to the paper, model's compound scaling starting from a 'good' baseline provides an network that achieves state-of-the-art on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet.
# ![efficientnets](https://raw.githubusercontent.com/pytorch/ignite/c22609796031f5831f054036895696c7e4df07ce/examples/notebooks/assets/efficientnets.png)
# 

# ## Imports

# In[ ]:


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid', color_codes=True)

from sklearn.metrics import confusion_matrix, cohen_kappa_score

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm, tqdm_notebook
import os, math
from random import shuffle  
from zipfile import ZipFile
from PIL import Image
from sklearn.utils import shuffle

get_ipython().system('ls ../input/*')


# In[ ]:


import fastai
from fastai import *
from fastai.vision import *
from fastai.callbacks import *

fastai.__version__


# In[ ]:


# check if the kernel is running in interactive/edit/debug mode https://www.kaggle.com/masterscrat/detect-if-notebook-is-running-interactively
def is_interactive():
   return 'runtime' in get_ipython().config.IPKernelApp.connection_file

print('Interactive?', is_interactive())


# In[ ]:


# from https://www.kaggle.com/chanhu/eye-inference-num-class-1-ver3
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(42)


# In[ ]:


package_path = '../input/efficientnet-pytorch/efficientnet-pytorch/EfficientNet-PyTorch-master'
sys.path.append(package_path)

from efficientnet_pytorch import EfficientNet


# In[ ]:


# copy pretrained weights for resnet34 to the folder fastai will search by default
Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)
model_path = '/tmp/.cache/torch/checkpoints/efficientNet.pth'
get_ipython().system('cp ../input/efficientnet*/efficientNet_*.pth {model_path}')


# ## Stratified Dataset

# In[ ]:


PATH = Path('../input/aptos2019-blindness-detection')

df_train = pd.read_csv(PATH/'train.csv', dtype={'id_code':str, 'diagnosis':int})
df_test = pd.read_csv(PATH/'test.csv')

# if is_interactive():  # fast debug mode
#     df_train = df_train.sample(1200)

_ = df_train.hist()


# In[ ]:


# create Stratified validation split (10%)
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=10, random_state=42)
tr_ids, val_ids = next(cv.split(df_train.id_code, df_train.diagnosis))
print(len(tr_ids), len(val_ids))
_ = df_train.loc[val_ids].hist()


# In[ ]:


# # append minority classes again to improve label balance
# minority = df_train.loc[tr_ids] # don't touch validation set
# minority = minority[(minority.diagnosis == 1) | (minority.diagnosis == 3) | (minority.diagnosis == 4)]
# df_balanced = pd.concat([df_train, minority])

# # def norm_data(Y): return (Y - 1.) / 4
# # def denorm_data(Y_): return (Y_ * 4.) + 1.
# # df_balanced.diagnosis = norm_data(df_balanced.diagnosis)

# _ = df_balanced.hist()


# In[ ]:


# create image data bunch
# aptos19_stats = ([0.42, 0.22, 0.075], [0.27, 0.15, 0.081])
src = (ImageList.from_df(df=df_train, path=PATH, folder='train_images', suffix='.png')
       .split_by_idx(val_ids)
       .label_from_df(cols='diagnosis', label_cls=FloatList))
data = ImageDataBunch.create_from_ll(src,
                                     ds_tfms=get_transforms(flip_vert=True,
                                                            max_rotate=2., max_zoom=1.02,
                                                            max_lighting=0.2, max_warp=0.05,
                                                            p_affine=0.9, p_lighting=0.8),
                                     size=256,
                                     bs=32,
                                     num_workers=os.cpu_count()
                                    ).normalize(imagenet_stats)
data


# In[ ]:


# show some sample images
data.show_batch(rows=3, figsize=(7,6), ds_type=DatasetType.Train)


# ## Define model

# In[ ]:


def EfficientNetB4(pretrained=False):
    """Constructs a EfficientNetB0 model for FastAI.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 1 }) ## Regressor
    if pretrained:
        model_state = torch.load(model_path)
        # load original weights apart from its head
        if '_fc.weight' in model_state.keys():
            model_state.pop('_fc.weight')
            model_state.pop('_fc.bias')
            res = model.load_state_dict(model_state, strict=False)
            assert str(res.missing_keys) == str(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
        else:
            # A basic remapping is required
            from collections import OrderedDict
            mapping = { i:o for i,o in zip(model_state.keys(), model.state_dict().keys()) }
            mapped_model_state = OrderedDict([
                (mapping[k], v) for k,v in model_state.items() if not mapping[k].startswith('_fc')
            ])
            res = model.load_state_dict(mapped_model_state, strict=False)
            print(res)
    return model


# In[ ]:


# create model
model = EfficientNetB4(pretrained=True)
# print model structure (hidden)
model


# ## Non-linear Kappa Optimizer

# In[ ]:


# inspired by https://www.kaggle.com/tanlikesmath/intro-aptos-diabetic-retinopathy-eda-starter
class KappaOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.coef = [0.5, 1.5, 2.5, 3.5]
        # define score function:
        self.func = self.quad_kappa
    
    def predict(self, preds):
        return self._predict(self.coef, preds)

    @classmethod
    def _predict(cls, coef, preds):
        if type(preds).__name__ == 'Tensor':
            y_hat = preds.clone().view(-1)
        else:
            y_hat = torch.FloatTensor(preds).view(-1)

        for i,pred in enumerate(y_hat):
            if   pred < coef[0]: y_hat[i] = 0
            elif pred < coef[1]: y_hat[i] = 1
            elif pred < coef[2]: y_hat[i] = 2
            elif pred < coef[3]: y_hat[i] = 3
            else:                y_hat[i] = 4
        return y_hat.int()
    
    def quad_kappa(self, preds, y):
        return self._quad_kappa(self.coef, preds, y)

    @classmethod
    def _quad_kappa(cls, coef, preds, y):
        y_hat = cls._predict(coef, preds)
        return cohen_kappa_score(y, y_hat, weights='quadratic')

    def fit(self, preds, y):
        ''' maximize quad_kappa '''
        print('Early score:', self.quad_kappa(preds, y))
        neg_kappa = lambda coef: -self._quad_kappa(coef, preds, y)
        opt_res = sp.optimize.minimize(neg_kappa, x0=self.coef, method='nelder-mead',
                                       options={'maxiter':1000, 'fatol':1e-20, 'xatol':1e-20})
        print(opt_res)
        self.coef = opt_res.x
        print('New score:', self.quad_kappa(preds, y))

    def forward(self, preds, y):
        ''' the pytorch loss function '''
        return torch.tensor(self.quad_kappa(preds, y))

kappa_opt = KappaOptimizer()


# In[ ]:


## some visual tests
preds = [-2.5, 0.3, 0.4,  0.5, 1.2, 1.41,  1.5, 2.4, 2.42,  2.5, 3.4, 3.43,  3.5, 4, 9]
y_hat = kappa_opt.predict(preds)
print(y_hat)
print('calc scores:', 
      kappa_opt(preds[::-1], y_hat.tolist()[::-1]), # inverse order
      kappa_opt(preds, y_hat+1),
      kappa_opt(preds, np.ones_like(y_hat)),
      kappa_opt(preds, 4-y_hat))

# shift predictions to test optimizer
n_preds = np.array(preds)+0.11
print('goal Y:', y_hat)
print('before:', kappa_opt.predict(n_preds))
kappa_opt.fit(n_preds, y_hat)
print('after: ', kappa_opt.predict(n_preds))

# reset
kappa_opt = KappaOptimizer()


# In[ ]:


# Adapted from FastAI master
# the weights of each label is 1/count of the labels of that class,
# so the label distribution on each batch is uniform.
# I.e. a batch of 100 images would have about 20 images of each class.
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler

class OverSamplingCallback(LearnerCallback):
    def __init__(self,learn:Learner, weights:torch.Tensor=None):
        super().__init__(learn)
        labels = self.learn.data.train_dl.dataset.y.items.astype(int)
        _,counts = np.unique(labels, return_counts=True)
#         counts = 1. / counts
        counts = 1. / np.sqrt(counts)  # non-linear weights
        self.weights = (weights if weights is not None else torch.DoubleTensor(counts[labels]))

    def on_train_begin(self, **kwargs):
        self.learn.data.train_dl.dl.batch_sampler = BatchSampler(
            WeightedRandomSampler(self.weights, len(self.learn.data.train_dl.dataset)),
                                  self.learn.data.train_dl.batch_size, False)

class StratifiedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.len = math.ceil(len(labels) / batch_size)

    def __len__(self): return self.len

    def __iter__(self):
        ids = np.arange(len(self.labels))
        folds = StratifiedKFold(n_splits=self.len, shuffle=True).split(ids, self.labels)
        for _,batch in folds:
            yield batch.tolist()

class StratifiedBatchCallback(LearnerCallback):
    def on_train_begin(self, **kwargs):
        self.learn.data.train_dl.dl.batch_sampler = StratifiedBatchSampler(
            self.learn.data.train_dl.dataset.y.items.astype(int),
            self.learn.data.train_dl.batch_size)


# ## Define Task

# In[ ]:


# build model (using EfficientNetB0)
learn = Learner(data, model,
                loss_func=MSELossFlat(),
                metrics=[kappa_opt],
                callback_fns=[BnFreeze,
#                               StratifiedBatchCallback,
#                               partial(GradientClipping, clip=0.2),
                              partial(SaveModelCallback, monitor='quad_kappa', name='bestmodel')]
               )
learn.split( lambda m: (model._conv_head,) )
learn.freeze()
learn.model_dir = '/tmp/'


# In[ ]:


# learn.summary()


# ## Training

# In[ ]:


learn.lr_find(end_lr=0.5)
learn.recorder.plot(suggestion=True)


# In[ ]:


# train head first
learn.fit_one_cycle(2, max_lr=5e-3, div_factor=15)
learn.save('stage-1')
learn.recorder.plot_losses()


# In[ ]:


# unfreeze and search appropriate learning rate for full training
learn.unfreeze()
learn.lr_find(start_lr=1e-10, wd=1e-3)
learn.recorder.plot(suggestion=True)


# In[ ]:


# train all layers
learn.fit_one_cycle(6, max_lr=slice(1e-5, 1e-3), div_factor=10, wd=1e-3)
learn.save('stage-2')
learn.recorder.plot_losses()


# In[ ]:


# schedule of the lr (left) and momentum (right) that the 1cycle policy uses
learn.recorder.plot_lr(show_moms=True)


# In[ ]:


# kappa scores
learn.recorder.plot_metrics()


# In[ ]:


# reload best model so far and look for a new learning rate
learn.load('bestmodel')
learn.lr_find(start_lr=1e-10)
learn.recorder.plot(suggestion=True)


# In[ ]:


# train all layers, now with some weight decay
learn.fit_one_cycle(12, max_lr=slice(2e-6, 2e-4), div_factor=20)
learn.save('stage-3')
learn.recorder.plot_losses()
# # schedule of the lr (left) and momentum (right) that the 1cycle policy uses
# learn.recorder.plot_lr(show_moms=True)


# ## Find best coefficients

# In[ ]:


learn.load('bestmodel')

# remove zoom from FastAI TTA
tta_params = {'beta':0.12, 'scale':1.0}


# In[ ]:


valid_preds = learn.get_preds(ds_type=DatasetType.Valid)
# valid_preds = learn.TTA(ds_type=DatasetType.Valid, **tta_params)
_ = pd.DataFrame(valid_preds[0].numpy().flatten()).hist()


# In[ ]:


kappa_opt.fit(valid_preds[0], valid_preds[1])


# In[ ]:


print('New coefficients:', kappa_opt.coef)
new_valid = kappa_opt.predict(valid_preds[0]).numpy()
_ = pd.DataFrame(new_valid).hist()


# ## TTA and submission

# In[ ]:


sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
sample_df.head()


# In[ ]:


# create test dataset
learn.data.add_test(ImageList.from_df(
    sample_df, PATH,
    folder='test_images',
    suffix='.png'
))


# In[ ]:


# test time augmentation
test_preds,_ = learn.get_preds(ds_type=DatasetType.Test)
# test_preds,_ = learn.TTA(ds_type=DatasetType.Test, **tta_params)
_ = pd.Series(test_preds.squeeze().tolist()).hist()


# In[ ]:


# apply optimised coefficients
sample_df.diagnosis = kappa_opt.predict(test_preds)
sample_df.head()


# In[ ]:


# save sub
sample_df.to_csv('submission.csv', index=False)
_ = sample_df.hist()


# In[ ]:


#move models back to root folder
get_ipython().system('mv {learn.model_dir}/*.pth .')
os.listdir()

