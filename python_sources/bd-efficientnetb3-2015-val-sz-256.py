#!/usr/bin/env python
# coding: utf-8

# # Blindness Detection: EfficientNet B3
# 
# * Img size: 256x256
# * Batch size: 64
# * Data: concat 2019 + 2015 training sets. Downsample class 0 to match class 2. Each epoch change sample of 0 class.
# * Validation: 2015 test set with class 0 downsampled to match class 2.
# * Preprocess: Preprocessing copied from [this](https://www.kaggle.com/joorarkesteijn/fast-cropping-preprocessing-and-augmentation) kernel which used ideas from [this](https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping) kernel.
# * Model head: multiclass (ordinal regression) outputs.
# * Loss: BCEWithLogitsLoss with modified label smoothing: convert `[1, 1, 0, 0, 0]` labels into `[0.95, 0.95, 0.05, 0.05, 0.05]`
# * Opt: Adam (fast.ai default)
# * Pseudo-labelling: add all test labels from submission.csv with 0.834 LB.
# * Augmentations: flip_lr, brightness, contrast, rotate(360)
# * Train: train just head for one epoch, train 15 epochs using [one cycle](https://arxiv.org/pdf/1803.09820).

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import time
import warnings
import os                   
from random import shuffle  
from zipfile import ZipFile

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from fastai import *
from fastai.vision import *
from fastai.callbacks import Callback
from fastai.callbacks import SaveModelCallback, EarlyStoppingCallback, ReduceLROnPlateauCallback
from fastai.data_block import MultiCategoryList
import cv2 
from tqdm import tqdm
from sklearn.utils import shuffle

sys.path.append('../input/efficientnet-pytorch/efficientnet-pytorch/EfficientNet-PyTorch-master')
import efficientnet_pytorch

style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ## Hyperparams

# In[ ]:


MODEL_NAME = 'efficientnetb03_change_zeros_ord_reg_label_smoothing'

IMG_SIZE = 256
BS = 64
SEED = 425


# In[ ]:


seed_everything(SEED)


# ## Augmentations

# In[ ]:


max_zoom = 1.5
p_affine = 0.75
max_lighting = 0.2
p_lighting = 0.75
scale = 2.
max_rotate = 360

train_tfms, val_tfms = [
    flip_lr(),
    brightness(change=(0.5 * (1-max_lighting), 0.5 * (1 + max_lighting)), p=p_lighting),
    contrast(scale=(1-max_lighting, 1/(1-max_lighting)), p=p_lighting),
    rotate(degrees=(-max_rotate, max_rotate), p=p_affine)
], []


# ## Data downsampling

# In[ ]:


def get_label(diagnosis):
        return ','.join([str(i) for i in range(diagnosis + 1)])


def get_train_df(seed, num_zeros=4000):
    val_preds_id = pd.read_csv('../input/bd-peter-and-lex-validation-set/val.csv')['id_code']

    df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
    df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

    df_train['is_valid'] = False
    # df_train.loc[df_train.id_code.isin(val_preds_id), 'is_valid'] = True
    df_train.id_code = '../input/aptos2019-blindness-detection/train_images/' + df_train.id_code + '.png'

    df_train.columns = ['image_path', 'diagnosis', 'is_valid']

    extra_training_df = pd.read_csv('../input/diabetic-retinopathy-resized/trainLabels.csv')
    extra_training_df['is_valid'] = False
    # extra_training_df.loc[extra_training_df.image.isin(val_preds_id), 'is_valid'] = True
    extra_training_df.image = '../input/diabetic-retinopathy-resized/resized_train/resized_train/' + extra_training_df.image + '.jpeg'
    extra_training_df.columns = ['image_path', 'diagnosis', 'is_valid']
    
    pseudo = pd.read_csv('../input/linearstackingblendedwithbestlbsub/linear-stacking-blended-with-best-lb-as-reg-v2-submission.csv')
    pseudo.id_code = '../input/aptos2019-blindness-detection/test_images/' + pseudo.id_code + '.png'
    pseudo['is_valid'] = False
    pseudo.columns = ['image_path', 'diagnosis', 'is_valid']
    
    test_labels_15_df = pd.read_csv('../input/resized-2015-2019-blindness-detection-images/labels/testLabels15.csv')
    del test_labels_15_df['Usage']
    test_labels_15_df.columns = ['image_id', 'diagnosis']
    test_labels_15_df['dataset_id'] = 'test_labels_15'
    test_labels_15_df['image_path'] = '../input/resized-2015-2019-blindness-detection-images/resized test 15/' + test_labels_15_df.image_id + '.jpg'
    test_labels_15_df['is_valid'] = True
    test_labels_15_df = test_labels_15_df[['image_path', 'diagnosis', 'is_valid']]

    df_train = pd.concat([
        df_train,
        extra_training_df[(extra_training_df.diagnosis == 0) & (extra_training_df.is_valid)],
        extra_training_df[(extra_training_df.diagnosis == 0) & ~(extra_training_df.is_valid)].sample(n=num_zeros, random_state=seed),
        extra_training_df[extra_training_df.diagnosis == 1],
        extra_training_df[extra_training_df.diagnosis == 2],
        extra_training_df[extra_training_df.diagnosis == 3],
        extra_training_df[extra_training_df.diagnosis == 4],
        pseudo,
        pd.concat([
            test_labels_15_df[test_labels_15_df.diagnosis == 0].sample(n=7900, random_state=420),
            test_labels_15_df[test_labels_15_df.diagnosis != 0]
        ]).sample(n=10_000, random_state=420),
    ]).sample(frac=1, random_state=seed)

    df_train['label'] = df_train.diagnosis.apply(get_label)
    
    return df_train


# Fixes bug with `show_batch` with multi label labels.

# In[ ]:



class ReconstructFixMultiCategoryList(MultiCategoryList):
    def reconstruct(self, t):
        try:
            return super().reconstruct(t)
        except Exception as e:
            return FloatItem(np.log(t))


# In[ ]:


sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')


# ## Image preprocessing
# 
# All image preprocessing this kernel taken from [this]() wonderful kernel.

# In[ ]:


# To remove irregularities along the circular boundary of the image
PARAM = 96

def Radius_Reduction(img,PARAM):
    h,w,c=img.shape
    Frame=np.zeros((h,w,c),dtype=np.uint8)
    cv2.circle(Frame,(int(math.floor(w/2)),int(math.floor(h/2))),int(math.floor((h*PARAM)/float(2*100))), (255,255,255), -1)
    Frame1=cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    img1 =cv2.bitwise_and(img,img,mask=Frame1)
    return img1


def info_image(im):
    # Compute the center (cx, cy) and radius of the eye
    cy = im.shape[0]//2
    midline = im[cy,:]
    midline = np.where(midline>midline.mean()/3)[0]
    if len(midline)>im.shape[1]//2:
        x_start, x_end = np.min(midline), np.max(midline)
    else: # This actually rarely happens p~1/10000
        x_start, x_end = im.shape[1]//10, 9*im.shape[1]//10
    cx = (x_start + x_end)/2
    r = (x_end - x_start)/2
    return cx, cy, r


def resize_image(im, img_size, augmentation=False):
    # Crops, resizes and potentially augments the image to IMG_SIZE
    cx, cy, r = info_image(im)
    scaling = img_size/(2*r)
    rotation = 0
    if augmentation:
        scaling *= 1 + 0.3 * (np.random.rand()-0.5)
        rotation = 360 * np.random.rand()
    M = cv2.getRotationMatrix2D((cx,cy), rotation, scaling)
    M[0,2] -= cx - img_size/2
    M[1,2] -= cy - img_size/2
    return cv2.warpAffine(im, M, (img_size, img_size)) # This is the most important line


def open_img(self, fn, size):
    "Open image in `fn`, subclass and overwrite for custom behavior."
    image = cv2.imread(fn)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_image(image, size)
    # image = subtract_median_bg_image(image)
    image = Radius_Reduction(image, PARAM)
    return Image(pil2tensor(image, np.float32).div_(255))
    

ImageList.open = lambda self, fn: open_img(self, fn, size=IMG_SIZE)


# ## Metrics and callbacks

# In[ ]:


def get_preds(arr):
    mask = arr == 0
    return np.clip(np.where(mask.any(1), mask.argmax(1), 5) - 1, 0, 4)

class ConfusionMatrix(Callback):
    "Computes the confusion matrix."

    def on_train_begin(self, **kwargs):
        self.n_classes = 0

    def on_epoch_begin(self, **kwargs):
        self.cm = None

    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        preds = torch.tensor(get_preds((torch.sigmoid(last_output) > 0.5).cpu().numpy()))
        
        targs = torch.tensor(get_preds(last_target.cpu().numpy()))

        if self.n_classes == 0:
            self.n_classes = last_output.shape[-1]
            self.x = torch.arange(0, self.n_classes)
        
        cm = ((preds==self.x[:, None]) & (targs==self.x[:, None, None])).sum(dim=2, dtype=torch.float32)
        if self.cm is None: self.cm =  cm
        else:               self.cm += cm

    def on_epoch_end(self, **kwargs):
        self.metric = self.cm
        

@dataclass
class KappaScore(ConfusionMatrix):
    "Compute the rate of agreement (Cohens Kappa)."
    weights:Optional[str]=None      # None, `linear`, or `quadratic`

    def on_epoch_end(self, last_metrics, **kwargs):
        sum0 = self.cm.sum(dim=0)
        sum1 = self.cm.sum(dim=1)
        expected = torch.einsum('i,j->ij', (sum0, sum1)) / sum0.sum()
        if self.weights is None:
            w = torch.ones((self.n_classes, self.n_classes))
            w[self.x, self.x] = 0
        elif self.weights == "linear" or self.weights == "quadratic":
            w = torch.zeros((self.n_classes, self.n_classes))
            w += torch.arange(self.n_classes, dtype=torch.float)
            w = torch.abs(w - torch.t(w)) if self.weights == "linear" else (w - torch.t(w)) ** 2
        else: raise ValueError('Unknown weights. Expected None, "linear", or "quadratic".')
        k = torch.sum(w * self.cm) / torch.sum(w * expected)
        return add_metrics(last_metrics, 1-k)


@dataclass
class ChangeDataOnEpoch(Callback):
    learn:Learner
    i:int
        
    def on_epoch_end(self, **kwargs):
        print(f'Data seed {self.i}')
        self.learn.data = get_data(seed=self.i)
        self.learn.data.add_tfm(batch_to_half)
        self.i += 1


# ## Loss function

# In[ ]:


class FlattenedLoss():
    "Same as `func`, but flattens input and target."
    def __init__(self, func, *args, axis:int=-1, floatify:bool=False, is_2d:bool=True, **kwargs):
        self.func,self.axis,self.floatify,self.is_2d = func(*args,**kwargs),axis,floatify,is_2d
        functools.update_wrapper(self, self.func)

    def __repr__(self): return f"FlattenedLoss of {self.func}"
    @property
    def reduction(self): return self.func.reduction
    @reduction.setter
    def reduction(self, v): self.func.reduction = v

    def __call__(self, input:Tensor, target:Tensor, **kwargs)->Rank0Tensor:
        input = input.transpose(self.axis,-1).contiguous()
        target = target.transpose(self.axis,-1).contiguous()
        if self.floatify: target = target.float()
            
        # Label smoothing experiment
        target = (target * 0.9 + 0.05)
        target[:,0] = 1

        input = input.view(-1,input.shape[-1]) if self.is_2d else input.view(-1)
        return self.func.__call__(input, target.view(-1), **kwargs)

    
def LabelSmoothBCEWithLogitsFlat(*args, axis:int=-1, floatify:bool=True, **kwargs):
    "Same as `nn.BCEWithLogitsLoss`, but flattens input and target."
    return FlattenedLoss(nn.BCEWithLogitsLoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)


# ## Get data

# In[ ]:


sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')


# In[ ]:


def get_data(seed, size=IMG_SIZE):
    df_train = get_train_df(seed)
    data = (
        ImageList.from_df(
            path='./',
            df=df_train,
            folder='.'
        )
    )
    data = (data.split_from_df('is_valid')
            .label_from_df('label', label_delim=',', label_cls=ReconstructFixMultiCategoryList)
            .transform(
                (train_tfms, val_tfms),
                resize_method=ResizeMethod.NO,
                padding_mode='zeros')
            .databunch(bs=BS)
            .normalize(imagenet_stats))
    data.add_test(ImageList.from_df(sample_df, '../input/aptos2019-blindness-detection', folder='test_images', suffix='.png'))
    return data


# In[ ]:


data = get_data(seed=1)


# In[ ]:


data.show_batch(figsize=(20, 16))


# In[ ]:


data.show_batch(figsize=(20, 16), ds_type=DatasetType.Valid)


# In[ ]:


data.show_batch(figsize=(20, 16), ds_type=DatasetType.Test)


# ## Get model

# In[ ]:


def get_efficientnet(name, pretrained, model_path):
    """Constructs a EfficientNetB5 model for FastAI.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = efficientnet_pytorch.EfficientNet.from_name(f'efficientnet-{name}', override_params={'num_classes': 5})
    if pretrained:
        model_state = torch.load(model_path)
        # load original weights apart from its head
        if '_fc.weight' in model_state.keys():
            model_state.pop('_fc.weight')
            model_state.pop('_fc.bias')
            res = model.load_state_dict(model_state, strict=False)
            print('Loaded pretrained')
            assert str(res.missing_keys) == str(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
        else:
            # A basic remapping is required
            from collections import OrderedDict
            mapping = { i:o for i,o in zip(model_state.keys(), model.state_dict().keys()) }
            mapped_model_state = OrderedDict([
                (mapping[k], v) for k,v in model_state.items() if not mapping[k].startswith('_fc')
            ])
            res = model.load_state_dict(mapped_model_state, strict=False)
    return model


# ## Training

# In[ ]:


start_time = time.time()


# In[ ]:


kappa = KappaScore(weights="quadratic")
change_data_cb = partial(ChangeDataOnEpoch, i=SEED)

model = get_efficientnet('b3', True, '../input/efficientnet-pytorch/efficientnet-b3-c8376fa2.pth')

learn = Learner(data, model, metrics=[kappa, accuracy_thresh], model_dir=".", callback_fns=[change_data_cb, BnFreeze])
learn.loss_func = LabelSmoothBCEWithLogitsFlat()
learn.split(lambda m: (m._conv_head,) );
learn = learn.to_fp16()
learn.freeze()


# In[ ]:


# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(1, 1e-2)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.unfreeze()


# In[ ]:


# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(15, 1e-3, callbacks=[SaveModelCallback(learn, name='best_model')])


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.load('best_model');


# In[ ]:


learn.validate()


# In[ ]:


duration = time.time() - start_time


# In[ ]:


print(f'Trained one fold in {duration} seconds')


# In[ ]:


val_items = learn.data.valid_dl.dataset.items 


# In[ ]:


val_preds, val_y = learn.get_preds(ds_type=DatasetType.Valid)


# In[ ]:


val_preds.shape


# In[ ]:


val_preds_df = pd.concat([
    pd.DataFrame({'id_code': [
        v.split('/')[-1].split('.')[0] for v in val_items
    ], 'diagnosis': val_y.argmax(1).numpy(), 'preds': get_preds((val_preds > 0.5).numpy())}),
    pd.DataFrame(val_preds.numpy())
], axis=1); val_preds_df.head(5)


# In[ ]:


val_preds_df.to_csv(f'{MODEL_NAME}_val_preds.csv')


# In[ ]:


metric = cohen_kappa_score(val_preds_df['diagnosis'], val_preds_df['preds'], weights='quadratic')


# In[ ]:


print(f'Val kappa score: {metric}')


# ## Make submission

# In[ ]:


start_time = time.time()


# In[ ]:


preds, y = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


preds


# In[ ]:


duration = time.time() - start_time


# In[ ]:


print(f'Made test predictions in {duration} seconds')


# In[ ]:


sample_df.diagnosis = get_preds((preds > 0.5).cpu().numpy())
sample_df.head(10)


# In[ ]:


sample_df.to_csv('submission.csv',index=False)


# In[ ]:


test_preds_df = pd.concat([
    sample_df,
    pd.DataFrame(preds.numpy())
], axis=1)
test_preds_df.head(5)


# In[ ]:


test_preds_df.to_csv('test_preds.csv', index=False)

