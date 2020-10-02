#!/usr/bin/env python
# coding: utf-8

# # Aptos 2019 - Best models blend (top 2-3%)
# 
# Simple weighted blend of a EfficientNet B3 with 256px images + EfficientNet B3 with 300px images and DenseNet101 with 320px images. Achieved 0.926 on the private leaderboard.
# 
# See [this](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107947) post for details about solution.

# In[ ]:


import time
import sys
import os

import cv2                  
         
from random import shuffle  
from zipfile import ZipFile

# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import numpy as np  
from tqdm import tqdm 
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from fastai import *
from fastai.vision import *
from fastai.callbacks import Callback
from fastai.callbacks import SaveModelCallback, EarlyStoppingCallback, ReduceLROnPlateauCallback

style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)


# ## Shared functions

# In[ ]:


def seed_everything(seed=999):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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
        pd.concat([
            test_labels_15_df[test_labels_15_df.diagnosis == 0].sample(n=7900, random_state=420),
            test_labels_15_df[test_labels_15_df.diagnosis != 0]
        ]).sample(n=10_000, random_state=420),
    ]).sample(frac=1, random_state=seed)

    df_train['label'] = df_train.diagnosis.apply(get_label)
    
    return df_train


def make_or_preds(model_name, learner, model_path, expected_val):
    learn.load(model_path);

    val_items = learn.data.valid_dl.dataset.items
    val_preds, val_y = learn.get_preds(ds_type=DatasetType.Valid)
    metric = cohen_kappa_score(val_y.argmax(1).numpy(), get_output_preds((val_preds > 0.5).numpy()), weights='quadratic')

    raw_preds = pd.DataFrame(val_preds.numpy())
    raw_preds.columns = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4']

    val_preds_df = pd.concat([
        pd.DataFrame({
            'id_code': [v.split('/')[-1].split('.')[0] for v in val_items],
            'diagnosis': val_y.argmax(1).numpy(),
            'preds': get_output_preds((val_preds > 0.5).numpy())
        }),
        raw_preds
    ], axis=1)

    val_preds_df.to_csv(f'{model_name}_val_preds.csv', index=False)

    test_items = learn.data.test_dl.dataset.items

    test_preds, __ = learn.get_preds(ds_type=DatasetType.Test)

    raw_test_preds = pd.DataFrame(test_preds.numpy())
    raw_test_preds.columns = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4']

    test_preds_df = pd.concat([
        pd.DataFrame({
            'id_code': [v.split('/')[-1].split('.')[0] for v in test_items],
            'preds': get_output_preds((test_preds > 0.5).numpy())
        }),
        raw_test_preds
    ], axis=1)

    test_preds_df.to_csv(f'{model_name}_test_preds.csv', index=False)

    print(f'Val kappa score: {metric} (expected: {expected_val})')


def avg_tta_score(model_name):
    no_flip = pd.read_csv(f'{model_name}_val_preds.csv').sort_values('id_code')
    flip = pd.read_csv(f'{model_name}-flip_val_preds.csv').sort_values('id_code')

    val_preds_avg = no_flip[['x_0', 'x_1', 'x_2', 'x_3', 'x_4']].values * 0.5 + flip[['x_0', 'x_1', 'x_2', 'x_3', 'x_4']].values * 0.5

    return cohen_kappa_score(flip.diagnosis, get_output_preds((val_preds_avg > 0.5)), weights='quadratic')


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


class ReconstructFixMultiCategoryList(MultiCategoryList):
    def reconstruct(self, t):
        try:
            return super().reconstruct(t)
        except Exception as e:
            return FloatItem(np.log(t))


# In[ ]:


sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')


# ## EfficientNet b3

# In[ ]:


SIGMA_X = 10

sys.path.append('../input/efficientnet-pytorch/efficientnet-pytorch/EfficientNet-PyTorch-master')
import efficientnet_pytorch

def get_data(seed, size, bs, tfms=((), ())):
    df_train = get_train_df(seed)
    ImageList.open = lambda self, fn: open_img(self, fn, size=size)
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
                tfms,
                resize_method=ResizeMethod.NO,
                padding_mode='zeros')
            .databunch(bs=bs)
            .normalize(imagenet_stats))
    data.add_test(ImageList.from_df(sample_df, '../input/aptos2019-blindness-detection', folder='test_images', suffix='.png'))
    return data


# To remove irregularities along the circular boundary of the image
PARAM = 96
def Radius_Reduction(img, PARAM):
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


def subtract_median_bg_image(im):
    k = np.max(im.shape)//20*2+1
    bg = cv2.medianBlur(im, k)
    return cv2.addWeighted (im, 4, bg, -4, 128)


def get_output_preds(arr):
    mask = arr == 0
    return np.clip(np.where(mask.any(1), mask.argmax(1), 5) - 1, 0, 4)


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


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img


def load_ben_color(path, img_size, sigmaX=SIGMA_X):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (img_size, img_size))
    image = cv2.addWeighted (image, 4, cv2.GaussianBlur(image, (0,0), sigmaX) , -4 ,128)
    return Image(pil2tensor(image, np.float32).div_(255))


def resize_image(im, img_size):
    # Crops, resizes and potentially augments the image to IMG_SIZE
    cx, cy, r = info_image(im)
    scaling = img_size/(2*r)
    rotation = 0
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


# ## EfficientNet B03

# In[ ]:


model = get_efficientnet('b3', True, '../input/efficientnet-pytorch/efficientnet-b3-c8376fa2.pth')


# In[ ]:


seed_everything(422)

data = get_data(seed=437, size=256, bs=64)
learn = Learner(data, model, model_dir=".", callback_fns=[BnFreeze])

make_or_preds(
    model_name='bd-efficientnet-b03-2015val-psu3-sz-256',
    learner=learn,
    model_path='../input/bd-efficientnet-b03-2015val-psu3-sz-256/best_model',
    expected_val=0.7273654273342949
)


# In[ ]:


seed_everything(422)

data = get_data(seed=437, size=256, bs=64, tfms=((), (flip_lr(p=1))))
learn = Learner(data, model, model_dir=".", callback_fns=[BnFreeze])

make_or_preds(
    model_name='bd-efficientnet-b03-2015val-psu3-sz-256-flip',
    learner=learn,
    model_path='../input/bd-efficientnet-b03-2015val-psu3-sz-256/best_model',
    expected_val=0.7273654273342949
)


# In[ ]:


avg_tta_score('bd-efficientnet-b03-2015val-psu3-sz-256')


# In[ ]:


seed_everything(420)

data = get_data(seed=420, size=300, bs=64)
learn = Learner(data, model, model_dir=".", callback_fns=[BnFreeze])

make_or_preds(
    model_name='bd-efficientnet-b03-2015val-psu3-sz-300',
    learner=learn,
    model_path='../input/bd-efficientnet-b03-2015val-psu3-sz-300/best_model',
    expected_val=0.7433966054032339
)


# In[ ]:


seed_everything(420)

data = get_data(seed=420, size=300, bs=64, tfms=((), (flip_lr(p=1))))
learn = Learner(data, model, model_dir=".", callback_fns=[BnFreeze])

make_or_preds(
    model_name='bd-efficientnet-b03-2015val-psu3-sz-300-flip',
    learner=learn,
    model_path='../input/bd-efficientnet-b03-2015val-psu3-sz-300/best_model',
    expected_val=0.7433966054032339
)


# In[ ]:


avg_tta_score('bd-efficientnet-b03-2015val-psu3-sz-300')


# ## DenseNet201

# In[ ]:


MODEL_NAME = 'densenet201_change_zeros_ord_reg_label_smoothing'

IMG_SIZE = 320
BS = 64
SEED = 423
SIGMA_X = 10


seed_everything(SEED)


def subtract_gaussian_bg_image(im):
    # k = np.max(im.shape)/10
    bg = cv2.GaussianBlur(im ,(0,0) , SIGMA_X)
    return cv2.addWeighted (im, 4, bg, -4, 128)


def open_img(self, fn, size):
    "Open image in `fn`, subclass and overwrite for custom behavior."
    image = cv2.imread(fn)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_image(image, size)

    # changing line here.
    image = subtract_gaussian_bg_image(image)
    image = Radius_Reduction(image, PARAM)
    return Image(pil2tensor(image, np.float32).div_(255))


def get_data(seed, size=IMG_SIZE, bs=BS, tfms=((), ())):
    df_train = get_train_df(seed)
    
    ImageList.open = lambda self, fn: open_img(self, fn, size=size)

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
                tfms,
                resize_method=ResizeMethod.NO,
                padding_mode='zeros')
            .databunch(bs=bs)
            .normalize(imagenet_stats))
    data.add_test(ImageList.from_df(sample_df, '../input/aptos2019-blindness-detection', folder='test_images', suffix='.png'))
    return data


# In[ ]:


data = get_data(seed=0, size=IMG_SIZE, bs=BS)
learn = cnn_learner(data, models.densenet201, model_dir=".", lin_ftrs=[2048], callback_fns=[BnFreeze], pretrained=False)

make_or_preds(
    model_name='bd-densenet201-2015val-psu3-2019-val',
    learner=learn,
    model_path='../input/bd-densenet201-2015val-psu3-2019-val/best_model',
    expected_val=0.690059677041891
)


# In[ ]:


data = get_data(seed=0, size=IMG_SIZE, bs=BS, tfms=((), (flip_lr(p=1))))
learn = cnn_learner(data, models.densenet201, model_dir=".", lin_ftrs=[2048], callback_fns=[BnFreeze], pretrained=False)

make_or_preds(
    model_name='bd-densenet201-2015val-psu3-2019-val-flip',
    learner=learn,
    model_path='../input/bd-densenet201-2015val-psu3-2019-val/best_model',
    expected_val=0.690059677041891
)


# In[ ]:


avg_tta_score('bd-densenet201-2015val-psu3-2019-val')


# ## Weighted Blend

# In[ ]:


columns = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4']

eff_b3_300_val_preds_no_flip = pd.read_csv('bd-efficientnet-b03-2015val-psu3-sz-300_val_preds.csv').sort_values('id_code')
eff_b3_300_val_preds_flip = pd.read_csv('bd-efficientnet-b03-2015val-psu3-sz-300-flip_val_preds.csv').sort_values('id_code')

eff_b3_300_val_preds = (eff_b3_300_val_preds_flip[columns] + eff_b3_300_val_preds_no_flip[columns]) / 2

eff_b3_256_val_preds_no_flip = pd.read_csv('bd-efficientnet-b03-2015val-psu3-sz-256_val_preds.csv').sort_values('id_code')
eff_b3_256_val_preds_flip = pd.read_csv('bd-efficientnet-b03-2015val-psu3-sz-256_val_preds.csv').sort_values('id_code')

eff_b3_256_val_preds = (eff_b3_256_val_preds_flip[columns] + eff_b3_256_val_preds_no_flip[columns]) / 2

densenet_201_val_preds_no_flip = pd.read_csv('bd-densenet201-2015val-psu3-2019-val_val_preds.csv').sort_values('id_code')
densenet_201_val_preds_flip = pd.read_csv('bd-densenet201-2015val-psu3-2019-val-flip_val_preds.csv').sort_values('id_code')

densenet_201_val_preds = (densenet_201_val_preds_flip[columns] + densenet_201_val_preds_no_flip[columns]) / 2


# In[ ]:


val_preds_avg = (
    eff_b3_300_val_preds.values * 0.4 +
    eff_b3_256_val_preds.values * 0.4 +
    densenet_201_val_preds.values * 0.2
)


# In[ ]:


preds = []
thres = []
for i in np.arange(0, 1, 0.005):
    thres.append(i)
    preds.append(cohen_kappa_score(
        eff_b3_300_val_preds_flip.sort_values('id_code').diagnosis,
        get_output_preds((val_preds_avg > i)), weights='quadratic'))


# In[ ]:


val_preds_and_threshold = sorted(list(zip(preds, thres)), key=lambda x: x[0], reverse=True)


# In[ ]:


val_preds_and_threshold[:10]


# In[ ]:


best_thresh = 0.45


# ## Make submission

# In[ ]:


columns = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4']

eff_b3_300_test_preds_no_flip = pd.read_csv('bd-efficientnet-b03-2015val-psu3-sz-300_test_preds.csv')
eff_b3_300_test_preds_flip = pd.read_csv('bd-efficientnet-b03-2015val-psu3-sz-300-flip_test_preds.csv')

eff_b3_300_test_preds = (eff_b3_300_test_preds_flip[columns] + eff_b3_300_test_preds_no_flip[columns]) / 2

eff_b3_256_test_preds_no_flip = pd.read_csv('bd-efficientnet-b03-2015val-psu3-sz-256_test_preds.csv')
eff_b3_256_test_preds_flip = pd.read_csv('bd-efficientnet-b03-2015val-psu3-sz-256-flip_test_preds.csv')

eff_b3_256_test_preds = (eff_b3_256_test_preds_flip[columns] + eff_b3_256_test_preds_no_flip[columns]) / 2

densenet_201_test_preds_no_flip = pd.read_csv('bd-densenet201-2015val-psu3-2019-val_test_preds.csv').sort_values('id_code')
densenet_201_test_preds_flip = pd.read_csv('bd-densenet201-2015val-psu3-2019-val-flip_test_preds.csv').sort_values('id_code')

densenet_201_test_preds = (densenet_201_test_preds_flip[columns] + densenet_201_test_preds_no_flip[columns]) / 2


# In[ ]:


test_preds_avg = (
    eff_b3_300_test_preds.values * 0.4 +
    eff_b3_256_test_preds.values * 0.4 +
    densenet_201_test_preds.values * 0.2
)


# In[ ]:


sample_df.diagnosis = get_output_preds((test_preds_avg > best_thresh)) 


# In[ ]:


sample_df.to_csv('submission.csv', index=False)


# In[ ]:


pd.read_csv('submission.csv').head(20)


# In[ ]:


pd.read_csv('submission.csv').diagnosis.value_counts()


# In[ ]:




