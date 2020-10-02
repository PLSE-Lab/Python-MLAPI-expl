#!/usr/bin/env python
# coding: utf-8

# ## Chainer Starter Kernel for iMet Collection 2019 - FGVC6
# 
# ### What is this kernel?
# 
# This is a baseline example using [Chainer](https://docs.chainer.org/en/v5.3.0/) and [ChainerCV](https://chainercv.readthedocs.io/en/stable/).  
# I share this kernel mainly for practice of writing kernel, and for sharing some (maybe useful) information e.g. training settings.
# 
# ### Summary of model, training, and inference
# #### base model: SEResNet152
# * pre-trained weights on ImageNet
# * obtain output of Global Average Pooling Layer (pool5) and feed it to Dense Layer
# * input shape: (ch, height, witdh) = (3, 128, 128)
# * preprocessing for images
#     * subtract per-channel mean of all train images, and devide by 255 (after data augmentation)
# 
# #### training
# * fine-tuning all over the model, not freezing any layer
# * data augmentation
#     * horizontal flip
#     * [random_distort](https://chainercv.readthedocs.io/en/stable/reference/links/ssd.html#random-distort)
#     * random_rotate(angle $\in$ [-10, 10])
#     * random_expand => [resize_with_random_interpolation](https://chainercv.readthedocs.io/en/stable/reference/links/ssd.html#resize-with-random-interpolation)
#     * random crop
# * max epoch: 20
# * batch size: 128
# * optimizer: NesterovSGD
#     * momentum = 0.9, weight decay = 1e-04
# * learning schedule: cosine anealing
#     * max_lr = 0.01, min_lr = 0.0001
#     * I ran only one cycle.
#     * Note: decaying learning rate by epoch, not iteration (for simple implementation)
# * loss: Focal Loss
#     * alpha = 0.5, gamma = 2
#     * Note:
#         * Loss for each sample is calculated by **summation** of focal loss for each class.
#         * Loss for mini-batch  is calculated by averaging loss for samples in it.
#             * At first I calculated each sample's loss by **averaging**, but it didn't work well.  
# * validation
#     * make one validation set, **not** perform k-fold cross validation
#     * **randomly** split, not considering target(attribute) frequency
#         * train : valid = 4 : 1
#     * check each epoch's f-beta score by threshold = 0.2
#         * since f-beta weights recall higher than precision
#         * [default threshold of fastai's implementation is 0.2](https://github.com/fastai/fastai/blob/master/fastai/metrics.py#L12)
# 
# #### inference
# * not using TTA
# * using best threshold for validation set
#     * Thresholds for all classes are same. 

# ## setup

# ### import

# In[ ]:


from time import time
beginning_time = time()


# In[ ]:


import os
import gc
import json
import sys
import random
from glob import glob

from PIL import Image
from collections import OrderedDict
from joblib import Parallel, delayed
from tqdm._tqdm_notebook import tqdm_notebook

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import cos, pi
from sklearn.metrics import precision_recall_curve

import seaborn as sns
from matplotlib import pyplot as plt

tqdm_notebook.pandas()
get_ipython().run_line_magic('matplotlib', 'inline')
print(os.listdir("../input"))


# #### import Chainer and ChainerCV

# In[ ]:


import chainer
from chainer import cuda, functions, links, datasets
from chainer import iterators, optimizers, training, reporter
from chainer import initializers, serializers

import chainercv
from chainercv import transforms
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation

print("chainercsv:", chainercv.__version__)
print("chainer:", chainer.__version__)


# #### set data path

# In[ ]:


DATA_DIR = "../input/imet-2019-fgvc6"
PRETRAINED_MODEL_DIR = "../input/chainercv-seresnet"


# In[ ]:


print("../input")
for path in glob("../input/*"):
    print("\t|- {}/".format(path.split("/")[-1]))
    for fname in os.listdir(path):
        print("\t\t|-{}".format(fname))


# ### set other imformation.

# In[ ]:


img_chmean_train = np.array([164.82181258, 155.93463791, 144.58968491], dtype="f")
# img_chmean_test = np.array([163.39652723, 154.7340003 , 143.86426686], dtype="f")


# ### settings

# In[ ]:


seed = 1086
settings = OrderedDict(
    # # model setting.
    base_model="SEResNet152",
    n_class=1103,
    image_size=[128, 128],
    
    # # training setting.
    # valid_fold=0,
    max_epoch=20,
    batch_size=128,
    da_select=[
        "random_distort",
        "random_lr_flip",
        "random_rotate",
        "random_expand", "resize_with_random",
        "random_crop"
    ],
    learning_schedule="cosine",
    epoch_per_cycle=20,
    optimizer="NesterovAG",
    learning_rate=0.01,
    learning_rate_min=0.0001,
    momentum=0.9,
    weight_decay_rate=1e-04,
    loss_func="FocalLoss",
    alpha=0.5,
    gamma=2,
)
settings["pretrained_model_path"] = "{}/{}".format(
    PRETRAINED_MODEL_DIR,
    {
        "SEResNet50": "se_resnet50_imagenet_converted_2018_06_25.npz",
        "SEResNet101": "se_resnet101_imagenet_converted_2018_06_25.npz",
        "SEResNet152": "se_resnet152_imagenet_converted_2018_06_25.npz",
    }[settings["base_model"]])


# ### classes and functions definition

# #### model
# * CNN model
# * wapper for training

# In[ ]:


base_class = getattr(chainercv.links, settings["base_model"])

class FeatureExtractor(base_class):
    """image feture extractor based on pretrained model."""
    
    def __init__(self, pretrained_model_path, extract_layers=["pool5"]):
        """Initialze."""
        super(FeatureExtractor, self).__init__(pretrained_model=pretrained_model_path)
        self._pick = extract_layers
        self.remove_unused()
    
    def __call__(self, x):
        """Simply Forward."""
        h = x
        for name in self.layer_names:
            h = self[name](h)
        return h
    
class Ext2Linear(chainer.Chain):
    """Chain to feed output of Extractor to Fully Connect."""
    
    def __init__(self, n_class, extractor):
        """Initialize."""
        super(Ext2Linear, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.fc = links.Linear(
                None, n_class, initialW=initializers.Normal(scale=0.01))

    def __call__(self, x):
        """Forward."""
        return self.fc(self.extractor(x))


# In[ ]:


class MultiLabelClassifier(links.Classifier):
    """Wrapper for multi label classification model."""
    
    def __init__(self, predictor, lossfun):
        """Initialize"""
        super(MultiLabelClassifier, self).__init__(predictor, lossfun)
        self.compute_accuracy = False
        self.f_beta = None
        self.metfun = self._fbeta_score
        
    def __call__(self, x, t):
        """Foward. calc loss and evaluation metric."""
        loss = super().__call__(x, t)
        self.f_beta = None
        self.f_beta = self.metfun(self.y, t)
        reporter.report({'f-beta': self.f_beta}, self)
        
        return loss
    
    def _fbeta_score(self, y_pred, t, beta=2, th=0.2, epsilon=1e-09):
        """
        calculate f-beta score.
        
        calculate f-bata score along **class-axis(axis=1)** and average them along sample-axis.
        """
        y_prob = functions.sigmoid(y_pred).data
        t_pred = (y_prob >= th).astype("i")
        true_pos = (t_pred * t).sum(axis=1)  # tp
        pred_pos = t_pred.sum(axis=1)  # tp + fp
        poss_pos = t.sum(axis=1)  # tp + fn
        precision = true_pos / (pred_pos + epsilon)
        recall = true_pos / (poss_pos + epsilon)
        f_beta_each_id = (1 + beta ** 2) * precision * recall / ((beta ** 2) * precision + recall + epsilon)
        return functions.mean(f_beta_each_id)


# #### loss
# * define Focal Loss

# In[ ]:


class FocalLoss:
    """
    Function for Focal loss.
    
    calculates focal loss **for each class**, **sum up** them along class-axis, and **average** them along sample-axis.
    Take data point x and its logit y = model(x),
    using prob p = (p_0, ..., p_C)^T = sigmoid(y) and label t,
    focal loss for each class i caluculated by:
    
        loss_{i}(p, t) = - \alpha' + (1 - p'_i) ** \gamma * ln(p'_i),
    
    where
        \alpha' = { \alpha (t_i = 1)
                  { 1 - \alpha (t_i = 0)
         p'_i   = { p_i (t_i = 1)
                = ( 1 - p_i (t_i = 0)
    """

    def __init__(self, alpha=0.25, gamma=2):
        """Initialize."""
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def __call__(self, y_pred, t, epsilon=1e-31):
        """
        Forward.
        
        p_dash = t * p + (1 - t) * (1 - p) = (1 - t) + (2 * t - 1) * p
        """
        p_dash = functions.clip(
            (1 - t) + (2 * t - 1) * functions.sigmoid(y_pred), epsilon, 1 - epsilon)
        alpha_dash = (1 - t) + (2 * t - 1) * self.alpha
        # # [y_pred: (bs, n_class), t: (bs: n_class)] => loss_by_sample_x_class: (bs, n_class)
        loss_by_sample_x_class = - alpha_dash * (1 - p_dash) ** self.gamma * functions.log(p_dash)
        # # loss_by_sample_x_class: (bs, n_class) => loss_by_sample: (bs, )
        loss_by_sample = functions.sum(loss_by_sample_x_class, axis=1)
        # # loss_by_sample: (bs,) => loss: (1, )
        return functions.mean(loss_by_sample)


# #### datasets
# * image preprocessing functions 
# * data augmentor class

# In[ ]:


def resize_pair(pair, size=settings["image_size"]):
    img, label = pair
    img = transforms.resize(img, size=size)
    return (img, label)

def scale_and_subtract_mean(pair, mean_value=img_chmean_train):
    img, label = pair
    img = (img - mean_value[:, None, None]) / 255.
    return (img, label)


# In[ ]:


class DataAugmentor():
    """DataAugmentor for Image Classification."""

    def __init__(
        self, image_size, image_mean,
        using_methods=["random_ud_flip","random_lr_flip", "random_90_rotate", "random_crop"]
    ):
        """Initialize."""
        self.img_size = image_size
        self.img_mean = image_mean
        self.img_mean_single = int(image_mean.mean())
        self.using_methods = using_methods
        
        self.func_dict = {
            "random_ud_flip": self._random_ud_flip,
            "random_lr_flip": self._random_lr_flip,
            "random_90_rotate": self._random_90_rotate,
            "random_rotate": self._random_rotate,
            "random_expand": self._random_expand,
            "resize_with_random": self._resize_with_random,
            "random_crop": self._random_crop,
            "random_distort": random_distort,
        }
        # # set da func by given order.
        self.da_funcs = [self.func_dict[um] for um in using_methods]
        
    def __call__(self, pair):
        """Forward"""
        img_arr, label = pair

        for func in self.da_funcs:
            img_arr = func(img_arr)
            
        return img_arr, label

    def _random_lr_flip(self, img_arr):
        """left-right flipping."""
        if np.random.randint(2):
            img_arr = img_arr[:, :, ::-1]
        return img_arr

    def _random_ud_flip(self, img_arr):
        """up-down flipping."""
        if np.random.randint(2):
            img_arr = img_arr[:, ::-1, :]
        return img_arr
    
    def _random_90_rotate(self, img_arr):
        """90 angle rotation."""
        if np.random.randint(2):
            img_arr = img_arr.transpose(0, 2, 1)[:, ::-1, :]
        return img_arr
    
    def _random_rotate(self, img_arr, max_angle=10):
        """random degree rotation."""
        angle = np.random.randint(-max_angle, max_angle + 1)
        if angle == 0:
            return img_arr
        return transforms.rotate(img_arr, angle, fill=self.img_mean_single)

    def _random_expand(self, img_arr):
        """random expansion"""
        if np.random.randint(2):
            return img_arr
        return transforms.random_expand(img_arr, fill=self.img_mean)

    def _resize_with_random(self, img_arr):
        """resize with random interpolation"""
        if img_arr.shape[-2:] == self.img_size:
            return img_arr
        return resize_with_random_interpolation(img_arr, self.img_size)
    
    def _random_crop(self, img_arr, rate=0.5):
        """Random Cropping."""
        crop_size = self.img_size
        resize_size = tuple(map(lambda x: int(x * 256 / 224), self.img_size))

        if np.random.randint(2):
            top = np.random.randint(0, resize_size[0] - crop_size[0])
            botom = top + crop_size[0]
            left = np.random.randint(0, resize_size[1] - crop_size[1])
            right = left + crop_size[1]
            img_arr = transforms.resize(img_arr, size=resize_size)[:, top:botom, left: right]

        return img_arr


# #### training
# * trainer extention for cosine anealing

# In[ ]:


class CosineShift(chainer.training.extension.Extension):
    """
    Cosine Anealing.
    
    reference link: https://github.com/takedarts/resnetfamily/blob/master/src/mylib/training/extensions/cosine_shift.py
    """
    def __init__(self, attr, value, period, period_mult=1, optimizer=None):
        self._attr = attr
        self._value = value
        self._period = period
        self._period_mult = period_mult
        self._optimizer = optimizer

        if not hasattr(self._value, '__getitem__'):
            self._value = (self._value, 0)

    def initialize(self, trainer):
        self._update_value(trainer)

    def __call__(self, trainer):
        self._update_value(trainer)

    def _update_value(self, trainer):
        optimizer = self._optimizer or trainer.updater.get_optimizer('main')
        epoch = trainer.updater.epoch

        period_range = self._period
        period_start = 0
        period_end = period_range

        while period_end <= epoch:
            period_start = period_end
            period_range *= self._period_mult
            period_end += period_range

        n_max, n_min = self._value
        t_cur = epoch - period_start
        t_i = period_range
        value = n_min + 0.5 * (n_max - n_min) * (1 + cos((t_cur / t_i) * pi))

        setattr(optimizer, self._attr, value)


# #### inference
# * perform predict
# * calculate f-beta score
# * find threshold
# * make pred_ids

# In[ ]:


def predict(model, val_iter, gpu_device=-1):
    val_pred_list = []
    val_label_list = []
    iter_num = 0
    epoch_test_start = time()

    while True:
        val_batch = val_iter.next()
        iter_num += 1
        print("\rtmp_valid_iteration: {:0>5}".format(iter_num), end="")
        feature_val, label_val = chainer.dataset.concat_examples(val_batch, gpu_device)

        # Forward the test data
        with chainer.no_backprop_mode() and chainer.using_config("train", False):
            prediction_val = model(feature_val)
            val_pred_list.append(prediction_val)
            val_label_list.append(label_val)
            prediction_val.unchain_backward()

        if val_iter.is_new_epoch:
            print(" => valid end: {:.2f} sec".format(time() - epoch_test_start))
            val_iter.epoch = 0
            val_iter.current_position = 0
            val_iter.is_new_epoch = False
            val_iter._pushed_position = None
            break

    val_pred_all = cuda.to_cpu(functions.concat(val_pred_list, axis=0).data)
    val_label_all = cuda.to_cpu(functions.concat(val_label_list, axis=0).data)
    return val_pred_all, val_label_all


# In[ ]:


def average_fbeta_score(y_prob, t, th=0.2, beta=2, epsilon=1e-09):
    t_pred = (y_prob >= th).astype(int)
    # # t_pred, t: (sample_num, n_class) => true_pps, predicted_pos, poss_pos : (sample_num,)
    true_pos = (t_pred * t).sum(axis=1)
    pred_pos = t_pred.sum(axis=1)
    poss_pos = t.sum(axis=1)

    p_arr = true_pos / (pred_pos + epsilon)
    r_arr = true_pos / (poss_pos + epsilon)
    # # p_arr, r_arr : (n_class,) => f_beta: (n_class)
    f_beta = (1 + beta ** 2) * p_arr * r_arr / ((beta ** 2) * p_arr + r_arr + epsilon)
    return f_beta.mean()

def search_best_threshold(y_prob, t, eval_func, search_range=[0.05, 0.95], interval=0.01):
    tmp_th = search_range[0]
    best_th = 0
    best_eval = -(10**9 + 7)
    while tmp_th < search_range[1]:
        eval_score = eval_func(y_prob, t, th=tmp_th)
        print(tmp_th, eval_score)
        if eval_score > best_eval:
            best_th = tmp_th
            best_eval = eval_score
        tmp_th += interval
    return best_th, best_eval

def make_pred_ids(test_pred, th):
    class_array = np.arange(test_pred.shape[1])
    test_cond = test_pred >= th
    pred_ids = [" ".join(map(str, class_array[cond])) for cond in test_cond]
    return pred_ids


# In[ ]:


print("[end of setup]: {:.3f}".format(time() - beginning_time))


# ## prepare data

# ### read data

# In[ ]:


train_df = pd.read_csv("{}/train.csv".format(DATA_DIR))
test_df = pd.read_csv("{}/sample_submission.csv".format(DATA_DIR))
labels_df = pd.read_csv("{}/labels.csv".format(DATA_DIR))


# ### make datasets

# #### label

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_attr_ohot = np.zeros((len(train_df), len(labels_df)), dtype="i")\nfor idx, attr_arr in enumerate(\n    train_df.attribute_ids.str.split(" ").apply(lambda l: list(map(int, l))).values):\n    train_attr_ohot[idx, attr_arr] = 1')


# In[ ]:


print(train_attr_ohot.shape)
train_attr_ohot[:5,:20]


# #### labeled datasets

# In[ ]:


all_train_dataset = datasets.LabeledImageDataset(
    pairs=list(zip((train_df.id + ".png").tolist(), train_attr_ohot)),
    root="{}/train".format(DATA_DIR))

# # For test set, I set dummy label.
test_dataset = datasets.LabeledImageDataset(
    pairs=list(zip((test_df.id + ".png").tolist(), [-1] * len(test_df))),
    root="{}/test".format(DATA_DIR))


# #### split all train into train and valid
# Here, I split train randomly.

# In[ ]:


train_dataset, valid_dataset = datasets.split_dataset_random(
    all_train_dataset, first_size=int(len(all_train_dataset) * 0.8), seed=seed)
print("train set:", len(train_dataset))
print("valid set:", len(valid_dataset))


# #### set transforms and data augmentations

# In[ ]:


# # train set, including data augmentation
train_dataset = datasets.TransformDataset(train_dataset, resize_pair)
train_dataset = datasets.TransformDataset(
    train_dataset, DataAugmentor(
        image_size=settings["image_size"], image_mean=img_chmean_train,
        using_methods=settings["da_select"])
)
train_dataset = datasets.TransformDataset(train_dataset, scale_and_subtract_mean)


# In[ ]:


# # validt set.
valid_dataset = datasets.TransformDataset(valid_dataset, resize_pair)
valid_dataset = datasets.TransformDataset(valid_dataset, scale_and_subtract_mean)
# # test set.
test_dataset = datasets.TransformDataset(test_dataset, resize_pair)
test_dataset = datasets.TransformDataset(test_dataset, scale_and_subtract_mean)


# In[ ]:


print("[end of preparing data]: {:.3f}".format(time() - beginning_time))


# ## make trainer

# ### model

# In[ ]:


model = Ext2Linear(
    settings["n_class"], FeatureExtractor(settings["pretrained_model_path"]))
train_model = MultiLabelClassifier(
    model, lossfun=FocalLoss(settings["alpha"], settings["gamma"]))


# ### optimizer

# In[ ]:


opt_class = getattr(optimizers, settings["optimizer"])
if settings["optimizer"] != "Adam":
    optimizer = opt_class(lr=settings["learning_rate"], momentum=settings["momentum"])
    optimizer.setup(train_model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(settings["weight_decay_rate"]))
else:
    optmizer = opt_class(
        alpha=settings["learning_rate"], weight_decay_rate=settings["weight_decay_rate"])
    optimizer.setup(train_model)


# ### iterator

# In[ ]:


train_iter = iterators.MultiprocessIterator(train_dataset, settings["batch_size"])
valid_iter = iterators.MultiprocessIterator(
    valid_dataset, settings["batch_size"], repeat=False, shuffle=False)


# ### updater, trainer

# In[ ]:


updater = training.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.trainer.Trainer(updater, stop_trigger=(settings["max_epoch"], "epoch"), out="training_result")


# ### trainer extensions

# In[ ]:


logging_attributes = ["epoch", "main/loss", "val/main/loss", "val/main/f-beta", "elapsed_time", "lr"]

# # cosine anealing.
trainer.extend(
    CosineShift('lr', [settings["learning_rate"], settings["learning_rate_min"]], settings["epoch_per_cycle"], 1))
# # evaluator.
trainer.extend(
    training.extensions.Evaluator(valid_iter, optimizer.target, device=0), name='val',trigger=(1, 'epoch'))
# # log.
trainer.extend(training.extensions.observe_lr(), trigger=(1, 'epoch'))
trainer.extend(training.extensions.LogReport(logging_attributes), trigger=(1, 'epoch'))
# # standard output.
trainer.extend(training.extensions.PrintReport(logging_attributes), trigger=(1, 'epoch'))
trainer.extend(training.extensions.ProgressBar(update_interval=200))
# # plots.
trainer.extend(
    training.extensions.PlotReport(["main/loss", "val/main/loss"], "epoch", file_name="loss.png"), trigger=(1, "epoch"))
trainer.extend(
    training.extensions.PlotReport(["val/main/f-beta"], "epoch", file_name="fbeta_at02.png"), trigger=(1, "epoch"))
# # snapshot.
trainer.extend(
    training.extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'), trigger=(10, 'epoch'))


# In[ ]:


print("[end of preparing trainer]: {:.3f}".format(time() - beginning_time))


# In[ ]:


gc.collect()


# ## training

# In[ ]:


get_ipython().run_cell_magic('time', '', 'trainer.run()')


# In[ ]:


# # save last model
trained_model = trainer.updater.get_optimizer('main').target.predictor
serializers.save_npz('{}/epoch{:0>3}.model'.format("training_result", settings["max_epoch"]), trained_model)


# ## inference

# ### find best thr for valid set

# In[ ]:


valid_iter = iterators.MultiprocessIterator(
    valid_dataset, settings["batch_size"], repeat=False, shuffle=False)
val_pred, val_label = predict(trained_model, valid_iter, gpu_device=0)


# In[ ]:


val_prob = functions.sigmoid(val_pred).data
best_th, best_fbeta = search_best_threshold(
    val_prob, val_label, eval_func=average_fbeta_score, search_range=[0.1, 0.9], interval=0.01)
print(best_th, best_fbeta)


# ### submit

# In[ ]:


test_iter = iterators.MultiprocessIterator(
    test_dataset, settings["batch_size"], repeat=False, shuffle=False)
test_pred, _ = predict(trained_model, test_iter, gpu_device=0)


# In[ ]:


test_prob = functions.sigmoid(test_pred).data
test_pred_ids = make_pred_ids(test_prob, th=best_th)


# In[ ]:


test_df.attribute_ids = test_pred_ids
print(test_df.shape)
test_df.head()


# In[ ]:


test_df.to_csv("submission.csv", index=False)

