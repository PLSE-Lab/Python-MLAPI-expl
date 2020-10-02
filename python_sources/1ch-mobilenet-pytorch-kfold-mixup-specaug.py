#!/usr/bin/env python
# coding: utf-8

# ### imports

# In[ ]:


import gc
import os
import pickle
import random
import time
from collections import Counter, defaultdict
from operator import itemgetter
from functools import partial
from pathlib import Path
from psutil import cpu_count

import numpy as np
import pandas as pd

import librosa
from PIL import Image, ImageOps

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, Optimizer, RMSprop, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, _LRScheduler
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from fastprogress import master_bar, progress_bar


# ### utils

# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 666
seed_everything(SEED)


# In[ ]:


N_JOBS = cpu_count()
os.environ['MKL_NUM_THREADS'] = str(N_JOBS)
os.environ['OMP_NUM_THREADS'] = str(N_JOBS)
DataLoader = partial(DataLoader, num_workers=N_JOBS)


# In[ ]:


# from official code https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8#scrollTo=cRCaCIb9oguU
def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class


# ### dataset

# In[ ]:


os.listdir('../input/creating-1ch-fat2019-preprocessed-data/work/fat2019_prep_mels1')


# In[ ]:


dataset_dir = Path('../input/freesound-audio-tagging-2019')
preprocessed_dir = Path('../input/creating-1ch-fat2019-preprocessed-data/work/fat2019_prep_mels1')


# In[ ]:


csvs = {
    'train_curated': dataset_dir / 'train_curated.csv',
    'trn_curated_trimmed': preprocessed_dir / 'trn_curated_trimmed.csv',
    'train_noisy': preprocessed_dir / 'trn_noisy_best50s.csv',
    'sample_submission': dataset_dir / 'sample_submission.csv',
}

dataset = {
    'train_curated': dataset_dir / 'train_curated',
    'train_noisy': dataset_dir / 'train_noisy',
    'test': dataset_dir / 'test',
}

mels = {
    'train_curated': preprocessed_dir / 'mels_train_curated.pkl',
    'train_noisy': preprocessed_dir / 'mels_trn_noisy_best50s.pkl',
    'test': preprocessed_dir / 'mels_test.pkl',  # NOTE: this data doesn't work at 2nd stage
}


# In[ ]:


train_df = pd.read_csv(csvs['trn_curated_trimmed'])# pd.concat([pd.read_csv(csvs['trn_curated_trimmed']), pd.read_csv(csvs['train_noisy'])], sort=True, ignore_index=True)
#train_noisy = pd.read_csv(csvs['train_noisy'])
#train_df = pd.concat([train_curated, train_noisy], sort=True, ignore_index=True)
#train_df.head()


# In[ ]:


test_df = pd.read_csv(csvs['sample_submission'])
#test_df.head()


# In[ ]:


labels = test_df.columns[1:].tolist()
#labels


# In[ ]:


num_classes = len(labels)
#num_classes


# In[ ]:


y_train = np.zeros((len(train_df), num_classes)).astype(int)
for i, row in enumerate(train_df['labels'].str.split(',')):
    for label in row:
        idx = labels.index(label)
        y_train[i, idx] = 1

y_train.shape


# In[ ]:


# with open(mels['train_curated'], 'rb') as curated, open(mels['train_noisy'], 'rb') as noisy:
#     x_train = pickle.load(curated)
#     x_train.extend(pickle.load(noisy))
with open(mels['train_curated'], 'rb') as curated:
    x_train = pickle.load(curated)

with open(mels['test'], 'rb') as test:
    x_test = pickle.load(test)
    
len(x_train), len(x_test)


# In[ ]:


def spec_augment(spec: np.ndarray, num_mask=2, 
                 freq_masking_max_percentage=0.3, time_masking_max_percentage=0.3):

    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
        
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)
        
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0
    
    return spec


# In[ ]:


class FATTrainDataset(Dataset):
    def __init__(self, mels, labels, transforms, crop=True):
        super().__init__()
        self.mels = mels
        self.labels = labels
        self.transforms = transforms
        self.crop = crop
        
    def __len__(self):
        return len(self.mels)
    
    def __getitem__(self, idx):
        mel = self.mels[idx]
        base_dim, time_dim = mel.shape
        max_masks = max(2, time_dim // base_dim)
        mel = spec_augment(mel, num_mask=random.randint(0, max_masks))
        if self.crop:
            crop = random.randint(0, time_dim - base_dim)
            mel = mel[:, crop:crop + base_dim]
        mel = self.transforms(mel)
        
        label = self.labels[idx]
        label = torch.from_numpy(label).float()
        
        return mel, label


# In[ ]:


class FATTestDataset(Dataset):
    def __init__(self, fnames, mels, transforms, crop=False):
        super().__init__()
        self.fnames = fnames
        self.mels = mels
        self.transforms = transforms
        self.crop = crop
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        new_idx = idx % len(self.fnames)
        
        mel = self.mels[new_idx]
        base_dim, time_dim = mel.shape
        max_masks = max(2, time_dim // base_dim)
        mel = spec_augment(mel, num_mask=random.randint(0, max_masks))
        if self.crop:
            crop = random.randint(0, time_dim - base_dim)
            mel = mel[:, crop:crop + base_dim]
        mel = self.transforms(mel)

        fname = self.fnames[new_idx]
        
        return mel, fname


# In[ ]:


transforms_dict = {
    'train': transforms.Compose([
        #transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        #transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ]),
}


# ### model

# In[ ]:


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(1, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


# In[ ]:


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mobilenet = MobileNetV2(num_classes=num_classes)

    def forward(self, x):
        return self.mobilenet(x)


# In[ ]:


Classifier(num_classes=num_classes)


# ### train

# In[ ]:


def get_1cycle_schedule(lr_max=1e-3, n_data_points=8000, epochs=200, batch_size=40):          
    """
    Creates a look-up table of learning rates for 1cycle schedule with cosine annealing
    See @sgugger's & @jeremyhoward's code in fastai library: https://github.com/fastai/fastai/blob/master/fastai/train.py
    Wrote this to use with my Keras and (non-fastai-)PyTorch codes.
    Note that in Keras, the LearningRateScheduler callback (https://keras.io/callbacks/#learningratescheduler) only operates once per epoch, not per batch
      So see below for Keras callback

    Keyword arguments:
    lr_max            chosen by user after lr_finder
    n_data_points     data points per epoch (e.g. size of training set)
    epochs            number of epochs
    batch_size        batch size
    Output:  
    lrs               look-up table of LR's, with length equal to total # of iterations
    Then you can use this in your PyTorch code by counting iteration number and setting
          optimizer.param_groups[0]['lr'] = lrs[iter_count]
    """
    pct_start, div_factor = 0.3, 25.        # @sgugger's parameters in fastai code
    lr_start = lr_max/div_factor
    lr_end = lr_start/1e4
    n_iter = n_data_points * epochs // batch_size     # number of iterations
    a1 = int(n_iter * pct_start)
    a2 = n_iter - a1

    # make look-up table
    lrs_first = np.linspace(lr_start, lr_max, a1)            # linear growth
    lrs_second = (lr_max-lr_end)*(1+np.cos(np.linspace(0,np.pi,a2)))/2 + lr_end  # cosine annealing
    lrs = np.concatenate((lrs_first, lrs_second))
    return lrs

class OneCycleScheduler(_LRScheduler):
    """My modification of Keras' Learning rate scheduler to do 1Cycle learning
       which increments per BATCH, not per epoch
    Keyword arguments
        **kwargs:  keyword arguments to pass to get_1cycle_schedule()
        Also, verbose: int. 0: quiet, 1: update messages.
        
    Sample usage (from my train.py):
        lrsched = OneCycleScheduler(lr_max=1e-4, n_data_points=X_train.shape[0], epochs=epochs, batch_size=batch_size, verbose=1)
    """
    def __init__(self, optimizer, **kwargs):
        self.lrs = get_1cycle_schedule(**kwargs)
        super(OneCycleScheduler, self).__init__(optimizer)
    
    def get_lr(self):
        return [self.lrs[self.last_epoch % len(self.lrs)]]


# In[ ]:


class CyclicLR(_LRScheduler):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.

    This implementation was adapted from the github repo: `bckenstler/CLR`_

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for each parameter group.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size_up (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        step_size_down (int): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up. Default: None
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            If specified, then 'mode' is ignored.
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Initial momentum which is the
            lower boundary in the cycle for each parameter group.
            Default: 0.8
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the difference of max_momentum
            and some scaling of the amplitude; therefore
            base_momentum may not actually be reached depending on
            scaling function. Default: 0.9
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()


    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self,
                 optimizer,
                 base_lr,
                 max_lr,
                 step_size_up=2000,
                 step_size_down=None,
                 mode='triangular',
                 gamma=1.,
                 scale_fn=None,
                 scale_mode='cycle',
                 cycle_momentum=True,
                 base_momentum=0.8,
                 max_momentum=0.9,
                 last_epoch=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        base_lrs = self._format_param('base_lr', optimizer, base_lr)
        if last_epoch == -1:
            for lr, group in zip(base_lrs, optimizer.param_groups):
                group['lr'] = lr

        self.max_lrs = self._format_param('max_lr', optimizer, max_lr)

        step_size_up = float(step_size_up)
        step_size_down = float(step_size_down) if step_size_down is not None else step_size_up
        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size

        if mode not in ['triangular', 'triangular2', 'exp_range']                 and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if 'momentum' not in optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')

            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for momentum, group in zip(base_momentums, optimizer.param_groups):
                    group['momentum'] = momentum
        self.base_momentums = list(map(lambda group: group['momentum'], optimizer.param_groups))
        self.max_momentums = self._format_param('max_momentum', optimizer, max_momentum)

        super(CyclicLR, self).__init__(optimizer, last_epoch)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1. + self.last_epoch / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_epoch)
            lrs.append(lr)

        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                base_height = (max_momentum - base_momentum) * scale_factor
                if self.scale_mode == 'cycle':
                    momentum = max_momentum - base_height * self.scale_fn(cycle)
                else:
                    momentum = max_momentum - base_height * self.scale_fn(self.last_epoch)
                momentums.append(momentum)
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['momentum'] = momentum

        return lrs


# In[ ]:


import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """Implements AdamW algorithm.

    It has been proposed in `Fixing Weight Decay Regularization in Adam`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. Fixing Weight Decay Regularization in Adam:
    https://arxiv.org/abs/1711.05101
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # according to the paper, this penalty should come after the bias correction
                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'], p.data)

        return loss


# In[ ]:


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# In[ ]:


def train_model(x_trn, x_val, y_trn, y_val, train_transforms, model, num_epochs=100, lr=1e-3, batch_size=71, fine_tune=False, weight_file_name='weight_best.pt'):
    test_batch_size = batch_size
    eta_min = 1e-7
    t_max = 20
    mixup_alpha = 1.0
    validation_tta = 10
    
    num_classes = y_train.shape[1]
    
    train_dataset = FATTrainDataset(x_trn, y_trn, train_transforms)
    valid_dataset = FATTrainDataset(x_val, y_val, train_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=test_batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss().cuda()
    if fine_tune:
        optimizer = AdamW(params=model.parameters(), lr=lr, weight_decay=lr/10)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=True)
    else:
        #optimizer = AdamW(params=model.parameters(), lr=lr, weight_decay=lr/10)
        optimizer = SGD(params=model.parameters(), lr=lr, weight_decay=lr/10)
        #scheduler = OneCycleScheduler(optimizer, lr_max=lr, n_data_points=len(x_trn), epochs=num_epochs, batch_size=batch_size)
        scheduler = CyclicLR(optimizer, base_lr=eta_min, max_lr=lr, cycle_momentum=True)
    

    best_epoch = -1
    best_lwlrap = 0.
    best_score = np.zeros((num_classes,))
    best_weight = np.zeros((num_classes,))

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            
            x_batch, targets_a, targets_b, lam = mixup_data(x_batch, y_batch, mixup_alpha)
            x_batch, targets_a, targets_b = map(Variable, (x_batch, targets_a, targets_b))

            preds = model(x_batch)
            loss = mixup_criterion(criterion, preds, targets_a, targets_b, lam)
            #loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if not fine_tune:
                scheduler.step()

            avg_loss += loss.item() / len(train_loader)

        model.eval()
        valid_preds = np.zeros((len(x_val), num_classes))
        avg_val_loss = 0.

        for _ in range(0, validation_tta):
            for i, (x_batch, y_batch) in enumerate(valid_loader):
                preds = model(x_batch.cuda()).detach()
                loss = criterion(preds, y_batch.cuda())

                preds = torch.sigmoid(preds)
                valid_preds[i * test_batch_size: (i+1) * test_batch_size] = preds.cpu().numpy()

                avg_val_loss += (loss.item() / len(valid_loader)) / validation_tta
            
        score, weight = calculate_per_class_lwlrap(y_val, valid_preds)
        lwlrap = (score * weight).sum()
        
        log_epoch = epoch % 5 == 0
        if lwlrap > best_lwlrap:
            log_epoch = True
            best_epoch = epoch + 1
            best_lwlrap = lwlrap
            best_score = score
            best_weight = weight
            torch.save(model.state_dict(), weight_file_name)
            
        if log_epoch:
            elapsed = time.time() - start_time
            print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  val_lwlrap: {lwlrap:.6f}  time: {elapsed:.0f}s')
            
        if fine_tune:
            scheduler.step(avg_val_loss)
#         else:
#             scheduler.step()
            
    return {
        'best_epoch': best_epoch,
        'best_lwlrap': best_lwlrap,
        'best_score': best_score,
        'best_weight': best_weight,
        'weight_file_name' : weight_file_name,
    }


# In[ ]:


def pick_best_result(result1, result2):
    if result2['best_lwlrap'] > result1['best_lwlrap']:
        return result2
    else:
        return result1


# In[ ]:


os.listdir('../input/fat2019models')


# In[ ]:


test_loader = DataLoader(FATTestDataset(test_df['fname'], x_test, transforms_dict['test'], crop=True), batch_size=256, shuffle=False)

kf = KFold(n_splits=5, random_state=SEED, shuffle=True)
predictions = []
results = []
for train_index, test_index in kf.split(np.arange(len(train_df))):
    fold = str(len(results) + 1)
    
    x_trn = list(itemgetter(*train_index)(x_train))
    y_trn = np.array(list(itemgetter(*train_index)(y_train)))
    
    x_val = list(itemgetter(*test_index)(x_train))
    y_val = np.array(list(itemgetter(*test_index)(y_train)))
    
    print("Stage 1:")
    model = Classifier(num_classes=num_classes)
    model.load_state_dict(torch.load('../input/fat2019models/' + fold + '_mobilenet_stage4.pt'))
    best_result = train_model(x_trn, x_val, y_trn, y_val, transforms_dict['train'], model.cuda(), num_epochs=100, lr=5e-5, batch_size=100, weight_file_name=fold + '_mobilenet_stage0.pt')

    print("Stage 2:")
    model = Classifier(num_classes=num_classes)
    model.load_state_dict(torch.load(best_result['weight_file_name']))
    best_result = train_model(x_trn, x_val, y_trn, y_val, transforms_dict['train'], model.cuda(), num_epochs=100, lr=1e-5, fine_tune=True, batch_size=100, weight_file_name=fold + '_mobilenet_stage4.pt')

    results.append(best_result)
    
    print("Predicting using:")
    print(best_result)
    model = Classifier(num_classes=num_classes)
    model.load_state_dict(torch.load(best_result['weight_file_name']))
    model.cuda()
    model.eval()
    
    for _ in range(0, 20):
        all_outputs = []
        for images, _ in test_loader:
            preds = torch.sigmoid(model(images.cuda()).detach())
            all_outputs.append(preds.cpu().numpy())
        fold_preds = np.concatenate(all_outputs)
        predictions.append(fold_preds)


# In[ ]:


print("Best results:")
print(results)

for idx, val in enumerate(results):
    df = pd.DataFrame({'labels': labels, 'scores': results[idx]['best_score'], 'weights': results[idx]['best_weight']}, columns=['labels', 'scores', 'weights'])
    df.to_csv(str(idx + 1) + 'scores.csv', encoding='utf-8', index=False)


# In[ ]:


test_df[labels] = np.mean(np.array(predictions), axis=0)
test_df.to_csv('submission.csv', index=False)
test_df.head()

