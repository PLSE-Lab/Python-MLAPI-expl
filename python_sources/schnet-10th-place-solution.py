#!/usr/bin/env python
# coding: utf-8

# This kernel only shows training on 1JHN for the first 20 epochs.
# If you want a description of our model, you can view this discussion:
# https://www.kaggle.com/c/champs-scalar-coupling/discussion/106271#latest-610727

# In[ ]:


import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
from timeit import default_timer as timer


# In[ ]:


get_ipython().system('conda install pytorch torchvision -c pytorch --yes')
get_ipython().system('pip install torch-scatter')
get_ipython().system('pip install torch_geometric')
get_ipython().system('pip install torch_sparse')
get_ipython().system('pip install torch_cluster')


# In[ ]:


get_ipython().system('pip install -U ase==3.17.0')
get_ipython().system('pip install schnetpack')


# In[ ]:


import dataclasses
from multiprocessing import cpu_count
from pprint import pformat
from time import time
from typing import Dict, Callable, List
import schnetpack.atomistic as atm
import schnetpack.nn.blocks
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from schnetpack.data import Structure, AtomsLoader
from schnetpack.datasets import *
from schnetpack.nn import shifted_softplus, Dense
from schnetpack.representation import SchNetInteraction
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch_scatter import scatter_mean
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[ ]:


def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)


    else:
        raise NotImplementedError
        

def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

        
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, fmt):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=fmt)


class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key) -> AverageMeter:
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}
    
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

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
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.8
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the difference of max_momentum
            and some scaling of the amplitude; therefore
            base_momentum may not actually be reached depending on
            scaling function. Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.9
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
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
        return self.gamma ** (x)

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
ARTIFACTS_DIR = 'data/artifacts'
TYPES = ['1JHN', '2JHN', '3JHN', '1JHC', '2JHC', '3JHC', '2JHH', '3JHH']

TYPE_WEIGHT = [
    6.566171,
    107.422157,
    4.083679,
    12.321967,
    39.061047,
    3.084091,
    7.886997,
    27.991149,
]
from abc import ABC, abstractmethod

import numpy as np
import optuna
from optuna import Trial, Study
from optuna.trial import FixedTrial


class Estimator(ABC):
    @abstractmethod
    def predict(self, X: np.ndarray):
        return NotImplemented


class CVOptimizer(ABC):
    def __init__(self, study: Study, is_one_cv=False, prune=False):
        # self.X = X
        # self.y = y
        self.study = study
        self.is_one_cv = is_one_cv
        self.prune = prune

    @abstractmethod
    def fit(self, trial: Trial, X_train, y_train, X_val, y_val, step) -> Estimator:
        return NotImplemented

    @abstractmethod
    def loss_fn(self, y_true, y_pred, trial: Trial):
        return NotImplemented

    @abstractmethod
    def split(self, X, y):
        return NotImplemented

    @abstractmethod
    def make_xy(self, trial: Trial):
        return NotImplemented

    def on_after_trial(self, trial: Trial, cv_models, cv_preds, loss_val):
        pass

    def best_models(self, return_preds=False):
        fixed_trial = FixedTrial(self.study.best_params)
        loss, models, cv_preds = self.objective(fixed_trial, return_model=True)

        if return_preds:
            return models, cv_preds
        return models

    def objective(self, trial: Trial, return_model=False):
        X, y = self.make_xy(trial)
        # X, y = self.X, self.y

        loss_cv_train = []

        cv_preds = np.zeros_like(y)
        cv_preds.fill(np.nan)

        models = []

        for step, (train_idx, val_idx) in enumerate(self.split(X, y)):
            X_train = X[train_idx]
            X_val = X[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            model = self.fit(trial, X_train, y_train, X_val, y_val, step)

            y_train_preds = model.predict(X_train)
            y_val_preds = model.predict(X_val)
            # from IPython.core.debugger import Pdb;
            # Pdb().set_trace()
            cv_preds[val_idx] = y_val_preds

            mask_done = ~np.isnan(cv_preds)
            intermediate_loss_train = self.loss_fn(y_train, y_train_preds, trial)
            intermediate_loss_val = self.loss_fn(y[mask_done], cv_preds[mask_done], trial)

            loss_cv_train.append(intermediate_loss_train)

            trial.report(intermediate_loss_val, step)
            if self.prune and trial.should_prune(step):
                raise optuna.structs.TrialPruned()

            models.append(model)

            if self.is_one_cv:
                break

        mask_done = ~np.isnan(cv_preds)
        loss_train = float(np.mean(loss_cv_train))
        loss_val = float(self.loss_fn(y[mask_done], cv_preds[mask_done], trial))

        trial.set_user_attr('train_loss', loss_train)
        trial.set_user_attr('val_loss', loss_val)
        trial.set_user_attr('is_one_cv', int(self.is_one_cv))

        self.on_after_trial(trial, models, cv_preds, loss_val)

        if return_model:
            return loss_val, models, cv_preds

        return loss_val

    def optimize(self, n_trials=100, n_jobs=1):
        self.study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs)

import os

import feather
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def get_main_train_df(coupling_type=None):
    df = pd.read_csv('data/input/train.csv')
    if coupling_type is None:
        return df
    if isinstance(coupling_type, str):
        coupling_type = [coupling_type]

    return df[df.type.isin(coupling_type)].reset_index(drop=True)


def get_main_test_df(coupling_type=None):
    df = pd.read_csv('data/input/test.csv')
    if coupling_type is None:
        return df
    if isinstance(coupling_type, str):
        coupling_type = [coupling_type]

    return df[df.type.isin(coupling_type)].reset_index(drop=True)


class CouplingProvider:
    DF_CACHE_PATH = 'tmp/df_CouplingProvider.fth'

    def __init__(self, scaler='std'):
        self.cols = [
            'scalar_coupling_constant',
            'fc',
            'sd',
            'pso',
            'dso',
        ]
        self.scaler = scaler
        df = self.init_df()
        self.grouped = df.groupby('molecule_name')
        self.type_encoder = get_type_onehot_encoder()

    def get_coupling_values(self, mol_name, types):
        g = self.grouped.get_group(mol_name)
        g = g[g.type.isin(types)]

        # Case for H2O
        if len(g.type.values) == 0:
            return np.array([]).reshape(-1, 2), np.array([]).reshape(-1, 13)

        encoded_types = self.type_encoder.transform(g.type.values.reshape(-1, 1))
        feats = np.concatenate((
            g[self.cols].values,
            encoded_types,
        ), axis=1)
        edge_index = g[['atom_index_0', 'atom_index_1']].values

        return edge_index, feats

    def init_df(self):
        if os.path.exists(self.DF_CACHE_PATH):
            print('load df from cache')
            return feather.read_dataframe(self.DF_CACHE_PATH)

        df = get_main_train_df()
        df = df.merge(pd.read_csv('data/input/scalar_coupling_contributions.csv').drop(columns=['type']),
                      on=['molecule_name', 'atom_index_0', 'atom_index_1'])
        types = sorted(df.type.unique())

        # Scale in for each types
        for t in types:
            df_tmp = df[df.type == t]
            if self.scaler == 'std':
                scaled = StandardScaler().fit_transform(df_tmp[self.cols])
            elif self.scaler == 'minmax':
                scaled = MinMaxScaler().fit_transform(df_tmp[self.cols])
            else:
                raise Exception('not supported scaler')
            df.loc[df.type == t, self.cols] = scaled

        df[self.cols] = df[self.cols].astype(np.float32)
        df.to_feather(self.DF_CACHE_PATH)
        return df
import torch
import torch.nn as nn
from schnetpack.datasets import *
from schnetpack.nn.activations import shifted_softplus


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*repeat_idx)
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


class ShiftedSoftplus(nn.Module):
    def __init__(self, inplace=False):
        super(ShiftedSoftplus, self).__init__()

    def forward(self, input):
        return shifted_softplus(input)


def l1_loss(y_pred, y_true):
    # TODO should use weight for each types?
    loss = torch.abs(y_true - y_pred)
    loss = loss.mean(dim=0)
    loss = torch.log(loss)
    loss = torch.sum(loss)

    return loss


def smooth_l1_loss(delta=1.):
    def loss_fn(y_pred, y_true):
        loss = torch.abs(y_true - y_pred)
        loss = torch.where(loss < delta, 0.5 * (loss ** 2), delta * loss - 0.5 * (delta ** 2))
        loss = loss.mean(dim=0)
        loss = torch.log(loss)
        loss = torch.sum(loss)
        return loss

    return loss_fn


def log_cosh_loss(y_pred, y_true):
    loss = torch.cosh(y_pred - y_true)
    loss = torch.log(loss)
    loss = loss.mean(dim=0)
    loss = torch.log(loss)
    loss = torch.sum(loss)
    return loss
import numpy as np


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta


# In[ ]:


PROPERTIES = [
    'scalar_coupling_constants',
    'scalar_coupling_types',
    'mulliken_charges',
    'gasteiger_charges',
    'dihedral_edge_index',
    'dihedral_edge_attr',
    'bond_edge_index',
    'bond_edge_attr',
    'sigma_iso',
    'log_omega',
    'kappa',
    # 'potential_energy',
    # 'dipole_moments',
    # 'homo',
    # 'lumo',
    # 'U0',
    # 'paths',
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(0)

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.options.display.width = 999


# In[ ]:


def l1_loss(y_pred, y_true):
    # TODO should use weight for each types?
    loss = torch.abs(y_true - y_pred)
    loss = loss.mean(dim=0)
    loss = torch.log(loss)
    loss = torch.sum(loss)

    return loss


def smooth_l1_loss(delta=1.):
    def loss_fn(y_pred, y_true):
        loss = torch.abs(y_true - y_pred)
        loss = torch.where(loss < delta, 0.5 * (loss ** 2), delta * loss - 0.5 * (delta ** 2))
        loss = loss.mean(dim=0)
        loss = torch.log(loss)
        loss = torch.sum(loss)
        return loss

    return loss_fn


def _logcosh(x):
    return x + F.softplus(-2. * x) - np.log(2.)


def log_cosh_loss(y_pred, y_true):
    loss = _logcosh(y_pred - y_true)
    loss = loss.mean(dim=0)
    loss = torch.log(loss)
    loss = torch.sum(loss)
    # loss = loss.mean()
    return loss


# In[ ]:


class SchNet(nn.Module):
    def __init__(self, n_atom_basis=128, n_filters=128, n_interactions=1, cutoff=5.0, n_gaussians=25,
                 normalize_filter=False, coupled_interactions=False,
                 return_intermediate=False, max_z=100, interaction_block=SchNetInteraction, trainable_gaussians=False,
                 distance_expansion=None):
        super(SchNet, self).__init__()

        n_extra_embeddings = 1
        # n_extra_embeddings = 0

        # atom type embeddings
        self.embedding = nn.Embedding(max_z, n_atom_basis - n_extra_embeddings, padding_idx=0)

        # spatial features
        self.distances = schnetpack.nn.neighbors.AtomDistances()
        if distance_expansion is None:
            self.distance_expansion = schnetpack.nn.acsf.GaussianSmearing(0.0, cutoff, n_gaussians,
                                                                          trainable=trainable_gaussians)
        else:
            self.distance_expansion = distance_expansion

        self.return_intermediate = return_intermediate

        # interaction network
        if coupled_interactions:
            self.interactions = nn.ModuleList([
                                                  SchNetInteraction(n_atom_basis=n_atom_basis,
                                                                    n_spatial_basis=n_gaussians,
                                                                    n_filters=n_filters,
                                                                    normalize_filter=normalize_filter)
                                              ] * n_interactions)
        else:
            self.interactions = nn.ModuleList([
                SchNetInteraction(n_atom_basis=n_atom_basis, n_spatial_basis=n_gaussians,
                                  n_filters=n_filters, normalize_filter=normalize_filter)
                for _ in range(n_interactions)
            ])

        n_dihedral_attrs = 2
        n_dihedral_feats = 16
        self.dihedral_conv = gnn.NNConv(n_atom_basis, n_dihedral_feats, nn.Sequential(
            Dense(n_dihedral_attrs, n_atom_basis, activation=F.relu),
            Dense(n_atom_basis, n_atom_basis * n_dihedral_feats),
        ))

        n_bond_attrs = 5
        n_bond_feats = 32
        self.bond_conv = gnn.NNConv(n_atom_basis, n_bond_feats, nn.Sequential(
            Dense(n_bond_attrs, n_atom_basis, activation=F.relu),
            Dense(n_atom_basis, n_atom_basis * n_bond_feats),
        ))

        self.init_fc = Dense(n_atom_basis + n_dihedral_feats + n_bond_feats, n_atom_basis, activation=shifted_softplus)

    def forward(self, inputs):
        atomic_numbers = inputs[Structure.Z]
        positions = inputs[Structure.R]
        cell = inputs[Structure.cell]
        cell_offset = inputs[Structure.cell_offset]
        neighbors = inputs[Structure.neighbors]
        neighbor_mask = inputs[Structure.neighbor_mask]
        
        # atom embedding
        x = self.embedding(atomic_numbers)
        x = torch.cat((
            x,
            inputs['mulliken_charges'],
            # inputs['gasteiger_charges'],
        ), dim=2)

        n_batch, n_atoms, n_embeddings = x.shape
        # print(n_batch, n_atoms, n_embeddings)

        # -------
        batch_range = torch.arange(0, n_batch).to(device)

        d_edge_index = inputs['dihedral_edge_index'].long()
        d_edge_index = d_edge_index + batch_range.view(-1, 1, 1) * n_atoms
        d_edge_index = d_edge_index.reshape(-1, 2)
        d_edge_mask = d_edge_index[:, 0] != d_edge_index[:, 1]
        d_edge_index = d_edge_index[d_edge_mask]

        assert inputs['dihedral_edge_attr'].shape[2] == 3, 'n dihedral feats is not 3.'
        d_edge_attr = inputs['dihedral_edge_attr'][:, :, :2]
        d_edge_attr = d_edge_attr.reshape(-1, 2)[d_edge_mask]

        # noinspection PyCallingNonCallable
        dihedral_feat = self.dihedral_conv(x.reshape(-1, n_embeddings), torch.t(d_edge_index), d_edge_attr)
        dihedral_feat = F.relu(dihedral_feat.view(n_batch, n_atoms, -1))
        # print(dihedral_feat.shape)
        # -------

        # -------
        a_edge_index = inputs['bond_edge_index'].long()
        a_edge_index = a_edge_index + batch_range.view(-1, 1, 1) * n_atoms
        a_edge_index = a_edge_index.reshape(-1, 2)
        a_edge_mask = a_edge_index[:, 0] != a_edge_index[:, 1]
        a_edge_index = a_edge_index[a_edge_mask]
        
        assert inputs['bond_edge_attr'].shape[2] == 5 # , 'n angle feats is not 2.'
        a_edge_attr = inputs['bond_edge_attr'][:, :, :5]
        a_edge_attr = a_edge_attr.reshape(-1, 5)[a_edge_mask]
        
        # noinspection PyCallingNonCallable
        bond_feat = self.bond_conv(x.reshape(-1, n_embeddings), torch.t(a_edge_index), a_edge_attr)
        bond_feat = F.relu(bond_feat.view(n_batch, n_atoms, -1))
        # print(angle_feat.shape)
        # -------

        # spatial features
        r_ij = self.distances(positions, neighbors, cell, cell_offset)
        f_ij = self.distance_expansion(r_ij)

        x = self.init_fc(torch.cat((x, dihedral_feat, bond_feat), dim=2))

        for interaction in self.interactions:
            v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
            x = x + v



        return x


# In[ ]:


class AtomPairwise(atm.OutputModule):
    def __init__(self, n_in=128, n_out=1, n_layers=2, n_neurons=None,
                 activation=schnetpack.nn.activations.shifted_softplus, return_contributions=False,
                 requires_dr=False, create_graph=False, mean=None, stddev=None, atomref=None, max_z=100,
                 train_embeddings=False):
        super(AtomPairwise, self).__init__(n_in, n_out, requires_dr)

        self.n_layers = n_layers

        self.out_net = nn.Sequential(
            schnetpack.nn.base.GetItem('atom_pair_rep'),
            schnetpack.nn.blocks.MLP(n_in * 2, n_out, n_neurons, n_layers, activation),
            # schnetpack.nn.blocks.MLP(n_in, n_out, n_neurons, n_layers, activation),
        )

        self.contributions_to_coupling_constant = Dense(4, 1)

    def forward(self, inputs):
        # atomic_numbers = inputs[Structure.Z]
        # atom_mask = inputs[Structure.atom_mask]
        atom_rep = inputs['representation']

        n_batch = atom_rep.shape[0]

        atom_index_0 = inputs['scalar_coupling_constants'][:, :, 0].long()
        atom_index_1 = inputs['scalar_coupling_constants'][:, :, 1].long()
        # atom_2 = inputs['paths'][:, :, 1].long()

        coupling_batch_idx = tile(torch.arange(0, n_batch), 0, atom_index_0.shape[1])

        atom_rep_0 = atom_rep[coupling_batch_idx, atom_index_0.view(-1)]
        atom_rep_1 = atom_rep[coupling_batch_idx, atom_index_1.view(-1)]
        #add pooling layer like set2set
        
        # atom_rep_2 = atom_rep[coupling_batch_idx, atom_2.view(-1)]

        atom_pair_rep = torch.cat((atom_rep_0, atom_rep_1), dim=1)
        # atom_pair_rep = torch.cat((atom_rep_0, atom_rep_1, atom_rep_2), dim=1)
        # atom_pair_rep = atom_pair_rep.view(n_batch, -1, atom_pair_rep.shape[-1])
        # print(atom_pair_rep.shape)

        scc = inputs['scalar_coupling_constants'].view(-1, 7)  # idx0, idx1, cc, fc, sd, pso, dso
        types = inputs['scalar_coupling_types'].long().view(-1)
        mask = scc[:, 0] != scc[:, 1]

        atom_pair_rep = atom_pair_rep[mask]

        inputs['atom_pair_rep'] = atom_pair_rep

        #predict just scc
        contributions = self.out_net(inputs)

        y = self.contributions_to_coupling_constant(contributions)
        y = torch.cat((y, contributions), dim=1)

        result = {
            'y_pred': y,
            'y_true': scc[mask, 2:],
            'types': types[mask]
        }

        return result


# In[ ]:


def make_atomwise_result(batch, result):
    atom_mask = batch[Structure.atom_mask]
    mask = atom_mask.byte()

    y_pred = result['yi'][mask]  # sigma_iso, log_omega, kappa
    y_true = torch.cat((
        batch['sigma_iso'][mask],
        batch['log_omega'][mask],
        batch['kappa'][mask],
    ), dim=1)

    return y_pred, y_true

def get_type_encoder():
    enc = LabelEncoder()
    enc.fit(TYPES)


def calc_loss_atomwise_detail(y_pred, y_true):
    with torch.no_grad():
        loss = torch.abs(y_pred - y_true)
        loss = loss.mean(dim=0)
        # loss = torch.log(loss)
        loss = loss.detach().cpu().numpy()
    detail = {
        'sigma_iso': loss[0],
        'log_omega': loss[1],
        'kappa': loss[2],
    }
    return detail


def calc_loss_contribution_detail(y_pred, y_true):
    with torch.no_grad():
        loss = torch.abs(y_pred - y_true)
        loss = loss.mean(dim=0)
        # loss = torch.log(loss)
        loss = loss.detach().cpu().numpy()
    detail = {
        'cc': loss[0],
        'fc': loss[1],
        'sd': loss[2],
        'pso': loss[3],
        'dso': loss[4],
    }
    return detail


def calc_loss_energy_detail(y_pred, y_true):
    with torch.no_grad():
        loss = torch.abs(y_pred - y_true)
        loss = loss.mean(dim=0)
        loss = torch.log(loss)
        loss = loss.detach().cpu().numpy()
    detail = {
        'alpha': loss[0],
        'r2': loss[1]
    }
    return detail


def calc_loss_type_detail(y_pred, y_true, types):
    abs_errs = torch.abs(y_pred - y_true)
    abs_errs_cc = abs_errs.detach().cpu().numpy()[:, 0]
    types = types.cpu().numpy().astype(int)

    maes = pd.DataFrame({
        'errs': abs_errs_cc,
        'types': types,
    }).groupby('types').agg({
        'errs': [np.mean, 'size']
    })
    maes = maes.reset_index()
    maes.columns = ['type', 'mae', 'n_data']
    # maes['log_mae'] = np.log(maes['log_mae'])

    return maes


# In[ ]:


ARTIFACTS_DIR = 'data/artifacts'
TYPES = ['1JHN', '2JHN', '3JHN', '1JHC', '2JHC', '3JHC', '2JHH', '3JHH']

TYPE_WEIGHT = [
    6.566171,
    107.422157,
    4.083679,
    12.321967,
    39.061047,
    3.084091,
    7.886997,
    27.991149,
]


# In[ ]:


def make_energy_result(batch, result):
    y_pred = result['y'].to(device)  # homo, lumo, U0
    
    features=[]
    if(VALIDATION):
        molNam=molNameVal
    else:
        molNam=molNameTrain
    for i in batch["_idx"]:
        features.append(qm9.get_group(molNam[i])[cols])
    features=np.concatenate(features,axis=0)
    features=torch.from_numpy(features).to(device)
    features=features.type(torch.FloatTensor)

    y_true = torch.cat((
        features[:,0].reshape(-1,1),
        features[:,1].reshape(-1,1)
    ), dim=1)
    y_true=y_true.to(device)
    y_pred=y_pred.to(device)
    return y_pred, y_true

def l1_loss(y_pred, y_true):
    # TODO should use weight for each types?
    loss = torch.abs(y_true - y_pred)
    loss = loss.mean(dim=0)
    loss = torch.log(loss)
    loss = torch.sum(loss)

    return loss


# In[ ]:


def get_type_encoder():
        enc = LabelEncoder()
        enc.fit(TYPES)


# In[ ]:


@dataclasses.dataclass()
class Conf:
        
    lr: float = 1e-4
    weight_decay: float = 1e-4

    clr_max_lr: float = 3e-3
    clr_base_lr: float = 3e-6
    clr_gamma: float = 0.99994

    train_batch: int = 32
    val_batch: int = 256

    n_atom_basis: int = 128
    n_interactions: int = 1
    coupled_interactions: bool = True
    n_filters: int = 128
    cutoff: float = 5.0
    n_gaussians: int = 25

    pairwise_layers: int = 2
    atomwise_layers: int = 2
    energy_layers: int = 2

    atomwise_weight: float = 1.

    pre_trained_path: str = None

    optim: str = 'adam'
    loss_fn: Callable = l1_loss

    epochs: int = 25
    is_save_epoch_fn: Callable = None
    resume_from: Dict[str, int] = None

    types: List[str] = None
    db_path: str = None

    seed: int = 0

    is_one_cv: bool = True

    device: str = device

    exp_name: str = 'schnet'
    exp_time: float = time()

    logger_step = None
    logger_epoch = None
    
    @staticmethod
    def create_logger(name, filename):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if not logger.hasHandlers():
            logger.addHandler(logging.FileHandler(filename))
        return logger
    
    def get_type_encoder():
        enc = LabelEncoder()
        enc.fit(TYPES)

    def __post_init__(self):
        self.out_dir=os.getcwd()
        if self.resume_from is not None:
            assert os.path.exists(self.out_dir), '{} does not exist.'.format(self.out_dir)

        ensure_dir(self.out_dir)

        self.logger_step = self.create_logger('step_logger_{}'.format(self.exp_time),
                                              '{}/step.log'.format(self.out_dir))
        self.logger_epoch = self.create_logger('epoch_logger_{}'.format(self.exp_time),
                                               '{}/epoch.log'.format(self.out_dir))
        self.type_encoder = get_type_encoder()

        with open('{}/conf.txt'.format(self.out_dir), 'w') as f:
            f.write(str(self))

        global device
        device = self.device

    #def out_dir(self):
    #    return 'data/experiments/{}/{}'.format(self.exp_name, self.exp_time)

    def __str__(self):
        return pformat(dataclasses.asdict(self))


# In[ ]:


conf=Conf(
    is_one_cv=True,

    device='cuda',

    # train_batch=32,
    train_batch=64,
    # train_batch=128,
    val_batch=256,

    lr=.001,
    clr_max_lr=3e-3,
    clr_base_lr=3e-6,
    # lr=3e-5,
    # clr_max_lr=1e-3,
    # clr_base_lr=1e-6,
    clr_gamma=0.999991,
    weight_decay=1e-4,
    # weight_decay=3e-3,

    n_atom_basis=256 + 64,
    n_interactions=6,
    coupled_interactions=False,
    n_filters=256,#256*2,
    pairwise_layers=3,#4,
    atomwise_layers=3,#4,
    n_gaussians=25,#40,

    # atomwise_weight=0.3,

    # loss_fn=log_cosh_loss,
    # loss_fn=smooth_l1_loss(),
    loss_fn=l1_loss,
    optim='adam',

    epochs= 20,    #CHANGE THIS TO 400-500
    # is_save_epoch_fn=lambda x: x % 20 == 19,

    # pre_trained_path='data/experiments/schnet-qm9/1563077515.9908571/0-380.ckpt',  # n_filter=128
    # pre_trained_path='data/experiments/schnet-qm9/1563103994.1368518/0-420.ckpt',  # n_filter=256
    # pre_trained_path='data/experiments/schnet-qm9/1563622942.702352/0-300.ckpt',  # n_filter=256
    # pre_trained_path='data/experiments/schnet-qm9/1563624328.5389972/0-300.ckpt',  # n_filter=256 basis=256
    types=['1JHN'],
    db_path="../input/schnet-data/CHAMPS_with_bond_train_1JHN.db",
    exp_time=time()
)


# In[ ]:


def log_hist(df_hist, logger: logging.Logger, types):
    df_hist["time"]=time_to_str((timer() - start),'min')
    last = df_hist.tail(1)
    best = df_hist.sort_values('val_loss_cc', ascending=True).head(1)
    summary = pd.concat((last, best)).reset_index(drop=True)
    summary['name'] = ['Last', 'Best']
    logger.debug(summary[[
                             'name',
                             'epoch',
                             'train_loss_coupling',
                             'train_loss_atomwise',
                             # 'train_loss_energy',
                             'val_loss_coupling',
                             'val_loss_atomwise',
                             # 'train_loss_total',
                             # 'val_loss_total',
                         ] + [
                             'train_loss_{}'.format(t) for t in types
                         ] + [
                             'val_loss_{}'.format(t) for t in types
                         ]])
    
    print(summary[[          
                             'name',
                             'epoch',
                             'train_loss_coupling',
                             # 'train_loss_energy',
                             'val_loss_coupling',
                             'val_loss_atomwise',
                             "time",
                             # 'train_loss_total',
                             # 'val_loss_total',
                         ] + [
                             'train_loss_{}'.format(t) for t in types
                         ] + [
                             'val_loss_{}'.format(t) for t in types
                         ]])


def write_on_board(df_hist, writer, conf: Conf):
    row = df_hist.tail(1).iloc[0]

    writer.add_scalars('{}/lr'.format(conf.exp_name), {
        '{}'.format(conf.exp_time): row.lr,
    }, row.epoch)

    for tag in ['cc', 'fc', 'sd', 'pso', 'dso']:
        writer.add_scalars('{}/loss/coupling/{}'.format(conf.exp_name, tag), {
            '{}_train'.format(conf.exp_time): row['train_loss_{}'.format(tag)],
            '{}_val'.format(conf.exp_time): row['val_loss_{}'.format(tag)],
        }, row.epoch)
    writer.add_scalars('{}/loss/coupling/total'.format(conf.exp_name), {
        '{}_train'.format(conf.exp_time): row.train_loss_coupling,
        '{}_val'.format(conf.exp_time): row.val_loss_coupling,
    }, row.epoch)

    for tag in ['sigma_iso', 'log_omega', 'kappa']:
        writer.add_scalars('{}/loss/atomwise/{}'.format(conf.exp_name, tag), {
            '{}_train'.format(conf.exp_time): row['train_loss_{}'.format(tag)],
            '{}_val'.format(conf.exp_time): row['val_loss_{}'.format(tag)],
        }, row.epoch)
    writer.add_scalars('{}/loss/atomwise/total'.format(conf.exp_name), {
        '{}_train'.format(conf.exp_time): row.train_loss_atomwise,
        '{}_val'.format(conf.exp_time): row.val_loss_atomwise,
    }, row.epoch)

    # for tag in ['homo', 'lumo', 'U0']:
    #     writer.add_scalars('{}/loss/energy/{}'.format(conf.exp_name, tag), {
    #         '{}_train'.format(conf.exp_time): row['train_loss_{}'.format(tag)],
    #         '{}_val'.format(conf.exp_time): row['val_loss_{}'.format(tag)],
    #     }, row.epoch)
    # writer.add_scalars('{}/loss/energy/total'.format(conf.exp_name), {
    #     '{}_train'.format(conf.exp_time): row.train_loss_energy,
    #     '{}_val'.format(conf.exp_time): row.val_loss_energy,
    # }, row.epoch)

    # writer.add_scalars('{}/loss_dipole'.format(exp_name), {
    #     '{}_train'.format(exp_id): row.train_loss_dipole,
    #     '{}_val'.format(exp_id): row.val_loss_dipole,
    # }, row.epoch)

    for tag in conf.types:
        writer.add_scalars('{}/loss/type/{}'.format(conf.exp_name, tag), {
            '{}_train'.format(conf.exp_time): row['train_loss_{}'.format(tag)],
            '{}_val'.format(conf.exp_time): row['val_loss_{}'.format(tag)],
        }, row.epoch)
    writer.add_scalars('{}/loss/type/total'.format(conf.exp_name), {
        '{}_train'.format(conf.exp_time): row['train_loss_total'],
        '{}_val'.format(conf.exp_time): row['val_loss_total'],
    }, row.epoch)


def load_pre_trained(conf: Conf):
    ckpt = torch.load(conf.pre_trained_path, map_location=device)

    reps = SchNet(
        n_interactions=conf.n_interactions,
        coupled_interactions=False,
        n_filters=conf.n_filters,
        n_atom_basis=conf.n_atom_basis
    )
    model = atm.AtomisticModel(reps, [
        atm.Atomwise(
            return_contributions=True,
            n_in=conf.n_atom_basis,
            n_out=1,
            # n_layers=3,
        ),
        atm.Atomwise(
            return_contributions=False,
            n_in=conf.n_atom_basis,
            n_out=3,
            # n_layers=3,
        ),
    ])
    model.load_state_dict(ckpt['model'], strict=False)

    return model

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
def get_type_encoder() -> LabelEncoder:
    enc = LabelEncoder()
    enc.fit(TYPES)
    return enc


# In[ ]:


VALIDATION=False
def train(loader, model: atm.AtomisticModel, optimizer, scheduler, conf: Conf):
    global VALIDATION
    VALIDATION=False
    meters = AverageMeterSet()
    model.train()

    for i, batch in enumerate(loader):
        scheduler.step()
        meters.update('lr', optimizer.param_groups[0]['lr'])

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }
        result = model(batch)

        loss_fn = conf.loss_fn
        
        # ----- Coupling Loss -----
        # y_pred_coupling, y_true_coupling, types = make_coupling_result(batch, result[model.output_modules[0]])
        coupling_res = result[model.output_modules[0]]
        n_pairs = len(coupling_res['types'])
        coupling_loss = loss_fn(coupling_res['y_pred'], coupling_res['y_true'])
        meters.update('loss_coupling', coupling_loss.item(), n_pairs)
 
        
        # ----- Atomwise Loss -----
        y_pred_atomwise, y_true_atomwise = make_atomwise_result(batch, result[model.output_modules[1]])
        n_atoms = len(y_pred_atomwise)
        atomwise_loss = loss_fn(y_pred_atomwise, y_true_atomwise)
        meters.update('loss_atomwise', atomwise_loss.item(), n_atoms)

    

        # ----- Dipole Moment Loss -----
        # dipole_loss = loss_fn(torch.abs(batch['dipole_moments'] - result[model.output_modules[2]]['y']))
        # n_mols = len(batch['dipole_moments'])

        # ----- Metric of coupling contributions -----
        contribution_detail = calc_loss_contribution_detail(coupling_res['y_pred'], coupling_res['y_true'])
        meters.update('loss_cc', contribution_detail['cc'], n_pairs)
        meters.update('loss_fc', contribution_detail['fc'], n_pairs)
        meters.update('loss_sd', contribution_detail['sd'], n_pairs)
        meters.update('loss_pso', contribution_detail['pso'], n_pairs)
        meters.update('loss_dso', contribution_detail['dso'], n_pairs)

        # ----- Metric of atomwise -----
        atomwise_detail = calc_loss_atomwise_detail(y_pred_atomwise, y_true_atomwise)
        meters.update('loss_sigma_iso', atomwise_detail['sigma_iso'], n_atoms)
        meters.update('loss_log_omega', atomwise_detail['log_omega'], n_atoms)
        meters.update('loss_kappa', atomwise_detail['kappa'], n_atoms)

        # ----- Metric of energy -----
        # meters.update('loss_homo', energy_detail['homo'])
        # meters.update('loss_lumo', energy_detail['lumo'])
        # meters.update('loss_U0', energy_detail['U0'])

        # ----- Metric of dipole -----
        # meters.update('loss_dipole', dipole_loss.item(), n_mols)

        # ----- Metric for each types -----
        type_detail = calc_loss_type_detail(coupling_res['y_pred'], coupling_res['y_true'], coupling_res['types'])
        conf.type_encoder = get_type_encoder()
        type_detail['type_name'] = conf.type_encoder.inverse_transform(type_detail.type.values)
        for _, row in type_detail.iterrows():
            meters.update('loss_{}'.format(row.type_name), row.mae, row.n_data)

        # ----- Total Loss -----
        # loss = coupling_loss
        loss = coupling_loss + atomwise_loss * conf.atomwise_weight
        # loss = coupling_loss + atomwise_loss + energy_loss
        # loss = coupling_loss + atomwise_loss + dipole_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {
        'lr': meters['lr'].avg,
        'train_loss_cc': meters['loss_cc'].avg,
        'train_loss_fc': meters['loss_fc'].avg,
        'train_loss_sd': meters['loss_sd'].avg,
        'train_loss_pso': meters['loss_pso'].avg,
        'train_loss_dso': meters['loss_dso'].avg,
        'train_loss_coupling': meters['loss_coupling'].avg,

        'train_loss_sigma_iso': meters['loss_sigma_iso'].avg,
        'train_loss_log_omega': meters['loss_log_omega'].avg,
        'train_loss_kappa': meters['loss_kappa'].avg,
        'train_loss_atomwise': meters['loss_atomwise'].avg,

        # 'train_loss_homo': meters['loss_homo'].avg,
        # 'train_loss_lumo': meters['loss_lumo'].avg,
        # 'train_loss_U0': meters['loss_U0'].avg,

        **{
            'train_loss_{}'.format(t): np.log(meters['loss_{}'.format(t)].avg)
            for t in conf.types
        },
        'train_loss_total': np.mean([
            np.log(meters['loss_{}'.format(t)].avg)
            for t in conf.types
        ]),

        # 'train_loss_dipole': meters['loss_dipole'].avg,
    }

def validate(loader, model, conf: Conf):
    global VALIDATION
    validation=True
    
    meters = AverageMeterSet()
    model.eval()

    for i, batch in enumerate(loader):
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        loss_fn = conf.loss_fn

        with torch.no_grad():
            result = model(batch)
            # ----- Coupling Loss -----
            # y_pred_coupling, y_true_coupling, types = make_coupling_result(batch, result[model.output_modules[0]])
            coupling_res = result[model.output_modules[0]]
            n_pairs = len(coupling_res['types'])

            coupling_loss = loss_fn(coupling_res['y_pred'], coupling_res['y_true'])
            meters.update('loss_coupling', coupling_loss.item(), n_pairs)
            
            
            # ----- Atomwise Loss -----
            y_pred_atomwise, y_true_atomwise = make_atomwise_result(batch, result[model.output_modules[1]])
            n_atoms = len(y_pred_atomwise)
            atomwise_loss = loss_fn(y_pred_atomwise, y_true_atomwise)
            meters.update('loss_atomwise', atomwise_loss.item(), n_atoms)


            # ----- Dipole Moment Loss -----
            # dipole_loss = loss_fn(torch.abs(batch['dipole_moments'] - result[model.output_modules[2]]['y']))
            # n_mols = len(batch['dipole_moments'])

            # ----- Metric of coupling contributions -----
            contribution_detail = calc_loss_contribution_detail(coupling_res['y_pred'], coupling_res['y_true'])
            meters.update('loss_cc', contribution_detail['cc'], n_pairs)
            meters.update('loss_fc', contribution_detail['fc'], n_pairs)
            meters.update('loss_sd', contribution_detail['sd'], n_pairs)
            meters.update('loss_pso', contribution_detail['pso'], n_pairs)
            meters.update('loss_dso', contribution_detail['dso'], n_pairs)

            # ----- Metric of atomwise -----
            atomwise_detail = calc_loss_atomwise_detail(y_pred_atomwise, y_true_atomwise)
            meters.update('loss_sigma_iso', atomwise_detail['sigma_iso'], n_atoms)
            meters.update('loss_log_omega', atomwise_detail['log_omega'], n_atoms)
            meters.update('loss_kappa', atomwise_detail['kappa'], n_atoms)

            # ----- Metric of energy -----
            # energy_detail = calc_loss_energy_detail(y_pred_energy, y_true_energy)
            # meters.update('loss_homo', energy_detail['homo'])
            # meters.update('loss_lumo', energy_detail['lumo'])
            # meters.update('loss_U0', energy_detail['U0'])

            # ----- Metric of dipole -----
            # meters.update('loss_dipole', dipole_loss.item(), n_mols)

            # ----- Metric for each types -----
            type_detail = calc_loss_type_detail(coupling_res['y_pred'], coupling_res['y_true'], coupling_res['types'])
            type_detail['type_name'] = conf.type_encoder.inverse_transform(type_detail.type.values)
            for _, row in type_detail.iterrows():
                meters.update('loss_{}'.format(row.type_name), row.mae, row.n_data)

    return {
        'val_loss_cc': meters['loss_cc'].avg,
        'val_loss_fc': meters['loss_fc'].avg,
        'val_loss_sd': meters['loss_sd'].avg,
        'val_loss_pso': meters['loss_pso'].avg,
        'val_loss_dso': meters['loss_dso'].avg,
        'val_loss_coupling': meters['loss_coupling'].avg,

        'val_loss_sigma_iso': meters['loss_sigma_iso'].avg,
        'val_loss_log_omega': meters['loss_log_omega'].avg,
        'val_loss_kappa': meters['loss_kappa'].avg,
        'val_loss_atomwise': meters['loss_atomwise'].avg,

        #'val_loss_homo': meters['loss_homo'].avg,
        # 'val_loss_lumo': meters['loss_lumo'].avg,
        # 'val_loss_U0': meters['loss_U0'].avg,

        **{
            'val_loss_{}'.format(t): np.log(meters['loss_{}'.format(t)].avg)
            for t in conf.types
        },
        'val_loss_total': np.mean([
            np.log(meters['loss_{}'.format(t)].avg)
            for t in conf.types
        ]),

        # 'val_loss_dipole': meters['loss_dipole'].avg,
    }


# In[ ]:


start = timer()
conf.db_path = '../input/schnet-data/CHAMPS_with_bond_train_1JHN.db'
db = schnetpack.data.AtomsData(conf.db_path, properties=PROPERTIES)
folds = KFold(n_splits=4, random_state=1, shuffle=True)
for cv, (train_idx, val_idx) in enumerate(folds.split(range(len(db)))):
    # Use partial data for small experiments
    # train_idx = np.random.RandomState(conf.seed).permutation(train_idx)[:21420]
    # val_idx = np.random.RandomState(conf.seed).permutation(val_idx)[:7140]

    # if conf.types == ['1JHC']:
    #     train_idx = np.setdiff1d(train_idx, [14])
    #     val_idx = np.setdiff1d(val_idx, [14])
    # if conf.types == ['2JHC']:
    #     train_idx = np.setdiff1d(train_idx, [12])
    #     val_idx = np.setdiff1d(val_idx, [12])
    # if conf.types == ['3JHC']:
    #     train_idx = np.setdiff1d(train_idx, [6])
    #     val_idx = np.setdiff1d(val_idx, [6])

    train_data = db.create_subset(train_idx)
    val_data = db.create_subset(val_idx)
    print(cv, len(train_data), len(val_data))
    train_loader = AtomsLoader(train_data, batch_size=conf.train_batch, shuffle=True, num_workers=4)
    val_loader = AtomsLoader(val_data, batch_size=conf.val_batch, num_workers=4)

    reps = SchNet(
        n_atom_basis=conf.n_atom_basis,
        n_interactions=conf.n_interactions,
        coupled_interactions=conf.coupled_interactions,
        n_filters=conf.n_filters,
        cutoff=conf.cutoff,
        n_gaussians=conf.n_gaussians,
    )
    if conf.pre_trained_path is not None:
        pre_trained = load_pre_trained(conf)
        reps.load_state_dict(pre_trained.representation.state_dict(), strict=False)
        # n_in = conf.n_atom_basis + 17
        n_in = conf.n_atom_basis + 33
    else:
        # n_in = conf.n_atom_basis + 16
        n_in = conf.n_atom_basis
        # n_in = conf.n_atom_basis + 32

    model = atm.AtomisticModel(reps, [
        AtomPairwise(
            n_in=n_in,
            n_out=4,
            n_layers=conf.pairwise_layers,
        ),
        atm.Atomwise(
            return_contributions=True,
            n_in=n_in,
            n_out=3,
            n_layers=conf.atomwise_layers,
        ),
            # atm.Energy(n_in=128),
            # atm.DipoleMoment(n_in=128),
            # atm.ElementalDipoleMoment(n_in=128),
        ])
    model = model.to(device)

    if conf.optim == 'adam':
        opt = Adam(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    elif conf.optim == 'adamw':
        opt = optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    else:
        raise Exception("Not supported optim {}".format(conf.optim))
    scheduler = CyclicLR(
        opt,
        base_lr=conf.clr_base_lr,
        max_lr=conf.clr_max_lr,
        step_size_up=len(train_loader) * 10,
        mode="exp_range",
        gamma=conf.clr_gamma,
        cycle_momentum=False,
    )
    early_stopping = EarlyStopping(patience=100)

    if conf.resume_from is not None:
        cv_resume = conf.resume_from['cv']
        start_epoch = conf.resume_from['epoch']
        if cv_resume != cv:
            continue
        ckpt = torch.load('{}/{}-{:03d}.ckpt'.format(conf.out_dir, cv, start_epoch))

        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

        writer = SummaryWriter(logdir=ckpt['writer_logdir'], purge_step=start_epoch)
        hist = pd.read_csv('{}/{}.csv'.format(conf.out_dir, cv)).to_dict('records')

        print('Loaded checkpoint cv {}, epoch {} from {}'.format(cv, start_epoch, conf.out_dir))
    else:
        hist = []
#         writer = SummaryWriter()
        start_epoch = 0

    for epoch in range(start_epoch, conf.epochs):
        train_result = train(train_loader, model, opt, scheduler, conf)
        val_result = validate(val_loader, model, conf)
        hist.append({
            'epoch': epoch,
            **train_result,
            **val_result,
        })
        df_hist = pd.DataFrame(hist)
#         print(df_hist)
        log_hist(df_hist, conf.logger_epoch, conf.types)
#         write_on_board(df_hist, writer, conf)

        if conf.is_save_epoch_fn is not None and conf.is_save_epoch_fn(epoch):
            torch.save({
                'model': model.state_dict(),
                'optimizer': opt.state_dict(),
                'scheduler': scheduler.state_dict(),
                'writer_logdir': writer.logdir,
            }, '{}/{}-{:03d}.ckpt'.format(conf.out_dir, cv, epoch + 1))
            df_hist.to_csv('{}/{}.csv'.format(conf.out_dir, cv))
            print('Saved checkpoint {}/{}-{:03d}.ckpt'.format(conf.out_dir, cv, epoch + 1))

        should_stop = early_stopping.step(val_result['val_loss_total'])
        if should_stop:
            print('Early stopping at {}'.format(epoch))
            break

    df_hist = pd.DataFrame(hist)
    best = df_hist.sort_values('val_loss_cc', ascending=True).head(1).iloc[0]
    print(best)

#     writer.close()
    if conf.is_one_cv:
        break

