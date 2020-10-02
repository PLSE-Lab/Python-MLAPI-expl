#!/usr/bin/env python
# coding: utf-8

# <h1>Protonet</h1>     
# **As discussed in this [thread](https://www.kaggle.com/c/humpback-whale-identification/discussion/81085#478051), I converted [this](https://github.com/daisukelab/protonet-fine-grained-clf) repository into kernel. My computer can not handle this much computation therefore I used this kernel to play with protonets. I thought it would be helpful for someone who want to do experiment with protonets. 
# The code here does not belongs to me but the kernel is in running form without no errors. **

# In[ ]:


import cv2


# In[ ]:


import os


PATH = os.path.dirname(os.path.realpath('../input/'))

DATA_PATH = '../input/'

EPSILON = 1e-8

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')


# In[ ]:


#few_shot.metric 1

import torch


def categorical_accuracy(y, y_pred):
    """Calculates categorical accuracy.
    # Arguments:
        y_pred: Prediction probabilities or logits of shape [batch_size, num_categories]
        y: Ground truth categories. Must have shape [batch_size,]
    """
    return torch.eq(y_pred.argmax(dim=-1), y).sum().item() / y_pred.shape[0]


NAMED_METRICS = {
    'categorical_accuracy': categorical_accuracy
}


# In[ ]:


#few_shot.utils.py 2
import torch
import os
import shutil
from typing import Tuple, List

#from config import EPSILON, PATH


def mkdir(dir):
    """Create a directory, ignoring exceptions

    # Arguments:
        dir: Path of directory to create
    """
    try:
        os.mkdir(dir)
    except:
        pass


def rmdir(dir):
    """Recursively remove a directory and contents, ignoring exceptions

   # Arguments:
       dir: Path of directory to recursively remove
   """
    try:
        shutil.rmtree(dir)
    except:
        pass


def setup_dirs():
    """Creates directories for this project."""
    mkdir(PATH + '/logs/')
    mkdir(PATH + '/logs/proto_nets')
    mkdir(PATH + '/logs/matching_nets')
    mkdir(PATH + '/models/')
    mkdir(PATH + '/models/proto_nets')
    mkdir(PATH + '/models/matching_nets')


def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))


def copy_weights(from_model: torch.nn.Module, to_model: torch.nn.Module):
    """Copies the weights from one model to another model.

    # Arguments:
        from_model: Model from which to source weights
        to_model: Model which will receive weights
    """
    if not from_model.__class__ == to_model.__class__:
        raise(ValueError("Models don't have the same architecture!"))

    for m_from, m_to in zip(from_model.modules(), to_model.modules()):
        is_linear = isinstance(m_to, torch.nn.Linear)
        is_conv = isinstance(m_to, torch.nn.Conv2d)
        is_bn = isinstance(m_to, torch.nn.BatchNorm2d)
        if is_linear or is_conv or is_bn:
            m_to.weight.data = m_from.weight.data.clone()
            if m_to.bias is not None:
                m_to.bias.data = m_from.bias.data.clone()


def autograd_graph(tensor: torch.Tensor) -> Tuple[
            List[torch.autograd.Function],
            List[Tuple[torch.autograd.Function, torch.autograd.Function]]
        ]:
    """Recursively retrieves the autograd graph for a particular tensor.

    # Arguments
        tensor: The Tensor to retrieve the autograd graph for

    # Returns
        nodes: List of torch.autograd.Functions that are the nodes of the autograd graph
        edges: List of (Function, Function) tuples that are the edges between the nodes of the autograd graph
    """
    nodes, edges = list(), list()

    def _add_nodes(tensor):
        if tensor not in nodes:
            nodes.append(tensor)

            if hasattr(tensor, 'next_functions'):
                for f in tensor.next_functions:
                    if f[0] is not None:
                        edges.append((f[0], tensor))
                        _add_nodes(f[0])

            if hasattr(tensor, 'saved_tensors'):
                for t in tensor.saved_tensors:
                    edges.append((t, tensor))
                    _add_nodes(t)

    _add_nodes(tensor.grad_fn)

    return nodes, edges


# In[ ]:


#few_shot.eval 3

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Callable, List, Union

#from few_shot.metrics import NAMED_METRICS


def evaluate(model: Module, dataloader: DataLoader, prepare_batch: Callable, metrics: List[Union[str, Callable]],
             loss_fn: Callable = None, prefix: str = 'val_', suffix: str = ''):
    """Evaluate a model on one or more metrics on a particular dataset
    # Arguments
        model: Model to evaluate
        dataloader: Instance of torch.utils.data.DataLoader representing the dataset
        prepare_batch: Callable to perform any desired preprocessing
        metrics: List of metrics to evaluate the model with. Metrics must either be a named metric (see `metrics.py`) or
            a Callable that takes predictions and ground truth labels and returns a scalar value
        loss_fn: Loss function to calculate over the dataset
        prefix: Prefix to prepend to the name of each metric - used to identify the dataset. Defaults to 'val_' as
            it is typical to evaluate on a held-out validation dataset
        suffix: Suffix to append to the name of each metric.
    """
    logs = {}
    seen = 0
    totals = {m: 0 for m in metrics}
    if loss_fn is not None:
        totals['loss'] = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, y = prepare_batch(batch)
            y_pred = model(x)

            seen += x.shape[0]

            if loss_fn is not None:
                totals['loss'] += loss_fn(y_pred, y).item() * x.shape[0]

            for m in metrics:
                if isinstance(m, str):
                    v = NAMED_METRICS[m](y, y_pred)
                else:
                    # Assume metric is a callable function
                    v = m(y, y_pred)

                totals[m] += v * x.shape[0]

    for m in ['loss'] + metrics:
        logs[prefix + m + suffix] = totals[m] / seen
    return logs


# In[ ]:


get_ipython().system('pip install albumentations')


# In[ ]:


#few_shot.callbacks 4

"""
Ports of Callback classes from the Keras library.
"""
from tqdm import tqdm
import numpy as np
import torch
from collections import OrderedDict, Iterable
import warnings
import os
import csv
import io

#from few_shot.eval import evaluate


class CallbackList(object):
    """Container abstracting a list of callbacks.
    # Arguments
        callbacks: List of `Callback` instances.
    """
    def __init__(self, callbacks):
        self.callbacks = [c for c in callbacks]

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        """Called right before processing a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Called at the end of training.
        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)


class Callback(object):
    def __init__(self):
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class DefaultCallback(Callback):
    """Records metrics over epochs by averaging over each batch.
    NB The metrics are calculated with a moving model
    """
    def on_epoch_begin(self, batch, logs=None):
        self.seen = 0
        self.totals = {}
        self.metrics = ['loss'] + self.params['metrics']

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 1) or 1
        self.seen += batch_size

        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k in self.metrics:
                if k in self.totals:
                    # Make value available to next callbacks.
                    logs[k] = self.totals[k] / self.seen


class ProgressBarLogger(Callback):
    """TQDM progress bar that displays the running average of loss and other metrics."""
    def __init__(self):
        super(ProgressBarLogger, self).__init__()

    def on_train_begin(self, logs=None):
        self.num_batches = self.params['num_batches']
        self.verbose = self.params['verbose']
        self.metrics = ['loss'] + self.params['metrics']
        self.epoch_metrics = self.params['epoch_metrics']

    def on_epoch_begin(self, epoch, logs=None):
        self.target = self.num_batches
        self.pbar = tqdm(total=self.target, desc='Epoch {}'.format(epoch))
        self.seen = 0

    def on_batch_begin(self, batch, logs=None):
        self.log_values = {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.seen += 1

        for k in self.metrics:
            if k in logs:
                self.log_values[k] = logs[k]

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.verbose and self.seen < self.target:
            self.pbar.update(1)
            self.pbar.set_postfix(self.log_values)

    def on_epoch_end(self, epoch, logs=None):
        # Update log values
        self.log_values = {}
        for k in self.metrics + self.epoch_metrics:
            if k in logs:
                self.log_values[k] = logs[k]

        if self.verbose:
            self.pbar.update(1)
            self.pbar.set_postfix(self.log_values)

        self.pbar.close()


class CSVLogger(Callback):
    """Callback that streams epoch results to a csv file.
    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.
    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = ''
        self._open_args = {'newline': '\n'}
        super(CSVLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'

        self.csv_file = open(self.filename,mode + self.file_flags,**self._open_args)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            fieldnames = ['epoch'] + self.keys
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None


class EvaluateMetrics(Callback):
    """Evaluates metrics on a dataset after every epoch.
    # Argments
        dataloader: torch.DataLoader of the dataset on which the model will be evaluated
        prefix: Prefix to prepend to the names of the metrics when they is logged. Defaults to 'val_' but can be changed
        if the model is to be evaluated on many datasets separately.
        suffix: Suffix to append to the names of the metrics when they is logged.
    """
    def __init__(self, dataloader, prefix='val_', suffix=''):
        super(EvaluateMetrics, self).__init__()
        self.dataloader = dataloader
        self.prefix = prefix
        self.suffix = suffix

    def on_train_begin(self, logs=None):
        self.metrics = self.params['metrics']
        self.prepare_batch = self.params['prepare_batch']
        self.loss_fn = self.params['loss_fn']

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update(
            evaluate(self.model, self.dataloader, self.prepare_batch, self.metrics, self.loss_fn, self.prefix, self.suffix)
        )


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.
    # Arguments
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        min_delta: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(self, monitor='val_loss', factor=0.1, patience=10,
                 verbose=0, mode='auto', min_delta=1e-4, cooldown=0, min_lr=0,
                 **kwargs):
        super(ReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        if mode not in ['auto', 'min', 'max']:
            raise ValueError('Mode must be one of (auto, min, max).')
        self.mode = mode
        self.monitor_op = None

        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if (self.mode == 'min' or
                (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self.optimiser = self.params['optimiser']
        self.min_lrs = [self.min_lr] * len(self.optimiser.param_groups)
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if len(self.optimiser.param_groups) == 1:
            logs['lr'] = self.optimiser.param_groups[0]['lr']
        else:
            for i, param_group in enumerate(self.optimiser.param_groups):
                logs['lr_{}'.format(i)] = param_group['lr']

        current = logs.get(self.monitor)

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.wait = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimiser.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.min_delta:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    def in_cooldown(self):
        return self.cooldown_counter > 0


class ModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options, which will be filled the value of `epoch` and keys in `logs`
    (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model checkpoints will be saved
    with the epoch number and the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            raise ValueError('Mode must be one of (auto, min, max).')

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less

        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        torch.save(self.model.state_dict(), filepath)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                torch.save(self.model.state_dict(), filepath)


class LearningRateScheduler(Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        self.optimiser = self.params['optimiser']

    def on_epoch_begin(self, epoch, logs=None):
        lrs = [self.schedule(epoch, param_group['lr']) for param_group in self.optimiser.param_groups]

        if not all(isinstance(lr, (float, np.float32, np.float64)) for lr in lrs):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        self.set_lr(epoch, lrs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if len(self.optimiser.param_groups) == 1:
            logs['lr'] = self.optimiser.param_groups[0]['lr']
        else:
            for i, param_group in enumerate(self.optimiser.param_groups):
                logs['lr_{}'.format(i)] = param_group['lr']

    def set_lr(self, epoch, lrs):
        for i, param_group in enumerate(self.optimiser.param_groups):
            new_lr = lrs[i]
            param_group['lr'] = new_lr
            if self.verbose:
                print('Epoch {:5d}: setting learning rate'
' of group {} to {:.4e}.'.format(epoch, i, new_lr))


# In[ ]:


get_ipython().system('pip install easydict')


# In[ ]:


#few_shot.core.py 5


from torch.utils.data import Sampler
from typing import List, Iterable, Callable, Tuple
import numpy as np
import torch

#from few_shot.metrics import categorical_accuracy
#from few_shot.callbacks import Callback


class NShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None):
        """PyTorch Sampler subclass that generates batches of n-shot, k-way, q-query tasks.
        Each n-shot task contains a "support set" of `k` sets of `n` samples and a "query set" of `k` sets
        of `q` samples. The support set and the query set are all grouped into one Tensor such that the first n * k
        samples are from the support set while the remaining q * k samples are from the query set.
        The support and query sets are sampled such that they are disjoint i.e. do not contain overlapping samples.
        # Arguments
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            episodes_per_epoch: Arbitrary number of batches of n-shot tasks to generate in one epoch
            n_shot: int. Number of samples for each class in the n-shot classification tasks.
            k_way: int. Number of classes in the n-shot classification tasks.
            q_queries: int. Number query samples for each class in the n-shot classification tasks.
            num_tasks: Number of n-shot tasks to group into a single batch
            fixed_tasks: If this argument is specified this Sampler will always generate tasks from
                the specified classes
        """
        super(NShotTaskSampler, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1.')

        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.k = k
        self.n = n
        self.q = q
        self.fixed_tasks = fixed_tasks

        self.i_task = 0

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []

            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    # Get random classes
                    episode_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k, replace=False)
                else:
                    # Loop through classes in fixed_tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1

                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]

                support_k = {k: None for k in episode_classes}
                for k in episode_classes:
                    # Select support examples
                    support = df[df['class_id'] == k].sample(self.n)
                    support_k[k] = support

                    for i, s in support.iterrows():
                        batch.append(s['id'])

                for k in episode_classes:
                    query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q)
                    for i, q in query.iterrows():
                        batch.append(q['id'])

            yield np.stack(batch)


class EvaluateFewShot(Callback):
    """Evaluate a network on  an n-shot, k-way classification tasks after every epoch.
    # Arguments
        eval_fn: Callable to perform few-shot classification. Examples include `proto_net_episode`,
            `matching_net_episode` and `meta_gradient_step` (MAML).
        num_tasks: int. Number of n-shot classification tasks to evaluate the model with.
        n_shot: int. Number of samples for each class in the n-shot classification tasks.
        k_way: int. Number of classes in the n-shot classification tasks.
        q_queries: int. Number query samples for each class in the n-shot classification tasks.
        task_loader: Instance of NShotWrapper class
        prepare_batch: function. The preprocessing function to apply to samples from the dataset.
        prefix: str. Prefix to identify dataset.
    """

    def __init__(self,
                 eval_fn: Callable,
                 num_tasks: int,
                 n_shot: int,
                 k_way: int,
                 q_queries: int,
                 taskloader: torch.utils.data.DataLoader,
                 prepare_batch: Callable,
                 prefix: str = 'val_',
                 **kwargs):
        super(EvaluateFewShot, self).__init__()
        self.eval_fn = eval_fn
        self.num_tasks = num_tasks
        self.n_shot = n_shot
        self.k_way = k_way
        self.q_queries = q_queries
        self.taskloader = taskloader
        self.prepare_batch = prepare_batch
        self.prefix = prefix
        self.kwargs = kwargs
        self.metric_name = f'{self.prefix}{self.n_shot}-shot_{self.k_way}-way_acc'

    def on_train_begin(self, logs=None):
        self.loss_fn = self.params['loss_fn']
        self.optimiser = self.params['optimiser']

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        seen = 0
        totals = {'loss': 0, self.metric_name: 0}
        for batch_index, batch in enumerate(self.taskloader):
            x, y = self.prepare_batch(batch)

            loss, y_pred = self.eval_fn(
                self.model,
                self.optimiser,
                self.loss_fn,
                x,
                y,
                n_shot=self.n_shot,
                k_way=self.k_way,
                q_queries=self.q_queries,
                train=False,
                **self.kwargs
            )

            seen += y_pred.shape[0]

            totals['loss'] += loss.item() * y_pred.shape[0]
            totals[self.metric_name] += categorical_accuracy(y, y_pred) * y_pred.shape[0]

        logs[self.prefix + 'loss'] = totals['loss'] / seen
        logs[self.metric_name] = totals[self.metric_name] / seen


def prepare_nshot_task(n: int, k: int, q: int) -> Callable:
    """Typical n-shot task preprocessing.
    # Arguments
        n: Number of samples for each class in the n-shot classification task
        k: Number of classes in the n-shot classification task
        q: Number of query samples for each class in the n-shot classification task
    # Returns
        prepare_nshot_task_: A Callable that processes a few shot tasks with specified n, k and q
    """
    def prepare_nshot_task_(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create 0-k label and move to GPU.
        TODO: Move to arbitrary device
        """
        x, y = batch
        x = x.float().cuda()
        # Create dummy 0-(num_classes - 1) label
        y = create_nshot_task_label(k, q).cuda()
        return x, y

    return prepare_nshot_task_


def create_nshot_task_label(k: int, q: int) -> torch.Tensor:
    """Creates an n-shot task label.
    Label has the structure:
        [0]*q + [1]*q + ... + [k-1]*q
    # TODO: Test this
    # Arguments
        k: Number of classes in the n-shot classification task
        q: Number of query samples for each class in the n-shot classification task
    # Returns
        y: Label vector for n-shot task of shape [q * k, ]
    """
    y = torch.arange(0, k, 1 / q).long()
    return y


# In[ ]:


#few_shot.train.py


"""
The `fit` function in this file implements a slightly modified version
of the Keras `model.fit()` API.
"""
import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Callable, List, Union

#from few_shot.callbacks import DefaultCallback, ProgressBarLogger, CallbackList, Callback
#from few_shot.metrics import NAMED_METRICS


def gradient_step(model: Module, optimiser: Optimizer, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor, **kwargs):
    """Takes a single gradient step.
    # Arguments
        model: Model to be fitted
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        x: Input samples
        y: Input targets
    """
    model.train()
    optimiser.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimiser.step()

    return loss, y_pred


def batch_metrics(model: Module, y_pred: torch.Tensor, y: torch.Tensor, metrics: List[Union[str, Callable]],
                  batch_logs: dict):
    """Calculates metrics for the current training batch
    # Arguments
        model: Model being fit
        y_pred: predictions for a particular batch
        y: labels for a particular batch
        batch_logs: Dictionary of logs for the current batch
    """
    model.eval()
    for m in metrics:
        if isinstance(m, str):
            batch_logs[m] = NAMED_METRICS[m](y, y_pred)
        else:
            # Assume metric is a callable function
            batch_logs = m(y, y_pred)

    return batch_logs


def fit(model: Module, optimiser: Optimizer, loss_fn: Callable, epochs: int, dataloader: DataLoader,
        prepare_batch: Callable, metrics: List[Union[str, Callable]] = None,
        epoch_metrics: List[str] = None, callbacks: List[Callback] = None,
        verbose: bool =True, fit_function: Callable = gradient_step, fit_function_kwargs: dict = {}):
    """Function to abstract away training loop.
    The benefit of this function is that allows training scripts to be much more readable and allows for easy re-use of
    common training functionality provided they are written as a subclass of voicemap.Callback (following the
    Keras API).
    # Arguments
        model: Model to be fitted.
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        epochs: Number of epochs of fitting to be performed
        dataloader: `torch.DataLoader` instance to fit the model to
        prepare_batch: Callable to perform any desired preprocessing
        metrics: Optional list of metrics to evaluate the model with
        epoch_metrics: Optional list of metrics on top of metrics at the end of epoch
        callbacks: Additional functionality to incorporate into training such as logging metrics to csv, model
            checkpointing, learning rate scheduling etc... See voicemap.callbacks for more.
        verbose: All print output is muted if this argument is `False`
        fit_function: Function for calculating gradients. Leave as default for simple supervised training on labelled
            batches. For more complex training procedures (meta-learning etc...) you will need to write your own
            fit_function
        fit_function_kwargs: Keyword arguments to pass to `fit_function`
    """
    # Determine number of samples:
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size

    callbacks = CallbackList([DefaultCallback(), ] + (callbacks or []) + [ProgressBarLogger(), ])
    callbacks.set_model(model)
    callbacks.set_params({
        'num_batches': num_batches,
        'batch_size': batch_size,
        'verbose': verbose,
        'metrics': (metrics or []),
        'epoch_metrics': (epoch_metrics or []),
        'prepare_batch': prepare_batch,
        'loss_fn': loss_fn,
        'optimiser': optimiser
    })

    if verbose:
        print('Begin training...')

    callbacks.on_train_begin()

    for epoch in range(1, epochs+1):
        callbacks.on_epoch_begin(epoch)

        epoch_logs = {}
        for batch_index, batch in enumerate(dataloader):
            batch_logs = dict(batch=batch_index, size=(batch_size or 1))

            callbacks.on_batch_begin(batch_index, batch_logs)

            x, y = prepare_batch(batch)

            loss, y_pred = fit_function(model, optimiser, loss_fn, x, y, **fit_function_kwargs)
            batch_logs['loss'] = loss.item()

            # Loops through all metrics
            batch_logs = batch_metrics(model, y_pred, y, metrics, batch_logs)

            callbacks.on_batch_end(batch_index, batch_logs)

        # Run on epoch end
        callbacks.on_epoch_end(epoch, epoch_logs)

    # Run on train end
    if verbose:
        print('Finished.')
    callbacks.on_train_end()


# In[ ]:


#few_shot.proto.py

import torch
from torch.optim import Optimizer
from torch.nn import Module
from typing import Callable

#from few_shot.utils import pairwise_distances


def proto_net_episode(model: Module,
                      optimiser: Optimizer,
                      loss_fn: Callable,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      n_shot: int,
                      k_way: int,
                      q_queries: int,
                      distance: str,
                      train: bool):
    """Performs a single training episode for a Prototypical Network.
    # Arguments
        model: Prototypical Network to be trained.
        optimiser: Optimiser to calculate gradient step
        loss_fn: Loss function to calculate between predictions and outputs. Should be cross-entropy
        x: Input samples of few shot classification task
        y: Input labels of few shot classification task
        n_shot: Number of examples per class in the support set
        k_way: Number of classes in the few shot classification task
        q_queries: Number of examples per class in the query set
        distance: Distance metric to use when calculating distance between class prototypes and queries
        train: Whether (True) or not (False) to perform a parameter update
    # Returns
        loss: Loss of the Prototypical Network on this task
        y_pred: Predicted class probabilities for the query set on this task
    """
    if train:
        # Zero gradients
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    # Embed all samples
    embeddings = model(x)

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot*k_way]
    queries = embeddings[n_shot*k_way:]
    prototypes = compute_prototypes(support, k_way, n_shot)

    # Calculate squared distances between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
    distances = pairwise_distances(queries, prototypes, distance)

    # Calculate log p_{phi} (y = k | x)
    log_p_y = (-distances).log_softmax(dim=1)
    loss = loss_fn(log_p_y, y)

    # Prediction probabilities are softmax over distances
    y_pred = (-distances).softmax(dim=1)

    if train:
        # Take gradient step
        loss.backward()
        optimiser.step()
    else:
        pass

    return loss, y_pred


def compute_prototypes(support: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Compute class prototypes from support samples.
    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task
    # Returns
        class_prototypes: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    class_prototypes = support.reshape(k, n, -1).mean(dim=1)
    return class_prototypes


# In[ ]:


#few_shot.models

from torch import nn
import numpy as np
import torch.nn.functional as F
import torch
from typing import Dict


##########
# Layers #
##########
class Flatten(nn.Module):
    """Converts N-dimensional Tensor of shape [batch_size, d1, d2, ..., dn] to 2-dimensional Tensor
    of shape [batch_size, d1*d2*...*dn].
    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class GlobalMaxPool1d(nn.Module):
    """Performs global max pooling over the entire length of a batched 1D tensor
    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return nn.functional.max_pool1d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


class GlobalAvgPool2d(nn.Module):
    """Performs global average pooling over the entire height and width of a batched 2D tensor
    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return nn.functional.avg_pool2d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.
    # Arguments
        in_channels:
        out_channels:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


def functional_conv_block(x: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor,
                          bn_weights, bn_biases) -> torch.Tensor:
    """Performs 3x3 convolution, ReLu activation, 2x2 max pooling in a functional fashion.
    # Arguments:
        x: Input Tensor for the conv block
        weights: Weights for the convolutional block
        biases: Biases for the convolutional block
        bn_weights:
        bn_biases:
    """
    x = F.conv2d(x, weights, biases, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x


##########
# Models #
##########
def get_few_shot_encoder(num_input_channels=1) -> nn.Module:
    """Creates a few shot encoder as used in Matching and Prototypical Networks
    # Arguments:
        num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
            miniImageNet = 3
    """
    return nn.Sequential(
        conv_block(num_input_channels, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        Flatten(),
    )


class FewShotClassifier(nn.Module):
    def __init__(self, num_input_channels: int, k_way: int, final_layer_size: int = 64):
        """Creates a few shot classifier as used in MAML.
        This network should be identical to the one created by `get_few_shot_encoder` but with a
        classification layer on top.
        # Arguments:
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            k_way: Number of classes the model will discriminate between
            final_layer_size: 64 for Omniglot, 1600 for miniImageNet
        """
        super(FewShotClassifier, self).__init__()
        self.conv1 = conv_block(num_input_channels, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)

        self.logits = nn.Linear(final_layer_size, k_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)

        return self.logits(x)

    def functional_forward(self, x, weights):
        """Applies the same forward pass using PyTorch functional operators using a specified set of weights."""

        for block in [1, 2, 3, 4]:
            x = functional_conv_block(x, weights[f'conv{block}.0.weight'], weights[f'conv{block}.0.bias'],
                                      weights.get(f'conv{block}.1.weight'), weights.get(f'conv{block}.1.bias'))

        x = x.view(x.size(0), -1)

        x = F.linear(x, weights['logits.weight'], weights['logits.bias'])

        return x


class MatchingNetwork(nn.Module):
    def __init__(self, n: int, k: int, q: int, fce: bool, num_input_channels: int,
                 lstm_layers: int, lstm_input_size: int, unrolling_steps: int, device: torch.device):
        """Creates a Matching Network as described in Vinyals et al.
        # Arguments:
            n: Number of examples per class in the support set
            k: Number of classes in the few shot classification task
            q: Number of examples per class in the query set
            fce: Whether or not to us fully conditional embeddings
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            lstm_layers: Number of LSTM layers in the bidrectional LSTM g that embeds the support set (fce = True)
            lstm_input_size: Input size for the bidirectional and Attention LSTM. This is determined by the embedding
                dimension of the few shot encoder which is in turn determined by the size of the input data. Hence we
                have Omniglot -> 64, miniImageNet -> 1600.
            unrolling_steps: Number of unrolling steps to run the Attention LSTM
            device: Device on which to run computation
        """
        super(MatchingNetwork, self).__init__()
        self.n = n
        self.k = k
        self.q = q
        self.fce = fce
        self.num_input_channels = num_input_channels
        self.encoder = get_few_shot_encoder(self.num_input_channels)
        if self.fce:
            self.g = BidrectionalLSTM(lstm_input_size, lstm_layers).to(device, dtype=torch.float)
            self.f = AttentionLSTM(lstm_input_size, unrolling_steps=unrolling_steps).to(device, dtype=torch.float)

    def forward(self, inputs):
        pass


class BidrectionalLSTM(nn.Module):
    def __init__(self, size: int, layers: int):
        """Bidirectional LSTM used to generate fully conditional embeddings (FCE) of the support set as described
        in the Matching Networks paper.
        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            layers: Number of LSTM layers
        """
        super(BidrectionalLSTM, self).__init__()
        self.num_layers = layers
        self.batch_size = 1
        # Force input size and hidden size to be the same in order to implement
        # the skip connection as described in Appendix A.1 and A.2 of Matching Networks
        self.lstm = nn.LSTM(input_size=size,
                            num_layers=layers,
                            hidden_size=size,
                            bidirectional=True)

    def forward(self, inputs):
        # Give None as initial state and Pytorch LSTM creates initial hidden states
        output, (hn, cn) = self.lstm(inputs, None)

        forward_output = output[:, :, :self.lstm.hidden_size]
        backward_output = output[:, :, self.lstm.hidden_size:]

        # g(x_i, S) = h_forward_i + h_backward_i + g'(x_i) as written in Appendix A.2
        # AKA A skip connection between inputs and outputs is used
        output = forward_output + backward_output + inputs
        return output, hn, cn


class AttentionLSTM(nn.Module):
    def __init__(self, size: int, unrolling_steps: int):
        """Attentional LSTM used to generate fully conditional embeddings (FCE) of the query set as described
        in the Matching Networks paper.
        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            unrolling_steps: Number of steps of attention over the support set to compute. Analogous to number of
                layers in a regular LSTM
        """
        super(AttentionLSTM, self).__init__()
        self.unrolling_steps = unrolling_steps
        self.lstm_cell = nn.LSTMCell(input_size=size,
                                     hidden_size=size)

    def forward(self, support, queries):
        # Get embedding dimension, d
        if support.shape[-1] != queries.shape[-1]:
            raise(ValueError("Support and query set have different embedding dimension!"))

        batch_size = queries.shape[0]
        embedding_dim = queries.shape[1]

        h_hat = torch.zeros_like(queries).cuda().float()
        c = torch.zeros(batch_size, embedding_dim).cuda().float()

        for k in range(self.unrolling_steps):
            # Calculate hidden state cf. equation (4) of appendix A.2
            h = h_hat + queries

            # Calculate softmax attentions between hidden states and support set embeddings
            # cf. equation (6) of appendix A.2
            attentions = torch.mm(h, support.t())
            attentions = attentions.softmax(dim=1)

            # Calculate readouts from support set embeddings cf. equation (5)
            readout = torch.mm(attentions, support)

            # Run LSTM cell cf. equation (3)
            # h_hat, c = self.lstm_cell(queries, (torch.cat([h, readout], dim=1), c))
            h_hat, c = self.lstm_cell(queries, (h + readout, c))

        h = h_hat + queries
        return h


# In[ ]:





# In[ ]:


#few_shot.matching

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.nn import Module
from torch.nn.modules.loss import _Loss as Loss

#from config import EPSILON
#from few_shot.core import create_nshot_task_label
#from few_shot.utils import pairwise_distances


def matching_net_episode(model: Module,
                         optimiser: Optimizer,
                         loss_fn: Loss,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         n_shot: int,
                         k_way: int,
                         q_queries: int,
                         distance: str,
                         fce: bool,
                         train: bool):
    """Performs a single training episode for a Matching Network.
    # Arguments
        model: Matching Network to be trained.
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        x: Input samples of few shot classification task
        y: Input labels of few shot classification task
        n_shot: Number of examples per class in the support set
        k_way: Number of classes in the few shot classification task
        q_queries: Number of examples per class in the query set
        distance: Distance metric to use when calculating distance between support and query set samples
        fce: Whether or not to us fully conditional embeddings
        train: Whether (True) or not (False) to perform a parameter update
    # Returns
        loss: Loss of the Matching Network on this task
        y_pred: Predicted class probabilities for the query set on this task
    """
    if train:
        # Zero gradients
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    # Embed all samples
    embeddings = model.encoder(x)

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot * k_way]
    queries = embeddings[n_shot * k_way:]

    # Optionally apply full context embeddings
    if fce:
        # LSTM requires input of shape (seq_len, batch, input_size). `support` is of
        # shape (k_way * n_shot, embedding_dim) and we want the LSTM to treat the
        # support set as a sequence so add a single dimension to transform support set
        # to the shape (k_way * n_shot, 1, embedding_dim) and then remove the batch dimension
        # afterwards

        # Calculate the fully conditional embedding, g, for support set samples as described
        # in appendix A.2 of the paper. g takes the form of a bidirectional LSTM with a
        # skip connection from inputs to outputs
        support, _, _ = model.g(support.unsqueeze(1))
        support = support.squeeze(1)

        # Calculate the fully conditional embedding, f, for the query set samples as described
        # in appendix A.1 of the paper.
        queries = model.f(support, queries)

    # Efficiently calculate distance between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
    distances = pairwise_distances(queries, support, distance)

    # Calculate "attention" as softmax over support-query distances
    attention = (-distances).softmax(dim=1)

    # Calculate predictions as in equation (1) from Matching Networks
    # y_hat = \sum_{i=1}^{k} a(x_hat, x_i) y_i
    y_pred = matching_net_predictions(attention, n_shot, k_way, q_queries)

    # Calculated loss with negative log likelihood
    # Clip predictions for numerical stability
    clipped_y_pred = y_pred.clamp(EPSILON, 1 - EPSILON)
    loss = loss_fn(clipped_y_pred.log(), y)

    if train:
        # Backpropagate gradients
        loss.backward()
        # I found training to be quite unstable so I clip the norm
        # of the gradient to be at most 1
        clip_grad_norm_(model.parameters(), 1)
        # Take gradient step
        optimiser.step()

    return loss, y_pred


def matching_net_predictions(attention: torch.Tensor, n: int, k: int, q: int) -> torch.Tensor:
    """Calculates Matching Network predictions based on equation (1) of the paper.
    The predictions are the weighted sum of the labels of the support set where the
    weights are the "attentions" (i.e. softmax over query-support distances) pointing
    from the query set samples to the support set samples.
    # Arguments
        attention: torch.Tensor containing softmax over query-support distances.
            Should be of shape (q * k, k * n)
        n: Number of support set samples per class, n-shot
        k: Number of classes in the episode, k-way
        q: Number of query samples per-class
    # Returns
        y_pred: Predicted class probabilities
    """
    if attention.shape != (q * k, k * n):
        raise(ValueError(f'Expecting attention Tensor to have shape (q * k, k * n) = ({q * k, k * n})'))

    # Create one hot label vector for the support set
    y_onehot = torch.zeros(k * n, k)

    # Unsqueeze to force y to be of shape (K*n, 1) as this
    # is needed for .scatter()
    y = create_nshot_task_label(k, n).unsqueeze(-1)
    y_onehot = y_onehot.scatter(1, y, 1)

    y_pred = torch.mm(attention, y_onehot.cuda().float())
    return y_pred


# In[ ]:


#few_shot.maml

import torch
from collections import OrderedDict
from torch.optim import Optimizer
from torch.nn import Module
from typing import Dict, List, Callable, Union

#from few_shot.core import create_nshot_task_label


def replace_grad(parameter_gradients, parameter_name):
    def replace_grad_(module):
        return parameter_gradients[parameter_name]

    return replace_grad_


def meta_gradient_step(model: Module,
                       optimiser: Optimizer,
                       loss_fn: Callable,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       n_shot: int,
                       k_way: int,
                       q_queries: int,
                       order: int,
                       inner_train_steps: int,
                       inner_lr: float,
                       train: bool,
                       device: Union[str, torch.device]):
    """
    Perform a gradient step on a meta-learner.
    # Arguments
        model: Base model of the meta-learner being trained
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        x: Input samples for all few shot tasks
        y: Input labels of all few shot tasks
        n_shot: Number of examples per class in the support set of each task
        k_way: Number of classes in the few shot classification task of each task
        q_queries: Number of examples per class in the query set of each task. The query set is used to calculate
            meta-gradients after applying the update to
        order: Whether to use 1st order MAML (update meta-learner weights with gradients of the updated weights on the
            query set) or 2nd order MAML (use 2nd order updates by differentiating through the gradients of the updated
            weights on the query with respect to the original weights).
        inner_train_steps: Number of gradient steps to fit the fast weights during each inner update
        inner_lr: Learning rate used to update the fast weights on the inner update
        train: Whether to update the meta-learner weights at the end of the episode.
        device: Device on which to run computation
    """
    data_shape = x.shape[2:]
    create_graph = (True if order == 2 else False) and train

    task_gradients = []
    task_losses = []
    task_predictions = []
    for meta_batch in x:
        # By construction x is a 5D tensor of shape: (meta_batch_size, n*k + q*k, channels, width, height)
        # Hence when we iterate over the first  dimension we are iterating through the meta batches
        x_task_train = meta_batch[:n_shot * k_way]
        x_task_val = meta_batch[n_shot * k_way:]

        # Create a fast model using the current meta model weights
        fast_weights = OrderedDict(model.named_parameters())

        # Train the model for `inner_train_steps` iterations
        for inner_batch in range(inner_train_steps):
            # Perform update of model weights
            y = create_nshot_task_label(k_way, n_shot).to(device)
            logits = model.functional_forward(x_task_train, fast_weights)
            loss = loss_fn(logits, y)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)

            # Update weights manually
            fast_weights = OrderedDict(
                (name, param - inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

        # Do a pass of the model on the validation data from the current task
        y = create_nshot_task_label(k_way, q_queries).to(device)
        logits = model.functional_forward(x_task_val, fast_weights)
        loss = loss_fn(logits, y)
        loss.backward(retain_graph=True)

        # Get post-update accuracies
        y_pred = logits.softmax(dim=1)
        task_predictions.append(y_pred)

        # Accumulate losses and gradients
        task_losses.append(loss)
        gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
        named_grads = {name: g for ((name, _), g) in zip(fast_weights.items(), gradients)}
        task_gradients.append(named_grads)

    if order == 1:
        if train:
            sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).mean(dim=0)
                                  for k in task_gradients[0].keys()}
            hooks = []
            for name, param in model.named_parameters():
                hooks.append(
                    param.register_hook(replace_grad(sum_task_gradients, name))
                )

            model.train()
            optimiser.zero_grad()
            # Dummy pass in order to create `loss` variable
            # Replace dummy gradients with mean task gradients using hooks
            logits = model(torch.zeros((k_way, ) + data_shape).to(device, dtype=torch.float))
            loss = loss_fn(logits, create_nshot_task_label(k_way, 1).to(device))
            loss.backward()
            optimiser.step()

            for h in hooks:
                h.remove()

        return torch.stack(task_losses).mean(), torch.cat(task_predictions)

    elif order == 2:
        model.train()
        optimiser.zero_grad()
        meta_batch_loss = torch.stack(task_losses).mean()

        if train:
            meta_batch_loss.backward()
            optimiser.step()

        return meta_batch_loss, torch.cat(task_predictions)
    else:
        raise ValueError('Order must be either 1 or 2.')


# In[ ]:


#dlclihe.util

import os
import sys
from pathlib import Path
import numpy as np
import warnings
warnings.simplefilter('ignore')
import numpy as np
np.warnings.filterwarnings('ignore')

import IPython
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from easydict import EasyDict
from tqdm import tqdm_notebook
import shutil
import datetime

## File utilities

def ensure_folder(folder):
    """Make sure a folder exists."""
    Path(folder).mkdir(exist_ok=True, parents=True)

def ensure_delete(folder_or_file):
    anything = Path(folder_or_file)
    if anything.is_dir():
        shutil.rmtree(str(folder_or_file))
    elif anything.exists():
        anything.unlink()

def copy_file(src, dst):
    """Copy source file to destination file."""
    assert Path(src).is_file()
    shutil.copy(str(src), str(dst))

def _copy_any(src, dst, symlinks):
    if Path(src).is_dir():
        if Path(dst).is_dir():
            dst = Path(dst)/Path(src).name
        assert not Path(dst).exists()
        shutil.copytree(src, dst, symlinks=symlinks)
    else:
        copy_file(src, dst)

def copy_any(src, dst, symlinks=True):
    """Copy any file or folder recursively.
    Source file can be list/array of files.
    """
    do_list_item(_copy_any, src, dst, symlinks)

def do_list_item(func, src, *prms):
    if isinstance(src, (list, tuple, np.ndarray)):
        result = True
        for element in src:
            result = do_list_item(func, element, *prms) and result
        return result
    else:
        return func(src, *prms)

def _move_file(src, dst):
    shutil.move(str(src), str(dst))

def move_file(src, dst):
    """Move source file to destination file/folder.
    Source file can be list/array of files.
    """
    do_list_item(_move_file, src, dst)

def symlink_file(fromfile, tofile):
    """Make fromfile's symlink as tofile."""
    Path(tofile).symlink_to(fromfile)

def make_copy_to(dest_folder, files, n_sample=None, operation=copy_file):
    """Do file copy like operation from files to dest_folder.
    
    If n_sample is set, it creates symlinks up to number of n_sample files.
    If n_sample is greater than len(files), symlinks are repeated twice or more until it reaches to n_sample.
    If n_sample is less than len(files), n_sample symlinks are created for the top n_sample samples in files."""
    dest_folder.mkdir(exist_ok=True, parents=True)
    if n_sample is None:
        n_sample = len(files)

    _done = False
    _dup = 0
    _count = 0
    while not _done: # yet
        for f in files:
            f = Path(f)
            name = f.stem+('_%d'%_dup)+f.suffix if 0 < _dup else f.name
            to_file = dest_folder / name
            operation(f, to_file)
            _count += 1
            _done = n_sample <= _count
            if _done: break
        _dup += 1
    print('Now', dest_folder, 'has', len(list(dest_folder.glob('*'))), 'files.')

## Log utilities

import logging
_loggers = {}
def get_logger(name=None, level=logging.DEBUG, format=None, print=True, output_file=None):
    """One liner to get logger.
    See test_log.py for example.
    """
    name = name or __name__
    if _loggers.get(name):
        return _loggers.get(name)
    else:
        log = logging.getLogger(name)
    formatter = logging.Formatter(format or '%(asctime)s %(name)s %(funcName)s [%(levelname)s]: %(message)s')
    def add_handler(handler):
        handler.setFormatter(formatter)
        handler.setLevel(level)
        log.addHandler(handler)
    if print:
        add_handler(logging.StreamHandler())
    if output_file:
        ensure_folder(Path(output_file).parent)
        add_handler(logging.FileHandler(output_file))
    log.setLevel(level)
    log.propagate = False
    _loggers[name] = log
    return log

## Multi process utilities

def caller_func_name(level=2):
    """Return caller function name."""
    return sys._getframe(level).f_code.co_name

def _file_mutex_filename(filename):
    return filename or '/tmp/'+Path(caller_func_name(level=3)).stem+'.lock'

def lock_file_mutex(filename=None):
    """Lock file mutex (usually placed under /tmp).
    Note that filename will be created based on caller function name.
    """
    filename = _file_mutex_filename(filename)
    with open(filename, 'w') as f:
        f.write('locked at {}'.format(datetime.datetime.now()))
def release_file_mutex(filename=None):
    """Release file mutex."""
    filename = _file_mutex_filename(filename)
    ensure_delete(filename)

def is_file_mutex_locked(filename=None):
    """Check if file mutex is locked or not."""
    filename = _file_mutex_filename(filename)
    return Path(filename).exists()

## Date utilities

def str_to_date(text):
    if '/' in text:
        temp_dt = datetime.datetime.strptime(text, '%Y/%m/%d')
    else:
        temp_dt = datetime.datetime.strptime(text, '%Y-%m-%d')
    return datetime.date(temp_dt.year, temp_dt.month, temp_dt.day)

def get_week_start_end_dates(week_no:int, year=None) -> [datetime.datetime, datetime.datetime]:
    """Get start and end date of an ISO calendar week.
    ISO week starts on Monday, and ends on Sunday.
    
    Arguments:
        week_no: ISO calendar week number
        year: Year to calculate, None will set this year
    Returns:
        [start_date:datetime, end_date:datetime]
    """
    if not year:
        year, this_week, this_day = datetime.datetime.today().isocalendar()
    start_date = datetime.datetime.strptime(f'{year}-W{week_no:02d}-1', "%G-W%V-%u").date()
    end_date = datetime.datetime.strptime(f'{year}-W{week_no:02d}-7', "%G-W%V-%u").date()
    return [start_date, end_date]

def get_this_week_no():
    """Get ISO calendar week no of today."""
    return datetime.datetime.today().isocalendar()[1]

## List utilities

def write_text_list(textfile, a_list):
    """Write list of str to a file with new lines."""
    with open(textfile, 'w') as f:
        f.write('\n'.join(a_list)+'\n')

def read_text_list(filename) -> list:
    """Read text file splitted as list of texts, stripped."""
    with open(filename) as f:
        lines = f.read().splitlines()
        return [l.strip() for l in lines]

from itertools import chain
def flatten_list(lists):
    return list(chain.from_iterable(lists))

# Thanks to https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
def all_elements_are_identical(iterator):
    """Check all elements in iterable like list are identical."""
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)

## Text utilities

# Thanks to https://github.com/dsindex/blog/wiki/%5Bpython%5D-difflib,-show-differences-between-two-strings
import difflib
def show_text_diff(text, n_text):
    """
    http://stackoverflow.com/a/788780
    Unify operations between two compared strings seqm is a difflib.
    SequenceMatcher instance whose a & b are strings
    """
    seqm = difflib.SequenceMatcher(None, text, n_text)
    output= []
    for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
        if opcode == 'equal':
            pass # output.append(seqm.a[a0:a1])
        elif opcode == 'insert':
            output.append("<INS>" + seqm.b[b0:b1] + "</INS>")
        elif opcode == 'delete':
            output.append("<DEL>" + seqm.a[a0:a1] + "</DEL>")
        elif opcode == 'replace':
            # seqm.a[a0:a1] -> seqm.b[b0:b1]
            output.append("<REPL>" + seqm.b[b0:b1] + "</REPL>")
        else:
            raise RuntimeError
    return ''.join(output)

import unicodedata
def unicode_visible_width(unistr):
    """Returns the number of printed characters in a Unicode string."""
    return sum([1 if unicodedata.east_asian_width(char) in ['N', 'Na'] else 2 for char in unistr])

## Pandas utilities

def df_to_csv_excel_friendly(df, filename, **args):
    """df.to_csv() to be excel friendly UTF-8 handling."""
    df.to_csv(filename, encoding='utf_8_sig', **args)

def df_merge_update(df_list_or_org_file, opt_joining_file=None):
    """Merge data frames while update duplicated index with following (joining) row.
    
    Usages:
        - df_merge_update([df1, df2, ...]) merges dfs in list.
        - df_merge_update(df1, df2) merges df1 and df2.
    """
    if opt_joining_file is not None:
        df_list = [df_list_or_org_file, opt_joining_file]
    else:
        df_list = df_list_or_org_file

    master = df_list[0]
    for df in df_list[1:]:
        tmp_df = pd.concat([master, df])
        master = tmp_df[~tmp_df.index.duplicated(keep='last')].sort_index()
    return master

def df_select_by_keyword(source_df, keyword, search_columns=None, as_mask=False):
    """Select data frame rows by a search keyword.
    Any row will be selected if any of its search columns contain the keyword.
    
    Returns:
        New data frame where rows have the keyword,
        or mask if as_mask is True.
    """
    search_columns = search_columns or source_df.columns
    masks = np.column_stack([source_df[col].str.contains(keyword, na=False) for col in search_columns])
    mask = masks.any(axis=1)
    if as_mask:
        return mask
    return source_df.loc[mask]

def df_select_by_keywords(source_df, keys_cols, and_or='or', as_mask=False):
    """Multi keyword version of df_select_by_keyword.
    Arguments:
        key_cols: dict defined as `{'keyword1': [search columns] or None, ...}`
    """
    masks = []
    for keyword in keys_cols:
        columns = keys_cols[keyword]
        mask = df_select_by_keyword(source_df, keyword, search_columns=columns, as_mask=True)
        masks.append(mask)
    mask = np.column_stack(masks).any(axis=1) if and_or == 'or' else            np.column_stack(masks).all(axis=1)
    if as_mask:
        return mask
    return source_df.loc[mask]

def df_str_replace(df, from_strs, to_str):
    """Apply str.replace to entire DataFrame inplace."""
    for i, row in df.iterrows():
        df.ix[i] = df.ix[i].str.replace(from_strs, to_str)

def df_cell_str_replace(df, from_str, to_str):
    """Replace cell string with new string if entire string matches."""
    for i, row in df.iterrows():
        for c in df.columns:
            df.at[i, c] = to_str if str(df.at[i, c]) == from_str else df.at[i, c]

_EXCEL_LIKE = ['.csv', '.xls', '.xlsx', '.xlsm']
def is_excel_file(filename):
    # not accepted if suffix == '.csv': return True
    return Path(filename).suffix.lower() in _EXCEL_LIKE

def is_csv_file(filename):
    return Path(filename).suffix.lower() == '.csv'

def pd_read_excel_keep_dtype(io, **args):
    """pd.read_excel() wrapper to do as described in pandas document:
    '... preserve data as stored in Excel and not interpret dtype'
    Details:
        - String '1' might be loaded as int 1 by pd.read_excel(file).
        - By setting `dtype=object` it will preserve it as string '1'.
    """
    return pd.read_excel(io, dtype=object, **args)

def pd_read_csv_as_str(filename, **args):
    """pd.read_csv() wrapper to preserve data type = str"""
    return pd.read_csv(filename, dtype=object, **args)

def df_load_excel_like(filename, preserve_dtype=True, **args):
    """Load Excel like files. (csv, xlsx, ...)"""
    if is_csv_file(filename):
        if preserve_dtype:
            return pd_read_csv_as_str(filename, **args)
        return pd.read_csv(filename, **args)
    if preserve_dtype:
        return pd_read_excel_keep_dtype(filename, **args)
    return pd.read_excel(filename, **args)

import codecs
def df_read_sjis_csv(filename, **args):
    """Read shift jis Japanese csv file.
    Thanks to https://qiita.com/niwaringo/items/d2a30e04e08da8eaa643
    """
    with codecs.open(filename, 'r', 'Shift-JIS', 'ignore') as file:
        return pd.read_table(file, delimiter=',', **args)

## Dataset utilities

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def flatten_y_if_onehot(y):
    """De-one-hot y, i.e. [0,1,0,0,...] to 1 for all y."""
    return y if len(np.array(y).shape) == 1 else np.argmax(y, axis = -1)

def get_class_distribution(y):
    """Calculate number of samples per class."""
    # y_cls can be one of [OH label, index of class, class label name string]
    # convert OH to index of class
    y_cls = flatten_y_if_onehot(y)
    # y_cls can be one of [index of class, class label name]
    classset = sorted(list(set(y_cls)))
    sample_distribution = {cur_cls:len([one for one in y_cls if one == cur_cls]) for cur_cls in classset}
    return sample_distribution

def get_class_distribution_list(y, num_classes):
    """Calculate number of samples per class as list"""
    dist = get_class_distribution(y)
    assert(y[0].__class__ != str) # class index or class OH label only
    list_dist = np.zeros((num_classes))
    for i in range(num_classes):
        if i in dist:
            list_dist[i] = dist[i]
    return list_dist

def _balance_class(X, y, min_or_max, sampler_class, random_state):
    """Balance class distribution with sampler_class."""
    y_cls = flatten_y_if_onehot(y)
    distribution = get_class_distribution(y_cls)
    classes = list(distribution.keys())
    counts  = list(distribution.values())
    nsamples = np.max(counts) if min_or_max == 'max'           else np.min(counts)
    flat_ratio = {cls:nsamples for cls in classes}
    Xidx = [[xidx] for xidx in range(len(X))]
    sampler_instance = sampler_class(ratio=flat_ratio, random_state=random_state)
    Xidx_resampled, y_cls_resampled = sampler_instance.fit_sample(Xidx, y_cls)
    sampled_index = [idx[0] for idx in Xidx_resampled]
    return np.array([X[idx] for idx in sampled_index]), np.array([y[idx] for idx in sampled_index])

def balance_class_by_over_sampling(X, y, random_state=42):
    """Balance class distribution with imbalanced-learn RandomOverSampler."""
    return  _balance_class(X, y, 'max', RandomOverSampler, random_state)

def balance_class_by_under_sampling(X, y, random_state=42):
    """Balance class distribution with imbalanced-learn RandomUnderSampler."""
    return  _balance_class(X, y, 'min', RandomUnderSampler, random_state)

def df_balance_class_by_over_sampling(df, label_column, random_state=42):
    """Balance class distribution in DataFrame with imbalanced-learn RandomOverSampler."""
    X, y = list(range(len(df))), list(df[label_column])
    X, _ = balance_class_by_over_sampling(X, y, random_state=random_state)
    return df.iloc[X].sort_index()

def df_balance_class_by_under_sampling(df, label_column, random_state=42):
    """Balance class distribution in DataFrame with imbalanced-learn RandomUnderSampler."""
    X, y = list(range(len(df))), list(df[label_column])
    X, _ = balance_class_by_under_sampling(X, y, random_state=random_state)
    return df.iloc[X].sort_index()

## Visualization utilities

def _expand_labels_from_y(y, labels):
    """Make sure y is index of label set."""
    if labels is None:
        labels = sorted(list(set(y)))
        y = [labels.index(_y) for _y in y]
    return y, labels

def visualize_class_balance(title, y, labels=None, sorted=False):
    y, labels = _expand_labels_from_y(y, labels)
    sample_dist_list = get_class_distribution_list(y, len(labels))
    if sorted:
        items = list(zip(labels, sample_dist_list))
        items.sort(key=lambda x:x[1], reverse=True)
        sample_dist_list = [x[1] for x in items]
        labels = [x[0] for x in items]
    index = range(len(labels))
    fig, ax = plt.subplots(1, 1, figsize = (16, 5))
    ax.bar(index, sample_dist_list)
    ax.set_xlabel('Label')
    ax.set_xticks(index)
    ax.set_xticklabels(labels, rotation='vertical')
    ax.set_ylabel('Number of Samples')
    ax.set_title(title)
    fig.show()

from collections import OrderedDict
def print_class_balance(title, y, labels=None, sorted=False):
    y, labels = _expand_labels_from_y(y, labels)
    distributions = get_class_distribution(y)
    dist_dic = {labels[cls]:distributions[cls] for cls in distributions}
    if sorted:
        items = list(dist_dic.items())
        items.sort(key=lambda x:x[1], reverse=True)
        dist_dic = OrderedDict(items) # sorted(dist_dic.items(), key=...) didn't work for some reason...
    print(title, '=', dist_dic)
    zeroclasses = [label for i, label in enumerate(labels) if i not in distributions.keys()]
    if 0 < len(zeroclasses):
        print(' 0 sample classes:', zeroclasses)

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def calculate_clf_metrics(y_true, y_pred, average='weighted'):
    """Calculate metrics: f1/recall/precision/accuracy.
    # Arguments
        y_true: GT, an index of label or one-hot encoding format.
        y_pred: Prediction output, index or one-hot.
        average: `average` parameter passed to sklearn.metrics functions.
    # Returns
        Four metrics: f1, recall, precision, accuracy.
    """
    y_true = flatten_y_if_onehot(y_true)
    y_pred = flatten_y_if_onehot(y_pred)
    if np.max(y_true) < 2 and np.max(y_pred) < 2:
        average = 'binary'

    f1 = f1_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    precision = precision_score(y_true, y_pred, average=average)
    accuracy = accuracy_score(y_true, y_pred)
    return f1, recall, precision, accuracy

def skew_bin_clf_preds(y_pred, binary_bias=None, logger=None):
    """Apply bias to prediction results for binary classification.
    Calculated as follows.
        p(y=1) := p(y=1) ^ binary_bias
        p(y=0) := 1 - p(y=0)
    0 < binary_bias < 1 will be optimistic with result=1.
    Inversely, 1 < binary_bias will make results pesimistic.
    """
    _preds = np.array(y_pred.copy())
    if binary_bias is not None:
        ps = np.power(_preds[:, 1], binary_bias)
        _preds[:, 1] = ps
        _preds[:, 0] = 1 - ps
        logger = get_logger() if logger is None else logger
        logger.info(f' @skew{"+" if binary_bias >= 0 else ""}{binary_bias}')
    return _preds

def print_clf_metrics(y_true, y_pred, average='weighted', binary_bias=None, title_prefix='', logger=None):
    """Calculate and print metrics: f1/recall/precision/accuracy.
    See calculate_clf_metrics() and skew_bin_clf_preds() for more detail.
    """
    # Add bias if binary_bias is set
    _preds = skew_bin_clf_preds(y_pred, binary_bias, logger=logger)
    # Calculate metrics
    f1, recall, precision, acc = calculate_clf_metrics(y_true, _preds, average=average)
    logger = get_logger() if logger is None else logger
    logger.info('{0:s}F1/Recall/Precision/Accuracy = {1:.4f}/{2:.4f}/{3:.4f}/{4:.4f}'           .format(title_prefix, f1, recall, precision, acc))

# Thanks to https://qiita.com/knknkn1162/items/be87cba14e38e2c0f656
def plt_japanese_font_ready():
    """Set font family with Japanese fonts.
    
    # How to install fonts:
        wget https://ipafont.ipa.go.jp/old/ipafont/IPAfont00303.php
        mv IPAfont00303.php IPAfont00303.zip
        unzip -q IPAfont00303.zip
        sudo cp IPAfont00303/*.ttf /usr/share/fonts/truetype/
    """
    plt.rcParams['font.family'] = 'IPAPGothic'

def plt_looks_good():
    """Plots will be looks good (at least to me)."""
    plt.rcParams["figure.figsize"] = [16, 10]
    plt.rcParams['font.size'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

# Thanks to http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_test, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """Plot confusion matrix."""
    po = np.get_printoptions()
    np.set_printoptions(precision=2)

    y_test = flatten_y_if_onehot(y_test)
    y_pred = flatten_y_if_onehot(y_pred)
    cm = confusion_matrix(y_test, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if title is None: title = 'Normalized confusion matrix'
    else:
        if title is None: title = 'Confusion matrix (not normalized)'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    np.set_printoptions(**po)


# In[ ]:


#dlcliche.math

import math
import numpy as np
import pandas as pd


def roundup(x, n=10):
    """Round up x to multiple of n."""
    return int(math.ceil(x / n)) * n


def running_mean(x, N):
    """Calculate running/rolling mean or moving average.
    Thanks to https://stackoverflow.com/a/27681394/6528729
    """
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def np_describe(arr):
    """Describe numpy array statistics.
    Thanks to https://qiita.com/AnchorBlues/items/051dc69e81705b52adad
    """
    return pd.DataFrame(pd.Series(arr.ravel()).describe()).transpose()


def np_softmax(z):
    """Numpy version softmax.
    Thanjs to https://stackoverflow.com/a/39558290/6528729
    """
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


class OnlineStats:
    """Calculate mean/variance of a vector online
    Thanks to https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """

    def __init__(self, length):
        self.K = np.zeros((length))
        self.Ex = np.zeros((length))
        self.Ex2 = np.zeros((length))
        self.n = 0

    def put(self, x):
        if self.n == 0:
            self.K = x
        self.n += 1
        d = x - self.K
        self.Ex += d
        self.Ex2 += d * d

    def undo(self, x):
        self.n -= 1
        d = x - self.K
        self.Ex -= d
        self.Ex2 -= d * d

    def mean(self):
        if self.n == 0:
            return np.zeros_like(self.K)
        return self.K + self.Ex / self.n

    def variance(self):
        if self.n < 2:
            return np.zeros_like(self.K)
        return (self.Ex2 - (self.Ex * self.Ex) / self.n) / (self.n - 1)

    def sigma(self):
        if self.n < 2:
            return np.zeros_like(self.K)
        return np.sqrt(self.variance())

    def count(self):
        return self.n


# In[ ]:


#dlcliche.image

# Thanks to https://github.com/ipython/ipython/issues/9732
from IPython import get_ipython
ipython = get_ipython()

# Determine if this is running in Jupyter notebook or not
if ipython:
    running_in_notebook = ipython.has_trait('kernel')

    if running_in_notebook:
        ipython.magic('reload_ext autoreload')
        ipython.magic('autoreload 2')
        ipython.magic('matplotlib inline')
else:
    # cannot even get ipython object...
    running_in_notebook = False


def fit_notebook_to_window():
    """Fit notebook width to width of browser window.
    Thanks to https://stackoverflow.com/questions/21971449/how-do-i-increase-the-cell-width-of-the-jupyter-ipython-notebook-in-my-browser
    """
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))


import cv2
import tqdm
import math
from PIL import Image
from multiprocessing import Pool

def resize_image(dest_folder, filename, shape):
    """Resize and save copy of image file to destination folder."""
    img = cv2.imread(str(filename))
    if shape is not None:
        img = cv2.resize(img, shape)
    outfile = str(Path(dest_folder)/Path(filename).name)
    cv2.imwrite(outfile, img)
    return outfile, (img.shape[1], img.shape[0]) # original size

def _resize_image_worker(args):
    return resize_image(args[0], args[1], args[2])

def resize_image_files(dest_folder, source_files, shape=(224, 224), num_threads=8, skip_if_any_there=False):
    """Make resized copy of listed images in parallel processes.
    Arguments:
        dest_folder: Destination folder to make copies.
        source_files: Source image files.
        shape: (Width, Depth) shape of copies. None will NOT resize and makes dead copy.
        num_threads: Number of parallel workers.
        skip_if_any_there: If True, skip processing processing if any file have already been done.
    Returns:
        List of image info (filename, original size) tuples, or None if skipped.
        ex)
        ```python
        [('tmp/8d6ed7c786dcbc93.jpg', (1024, 508)),
         ('tmp/8d6ee9921e4aeb18.jpg', (891, 1024)),
         ('tmp/8d6f00feedb09efa.jpg', (1024, 683))]
        ```
    """
    if skip_if_any_there:
        if (Path(dest_folder)/Path(source_files[0]).name).exists():
            return None
    # Create destination folder if needed
    ensure_folder(dest_folder)
    # Do resize
    if running_in_notebook:  # Workaround: not using pool on notebook
        returns = []
        for f in tqdm.tqdm(source_files, total=len(source_files)):
            returns.append(resize_image(dest_folder, f, shape))
    else:
        with Pool(num_threads) as p:
            args = [[dest_folder, f, shape] for f in source_files]
            returns = list(tqdm.tqdm(p.imap(_resize_image_worker, args), total=len(args)))
    return returns

def _get_shape_worker(filename):
    return Image.open(filename).size # Image.open() is much faster than cv2.imread()

def read_file_shapes(files, num_threads=8):
    """Read shape of files in parallel."""
    if running_in_notebook:  # Workaround: not using pool on notebook
        shapes = []
        for f in tqdm.tqdm(files, total=len(files)):
            shapes.append(_get_shape_worker(f))
    else:
        with Pool(num_threads) as p:
            shapes = list(tqdm.tqdm(p.imap(_get_shape_worker, files), total=len(files)))
    return np.array(shapes)

def load_rgb_image(filename):
    """Load image file and make sure that format is RGB."""
    img = cv2.imread(str(filename))
    if img is None:
        raise ValueError(f'Failed to load {filename}.')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def convert_mono_to_jpg(fromfile, tofile):
    """Convert monochrome image to color jpeg format.
    Linear copy to RGB channels. 
    https://docs.opencv.org/3.1.0/de/d25/imgproc_color_conversions.html
    Args:
        fromfile: float png mono image
        tofile: RGB color jpeg image.
    """
    img = np.array(Image.open(fromfile)) # TODO Fix this to cv2.imread
    img = img - np.min(img)
    img = img / (np.max(img) + 1e-4)
    img = (img * 255).astype(np.uint8) # [0,1) float to [0,255] uint8
    img = np.repeat(img[..., np.newaxis], 3, axis=-1) # mono to RGB color
    img = Image.fromarray(img)
    tofile = Path(tofile)
    img.save(tofile.with_suffix('.jpg'), 'JPEG', quality=100)

# Borrowing from fast.ai course notebook
from matplotlib import patches, patheffects
def subplot_matrix(rows, columns, figsize=(12, 12)):
    """Subplot utility for drawing matrix of images.
    # Usage
    Following will show images in 2x3 matrix.
    ```python
    axes = subplot_matrix(2, 3)
    for img, ax in zip(images, axes):
        show_image(img, ax=ax)
    ```
    """
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    return list(axes.flat) if 1 < rows*columns else [axes]


def show_np_image(img, figsize=None, ax=None):
    """Show numpy object image with figsize on axes of subplot.
    Using this with subplot_matrix() will make it easy to plot matrix of images.
    # Returns
        Axes of subplot created, or given."""
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax
def _draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])
def ax_draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
    _draw_outline(patch, 4)
def ax_draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    _draw_outline(text, 1)
def ax_draw_bbox(ax, bbox, class_name):
    """Object Detection Helper: Draw single bounding box with class name on top of image."""
    ax_draw_rect(ax, bbox)
    ax_draw_text(ax, bbox[:2], class_name)

def show_np_od_data(image, bboxes, labels, class_names=None, figsize=None):
    """Object Detection Helper: Show numpy object detector data (set of an image, bboxes and labels)."""
    ax = show_np_image(image, figsize=figsize)
    for bbox, label in zip(bboxes, labels):
        if class_names is not None:
            label = class_names[label]
        ax_draw_bbox(ax, bbox, label)
    plt.show()

def union_of_bboxes(height, width, bboxes, erosion_rate=0.0, to_int=False):
    """Calculate union bounding box of boxes.
    # Arguments
        height (float): Height of image or space.
        width (float): Width of image or space.
        bboxes (list): List like bounding boxes. Format is `[x_min, y_min, x_max, y_max]`.
        erosion_rate (float): How much each bounding box can be shrinked, useful for erosive cropping.
            Set this in range [0, 1]. 0 will not be erosive at all, 1.0 can make any bbox to lose its volume.
        to_int (bool): Returns as int if True.
    """
    x1, y1 = width, height
    x2, y2 = 0, 0
    for b in bboxes:
        w, h = b[2]-b[0], b[3]-b[1]
        lim_x1, lim_y1 = b[0] + erosion_rate*w, b[1] + erosion_rate*h
        lim_x2, lim_y2 = b[2] - erosion_rate*w, b[3] - erosion_rate*h
        x1, y1 = np.min([x1, lim_x1]), np.min([y1, lim_y1])
        x2, y2 = np.max([x2, lim_x2]), np.max([y2, lim_y2])
        #print(b, [lim_x1, lim_y1, lim_x2, lim_y2], [x1, y1, x2, y2])
    if to_int:
        x1, y1 = int(math.floor(x1)), int(math.floor(y1))
        x2, y2 = int(np.min([width, math.ceil(x2)])), int(np.min([height, math.ceil(y2)]))
    return x1, y1, x2, y2


# In[ ]:


#few_shot.extmodel_proto_net_clf.py


"""
For testing what if we use ImageNet pretrained model as ProtoNet??
"""
#from dlcliche.utils import *
#from dlcliche.math import *
#from dlcliche.image import show_np_image, subplot_matrix

from torchvision import models
from torch import nn
import torch
from tqdm import tqdm

# TODO: Support cpu environment

class BasePretrainedModel(nn.Module):
    def __init__(self, base_model=models.resnet18, n_embs=512, print_shape=False):
        super(BasePretrainedModel, self).__init__()
        resnet = base_model(pretrained=True)
        self.body = nn.Sequential(*list(resnet.children())[:-1])
        self.n_embs = n_embs
        self.print_shape = print_shape

    def forward(self, x):
        x = self.body(x)
        if self.print_shape:
            print(x.shape)
        return x.view(-1, self.n_embs)


class ExtModelProtoNetClf(object):
    """ProtoNet as conventional classifier using external model.
    Created for testing what if we use ImageNet pretrained model for getting embeddings.
    TODO Fix bad design for member-call-order dependency...
    """

    def __init__(self, model, classes, device):
        model.to(device)
        model.eval()
        self.model = model
        self.classes = classes
        self.device = device
        self.prototypes = None
        self.n_embeddings = None  # First get_embeddings() will set this
        self.n_classes = len(classes)
        self.log = get_logger()

    
    def _make_null_prototypes(self):
        self.prototypes = [OnlineStats(self.n_embeddings)for _ in range(self.n_classes)]
    
    def get_embeddings(self, dl, visualize=False):
        """Get embeddings for all samples available in dataloader."""
        gts, cur = [], 0
        with torch.no_grad():
            for batch_index, (X, y_gt) in tqdm(enumerate(dl), total=len(dl)):
                dev_X, y_gt = X.to(self.device), list(y_gt)
                this_embs = self.model(dev_X).cpu().detach().numpy()
                if cur == 0:
                    self.n_embeddings = this_embs.shape[-1]
                    embs = np.zeros((len(dl.dataset), self.n_embeddings))

                if visualize:
                    for i, ax in enumerate(subplot_matrix(columns=4, rows=2, figsize=(16, 8))):
                        if len(dl) <= batch_index * 8 + i: break
                        show_np_image(np.transpose(X[i].cpu().detach().numpy(), [1, 2, 0]), ax=ax)
                    plt.show()

                for i in range(len(this_embs)):
                    embs[cur] = this_embs[i]
                    gts.append(y_gt[i])
                    cur += 1
        return np.array(embs), gts

    def make_prototypes(self, support_set_dl, repeat=1, update=False, visualize=False):
        """Calculate prototypes by accumulating embeddings of all samples in given support set.
        Args:
             support_set_dl: support set dataloader.
             repeat: test parameter for what if we get prototype with augmented samples.
             update: set True if you don't want to update prototypes with new samples from dataloader.
        """
        # Get embeddings of support set samples
        embs, gts = self.get_embeddings(support_set_dl, visualize=visualize)
        # Make prototypes if not there
        
        if update:
            self.log.info('Using current prototypes.')
        else:
            self.log.info('Making new prototypes.')
            self._make_null_prototypes()
        
        # Update prototypes (just by feeding to online stat class)
        
        for i in range(repeat):
            for emb, cls in zip(embs, gts):
                if not isinstance(cls, int):
                    cls = self.classes.index(cls)
                self.prototypes[cls].put(emb)
            if i < repeat - 1:
                embs, gts = self.get_embeddings(support_set_dl)  # no visualization
        
    def predict_embeddings(self, X_embs, softmax=True):
        preds = np.zeros((len(X_embs), self.n_classes))
        proto_embs = [p.mean() for p in self.prototypes]
        for idx_sample, x in tqdm(enumerate(X_embs), total=len(X_embs)):
            for idx_class, proto in enumerate(proto_embs):
                preds[idx_sample, idx_class] = -np.log(np.sum((x - proto)**2) + 1e-300) # preventing log(0)
            if softmax:
                preds[idx_sample, :] = np_softmax(preds[idx_sample])
        return preds
    def predict(self, data_loader):
        embs, y_gts = self.get_embeddings(data_loader)
        return self.predict_embeddings(embs), y_gts

    def evaluate(self, data_loader):
        y_hat, y_gts = self.predict(data_loader)
        return calculate_clf_metrics(y_gts, y_hat)


# In[ ]:





# In[ ]:


#few_shot.datasets.py


from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

#from config import DATA_PATH


class OmniglotDataset(Dataset):
    def __init__(self, subset):
        """Dataset class representing Omniglot dataset
        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):
        instance = io.imread(self.datasetid_to_filepath[item])
        # Reindex to channels first format as supported by pytorch
        instance = instance[np.newaxis, :, :]

        # Normalise to 0-1
        instance = (instance - instance.min()) / (instance.max() - instance.min())

        label = self.datasetid_to_class_id[item]

        return torch.from_numpy(instance), label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.
        # Arguments
            subset: Name of the subset
        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            Omniglot dataset dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/Omniglot/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/Omniglot/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            alphabet = root.split('/')[-2]
            class_name = '{}.{}'.format(alphabet, root.split('/')[-1])

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'alphabet': alphabet,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images


class MiniImageNet(Dataset):
    def __init__(self, subset):
        """Dataset class representing miniImageNet dataset
        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.
        # Arguments
            subset: Name of the subset
        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            miniImageNet dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images


class DummyDataset(Dataset):
    def __init__(self, samples_per_class=10, n_classes=10, n_features=1):
        """Dummy dataset for debugging/testing purposes
        A sample from the DummyDataset has (n_features + 1) features. The first feature is the index of the sample
        in the data and the remaining features are the class index.
        # Arguments
            samples_per_class: Number of samples per class in the dataset
            n_classes: Number of distinct classes in the dataset
            n_features: Number of extra features each sample should have.
        """
        self.samples_per_class = samples_per_class
        self.n_classes = n_classes
        self.n_features = n_features

        # Create a dataframe to be consistent with other Datasets
        self.df = pd.DataFrame({
            'class_id': [i % self.n_classes for i in range(len(self))]
        })
        self.df = self.df.assign(id=self.df.index.values)

    def __len__(self):
        return self.samples_per_class * self.n_classes

    def __getitem__(self, item):
        class_id = item % self.n_classes
        return np.array([item] + [class_id]*self.n_features, dtype=np.float), float(class_id)


# In[ ]:





# In[ ]:





# In[ ]:


#app_utils_clf

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn

#from few_shot.models import get_few_shot_encoder, Flatten
#from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
#from few_shot.proto import proto_net_episode
#from few_shot.train import fit
#from few_shot.callbacks import *

#from dlcliche.utils import *

assert torch.cuda.is_available()
device = torch.device('cuda')


def show_normalized_image(img, ax=None, mono=False):
    if mono:
        img.numpy()[..., np.newaxis]
    np_img = img.numpy().transpose(1, 2, 0)
    lifted = np_img - np.min(np_img)
    ranged = lifted / np.max(lifted)
    show_np_image(ranged, ax=ax)


class MonoTo3ChLayer(nn.Module):
    def __init__(self):
        super(MonoTo3ChLayer, self).__init__()
    def forward(self, x):
        x.unsqueeze_(1)
        return x.repeat(1, 3, 1, 1)


def _get_model(weight_file, device, model_fn, mono):
    base_model = model_fn(pretrained=True)
    feature_model = nn.Sequential(*list(base_model.children())[:-1],
                                  nn.AdaptiveAvgPool2d(1),
                                  Flatten())
    # Load initial weights
    if weight_file is not None:
        feature_model.load_state_dict(torch.load(weight_file))
    # Add mono image input layer at the bottom of feature model
    if mono:
        feature_model = nn.Sequential(MonoTo3ChLayer(), feature_model)
    if device is not None:
        feature_model.to(device)

    feature_model.eval()
    return feature_model


def get_resnet101(weight_file=None, device=None, mono=False):
    return _get_model(weight_file, device, models.resnet101, mono=mono)


def get_resnet50(weight_file=None, device=None, mono=False):
    return _get_model(weight_file, device, models.resnet50, mono=mono)


def get_resnet34(weight_file=None, device=None, mono=False):
    return _get_model(weight_file, device, models.resnet34, mono=mono)


def get_resnet18(weight_file=None, device=None, mono=False):
    return _get_model(weight_file, device, models.resnet18, mono=mono)


def get_densenet121(weight_file=None, device=None, mono=False):
    return _get_model(weight_file, device, models.densenet121, mono=mono)


def train_proto_net(args, model, device, n_epochs,
                    background_taskloader,
                    evaluation_taskloader,
                    path='.',
                    lr=3e-3,
                    drop_lr_every=100,
                    evaluation_episodes=100,
                    episodes_per_epoch=100,
                   ):
    # Prepare model
    model.to(device, dtype=torch.float)
    model.train(True)

    # Prepare training etc.
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.NLLLoss().cuda()
    #ensure_folder(path)
    #ensure_folder(path)

    def lr_schedule(epoch, lr):
        if epoch % drop_lr_every == 0:
            return lr / 2
        else:
            return lr

    callbacks = [
        EvaluateFewShot(
            eval_fn=proto_net_episode,
            num_tasks=evaluation_episodes,
            n_shot=args.n_test,
            k_way=args.k_test,
            q_queries=args.q_test,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
            distance=args.distance
        ),
        ModelCheckpoint(
            filepath=path + '/models/'+args.param_str+'_e{epoch:02d}.pth',
            monitor=args.checkpoint_monitor or f'val_{args.n_test}-shot_{args.k_test}-way_acc',
            period=args.checkpoint_period or 100,
        ),
        LearningRateScheduler(schedule=lr_schedule),
        CSVLogger(path +'train.csv'),
    ]

    fit(
        model,
        optimizer,
        loss_fn,
        epochs=n_epochs,
        dataloader=background_taskloader,
        prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        epoch_metrics=[f'val_{args.n_test}-shot_{args.k_test}-way_acc'],
        fit_function=proto_net_episode,
        fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                             'distance': args.distance},
)


# In[ ]:


#whale.whale_utils

#from dlcliche.image import *
#from dlcliche.math import *
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d
from IPython.display import display
import glob
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import albumentations as A

#sys.path.append('..') # app
#sys.path.append('../..') # root
#from few_shot.extmodel_proto_net_clf import ExtModelProtoNetClf
#from app_utils_clf import *


def _get_test_images(data_test):
    return [str(f).replace(data_test+'/', '') for f in glob.glob(data_test+'/*.jpg')]


def get_aug(re_size=224, to_size=224, train=True):
    augs = [A.Resize(height=re_size, width=re_size)]
    if train:
        augs.extend([
            A.RandomCrop(height=to_size, width=to_size),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=30, p=0.75),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.75),
            A.Blur(p=0.5),
            A.Cutout(max_h_size=to_size//12, max_w_size=to_size//12, p=0.5),
        ])
    else:
        augs.extend([A.CenterCrop(height=to_size, width=to_size)])
    return A.Compose(augs + [A.Normalize()])


def get_img_loader(folder, to_gray=False):
    def _loader(filename):
        img = cv2.imread(folder + '/' + str(filename))
        if to_gray:
            img = np.mean(img, axis=-1).astype(np.uint8)
            img = np.stack((img,)*3, axis=-1)
        return img
    return _loader


class WhaleImages(Dataset):
    def __init__(self, path, images, labels, re_size=256, to_size=224, train=True):
        self.datasetid_to_filepath = images
        self.datasetid_to_class_id = labels
        self.classes = sorted(list(set(labels)))
        
        self.df = pd.DataFrame({'class_id':labels, 'id':list(range(len(images)))})

        self.loader = get_img_loader(path, to_gray=True)
        self.transform = get_aug(re_size=re_size, to_size=to_size, train=train)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, item):
        instance = self.loader(self.datasetid_to_filepath[item])
        instance = self.transform(image=instance)['image']
        instance = self.to_tensor(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.cls2imgs)


def plot_prototype_2d_space_distribution(prototypes):
    X = prototypes
    pca = PCA(n_components=2)
    X_pca = pca.fit(X).transform(X)
    print('PCA: Explained variance ratio: %s'
          % str(pca.explained_variance_ratio_))
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=.6)
    plt.title('Prototype Distribution PCA')
    plt.xlim((-4, 4))
    plt.ylim((-3, 3))
    plt.show()
    return X_pca


def plot_prototype_3d_space_distribution(prototypes):
    X = prototypes
    pca = PCA(n_components=3)
    X_pca = pca.fit(X).transform(X)
    print('PCA: Explained variance ratio: %s'
          % str(pca.explained_variance_ratio_))
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(X_pca[:, 0],X_pca[:, 1],X_pca[:, 2])
    ax.set_title('Prototype Distribution PCA')
    ax.set_xlim((-4, 4))
    ax.set_ylim((-3, 3))
    ax.set_zlim((-3, 3))
    plt.show()
    return X_pca


def get_classes(data='data', except_new_whale=True, append_new_whale_last=True):
    df = pd.read_csv(data+'/train.csv')
    if except_new_whale:
        df = df[df.Id != 'new_whale']
    classes = sorted(list(set(df.Id.values)))
    if append_new_whale_last:
        classes.append('new_whale')
    return classes


def calculate_results(weight, SZ, get_model_fn, device, train_csv='../input/train.csv',
                      data_train='../input/train', data_test='../input/test'):
    # Training samples
    df = pd.read_csv(train_csv)
    df = df[df.Id != 'new_whale']
    images = df.Image.values
    labels = df.Id.values

    # Test samples
    test_images = _get_test_images(data_test)
    dummy_test_gts = list(range(len(test_images)))

    print(f'Training samples: {len(images)}, # of labels: {len(list(set(labels)))}.')
    print(f'Test samples: {len(test_images)}.')
    print(f'Work in progress for {weight}...')

    def get_dl(images, labels, folder, SZ=SZ, batch_size=64):
        ds = WhaleImages(folder, images, labels, re_size=SZ, to_size=SZ, train=False)
        dl = DataLoader(ds, batch_size=batch_size)
        return dl

    # Make prototypes
    trn_dl = get_dl(images, labels, data_train)
    model = get_model_fn(device=device, weight_file=weight+'.pth')
    proto_net = ExtModelProtoNetClf(model, trn_dl.dataset.classes, device)

    proto_net.make_prototypes(trn_dl)

    # Calculate distances
    test_dl = get_dl(test_images, dummy_test_gts, data_test)
    test_embs, gts = proto_net.get_embeddings(test_dl)
    test_dists = proto_net.predict_embeddings(test_embs, softmax=False)

    np.save(f'test_dists_{weight}.npy', test_dists)
    np.save(f'prototypes_{weight}.npy', np.array([x.mean() for x in proto_net.prototypes]))


# Thanks to https://github.com/radekosmulski/whale/blob/master/utils.py
def top_5_preds(preds): return np.argsort(preds.numpy())[:, ::-1][:, :5]

def top_5_pred_labels(preds, classes):
    top_5 = top_5_preds(preds)
    labels = []
    for i in range(top_5.shape[0]):
        labels.append(' '.join([classes[idx] for idx in top_5[i]]))
    return labels


def prepare_submission(submission_filename, test_dists, new_whale_thresh, data_test, classes):
    def _create_proto_submission(preds, name, classes):
        sub = pd.DataFrame({'Image': _get_test_images(data_test)})
        sub['Id'] = [classes[i] if not isinstance(i, str) else i for i in 
                     top_5_pred_labels(torch.tensor(preds), classes)]
        ensure_folder('subs')
        sub.to_csv(f'subs/{name}.csv.gz', index=False, compression='gzip')

    dist_new_whale = np.ones_like(test_dists[:, :1])
    dist_new_whale[:] = new_whale_thresh
    final_answer = np.c_[test_dists, dist_new_whale]

    _create_proto_submission(final_answer, submission_filename, classes)
    print(submission_filename,pd.read_csv(f'subs/{submission_filename}.csv.gz').Id.str.split().apply(lambda x: x[0] == 'new_whale').mean(),len(set(pd.read_csv(f'subs/{submission_filename}.csv.gz').Id.str.split().apply(lambda x: x[0]).values)))
    display(pd.read_csv(f'subs/{submission_filename}.csv.gz').head())


# In[ ]:





# In[ ]:


#whale.train

#from dlcliche.image import *
#sys.path.append('..') # app
#sys.path.append('../..') # root
from easydict import EasyDict
#from app_utils_clf import *
#from whale_utils import *
#from config import DATA_PATH

# Basic training parameters
args = EasyDict()
args.distance = 'l2'
args.n_train = 1
args.n_test = 1
args.q_train = 1
args.q_test = 1

args.k_train = 50
args.k_test = 10
SZ = 224
RE_SZ = 256

args.n_epochs = 1
args.drop_lr_every = 50
args.lr = 3e-3
args.init_weight = None

data_train = DATA_PATH+'/train'
data_test  = DATA_PATH+'/test'

args.param_str = f'app_whale_n{args.n_train}_k{args.k_train}_q{args.q_train}'
args.checkpoint_monitor = 'categorical_accuracy'
args.checkpoint_period = 50

print(f'Training {args.param_str}.')

# Data
df = pd.read_csv(DATA_PATH+'train.csv')
df = df[df.Id != 'new_whale']
ids = df.Id.values
classes = sorted(list(set(ids)))
images = df.Image.values
all_cls2imgs = {cls:images[ids == cls] for cls in classes}

trn_images = [image for image, _id in zip(images, ids) if len(all_cls2imgs[_id]) >= 2]
trn_labels = [_id   for image, _id in zip(images, ids) if len(all_cls2imgs[_id]) >= 2]
val_images = [image for image, _id in zip(images, ids) if len(all_cls2imgs[_id]) == 2]
val_labels = [_id   for image, _id in zip(images, ids) if len(all_cls2imgs[_id]) == 2]

args.episodes_per_epoch = len(trn_images) // args.k_train + 1
args.evaluation_episodes = 100 # setting small value, anyway validation set is almost useless here

print(f'Samples = {len(trn_images)}, {len(val_images)}')

# Model
feature_model = get_resnet18(device=device, weight_file=args.init_weight)

# Dataloader
background = WhaleImages(data_train, trn_images, trn_labels, re_size=RE_SZ, to_size=SZ)
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, args.episodes_per_epoch, args.n_train, args.k_train, args.q_train)
)
evaluation = WhaleImages(data_train, val_images, val_labels, re_size=RE_SZ, to_size=SZ, train=False)
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, args.episodes_per_epoch, args.n_test, args.k_test, args.q_test) 
)

# Train
train_proto_net(args,
                model=feature_model,
                device=device,
                path='.',
                n_epochs=args.n_epochs,
                background_taskloader=background_taskloader,
                evaluation_taskloader=evaluation_taskloader,
                drop_lr_every=args.drop_lr_every,
                evaluation_episodes=args.evaluation_episodes,
                episodes_per_epoch=args.episodes_per_epoch,
                lr=args.lr,
               )
torch.save(feature_model.state_dict(), f'{args.param_str}_epoch{args.n_epochs}.pth')


# In[ ]:


os.listdir('../input/')


# In[ ]:


from easydict import EasyDict
#from app_utils_clf import *
#from whale_utils import *
#from config import DATA_PATH

weight = f'{args.param_str}_epoch{args.n_epochs}'
calculate_results(weight=weight, SZ=224, get_model_fn=get_resnet18, device=device,train_csv='../input/humpback-whale-identification/train.csv', data_train='../input/humpback-whale-identification/train', data_test='../input/humpback-whale-identification/test')


# In[ ]:


test_dists = np.load(f'test_dists_{weight}.npy')
np_describe(test_dists)
prepare_submission(weight, test_dists, data_test='../input/test',classes=get_classes(data=DATA_PATH), new_whale_thresh=-1.85)

