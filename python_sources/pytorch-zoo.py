# The code for this is from my pip package: (https://pypi.org/project/pytorch-zoo/)

## Imports

import os
import random
import pickle
import requests

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Data classes

class DynamicSampler(data.BatchSampler):
    """A dynamic batch length data sampler. To be used with `trim_tensors`.

    Implementation adapted from https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/94779 and https://github.com/pytorch/pytorch/blob/master/torch/utils/data/sampler.py

    Args:
        sampler (torch.utils.data.Sampler): Base sampler.
        batch_size (int): Size of minibatch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if its size would be less than ``batch_size``.
    """
    def __iter__(self):

        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            count_zeros = torch.sum(self.sampler.data_source[idx][0] == 0)
            count_zeros = int(count_zeros / 64) 
            if len(buckets[count_zeros]) == 0:  buckets[count_zeros] = []

            buckets[count_zeros].append(idx)

            if len(buckets[count_zeros]) == self.batch_size:
                batch = list(buckets[count_zeros])
                yield batch
                yielded += 1
                buckets[count_zeros] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

        assert len(self) == yielded, "produced an inccorect number of batches. expected %i, but yielded %i" %(len(self), yielded)

def trim_tensors(tensors):
    """Trim padding off of a batch of tensors to the smallest possible length. To be used with `DynamicSampler`.

    Implementation adapted from https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/94779

    Args:
        tensors ([torch.tensor]): list of tensors to trim.

    Returns:
        ([torch.tensor]): list of trimmed tensors. 
    """

    max_len = torch.max(torch.sum( (tensors[0] != 0  ), 1))
    if max_len > 2: 
        tensors = [tsr[:, :max_len] for tsr in tensors]
    return tensors

## Losses

def _mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def _lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs.requires_grad_()
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.elu(errors_sorted) + 1, grad.requires_grad_())
    return loss


def _flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_hinge(logits, labels, per_image=True):
    """The binary Lovasz Hinge loss for semantic segmentation.

    Implementation adapted from https://github.com/bermanmaxim/LovaszSoftmax
    
    Args:
        logits (torch.tensor): Logits at each pixel (between -\infty and +\infty).
        labels (torch.tensor): Binary ground truth masks (0 or 1).
        per_image (bool, optional): Compute the loss per image instead of per batch.
                                    Defaults to True.

    Shape:
        - Input:
            - logits: (batch, height, width)
            - labels: (batch, height, width)
        - Output: (batch)

    Returns:
        torch.tensor: The lovasz hinge loss
    """
    if per_image:
        loss = _mean(
            _lovasz_hinge_flat(
                *_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), None)
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, None))
    return loss

class DiceLoss(nn.Module):
    """The dice loss for semantic segmentation

    Implementation adapted from https://www.kaggle.com/soulmachine/siim-deeplabv3

    Shape:
        - Input:
            - logits: (batch, *)
            - targets: (batch, *) _same shape as logits_
        - Output: (1)

    Returns:
        torch.tensor: The dice loss
        
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets):
        N = targets.size(0)
        preds = torch.sigmoid(logits)

        EPSILON = 1

        preds_flat = preds.view(N, -1)
        targets_flat = targets.view(N, -1)

        intersection = (preds_flat * targets_flat).sum()
        union = (preds_flat + targets_flat).sum()

        loss = (2.0 * intersection + EPSILON) / (union + EPSILON)
        loss = 1 - loss / N

        return loss
    
## Modules

class SqueezeAndExcitation(nn.Module):
    """The channel-wise SE (Squeeze and Excitation) block from the [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) paper.

    Implementation adapted from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939 and https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    
    Args:
        in_ch (int): The number of channels in the feature map of the input.
        r (int): The reduction ratio of the intermidiate channels.
                Default: 16.

    Shape:
        - Input: (batch, channels, height, width)
        - Output: (batch, channels, height, width) (same shape as input)
    """

    def __init__(self, in_ch, r=16):
        super(SqueezeAndExcitation, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        input_x = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)

        return x


class ChannelSqueezeAndSpatialExcitation(nn.Module):
    """The sSE (Channel Squeeze and Spatial Excitation) block from the 
    [Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579) paper.

    Implementation adapted from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    
    Args:
        in_ch (int): The number of channels in the feature map of the input.

    Shape:
        - Input: (batch, channels, height, width)
        - Output: (batch, channels, height, width) (same shape as input)
    """

    def __init__(self, in_ch):
        super(ChannelSqueezeAndSpatialExcitation, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x

        x = self.conv(x)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)

        return x


class ConcurrentSpatialAndChannelSqueezeAndChannelExcitation(nn.Module):
    """The scSE (Concurrent Spatial and Channel Squeeze and Channel Excitation) block from the 
    [Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579) paper.
    
    Implementation adapted from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178

    Args:
        in_ch (int): The number of channels in the feature map of the input.
        r (int): The reduction ratio of the intermidiate channels.
                Default: 16.

    Shape:
        - Input: (batch, channels, height, width)
        - Output: (batch, channels, height, width) (same shape as input)
    """

    def __init__(self, in_ch, r):
        super(ConcurrentSpatialAndChannelSqueezeAndChannelExcitation, self).__init__()

        self.SqueezeAndExcitation = SqueezeAndExcitation(in_ch, r)
        self.ChannelSqueezeAndSpatialExcitation = ChannelSqueezeAndSpatialExcitation(
            in_ch
        )

    def forward(self, x):
        cse = self.SqueezeAndExcitation(x)
        sse = self.ChannelSqueezeAndSpatialExcitation(x)

        x = torch.add(cse, sse)

        return x


class GaussianNoise(nn.Module):
    """A gaussian noise module.

    Args:
        stddev (float): The standard deviation of the normal distribution.
                        Default: 0.1.

    Shape:
        - Input: (batch, *)
        - Output: (batch, *) (same shape as input)
    """

    def __init__(self, stddev=0.1):
        super(GaussianNoise, self).__init__()

        self.stddev = stddev

    def forward(self, x):
        noise = torch.empty_like(x)
        noise.normal_(0, self.stddev)

        return x + noise

## Schedulers

class CyclicMomentum(object):
    """
    Cyclical Momentum

    Pytorch's [cyclical learning rates](https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py), but for momentum, which leads to better results when used with cyclic learning rates, as shown in [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820).
    
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
        base_momentum (float or list): Initial momentum which is the
            lower boundary in the cycle for each param groups.
            Default: 0.8
        max_momentum (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the sum of base_momentum
            and some scaling of the amplitude; therefore
            max_momentum may not actually be reached depending on
            scaling function. Default: 0.9
        step_size (int): Number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch. Default: 2000
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
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_iteration (int): The index of the last batch. Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicMomentum(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.batch_step()
        >>>         train_batch(...)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(
        self,
        optimizer,
        base_momentum=0.8,
        max_momentum=0.9,
        step_size=2000,
        mode="triangular",
        gamma=1.0,
        scale_fn=None,
        scale_mode="cycle",
        last_batch_iteration=-1,
    ):

        self.optimizer = optimizer

        if isinstance(base_momentum, list) or isinstance(base_momentum, tuple):
            self.base_momentums = list(base_momentum)
        else:
            self.base_momentums = [base_momentum] * len(optimizer.param_groups)

        if isinstance(max_momentum, list) or isinstance(max_momentum, tuple):
            self.max_momentums = list(max_momentum)
        else:
            self.max_momentums = [max_momentum] * len(optimizer.param_groups)

        self.step_size = step_size

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == "triangular":
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = "cycle"
            elif self.mode == "triangular2":
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = "cycle"
            elif self.mode == "exp_range":
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = "iterations"
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1

        self.last_batch_iteration = batch_iteration

        # update momentum here
        for param_group, momentum in zip(
            self.optimizer.param_groups, self.get_momentum()
        ):
            param_group["momentum"] = momentum

    def _triangular_scale_fn(self, x):
        return 1.0

    def _triangular2_scale_fn(self, x):
        return 1 / (2.0 ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma ** (x)

    def get_momentum(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        momentums = []
        param_momentums = zip(
            self.optimizer.param_groups, self.base_momentums, self.max_momentums
        )

        for param_group, base_momentum, max_momentum in param_momentums:
            base_height = (max_momentum - base_momentum) * np.maximum(0, (x))

            if self.scale_mode == "cycle":
                momentum = base_momentum + base_height * self.scale_fn(cycle)
            else:
                momentum = base_momentum + base_height * self.scale_fn(
                    self.last_batch_iteration
                )
            momentums.append(momentum)

        return momentums

## Other utils

def notify(obj, key):
    """Send a notification to your phone with IFTTT

    Setup a IFTTT webhook with https://medium.com/datadriveninvestor/monitor-progress-of-your-training-remotely-f9404d71b720
    
    Args:
        obj (Object): Object to send to IFTTT
        key ([type]): IFTTT webhook key
    """
    requests.post(f"https://maker.ifttt.com/trigger/notify/with/key/{key}", data=obj)


def seed_environment(seed):
    """Set random seeds for python, numpy, and pytorch to ensure reproducible research.
    
    Args:
        seed (int): The random seed to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def gpu_usage(device=device, digits=4):
    """Prints the amount of GPU memory currently allocated in GB.
    
    Args:
        device (torch.device, optional): The device you want to check.
                                        Defaults to device.
        digits (int, optional): The number of digits of precision.
                                Defaults to 4.
    """
    print(
        f"GPU Usage: {round((torch.cuda.memory_allocated(device=device) / 1e9), digits)} GB\n"
    )


def n_params(model):
    """Return the number of parameters in a pytorch model.
    
    Args:
        model (nn.Module): The model to analyze.
    
    Returns:
        int: The number of parameters in the model.
    """
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def save_model(model, fold):
    """Save a trained pytorch model on a particular cross-validation fold to disk. 

    Implementation adapted from https://github.com/floydhub/save-and-resume.

    Args:
        model (nn.Module): The model to save.
        fold (int): The cross-validation fold the model was trained on.
    """
    filename = f"./checkpoint-{fold}.pt"
    torch.save(model.state_dict(), filename)


def load_model(model, fold):
    """Load a trained pytorch model saved to disk using `save_model`.
    
    Args:
        model (nn.Module): The model to save.
        fold (int): Which saved model fold to load.
    
    Returns:
        nn.Module: The same model that was passed in, but with the pretrained weights loaded.
    """
    model.load_state_dict(torch.load(f"./checkpoint-{fold}.pt"))

    return model


def save(obj, filename):
    """Save an object to disk.
    
    Args:
        obj (Object): The object to save.
        filename (String): The name of the file to save the object to.
    """
    with open(f"{filename}", "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(path):
    """Load an object saved to disk with `save`.
    
    Args:
        path (String): The path to the saved object.
    
    Returns:
        Object: The loaded object.
    """
    with open(path, "rb") as handle:
        obj = pickle.load(handle)

    return obj

def masked_softmax(vector, mask, dim=-1, memory_efficient=False, mask_fill_value=-1e32):
    """A masked softmax module to correctly implement attention in Pytorch.

    Implementation adapted from: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    
    Args:
        vector (torch.tensor): The tensor to softmax.
        mask (torch.tensor): The tensor to indicate which indices are to be masked and not included in the softmax operation.
        dim (int, optional): The dimension to softmax over.
                            Defaults to -1.
        memory_efficient (bool, optional): Whether to use a less precise, but more memory efficient implementation of masked softmax.
                                            Defaults to False.
        mask_fill_value ([type], optional): The value to fill masked values with if `memory_efficient` is `True`.
                                            Defaults to -1e32.
    
    Returns:
        torch.tensor: The masked softmaxed output
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


# From: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py#L276-L307
def masked_log_softmax(vector, mask, dim=-1):
    """A masked log-softmax module to correctly implement attention in Pytorch.

    Implementation adapted from: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.

    Args:
        vector (torch.tensor): The tensor to log-softmax.
        mask (torch.tensor): The tensor to indicate which indices are to be masked and not included in the log-softmax operation.
        dim (int, optional): The dimension to log-softmax over.
                            Defaults to -1.
    
    Returns:
        torch.tensor: The masked log-softmaxed output

    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)
