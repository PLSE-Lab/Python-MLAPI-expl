#!/usr/bin/env python
# coding: utf-8

# # QWK metric in PyTorch
# Since the metric is not implemented natively in PyTorch, I decided to implement it by myself. I followed the very good explanation by @reigHns in his [notebook](https://www.kaggle.com/reighns/understanding-the-quadratic-weighted-kappa) and realized it using only pure `torch` functions.
# 
# The function takes as input a `onehot` encoding of the target, while it can take a list of probabilities (apply `softmax` to the network output) or, in case of `binned` labels, the sum of the sigmoid logits from the network (apply `sigmoid` and `sum` to the network output).
# 
# ## Drawbacks of QWK metric (and loss)
# Since the coefficient is calculated with the outer product of the `outputs` and `targets` histograms, if there is no guess in any class (in any item of batch), a division by `0` occurs, leading the metric (and loss) to `nan`. It is clear that, the more the batch size, the less is the probability of such an event. Higher batch sizes (10, maybe) should be considered; in addition, the last batch of `Dataloader` should be dropped to avoid this problem (flag `drop_last` in the PyTorch `Dataloader` API). In the end, this metric is quite unstable for little batch sizes, so it should be used carefully.

# In[ ]:


import torch


# In[ ]:


def quadratic_kappa_coefficient(output, target):
    n_classes = target.shape[-1]
    weights = torch.arange(0, n_classes, dtype=torch.float32, device=output.device) / (n_classes - 1)
    weights = (weights - torch.unsqueeze(weights, -1)) ** 2

    C = (output.t() @ target).t()  # confusion matrix

    hist_true = torch.sum(target, dim=0).unsqueeze(-1)
    hist_pred = torch.sum(output, dim=0).unsqueeze(-1)

    E = hist_true @ hist_pred.t()  # Outer product of histograms
    E = E / C.sum() # Normalize to the sum of C.

    num = weights * C
    den = weights * E

    QWK = 1 - torch.sum(num) / torch.sum(den)
    return QWK


# ## Use example from notebook
# Referring to the aforementioned notebook, each tensor is less by 1 because in that case there were 5 classes and it started from 1. To use the `torch` functions we must start from 0.
# 
# In this case, there are 5 classes and 10 items.  (i.e., batch size = 10).

# In[ ]:


target = torch.tensor([2,2,2,3,4,5,5,5,5,5]) - 1
output = torch.tensor([2,2,2,3,2,1,1,1,1,3]) - 1

output.shape, target.shape


# Transform both arrays to `onehot` encoding.

# In[ ]:


import torch.nn.functional as F

target_onehot = F.one_hot(target, 5)
output_onehot = F.one_hot(output, 5)

output_onehot.shape, target_onehot.shape


# Show same result from the aforementioned notebook. I have already checked the values and dimensions of the `C`, `E` and `weights` matrices, but feel free to check by yourself.

# In[ ]:


quadratic_kappa_coefficient(output_onehot.type(torch.float32), target_onehot.type(torch.float32))


# ## Implementation of loss function
# This method can be easily integrated inside a `torch.nn.Module` to build, for example, the correlated loss.

# In[ ]:


def quadratic_kappa_loss(output, target, scale=2.0):
    QWK = quadratic_kappa_coefficient(output, target)
    loss = -torch.log(torch.sigmoid(scale * QWK))
    return loss

class QWKLoss(torch.nn.Module):
    def __init__(self, scale=2.0):
        super().__init__()
        self.scale = scale

    def forward(self, output, target):
        # Keep trace of output dtype for half precision training
        target = F.one_hot(target.squeeze(), num_classes=6).to(target.device).type(output.dtype)
        output = torch.softmax(output, dim=1)
        return quadratic_kappa_loss(output, target, self.scale)


# ## Implementation of metric function
# Moreover, it can be useful to define the metric function, even with binned labels for network training.

# In[ ]:


class QWKMetric(torch.nn.Module):
    def __init__(self, binned=False):
        super().__init__()
        self.binned = binned

    def forward(self, output, target):
        # Keep trace of dtype for half precision training
        dtype = output.dtype
        target = F.one_hot(target.squeeze(), num_classes=6).to(target.device).type(dtype)
        if self.binned:
            output = torch.sigmoid(output).sum(1).round().long()
            output = F.one_hot(output.squeeze(), num_classes=6).to(output.device).type(dtype)
        else:
            output = torch.softmax(output, dim=1)
        return quadratic_kappa_coefficient(output, target)


# In[ ]:


target = torch.randint(0, 6, (10, 1)).squeeze()
print("target: ", target)  # target class coming directly from the isup grades

output = torch.rand(10, 6)  # Logits from network, trained with not binned target
print("output: ", output)


# ## Binned metric and not binned loss examples

# In[ ]:


nb_loss = QWKLoss()
b_metric = QWKMetric(binned=True)
nbl = nb_loss(output, target)
bl = b_metric(output, target)
print("not binned loss: ", nbl.item())
print("binned metric: ", bl.item())


# In[ ]:




