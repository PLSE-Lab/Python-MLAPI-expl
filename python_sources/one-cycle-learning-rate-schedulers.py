#!/usr/bin/env python
# coding: utf-8

# # The 1cycle learning rate scheduler
# 
# ## Cyclic learning rates
# **Cyclic learning rates** (and cyclic momentum, which usually goes hand-in-hand) is a learning rate scheduling technique for (1) faster training of a network and (2) a finer understanding of the optimal learning rate. Cyclic learning rates have an effect on the model training process known somewhat fancifully as "superconvergence".
# 
# To apply cyclic learning rate and cyclic momentum to a run, begin by specifying a minimum and maximum learning rate and a minimum and maximum momentum. Over the course of a training run, the learning rate will be inversely scaled from its minimum to its maximum value and then back again, while the inverse will occur with the momentum. At the very end of training the learning rate will be reduced even further, an order of magnitude or two below the minimum learning rate, in order to squeeze out the last bit of convergence.
# 
# This was graphed in [the paper that introduced this idea](https://arxiv.org/abs/1803.09820) as follows:
# 
# ![](https://i.imgur.com/cs84ifb.png)
# 
# The maximum should be the value picked with a learning rate finder procedure, and the minimum value can be ten times lower.
# 
# ## One-cycle training
# This cyclic learning rate policy is meant to be applied over one entire learning cycle: e.g. one epoch. Fast.AI calls this the **one cycle training**. After each cycle, you are supposed to re-apply the learning rate finder to find new good values, and then fit another cycle, until no more training occurs; hence the name.
# 
# ## Why it works
# Optimally, learning rate and momentum should be set to a value which just causes the network to begin to diverge at its peak. The remainder of the training regimen performs learning rate annealing: the learning rate is gradually reduced, allowing the model to settle into the point of minimality in the cost surface. Momentum is counterproductive when the learning rate is very high, which is why momentum is annealed in the opposite of the way in which the learning rate is annealed in the optimizer.
# 
# Cyclic learning rates work well in practice because it combines the fastest possible learning rate for the macro discovery of the true minimum region on the cost surface with the micro discovery of the optimal point within that region as the learning rate is annealed.
# 
# Cyclic learning rate scheduling thus combines discovery of the maximum practical learning rate (given a certain batch size) with learning rate annealing. You can use cyclic learning to discover the maximum practical learning rate (too high and it will diverge, too low and it will not stabilize at the top), and then switch to adaptive learning rate scheduling (which is what most `keras` users use) with that learning rate, if you so desire.
# 
# ## Practical outcome
# Using the cyclic learning as presented in the paper doesn't result in a state-of-the-art model. However, with proper tuning, it achieves a near-state-of-the-art result in a fraction of the number of compute cycles. This makes it much more practical to train, and is what the author has termed "superconvergence".
# 
# ## Fast.AI modifications
# 
# The Fast.AI library uses a modified version of the cyclic learning rate presented in the original paper as its default learning rate scheduler when using the `fit_one_cycle` method. Instead of using linear descent for the second half of the learning curve, Fast.AI uses cosine descent:
# 
# ![](https://i.imgur.com/5PI4Om1.png)
# 
# ## Availability
# Besides Fast.AI, PyTorch also provides an implementation of this learning rate scheduler: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.OneCycleLR.
# 
# ## Comparison to other learning schedulers
# As best I can tell, there are currently four different learning rate schedulers that are commonly used:
# 
# * Simple annealing.
# * Adaptive learning rate scheduling, e.g. `ReduceLROnPlateau`, which is the default most people in the `keras` community opted for historically.
# * One-cycle learning rate scheduling.
# * Stochastic gradient descent with warm restarts (the subject of a future notebook).
# 
# The Fast.AI course uses one-cycle learning rate scheduling throughout the course, as it is considered the state of the art for training models to convergence expediently.
# 
# ## References
# * [callbacks.one_cycle](https://docs.fast.ai/callbacks.one_cycle.html)
# * ["The 1cycle policy"](https://sgugger.github.io/the-1cycle-policy.html)
# * ["Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"](https://towardsdatascience.com/https-medium-com-super-convergence-very-fast-training-of-neural-networks-using-large-learning-rates-decb689b9eb0)
