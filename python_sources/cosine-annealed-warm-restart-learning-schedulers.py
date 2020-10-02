#!/usr/bin/env python
# coding: utf-8

# # Cosine annealed warm restart learning schedulers
# 
# In a previous notebook, [one-cycle learning rate schedulers](https://www.kaggle.com/residentmario/one-cycle-learning-rate-schedulers), I discussed the one-cycle learning rate scheduler, and briefly mentioned the cosine annealed warm restart learning schedule as one commonly-used alternative.
# 
# ## How it works
# The cosine annealed warm restart learning schedule has two parts, cosine annealing and warm restarts.
# 
# **Cosine annealing** means that the cosine function is used as the learning rate annealing function. The cosine function has been shown to perform better than alternatives like simple linear annealing in practice.
# 
# **Warm restarts** is the interesting part: it means that every so often, the learning rate is *restated*, e.g. reraised back up. The original proposal for warm restarts was triangular with height descent:
# 
# ![](https://i.imgur.com/7y0inDP.png)
# 
# Fast.AI uses cosine annealing instead of linear annealing (as per the section above), resets the learner height all the way back to the original learning rate, and adds a multiplicative term, set to 2 by default, which multiplicatively increases the period length of each consecutive restart. All of these features are available in PyTorch under the name "cosine annealing with warm restarts" ([here](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)), hence the name of this kernel.
# 
# ## Why it works
# Cosine annealing has an ideas core to a good learning rate scheduler nowadays: periods with high learning rates and periods with low ones. The function of the periods of high learning rates in the scheduler is to prevent the learner from getting stuck in a local cost minima; the function of the periods of low learning rates in the scheduler is to allow it to converge to a near-true-optimal point within the (hopefully) global minima it finds.
# 
# Cosine annealing with warm restarts was originally presented alongside stochastic gradient descent, and termed in combination as "stochastic gradient descent with warm restarts", or sometimes SGDR. However it's a more general technique that can be applied to optimizers besides SGD. A summary of the algorithm that comes recommended is this blog post: ["Exploring Stochastic Gradient Descent with Restarts (SGDR)"](https://medium.com/38th-street-studios/exploring-stochastic-gradient-descent-with-restarts-sgdr-fa206c38a74e). A good summary image that captures the essence of the argument for cosine annealing is as follows:
# 
# ![](https://i.imgur.com/nJFP4UE.png)
# 
# ## In practice
# This learning rate scheduler [is available in PyTorch](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts).
# 
# Fast.AI (which I take as a reference on what is state-of-the-art at this time) uses this learning rate scheduler by default for `fit`, but uses `fit_one_cycle` throughout its material. Before the introduction of one-cycle learning into the library, [this was the default learning rate scheduler](https://forums.fast.ai/t/understanding-cycle-len-and-cycle-mult/9413/10). In a recent tweet, Jeremey Howard posited that [he doesn't know of any application where cyclic learning rates beat the one-cycle learning scheduler](https://twitter.com/jeremyphoward/status/1158388577824198659).
