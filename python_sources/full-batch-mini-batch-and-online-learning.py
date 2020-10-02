#!/usr/bin/env python
# coding: utf-8

# # Full batch, mini-batch, and online learning
# 
# ## Definitions
# 
# One important optimization to make early into the process of building a neural network is selecting an appropriate batch size.
# 
# Neural networks are trained in a series of **epochs**. Each epoch consists of one forward pass and one backpropogation pass over all of the provided training samples. Naively, we can compute the true gradient by computing the gradient value of each training case independently, then summing together the resultant vectors. This is known as **full batch learning**, and it provides an exact answer to the question of which stepping direction is optimal, as far as gradient descent is concerned.
# 
# Alternatively, we may choose to update the training weights several times over the course of a single epoch. In this case, we are no longer computing the true gradient; instead we are computing an *approximation* of the true gradient, using however many training samples are included in each split of the epoch. This is known as **mini-batch learning**.
# 
# In the most extreme case we may choose to adjust the gradient after every single forward and backwards pass. This is known as **online learning**.
# 
# The amount of data included in each sub-epoch weight change is known as the **batch size**. For example, with a training dataset of 1000 samples, a full batch size would be 1000, a mini-batch size would be 500 or 200 or 100, and an online batch size would be just 1.
# 
# ## Tradeoffs
# 
# There are many tradeoffs implicit in the choice of batch size.
# 
# Full batch learn is simpler to reason about, as we know that every step the learner makes is precisely aligned with the true gradient. In this sense it is less random than mini-batch or online learning, both of which take steps that are dependent on the randomness of the batch selection process. As a result full batch learning will always want to take nice smooth steps towards the globally optimal decision point; an attractive property, to be sure. However, to be efficient full batch learning requires that the entire training dataset be retained in memory, and hence it hits the scaling ceiling very very quickly.
# 
# Mini-batch learning is more exposed to randomness in the dataset and in the choice of the batch size, resulting in weight steps that look significantly more random than full batch steps. The smaller the batch size, the greater the randomness. On the other hand, assuming an appropriate batch size is chosen, they train much more quickly than full batch learners. Full batch learners must perform the full dataset scan for every single weight update. Mini-batch learners get to perform that same weight update multiple times per dataset scan. Assuming you choose an representative batch size this results in multiplicatively faster training.
# 
# Online learners are the most random of all. Because the steps they take are all over the place, they're significantly harder to debug. For this reason they are not usually used in static applications. However they are useful for applications that perform machine learning a runtime, as they eliminate the need for an expensive batch recomputation on reaching arbitrary input volume thresholds.
# 
# ## Best practices
# 
# In practice, most practitioners use mini-batch learning as their go-to. Full batch learning is generally reserved for very small datasets, and online learning is primarily the domain of sophisticated production data systems that live in production. The process of determining the correct batch size is left to experimentation, as it is highly data-dependent. The choice of batch size does not make a strong difference between different model architecture (or rather, it does not have nearly as strong an effect as many other models optimizations you should examine first). So to determine your batch size, it's reasonable to build a toy model, try a few different batch sizes, pick the one that seems to converge the fastest, and proceed from there. Even if you change the model, the batch size you chose earlier on will retain being a good choice.
# 
# Batch sizes that are multiples of powers of 2 are common. E.g. 32, 64, 128, and so on. I'm not sure whether this is just a stylistic point, or if there is actually any optimization going on around being a binary power.
# 
# Batch computations are heavily vectorized in their implementation, so the difference in processing speed between a 32 batch and a 64 batch is less than double, as you would naively assume. Smaller mini-batches are less efficient than larger ones in gradient calculation terms, but can make up for it with faster model convergence speeds, which may necessitate fewer epochs total. Online learning is least efficient of all, as it features zero vectorization, and is exponentially slower than online learning (on the toy dataset we will see shortly, 1000 observations with 3 output classes and 100 epochs take less than a second to train full-batch, and one point five minutes to train online).
# 
# On the other hand, there is evidence in practice that "large" batches tend to converge to minima with poorer generalization characteristics. This is because larger batches are more likely to converge to so-called "sharp" minima, e.g. sink values that are reasonably good, but do not provide the best problem solutions, but which have steep sides, and thus the learners are less able to escape from. Smaller batches are more likely to converge to "flat" minima. They are more able to escape these sinks, if need be. [Reference here](https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network).
# 
# Furthermore, when you put $m$ more examples into a mini-batch, you reduce the uncertaincy in the gradient by a factor of only $O(\sqrt{m})$ or so ([source](https://www.quora.com/In-deep-learning-why-dont-we-use-the-whole-training-set-to-compute-the-gradient)).
# 
# Another factor to consider when selecting a batch size is the learning rate. The interplay between the learning rate and the batch size is subtle, but important. A larger learning rate will compensate for a reliable slow-learning gradient, and a smaller learning rate will compensate for a more random fast-learning gradient. This visualization, taken from [this blog post on the subject](https://miguel-data-sc.github.io/2017-11-05-first/), shows a comparison between larger and smaller batch sizes, the learning rate, and the error rate the model ultimately converges to:
# 
# ![](https://i.imgur.com/uuMRabv.png)
# 
# For every model there is always a tuple of batch size and learning rate which forms a minimum amongst the curves in this plot. To find that minimum, you need to perform some kind of [hyperparameter search](https://www.kaggle.com/residentmario/gaming-cross-validation-and-hyperparameter-search/) against this two-dimensional space. This is itself a simple optimization problem; but since finding a new point in this space requires training an entire model from start to finish, it's a very expensive search space. If your model trains quickly enough, you may get away with selecting the best-performing model amongst a cross of $\text{learning_rate} \times \text{batch_size}$ settings. The [fastai](https://github.com/fastai/fastai) course seems to have [made some waves](https://www.kdnuggets.com/2017/11/estimating-optimal-learning-rate-deep-neural-network.html) by including a smarter optimizer called `lr_optimizer` which performs this parameter search for you automatically, using a more efficient search technique.

# ## Demonstration
# 
# Having discussed the characteristics of batch learning in general, let's look at a specific example demonstrating how large an impact the choice has on convergence speed.
# 
# The synthetic dataset that follows features three predictor variables and three target classes. The target classes are mutually exclusive, and correspond to the simple maximum of the three predictor variables. So for example the observation `[0.1, 0.2, 0.5]` will be classed `3`. `[0.9, 0.12, 0.5]` will be classed `1`. and so on.

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
import pandas as pd

X_train = np.random.random((1000, 3))
y_train = pd.get_dummies(np.argmax(X_train[:, :3], axis=1)).values
X_test = np.random.random((100, 3))
y_test = pd.get_dummies(np.argmax(X_test[:, :3], axis=1)).values


# The `keras` library features a `callbacks` parameter, available at `fit` time, which can be invoked at specific checkpoints during the model learning process. This callback can be used to retain historical information about the model being trained. In our case, I'm using a `LambdaCallback` to retain historical weight information from the first layer only.
# 
# The model itself is a relatively standard classifier model, with two nine-node `relu` activation layers and an output `softmax`.

# In[ ]:


# Reusable model fit wrapper.
def epocher(batch_size):
    # Create the print weights callback.
    from keras.callbacks import LambdaCallback
    history = []
    print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: history.append(clf.layers[0].get_weights()))

    # Build the model.
    clf = Sequential()
    clf.add(Dense(9, activation='relu', input_dim=3))
    clf.add(Dense(9, activation='relu'))
    clf.add(Dense(3, activation='softmax'))
    clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

    # Perform training.
    clf.fit(X_train, y_train, epochs=100, batch_size=batch_size, callbacks=[print_weights])
    
    # Return
    return history, clf


# Let's go ahead and train our first model, a full-batch model. There are 1000 training cases, so we set of a `batch_size` of 1000 in this case. Training is done over 100 epochs.

# In[ ]:


history, clf = epocher(batch_size=1000)


# We see that the model improves from ~33% accuracy to ~36% accuracy over the course of training.
# 
# Each call to `get_weights()` returns an array with the following structure:
# 
# ```python
# [[
#     [node_1_X_1_weight, node_1_X_2_weight, ..., node_1_X_9_weight],
#     [node_1_X_2_weight, node_2_X_2_weight, ...],
#     [node_1_X_3_weight, node_2_X_3_weight, ...]
# ],
#     [bias_node_1_weight, bias_node_2_weight, ..., bias_node_9_weight]
# ]
# ```
# 
# The first array contains the list of node weights used on each of the input variables. The second array contains the list of per-node biases, which are global across all variable observations. Thus this nine-node layer has $9 \times 3+ 9 = 9 \times 4 = 36\:$ weights in total!
# 
# To see what happens to the weights, let's pick a representative weight: the fist one on the first node will do. Here's what that weight does over the course of training:

# In[ ]:


import matplotlib.pyplot as plt

plt.plot([history[n][0][0][0] for n in range(100)], range(100))


# Notice that this is a smooth curve.
# 
# Now what if we use a very small batch. Say, a batch size of 10?

# In[ ]:


history, clf = epocher(batch_size=10)


# Notice how the resulting model is *vastly* more accurate than our first model. A batch size of 10 means one hundred gradient steps per epoch, instead of just one, so this model gets to adjust its gradient two orders of magnitude more times. Thus despite performing the same number of total round trips, our weights are ultimately much closer to the correct ones (though it's obviously almost certain that we've overfit in the process). We've done far more learning overall.
# 
# Now notice the path that our representative weight took in this training cycle:

# In[ ]:


plt.plot([history[n][0][0][0] for n in range(100)], range(100))


# This path is much more jagged and random and less smooth than that of the full-batch learner. It's still reasonable, however. Why? Because our toy dataset is very simple, we can expect the vast majority of the information in the model to get expressed even in the tiny ten-item batches we have chosen. If our dataset relationships were more complex and noisy, we could expect the performance of this small a batch to degrade, and as a consequence the "weight image" to be much more scattered.

# In[ ]:


# history, clf = epocher(batch_size=1)


# I omit an online learning algorithm, because it takes too long to train and I think you can see where this is going!
