#!/usr/bin/env python
# coding: utf-8

# # Batch normalization and its successors
# * Batch normalization is a practitioner's technique introduced in 2015 that has been shown to radically increase the rate at which neural networks converge in almost every setting in which its been tried. Batch normalization (or one of its evolved alternatives) is now used in basically every SOTA (usually alongside another similar practitioner's technique, residual connections).
# * Note that batch normalization and the related ideas is just one potential form of neural network regularization: the one that has been found to be most widely applicable and most useful for SOTA results. There are many other techniques in this space. I have a notebook with an overview of some of the various techniques that have been tried: ["Forms of neural network regularization"](https://www.kaggle.com/residentmario/forms-of-neural-network-regularization).
# 
# 
# ## Batch norm
# * Batch normalization is (obviously) a form of normalization. The "batch" part comes from the fact that it is applied on a batch-by-batch basis.
# * Batch normalization scales each dimension of the input to a succeeding layer/output from a previous layer so that they have mean zero ($\mu = 0$) and unary variance ($\sigma^2 = 1$).
# * Each dimension of the input is normalized separately. For example, if the input is a vector $<x_1, x_2, x_3>$, three things will normalized: $x_1$, $x_2$, and $x_3$.
# * If batch normalization is performed prior to the first hidden layer of the model then it's just minibatch data normalization. When batch normalization is performed between hidden layers of the model, it's minibatch hidden vector normalization.
# * Batch normalization requires reasonably large batch sizes (because it is an estimator of the population mean and variance, and not the true measure, it is sensitive to sample size). It cannot be used for online learning (batch size of 1), but even batch sizes as high as 128 or more can be problematic if the input sees skewed-enough results.
# * At inference time, the population statistic for the entire dataset is used. This requires some bookkeeping during the network training phase. 
# * Because the normalization differs between train time and test time, the reproducibility of the model is harmed! Results generated during train time and results generated during test time will be different. The more layers you batch normalize, and the more layers the model has, the greater the potential divergence.
# * In terms of theory, there's been a lot of arguing about what the reason why batch norm works is. The paper that introduced it posited that it deals with "interval covariance shift", but this has since been debunked. No durable theoretically justification is currently known.
# 
# 
# ## Weight norm
# * Weight normalization is a different form of normalization. Instead of scaling the hidden layer inputs/outputs, we scale the weights themselves.
# * Instead of learning the weights directly, the network is redirected to learn a function of the $w$ vector that looks like so:
# 
#   $$w = \frac{g}{||v||}v$$
#   
#   Where $||v||$ is the unit norm of $v$ and $g$ is a scalar value. $v$ has the same dimensionality of $w$. Both $v$ and $g$ are learned parameters; in effect under this scheme we are learning one additional parameter as part of the weight, a scaling factor $g$.
# * Detaching vector direction from vector weight in this manner separates these two concerns during model training, and has been shown to act as a form of learnable controllable normalization in practice (even though there's no explicit take-to-zero-mean-and-unit-variance component).
# * Because weight normalization learns weight parameters, it does not have the train-test results divergence that batch normalization has, nor is it sensitive to minibatch size, or to the scale of hidden vector input.
# * Weight normalization came out of OpenAI.
# * One mathematically interesting property of weight normalization is that it reduces the rank of the matrix it is applied to.
# 
# 
# ## Layer norm
# * Layer normalization came of out of Geoffrey Hinton's lab.
# * Layer normalization is batch normalization along the feature dimension instead of along the batch dimension. In other words, instead of normalizing the features of one input by looking at the corresponding feature values in other samples, we normalize the fatures of one input by looking at the other features of the vector.
# * Layer normalization has the obvious advantage that, like weight normalization, it insensitive to batch size. However, my intuition is that it requires very scale-similar features in order to work well.
# * Layer normalization is most often used in RNNs.
# 
# 
# ## Instance norm
# * Instance normalization is a further specialization of layer normalization specific to the image processing setting (e.g. to CNNs).
# * Instance normalization is layer normalization applied across input channels. E.g. with an RBG image input this means that each pixel of the image would be normalized individually across the R, G, B channels; the same technique would continue to be applied to hidden layers deeper in the network.
# * Instance normalization primarily creates robustness to contrast.
# * This technique has found success in applications in style transfer tasks and in training GANs. It has *not* been found to outperform batch normalization in most other image processing tasks, however.
# 
# 
# ## Group normalization
# * Group normalization is instance normalization applied across groups of channels. As with instance normalization, it is specific to the image task context.
# * Group normalization has performance characteristics similar to batch normalization in those image processing tasks not appropriate for instance normalization (e.g. not style transfer).
# 
# ![](https://i.imgur.com/kc2OPjA.png)
# 
# ## Spectral normalization
# * Spectal normalization is a form of normalization specific to GANs, which is used in GuaGAN, hence why I am interested in covering it alongside other techniques here.
# * Spectal normalization creates the constraint (typically in the discriminator module) that the weights matrix exhibits **Lipschitz continuity**. This property essentially means that the gradient created by the weights matrix exhibits a certain (constant-controllabe!) level of functional smoothness in all directions. The Wikipedia article has a great visualization of what this looks like, using a double cone on a linear function [here](https://en.wikipedia.org/wiki/Lipschitz_continuity).
# 
#   This is achieved by normalizing the weights matrix by the matrix's largest eigenvalue. This value is equivalently the spectral norm (the norm with $p=\infty$; see the notebook ["L1 versus L2 norms"](https://www.kaggle.com/residentmario/l1-norms-versus-l2-norms)) of the matrix. See [this StatsOverflow Q&A](https://math.stackexchange.com/questions/188202/meaning-of-the-spectral-norm-of-a-matrix) for a bit of explanation.
# 
# ## Others
# * There are a lot of normalization techniques out there. Like, a lot a lot. There's a solid chance if you look at a SOTA model that it's using some fort of normalization unfamiliar to you, so you just kind of have to read about it a bit.
