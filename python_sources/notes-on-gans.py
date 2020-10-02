#!/usr/bin/env python
# coding: utf-8

# # Notes on GANs
# 
# This kernel contains my high-level notes on GANs: what they are, how they are trained, and what their most common configurations are. These notes are largely drawn from Google's [Machine Learning Developer Materials on GANs](https://developers.google.com/machine-learning/gan), which are an extension of MLCC.
# 
# ## High-level overview
# GANs, or **generative adversial networks**, are one of the newest machine learning model archetypes, dating back to circa-2014 Ian Goodfellow. GANs work by pairing a *generative* network, trained to build compelling synthetic data, with a *discriminative* network, which is trained to detect the different between real input and fake input. These two networks compete in a game (commonly a zero-sum game), where each network is trying to beat the other.
# 
# GANs were originally proposed as a generative model for application in unsupervised learning.
# 
# Here **generative model** means that the model constructs an internal representation of the data space. Generative models are in opposition to discriminatory models, which learn to select an output given a set of inputs. Put another way, a generative model learns all of the joint distributions of the data ($P(X, y)$); a discrimatory model learns just their conditional probabilities ($P(y|X)$). A generative model can potentially be used to draw new samples of data that are probable; discriminatory models cannot. An example of a generative algorithm is kernel density estimation or naive Bayes; an example of a discriminatory model is regression or a decision tree.
# 
# In application to unsupervised learning, a GAN could be used to cluster data inputs, perform dimensionality reduction by piping the input representations to compressed output spaces (a la autoencoders), or to generate new synthetic data usable by a supervised learning algorithm.
# 
# Although they are most famous for generating photorealistic output, e.g. deepfakes, GANs are beginning to find applications in other forms of learning (convolutional and recurrant) as well.
# 
# One interesting application of GANs is to video game texture upscaling. Low-resolution images are scaled using a GAN, then downsampled back to their native resolution. This results in a texture map which is sharper than its original, in terms of anti-aliasing etecetera.
# 
# ## The objective function
# 
# The generator and discriminator networks are trained simultaneously, but using different loss functions and separate backpropogation regimens.
# 
# The Google Developers tutorial segment (an extension to MLCC) on GANs defines three common loss functions.
# 
# ---
# 
# The first is **minimax loss**. This is the loss function that was used in the original 2014 GAN paper. This loss function has the following formulation:
# 
# $$E_x[\log{D(x)}] + E_z[\log{(1 - D(G(z))}]$$
# 
# $D(x)$ is the discriminator's probability estimate for whether or not $x$ is real. $E_x[\log{D(x)}]$ is the expected value over the log-probabilities for all *real* data instances. $G(z)$ is the generator's output for input noise $z$. $D(G(z))$ is the discriminator's estimate for whether or not the generated output is real, hence $1 - D(G(z))$ is the discriminator's estimated correctness, and $E_z[\log{(1 - D(G(z))}]$ is the expectation over the log-probabilities of all *fake* data instances. These two components&mdash;the discriminator's score on real input and its score on fake input&mdash;are combined using simple addition.
# 
# Minimax loss is derived from cross-entropy, which itself is a generalization of the log-loss objective function used in logistic regression. The $\log{(\cdot)}$ formulation is convenient for reasons of numerical stability. It is called minimax because in game theory, given a zero-sum game, minimax loss has been shown to be the objective function which maximizes the minimum gain (or equivalently, and perhaps more clearly, the objective function which minimizes the expected losses).
# 
# The discriminator attempts to maximize this function (thus maximizing how often it sees through the generator) whilst the generator attempts to minimize this function (thus maximizing how often it fools the discriminator).
# 
# ---
# 
# The original GAN place notes that minimax loss can cause a GAN to plateau (stop learning early) is early-regimen training cases where the discriminator's job is very easy (question to self: why is this true?). The paper suggests using maximizing $\log{(D(G(z))}$, or minimizing $1 - \log{D(G(z))}$, as the generator loss as a workaround. This modified minimax loss is also implemented in e.g. TensorFlow.
# 
# ---
# 
# The last loss mentioned is **Wasserstein loss**. This is the loss that is used in TensorFlow by default. This kind of loss is also known as "earth mover's distance". Wasserstein loss does not constrain that the outputs of the models be valid probabilities (e.g. in the range $[0, 1]$); it instead simply wants output values to be larger for real inputs than for fake ones.
# 
# In Wasserstein loss the discriminator attempts to maximize $D(x) - D(G(z))$ whilst the generator attempts to maximize $D(G(z))$. So formulaicly, Wasserstein loss is minimax loss without the $\log{(\cdot)}$ or $E$ terms.
# 
# Wikipedia has the following intuition to share about Earth mover's distance ([link](https://en.wikipedia.org/wiki/Earth_mover%27s_distance)):
# 
# > Informally, if the distributions are interpreted as two different ways of piling up a certain amount of dirt over the region D, the EMD is the minimum cost of turning one pile into the other; where the cost is assumed to be amount of dirt moved times the distance by which it is moved.
# 
# You know, since distance is arbitrary.
# 
# Since Wasserstein loss doesn't output probabilities, the value 0.5 can't be used as a threshold for determining whether an image is real or not. If classification is ultimately your goal you will need to determine the optimal threshold value yourself.
# 
# Because the Wasserstein metric can flex to arbitrarily large or small values, it is less eager to plateau (the vanishing gradient problem) than minimax loss. On the other hand, it requires that model weights be clipped to certain boundary values, as the metric is also more vulnerable to performing overly large weight updates.
# 
# ## Training procedure
# 
# The discriminator is a convolutional neural network which is fed both real and fake images (output from the generator) at train time. It performs backpropogation on batches of such images.
# 
# The generator is a deconvolutional neural network which is fed random (or nonrandom, if you want to shape the input) noise at train time. It outputs its modeled data, then passes that modeled data to the discriminator, which generates predictions. This *entire sequence* is trained using backpropogation at once; e.g. the discriminator is trained in a manner including the generator.
# 
# ![](https://i.imgur.com/PocG3dJ.png)
# 
# According to my reading of the materials, there are a couple of ways of handling generator training. One is to freeze the discriminator layers. The other is to leave the discriminator layers unfrozen, but through away the discriminator component of the weight updates, e.g. only use the generator component weight updates.
# 
# Training the GAN as a whole requires training the two different networks in an interleved sequential manner. We train the discriminator for a few batches, then the generator, then the discriminator, and so on.
# 
# In practice, the performance of the generator near the end of training often outstrips that of the discriminator. Oftentimes the stable state that is reached is one in which the performance of the discriminator is very near that of a coin flip. This can make it difficult to determine whether or not convergence has been reached.
# 
# ## Common problems
# 
# GANs have a lot of training problems, largely stemming from the complex interplay between the generator and discriminator modules.
# 
# * If the discriminator advances faster than the generator, it induces vanishing gradients in the generator, causing training to plateau.
# * **Mode collapse**. This is a uniquely GAN problem, having everything to do with GAN architecture:
# 
#     > Usually you want your GAN to produce a wide variety of outputs. You want, for example, a different face for every random input to your face generator.
#     >
#     > However, if a generator produces an especially plausible output, the generator may learn to produce only that output. In fact, the generator is always trying to find the one output that seems most plausible to the discriminator.
#     > 
#     > If the generator starts producing the same output (or a small set of outputs) over and over again, the discriminator's best strategy is to learn to always reject that output. But if the next generation of discriminator gets stuck in a local minimum and doesn't find the best strategy, then it's too easy for the next generator iteration to find the most plausible output for the current discriminator.
#     > 
#     > Each iteration of generator over-optimizes for a particular discriminator, and the discriminator never manages to learn its way out of the trap. As a result the generators rotate through a small set of output types. This form of GAN failure is called mode collapse.
#     
#  So-called unrolled GANs are an attempt to ameliorate this problem. They do so by using a loss function incorporating both the current discriminator and discriminators from the future. "Unrolling" the discriminator...just like unrolling time-steps in a recurrant neural network. 
# * Failure to converge, due to differences in discriminator versus generator skill that cause the overall model to get stuck in a broad (non model collapse) local optimum. This can be solved with regularization: noise on the input to the discriminator, or the regularization of the discriminator model weights.
# 
# ## Worked example
# 
# The layer architecture actually used in a GAN is not particularly fancy; just a deconvolutional network (which is exactly what it sounds like) attached to a convolutional network.
# 
# For a worked example in raw TensorFlow see [Google's TF-GAN Colab notebook](https://colab.research.google.com/github/tensorflow/gan/blob/master/tensorflow_gan/examples/colab_notebooks/tfgan_tutorial.ipynb?utm_source=ss-gan&utm_campaign=colab-external&utm_medium=referral&utm_content=tfgan-intro#scrollTo=qxrYrU887Mns).
# 
# For a worked example in Keras see the following article: ["A brief introduction to GANs and how to code them"](https://medium.com/sigmoid/a-brief-introduction-to-gans-and-how-to-code-them-2620ee465c30).
# 
# Both of these articles use, you guessed it, MNIST.
