#!/usr/bin/env python
# coding: utf-8

# # State of ML frameworks (late 2019)
# 
# ## Octave
# * Andrew Ng's first ML course was famously written in Octave, a GNU MATLAB sort-of-clone. Don't use Octave.
# 
# ## Theano
# * This was the research framework of choice for many years. First came around in 2007. Developed in Monteal by a team led by Yoshua Bengio.
# * Theano was sunsetted in 2017. An academic lab could not keep feature development parity with the engineering teams at the big tech companies.
# 
# ## Caffe
# * A similar research effort, written by a PhD student at UC Berkeley and maintained for years afterwards by a community of practitioners.
# * In 2017 a Caffe2 initiative was announced, sponsored by said PhD student's now-employer Facebook. This didn't get very far; Caffe2 was quote-unquote "merged" into PyTorch in 2018.
# * Caffe was sunsetted in 2017.
# 
# ## Keras
# * Keras has the best-designed API amongsts machine learning frameworks, as well as (from an early point in time in its development history) the explicit goal of being multi-framework.
# * E.g. allowing you to build models with a clean API that work on top of many different frameworks.
# * Keras was for a long time (and still is as of now) the preferred machine learning framework for models built and submitted to Kaggle&mdash;a good barometer of simplicity and ease of use. It isn't research-grade, but it's great for practitioners.
# * Keras started out based on Theano, but eventually switched to TensorFlow as its default backend. It also supported CNTK.
# * Theano went out of production in 2017. CNTK went out of production in 2019. This left Keras with only the TensorFlow backend.
# * At about the same time, TensorFlow announced that it was adopting the Keras API interally lock, stock, barrel.
# * As a result, as of October 2019, [Keras is being sunsetted](https://keras.io/). `tf.keras`, which has Google's  development investment, is eclipsing the Keras library itself, at the same time that Keras has lost its other backends.
# * It's unclear how quickly the ecosystem of tools built on top of Keras, e.g. `keras-preprocessing` and the like, will adopt to `tf.keras`...or if they will at all...
# 
# ## TensorFlow
# * The dominant industry framework at the moment, but quickly being eclipsed by PyTorch in research.
# * Hurt by the fact that it's changed APIs so many times; though the Keras API is here to stay now.
# * Historically used a static execution graph, which was "un-Pythonic" but had the advantages of portability and optimizability. Now has an eager mode as well, which is dynamic, and competes with PyTorch.
# * This is Google's effort.
# 
# ## PyTorch
# * The dominant research framework at the moment, but nowhere near the level of productionization tooling that TensorFlow has.
# * Historically featured eager execution. Recently added static execution (TensorFlow's old mode).
# * This is Facebook's effort.
# 
# ## CNTK
# * Microsoft's framework.
# * Didn't find industry traction. Sunsetted in 2019, as Microsoft decided that it was going to put its resources towards MXNet instead.
# 
# ## Chainer
# * A framework developed by a small-ish Japanese company which has a small following.
# * Still actively under development, but unlikely to win much additional market share.
# 
# ## MXNet
# * Apache project.
# * While it didn't start out that way, Amazon (which never developed any ML language with any significant market share) and Microsoft (which developed and failed to market-adopt CNTK) found themselves "out in the cold" in terms of not owning any of the major frameworks (Facebook has PyTorch, Google has Tensorflow). Not wanting to allow a competitor to own their platforms, after a bit of scramble, they co-adopted Apache MXNet as their preferred framework, and sunsetted various internal efforts (most notably, CNTK) that were not MXNet.
# * MXNet is considered the "preferred framework" on Amazon AWS and Microsoft Azure.
# 
# ## Gluon
# * A high-level API out of Amazon that used to support CNTK and MXNet. Now supports MXNet only. Represents an attempt to "Keras-ify" MXNet.
# * Uncertain development and adoption status.
# 
# ## TLDR
# * Mainline Keras is going to sunset soon due to shifts in industry that have deprecated its backends. That sucks.
# * The choice of framework right now boils down to TensorFlow or PyTorch (for now). Maybe one day MXNet will have significant market share and join the party.
# * PyTorch is probably the better of the two frameworks to back up to. [Reasons why covered here](https://www.reddit.com/r/MachineLearning/comments/b6wgmo/d_tensorflow_is_dead_long_live_tensorflow/ejokfgq?utm_source=share&utm_medium=web2x). But you might have to cross-train.
