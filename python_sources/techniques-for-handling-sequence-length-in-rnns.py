#!/usr/bin/env python
# coding: utf-8

# # Techniques for handling sequence length in RNNs
# 
# RNNs, or recurrent neural networks, can perform tasks on word, sentence, paragraph, or even document length input.
# 
# However, this invites a couple of challenges involving the length of the input: (1) how to handle short and variable-length inputs (e.g. sentences of different lengths) and (2) what to do with very long inputs (which are computationally intensive).
# 
# ## Techniques for handling input sequences that are too short
# 
# The top rated answer to [an SO question on this subject](https://datascience.stackexchange.com/questions/48796/how-to-feed-lstm-with-different-input-array-sizes) points to three ways of handling this problem in practice.
# 
# ### First approach: mask everything to the same length
# The first and simplest way of handling variable length input is to set a special mask value in the dataset, and pad out the length of each input to the standard length with this mask value set for all additional entries created. Then, create a `Masking` layer in the model, placed ahead of all downstream layers. Those layers (*assuming they support masking*) will then proceed to ignore the masked values in their calculations.
# 
# Note that masking restricts us to layers that support masking. I'm not sure how restrictive this is, however; I assume not very.
# 
# In the case of an LSTM layer, masking is equivalent to skipping the propogation of the input values through the masked time steps. So a length five vector whose last five layers are masked will only go through the first five layers of an LSTM with 10 time steps. The output of the fifth time step will instead be propogated directly to the output layer.
# 
# The masking layer is covered [here](https://keras.io/layers/core/) in the Keras documentation.
# 
# This is the approach that you will want to use in most cases.
# 
# ### Second approach: set batch size to 1 and have keras mask automatically
# If we set `batch_size` to 1 (e.g. we perform stochastic or "live" training) and set the requisite length for the input layer to `None`, Keras is smart enough to mask your input automatically. This obviously only works if you are OK with batches with just a single sample in them, which is rarely what you want.
# 
# ### Third approach: bucketize sequences by length and mask per-bucket
# If the length of the sequence is informative, then perhaps the model will see improved performance with samples batched by size. This feels like a stretch to me, but it does represent another approach to this problem: batch the sentences by length, mask those sentences just-in-time, and feed those batches to the model. This requires building a custom data generator.
# 
# ---
# 
# [Overall the SO answer is very authoritative](https://datascience.stackexchange.com/questions/48796/how-to-feed-lstm-with-different-input-array-sizes).
# 
# ## Techniques for handling long input sequences
# 
# The opposite problem occurs when we have extremely long sequences. Such sequences are a problem because they potentially drive up the time cost of training by heaps and heaps. The "Machine Learning Mastery" blog has [a pretty authoritative article](https://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/) on this. In general, for our purposes, 250 to 500 sequence tokens is the practical limit to maximum sequence length.
# 
# ### Naive approaches
# 
# The simplest approach to long sequences is to simply truncate them, usually at the end but potentially at the beginning. This represents data loss, but might not be all that bad.
# 
# Other approaches are to summarize the sentence by e.g. removing stopwords, or otherwise to randomly pick words out of the sequence.
# 
# ### Truncated backpropogation
# "Truncated backpropogation through time" is a complicated name for a simple idea that is relatively in vogue right now: performing gradient updates based on backpropogation passes through a tail-end subset of the LSTM layers. This coarse approach results in less accurate updates, especially to the earlier LSTM time-step layers, but also makes for much faster training times.
# 
# Truncated backpropogation has the potential to make longer sequences tenable. I will cover this technique in more detail in a future notebook.
# 
# ### Encoder-decoders
# Autoencoders can be used to compress the sentence to a smaller space. This would essentially involve training an autoencoder, then feeding the input to just the first half of the model, the constraining half. The output vectors will be mathematically compressed word embeddings according to some loss function, so they will no longer be mappable directly back to the sequence they came from.
# 
# A step up from autoencoders, both in complexity and accuracy, are sequence-to-sequence RNNs, a new LSTM-based RNN architecture that maps sequences of one length to sequences of another length. Seq-to-seq RNNs have been very successful in translation domains; they power e.g. Google Translate (recall that long Wired article on what happened at Google Translate). The subject of a future notebook!
