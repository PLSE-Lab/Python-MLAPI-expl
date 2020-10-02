#!/usr/bin/env python
# coding: utf-8

# # Notes on the transformer architecture
# 
# This notebook contains my notes on the transformer architecture, based on [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/).
# 
# 
# * **Transformers** were one of the hottest new ideas in 2019. Transformers were introduced by the paper "[Attention is all you need](https://arxiv.org/abs/1706.03762)". This is a model architecture for the machine translation task which uses attention to outperform the previous SOTA, sequence-to-sequence models with attention, on many tasks.
# * Transformers are a sequence-to-sequence model architecture: they still have a separate encoder and decoder component. The difference is in their use of an improved form of attention known as self-attention, which in addition to its expressive power has the computational advantage that it's expressable as a matrix operation.
# * The core architecture consists of a stack of encoders fully connected to a stack of decoders. Each encoder consists of two blocks: a self-attention component, and a feed forward network. Each decoder consists of three blocks: a self-attention component, an encoder-decoder attention component, and a feed forward component.
# 
# ![](https://i.imgur.com/mTJRnlO.png)
# 
# 
# ## Self-attention
# * Vanilla **attention** is the use of a (typically linear) combination of previous hidden vectors in the input sequence (a sentence, in the case of a sequence-to-sequence model). This combination of hidden vectors is typically concatenated with the current word's vector and fed to the next layer of the model.
# * Here's how attention is typically implemented in slightly more detail (in e.g. the PyTorch documentation's example). A zero matrix is created of sentence length. Each pass through the network populates a column in the matrix with a hidden vector for that word and word position. A hidden vector is generated for the next word. The prior hidden vector matrix is combined via a linear layer into a single combinatorial "prior context" vector, which is then concatenated to current word's hidden vector and passed to the next layer of the network.
# * **Self-attention** is a new spin on the attention technique. Instead of looking at prior hidden vectors when considering a word embedding, self-attention is a weighted combination of all other word embeddings (including those that appear later in the sentence):
# 
# ![](https://i.imgur.com/PHWQnbX.png)
# 
# * How self-attention is implemented:
#   1. The word embedding is transformed into three separate matrices -- queries, keys, and values -- via multiplication of the word embedding against three matrices with learned weights.
#   2. For each word embedding, for each word embedding (e.g. $n^2$ times), calculate the dot product of the current word's *query vector* with that word's *key vector*. E.g. for `w1` in a sentence with just two words in it we would calculate: `{q1 dot k1, q1 dot k2}`.
#   2. The query and key vectors are turned into a single value via the dot product.
#   3. They are then adjusted via normalization and softmax.
#   4. Multipy each word embedding's value vector by its softmax (this squeeze out low-value words, which will have a low softmax vector).
#   5. Sum the results.
# * The first layer of encoders is given the result of a word embedding trained on the raw words. This can be a tokenization in the simple case, e.g. a one-hot encoding, or more practically the result of an embedding algorithm like e.g. BERT. Every encoder in the stack after that is given the previous encoder's outputs.
# * This system replaces a linear sum layer, which has one weight per word position in the input, with three matrices of weights per word. The complexity of the representation is much higher in this layer than it is in a regular attention layer. On the other hand, the extent of what is learned with self-attention is so high that we can ditch using recurrent neural network layers (which learn essentially this same idea via forget gates and whatnot) and use simpler layers elsewhere. The Transformer architecture does not use RNNs!
# * The other great thing about the self-attention calculation is that because it involves only dot multiplication and some scalar operations on vectors, it can easily be written as a (fast, parallelizable) single-pass matrix operation. This makes both the forward and backwards passes on this layer fast to calculate.
# 
# 
# * A further innovation of the paper is **multi-headed attention**. The idea is that by repeating the self-attention mechanism multiple times the model can learn to seperate different kinds of useful semantic information onto different channels (or heads):
# 
# ![](https://i.imgur.com/biHuOOV.png)
# 
# * This requires as many more weights and much more matrix multiplication as you choose to have attention layers. However, mathematiclly speaking this still one matrix operation with an extra dimension. To compress the space of the output representation down, a weight matrix is learned that is multiplied against the concatenated output matrices to generate a slimmed-down representation.
# 
# 
# ## Positional encoding
# * Self-attention is naively position-unaware, e.g. it encoded `"The mouse ate the cat."` and `"The cat ate the mouse."` the same way. Contrast this with vanilla attention and with recurrant neural network layers, which are both implicitly position-aware because they process embeddings sequentially.
# * Adding this information into the model requires some form of **positional encoding**. In the paper transformer this is achieved by adding a positional vector to the word embedding *prior* to input to the self-attention layer. The position vector is a `(sin, cos)` structure: e.g. the first half of it is a bunch of `sin` operations on the current position, and the second half is a bunch of `cos` operations. Adding this to the embedding creates a learnable time dependency in the data; if this positional dependency is important to the task the model can use this information to learn it.
# 
# ![](https://i.imgur.com/elikwnF.png)
# 
# 
# ## Normalization and residual connections
# * Residual layer connections are used (of course) in both encoder and decoder blocks
# * The model uses layer normalization as well (layer normalization is a layer-wise alternative to batch normalization) in both encoder and decoder blocks.
# 
# ## Feed-forward
# * The penultimate layer in the block is a sequence of feed-forward networks. Each word in the sentence (up to the capped sentence length) is given its own feed-forward network.
# * Thus each position in the sentence is learned independently of each other position. Compare this to e.g. the introductory character-level RNN, which has just one feedforward layer operating on its input.
# * Having one feedforward layer per word in the sentence is computationally equivalent to having a recurrent neural network, *except* that because the feedforward layers are truly independent of one another (e.g. they are not connected to one another whatsoever), something made possible by using the self-attention mechanism to eliminate time dependence, they are embarrassingly parallel, and hence very computationally tractable.
# 
# ## Decoder architecture
# * Like the encoder, the decoder is stacked, with further layers of the decoder taking the outputs of the previous decoder as input.
# * While the encoder has two layers (besides normalization and resnet connection bits), the decoder has three:
#   * A self-attention layer that is semantically the same as the encoder one.
#   * A encoder-decoder attention layer, which operates on the output of the decoder's self-attention layer *and* the output of the final encoder as input.
#   * Position-independent feed-forward networks which pass into the output layer.
# * The decoder network's self-attention layer has all of the positions after the current word position masked (to `-inf`, apparently), e.g. it's only allowed to use information which occurred prior to its current position in the sentence.
# * The encoder-decoder attention layer is a self-attention layer with different sources:
#   * It takes its query matrix from the previous layer, the self-attention layer.
#   * It takes its key and value matrices from the output of the encoder.
#   * The sharing of the key and value matrices from the encoder with the decoder is thus the transfer-of-learning component of the encoder-decoder architecture.
# 
# ## Output layers
# * The output layers are two: a linear layer followed by a softmax layer. The linear layer performs a decompression task: it takes the final encoded representation for the output word and turns it around into log-odds for each word in the recipient language's volcabulary. A softmax running on this then turns that into probabilities and we take an `argmax` of that. Standard stuff for a language generation model.
# * The loss function is cross-entropy of KL divergence.
# * As an optimization, **beam search** is also possible: take the top N most likely words, and continue to search through the sentence from there with lookahead (I describe beam search in a bit more detail in this notebook: ["Seq-to-seq models, attention, teacher forcing"](https://www.kaggle.com/residentmario/seq-to-seq-rnn-models-attention-teacher-forcing/)).
# 
# ## Comparison to RNNs
# * A recurrant neural network is a of layer some kind time distributed over an input.
# * The transformer architecture does away with the time dependence aspect of the RNN architecture by dealing with those aspects of learning in a completely separate architecture. Thus whilst transformers have as many linear layers as there are words in the maximum-length sentence, just like with RNNs, these layers are disjoint, not time-dependent. And hence easier to compute, because they're embarrassingly parallel.
# * LSTMs and GRUs augment the basic feedforward layers of a vanilla RNN with forget gates and similar architecture to allow retention of relevant information. As with the feedfoward networks, precomputing this aspect of the network is more efficient (it is fully reducible to linear algebra) and obviates the need for these more complex layer types in our model.
# * Transformers are not better than traditional RNNs in all applications, RNNs still win in some contexts, but in those applications where they match or beat traditional RNNs they do so with lower computational cost.
