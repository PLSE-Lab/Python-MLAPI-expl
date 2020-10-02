#!/usr/bin/env python
# coding: utf-8

# ## Notes on recurrent neural networks
# 
# These notes are based largely on Colah's excellent blog post ["Understanding LSTM Networks"](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).
# 
# ### High-level overview
# Recurrent neural networks, or RNNs, are the top-performing model architecture for sequential data. They work very differently from feedforward neural networks and convolutional neural networks, which are both fundamentally linear, with data passing through layered steps. RNNs are fundamentally sequential: they may have one-to-many, one many-to-one, or many-to-many input structures.
# 
# Conceptually speaking, you can think of an RNN as a sequence of feedforward neural networks. The output from the first feedforward network is used by the second feedforward network, and that output is used by the third feedforward network, and so on down the chain.
# 
# Practically speaking, the way that this is acheived is that each output node on each layer serves simultaneously as an output node for the current sequence layer as well as an input node to the next sequence layer.
# 
# In visualization this dynamic is usually modelled as a node that is connected to itself:
# 
# ![](https://i.imgur.com/Gfqrphc.png)
# 
# An RNN is thus representable as a huge fold-out feedforward network that has this special self-referential structure. All of the normal learning routines continue to operate as you would expect them to, albeit with scoring that is multiplied across the many elements of the total output sequence.
# 
# RNNs are natively applicable to data structures which are sequential in nature. However, they are also applicable to any dataset that you can restructure in a sequential manner. For example, you can train an RNN to recognize and read digit sequences on license plates or door decals by feeding it sequential pixel-by-pixel input in a left-right manner.
# 
# ### LSTMs
# All of the cutting edge recurrent neural networks used in practice are a particular subtype of neural network known as an LSTM. LSTMs solve the **long-term memory problem**.
# 
# RNNs suffer from the limitation that they often cannot connect old but relevant information to new insight. They have a limited "memory" for the information that informs their decisions, and data that extends past that memory window is forgot and left unused. In theory RNNs *can* learn to propogate messages across long periods of time, but in practice they simply do not learn to do so in a convergent way. LSTMs solve this problem, and in doing so, greatly enhance the usefulness of the models.
# 
# In ordinary RNNs there are layers in between the output nodal layers. In the simplest case these layers may be a single `tanh` activation layer. In LSTMs these are replaced by a structure of four layers which interact in a special way.
# 
# Colah's blog has the following visualization of how it works:
# 
# ![](https://i.imgur.com/fdRdXbB.png)
# 
# The core of the LSTM is the topmost structure, the cell state. The cell state runs linearly through the entire sequence of neurons, but its intake of information is regulated by the rest of the structures in the diagram---the gates. In a vanilla LSTM training proceeds in three steps, corresponding with and regulated by three gates:
# 
# 1. The $\sigma$ gate is simplest; it controls which of the input signals are let through (1), ignored (0), or somewhere in between (it is called $\sigma$ because it uses sigmoid activation). This essentially controls retention of previous messages.
# 2. The next gate, $\tilde{C}_t$, is a $\tahn{x}$ activation on a group of inputs whose contribution is again regulated by another $\sigma$ activation gate. This controls the injection of new information into the RNN. For example, displacing an old understanding of the gender of a word with a new understanding of the gender of the next word the network is looking at.
# 3. The last gate controls the output; it does not input information into the cell state.
# 
# ### LSTM variants
# LSTMs have many variants. Some of the major ideas are:
# * **peephole connections** &mdash; The three gates are given an additional input: the cell state at that time.
# * **coupled forget and input gates** &mdash; The forget gate and the input gate make a paired decision: in order for new information to be introduced, old information must be forgotten/displaced.
# * **gated recurrant unit** &mdash; A complex transform that merges the forget and input gates and makes changes to interactions with the cell state. Ultimately simplifies model architecture.
