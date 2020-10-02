#!/usr/bin/env python
# coding: utf-8

# # Notes on deep belief networks
# 
# **Deep belief networks** are a machine learning algorithm that looks remarkably similar to a deep neural network, but is actually not.
# 
# ## First, some works about restricted boltzmann machines
# In order to understand DBNs, one must first understand Restricted Boltzmann Machines, or RBMs. RMBs are a shallow feedforward "neural network alike" model that was invented by Geoffrey Hinton as a methodology for determining good starting weights for feedforward neural networks (a process known as **pretraining**) in a computationally efficient manner. An RBM consists of a two-layer network of fully connected nodes with both forward and backwards connections (a cycle). The forwards and backwards connections have the restriction that they share weights. On each pass through the network (either forwards or backwards), a gradient update is performed which affects the forward connections and the backwards connections simultaneously. This is, notably, *not* backpropogation! Instead a training regimen known as contrastive digestion is used, which is based on a metric known as KL divergence.
# 
# I cover RBMs in detail, including providing a worked example, in the notebook ["Restricted Boltzmann Machines and Pretraining"](https://www.kaggle.com/residentmario/restricted-boltzmann-machines-and-pretraining).
# 
# ## Deep belief network architecture
# 
# A deep belief network consists of a *sequence* of restricted boltzmann machines which are sequentially connected. Each of the Boltzmann machines layers is trained until convergence, then frozen; the result of the "output" layer of the machine is then fed as input to the next Boltzmann machine in the sequence, which is then itself trained until convergence, and so forth, until the entire network has been trained.
# 
# The following much-shared diagram shows the overall architecture:
# 
# ![](https://i.imgur.com/CodLXbO.png)
# 
# Notice that besides the output layer, every pair of layers in the hidden plus input layers compose a RBM.
# 
# ## Applications
# 
# Deep belief networks can substitute for a deep feedforward network or, given more complex arrangements, even a convolutional neural network. They have the advantages that they are less computationally expensive (they grow linearly in computational complexity with the number of layers, instead of exponentially, as with feedforward neural networks); and that they are significantly less vulnerable to the vanishing gradients problem.
# 
# However, because deep belief networks impart significant restrictions on their weight connections, they are also vastly less expressive than deep neural network, which outperform them on tasks for which sufficient input data is available.
# 
# Even in their prime, deep belief networks were rarely used in direct application. They were instead used as a pretraining step: a deep belief network with the same overall architecture as a corresponding deep neural network is defined and trained. Its weights are then taken and placed into the corresponding deep neural network, which is then fine-tuned and put to application.
# 
# Deep belief networks eventually fell out of favor in this application as well. For one thing, RBMs are just a special case of autoencoders, which were  found to be more broadly flexible and useful both for pretraining and for other applications. For another thing, the introduction of ReLU activation and its further refinement into leaky ReLU, along with the introduction of more sophisticated optimizers, learning late schedulers, and dropout techniques, have worked to greatly aleviate the vanishing gradient problem in practice, at the same time that increased data volumes and compute power have made direct deep neural network applications to problems more tractable.
# 
# See this Cross Validated question-answer pair for slightly more detail: ["Why are deep belief networks (DBNs) rarely used?"](https://stats.stackexchange.com/questions/261751/why-are-deep-belief-networks-dbn-rarely-used).
