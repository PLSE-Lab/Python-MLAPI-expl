#!/usr/bin/env python
# coding: utf-8

# # PyTorch design patterns
# This notebook is a high-level analysis of the architecture of PyTorch. Since building neural networks in PyTorch means extending the PyTorch system architecture, understanding that architecture is more important than you might at first think!
# 
# This notebook draws heavily from a series of articles from the Paperspace blog titled [PyTorch 101](https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/).
# 
# This notebook covers what might be considered "intermediate topics", it does not go into detail on e.g. every individual step of the training routine.
# 
# ## Autograd
# **Autograd** is the name of the automatic differenciation engine which underlies PyTorch. Optimizing a neural network means computing the gradient of the network with respect to the (potentially millions) of parameters included in the network on a batch of data, then updating those weights based on that gradient. This requires constructing and consuming a computational graph. Autograd does this.
# 
# Autograd operates on Torch tensor objects. Tensor objects may be created by the user by hand, or created by autograd as the output of a certain computation. Tensors have two system properties relevant to the computational engine: `requires_grad`, a boolean property that controls whether or not the tensor is elegible for gradient descent, and `grad_fn`, the function that created the operator. User-initialized tensors will have a user-set `requires_grad` value and a `grad_fn` of `None`. System-created tensors will have a `grad_fn` set by the system.
# 
# `grad_fn` is what actually links the different tensors in the computational graph together. It's what PyTorch uses to determine what to do during forward and backwards propogation passes.
# 
# One important difference between the default behaviors in PyTorch and TensorFlow is that the computational graph in PyTorch is dynamically determined, whilst the TensorFlow computational graph is statically compiled. In PyTorch, the computational graph is available in Python, and can trivially be modified mid-batch; in TensorFlow this is much harder to do, as the computational graph is a hard-coded non-Python struct. Because it is compiled, the TensorFlow computational graph is theoretically faster, but this is not often true in practice. Note that, to address this weakness, TensorFlow recently added its own dynamic mode (which it calls eager mode). To address the weaknesses of its dynamic graph (primarily portability), meanwhile, PyTorch recently added a compiled mode. Co-evolution!
# 
# How does PyTorch consume its dynamic graph? Each time `forward` is called on a model, the underlying computational graph is reconstructed. Each time `backwards` is called on a model, the underlying computational graph *is consumed*: e.g. as all of the gradients are calculated, the non-leaf nodes in the computational graph are destroyed, and the memory holding the intermediate values is deallocated. As a performance optimization, you can choose to retain the graph by running `backward(retain_graph=True)`. The [article](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/) states that this has the side effect of gradient accumulation, but this doesn't seem correct to me; [running `optimizer.zero_grad`](https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/3) should be all you need to zero the gradients between batch runs.
# 
# One implication of PyTorch having a dynamic graph is that inputs passed through the graph during `forward` operations must be cached, so that they may be applied to gradient calculation (and then discarded) during `backwards`. This is undesirable when performing non-training prediction tasks. PyTorch provides a `torch.no_grad()` context manager for this scenario: any operations performed inside the `no_grad` context will not cause intermediate value propogation.
# 
# ## Data loaders
# All differentiation is to take place on tensors, but the input to your model can be in any format (you can wrap the input in tensors as needed later on). My personal preference is to output data from data transformers as pandas DataFrames for columnar data or numpy arrays for all other data (sentence tokens, images), provide input to the model as numpy arrays (what scikit-learn does), and perform tensor wrapping after input.
# 
# When performing training, it is helpful to have just-in-time batching of our inputs in a class-based form. This allows fancy features like GPU pre-loading, parallelized data fetching, and so on. In Keras this is the function of `DataGenerator`, in PyTorch, this is the function of `torch.utils.data.Dataloader`.
# 
# The `DataLoader` itself takes a `Dataset` object as input. A `Dataset` object can be used in a modular way, e.g. you can extend it to create a  `Cifar10Dataset` object or `MNISTDataset` or whatever, and the article demonstrates how this is done. I am not sure whether or not I will personally follow the `Dataset` pattern, as it seems more ergonomic to me to define my own custom class for this purpose using the `scikit-learn` design patterns, then wrap that output with a `TensorDataset` before feeding it to the `DataLoader` object. This is what I did e.g. [here](https://www.kaggle.com/residentmario/pytorch-tabular-feedforward-v-3-gpu-first-full).
# 
# The `DataLoader` affords you control over `batch_size`, `shuffle`, and a few other optional things.
# 
# ## Class and functional composition
# PyTorch has `torch.nn.Module` class-based layers, and `torch.nn.Functional` functional layers. Functional layers are those layers which do not  need to maintain any state. For example, `MaxPooling2d`. Class-based layers are those layers that do need to maintain state. For example, `Conv2d`.
# 
# Many layers are available from both API paths, complicating things somewhat. But as a rule of thumb, you should *only* use functional layer definitions for layers that do not need state.
# 
# The design of PyTorch also allows for class composition. A layer or block or submodel may be implemented as its own `torch.nn.Module`, which can then be mixed into the definition for the higher-level model.
# 
# The Pytorch `Module` object registers (and tracks) two types of objects internally: *layers* and *parameters*. These objects must be registered properties of the `Module` object in order to be portable. What I mean by this is as follows: by default, layers and parameters are initialized on CPU. To move a model to GPU you run `model.gpu` on it, which copies all of the necessary resources. This is only possible if the model is aware of the resources that need to be copied, so PyTorch needs some discovery mechanism for such resources. They settled on enumerating the properties of the `Module` object and transfering over all layer or parameter objects in the list.
# 
# Parameters are wrapped tensors that represent weights in the model or other types of, well, parameters. A parameter may be created by wrapping a tensor object in a `nn.Parameter`. Every stateful layer implicitly allocates some number of parameter objects, which may be inspected by calling `model.parameters()`. Parameters are distinct from tensors in exactly one way: parmaeters assigned to `Module` object fields are copied automatically, tensors assigned to `Module` object fields are not. This can be a useful feature: e.g. you can cache tensors this way. Note that many models don't need parameters at all.
# 
# Layers are layers. You're expected to register these in your model `__init__` function. Not much else to say there!
# 
# You can register a list of layers or parameters all at once using `nn.ModuleList` and `nn.ParameterList`, if so inclined.
# 
# ## Weight initialization
# One design pattern for instantiating the weights on a layer in your model is to use `model.modules()` or `model.named_modules()`, roughly as follows:
# 
# ```
# for module in Net.modules():
#   if isinstance(module, nn.Conv2d):
#     weights = module.weight
# ```
# 
# This works well if you want to initialize weights all at once, e.g. you are performing pre-training.
# 
# To initialize weights based on type instead (e.g. 0-init for bias terms, orthogonal elsewhere) you want to use `model.named_parameters()`.
# 
# ## Discriminative learning
# Discriminative learning is the practice of applying different learning weights to different layers of the model. Generally speaking, you want smaller weights in the earlier sections of the model and larger weights in the later sections.
# 
# In PyTorch, the optimizer is applied to parameters in a list-of-sublists manner. Optimizers like e.g. `torch.optim.SGD` take either a list of parameters or a structured list of lists of parameters are input, and apply the corresponding hyperparameters to the requisite layers. Here's a short but succient demo:
# 
# ```
# class myNet(nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.fc1 = nn.Linear(10,5)
#     self.fc2 = nn.Linear(5,2)
#     
#   def forward(self, x):
#     return self.fc2(self.fc1(x))
# 
# Net = myNet()
# optimiser = torch.optim.SGD(Net.parameters(), lr = 0.5)
# 
# optimiser = torch.optim.SGD([{"params": Net.fc1.parameters(), 'lr' : 0.001, "momentum" : 0.99},
#                              {"params": Net.fc2.parameters()}], lr = 0.01, momentum = 0.9)])
# ```
# 
# ## Hooks
# Like Keras, PyTorch implements hooks, which can be used to perform actions based on network state. Hooks can be registered on either `forward` or `backwards` passes, and on either tensors (backwards only) or modules. Hooks are the primary method by which you attach loggers to the model.
# 
# Hoods on a tensor have the following type signature: `hook(grad) -> Tensor or None`. If `None` is returned, the tensor is not modified; if a tensor is returned, the old tensor is replaced with the ouput tensor *before* backpropogation actually takes place. This hook can be used to modify the gradient on the fly, if so desired, but it's mainly used for logging the gradient values and other forms of intermediate state to as desired (recall that this intermediate state is destroyed as the computational graph is consumed).
# 
# The article considers hooks on modules bad practice. They're only truly necessary when you want to retrieve intermediate feature maps from an opaque sequential submodule, but you can just rewrite your code in this case to get what you need directly. This is a good point of view.
