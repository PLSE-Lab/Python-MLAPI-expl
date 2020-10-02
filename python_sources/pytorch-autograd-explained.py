#!/usr/bin/env python
# coding: utf-8

# # Pytorch autograd explained
# I've written some previous notes on the design of the PyTorch API in general. This notebook contains some deeper notes on PyTorch autograd specifically, because understanding how this works at a slightly deeper level is valuable.
# 
# * In the past PyTorch had both `Tensor` objects and a wrapper on tensor objects known as `Variable`.
# * This changed in PyTorch version 0.4.0, which did away with the `Variable` wrapper and collapsed its properties and use cases into the `Tensor` object.
# * When working with a variable, it was possible to get a view of the underlying tensor using the `.data` accessor. The tensor retrieved is a view: it has `requires_grad=False` and is not attached to the computational graph that its `Variable` is attached to.
#   
#   This is necessary because arbitrary operations on a tensor are not supported by autograd&mdash;only supported operations defined by the PyTorch API are.
#   
#   After `Variable` was deprecated, the properties of the `Tensor` object were changed to those formerly assigned on `Variable`. The `data` accessor was retained, for backwards compatibility, with much the same behavior: it returns a view on the tensor that has `requires_grad=False` and is detached from the computational graph.
#   
#   However, actually _using_ this attribute is considered an anti-pattern. You should be using `detach()` instead. More on that later.
# * Besides `Tensor` and the deprecated `Variable`, there is one other wrapper class: `Parameter`. A `Parameter` is no more and no less than a tensor that has been attached to the PyTorch API's orginazatory class for a layer or model, a `nn.Module` object. Tensors which have been made into parameters have two additional properties:
#   * They move with the model. E.g. if you run `model.cuda()`, all of the model parameters will be transfered automatically.
#   * They are enumerable via the `parameters` and `named_parameters` properties of the `nn.Module` object.
# * To create a parameter, wrap a tensor in a `Parameter` class and assign it a module property. E.g. `self.X = nn.Parameter(nn.Tensor([1]))`.
# * Parameters created in a submodule (e.g. a `nn.Module` object assigned as a module property) are automatically added to the parent module's parameter list. This is e.g. how model weights are accessible from the model object. It is of course possible to iterate only over one's own parameters instead of recursively over one and one's children's parameters.
# 
# 
# * A simplified model of a PyTorch tensor is as an object containing the following properties:
#   * `data` &mdash; a self-reference (per the above).
#   * `required_grad` &mdash; whether or not this tensor is/should be connected to the computational graph.
#   * `grad` &mdash; if `required_grad` is true, this prop will be a sub-tensor that collects the gradients against this tensor accumulated during `backwards()`.
#   * `grad_fn` &mdash; This is a reference to the most recent operation which generated this tensor. PyTorch performs automatic differentiation by looking through the `grad_fn` list.
#   * `is_leaf` &mdash; Whether or not this is a leaf node (more on this later).
# 
# 
# * `requires_grad` is logically dominant: if a tensor is the function of tensor operations that involve at least one tensor with `requires_grad` is true, it will itself have `requires_grad` set to true. Conversely, if all precursors to a tensor do not require gradient, the output won't require it either.
# * The rules for `is_leaf` are more complicated.
# 
#   All Tensors that have `requires_grad` set to False will be leaf tensors. Tensors that have `requires_grad=True` will be leaf tensors IFF they were created by the user (and thus have `grad_fn=None`).
#   
#   Only leaf nodes accumulate gradients on the `grad` attribute. Essentially what the PyTorch API is doing is that it's assuming that you will initialize your weights tensors directly in your code, e.g. you will not make them dependent on prior computations (because there is no reason to do that). As long as you do this, the leaf rules will have the behavior that you expect: all of the tensors that you initialize (via `torch.tensor` init) will be subject to backpropogation.
#   
#   Unlike `requires_grad`, you can escape this pattern if you wish, by using the `retain_graph` method on a non-leaf tensor. This will then also let you set `requires_grad,` if you want.
# 
# 
# * OK, so how is backpropogration handled?
# * This next bit comes from [this excellent YouTube video walking through some example backprop works](https://www.youtube.com/watch?v=MswxJw-8PvE).
# * When you call a PyTorch op, e.g. multiplication, the method will use a context reference it is passed as part of the op to accumulate into a backwards graph. For example, consider the following: 
# 
# ![](https://i.imgur.com/xaWu3j1.png)
# 
# * Running `backward` calls the `MulBackward` operator with an initial value of `1`. The `MulBackward` operator has an internal list of prior operations I needs to run. `MulBackwards` computes the derivative of `C` w.r.t. `A` and `B`:
# 
#   $$\frac{d}{da}(c) = \frac{d}{da}\left[a * 3\right] = 3$$
#   $$\frac{d}{db}(c) = \frac{d}{db}\left[a * 3\right] = 0$$
# 
#   Notice that the derivative w.r.t. B is zero because B is constant because `requires_grad=False`. The other value, 3, is passed to an `AccumulateGrad` handler for `A`. A's `grad` attribute then is updated to this value.
# * One interesting detail of backpropogration is that under certain operations, the value of a constant tensor doesn't matter, but under other operations, it does.
# 
#   For example suppose a gradient-requiring tensor B and a gradient-nonrequiring tensor A. If we have `C = A + B`, the value of A doesn't matter because the constant factor will always be eliminated. If `C = AB`, then it does matter, because its value will be.
#   
#   PyTorch prevents in-place operations (like `A += 1`) from causing users to accidentally incorrectly accumulating gradients they've already computed using an internal `_version` attribute, which is incremented every time a tensor is operated on. `backwards` checks this value against an expected version stored in the backwards graph and raises if it notices a version mismatch.
# 
# 
# * Even though the computational graph is not something you typically interact with directly, it is nevertheless a (potentially large) object that's allocated in the background, one which automatically does things as you interact with your code. For this reason you need to be careful if your interactions with it.
# * Don't inadvertently collect gradients; make sure to call `zero_grad` after `backward` to clear them.
# * Calling `zero_grad` resets accumulated gradients but it does not consume the background computational graph. Even though PyTorch builds and consumes the computational graph repeatedly as it does its work, an old computational graph is preserved, so long as references exist to it. Therefore to avoid a memory leak you want to `detach` all of your outputs as soon as appropriate. This will pull a tensor reference off of the computational graph: it becomes a leaf node with `requires_grad=False` and `is_leaf=True` and no `grad_fn`. Once you've broken all references, the background computational graph is free to be garbage collected.
