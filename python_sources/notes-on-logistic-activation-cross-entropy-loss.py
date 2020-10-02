#!/usr/bin/env python
# coding: utf-8

# # Notes on logistic activation, cross-entropy loss
# 
# ## Introduction to logistic activation
# 
# Nodes in a neural network must apply an activation function to their tensor inputs in order to squash together the value they emit to the next node in the network. The choice of activation function is one of the most important choices in designing a neural network, as changing the activation changes the shape of the relationships which may be learned by the network.
# 
# I covered linear activation versus non-linear activation, as well as one specific activation function (softmax) in the ["Linear and non-linear activation, and softmax"](https://www.kaggle.com/residentmario/linear-and-non-linear-activation-and-softmax/) notebook. In this notebook we'll look at another non-linear activation function, **logistic activation**, in combination with a specific loss function.
# 
# The logistic activation function is defined as:
# 
# $$f(x) = \frac{1}{1 + e^{-wx}}$$
# 
# Where $x$ is the input value and $w$ is the weight learned for the node. The logistic function should be familiar because it's the same function used in logistic regression: I cover it in detail in [this past notebook](https://www.kaggle.com/residentmario/logistic-regression-with-wta-tennis-matches/). The properties of the logistic function when used in a neural network are similar. It's still a good choice of activation for classification tasks, and still a poor choice for regression tasks.
# 
# The outputs of individual logistic regressions amongst non-output nodes are obviously no longer directly interpretable as probabilities; they are instead just inputs for nodes further down the line. If you use logistic activation for your *output* nodes, however, and also use a loss function that penalizes based on probabilities, you can interpret those outputs (and thus, the output of the neural network overall) as probabilities. We saw in the previous notebook that softmax activation shares this property. However, softmax activation has one important advantage: it is tuned so that the sums of the values emitted by the output layer is always 1. To achieve this, it uses a more complicated formula, and one which takes into account the value contributions of neighboring nodes in the layer. Logistic activation is simpler formulaicly, but as a result does not include this correction.
# 
# When there is only one output layer, e.g. a binary classification task, this doesn't matter, as there are no neighbors to account for. But it does matter if you are performing multiclassification. Because they lack the softmax "continuity correction", In the multiclass case, logistic regression results may sum to well more than or well less than 1!
# 
# Thus logistic regression is an appropriate output layer for binary classification tasks, but not an appropriate output layer function for multiclassification tasks. 
# 
# Because it is defined in the domain $[-\infty, \infty]$ but in the range $[0, 1]$, it has the desirable property that even when the weight becomes very large in the positive or negative direction, the value predicted still doesn't move very much. Neural networks often suffer from a problem known as "exploding gradients" (covered in a later notebook). Although it's better to stop gradients from exploding to begin with, if you do get this problem, having this property keeps your model from going haywire.  Most of the rest of the non-linear activation functions share this property. For example, softmax is also clamped.
# 
# On the other hand, logistic activation has the weakness that its slope near zero is very large. This can make it harder to converge a weight on the correct value, as it makes it easier for an optimization step passing through this region of the curve to jump to, and past, the optimal stopping point. This is the main reason why logistic activation, simple though it may be, is rarely used in more complex and more "state of the art" neural networks, which use activation functions (like ReLu) which are more complicated and harder to train, but also have better convergence properties.
# 
# On the other hand, logistic activation is the simplest and most interpretable of the non-linear activation functions. It has a convenient derivative: $f'(x) = f(x)(1 - f(x))$. For this reason it is also the fastest non-linear activation to train. It is ideal for prototyping.
# 
# ## Introduction to cross-entropy loss
# 
# A neural network learns by performing optimizations against a loss function. The choice of loss function is important, as it fundamentally changes what characteristics of the data the network targets and the way in which it learns them. Neural network tasks are either regression tasks (approximate a continuous value) or categorical tasks (choose a discrete value, representing a category). For categorical learning tasks, the most popular loss function is **cross entropy loss**. Cross entropy loss is defined mathematically as:
# 
# $$C = - \sum_j t_j \log{y_j}$$
# 
# Where $t_j$ is the model prediction and $y_j$ the ground truth for a single observation.
# 
# This formula comes from information theory, where it plays an interesting role. Suppose that model outputs and ground truth outputs are the result of functions $t_j = q(x_j)$ and $y_j = p(x_j)$, respectively. Then it can be shown that, assuming $p$ and $q$ are both discrete probability distributions, a compression scheme optimized for the approximate distribution $q$, rather than the true distribution $p$, will need, on average, $C$ bits to encode an event from the set. Proving this requires knowing some other theorems from information theory. But it's good to keep in mind this formula's origin.
# 
# Cross-entropy is almost always what is used in combination with logistic activation. This is because when used in conjunction like this, cross-entropy has some nice properties.
# 
# Let us use logistic activation. Let $q(x)$ be the model prediction, and $w$ the weight. Then $q_{y=1} = \hat{y} = \frac{1}{1 + \exp{-wx}}$. $q_{y=0} = 1 - \hat{y}$. Similarly, $p_{y=1}=y$ and $p_{y=0} = 1-y$. Recall that $H(p, q) = - \sum_i p_i \log{q_i}$. Plugging the logistic regression function into this formula in place of $\hat{y}$ eventually results in the following expression, after simplification:
# 
# $$H(p, q) = - \sum_i p_i \log{q_i} = -y \log{\hat{y}} - (1-y)\log{1 - \hat{y}}$$
# 
# In the cross-entropy scheme, this is the cost of a single prediction. Our overall cost will be the sum of these costs:
# 
# $$J = \frac{1}{N}\sum_{n =1}^N H(p_n, q_n) = - \frac{1}{N} \sum_{n=1}^N \left[ y_n \log{\hat{y}_n} + (1 - y_n) \log{(1 - \hat{y}_n)}\right]$$
# 
# This is a nice closed-form loss expression for the loss the network produces that only uses $y$ and $\hat{y}$ in the computation.
# 
# Of course, during training the bulk of our time is spent performing backpropogation, and therefore computing the derivative $\frac{\partial J}{\partial w}$. If we perform this computation, we discover the fact that:
# 
# $$\frac{\partial J}{\partial w_{jk}^1} = \sum_{i =1}^{N_{out}}(y_i - t_i)(w_{ji})(h_j ( 1- h_j ))(x_k)$$
# 
# Where $h_j$ is the tensor output from the hidden unit, $w_{ji}$ is its weight, $s_j^1$ is its input sum, $y_i$ is the ground truth value, and $t_i$ is the target value. This derivation is worked out in [this reference sheet](https://www.ics.uci.edu/~pjsadows/notes.pdf).
# 
# This is the derivative of the cumulative cost with respect to a hidden layer node. This is a nice closed form expression for the derivative which may be applied recursively to further logistic units deeper into the network. Having this in tow allows us to perform training fast!
# 
# That's with logistic activation. How about softmax? Softmax layers are usually used on the output layer, and not for the hidden layers (for hidden layers there are easier-to-trian activations that perform better). When used against a softmax output layer, we get a really nice derivative:
# 
# $$\frac{\partial C}{\partial z_i} = y_i - t_i$$
# 
# Thus cross-entropy in combination with softmax has the nice properties that the gradient is constrained to $[-1, 1]$, and directly interpretable as the mean difference between the true and target values (which you should remember, since this is classification, will always be 0 or 1). This *exactly* cancels out the steepness of the softmax function near 0 (a property that it shares with logistic activation, which we described in the previous section).
# 
# Cross-entropy loss is so often used with softmax that sometimes the misnomer [softmax loss](https://www.quora.com/Is-the-softmax-loss-the-same-as-the-cross-entropy-loss) is used to name the combination of the two.
