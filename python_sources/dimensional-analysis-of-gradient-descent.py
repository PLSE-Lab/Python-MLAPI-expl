#!/usr/bin/env python
# coding: utf-8

# This notebook explores the dimensional unit of the learning rate hyperparameter $\alpha$ used in the gradient descent algorithm and related machine learning optimization algorithms (RMSprop, and Adam).
# 
# The bottom line is the following:
# * If you normalize both the inputs $X$ and the outputs $Y$ before applying machine learning algorithms, then $\alpha$ will always be unitless.
# * If you do not normalize both the inputs $X$ and the outputs $Y$ before applying machine learning algorithms, then the dimensional unit of $\alpha$ can depend on the units of $X$ and $Y$ and also upon the optimization algorithm selected.
# * It is beneficial for $\alpha$ to be unitless for the following reasons:
#   * It can speed-up gradient descent
#     * Especially when X contains multiple input features $X_1, X_2, ..., X_N$ where different input features have different dimensional units.
#     * Especially when Y predicts multiple output features $Y_1, Y_2, ..., Y_O$ where different output features have different dimensional units.
#   * It can allow one to switch between between optimization algorithms without the need to re-tune $\alpha$ separately for the newly selected algorithm. 
# 
# So the final recommendation is to always normalize both $X$ and $Y$ before applying machine learning algorithms.  Of course this implies that all of the $\hat{Y}$ values that the machine learning algorithm will predict will need to be de-normalized using the inverse transformation to arrive at the final real-world predicted values for $\hat{Y}$, but that is fine.

# Let's explore the details now.  Assume we have a simple neural network with no hidden layer and a linear activation function for the output layer.  In this case:
# $\hat{Y} = W X + b \tag{1}$
# 
# Assume that neither $X$ nor $Y$ has been normalized and that $X$ contains 2 input features ($X_1$, $X_2$) and $Y$ contains ground truth values for 2 output features ($Y_1$, $Y_2$), where the dimensional unit of $X_1$ is $U_1$, the unit of $X_2$ is $U_2$, the unit of $Y_1$ is $V_1$ and the unit of $Y_2$ is $V_2$.

# In this case:
# $$\begin{bmatrix}\hat{Y_1}\\ \hat{Y_2}\end{bmatrix} = \begin{bmatrix}W_{11} & W_{12}\\ W_{21} & W_{22}\end{bmatrix}\begin{bmatrix}X_1\\ X_2\end{bmatrix} + \begin{bmatrix}b_1\\ b_2\end{bmatrix}\tag{2}$$
# <br>
# where the corresponding dimensional unit for each number is as follows:
# $$\begin{bmatrix}V_1\\ V_2\end{bmatrix} = \begin{bmatrix} \frac{V_1}{U_1} & \frac{V_1}{U_2} \\ \frac{V_2}{U_1} & \frac{V_2}{U_2} \end{bmatrix}\begin{bmatrix}U_1\\ U_2\end{bmatrix} + \begin{bmatrix}V_1\\ V_2\end{bmatrix}\tag{3}$$

# # Gradient Descent requires X to be normalized

# In gradient descent we use the following formulas to update the parameters on each iteration:
# $$W_{11} := W_{11} - \alpha \frac{\delta J_1}{\delta W_{11}} \tag{4}$$
# <br>
# $$W_{12} := W_{12} - \alpha \frac{\delta J_1}{\delta W_{12}} \tag{5}$$
# <br>
# $$W_{21} := W_{21} - \alpha \frac{\delta J_2}{\delta W_{21}} \tag{6}$$
# <br>
# $$W_{22} := W_{22} - \alpha \frac{\delta J_2}{\delta W_{22}} \tag{7}$$
# <br>
# $$b_1 := b_1 - \alpha \frac{\delta J_1}{\delta b_1} \tag{8}$$
# <br>
# $$b_2 := b_2 - \alpha \frac{\delta J_2}{\delta b_2} \tag{9}$$

# Where $J_1$ and $J_2$ are two objective functions that will be co-minimized by the gradient descent, and are defined as follows:
# $$J_1 = \frac{1}{m} \sum_{i=1}^{m} (\hat{Y}_1^{(i)} - Y_1^{(i)})^2 \text{ with unit } V_1^2\tag{10}$$
# $$J_2 = \frac{1}{m} \sum_{i=1}^{m} (\hat{Y}_2^{(i)} - Y_2^{(i)})^2 \text{ with unit } V_2^2\tag{11}$$

# Substituting (2) into (10) and (11):
# $$J_1 = \frac{1}{m} \sum_{i=1}^{m} (W_{11} X_1^{(i)} + W_{12} X_2^{(i)} + b_1 - Y_1^{(i)})^2 \tag{12}$$
# $$J_2 = \frac{1}{m} \sum_{i=1}^{m} (W_{21} X_1^{(i)} + W_{22} X_2^{(i)} + b_2 - Y_2^{(i)})^2 \tag{13}$$

# Using (12) and (13) the partial derivitives are as follows:
# $$ \frac{\delta J_1}{\delta W_{11}} = \frac{2}{m} \sum_{i=1}^{m} (\hat{Y}_1^{(i)} - Y_1^{(i)}) X_1^{(i)} \text{ with unit } V_1 U_1 \tag{14}$$
# <br>
# $$ \frac{\delta J_1}{\delta W_{12}} = \frac{2}{m} \sum_{i=1}^{m} (\hat{Y}_1^{(i)} - Y_1^{(i)}) X_2^{(i)} \text{ with unit } V_1 U_2  \tag{15}$$
# <br>
# $$ \frac{\delta J_2}{\delta W_{21}} = \frac{2}{m} \sum_{i=1}^{m} (\hat{Y}_2^{(i)} - Y_2^{(i)}) X_1^{(i)} \text{ with unit } V_2 U_1  \tag{16}$$
# <br>
# $$ \frac{\delta J_2}{\delta W_{22}} = \frac{2}{m} \sum_{i=1}^{m} (\hat{Y}_2^{(i)} - Y_2^{(i)}) X_2^{(i)} \text{ with unit } V_2 U_2  \tag{17}$$
# <br>
# $$ \frac{\delta J_1}{\delta b_{1}} = \frac{2}{m} \sum_{i=1}^{m} (\hat{Y}_1^{(i)} - Y_1^{(i)}) \text{ with unit } V_1  \tag{18}$$
# <br>
# $$ \frac{\delta J_2}{\delta b_{2}} = \frac{2}{m} \sum_{i=1}^{m} (\hat{Y}_2^{(i)} - Y_2^{(i)}) \text{ with unit } V_2  \tag{19}$$

# Now that we know the units for all of the partial derivatives, we can rewrite equations (4)-(9) now showing the units of each term as follows:
# $$W_{11} := W_{11} - \alpha \frac{\delta J_1}{\delta W_{11}} \text{ with units } \frac{V_1}{U_1} := \frac{V_1}{U_1} - \alpha V_1 U_1 \tag{4}$$
# <br>
# $$W_{12} := W_{12} - \alpha \frac{\delta J_1}{\delta W_{12}} \text{ with units } \frac{V_1}{U_2} := \frac{V_1}{U_2} - \alpha V_1 U_2 \tag{5}$$
# <br>
# $$W_{21} := W_{21} - \alpha \frac{\delta J_2}{\delta W_{21}} \text{ with units } \frac{V_2}{U_1} := \frac{V_2}{U_1} - \alpha V_2 U_1 \tag{6}$$
# <br>
# $$W_{22} := W_{22} - \alpha \frac{\delta J_2}{\delta W_{22}} \text{ with units } \frac{V_2}{U_2} := \frac{V_2}{U_2} - \alpha V_2 U_2 \tag{7}$$
# <br>
# $$b_1 := b_1 - \alpha \frac{\delta J_1}{\delta b_1} \text{ with units } V_1 := V_1 - \alpha V_1 \tag{8}$$
# <br>
# $$b_2 := b_2 - \alpha \frac{\delta J_2}{\delta b_2} \text{ with units } V_2 := V_2 - \alpha V_2 \tag{9}$$

# Now dimensional analysis shows that there is a conflict regarding the unit of $\alpha$.
# 
# 
# For equations (8) and (9) to be valid, it is clear that $\alpha$ must be unitless.
# <br>
# For equations (4) and (6) to be valid, it is clear that $\alpha$ must have units of $\large \frac{1}{U_1^2}$.
# <br>
# For equations (5) and (7) to be valid, it is clear that $\alpha$ must have units of $\large \frac{1}{U_2^2}$.
# 

# Of course, one can simply ignore this dimensional analysis and proceed to use gradient descent with non-normalized $X$ data, and it will still work, however:
# * $b_1$ and $b_2$ will learn at learning rate $\alpha$
# * $W_{11}$ and $W_{21}$ will learn at an effective learning rate of $\alpha_1 = \large \frac{\alpha}{U_1^2}$
# * $W_{12}$ and $W_{22}$ will learn at an effective learning rate of $\alpha_2 = \large \frac{\alpha}{U_2^2}$.
# <br><br>
# For example, if $U_1 = s$ and $U_2 = ms$ and $V_1 = V_2 = m$ and $\alpha = 0.01 = 10^{-2}$:
# * $b_1$ and $b_2$ will have units $m$ and will learn at learning rate $\alpha = 10^{-2}$
# * $W_{11}$ and $W_{21}$ will have units $\frac{m}{s}$ and will learn at an effective learning rate of $\alpha_1 = \large \frac{10^{-2}}{s^{2}}$
# * $W_{12}$ and $W_{22}$ will have units $\frac{m}{ms} = \frac{km}{s}$ and will learn at an effective learning rate of $\alpha_2 = 10^{-2} / (10^{-3})^2 = \large \frac{10^{4}}{s^2}$.

# Based on the above analysis, the only way to make all of the gradient descent update equations valid simultaneously is to normalize the $X$ values, so that $X_1$ and $X_2$ will be unitless, and then we can update equations (4) to (9) using $U_1 = U_2 = 1$ as follows:
# $$W_{11} := W_{11} - \alpha \frac{\delta J_1}{\delta W_{11}} \text{ with units } V_1 := V_1 - \alpha V_1 \tag{4}$$
# <br>
# $$W_{12} := W_{12} - \alpha \frac{\delta J_1}{\delta W_{12}} \text{ with units } V_1 := V_1 - \alpha V_1 \tag{5}$$
# <br>
# $$W_{21} := W_{21} - \alpha \frac{\delta J_2}{\delta W_{21}} \text{ with units } V_2 := V_2 - \alpha V_2 \tag{6}$$
# <br>
# $$W_{22} := W_{22} - \alpha \frac{\delta J_2}{\delta W_{22}} \text{ with units } V_2 := V_2 - \alpha V_2 \tag{7}$$
# <br>
# $$b_1 := b_1 - \alpha \frac{\delta J_1}{\delta b_1} \text{ with units } V_1 := V_1 - \alpha V_1 \tag{8}$$
# <br>
# $$b_2 := b_2 - \alpha \frac{\delta J_2}{\delta b_2} \text{ with units } V_2 := V_2 - \alpha V_2 \tag{9}$$

# Notice that after normalizing the $X$ values, then $\alpha$ becomes unitless in all of the equations above.  This is good.  Not only does it make all of the equations valid from a dimensional analysis point of view, it also means that all of the $W$ and $b$ values will learn from the data at the same learning rate, which will allow the optimum to be found more quickly.
# <br>
# 
# 

# # RMSprop requires Y to be normalized

# In RMSprop we use the following formulas to update the parameters on each iteration:
# $$W_{11} := W_{11} - \alpha \frac{\frac{\delta J_1}{\delta W_{11}}}{\sqrt{S_{W_{11}}} + \epsilon} \text{ with units } \frac{V_1}{U_1} := \frac{V_1}{U_1} - \alpha \frac{V_1 U_1}{V_1 U_1 + \epsilon} \tag{20}$$
# $$W_{12} := W_{12} - \alpha \frac{\frac{\delta J_1}{\delta W_{12}}}{\sqrt{S_{W_{12}}} + \epsilon} \text{ with units } \frac{V_1}{U_2} := \frac{V_1}{U_2} - \alpha \frac{V_1 U_2}{V_1 U_2 + \epsilon} \tag{21}$$
# $$W_{21} := W_{21} - \alpha \frac{\frac{\delta J_2}{\delta W_{21}}}{\sqrt{S_{W_{21}}} + \epsilon} \text{ with units } \frac{V_2}{U_1} := \frac{V_2}{U_1} - \alpha \frac{V_2 U_1}{V_2 U_1 + \epsilon} \tag{22}$$
# $$W_{22} := W_{22} - \alpha \frac{\frac{\delta J_2}{\delta W_{22}}}{\sqrt{S_{W_{22}}} + \epsilon} \text{ with units } \frac{V_2}{U_2} := \frac{V_2}{U_2} - \alpha \frac{V_2 U_2}{V_2 U_2 + \epsilon} \tag{23}$$
# $$b_1 := b_1 - \alpha \frac{\frac{\delta J_1}{\delta b_1}}{\sqrt{S_{b_1}} + \epsilon} \text{ with units } V_1 := V_1 - \alpha \frac{V_1}{V_1 + \epsilon} \tag{24}$$
# $$b_2 := b_2 - \alpha \frac{\frac{\delta J_2}{\delta b_2}}{\sqrt{S_{b_2}} + \epsilon} \text{ with units } V_2 := V_2 - \alpha \frac{V_2}{V_2 + \epsilon} \tag{25}$$

# Using dimensional analysis, we can see conflicts for the units for both $\alpha$ and $\epsilon$.
# 
# 
# In every equation $\epsilon$ needs to have a different unit which matches the unit of the other term that it is added to such as $V_1 U_1$, $V_1 U_2$, ...
# 
# 
# In every equation $\alpha$ needs to have a different unit which matches the unit of the parameter that is being updated.
# 
# In this case normalizing only the $X$ values is not sufficient as it results in the following:
# $$W_{11} := W_{11} - \alpha \frac{\frac{\delta J_1}{\delta W_{11}}}{\sqrt{S_{W_{11}}} + \epsilon} \text{ with units } V_1 := V_1 - \alpha \frac{V_1}{V_1 + \epsilon} \tag{20}$$
# $$W_{12} := W_{12} - \alpha \frac{\frac{\delta J_1}{\delta W_{12}}}{\sqrt{S_{W_{12}}} + \epsilon} \text{ with units } V_1 := V_1 - \alpha \frac{V_1}{V_1 + \epsilon} \tag{21}$$
# $$W_{21} := W_{21} - \alpha \frac{\frac{\delta J_2}{\delta W_{21}}}{\sqrt{S_{W_{21}}} + \epsilon} \text{ with units } V_2 := V_2 - \alpha \frac{V_2}{V_2 + \epsilon} \tag{22}$$
# $$W_{22} := W_{22} - \alpha \frac{\frac{\delta J_2}{\delta W_{22}}}{\sqrt{S_{W_{22}}} + \epsilon} \text{ with units } V_2 := V_2 - \alpha \frac{V_2}{V_2 + \epsilon} \tag{23}$$
# $$b_1 := b_1 - \alpha \frac{\frac{\delta J_1}{\delta b_1}}{\sqrt{S_{b_1}} + \epsilon} \text{ with units } V_1 := V_1 - \alpha \frac{V_1}{V_1 + \epsilon} \tag{24}$$
# $$b_2 := b_2 - \alpha \frac{\frac{\delta J_2}{\delta b_2}}{\sqrt{S_{b_2}} + \epsilon} \text{ with units } V_2 := V_2 - \alpha \frac{V_2}{V_2 + \epsilon} \tag{25}$$

# Equations (20) (21) and (24) require the unit of both $\alpha$ and $\epsilon$ to be $V_1$.
# 
# 
# Equations (22) (23) and (25) require the unit of both $\alpha$ and $\epsilon$ to be $V_2$.
# 
# The only way to resolve this conflict is to also normalize the $Y$ values, which makes all of the parameters ($W$ and $b$) and both of the hyper-parameters ($\alpha$ and $\epsilon$) unitless.
# 
# Of course, if there is only one output $Y$, then there is no conflict and so there is no need to normalize $Y$ data in that case.  But normalizing Y values never hurts, so in general it is a good idea.  Even when you only have one output, normalizing the $Y$ data can help to avoid exploding gradients in deep networks.
# 

# # Adam requires Y to be normalized

# The Adam algorithm is a combination of gradient descent with momentum and with RMSprop, so all of the points made above for RMSprop also apply to the Adam optimizer. 

# # Considerations for more complex networks

# All of the analysis above was done with a simplistic network, with no hidden layers, and with only linear activations.  Next consider using a non-linear activation function such as a sigmoid.  In that case the dimensional analysis becomes more complicated.
# $$Z = W X + b$$
# $$\hat{Y} = \sigma(Z) = \frac{1}{1 + e^{-Z}}$$
# 
# In this case, if you don't normalize the $Y$ values, then the dimensional units of Z, W, and b become quite complex.  For example, if the unit of $Y$ is $V$ (volts), then what would the unit be for $Z$?
# 
# If you normalize the $Y$ values then $Y$ is unitless and $Z$ is unitless.  If you also normalize the $X$ values, then the $W$ and $b$ values are all unitless, and the hyper-parameters like $\alpha$ and $\epsilon$ are all unitless too.  All gradients would be unitless.  All losses would be unitless.
# 
# As, you can see, by normalizing both $X$ and $Y$ all of the problems that would be pointed out by dimensional analysis vanish, and all of the hyper-parameters become unitless, which makes them easier to tune, and loss values become unitless, which makes the loss values comparable from one dataset to another.

# In[ ]:




