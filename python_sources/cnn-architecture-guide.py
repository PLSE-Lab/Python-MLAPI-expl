#!/usr/bin/env python
# coding: utf-8

# # CNN Architecture
# 
# ![image.png](attachment:image.png)
# 
# In this notebook, we introduce various components of convolutional neural networks and how they can be used to assemble effective CNN classifiers. For more information about training CNN models, please see the following notebooks where we use ensemble CNN models to classify [digits](https://nbviewer.jupyter.org/github/cschupbach/deep_learning_cnn/blob/master/digits_ensemble.ipynb) and [fashion](https://nbviewer.jupyter.org/github/cschupbach/deep_learning_cnn/blob/master/fashion_ensemble.ipynb) items, achieving testing accuracies of 99.67% and 93.55%, respectively. Note that these examples utilize the MNIST datasets of the Keras distribution, where the `digits` dataset is composed of 60000 training samples and 10000 testing samples.

# **Import modules**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as pltgs
import torch
import keras
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras import backend as K
from keras.utils import to_categorical


# Create class `Plot()` for notebook network plots:

# In[ ]:


class Plot():
    def __init__(self, width=10, height=4, cmap='magma', size=12, bot=0.1):
        self.width = width
        self.height = height
        self.cmap = cmap
        self.size = size
        self.fig = plt.figure(figsize=(self.width, self.height))
        self.bot = bot

    def _network_dict(self, network_id):
        c = 'Convolution\n+ '
        dictionary = {
            0: [c + 'ReLU', r'Max Pooling $(2 \times 2)$'],
            1: [c + 'Tanh', r'Max Pooling $(2 \times 2)$'],
            2: [c + 'ReLU', r'Average Pooling $(2 \times 2)$'],
            3: [c + 'Tanh', r'Average Pooling $(2 \times 2)$'],
            4: [c + 'Tanh', r'Max Pooling $(2 \times 2)$', c + 'ReLU'],
            5: [c + 'ReLU', r'Max Pooling $(2 \times 2)$', c + 'ReLU',
                c + 'ReLU', r'Max Pooling $(2 \times 2)$']
        }
        return dictionary[network_id]

    def _no_ticks(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])
        return None

    def _get_lims(self, ax):
        return np.array(ax.get_xlim()), np.array(ax.get_ylim())

    def _shape_label(self, x, xlim, ylim):
        _shape = tuple([x.shape[i] for i in [1, 2, 0]])
        plt.text(np.mean(xlim), np.max(ylim)*1.05, _shape,
                 ha='center', va='top', size=self.size)
        return None

    def _level_locs(self, levels):
        x1 = [1 / (levels * 4)]
        x1 += [(i + 0.8) / levels for i in range(1, levels-1)]
        x2 = [(i + 0.7) / levels for i in range(1, levels-1)]
        x2 += [(levels - 0.25) / levels]
        return np.array(x1), np.array(x2)


    def _network_desc(self, levels, network_id):
        x1, x2 = self._level_locs(levels)
        labels = self._network_dict(network_id)
        gs = pltgs.GridSpec(1, 1, left=0, right=1, top=self.bot, bottom=0)
        ax = self.fig.add_subplot(gs[0,0])
        for i in range(len(x1)):
            ax.plot([x1[i], x1[i]], [0.8, 0.7], c='k')
            ax.plot([x1[i], x2[i]], [0.7, 0.7], c='k')
            ax.plot([x2[i], x2[i]], [0.8, 0.7], c='k')
            ax.text((x1[i] + x2[i]) / 2, 0.55, labels[i],
                     ha='center', va='top', size=self.size)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.axis('off')
        return None

    def _conv_desc(self, levels, activation):
        x1, x2 = self._level_locs(levels)
        label = 'Convolution'
        if len(activation) > 0:
            label += f'\n+ {activation}'
        gs = pltgs.GridSpec(1, 1, left=0, right=1, top=1, bottom=0)
        ax = self.fig.add_subplot(gs[0,0])
        for i in range(len(x1)):
            ax.plot([0.27, 0.45], [0.50, 0.50], c='k')
            ax.plot([0.45, 0.44], [0.50, 0.52], c='k')
            ax.plot([0.45, 0.44], [0.50, 0.48], c='k')
            ax.text((0.27 + 0.45) / 2, 0.47, label, ha='center', va='top',
                    size=self.size)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.axis('off')
        return None

    def _plot_input(self, x_input, levels):
        gs = pltgs.GridSpec(1, 2, left=0, right=1/levels, top=1,
                            bottom=self.bot)
        ax = self.fig.add_subplot(gs[0,0])
        ax.imshow(x_input[0,0], cmap=self.cmap, aspect='equal')
        xlim, ylim = self._get_lims(ax)
        plt.text(np.mean(xlim), -np.max(ylim)*0.05, 'Input',
                 ha='center', va='bottom', size=self.size)
        self._shape_label(x_input[0], xlim, ylim)
        self._no_ticks(ax)
        return None

    def _network_layout(self, x_input, x_list):
        input_size = x_input[0,0].shape[0]
        xrng = np.arange(len(x_list))
        x = [x_list[i][0,:,:,:] for i in xrng]
        layers = [x[i].shape[0] for i in xrng]
        size_ratio = [x[i].shape[1] / input_size for i in xrng]
        ws = [(1 / ((layers[i] - 1) * size_ratio[i])) - 1 for i in xrng]
        hs = [(1 / ((layers[i] - 1) * size_ratio[i])) - 1 for i in xrng]
        return x, xrng, layers, ws, hs

    def _plot_network(self, x_input, x_list, levels):
        x, xrng, layers, ws, hs = self._network_layout(x_input, x_list)
        for i in xrng:
            gs = pltgs.GridSpec(layers[i], layers[i], left=(i+1)/levels,
                                right=(i+2)/levels, top=1, bottom=self.bot,
                                wspace=ws[i], hspace=hs[i])
            for j in range(layers[i]):
                ax = self.fig.add_subplot(gs[j,j])
                ax.imshow(x[i][j], cmap=self.cmap, aspect='equal')
                self._no_ticks(ax)
            xlim, ylim = self._get_lims(ax)
            self._shape_label(x[i], xlim, ylim)
        return None

    def network(self, x_input, x_list, activation='', network_id=0,
                channels='first'):
        if channels == 'last':
            x_input = np.transpose(x_input, [0, 3, 1, 2])
            x_list = [np.transpose(x, [0, 3, 1, 2]) for x in x_list]
        levels = len(x_list) + 1
        if levels > 2:
            self._network_desc(levels, network_id)
        else:
            self.bot = 0
            self._conv_desc(levels, activation)
        self._plot_input(x_input, levels)
        self._plot_network(x_input, x_list, levels)
        return None


# Define additional helper/plotting functions:

# In[ ]:


def normalize(x):
    return x.astype('float32') / 255


def plot_samples(x_train, y_train):
    class_dict = np.arange(10)
    sample_list = [x_train[y_train[:,i].astype(bool)][:12] for i in range(10)]
    samples = np.concatenate(sample_list)
    gs = pltgs.GridSpec(10, 12, hspace=-0.025, wspace=-0.025)
    fig = plt.figure(figsize=(10, 8.5))
    yloc = np.linspace(0.95, 0.05, 10)
    for i in range(120):
        ax = fig.add_subplot(gs[i//12, i%12])
        ax.imshow(samples[i,:,:,0], cmap='magma')
        ax.set_xticks([])
        ax.set_yticks([])
    return None


def display_conv(x, conv, activation):
    x1 = K.eval(conv(x))
    Plot().network(K.eval(x), [x1], activation=activation, channels='last')
    return None


def display_conv_pool(x, conv_list, network_id, height=5):
    x_list = [conv_list[0](x)]
    for i in range(1, len(conv_list)):
        x_list += [conv_list[i](x_list[i-1])]
    x_list = [K.eval(x_list[i]) for i in range(len(conv_list))]
    Plot(height=height).network(K.eval(x), x_list, network_id=network_id,
                                channels='last')
    return None


def display_fc(model, x):
    label = model.get_layer(index=-1).name
    fc = K.eval(model(x))
    plt.figure(figsize=(10, 0.5))
    plt.imshow(fc, cmap='magma', aspect='auto')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    xloc = np.array(plt.gca().get_xlim())
    plt.text(xloc[0], -0.6, label, ha='left', va='bottom', size=12)
    plt.text(np.mean(xloc), 0.7, fc.shape, ha='center', va='top', size=12)
    return None


# **Read in data**:

# In[ ]:


train = pd.read_csv('../input/digit-recognizer/train.csv').to_numpy()
test = pd.read_csv('../input/digit-recognizer/test.csv').to_numpy()

x_train = normalize(train[:,1:]).reshape(-1, 28, 28, 1)
x_test = normalize(test).reshape(-1, 28, 28, 1)
y_train = to_categorical(train[:,0])

print(f'  Train data   shape = {x_train.shape}')
print(f'   Test data   shape = {x_test.shape}')
print(f'Train labels   shape = {y_train.shape}')


# The first 12 samples from each class are shown below.

# In[ ]:


plot_samples(x_train, y_train)


# # Feature Learning
# 
# Convolutional neural networks (CNN) classify high-dimensional input data by extracting feature maps in an iterative process known as feature learning.
# 
# $$\mathsf{input} \ \ \longrightarrow \ \ \mathsf{feature \ learning} \ \ \longrightarrow \ \ \mathsf{classification} \\ $$
# 
# The feature learning segment of the network is a set of feature maps or layers primarily constructed using convolutional kernels convolved with the input layers in a sequence.
# 
# $$\mathsf{input} \ \ \circledast \ \ \text{conv}_{1} \ \ \circledast \ \cdots \ \circledast \ \text{conv}_{N} \ \ \longrightarrow \ \mathsf{classification}$$

# ## Convolutional Layers
# ### Input/Output
# 
# Here we discuss the role of convolutional kernels/filters in CNN architecture. Given a convolutional kernel function $\Omega$, the kernel of $\omega$ is convolved with an input layer $x_\text{in}$, such that,
# 
# $$x_\text{out} = \Omega(x_\text{in})\tag{1} \\ $$
# 
# where $x_\text{out}$ is the output layer. The shape of the output layer $x_\text{out}$ is dependent on the shape of the input layer $x_\text{in} \in \mathbb{R}^{H_\text{in} \times W_\text{in} \times D_\text{in}}$ and the hyperparameters of the kernel function $\Omega$: the number of kernels $K$, kernel size $F$, the stride $S$, and the amount of padding $P$ used on the input border.
# 
# $$x_\text{in} \in \mathbb{R}^{H_\text{in} \times W_\text{in} \times D_\text{in}} \quad\overset{\Omega}{\longrightarrow}\quad x_\text{out} \in \mathbb{R}^{H_\text{out} \times W_\text{out} \times D_\text{out}}$$
# 
# where
# 
# $$H_\text{out} = \frac{H_\text{in} - F + 2P}{S} + 1\tag{2} \\ $$
# 
# $$W_\text{out} = \frac{W_\text{in} - F + 2P}{S} + 1\tag{3} \\ $$
# 
# $$D_\text{out} = K\tag{4}$$
# 
# 
# #### Example 1
# 
# Let the input layer $x_\text{in}$ be a tensor of size ${5 \times 5 \times 1}$ and the hyperparameters of $\Omega$ be $K=2$, $F=3$, $S=1$ and $P=0$, such that,
# 
# $$H_\text{out} = \frac{H_\text{in} - F + 2P}{S} + 1 = \frac{5 - 3 + 2(0)}{1} + 1 = 3 \\ $$
# 
# $$W_\text{out} = \frac{W_\text{in} - F + 2P}{S} + 1 = \frac{5 - 3 + 2(0)}{1} + 1 = 3 \\ $$
# 
# $$D_\text{out} = K = 2 \\ $$
# 
# Therefore, the output $x_\text{out} \in \mathbb{R}^{3 \times 3 \times 2}$ is a feature map with 2 layers and each layer is a $3 \times 3$ matrix. For additional information about specific kernel functions and convolution, feel free to check out my GitHub [image processing](https://github.com/cschupbach/image_processing) repository.

# ### PyTorch
# Example 2 briefly introduces the [torch.nn](https://pytorch.org/docs/stable/nn.html) module as an introduction to the PyTorch library. While the remainder of the notebook uses the Keras library, both PyTorch and Keras offer similar tools for constructing convolutional neural networks.
# 
# Note that PyTorch documentation specifies use of an input tensor in *channels first* format. This means the default input tensor for 2-dimensional convolution is expected to have the shape $N \times C_\text{in} \times H_\text{in} \times W_\text{in}$, where $N$ is the batch size, $C_\text{in}$ is the number of input channels/layers, $H_\text{in}$ is the input height or the number of rows, and $W_\text{in}$ is the width or the number of columns. Alternatively, Keras 2-dimensional convolution expects an input tensor of shape $N \times H_\text{in} \times W_\text{in} \times C_\text{in}$, which is known as *channels last* format.

# #### Example 2
# 
# Let's use the 40th training image as our sample input `x`.

# In[ ]:


x = x_train[39:40,:,:,:1]


# Currently `x` is a NumPy array of shape $1 \times 28 \times 28 \times 1$. For convolution using PyTorch, we need `x` to be a PyTorch tensor in *channels first* form (currently in *channels last*). We solve this requirement below.

# In[ ]:


x = np.transpose(x, [0, 3, 1, 2])
x = torch.Tensor(x)
print(f'x = {x.shape}')


# Below, we use the `torch.nn` function `Conv2d` to create two 2-dimensional convolution objects, `conv1` and `conv2`. The first convolution object `conv1` uses 3 kernel filters ($K=3$) and no zero padding ($P=0$) of the input border, while `conv2` uses 6 kernel filters ($K=6$) and adds one pixel of zero padding ($P=1$) to the input border. Both convolution objects have kernel size $F=3$, stride $S=1$, and one input channel, given that the input tensor is of shape/size $1 \times 1 \times 28 \times 28$.

# In[ ]:


conv1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3,
                        stride=1, padding=0)


# In[ ]:


conv2 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3,
                        stride=1, padding=1)


# The kernels of each convolution object convolve the input `x` as follows:

# In[ ]:


x1 = conv1(x).detach().numpy()
x2 = conv2(x).detach().numpy()


# We apply the `network` function of object `Plot` to display the results of each convolution as follows:

# In[ ]:


Plot().network(x, [x1], channels='first')


# In[ ]:


Plot().network(x, [x2], channels='first')


# ## Activation Functions

# Most CNN architectures also apply an activation function to convolutional layers. We use activation functions to increase non-linearity in the network. In theory, the output layers of convolution with activation have a thresholding effect similar to that of action potentials in the biological firing of neurons.
# 
# $$\text{input} \ \ \circledast \ \ \text{conv}_{1} \ + \ \text{activation}_1 \ \circledast \ \cdots \ \circledast \ \text{conv}_{N} \ + \ \text{activation}_N \longrightarrow \ \text{classification}$$

# ### Rectified Linear Unit  (ReLU)
# $$f(x) = \begin{cases} 0 & \text{for } \ x \leq 0 \\
# x & \text{for } \ x \gt 0 \end{cases}$$

# Here we provide an example of convolution with ReLU activation using Keras. Because the Keras library uses *channels last* format (by default) for 2-dimensional convolution, we revert tensor `x` back into an array, transform to *channels last* format, and express the array as a Keras tensor. 

# In[ ]:


x = x.detach().numpy()
x = np.transpose(x, [0, 2, 3, 1])
x = K.constant(x)


# We create two 2-dimensional convolution objects with ReLU activation, `relu1` and `relu2`, using the Keras function `Conv2D` below. The first convolution object `relu1` uses 3 kernel filters ($K=3$) and no zero padding ($P=0$) of the input border, while `relu2` uses 6 kernel filters ($K=6$) and adds one pixel of zero padding ($P=1$) to the input border. Both convolution objects have kernel size $F=3$ and stride $S=1$ (default). Note that the zero padding parameter is either `'valid'` (default), where $H_\text{out}$ and $W_\text{out}$ are according to equations (2) and (3) with $P=0$, or `'same'`, where $H_\text{out} = H_\text{in}$ and $W_\text{out}=W_\text{in}$.

# In[ ]:


relu1 = Conv2D(filters=3, kernel_size=3, strides=1, padding='valid',
               activation='relu')
relu2 = Conv2D(filters=6, kernel_size=3, strides=1, padding='same',
               activation='relu')


# We use the `display_conv` function to display the resulting ReLU-activated convolutional layers as follows:

# In[ ]:


display_conv(x, relu1, activation='ReLU')


# In[ ]:


display_conv(x, relu2, activation='ReLU')


# ### Softmax
# $$P(y=j| \mathbf{x}) = \dfrac{e^{\mathbf{x}^\mathsf{T}\mathbf{w}_j}}{\sum_{k = 1}^{k} e^{\mathbf{x}^\mathsf{T}\mathbf{w}_k}}$$

# In[ ]:


softmax = Conv2D(filters=3, kernel_size=3, activation='softmax')


# In[ ]:


display_conv(x, softmax, activation='Softmax')


# ### Sigmoid
# $$f(x) = \sigma(x) = \dfrac{1}{1 + e^{-x}}$$

# In[ ]:


sigmoid = Conv2D(filters=3, kernel_size=3, activation='sigmoid')


# In[ ]:


display_conv(x, sigmoid, activation='Sigmoid')


# ### Hyperbolic Tangent (tanh)
# $$f(x) = \tanh(x) = \dfrac{e^x - e^{-x}}{e^x + e^{-x}}$$

# In[ ]:


tanh = Conv2D(filters=3, kernel_size=3, activation='tanh')


# In[ ]:


display_conv(x, tanh, activation='Tanh')


# ### Exponential Linear Unit (ELU)
# $$f(\alpha, x) = \begin{cases} \alpha (e^x - 1) & \text{for } \ x \leq 0 \\
# x & \text{for } \ x \gt 0 \end{cases}$$

# In[ ]:


elu = Conv2D(filters=3, kernel_size=3, activation='elu')


# In[ ]:


display_conv(x, elu, activation='ELU')


# ### Softplus
# $$f(x) = \ln(1 + e^x)$$

# In[ ]:


softplus = Conv2D(filters=3, kernel_size=3, activation='softplus')


# In[ ]:


display_conv(x, softplus, activation='Softplus')


# ### Softsign
# $$f(x) = \dfrac{x}{1 + |x|}$$

# In[ ]:


softsign = Conv2D(filters=3, kernel_size=3, activation='softsign')


# In[ ]:


display_conv(x, softsign, activation='Softsign')


# ### Others
# 
# - Leaky ReLU
# 
# - PReLU
# 
# - RReLU
# 
# - SELU
# 
# - GELU

# ## Pooling
# 
# Pooling methods are often used between certain convolutions for dimensionality reduction. The most common method for pooling input layers is *max* pooling using a $2 \times 2$ kernel with non-overlapping strides. In certain cases with very large input, pooling with a $3 \times 3$ kernel may be effective. Typically, *max* pooling produces better results than the less common *average* pooling.
# 
# ### Max Pooling

# In[ ]:


relu = Conv2D(filters=6, kernel_size=3, padding='same', activation='relu')
tanh = Conv2D(filters=6, kernel_size=3, padding='same', activation='tanh')


# In[ ]:


maxpool = MaxPooling2D(pool_size=(2, 2))


# We use the `display_conv_pool` function to display the resulting layers as follows:

# In[ ]:


display_conv_pool(x, [relu, maxpool], network_id=0)


# In[ ]:


display_conv_pool(x, [tanh, maxpool], network_id=1)


# ### Average Pooling

# In[ ]:


avgpool = AveragePooling2D(pool_size=(2, 2))


# In[ ]:


display_conv_pool(x, [relu, avgpool], network_id=2)


# In[ ]:


display_conv_pool(x, [tanh, avgpool], network_id=3)


# ## Stacking Layers

# In[ ]:


relu = Conv2D(filters=6, kernel_size=3, padding='same', activation='relu')
tanh = Conv2D(filters=6, kernel_size=3, padding='same', activation='tanh')


# In[ ]:


display_conv_pool(x, [tanh, maxpool, relu], network_id=4)


# In[ ]:


maxpool = MaxPooling2D(pool_size=(2, 2))
relu1 = Conv2D(filters=8, kernel_size=3, padding='same', activation='relu')
relu2 = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')
relu3 = Conv2D(filters=16, kernel_size=3, activation='relu')


# In[ ]:


display_conv_pool(x, [relu1, maxpool, relu2, relu3, maxpool],
                  network_id=5)


# A common CNN feature learning architecture is similar to that shown above, where feature maps are constructed with ReLU-activated convolutional layers and intermittent max pooling. It's also fairly common to normalize the batch of input layers prior to convolution in the feature learning segment.

# # Classification
# 
# In the classification segment of the network, we flatten the final feature learning layer to form fully connected (FC) layers.
# 
# ## Fully Connected (FC) Layers
# 
# Here, we introduce FC layers by building a network of feature learning layers using Keras [Sequential](https://keras.io/api/models/sequential/) class. We initialize a sequential object and add feature learning layers to the model as follows:

# In[ ]:


model = Sequential()
model.add(Conv2D(8, 3, padding='same', activation='relu',
                 input_shape=(28, 28, 1), name='conv_1'))
model.add(Conv2D(8, 3, padding='same', activation='relu', name='conv_2'))
model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_1'))
model.add(Conv2D(16, 3, activation='relu', name='conv_3'))
model.add(Conv2D(16, 3, activation='relu', name='conv_4'))
model.add(Conv2D(16, 3, activation='relu', name='conv_5'))
model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_2'))


# ![image.png](attachment:image.png)
# 
# **Figure 1.** Example of fully connected layers where the flattened input layer is fed into a sequence of dense, fully connected layers. Neurons of each fully connected layer are linked to all activations in the prior layer.

# ### Flatten
# The output layers of the feature learning network above has the shape $4 \times 4 \times 16$. To begin the classification segment of the CNN, we flatten the output layers forming input array of length 256 similar to that of the simplified neural network shown in Figure 1 [[1](https://www.researchgate.net/publication/331525817_Temporal_Convolutional_Neural_Network_for_the_Classification_of_Satellite_Image_Time_Series)] and displayed the fully connected layer using the `display_fc` function.

# In[ ]:


model.add(Flatten(name='flatten_1'))
display_fc(model, x)


# ### Dense
# We sequentially add to the model two ReLU-activated fully connected layers of size 128 and a Softmax-activated output layer with a length equal to the number of classes (in our case 10).

# In[ ]:


model.add(Dense(128, activation='relu', name='dense_1'))
display_fc(model, x)


# In[ ]:


model.add(Dense(128, activation='relu', name='dense_2'))
display_fc(model, x)


# In[ ]:


model.add(Dense(10, activation='softmax', name='dense_3'))
display_fc(model, x)


# ### Network Summary

# The network can be summarized as follows:

# In[ ]:


model.summary()


# # Additional Layers and Functions
# 
# Many additional layers and layer functions may be added to fine tune the base architecture of the CNN shown above based on the data being classified. While a few common additions are discussed below, we highly suggest reviewing documentation for the [Keras Layers API](https://keras.io/api/layers/).
# 
# ## Batch Normalization
# We use batch normalization to scale values of inputs to mean zero and unit variance. This helps account for a problem called covariate shift, where the distribution of data varies across training and testing batch inputs. For some datasets, batch normalization may increase efficiency by reducing the number of epochs required to effectively train the CNN. However, batch normalization may not be suitable in some instances due to an increase in the amount of time required to complete an epoch. Results typically depend on the data and where batch normalizations are added in the model. For a datasets like `digits`, where the distribution of pixel values of is likely similar across large batches of input data, batch normalization may not be worth the increase in computational complexity.
# 
# ## Dropout Layers
# A dropout layer is a regularization method where a proportion of input values are randomly set to zero. Doing this reduces the amount of overfitting while training and results in a fit more robust to potential outliers. We typically apply dropout layers to inputs of the fully connected layers during the classification segment of the network. 
# 
# ## L1/L2 Regularization
# Regularization techniques like L1 and L2 regularization can be applied to layers of the network to reduce/prevent overfitting. The regularizers are added as penalty terms to the weight parameters of the loss function to reduce model complexity.

# # Citations
# 
# 1. C. Pelletier, G. Webb, and F. Petitjean, "Temporal Convolutional Neural Network for the Classification of Satellite Image Time Series," *Remote Sensing*, vol. 11, no. 5, p. 523, 2019.
