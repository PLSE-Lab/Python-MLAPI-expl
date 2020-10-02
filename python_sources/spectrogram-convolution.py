#!/usr/bin/env python
# coding: utf-8

# ## **Can convolution extract useful features from a spectrogram of seismic measurements?**
# 
# *[See [here](https://www.kaggle.com/michael422/tf-lstm-on-acoustic-signals) for a related article on using [Recurrent Neural Networks](https://www.kaggle.com/michael422/tf-lstm-on-acoustic-signals) on seismic data.]*
# 
# [Spectrograms](https://en.wikipedia.org/wiki/Spectrogram) evaluate a 1-dimensional signal input, and produce a 2-dimensional output which is a [Fourier transform](https://en.wikipedia.org/wiki/Fourier_transform) moving through time.  The approach is to divide the signal into frames, then apply a discrete Fourier transform ([FFT](https://en.wikipedia.org/wiki/Fast_Fourier_transform)) to each of the frames using a smoothing kernel or window function.  The output indicates the power of the signal in each of a fixed number of frequency bins (example below).
# 
# 1-dimensional convolutional neural networks are often used in signal processing.  2-dimensional convolution can also be used with spectrograms ([example](https://arxiv.org/abs/1701.02720)).  In the context of speech recognition, a variety of additional preprocessing techniques suited to the audio spectrum of human speech are applied.  Here, we will attempt convolution with minimal preprocessing on log spectrograms generated from the LANL labquake data.
# 
# We use the first 400MM records in order to process efficiently in-memory:

# In[ ]:


import os
import numpy as np
import pandas as pd
from datetime import datetime
from functools import partial
import tensorflow as tf
from tqdm import tqdm_notebook, tnrange
from scipy import signal
import matplotlib.pyplot as plt
np.warnings.filterwarnings('ignore')

rng = np.random.RandomState(datetime.now().microsecond)

data_path = '../input/LANL-Earthquake-Prediction'
tf_path = os.path.join(data_path, 'tf')

nrows = 400000000   # 629145480
data = (pd.read_csv(os.path.join(data_path, 'train.csv'), nrows=nrows,
                   dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
            .values)
print(f'input data shape: {data.shape}')


# First, downsample the training data.  This will aid performance and reduce some of the noise in the raw signal.  The [scipy.signal.decimate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html) function applies a low-pass (noise-reducing) filter, and samples the data every n-th value based on the downsample rate

# In[ ]:


def downsample(data, downsample_rate):
    trunc = int((data.shape[0] // downsample_rate) * downsample_rate)
    data = data[:trunc, :]
    x = signal.decimate(data[:, 0].ravel(), downsample_rate)
    y_ = data[:, 1].reshape((-1, downsample_rate)).T
    y = y_.mean(axis=0)
    return x, y

RAW_SEQ_LEN = 150000
DOWNSAMPLE_RATE = 4

assert (RAW_SEQ_LEN / DOWNSAMPLE_RATE) % 1 == 0
downsampled_seq_len = int(RAW_SEQ_LEN/DOWNSAMPLE_RATE)

X_train_, y_train_ = downsample(data, DOWNSAMPLE_RATE)

print(f'downsample rate: {DOWNSAMPLE_RATE}')
print(f'downsampled seq len: {downsampled_seq_len}')
print(f'X shape: {X_train_.shape} y shape: {y_train_.shape}')


# Next, define a spectrogram function using [scipy.signal.spectrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html).  The choice of window function, segment length, and overlap all affect the output.  In this case, we use a triangular smoothing window, a frame length of 256, and an overlap of 64.  The function will return a log spectrogram, standardized by arguments we will determine next.
# 
# For more background on applications of spectrograms to machine learning, see Haytham Fayek's [blog](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html) post.

# In[ ]:


def spectrogram(sig_in, dsamp):
    nperseg = 256 # default 256
    noverlap = nperseg // 4 # default: nperseg // 8
    fs = 4000000 // dsamp # raw signal sample rate is 4MHz
    window = 'triang'
    scaling = 'density' # {'density', 'spectrum'}
    detrend = 'linear' # {'linear', 'constant', False}
    eps = 1e-11
    f, t, Sxx = signal.spectrogram(sig_in, nperseg=nperseg, noverlap=noverlap,
                                   fs=fs, window=window,
                                   scaling=scaling, detrend=detrend)
    return f, t, np.log(Sxx + eps)


# Visualize a sample signal and the log spectrogram generated from it.  The vertical axis is a range of frequency bands ranging 0-500kHz.  The axis is inverted so that high frequencies are at the top.

# In[ ]:


start = rng.randint(0, X_train_.shape[0] - downsampled_seq_len)
sig_in = X_train_[start:start+downsampled_seq_len]

f, t, Sxx_out = spectrogram(sig_in, DOWNSAMPLE_RATE)

fig = plt.figure(figsize=(18, 8))
ax = fig.add_subplot(2, 1, 1)
ax.margins(x=0.003)
plt.plot(X_train_[start:start+downsampled_seq_len])
plt.title('downsampled signal:', fontsize=18, loc='left')
plt.axis('off')

ax = fig.add_subplot(2, 1, 2)
cmap = plt.get_cmap('magma')
spec = plt.pcolormesh(t, f, Sxx_out, cmap=cmap)
plt.title('normalized log spectrogram (aspect stretched):',
          fontsize=18, loc='left')
plt.axis('off');


# Using convolution on spectrographs is appealing on the surface, since spectrograms are 2-dimensional and visually intuitive.  However, the height and width of a photograph measure the same thing: spatial distance (3-dimensional distance projected to 2 dimensions).  2D convolution leverages the notion that objects in a photograph can shift, rotate, and warp (due to perspective), but the object is still the same.  Spectrograms are different.  The horizontal axis measures time, while the vertical axis corresponds to frequency.  See [this article](https://towardsdatascience.com/whats-wrong-with-spectrograms-and-cnns-for-audio-processing-311377d7ccd) for a more detailed discussion.
# 
# In fact, a spectrogram is a just time series of frequency measurements.  Local features (periodic, repeating signals) are present in most time series on multiple scales.  Convolution is one way to extract these recurring patterns if they exist.
# 
# Following (https://arxiv.org/abs/1701.02720) we will augment the features with a delta spectrogram (representing the change in FFT response between time frames) and the delta-delta spectrogram, which is the second-order change. Since these 2D matrices are all indexed by the same frequency bands and time periods, we will stack the matrices along a third dimension, creating features that are h x w x 3.

# In[ ]:


N_CHANNELS = 3

def get_3d_spec(Sxx_in, moments=None):
    if moments is not None:
        (base_mean, base_std, delta_mean, delta_std,
             delta2_mean, delta2_std) = moments
    else:
        base_mean, delta_mean, delta2_mean = (0, 0, 0)
        base_std, delta_std, delta2_std = (1, 1, 1)
    h, w = Sxx_in.shape
    right1 = np.concatenate([Sxx_in[:, 0].reshape((h, -1)), Sxx_in], axis=1)[:, :-1]
    delta = (Sxx_in - right1)[:, 1:]
    delta_pad = delta[:, 0].reshape((h, -1))
    delta = np.concatenate([delta_pad, delta], axis=1)
    right2 = np.concatenate([delta[:, 0].reshape((h, -1)), delta], axis=1)[:, :-1]
    delta2 = (delta - right2)[:, 1:]
    delta2_pad = delta2[:, 0].reshape((h, -1))
    delta2 = np.concatenate([delta2_pad, delta2], axis=1)
    base = (Sxx_in - base_mean) / base_std
    delta = (delta - delta_mean) / delta_std
    delta2 = (delta2 - delta2_mean) / delta2_std
    stacked = [arr.reshape((h, w, 1)) for arr in (base, delta, delta2)]
    return np.concatenate(stacked, axis=2)


# Sample the training data and process spectrograms for the samples in order to get factors for standardizing inputs to the model.  There are three pairs of moments required to standardized the inputs: a mean and sigma for the spectrograms as well as the deltas and delta-deltas:

# In[ ]:


def get_moments(X, downsampled_seq_len, samp=1000):
    nrows = X.shape[0]
    start_list = [rng.randint(0, nrows-downsampled_seq_len)
                  for i in range(samp)]
    Sxx_samples = [spectrogram(X[start:start+downsampled_seq_len],
                               dsamp=DOWNSAMPLE_RATE)[2]
                   for start in start_list]
    sxx_h, sxx_w = Sxx_samples[0].shape
    Sxx_3d_samples = np.array([get_3d_spec(Sxx) for Sxx in Sxx_samples])
    base_mean = Sxx_3d_samples[:, :, :, 0].mean()
    base_std = Sxx_3d_samples[:, :, :, 0].std()
    delta_mean = Sxx_3d_samples[:, :, :, 1].mean()
    delta_std = Sxx_3d_samples[:, :, :, 1].std()
    delta2_mean = Sxx_3d_samples[:, :, :, 2].mean()
    delta2_std = Sxx_3d_samples[:, :, :, 2].std()
    return (sxx_h, sxx_w), (base_mean, base_std, delta_mean,
                            delta_std, delta2_mean, delta2_std)

(sxx_h, sxx_w), moments = get_moments(X_train_, downsampled_seq_len)
print(f'spectrogram dims: {sxx_h}x{sxx_w}')
print(f'base spectrogram mean, sigma: {moments[0]:.4f}, {moments[1]:.4f}')
print(f'delta spectrogram mean, sigma: {moments[2]:.4f}, {moments[3]:.4f}')
print(f'delta-delta spectrogram mean, sigma: {moments[4]:.4f}, {moments[5]:.4f}')


# Visualize a sample  spectrogram as well as the delta and delta-delta spectrograms:

# In[ ]:


start = 21307218
sig_in = X_train_[start:start+downsampled_seq_len]

f, t, Sxx = spectrogram(sig_in, DOWNSAMPLE_RATE)
s3d = get_3d_spec(Sxx, moments)

fig = plt.figure(figsize=(18, 16))
ax = fig.add_subplot(4, 1, 1)
ax.margins(x=0.003)
plt.plot(X_train_[start:start+downsampled_seq_len])
plt.title('downsampled signal:', fontsize=14, loc='left')
plt.axis('off')

ax = fig.add_subplot(4, 1, 2)
cmap = plt.get_cmap('magma')
spec = plt.pcolormesh(t, f, s3d[:, :, 0], cmap=cmap)
plt.title('log spectrogram (aspect stretched):',
          fontsize=14, loc='left')
plt.axis('off')

ax = fig.add_subplot(4, 1, 3)
spec = plt.pcolormesh(t, f, s3d[:, :, 1], cmap=cmap)
plt.title('delta log spectrogram:', fontsize=14, loc='left')
plt.axis('off')

ax = fig.add_subplot(4, 1, 4)
spec = plt.pcolormesh(t, f, s3d[:, :, 2], cmap=cmap)
plt.title('delta-delta log spectrogram:', fontsize=14, loc='left')
plt.axis('off');


# The delta spectrograms add useful features that emphasize abrupt changes in the signal at different frequencies.
# 
# In this application, we do not use filter banks, which transform the frequency band of the spectrogram to a log scale, leaving constant spacing between harmonic frequencies.  Filter banks are frequently used in analysis of speech and other audible signals.  However, the signals in this dataset are mostly well above 50kHz (see below), while human hearing is limited to about 20kHz, so Mel-frequency banks are not suitable.  A custom log-scale filter bank might bring out the harmonics in this particular spectrum, but we will keep things simple for now.

# In[ ]:


plt.figure(figsize=(18, 4))
cmap = plt.get_cmap('magma')
ax = plt.pcolormesh(t, f/1000, s3d[:, :, 0], cmap=cmap)
plt.ylabel(r'frequency (kHz)', fontsize=14)
ax.axes.tick_params(axis='y', labelsize=14)
ax.axes.get_xaxis().set_visible(False);


# Split the data into train and validation sets.  The train and validation data have different distributions, which will lead to unexpected validation results below.

# In[ ]:


TRAIN_VAL_RATIO = 0.7

train_size = int(X_train_.shape[0] * TRAIN_VAL_RATIO)
X_train = X_train_[:train_size]
X_val = X_train_[train_size:]
y_train = y_train_[:train_size]
y_val = y_train_[train_size:]

print(f'train mean ttf: {y_train.mean():.4f}, val mean ttf: {y_val.mean():.4f}')
print(f'train 1st quartile ttf: {np.quantile(y_train, 0.25):.4f}, ' +
      f'val 1st quartile ttf: {np.quantile(y_val, 0.25):.4f}')
print(f'train 3rd quartile ttf: {np.quantile(y_train, 0.75):.4f}, ' +
      f'val 3rd quartile ttf: {np.quantile(y_val, 0.75):.4f}')


# Define a batching function to sample the input signal.  2d convolution takes a rank-4 tensor as input:
# 
#     (batch x image height x image width x channels)
# Image files have three color channels.  The spectrograms stacked with their deltas and delta-deltas also have a final dim of 3.

# In[ ]:


def get_batch(X, y, n_samples, seq_len, moments, sxx_h, sxx_w, dsamp_rate):
    start_list = [rng.randint(0, X.shape[0]-seq_len) for i in range(n_samples)]
    X_batch = []
    for start in start_list:
        sig_in = X[start:start+seq_len]
        _, _, Sxx = spectrogram(sig_in, dsamp_rate)
        
        Sxx3d = get_3d_spec(Sxx, moments)
        X_batch.append(Sxx3d)
    n_channels = X_batch[0].shape[-1]
    X_batch = np.array(X_batch).reshape(n_samples, sxx_h, sxx_w, n_channels)
    y_batch = [y[start+seq_len-1] for start in start_list]
    return X_batch, y_batch


# Nex, build an inner layer for the convnet.  We will use three layers of convolution, followed by a pooling layer.  This inner layer includes options for several regularization and optimization features:
# 
# 1. [Batch normalization](https://arxiv.org/abs/1502.03167).  This aids learning in deeper neural networks.  We will not use batch norm in this kernel.
# 
# 1. [Local response regularization](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) (LRN):  This layer encourages adjacent convolutional kernels to train on different features, but it introduces another set of hyperparameters to be tuned.
# 
# 1. Residual connections: This allows information from the inputs to reach the pooling layer with only one layer of convolution, using a "skip" layer.  The skip layer is added element-wise to the outputs of the final convolutional sub-layer.  This is particularly helpful when there are many iterations of the interior layer.  Residual connections are used in [ResNet](https://arxiv.org/abs/1512.03385) and in many other deep learning models.
# 
# For the purpose of demonstration, we will use LRN on the pooling layers only.  We will also enable residuals.  In practice, these elements  were designed for use with large, deep image classification networks, and may not be suitable for this particular learning task.  In fact, even though the data from the spectrogram is visually informative, its structure is very different from the photographs that convnets are typically trained on.

# In[ ]:



def layer(inputs, filt, kn, stride, act, ptype, psize, pstride, ppad,
          use_lrn_conv, use_lrn_pool, use_bn=False, use_res=False, name=None):
    '''Internal layer composed of 2x conv1d, then pool.
    Args:
        inputs: tf.Tensor rank 4
        filt: int, number of maps to use
        kn: list of tuples (h, w), kernel size
        stride: list of tuples (h, w), stride for each conv layer
        act: tf.nn object or None, type of activation after each conv layer
        ptype: str in {'max', 'avg'}: type of pooling layer
        psize: tuple (h, w), size of pooling layer
        pstride: tuple (h, w), stride of pooling layer
        ppad: str in {'same', 'valid'}, padding type for pooling layer
        use_lrn: bool, optional local response normalization
        use_bn: bool, optional batch norm
        use_res: bool, optional residual connection around conv layers
        name: str, name for the layer

    Returns: tf.Tensor rank 4, layer output
    '''
    with graph.as_default():
        batch_norm = partial(tf.layers.batch_normalization, training=train_flag,
                     momentum=bn_momentum)
        lrn = partial(tf.nn.local_response_normalization, depth_radius=5, bias=1,
                      alpha=1, beta=0.5)

        conv1 = conv2d(inputs, filters=filt, kernel_size=kn[0],
                       strides=stride[0], name=name+'_conv1')
        if use_lrn_conv[0]:
            conv1 = lrn(conv1, name=name+'_conv1_lrn')
        if use_bn: conv1 = batch_norm(conv1, name=name+'_conv1_bn')
        if act is not None:
            conv1 = act(conv1, name=name+'_conv1_act')

        conv2 = conv2d(conv1, filters=filt, kernel_size=kn[1],
                       strides=stride[1], name=name+'_conv2')
        if use_lrn_conv[1]:
            conv2 = lrn(conv2, name=name+'_conv2_lrn')
        if use_bn: conv2 = batch_norm(conv2, name=name+'_conv2_bn')

        if use_res:
            stride_h = np.prod([i[0] for i in stride])
            stride_w = np.prod([i[1] for i in stride])
            kn_h = np.prod([i[0] for i in kn])
            kn_w = np.prod([i[1] for i in kn])
            resid = conv2d(inputs, filters=filt, kernel_size=(kn_h, kn_w),
                           strides=(stride_h, stride_w), name=name+'_skip')

            conv_final = tf.add(conv2, resid, name=name+'_resid')
        else:
            conv_final = tf.identity(conv2, name=name+'_conv_final')

        if act is not None:
            conv_final = act(conv_final, name+'_conv2_act')

        if ptype == 'max':
            pool = tf.layers.max_pooling2d(conv_final, pool_size=psize,
                        strides=pstride, padding=ppad, name=name+'_max_pool')
        elif ptype == 'avg':
            pool = tf.layers.average_pooling2d(conv_final, pool_size=psize,
                        strides=pstride, padding=ppad, name=name+'_avg_pool')
        else:
            pool = tf.identity(conv_final, name=name+'_pool_passthru')
        
        if use_lrn_pool:
            pool = lrn(pool, name=name+'_pool_lrn')

        return pool


# A graph of the internal layer (from Tensorboard) makes this a bit clearer.  The two nodes between each convolutional layer are the LRN layer and the activation:

# In[ ]:


from IPython.display import Image
Image(filename=r'../input/images/graph.JPG', height=300, width=300)


# Next, set hyperparameters for the model.  For information about convnet design, see this [Stanford class](https://cs231n.github.io/convolutional-networks/):  For more about implementing convnets in TensorFlow, see [here](https://www.tensorflow.org/tutorials/estimators/cnn).
# 
# Convolutional neural nets have a lot of hyperparameters.  Kernel, stride, and pool dimensions all have a big impact on the quality of the model.
# 
# We use mean squared error (MSE) as the objective function.  Mean absolute error (MAE) is also an option, but MSE more heavily penalizes predictions that are far from the actual value.  (Symmetric) exponential error is another option, but can lead to overfitting because outliers have an outsized effect on training loss.

# In[ ]:


init_mode = 'FAN_AVG'
init_uniform = True

# convolutional layer(s):
n_layers = 2
filters=[64, 128]
kernels = [[(3, 6), (4, 8)],
           [(3, 6), (4, 8)]
          ]
strides = [[(1, 2), (2, 2)],
           [(1, 2), (2, 2)]
          ]
conv_activation = tf.nn.elu
use_conv_bn = [0, 0]
bn_momentum = 0.9999 # only if use_conv_bn
use_lrn_conv = [[0, 0, 0],
                [0, 0, 0]
               ]
use_lrn_pool = [1, 1]
use_residuals = [1, 1]
ptypes = ['max', 'avg']
ppads = ['valid', 'valid']
pools = [(5, 5,), (5, 5,)]
pstrides = [(1, 1), (1, 1)]

# feedforward layer:
dim_ff = [1024]
ff_activation = tf.nn.relu
drop_rate = [0.4]

# train config:
n_epochs = 2500
eta = 2e-8
train_batch_size = 50
eval_batch_size = 50
n_eval_batches = 10
optimizer_name = 'adam'  # ['sgd', 'adam', 'adagrad', 'adadelta']
objective = 'MSE'  # ['MSE', 'MAE', 'exp_error']
tensorboard_eval_interval = 20
stdout_eval_interval = 100

optimizers = {
    'sgd': tf.train.GradientDescentOptimizer,
    'adam': tf.train.AdamOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'adadelta': tf.train.AdadeltaOptimizer
    }

init = tf.contrib.layers.variance_scaling_initializer(mode=init_mode,
                                                      uniform=init_uniform)


# Now construct the model graph.  This particular model has the following structure:
# * one or more of the inner layers defined above, each of which has 2x convolutional sub-layers followed by a pooling layer.
# * a Flatten() layer to collapse the 2-dimensional outputs from convolution to a 1-d vector
# * a [dropout](http://jmlr.org/papers/v15/srivastava14a.html) layer for regularization
# * 1x feedforward layer
# * a linear layer to produce a scalar prediction for the target value (time to failure).
# 
# This structure is verly loosely based on features of AlexNet and ResNet, and is not tailored to this particular task.

# In[ ]:


try:
    sess.close()
except: pass

global graph
graph = tf.Graph()
tf.reset_default_graph()
with graph.as_default():

    X = tf.placeholder(tf.float32,
                       shape=(None, sxx_h, sxx_w, N_CHANNELS),
                       name='X')
    y = tf.placeholder(tf.float32, shape=(None), name='y')
    train_flag = tf.placeholder_with_default(False, shape=(), name='train_flag')

    conv2d = partial(tf.layers.conv2d, padding='same',
                     activation=conv_activation,
                     kernel_initializer=init)

    lrn = partial(tf.nn.local_response_normalization,
                  depth_radius=5, bias=1, alpha=1, beta=0.5)
    
    # unroll internal (3x conv1d+pool) layers
    layers = [X]
    for i in range(n_layers):
        layers.append(
            layer(layers[i], filt=filters[i], kn=kernels[i], stride=strides[i],
                  act=conv_activation, ptype=ptypes[i], psize=pools[i],
                  pstride=pstrides[i], ppad=ppads[i],
                  use_lrn_conv=use_lrn_conv[i], use_lrn_pool=use_lrn_pool[i],
                  use_bn=use_conv_bn[i], use_res=use_residuals[i], name='L'+str(i+1)))

    # feedforward layer(s)
    dense = partial(tf.layers.dense,
                    activation=ff_activation,
                    kernel_initializer=init)

    dropout = partial(tf.layers.dropout,
                      training=train_flag)
    
    flat = tf.layers.Flatten()(layers[-1])
    ff_layers = [flat]
    for i, width in enumerate(dim_ff):
        drop_layer = dropout(ff_layers[-1], rate=drop_rate[i], name='drop_' + str(i+1))
        ff_layer = dense(drop_layer, width, name='ff_' + str(i+1))
        ff_layers.append(ff_layer)
    outputs = tf.layers.dense(ff_layers[-1], 1, name='outputs')

    MAE = tf.reduce_mean(tf.abs(outputs - y))
    MSE = tf.reduce_mean(tf.square(outputs - y))
    exp_error = tf.reduce_mean(tf.subtract(tf.exp(tf.abs(outputs - y)), 1))

    optimizer_fn = optimizers[optimizer_name]

    if objective == 'MAE':
        training_op = optimizer_fn(learning_rate=eta).minimize(MAE)
    elif objective == 'MSE':
        training_op = optimizer_fn(learning_rate=eta).minimize(MSE)
    elif objective == 'exp_error':
        training_op = optimizer_fn(learning_rate=eta).minimize(exp_error)

    init = tf.global_variables_initializer()


# Train and evaluate the model.

# In[ ]:


sess = tf.InteractiveSession(graph=graph)
init.run(session=sess)
for epoch in tnrange(n_epochs):
    X_batch, y_batch = get_batch(X_train, y_train, train_batch_size, downsampled_seq_len,
                                 moments, sxx_h, sxx_w, DOWNSAMPLE_RATE)
    sess.run(training_op, feed_dict={X: X_batch, y: y_batch, train_flag: True})
    if (epoch+1) % stdout_eval_interval == 0:
        evals_out = {'train': {'mse': [], 'mae': []}, 'eval': {'mse': [], 'mae': []}}
        for i in range(n_eval_batches):
            X_train_batch, y_train_batch = get_batch(X_train, y_train, train_batch_size,
                    downsampled_seq_len, moments, sxx_h, sxx_w, DOWNSAMPLE_RATE)
            evals_out['train']['mse'].append(
                MSE.eval(feed_dict={X: X_train_batch, y: y_train_batch, train_flag:False}))
            evals_out['train']['mae'].append(
                MAE.eval(feed_dict={X: X_train_batch, y: y_train_batch, train_flag:False}))
            X_val_batch, y_val_batch = get_batch(X_val, y_val, eval_batch_size,
                    downsampled_seq_len, moments, sxx_h, sxx_w, DOWNSAMPLE_RATE)
            evals_out['eval']['mse'].append(
                MSE.eval(feed_dict={X: X_val_batch, y: y_val_batch, train_flag:False}))
            evals_out['eval']['mae'].append(
                MAE.eval(feed_dict={X: X_val_batch, y: y_val_batch, train_flag:False}))
        print(f"round {epoch+1} " +
              f"train MSE: {sum(evals_out['train']['mse'])/n_eval_batches:.4f} " +
              f"train MAE: {sum(evals_out['train']['mae'])/n_eval_batches:.4f} " +
              f"val MSE: {sum(evals_out['eval']['mse'])/n_eval_batches:.4f} " +
              f"val MAE: {sum(evals_out['eval']['mae'])/n_eval_batches:.4f} ")


# Uh oh, the model trains for ~1700 its, but then learning plateaus.  A cursory examination of the outputs shows that the model is severaly underfitting.  It's probably worth exploring different convolutional structures that are better suited to this task, plus quite a bit of hyperparameter tuning.

# In[ ]:


test_path = os.path.join(data_path, 'test')
files = os.listdir(test_path)
print('total files', len(files)) # 2624
preds = {'seg_id': [], 'time_to_failure': []}
for fname in tqdm_notebook(files):
    path = os.path.join(test_path, fname)
    obs = np.array(pd.read_csv(path).values)
    data = np.c_[obs, np.zeros(len(obs))]
    x, y = downsample(data, DOWNSAMPLE_RATE)
    _, _, Sxx = spectrogram(x, DOWNSAMPLE_RATE)
    Sxx3d = get_3d_spec(Sxx, moments)
    x_feed = Sxx3d.reshape((1, sxx_h, sxx_w, N_CHANNELS))
    pred = float(outputs.eval(feed_dict={X:x_feed}))
    preds['time_to_failure'].append(pred)
    preds['seg_id'].append(fname.split('.')[0])
preds_df = pd.DataFrame(preds)


# In[ ]:


preds_df.to_csv('submissions.csv', index=False)

