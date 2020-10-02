#!/usr/bin/env python
# coding: utf-8

# # Motivation
# The motivation for putting this together was [@JoelGru's](https://twitter.com/joelgrus) presentation called ["I don't like notebooks"](https://conferences.oreilly.com/jupyter/jup-ny/public/schedule/detail/68282) at JupyterCon. He made a number of very good points and one of the most critical ones was about bad habits and lack of testing. I attempt to address this in this work
# 
# ## Overview
# The idea for this document is to provide a practical introduction to using doctests (having tests inside the docstring). It is not meant to replace the [main documentation](https://docs.python.org/3/library/doctest.html) but rather to show data-scientists and deep-learning people how to use it effectively. If you have any changes or suggestions please make comments to the document or reach out on Kaggle
# 
# 

# #### Getting Started
# 
# Interactive `doctest` snippet. The snippet below lets us use a simple annotation (`@autotest`) above any function to run the doctest immediately. In real projects you can run doctest seperately using `doctest` or as part of `py.test` using the correct `setup.cfg` file or command line arguments. You will thus not need the annotation here. For notebooks the annotation makes it much easier to build tests on the fly

# In[ ]:


# tests help notebooks stay managable
import doctest
import copy
import functools

def autotest(func):
    globs = copy.copy(globals())
    globs.update({func.__name__: func})
    doctest.run_docstring_examples(
        func, globs, verbose=True, name=func.__name__)
    return func


# # Simple Examples
# Just to get started with doctest and how it generally works

# ## Trivial Sample / Intro
# no >>> code means nothing actually happens

# In[ ]:


@autotest
def add_1(x):
  """Add 1 to a value"""
  return x+1


# we can add one test by adding the `>>> add_1(4)` line, but we see that it fails because we don't specify any output

# In[ ]:


@autotest
def add_1(x):
  """Add 1 to a value
  >>> add_1(4)
  """
  return x+1


# now everything works

# In[ ]:


@autotest
def add_1(x):
  """Add 1 to a value
  >>> add_1(4)
  5
  """
  return x+1


# You can even check exceptions by using traceback, elipses and then the error type

# In[ ]:


@autotest
def add_1(x):
  """Add 1 to a value.
  :param x: a number to add 1 to
  :return: the number plus 1
  
  >>> add_1(4)
  5
  >>> add_1("bob")
  Traceback (most recent call last):
       ...
  TypeError: must be str, not int
  """
  return x+1


# The largest benefit of doctests is when using the `help` or `?` tools inside IPython/Jupyter you can see all of the tests, results and along with the general function info. This makes it substantially easier to figure out how to use a function since ready-to-go examples have already been prepared.

# In[ ]:


help(add_1)


# ## Real Sample
# 
# Here is a slightly more complicated real sample (requires webcolors library) which finds the closest color to a given entry

# In[ ]:


get_ipython().system('pip install -q webcolors')
import webcolors
@autotest
def closest_color(requested_color):
    # type: (Tuple[float, float, float]) -> str
    """
    Finds the closest color to a given item
    :param requested_color: r,g,b values for a color
    :return: name of the closest color
    
    Examples:
    >>> closest_color((128, 0, 255))
    'darkviolet'
    >>> closest_color((255, 128, 128))
    'salmon'
    >>> closest_color([0])
    Traceback (most recent call last):
        ...
    ValueError: Invalid size
    """
    if len(requested_color)!=3:
        raise ValueError('Invalid size')
    min_colors = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


# ## More complicated exampe - Try
# Here is a more complicated example where we make and test another decorator which trys a function and returns a default value if it fails. This is particularly useful for parsing messy data where some columns that should contain numbers contain bizzare strings like (empty or just "")

# In[ ]:


from warnings import warn
@autotest
def try_default(errors=(Exception,), default_value='', verbose=False,
                warning=False):
    """
    A decorator for making failing functions easier to use
    :param errors:
    :param default_value:
    :param verbose:
    :return:
    >>> int('a')
    Traceback (most recent call last):
         ...
    ValueError: invalid literal for int() with base 10: 'a'
    >>> safe_int = try_default(default_value=-1, verbose=True, warning=False)(int)
    >>> safe_int('a')
    Failed calling int with :('a',),{}, because of ValueError("invalid literal for int() with base 10: 'a'",)
    -1
    >>> quiet_int = try_default(default_value=0, verbose=False, warning=False)(int)
    >>> quiet_int('a')
    0
    """

    def decorator(func):
        def new_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except errors as e:
                if verbose:
                    out_msg = "Failed calling {} with :{},{}, because of {}".format(
                        func.__name__,
                        args, kwargs, repr(e))
                    if warning:
                        warn(out_msg, RuntimeWarning)
                    else:
                        print(out_msg)
                return default_value

        return new_func

    return decorator


# # Matrix Examples
# Since matrices, arrays, tensors, ... come up quite frequently in real-world python use-cases, we show some examples of testing matrix code even though the output is often ugly

# In[ ]:


import numpy as np
# make outputs look nicer
pprint = lambda x, p=2: print(np.array_str(x, max_line_width=80, precision=
p))


# ## In-ND
# The `np.in1d` function in numpy works quite well to see which elements in a 1D array ($x$) are inside a list ($y$). The result is a boolean array the same size as the input where each element is if the item at that index in $x$ was in $y$. The function below works on ND arrays.

# In[ ]:


@autotest
def in_nd(x, y):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """
    A simple wrapper for the in1d function to work on ND data
    :param x:
    :param y:
    :return:
    >>> t_img = np.arange(6).reshape((2,3))
    >>> pprint(t_img)
    [[0 1 2]
     [3 4 5]]
    >>> pprint(in_nd(t_img, [4,5]))
    [[False False False]
     [False  True  True]]
    """
    return np.in1d(x.ravel(), y).reshape(x.shape)


# ## Meshgrid-like
# Here is a simple function which makes a meshgrid with the same dimensions as the input

# In[ ]:


@autotest
def meshgridnd_like(in_img,
                    rng_func=range):
    """
    Makes a n-d meshgrid in the shape of the input image.
    
    >>> import numpy as np
    >>> xx, yy = meshgridnd_like(np.ones((3,2)))
    >>> xx.shape
    (3, 2)
    >>> xx
    array([[0, 0],
           [1, 1],
           [2, 2]])
    >>> xx[:,0]
    array([0, 1, 2])
    >>> yy
    array([[0, 1],
           [0, 1],
           [0, 1]])
    >>> yy[0,:]
    array([0, 1])
    >>> xx, yy, zz = meshgridnd_like(np.ones((2,3,4)))
    >>> xx.shape
    (2, 3, 4)
    >>> xx[:,0,0]
    array([0, 1])
    >>> yy[0,:,0]
    array([0, 1, 2])
    >>> zz[0,0,:]
    array([0, 1, 2, 3])
    >>> zz.astype(int)
    array([[[0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3]],
    <BLANKLINE>
           [[0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3]]])
    """
    new_shape = list(in_img.shape)
    all_range = [rng_func(i_len) for i_len in new_shape]
    return tuple([x_arr.swapaxes(0, 1) for x_arr in np.meshgrid(*all_range)])


# ## Difference Padding
# Here we want to take the difference along a dimension and then pad it so that it retains the same dimensions

# In[ ]:


@autotest
def diffpad(in_x, 
            n=1,
            axis=0,
            starting_value=0):
  """
  Run diff and pad the results to keep the same 
  If the starting_value is the same then np.cumsum should exactly undo diffpad
  >>> diffpad([1, 2, 3], axis=0)
  array([0, 1, 1])
  >>> np.cumsum(diffpad([1, 2, 3], axis=0, starting_value=1))
  array([1, 2, 3])
  >>> diffpad(np.cumsum([0, 1, 2]), axis=0)
  array([0, 1, 2])
  >>> diffpad(np.eye(3), axis=0)
  array([[ 0.,  0.,  0.],
         [-1.,  1.,  0.],
         [ 0., -1.,  1.]])
  >>> diffpad(np.eye(3), axis=1)
  array([[ 0., -1.,  0.],
         [ 0.,  1., -1.],
         [ 0.,  0.,  1.]])
  """
  if axis<0: 
    raise ValueError("Axis must be nonneggative")
  d_x = np.diff(in_x, n=n, axis=axis)
  return np.pad(d_x, [(n, 0) if i==axis else (0,0) 
                      for i, _ in enumerate(np.shape(in_x))], 
                mode='constant', constant_values=starting_value)


# ## Numpy Aware JSON Encoder
# JSON typically chokes on numpy arrays since it does not have a default encoder for them (and many arrays would be massive if just naively converted to JSON). Here we implement the naive conversion for exporting small arrays. We see here that it also works on classes

# In[ ]:


import json
@autotest
class NumpyAwareJSONEncoder(json.JSONEncoder):
    """
    A JSON plugin that allows numpy data to be serialized 
    correctly (if inefficiently)
    >>> json.dumps(np.eye(3))
    Traceback (most recent call last):
        ...
    TypeError: Object of type 'ndarray' is not JSON serializable
    >>> json.dumps(np.eye(3).astype(int), cls=NumpyAwareJSONEncoder)
    '[[1, 0, 0], [0, 1, 0], [0, 0, 1]]'
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):  # and obj.ndim == 1:
            return obj.tolist()
        if isinstance(obj, np.number):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
  


# # Deep Learning Examples
# Deep learning can be a bit messier and require graphs, setups, CPU/GPU configs and so is often less well suited to doctest-style testing. With a few simple helper functions it can be made easier and so we show a few examples below

# In[ ]:


from keras import models, layers
import tensorflow as tf
def _setup_and_test(in_func, *in_arrs, is_list=False, round=False):
    """
    For setting up a simple graph and testing it
    :param in_func:
    :param in_arrs:
    :param is_list:
    :return:
    """
    with tf.Graph().as_default() as g:
        in_vals = [tf.placeholder(dtype=tf.float32, shape=in_arr.shape) for
                   in_arr in in_arrs]
        out_val = in_func(*in_vals)
        if not is_list:
            print('setup_net', [in_arr.shape for in_arr in in_arrs],
                  out_val.shape)
            out_list = [out_val]
        else:
            out_list = list(out_val)
    with tf.Session(graph=g) as c_sess:
        sess_out = c_sess.run(fetches=out_list,
                              feed_dict={in_val: in_arr
                                         for in_val, in_arr in
                                         zip(in_vals, in_arrs)})
        if is_list:
            o_val = sess_out
        else:
            o_val = sess_out[0]
        if round:
            return (np.array(o_val) * 100).astype(int) / 100
        else:
            return o_val


# ## Spatial Gradient 2D
# The function serves to add the equivalent of the `numpy.gradient` command to tensorflow. The edges are different but the results are largely the same.

# In[ ]:


@autotest
def spatial_gradient_2d_tf(in_img):
    # type: (tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    """
    Calculate the 2d spatial gradient in x,y using tensorflow
    The channel dimension is completely ignored and the batches are kept
    consistent.
    :param in_img: a 4d tensor sized as batch, x, y, channel
    :return:
    NOTE:: the doctests are written to only test the main region, due to
    boundary issues the edges are different (not massively) between
    np.gradient and this function, eventually a better edge scaling should
    be implemented
    >>> _testimg = np.ones((4, 4))
    >>> _testimg = np.sum(np.power(np.stack(meshgridnd_like(_testimg),-1),2),-1)
    >>> _testimg = np.expand_dims(np.expand_dims(_testimg,0),-1)
    >>> dx, dy = _setup_and_test(spatial_gradient_2d_tf, _testimg, is_list=True)
    >>> dx.shape, dy.shape
    ((1, 4, 4, 1), (1, 4, 4, 1))
    >>> ndx, ndy = np.gradient(_testimg[0,:,:,0])
    >>> (ndx.shape, ndy.shape)
    ((4, 4), (4, 4))
    >>> [(a,b) for a,b in zip(ndx[:,0],dx[0,1:-1,0,0])]
    [(1.0, 2.0), (2.0, 4.0)]
    >>> [(a,b) for a,b in zip(ndy[0,:],dy[0,0,1:-1,0])]
    [(1.0, 2.0), (2.0, 4.0)]
    >>> np.sum(ndx-dx[0,:,:,0],(1))
    array([ 2.,  0.,  0., 10.])
    >>> np.sum(ndy-dy[0,:,:,0], (0))
    array([ 2.,  0.,  0., 10.])
    """
    with tf.variable_scope('spatial_gradient_2d'):
        pad_r = tf.pad(in_img, [[0, 0], [1, 1], [1, 1], [0, 0]],
                       "SYMMETRIC")
        dx_img = pad_r[:, 2:, 1:-1, :] - pad_r[:, 0:-2, 1:-1, :]
        dy_img = pad_r[:, 1:-1, 2:, :] - pad_r[:, 1:-1, 0:-2, :]
        return (0.5 * dx_img, 0.5 * dy_img)


# ## Create a layer which takes the difference of the input
# We use a hard-coded convolution weight and disable `.trainable`

# In[ ]:


@autotest
def diff_filter_layer(length=2, depth=1):
  """Calculates difference of input in 1D.
  
  >>> c_layer = diff_filter_layer()
  >>> t_in = layers.Input((5, 1))
  >>> t_out = c_layer(t_in)
  >>> t_model = models.Model(inputs=[t_in], outputs=[t_out])
  >>> t_model.predict(np.ones((1, 5, 1)))[0, :, 0]
  array([0., 0., 0., 0., 0.], dtype=float32)
  >>> t_model.predict(np.arange(5).reshape((1, 5, 1)))[0, :, 0]
  array([0., 1., 1., 1., 1.], dtype=float32)
  >>> c_layer_3d = diff_filter_layer(depth=3)
  >>> t_in_3d = layers.Input((4, 3))
  >>> t_out_3d = c_layer_3d(t_in_3d)
  >>> t_model_3d = models.Model(inputs=[t_in_3d], outputs=[t_out_3d])
  >>> t_model_3d.predict(np.ones((1, 4, 3)))[0, :, 0]
  array([0., 0., 0., 0.], dtype=float32)
  >>> fake_in = np.arange(12).reshape((1, 4, 3))
  >>> fake_in[0]
  array([[ 0,  1,  2],
         [ 3,  4,  5],
         [ 6,  7,  8],
         [ 9, 10, 11]])
  >>> t_model_3d.predict(fake_in)[0]
  array([[0., 0., 0.],
         [3., 3., 3.],
         [3., 3., 3.],
         [3., 3., 3.]], dtype=float32)
  """
  coef = np.zeros((length, depth, depth))
  i=length//2-1 # offset in middle
  for j in range(depth):
    coef[i,j,j] = -1
    coef[i+1,j,j] = 1
  c_layer = layers.Conv1D(depth, 
                          (coef.shape[0],), 
                          weights=[coef],
                          use_bias=False,
                          activation='linear',
                          padding='valid',
                          name='diff'
               )
  c_layer.trainable = False
  def _diff_module(x):
    diff_x = c_layer(x)
    needed_padding = length-1
    right_pad = np.clip(needed_padding//2-1, 0, 1e99).astype(int)
    left_pad = needed_padding-right_pad
    return layers.ZeroPadding1D((left_pad, right_pad), name='PaddingDiffEdges')(diff_x)
  return _diff_module


# ## Overfit Model on One Dataset
# One of the points Andre Karpathy makes in his [recipe blog post](http://karpathy.github.io/2019/04/25/recipe/) is your model should be able to overfit on a single test batch. This can also be incorporated as a unit test for the model which runs to make sure the model, loss, and training are properly implemented. We show here the model overfitting on a random batch over a -5 and 5 range.

# In[ ]:


from keras import models, layers
@autotest
def build_counting_model(
    input_shape, 
    layer_count=2,
    depth=3,
    max_output=5
    ):
  """Builds a regression model that counts things.
  >>> basic_shape = (9, 9, 1)
  >>> tf.random.set_random_seed(0)
  >>> np.random.seed(0)
  >>> my_model = build_counting_model(basic_shape)
  >>> len(my_model.layers)
  6
  >>> my_model.compile(optimizer='adam', loss='mse')
  >>> x_data = np.random.uniform(size=(5,)+basic_shape)
  >>> y_data = np.linspace(-4, 4, 5)
  >>> _ = my_model.fit(x_data, y_data, epochs=100, verbose=False)
  >>> my_model.evaluate(x_data, y_data, verbose=False)<0.1
  True
  """
  cnt_model = models.Sequential()
  cnt_model.add(layers.BatchNormalization(input_shape=input_shape))
  for i in range(layer_count):
    cnt_model.add(layers.Conv2D(depth*2**i, kernel_size=(3, 3), padding='same'))
  cnt_model.add(layers.Flatten())
  cnt_model.add(layers.Dense(2*max_output, activation='tanh'))
  cnt_model.add(layers.Dense(1, activation='linear', use_bias=False))
  return cnt_model
  


# # Scientific Computing Examples
# Here are a few general scientific computing examples

# ## Get Local Maxima
# Here we have a function to get the local maxima and ignore multiple values above a threshold

# In[ ]:


from scipy.signal import argrelextrema
@autotest
def get_local_maxi(s_vec, 
                   jitter_amount=1e-5, 
                   min_width=5, 
                   cutoff=None # type: Optional[float]
                  ): 
  # type: (...) -> List
  """Get the local maximums.
  
  The standard functions struggle with flat peaks
  
  >>> np.random.seed(2019)
  >>> get_local_maxi([0, 1, 1, 0])
  array([2])
  >>> get_local_maxi([1, 1, 1, 0])
  array([0])
  >>> get_local_maxi([1, 1, 1, 0, 1])
  array([0])
  >>> get_local_maxi([1, 0, 0, 0, 0, 1], min_width=1)
  array([0, 2, 5])
  >>> get_local_maxi([1, 0, 0, 0, 0, 1], min_width=1, cutoff=0.5)
  array([0, 5])
  """
  s_range = np.max(s_vec)-np.min(s_vec)
  if s_range==0:
    return []
  
  j_vec = np.array(s_vec)+    jitter_amount*s_range*np.random.uniform(-1, 1, size=np.shape(s_vec))
  max_idx = argrelextrema(j_vec, np.greater_equal, order=min_width)[0]
  if cutoff is not None:
    max_idx = np.array([k for k in max_idx if s_vec[k]>cutoff])
  return max_idx


# ## Add a wobble to a signal
# Here we add a wobble to a signal

# In[ ]:


from scipy.signal import detrend
@autotest
def scale_wobble(
    in_vec, # type: np.ndarray
    wobble_scale, # type: float
    axis=0
):
  # type: (...) -> np.ndarray
  """Scale the amount of bobble in headpose.
  
  :param wobble_scale: scalar for how much to multiply the amplitude
  >>> scale_wobble([0, 1, 0, 1, 0], 2)
  array([-0.4,  1.6, -0.4,  1.6, -0.4])
  """
  amp_vec = detrend(in_vec, axis=axis)
  trend_vec = in_vec-amp_vec
  return amp_vec*wobble_scale+trend_vec


# ## Perform a dilation on an labelmap
# Taken a labeled image (the value of a pixel is a label) and expand each region until the entire image is filled

# In[ ]:


from scipy.ndimage import grey_dilation
@autotest
def idx_dilation(
    in_img,
    in_mask=None,
    step_size=(3, 3),
    max_iter=100,
    pre_erode=None,  # type: Optional[Tuple[int, int]]
):
    """Index based dilation
    >>> idx_dilation(np.diag(range(3)))
    array([[1, 1, 1],
           [1, 1, 2],
           [1, 2, 2]])
    >>> idx_dilation(np.diag(range(5)))
    array([[1, 1, 1, 2, 2],
           [1, 1, 2, 2, 3],
           [1, 2, 2, 3, 3],
           [2, 2, 3, 3, 4],
           [2, 3, 3, 4, 4]])
    """
    if in_mask is None:
        in_mask = np.ones_like(in_img)
    cur_img = in_img.astype(int)

    if pre_erode is not None:
        cur_img = grey_erosion(cur_img, pre_erode)

    out_img = cur_img.copy()
    matches = False
    iter_cnt = 0
    while not matches:
        out_img = grey_dilation(cur_img, step_size) * in_mask
        out_img[cur_img > 0] = cur_img[
            cur_img > 0
        ]  # preserve the last labels where necessary
        matches = np.sum(np.abs(out_img - cur_img)) < 1
        cur_img = out_img
        iter_cnt += 1
        if iter_cnt > max_iter:
            print("Did not converge")
            print(cur_img)
            print(out_img)
            matches = True
    return out_img


# In[ ]:




