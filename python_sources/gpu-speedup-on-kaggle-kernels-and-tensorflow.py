#!/usr/bin/env python
# coding: utf-8

# # GPU Speedup on Kaggle Kernels and Tensorflow
# 
# This kernel provides an example of running a GPU on Kaggle Kernels. If you fork this kernel and rerun it, it will automatically run in a GPU session. For new kernels, you can flip the GPU from "off" to "on" at any point while you're editing the kernel, using the menu on the right. This will restart your interactive kernel session in a new VM that has a GPU attached. When you press "Commit & Run", the commited version will run on a GPU as well.
# 
# First, we'll import Tensorflow and verify that it finds the GPU.

# In[ ]:


import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))


# Next, we'll time an image filter convolution on the CPU vs. the GPU.

# In[ ]:


config = tf.ConfigProto()

with tf.device('/cpu:0'):
    random_image_cpu = tf.random_normal((100, 100, 100, 3))
    net_cpu = tf.layers.conv2d(random_image_cpu, 32, 7)
    net_cpu = tf.reduce_sum(net_cpu)

with tf.device('/gpu:0'):
    random_image_gpu = tf.random_normal((100, 100, 100, 3))
    net_gpu = tf.layers.conv2d(random_image_gpu, 32, 7)
    net_gpu = tf.reduce_sum(net_gpu)

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

cpu = lambda : sess.run(net_cpu)
gpu = lambda : sess.run(net_gpu)
  
print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))

sess.close()


# If we rerun this, we'll see that the GPU goes even faster.

# In[ ]:


sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

cpu = lambda : sess.run(net_cpu)
gpu = lambda : sess.run(net_gpu)
  
print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))

sess.close()


# In[ ]:




