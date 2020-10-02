#!/usr/bin/env python
# coding: utf-8

# # Introduction #
# 
# [TensorFlow Datasets](https://www.tensorflow.org/datasets/overview) (TFDS) is a library implementing the [Extract, Transform, Load](https://en.wikipedia.org/wiki/Extract%2C_transform%2C_load) process for Tensorflow. It contains utilities to assist in downloading and preparing data for use with [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) pipelines. When prepared with TFDS, it is easy to use the dataset with TensorFlow models on Cloud TPUs.
# 
# 
# # Preparing a Dataset for Kaggle #
# 
# We'll create the [TensorFlow Flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers) dataset. You can see the complete catalog [here](https://www.tensorflow.org/datasets/catalog/overview), or in a Python REPL run `tfds.list_builders()`.
# 
# At the command line:
# 
# ```sh
# pip install --upgrade tensorflow-datasets # if it's not installed already
# cd /home/datascientist/ # decide where to create your data directory
# mkdir datasets # name doesn't matter
# cd datasets
# ipython
# ```
# 
# In the Python REPL:
# 
# ```python
# import tensorflow_datasets as tfds
# builder = tfds.builder('tf_flowers', data_dir='/home/datascientist/datasets')
# builder.download_and_prepare() # this can take a few minutes
# exit
# ```
# 
# Back at the command line:
# 
# ```sh
# mkdir temp
# mv tf_flowers temp
# mv temp tf_flowers
# zip -r tf_flowers.zip tf_flowers
# ```
# 
# Now you can upload the zip file to Kaggle!
