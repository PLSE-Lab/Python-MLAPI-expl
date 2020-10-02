#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# Input data files are available in the "../input/" directory.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET # for parsing XML
import gc
import math
import shutil
import random
import os


# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard.notebook')
get_ipython().run_line_magic('tensorboard', '--logdir logs')
os.mkdir("/kaggle/working/logs")


# In[ ]:


NUM_EPOCHS = 130
BATCH_SIZE = 32
DIM_X = 64
DIM_Y = 64
DIM_Z = 3
D_LEARNING_RATE = 0.0015
G_LEARNING_RATE = 0.0015
BETA1 = 0.5
Z_NOISE_DIM = 128
F_DIM = 128
FULL_OUTPUT_PATH = "../output_dir/"
os.mkdir(FULL_OUTPUT_PATH)

def image_from_array(image_arr):
    return np.reshape(image_arr, (DIM_X, DIM_Y, DIM_Z))

def transform_image(image):
    # back from -1,1 to 0,1
    image = image / 2 + 0.5
    # from 0,1 to 0,255
    image = image * 255
    return image


# In[ ]:


def random_crop_inside_bbox(image,bbox):
    img = tf.image.crop_to_bounding_box(image, bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0])
    img = tf.image.central_crop(img,0.5)
    imgs = tf.image.resize_bilinear(tf.expand_dims(img, 0),[64,64])
    return tf.squeeze(imgs)


# In[ ]:


def read_image(filename, bbox):
    """
    This function does blah blah.
    """
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    
    # image = tf.image.random_flip_up_down(image)
    # image = random_rotation(image)
    
    #image = tf.image.crop_to_bounding_box(image, bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0])
    #image = tf.image.resize_image_with_pad(image,DIM_X,DIM_Y)
    height = tf.shape(image)[1]
    width = tf.shape(image)[0]
    image = tf.image.crop_and_resize(tf.expand_dims(image, 0), [[bbox[1]/width, bbox[0]/height, bbox[3]/width, bbox[2]/height]],box_ind = [0], crop_size = [64,64])
    image = tf.squeeze(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


# In[ ]:


def prepare_data():
    filenames = []
    labels = []
    bboxes = []
    breeds = os.listdir('../input/annotation/Annotation/')
    for breed in breeds:
        dogs = os.listdir('../input/annotation/Annotation/' + breed + "/")
        for dog in dogs:
            if os.path.exists('../input/all-dogs/all-dogs/' + dog + '.jpg'):
                tree = ET.parse('../input/annotation/Annotation/' + breed + '/' + dog)
                root = tree.getroot()
                objects = root.findall('object')
                for o in objects:
                    bndbox = o.find('bndbox') # reading bound box
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    label = o.find('name').text
                filenames.append('../input/all-dogs/all-dogs/' + dog + '.jpg')
                bboxes.append([xmin, ymin, xmax, ymax])
    return filenames, bboxes


# In[ ]:


def create_dataset(filenames, b_boxes):
    """
    This function creates a tf.data.Dataset from a folder containing images
    """   
    dataset = (tf.data.Dataset.from_tensor_slices((filenames, b_boxes))
                .apply(tf.data.experimental.shuffle_and_repeat(buffer_size=len(filenames), count=NUM_EPOCHS))
                .map(lambda filename, b_box: read_image(
                    filename,b_box), num_parallel_calls=8)
                .batch(BATCH_SIZE)
                .prefetch(1))

    return dataset, len(filenames)


# In[ ]:


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, is_training=train, scope=self.name)

d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')

g_bn0 = batch_norm(name='g_bn0')
g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')
g_bn3 = batch_norm(name='g_bn3')

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[
            1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable(
            'biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])
        biases = tf.get_variable(
            'biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        try:
            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
        except ValueError as err:
            msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
            err.args = err.args + (msg,)
            raise
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def discriminator(x, reuse=False,  decaying_noise=None):
    df_dim = F_DIM
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        x0 = x + decaying_noise
        conv_1 = conv2d(x0, df_dim, name='d_h0_conv')
        
        h0 = lrelu(conv_1)
        h1 = lrelu(d_bn1(conv2d(h0, df_dim*2, name='d_h1_conv')))
        h2 = lrelu(d_bn2(conv2d(h1, df_dim*4, name='d_h2_conv')))
        h3 = lrelu(d_bn3(conv2d(h2, df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [BATCH_SIZE, -1]), 1, 'd_h4_lin')

        return tf.nn.sigmoid(h4), h4


def generator(z):
    gf_dim = F_DIM
    with tf.variable_scope("generator") as scope:
        
        s_h, s_w = DIM_X, DIM_Y
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        z_, h0_w, h0_b = linear(z, gf_dim*8*s_h16*s_w16,
                                'g_h0_lin', with_w=True)
        h0 = tf.reshape(z_, [-1, s_h16, s_w16, gf_dim * 8])
        h0 = tf.nn.relu(g_bn0(h0))
        h1, h1_w, h1_b = deconv2d(
            h0, [BATCH_SIZE, s_h8, s_w8, gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(g_bn1(h1))
        h2, h2_w, h2_b = deconv2d(
            h1, [BATCH_SIZE, s_h4, s_w4, gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(g_bn2(h2))
        h3, h3_w, h3_b = deconv2d(
            h2, [BATCH_SIZE, s_h2, s_w2, gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(g_bn3(h3))
        h4, h4_w, h4_b = deconv2d(
            h3, [BATCH_SIZE, s_h, s_w, DIM_Z], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)


def sampler(z, batch_size=BATCH_SIZE):
    gf_dim = F_DIM
    with tf.variable_scope("generator") as scope:
        scope.reuse_variables()

        s_h, s_w = DIM_X, DIM_Y
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        h0 = tf.reshape(
            linear(z, gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, gf_dim * 8])
        h0 = tf.nn.relu(g_bn0(h0, train=False))

        h1 = deconv2d(h0, [batch_size, s_h8, s_w8, gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(g_bn1(h1, train=False))

        h2 = deconv2d(h1, [batch_size, s_h4, s_w4, gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2, train=False))

        h3 = deconv2d(h2, [batch_size, s_h2, s_w2, gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3, train=False))

        h4 = deconv2d(h3, [batch_size, s_h, s_w, DIM_Z], name='g_h4')

    return tf.nn.tanh(h4)


# In[ ]:


def model_inputs(image_width, image_height, image_channels, z_dim, batch_size = BATCH_SIZE):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """
    real_input_images = tf.placeholder(tf.float32, [BATCH_SIZE] + [image_width, image_height, image_channels], name='real_images')
    input_z = tf.placeholder(tf.float32, [None, z_dim], name='input_z')

    return real_input_images, input_z

def model_loss(input_real, input_z, smooth_factor=0.1, decaying_noise=None):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    fake_samples = generator(input_z)
    tf.summary.image("G", fake_samples, max_outputs=5, collections=["g_imgs"])
    tf.summary.image("D", input_real, max_outputs=5, collections=["d_imgs"])
    d_model_real, d_logits_real = discriminator(input_real, reuse=False, decaying_noise=decaying_noise)
    d_model_fake, d_logits_fake = discriminator(fake_samples, reuse=True, decaying_noise=decaying_noise)
        
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real) * (1 - smooth_factor)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))    
    d_loss = d_loss_real + d_loss_fake
    

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))
    
    
    return d_loss, g_loss

def model_opt(d_loss, g_loss, d_learning_rate, g_learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(d_learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(g_learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    tf.summary.scalar("g_loss", g_loss, collections=["g_summ"])
    tf.summary.scalar("d_loss", d_loss, collections=["d_summ"])
    return d_train_opt, g_train_opt


# In[ ]:


filenames, b_boxes = prepare_data()
dataset, dataset_len = create_dataset(filenames, b_boxes)


# In[ ]:


iterator = dataset.make_initializable_iterator()
global_step = tf.Variable(0, trainable=False, name='global_step')
increment_global_step = tf.assign_add(global_step, 1, name='increment_global_step')

# Model
input_real, input_z = model_inputs(DIM_X, DIM_Y, DIM_Z, Z_NOISE_DIM)
total_steps = (dataset_len / BATCH_SIZE) * NUM_EPOCHS
starter_stdev = 0.1

##decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
decaying_stdev = tf.train.exponential_decay(starter_stdev, global_step, total_steps * 10, 0.0001)

decaying_noise = tf.random_normal(shape=tf.shape(input_real), mean=0.0, stddev=decaying_stdev, dtype=tf.float32)
tf.summary.scalar("stdev", tf.keras.backend.std(decaying_noise), collections=["d_summ"])
d_loss, g_loss = model_loss(input_real, input_z, decaying_noise=decaying_noise)
d_train_opt, g_train_opt = model_opt(d_loss, g_loss, D_LEARNING_RATE, G_LEARNING_RATE, BETA1)

z_batch_tensor = tf.random.uniform((BATCH_SIZE, Z_NOISE_DIM), dtype=tf.float32, minval=-1, maxval=1)


# In[ ]:


sess = tf.Session()


# In[ ]:


steps = 0
sess.run(tf.global_variables_initializer())
sess.run(iterator.initializer)
next_batch = iterator.get_next()

writer = tf.summary.FileWriter("/kaggle/working/logs/", sess.graph)
g_imgs = tf.summary.merge_all(key="g_imgs")
d_imgs = tf.summary.merge_all(key="d_imgs")
g_summ = tf.summary.merge_all(key="g_summ")
d_summ = tf.summary.merge_all(key="d_summ")

while True:
    try:
        steps = sess.run(increment_global_step)
        try:
            batch_imgs = sess.run(next_batch)
        except Exception as e:
            print(e)
            continue
        else:
            if len(batch_imgs) != BATCH_SIZE:
                break
        batch_z = sess.run(z_batch_tensor)

        _, d_loss_sum_str = sess.run([d_train_opt, d_summ], feed_dict={
                                     input_real: batch_imgs, input_z: batch_z})
        writer.add_summary(d_loss_sum_str, steps)

        _, g_sum_str = sess.run([g_train_opt, g_summ], feed_dict={
            input_real: batch_imgs, input_z: batch_z})
        writer.add_summary(g_sum_str, steps)

        _, g_sum_str = sess.run([g_train_opt, g_summ], feed_dict={
            input_real: batch_imgs, input_z: batch_z})

        writer.add_summary(g_sum_str, steps)

        if steps % 20 == 0:
            g_imgs_summ, d_imgs_summ = sess.run([g_imgs,d_imgs], feed_dict={
                               input_z: batch_z, input_real : batch_imgs})
            writer.add_summary(g_imgs_summ, steps)
            writer.add_summary(d_imgs_summ, steps)
            
        if steps % 100 == 0:
            train_loss_d, train_loss_g = sess.run([d_loss, g_loss], feed_dict={
                                                  input_real: batch_imgs, input_z: batch_z})

            print("Step {} de {}".format(steps % int(dataset_len/BATCH_SIZE)+1, int(dataset_len/BATCH_SIZE)+1),
                  "-- Epoch [{} de {}]".format(int(steps *
                                                   BATCH_SIZE/dataset_len), NUM_EPOCHS),
                  "-- Global step {}".format(steps),
                  "-- Discriminator Loss: {:.4f}".format(train_loss_d),
                  "-- Generator Loss: {:.4f}".format(train_loss_g))
        gc.collect()

    except tf.errors.OutOfRangeError:
        print("End")
        break
        


# In[ ]:


def save_images(imgs_batch, curr_batch=1):
    for i, image in enumerate(imgs_batch):
        filepath = FULL_OUTPUT_PATH + "_" + str(curr_batch) + "_" + str(i) + "_" + ".png"
        output_image = Image.fromarray(transform_image(
            image_from_array(image)).astype(np.uint8))
        output_image.save(filepath)

def generate_samples(sess, z_batch_tensor, input_z, num_samples):
    """
    This function does blah blah.
    """    
    num_batches = (num_samples // BATCH_SIZE) + 1
    for batch_count in range(num_batches):
        example_z = sess.run(z_batch_tensor)
        samples = sess.run(sampler(input_z),
                       feed_dict={input_z: example_z})
        imgs = [img[:, :, :] for img in samples]
        save_images(imgs, batch_count)


# In[ ]:


generate_samples(sess,z_batch_tensor, input_z, 10000)


# In[ ]:


path, dirs, files = next(os.walk(FULL_OUTPUT_PATH))
exceed = len(files) - 10000
i = 0
for file in os.listdir(FULL_OUTPUT_PATH):
    if i < exceed:
        os.remove(FULL_OUTPUT_PATH + file)
        i += 1


# In[ ]:


path, dirs, files = next(os.walk(FULL_OUTPUT_PATH))
exceed = len(files) - 10000
print(exceed)
shutil.make_archive('images', 'zip', FULL_OUTPUT_PATH)

