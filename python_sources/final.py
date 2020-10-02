import numpy as np
import cv2
from tqdm import tqdm
import os
import pandas as pd
from multiprocessing import Pool
from functools import partial
import os
import gc
from sklearn.model_selection import train_test_split
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import time
from pympler import asizeof
from skimage import feature

from platform import python_version
print(python_version())

import os
import multiprocessing

mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  # e.g. 4015976448
mem_gib = mem_bytes/(1024.**3)  # e.g. 3.74
print("RAM: %f GB" % mem_gib)
print("CORES: %d" % multiprocessing.cpu_count())



def print_progress(it, mIoU, loss, t_v):
    # Calculate the accuracy on the training-set.
    now = time.strftime("%c")
    print("Iteration " + str(it) + " --- mIoU: " + str(mIoU) + " --- Loss: " + str(loss) + " --- " + t_v + " " + now);


def stretch_contrast(image, max_val=255.0, p=0.5):
    o_min = 1e-6
    i_min = np.percentile(image, p)

    o_max = max_val - 1e-6
    i_max = np.percentile(image, 100 - p)

    out = image - i_min
    out *= o_max / (i_max - i_min)
    out[out < o_min] = o_min
    out[out > o_max] = o_max

    return out.astype(np.uint8)


def conv_layer(input, filters, kernel_size, strides, k_init=None, k_reg=None, training=True, activation=None):
    l = tf.layers.conv2d(inputs=input, filters=filters, kernel_size=kernel_size, strides=strides,
                         kernel_initializer=k_init, kernel_regularizer=k_reg, padding='same', trainable=True);

    l = tf.layers.batch_normalization(l, training=training);

    if activation == tf.nn.relu:
        l = tf.nn.relu(l);

    return l;


def convt_layer(input, filters, kernel_size, strides, k_init=None, k_reg=None, training=True, activation=None):
    l = tf.layers.conv2d_transpose(inputs=input, filters=filters, kernel_size=kernel_size, strides=strides,
                                   activation=activation, kernel_initializer=k_init, kernel_regularizer=k_reg,
                                   padding='same', trainable=True);

    l = tf.layers.batch_normalization(l, training=training);

    if activation == tf.nn.relu:
        l = tf.nn.relu(l);

    return l;


def resnet_simple_block(block_input, filters=16, k_init=None, k_reg=None, training=True):
    # print("Resnet block");
    l = block_input;
    l = conv_layer(l, filters, kernel_size=[1, 1], strides=1, k_init=k_init, k_reg=k_reg, activation=tf.nn.relu,
                   training=training);
    l = conv_layer(l, filters, kernel_size=[3, 1], strides=1, k_init=k_init, k_reg=k_reg, activation=tf.nn.relu,
                   training=training);
    l = conv_layer(l, filters, kernel_size=[1, 3], strides=1, k_init=k_init, k_reg=k_reg, activation=tf.nn.relu,
                   training=training);
    l = conv_layer(l, filters, kernel_size=[1, 1], strides=1, k_init=k_init, k_reg=k_reg, training=training);
    res = tf.concat([block_input, l], axis=3);
    return res;


def unet(x, enc_len=5, k_init=None, k_reg=None, training=True):
    enc = [None for i in range(enc_len)];
    dec = [None for i in range(enc_len)];

    enc_ = x;

    with tf.variable_scope('unet'):
        with tf.variable_scope('encoder'):
            for i in range(enc_len):
                enc[i] = resnet_simple_block(enc_, k_init=k_init, k_reg=k_reg, training=training);
                enc_ = tf.layers.max_pooling2d(inputs=enc[i], pool_size=[2, 2], strides=2);
                enc_ = tf.nn.relu(enc_);

        dec_ = enc_;

        with tf.variable_scope('decoder'):
            for i in range(enc_len):
                dec[i] = convt_layer(input=dec_, filters=16, kernel_size=[2, 2], strides=2, k_init=k_init, k_reg=k_reg,
                                     training=training);
                dec_ = tf.concat([dec[i], enc[enc_len - 1 - i]], axis=3);
                dec_ = resnet_simple_block(dec_, k_init=k_init, k_reg=k_reg, training=training);
                dec_ = tf.nn.relu(dec_);

    return dec_;


def transform(x, y):
    x_fl = tf.image.flip_left_right(x);
    y_fl = tf.image.flip_left_right(y);

    x_ud = tf.image.flip_up_down(x);
    y_ud = tf.image.flip_up_down(y);

    # x_rot90 = tf.image.rot90(x);
    # y_rot90 = tf.image.rot90(y);

    x_ = tf.concat([x, x_fl, x_ud], axis=0);
    y_ = tf.concat([y, y_fl, y_ud], axis=0);

    return x_, y_;


def no_transform(x_, y_true_norm):
    return x_, y_true_norm;


def add_features(x, f):
    #
    x_g1 = tf.image.adjust_gamma(image=x, gamma=1.15);
    x_g2 = tf.image.adjust_gamma(image=x, gamma=0.85);
    x_c1 = tf.image.adjust_contrast(images=x, contrast_factor=0.15);
    f = tf.concat([f, x_g1, x_g2, x_c1], axis=3);
    return f;


def dice_coef(y_true, y_pred):
    y_true_f = tf.layers.flatten(y_true)
    y_pred_f = tf.layers.flatten(y_pred)
    smooth = tf.constant(0.00001)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred, numLabels=5):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return tf.constant(1.) - dice/numLabels


def get_images_names(base_dir):
    # annotations
    annot_folder = 'annotations'

    entries = os.listdir(base_dir)
    ds_folders = [item for item in entries if os.path.isdir(os.path.join(base_dir, item))]

    mask_files = []

    for i, ds in enumerate(ds_folders, 1):
        # print("Importing images in folder " + ds)
        mask_list = os.listdir(os.path.join(base_dir, ds, annot_folder))
        mask_list = [os.path.join(base_dir, ds, annot_folder, m) for m in mask_list]

        if i == 1:
            mask_files.extend(mask_list)
        else:
            mask_files.extend(mask_list)

    mask_files = [m for m in mask_files if Path(m).is_file()]

    # image data
    rgb_folder = 'images/rgb'
    nir_folder = 'images/nir'
    # annotations
    annot_folder = 'annotations'

    rgb = []
    mask = []
    nir = []

    for m in mask_files:
        head, image_name = os.path.split(m)
        head, _ = os.path.split(head)

        rgb.append(os.path.join(head, rgb_folder, image_name))
        nir.append(os.path.join(head, nir_folder, image_name))
        mask.append(os.path.join(head, annot_folder, image_name))

    return rgb, mask, nir


def stretch_contrast(im,axis):
    im_min = tf.cast(tf.argmin(im,dimension=2), tf.float32)
    im = tf.subtract(im,im_min)
    im_max = tf.cast(tf.argmax(im,dimension=2), tf.float32)
    im_scale = tf.div(tf.constant(255.0),im_max)
    im = tf.multiply(im,im_scale)
    return im;
    
    
def normalize_tensor(im):
    im_min = tf.cast(tf.reduce_min(im), tf.float32)
    im = tf.subtract(im,im_min)
    im_max = tf.cast(tf.reduce_max(im), tf.float32)
    im = tf.div(im,im_max)
    return im
    
    
def mapping(x, y, nir):
    x_tensor = tf.div(tf.cast(tf.image.decode_png(tf.read_file(x), channels=3),tf.float32), 255.)
    y_tensor = tf.cast(tf.div(tf.image.decode_png(tf.read_file(y), channels=3), 255), tf.uint8)
    nir_tensor = tf.div(tf.cast(tf.image.decode_png(tf.read_file(nir), channels=1),tf.float32), 255.)
    
    r_tensor = tf.slice(x_tensor,[0,0,0],[256,336,1])
    
    nvdi_tensor = normalize_tensor((nir_tensor - r_tensor) / (nir_tensor+r_tensor))
    #x_tensor = tf.image.adjust_contrast(x_tensor,contrast_factor=2)
    #nir_tensor = tf.image.adjust_contrast(nir_tensor,contrast_factor=2)
    #x_tensor = stretch_contrast(x_tensor,0)
    print(r_tensor.shape)
    x_tensor = tf.concat([x_tensor, nir_tensor, nvdi_tensor], axis=2)
    print(x_tensor.shape)
    return x_tensor, y_tensor


def main():
    batch_size = 5
    learning_rate = .001
    lr_decay = .999
    reg = .0    
    kver = 158
    inputs = 'rgb'
    debug = True
    
    print("Loading images...")

    x, y, nir = get_images_names(
        '../input/ijrr_sugarbeets_2016_annotations__336_256/ijrr_sugarbeets_2016_annotations__336_256')

    x_print = x[-10:]
    y_print = y[-10:]
    nir_print = nir[-10:0]
    
    i = 0
    for im in x_print:
        temp = cv2.imread(im)
        cv2.imwrite("{0}_x.jpg".format(i), temp)
        i = i + 1
    
    i = 0
    for im in y_print:
        temp = cv2.imread(im)
        cv2.imwrite("{0}_y.jpg".format(i), temp)
        i = i + 1
        
    i = 0
    for im in nir_print:
        temp = cv2.imread(im,cv2.IMREAD_GRAYSCALE)
        cv2.imwrite("{0}_nir.jpg".format(i), temp)
        i = i + 1
    
    with open("results.txt", "w+") as f:
        f.write('Test results: \n')
        
    exit()
    
    if debug == True:
        max_epochs = 1
        enc_layers = 2
        x_test = x[-10:]
        y_test = y[-10:]
        nir_test = nir[-10:]
        x = x[:50]
        y = y[:50]
        nir = nir[:50]
    else:
        max_epochs = 30
        enc_layers = 4
        x_test = x[-1200:]
        y_test = y[-1200:]
        nir_test = nir[-1200:]
        x = x[:12330 - 1200]
        y = y[:12330 - 1200]
        nir = nir[:12330 - 1200]
    
    
    '''
    Building the computational graph
    '''

    tf.reset_default_graph();
    tf.logging.set_verbosity(tf.logging.ERROR)

    # create the training, validation and test datasets
    train_set = tf.data.Dataset.from_tensor_slices((x, y, nir)). \
        map(mapping, num_parallel_calls=tf.constant(2)). \
        batch(batch_size)
        #prefetch(1). \
        #cache()

    test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test, nir_test)). \
        map(mapping, num_parallel_calls=tf.constant(2)). \
        batch(batch_size)
        #prefetch(1). \
        #cache()

    # create a iterator of the correct shape and type
    iterator = tf.data.Iterator.from_structure(train_set.output_types, train_set.output_shapes)

    # create the initialisation operations
    train_init_op = iterator.make_initializer(train_set)
    test_init_op = iterator.make_initializer(test_set)

    next_element = iterator.get_next()

    x = next_element[0]
    y = next_element[1]
    
    # inputs
    with tf.variable_scope('input'):
        with tf.device('/cpu:0'):
            lr = tf.placeholder(tf.float32, name='learning_rate');
            is_train = tf.placeholder(tf.bool, name="is_train");

    # tf.summary.image("input_x", x);
    # tf.summary.image("input_y", y_true);

    y_true_labels = tf.reshape(tf.argmax(y, axis=3), [-1, 256, 336, 1])

    xavier = tf.contrib.layers.xavier_initializer();

    reg_regr = tf.contrib.layers.l2_regularizer(scale=reg)

    with tf.device('/gpu:0'):
        out = unet(x=x, enc_len=enc_layers, k_init=xavier, k_reg=reg_regr, training=is_train)

    with tf.variable_scope('output'):
        y_pred = conv_layer(input=out, filters=3, kernel_size=(1, 1), strides=(1, 1), k_init=xavier, k_reg=reg_regr,
                            training=is_train);
        y_pred = tf.nn.softmax(y_pred, axis=3);

        y_pred_th = tf.reshape(tf.argmax(y_pred, axis=3), [-1, 256, 336, 1]);
        # tf.summary.image("output", y_pred_th);

    with tf.variable_scope('loss'):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y_true_labels,
                                                      logits=y_pred);
        #loss = dice_loss(tf.cast(y_true_labels, tf.float32), tf.cast(y_pred_th, tf.float32));
        dice = dice_coef_multilabel(tf.cast(y, tf.float32),tf.cast(y_pred, tf.float32),3)
        loss = loss + dice;
        tf.summary.scalar("loss", loss);

    # print("Trainable encoder variables: " + str(len(tf.trainable_variables(scope='encoder'))))
    # print("Trainable decoder variables: " + str(len(tf.trainable_variables(scope='decoder'))))
    # print("Trainable output variables: " + str(len(tf.trainable_variables(scope='output'))))

    with tf.variable_scope('metrics'):
        IoU, IoU_op = tf.metrics.mean_iou(labels=tf.cast(y_true_labels, tf.int32),
                                          predictions=y_pred_th, num_classes=3, name='IoU');
        tf.summary.scalar("mIoU", IoU);

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # print(update_ops)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='op').minimize(loss);

    merged_summary = tf.summary.merge_all();

    '''
    End of building the computational graph
    '''

    # Counting trainable variables
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Total model parameters: " + str(total_parameters))

    saver = tf.train.Saver(max_to_keep=1);

    session = tf.Session(graph=tf.get_default_graph())

    print("All global variables count: " + str(len(tf.all_variables())));
    uninitialized_vars = []
    for var in tf.all_variables():
        try:
            session.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)

    session.run(tf.initialize_variables(uninitialized_vars))

    print("Uninitialized global variabls count:" + str(len(uninitialized_vars)));

    print("All local variables count: " + str(len(tf.local_variables())));
    uninitialized_vars = []
    for var in tf.local_variables():
        try:
            session.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)

    session.run(tf.initialize_variables(uninitialized_vars))

    print("Uninitialized local variables count:" + str(len(uninitialized_vars)));

    folder = "lr_" + str(learning_rate) + "_epochs_" + str(max_epochs) + "_layers_" + str(enc_layers) + "_kver_" + str(
        kver)
    folder = folder + "_reg_" + str(reg)
    train_writer = tf.summary.FileWriter("logs/train_" + folder, session.graph);
    val_writer = tf.summary.FileWriter("logs/val_" + folder, session.graph);

    '''
    Starting training
    '''
    print("Starting training...");

    lrate = learning_rate;
    last_mIoU = 0;
    step = 0;
    for _ in range(max_epochs):
        # Train loop
        session.run(train_init_op);
        feed_dict_train = {'input/learning_rate:0': lrate, is_train: True};
        mIoU_over_steps = [];
        loss_over_steps = [];
        while True:
            try:
                _, _, mIoU, xent, s = session.run([optimizer, IoU_op, IoU, loss, merged_summary],feed_dict=feed_dict_train)
                mIoU_over_steps.append(mIoU);
                loss_over_steps.append(xent);
                if step % 100 == 0:
                    if debug:
                        x_prueba = np.asarray(session.run(x))
                        x_prueba_channels = len(x_prueba[0,0,0])
                        print(x_prueba.shape)
                        for i in range(x_prueba_channels):
                            print("Channel " + str(i) +":")
                            print([np.amin(x_prueba[:,:,:,i]),np.amax(x_prueba[:,:,:,i])])

                    print_progress(step, np.mean(mIoU_over_steps), np.mean(loss_over_steps), "train");
                    train_writer.add_summary(s, step);
                    mIoU_over_steps = [];
                    loss_over_steps = [];
                step += 1
            except tf.errors.OutOfRangeError:
                break
        # Test loop
        session.run(test_init_op);
        feed_dict_train = {is_train: False};
        mIoU_over_steps = [];
        loss_over_steps = [];
        while True:
            try:
                _, mIoU, xent, s = session.run([IoU_op, IoU, loss, merged_summary], feed_dict=feed_dict_train)
                last_mIoU = mIoU;
                mIoU_over_steps.append(mIoU);
                loss_over_steps.append(xent);
            except tf.errors.OutOfRangeError:
                break
        print_progress(step, np.mean(mIoU_over_steps), np.mean(loss_over_steps), "validation");
        val_writer.add_summary(s, step);

    # Test loop
    session.run(test_init_op);
    feed_dict_train = {is_train: False};
    mIoUList = []
    images_x = []
    images_y = []
    images_pred = []
    while True:
        try:
            _, mIoU, x_final,y_final,y_final_true = session.run([IoU_op, IoU,x,y_pred,y], feed_dict=feed_dict_train)
            mIoUList.append(mIoU)
            images_x.extend(x_final[:,:,:,0:3])
            images_y.extend(y_final_true)
            images_pred.extend(y_final)
        except tf.errors.OutOfRangeError:
            break

    print(len(images_x))
    print(images_x[0].shape)
    
    session.close()

    top_5_idx = np.argsort(mIoUList)[-5:]
    
    rank = 1
    
    for i in top_5_idx:
        IoU = mIoUList[i]
        final_x_image = np.array(images_x[i])*255
        final_y_image = np.array(images_y[i])*255
        final_pred_image = np.array(images_pred[i])*255
        # Stretch final pred image
        minvalue = np.amin(final_pred_image[:,:,0])
        final_pred_image[:,:,0] = final_pred_image[:,:,0] - minvalue
        maxvalue = np.amax(final_pred_image[:,:,0])
        final_pred_image[:,:,0] = final_pred_image[:,:,0] * 255 / maxvalue
    
        minvalue = np.amin(final_pred_image[:,:,1])
        final_pred_image[:,:,1] = final_pred_image[:,:,1] - minvalue
        maxvalue = np.amax(final_pred_image[:,:,1])
        final_pred_image[:,:,1] = final_pred_image[:,:,1] * 255 / maxvalue
    
        minvalue = np.amin(final_pred_image[:,:,2])
        final_pred_image[:,:,2] = final_pred_image[:,:,2] - minvalue
        maxvalue = np.amax(final_pred_image[:,:,2])
        final_pred_image[:,:,2] = final_pred_image[:,:,2] * 255 / maxvalue
    
        cv2.imwrite("{0}_{1}_x.jpg".format(rank,IoU), final_x_image)
        cv2.imwrite("{0}_{1}_y.jpg".format(rank,IoU), final_y_image)
        cv2.imwrite("{0}_{1}_pred.jpg".format(rank,IoU), final_pred_image)
        rank = rank + 1

    del session
    gc.collect()

    with open("results.txt", "w+") as f:
        f.write('Test results: \n')
        f.write('mIoU = ' + str(np.mean(mIoU_over_steps)) + '\n')
        f.write('loss = ' + str(np.mean(loss_over_steps)) + '\n')


if __name__ == "__main__":
    main();