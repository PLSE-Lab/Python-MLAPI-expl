#!/usr/bin/env python
# coding: utf-8

#  # YOLO v3 Object Detection in Tensorflow
#  [GitHub repo link](https://github.com/heartkilla/yolo-v3) <br>
#  UPDATE: Video processing
#  <a id="top"></a> <br>
# ## Content
# 1. [What is Yolo?](#1)
# 2. [Dependencies](#2)
# 3. [Model hyperparameters](#3)
# 4. [Model definiton](#4)
# 5. [Utility functions](#5)
# 6. [Converting weights to Tensorflow format](#6)
# 7. [Running the model](#7)
# 8. [Video processing](#8)
# 9. [To-Do list](#9)
# 10. [Acknowledgements](#10)

# <a id="1"></a> 
# ## 1. What is Yolo?
# ![](https://cdn-images-1.medium.com/max/1000/1*wnr2e-W3WvYk_G51Y4oMCQ.png)

# <a id="2"></a> 
# ## 2. Dependencies
# To build Yolo we're going to need Tensorflow (deep learning), NumPy (numerical computation) and Pillow (image processing) libraries. Also we're going to use seaborn's color palette for bounding boxes colors. Finally, let's import IPython function `display()` to display images in the notebook.

# In[ ]:


import numpy as np
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
from seaborn import color_palette
import cv2
import tensorflow as tf
import tensorflow.keras as keras


# <a id="3"></a>
# ## 3. Model hyperparameters
# Next, we define some configurations for Yolo. 

# In[ ]:


_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1
_ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
_MODEL_SIZE = (416, 416)


# ### 9 Anchor Boxes
# ![](data:image/png;base64,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)

# `_MODEL_SIZE` refers to the input size of the model.
# 
# Let's look at other parameters step-by-step.

# ### Batch normalization
# Almost every convolutional layer in Yolo has batch normalization after it. It helps the model train faster and reduces variance between units (and total variance as well). Batch normalization is defined as follows.
# <br>
# ![](https://hsto.org/files/005/d19/2bd/005d192bd6274c298f75896498aea377.png)
# <br>
# `_BATCH_NORM_EPSILON` refers to epsilon in this formula, whereas `_BATCH_NORM_DECAY` refers to momentum, which is used for computing moving average and variance. We use them in forward propagation during inference (after training).
# <br>
# <br>
# `moving_average = momentum * moving_average + (1 - momentum) * current_average`

# ### Leaky ReLU
# Leaky ReLU is a slight modification of ReLU activation function. The idea behind Leaky ReLU is to prevent so-called "neuron dying" when a large number of activations become 0. 
# <br>
# ![](https://i1.wp.com/sefiks.com/wp-content/uploads/2018/02/prelu.jpg?resize=300%2C201&ssl=1)
# `_LEAKY_RELU` refers to alpha.

# ### Anchors
# ![](https://2.bp.blogspot.com/-_R-w_tWHdzc/WzJPsol7qFI/AAAAAAABbgg/Jsf-AO3qH0A9oiCeU0LQxN-wdirlOz4WgCLcBGAs/s400/%25E8%259E%25A2%25E5%25B9%2595%25E5%25BF%25AB%25E7%2585%25A7%2B2018-06-26%2B%25E4%25B8%258B%25E5%258D%258810.36.51.png)
# Anchors are sort of bounding box priors, that were calculated on the COCO dataset using k-means clustering. We are going to predict the width and height of the box as offsets from cluster centroids. The center coordinates of the box relative to the location of filter application are predicted using a sigmoid function.
# <br>
# $$b_{x} = \sigma(t_{x})+c_{x}$$
# $$b_{y} = \sigma(t_{y})+c_{y}$$
# $$b_{w} = p_{w}e^{t_{w}}$$
# $$b_{h} = p_{h}e^{t_{h}}$$
# <br>
# Where $b_{x}$ and $b_{y}$ are the center coordinates of the box, $b_{w}$ and $b_{h}$ are the width and height of the box, $c_{x}$ and $c_{y}$ are the location of filter application and $t_{i}$ are predicted during regression.

# <a id="4"></a>
# ## 4. Model definition
# I refered to the official ResNet implementation in Tensorflow in terms of how to arange the code. 

# ### Batch norm and fixed padding
# It's useful to define `batch_norm` function since the model uses batch norms with shared parameters heavily. Also, same as ResNet, Yolo uses convolution with fixed padding, which means that padding is defined only by the size of the kernel.

# In[ ]:


def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
        scale=True, training=training)


def fixed_padding(inputs, kernel_size, data_format):
    """ResNet implementation of fixed padding.

    Pads the input along the spatial dimensions independently of input size.

    Args:
        inputs: Tensor input to be padded.
        kernel_size: The kernel to be used in the conv2d or max_pool2d.
        data_format: The input format.
    Returns:
        A tensor with the same format as the input.
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end],
                                        [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, data_format, strides=1):
    """Strided 2-D convolution with explicit padding."""
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size,
        strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False, data_format=data_format)


# ### Feature extraction: Darknet-53
# For feature extraction Yolo uses Darknet-53 neural net pretrained on ImageNet. Same as ResNet,  Darknet-53 has shortcut (residual) connections, which help information from earlier layers flow further. We omit the last 3 layers (Avgpool, Connected and Softmax) since we only need the features.

# ### residual block
# ![](https://miro.medium.com/max/513/1*D0F3UitQ2l5Q0Ak-tjEdJg.png)

# In[ ]:


def darknet53_residual_block(inputs, filters, training, data_format,
                             strides=1):
    """Creates a residual block for Darknet."""
    shortcut = inputs

    inputs = conv2d_fixed_padding(
        inputs, filters=filters, kernel_size=1, strides=strides,
        data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv2d_fixed_padding(
        inputs, filters=2 * filters, kernel_size=3, strides=strides,
        data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs += shortcut

    return inputs


# In[ ]:


def darknet53(inputs, training, data_format):
    """Creates Darknet53 model for feature extraction."""
    inputs = conv2d_fixed_padding(inputs, filters=32, kernel_size=3,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    inputs = conv2d_fixed_padding(inputs, filters=64, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = darknet53_residual_block(inputs, filters=32, training=training,
                                      data_format=data_format)

    inputs = conv2d_fixed_padding(inputs, filters=128, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(2):
        inputs = darknet53_residual_block(inputs, filters=64,
                                          training=training,
                                          data_format=data_format)

    inputs = conv2d_fixed_padding(inputs, filters=256, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(8):
        inputs = darknet53_residual_block(inputs, filters=128,
                                          training=training,
                                          data_format=data_format)

    route1 = inputs

    inputs = conv2d_fixed_padding(inputs, filters=512, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(8):
        inputs = darknet53_residual_block(inputs, filters=256,
                                          training=training,
                                          data_format=data_format)

    route2 = inputs

    inputs = conv2d_fixed_padding(inputs, filters=1024, kernel_size=3,
                                  strides=2, data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    for _ in range(4):
        inputs = darknet53_residual_block(inputs, filters=512,
                                          training=training,
                                          data_format=data_format)

    return route1, route2, inputs


# ### Convolution layers
# Yolo has a large number of convolutional layers. It's useful to group them in blocks.

# In[ ]:


def yolo_convolution_block(inputs, filters, training, data_format):
    """Creates convolution operations layer used after Darknet."""
    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv2d_fixed_padding(inputs, filters=2 * filters, kernel_size=3,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv2d_fixed_padding(inputs, filters=2 * filters, kernel_size=3,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    inputs = conv2d_fixed_padding(inputs, filters=filters, kernel_size=1,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    route = inputs

    inputs = conv2d_fixed_padding(inputs, filters=2 * filters, kernel_size=3,
                                  data_format=data_format)
    inputs = batch_norm(inputs, training=training, data_format=data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)

    return route, inputs


# ### Detection layers
# Yolo has 3 detection layers, that detect on 3 different scales using respective anchors. For each cell in the feature map the detection layer predicts `n_anchors * (5 + n_classes)` values using 1x1 convolution. For each scale we have `n_anchors = 3`. `5 + n_classes` means that respectively to each of 3 anchors we are going to predict 4 coordinates of the box, its confidence score (the probability of containing an object) and class probabilities. 

# In[ ]:


def yolo_layer(inputs, n_classes, anchors, img_size, data_format):
    """Creates Yolo final detection layer.

    Detects boxes with respect to anchors.

    Args:
        inputs: Tensor input.
        n_classes: Number of labels.
        anchors: A list of anchor sizes.
        img_size: The input size of the model.
        data_format: The input format.

    Returns:
        Tensor output.
    """
    n_anchors = len(anchors)

    inputs = tf.layers.conv2d(inputs, filters=n_anchors * (5 + n_classes),
                              kernel_size=1, strides=1, use_bias=True,
                              data_format=data_format)

    shape = inputs.get_shape().as_list()
    grid_shape = shape[2:4] if data_format == 'channels_first' else shape[1:3]
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
    inputs = tf.reshape(inputs, [-1, n_anchors * grid_shape[0] * grid_shape[1],
                                 5 + n_classes])

    strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])

    box_centers, box_shapes, confidence, classes =         tf.split(inputs, [2, 2, 1, n_classes], axis=-1)

    x = tf.range(grid_shape[0], dtype=tf.float32)
    y = tf.range(grid_shape[1], dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(x, y)
    x_offset = tf.reshape(x_offset, (-1, 1))
    y_offset = tf.reshape(y_offset, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
    x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
    box_centers = tf.nn.sigmoid(box_centers)
    box_centers = (box_centers + x_y_offset) * strides

    anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
    box_shapes = tf.exp(box_shapes) * tf.to_float(anchors)

    confidence = tf.nn.sigmoid(confidence)

    classes = tf.nn.sigmoid(classes)

    inputs = tf.concat([box_centers, box_shapes,
                        confidence, classes], axis=-1)

    return inputs


# ### Upsample layer
# In order to concatenate with shortcut outputs from Darknet-53 before applying detection on a different scale, we are going to upsample the feature map using nearest neighbor interpolation.

# In[ ]:


def upsample(inputs, out_shape, data_format):
    """Upsamples to `out_shape` using nearest neighbor interpolation."""
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        new_height = out_shape[3]
        new_width = out_shape[2]
    else:
        new_height = out_shape[2]
        new_width = out_shape[1]

    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    return inputs


# ### Non-max suppression
# The model is going to produce a lot of boxes, so we need a way to discard the boxes with low confidence scores. Also, to avoid having multiple boxes for one object, we will discard the boxes with high overlap as well using non-max suppresion for each class.

# In[ ]:


def build_boxes(inputs):
    """Computes top left and bottom right points of the boxes."""
    center_x, center_y, width, height, confidence, classes =         tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    top_left_x = center_x - width / 2
    top_left_y = center_y - height / 2
    bottom_right_x = center_x + width / 2
    bottom_right_y = center_y + height / 2

    boxes = tf.concat([top_left_x, top_left_y,
                       bottom_right_x, bottom_right_y,
                       confidence, classes], axis=-1)

    return boxes


def non_max_suppression(inputs, n_classes, max_output_size, iou_threshold,
                        confidence_threshold):
    """Performs non-max suppression separately for each class.

    Args:
        inputs: Tensor input.
        n_classes: Number of classes.
        max_output_size: Max number of boxes to be selected for each class.
        iou_threshold: Threshold for the IOU.
        confidence_threshold: Threshold for the confidence score.
    Returns:
        A list containing class-to-boxes dictionaries
            for each sample in the batch.
    """
    batch = tf.unstack(inputs)
    boxes_dicts = []
    for boxes in batch:
        boxes = tf.boolean_mask(boxes, boxes[:, 4] > confidence_threshold)
        classes = tf.argmax(boxes[:, 5:], axis=-1)
        classes = tf.expand_dims(tf.to_float(classes), axis=-1)
        boxes = tf.concat([boxes[:, :5], classes], axis=-1)

        boxes_dict = dict()
        for cls in range(n_classes):
            mask = tf.equal(boxes[:, 5], cls)
            mask_shape = mask.get_shape()
            if mask_shape.ndims != 0:
                class_boxes = tf.boolean_mask(boxes, mask)
                boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes,
                                                              [4, 1, -1],
                                                              axis=-1)
                boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
                indices = tf.image.non_max_suppression(boxes_coords,
                                                       boxes_conf_scores,
                                                       max_output_size,
                                                       iou_threshold)
                class_boxes = tf.gather(class_boxes, indices)
                boxes_dict[cls] = class_boxes[:, :5]

        boxes_dicts.append(boxes_dict)

    return boxes_dicts


# ### Final model class
# Finally, let's define the model class using all of the layers described previously. 

# ### YOLO v3 architecture 
# ![](https://2.bp.blogspot.com/-VhL37ZWApqg/WzJQgGURTJI/AAAAAAABbgo/nPnrC3l_lSEIJNFCJY9TuOBcukXk0cQcgCEwYBhgL/s640/yolov3_structure.png)

# In[ ]:


class Yolo_v3:
    """Yolo v3 model class."""

    def __init__(self, n_classes, model_size, max_output_size, iou_threshold,
                 confidence_threshold, data_format=None):
        """Creates the model.

        Args:
            n_classes: Number of class labels.
            model_size: The input size of the model.
            max_output_size: Max number of boxes to be selected for each class.
            iou_threshold: Threshold for the IOU.
            confidence_threshold: Threshold for the confidence score.
            data_format: The input format.

        Returns:
            None.
        """
        if not data_format:
            if tf.test.is_built_with_cuda():
                data_format = 'channels_first'
            else:
                data_format = 'channels_last'

        self.n_classes = n_classes
        self.model_size = model_size
        self.max_output_size = max_output_size
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.data_format = data_format

    def __call__(self, inputs, training):
        """Add operations to detect boxes for a batch of input images.

        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean, whether to use in training or inference mode.

        Returns:
            A list containing class-to-boxes dictionaries
                for each sample in the batch.
        """
        with tf.variable_scope('yolo_v3_model'):
            if self.data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            inputs = inputs / 255

            route1, route2, inputs = darknet53(inputs, training=training,
                                               data_format=self.data_format)

            route, inputs = yolo_convolution_block(
                inputs, filters=512, training=training,
                data_format=self.data_format)
            detect1 = yolo_layer(inputs, n_classes=self.n_classes,
                                 anchors=_ANCHORS[6:9],
                                 img_size=self.model_size,
                                 data_format=self.data_format)

            inputs = conv2d_fixed_padding(route, filters=256, kernel_size=1,
                                          data_format=self.data_format)
            inputs = batch_norm(inputs, training=training,
                                data_format=self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
            upsample_size = route2.get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_size,
                              data_format=self.data_format)
            axis = 1 if self.data_format == 'channels_first' else 3
            inputs = tf.concat([inputs, route2], axis=axis)
            route, inputs = yolo_convolution_block(
                inputs, filters=256, training=training,
                data_format=self.data_format)
            detect2 = yolo_layer(inputs, n_classes=self.n_classes,
                                 anchors=_ANCHORS[3:6],
                                 img_size=self.model_size,
                                 data_format=self.data_format)

            inputs = conv2d_fixed_padding(route, filters=128, kernel_size=1,
                                          data_format=self.data_format)
            inputs = batch_norm(inputs, training=training,
                                data_format=self.data_format)
            inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
            upsample_size = route1.get_shape().as_list()
            inputs = upsample(inputs, out_shape=upsample_size,
                              data_format=self.data_format)
            inputs = tf.concat([inputs, route1], axis=axis)
            route, inputs = yolo_convolution_block(
                inputs, filters=128, training=training,
                data_format=self.data_format)
            detect3 = yolo_layer(inputs, n_classes=self.n_classes,
                                 anchors=_ANCHORS[0:3],
                                 img_size=self.model_size,
                                 data_format=self.data_format)

            inputs = tf.concat([detect1, detect2, detect3], axis=1)

            inputs = build_boxes(inputs)

            boxes_dicts = non_max_suppression(
                inputs, n_classes=self.n_classes,
                max_output_size=self.max_output_size,
                iou_threshold=self.iou_threshold,
                confidence_threshold=self.confidence_threshold)

            return boxes_dicts


# <a id="5"></a>
# ## 5. Utility functions
# Here are some utility functions that will help us load images as NumPy arrays, load class names from the official file and draw the predicted boxes.

# In[ ]:


def load_images(img_names, model_size):
    """Loads images in a 4D array.

    Args:
        img_names: A list of images names.
        model_size: The input size of the model.
        data_format: A format for the array returned
            ('channels_first' or 'channels_last').

    Returns:
        A 4D NumPy array.
    """
    imgs = []

    for img_name in img_names:
        img = Image.open(img_name)
        img = img.resize(size=model_size)
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        imgs.append(img)

    imgs = np.concatenate(imgs)

    return imgs


def load_class_names(file_name):
    """Returns a list of class names read from `file_name`."""
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names


def draw_boxes(img_names, boxes_dicts, class_names, model_size):
    """Draws detected boxes.
    Args:
        img_names: A list of input images names.
        boxes_dict: A class-to-boxes dictionary.
        class_names: A class names list.
        model_size: The input size of the model.

    Returns:
        None.
    """
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    for num, img_name, boxes_dict in zip(range(len(img_names)), img_names,
                                         boxes_dicts):
        img = Image.open(img_name)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font='../input/futur.ttf',
                                  size=(img.size[0] + img.size[1]) // 100)
        resize_factor =             (img.size[0] / model_size[0], img.size[1] / model_size[1])
        for cls in range(len(class_names)):
            boxes = boxes_dict[cls]
            if np.size(boxes) != 0:
                color = colors[cls]
                for box in boxes:
                    xy, confidence = box[:4], box[4]
                    xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
                    x0, y0 = xy[0], xy[1]
                    thickness = (img.size[0] + img.size[1]) // 200
                    for t in np.linspace(0, 1, thickness):
                        xy[0], xy[1] = xy[0] + t, xy[1] + t
                        xy[2], xy[3] = xy[2] - t, xy[3] - t
                        draw.rectangle(xy, outline=tuple(color))
                    text = '{} {:.1f}%'.format(class_names[cls],
                                               confidence * 100)
                    text_size = draw.textsize(text, font=font)
                    draw.rectangle(
                        [x0, y0 - text_size[1], x0 + text_size[0], y0],
                        fill=tuple(color))
                    draw.text((x0, y0 - text_size[1]), text, fill='black',
                              font=font)

        display(img)


# <a id="6"></a>
# ## 6. Converting weights to Tensorflow format
# Now it's time to load the official weights. We are going to iterate through the file and gradually create `tf.assign` operations.

# In[ ]:


def load_weights(variables, file_name):
    """Reshapes and loads official pretrained Yolo weights.

    Args:
        variables: A list of tf.Variable to be assigned.
        file_name: A name of a file containing weights.

    Returns:
        A list of assign operations.
    """
    with open(file_name, "rb") as f:
        # Skip first 5 values containing irrelevant info
        np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

        assign_ops = []
        ptr = 0

        # Load weights for Darknet part.
        # Each convolution layer has batch normalization.
        for i in range(52):
            conv_var = variables[5 * i]
            gamma, beta, mean, variance = variables[5 * i + 1:5 * i + 5]
            batch_norm_vars = [beta, gamma, mean, variance]

            for var in batch_norm_vars:
                shape = var.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape(shape)
                ptr += num_params
                assign_ops.append(tf.assign(var, var_weights))

            shape = conv_var.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(conv_var, var_weights))

        # Loading weights for Yolo part.
        # 7th, 15th and 23rd convolution layer has biases and no batch norm.
        ranges = [range(0, 6), range(6, 13), range(13, 20)]
        unnormalized = [6, 13, 20]
        for j in range(3):
            for i in ranges[j]:
                current = 52 * 5 + 5 * i + j * 2
                conv_var = variables[current]
                gamma, beta, mean, variance =                      variables[current + 1:current + 5]
                batch_norm_vars = [beta, gamma, mean, variance]

                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights))

                shape = conv_var.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape(
                    (shape[3], shape[2], shape[0], shape[1]))
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                assign_ops.append(tf.assign(conv_var, var_weights))

            bias = variables[52 * 5 + unnormalized[j] * 5 + j * 2 + 1]
            shape = bias.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(shape)
            ptr += num_params
            assign_ops.append(tf.assign(bias, var_weights))

            conv_var = variables[52 * 5 + unnormalized[j] * 5 + j * 2]
            shape = conv_var.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(conv_var, var_weights))

    return assign_ops


# <a id="7"></a>
# ## 7. Running the model
# Now we can run the model using some sample images.

# ### Sample images

# In[ ]:


img_names = ['../input/dog.jpg', '../input/office.jpg']
for img in img_names: display(Image.open(img))


# ### Detections
# Testing the model with IoU (Interception over Union ratio used in non-max suppression) threshold and confidence threshold both set to 0.5.

# In[ ]:


batch_size = len(img_names)
batch = load_images(img_names, model_size=_MODEL_SIZE)
class_names = load_class_names('../input/coco.names')
n_classes = len(class_names)
max_output_size = 10
iou_threshold = 0.5
confidence_threshold = 0.5

model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE,
                max_output_size=max_output_size,
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold)

inputs = tf.placeholder(tf.float32, [batch_size, 416, 416, 3])

detections = model(inputs, training=False)

model_vars = tf.global_variables(scope='yolo_v3_model')
assign_ops = load_weights(model_vars, '../input/yolov3.weights')

with tf.Session() as sess:
    sess.run(assign_ops)
    detection_result = sess.run(detections, feed_dict={inputs: batch})
    
draw_boxes(img_names, detection_result, class_names, _MODEL_SIZE)


# <a id="8"></a>
# ## 8. Video processing
# I also applied the same algorithm to video detections. The code is available on my [GitHub repo](https://github.com/heartkilla/yolo-v3). <br>
# Here is an example of applying Yolo to a video I found on YouTube. ([A Street Walk in Shinjuku, Tokyo, Japan](https://www.youtube.com/watch?v=kZ7caIK4RXI))
# ![](https://github.com/heartkilla/yolo-v3/blob/master/data/detection_examples/detections.gif)

# In[ ]:


from IPython.display import Image
with open('../input/detections.gif','rb') as f:
    display(Image(data=f.read(), format='png'))


# <a id="9"></a>
# ## 9. To-Do list
# * training

# <a id="10"></a>
# ## 10. Acknowledgements
# * [Yolo v3 official paper](https://arxiv.org/abs/1804.02767)
# * [A Tensorflow Slim implementation](https://github.com/mystic123/tensorflow-yolo-v3)
# * [ResNet official implementation](https://github.com/tensorflow/models/tree/master/official/resnet)
# * [DeviceHive video analysis repo](https://github.com/devicehive/devicehive-video-analysis)
# * [A Street Walk in Shinjuku, Tokyo, Japan](https://www.youtube.com/watch?v=kZ7caIK4RXI)
# 
# Special thanks to [Paul]( https://www.kaggle.com/paultimothymooney) for posting this kernel in [Kaggle Data Notes Newsletter](https://www.kaggle.com/page/data-notes).
# 
