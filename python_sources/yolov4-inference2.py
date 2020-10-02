#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
from keras.models import Sequential
import keras
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D,MaxPool2D,Dropout,Reshape,Add,Conv2DTranspose,Concatenate
from keras.layers import LeakyReLU
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches


# In[ ]:


import tensorflow as tf
# import tensorflow_addons as tfa
class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn: conv = BatchNormalization()(conv)
    if activate == True:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
            # conv = softplus(conv)
            # conv = conv * tf.math.tanh(tf.math.softplus(conv))
            # conv = conv * tf.tanh(softplus(conv))
            # conv = tf.nn.leaky_relu(conv, alpha=0.1)
            # conv = tfa.activations.mish(conv)
            # conv = conv * tf.nn.tanh(tf.keras.activations.relu(tf.nn.softplus(conv), max_value=20))
            # conv = tf.nn.softplus(conv)
            # conv = tf.keras.activations.relu(tf.nn.softplus(conv), max_value=20)

    return conv
def softplus(x, threshold = 20.):
    def f1():
        return x
    def f2():
        return tf.exp(x)
    def f3():
        return tf.math.log(1 + tf.exp(x))
    # mask = tf.greater(x, threshold)
    # x = tf.exp(x[mask])
    # return tf.exp(x)
    return tf.case([(tf.greater(x, tf.constant(threshold)), lambda:f1()), (tf.less(x, tf.constant(-threshold)), lambda:f2())], default=lambda:f3())
    # return tf.case([(tf.greater(x, threshold), lambda:f1())])
def mish(x):
    return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)
    # return tf.keras.layers.Lambda(lambda x: softplus(x))(x)
    # return tf.keras.layers.Lambda(lambda x: x * tf.tanh(softplus(x)))(x)

def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    residual_output = short_cut + conv
    return residual_output

# def block_tiny(input_layer, input_channel, filter_num1, activate_type='leaky'):
#     conv = convolutional(input_layer, filters_shape=(3, 3, input_channel, filter_num1), activate_type=activate_type)
#     short_cut = input_layer
#     conv = convolutional(conv, filters_shape=(3, 3, input_channel, filter_num1), activate_type=activate_type)
#
#     input_data = tf.concat([conv, short_cut], axis=-1)
#     return residual_output

def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]

def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')


# In[ ]:


def cspdarknet53(input_data):

    input_data = convolutional(input_data, (3, 3,  3,  32), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")

    route = input_data
    route = convolutional(route, (1, 1, 64, 64), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

    input_data = tf.concat([input_data, route], axis=-1)
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 128, 64), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    for i in range(2):
        input_data = residual_block(input_data, 64,  64, 64, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 256, 128), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
    for i in range(4):
        input_data = residual_block(input_data, 128, 128, 128, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 512, 256), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
    for i in range(4):
        input_data = residual_block(input_data, 256, 256, 256, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
    route = input_data
    route = convolutional(route, (1, 1, 1024, 512), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")
    for i in range(4):
        input_data = residual_block(input_data, 512, 512, 512, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 1024, 512))
    input_data = convolutional(input_data, (3, 3, 512, 1024))
    input_data = convolutional(input_data, (1, 1, 1024, 512))

    input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1)
                            , tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)
    input_data = convolutional(input_data, (1, 1, 2048, 512))
    input_data = convolutional(input_data, (3, 3, 512, 1024))
    input_data = convolutional(input_data, (1, 1, 1024, 512))

    return route_1, route_2, input_data


# In[ ]:


def YOLOv4(input_layer, NUM_CLASS=0):
    route_1, route_2, conv = cspdarknet53(input_layer)

    route = conv
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = upsample(conv)
    route_2 = convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    #conv = convolutional(conv, (1, 1, 512, 256))
    #conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)
    route_1 = convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    #conv = convolutional(conv, (1, 1, 256, 128))
    #conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = convolutional(conv, (1, 1, 256, 25 ), activate=False, bn=False)

    conv = convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    #conv = convolutional(conv, (1, 1, 512, 256))
    #conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = convolutional(conv, (1, 1, 512, 25 ), activate=False, bn=False)

    conv = convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    #conv = convolutional(conv, (1, 1, 1024, 512))
    #conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))

    conv = convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = convolutional(conv, (1, 1, 1024, 25 ), activate=False, bn=False)
    
    conv_sbbox=Reshape((32,32,5,5),name='out1')(conv_sbbox)
    conv_mbbox=Reshape((16,16,5,5),name='out2')(conv_mbbox)
    conv_lbbox=Reshape((8,8,5,5),name='out3')(conv_lbbox)

    

    return [conv_sbbox, conv_mbbox, conv_lbbox]


# In[ ]:


def decode_train(conv,clusters,shape,name):
    clust=np.zeros((shape,shape,5,2))
    for i in range(shape):
        for j in range(shape):
            clust[i][j]=clusters
    clust=tf.convert_to_tensor(tf.cast(clust,dtype='float32'))
        
    xy=tf.sigmoid(conv[...,0:2])
    wh=tf.exp(conv[...,2:4])*clust
    prob=tf.sigmoid(conv[...,4:])
                
                
    return tf.concat([xy,wh,prob],axis=-1,name=name)


# In[ ]:


clusters=np.array([[0.04980469, 0.06445312],
       [0.1171875 , 0.10253906],
       [0.06445312, 0.03710938],
       [0.09179688, 0.05859375],
       [0.07226562, 0.08496094]])


# In[ ]:


def my_model(clusters,input_shape):
    input_data=Input(input_shape)
    [conv_sbbox, conv_mbbox, conv_lbbox] = YOLOv4(input_data)

    conv_sbbox=decode_train(conv_sbbox,clusters,32,'sbbox')
    conv_mbbox=decode_train(conv_mbbox,clusters,16,'mbbox')
    conv_lbbox=decode_train(conv_lbbox,clusters,8,'lbbox')
        
    
    
    
    model=Model(input_data,[conv_sbbox, conv_mbbox, conv_lbbox])
    
    return model


# In[ ]:


def bbox_ciou(bboxes1, bboxes2):
    """
    Complete IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up

    c_2 = enclose_section[..., 0] ** 2 + enclose_section[..., 1] ** 2

    center_diagonal = bboxes2[..., :2] - bboxes1[..., :2]

    rho_2 = center_diagonal[..., 0] ** 2 + center_diagonal[..., 1] ** 2

    diou = iou - tf.math.divide_no_nan(rho_2, c_2)

    v = (
        (
            tf.math.atan(
                tf.math.divide_no_nan(bboxes1[..., 2], bboxes1[..., 3])
            )
            - tf.math.atan(
                tf.math.divide_no_nan(bboxes2[..., 2], bboxes2[..., 3])
            )
        )
        * 2/np.pi
        
    ) ** 2

    alpha = tf.math.divide_no_nan(v, 1 - iou + v)

    ciou = diou - alpha * v

    return ciou


# In[ ]:


def log_loss(y_true,y_pred):
    return -y_true * tf.math.log(y_pred+0.0000000001)-( (1-y_true)*tf.math.log(1-y_pred+0.000000000001) )


# In[ ]:


def yolo_loss(y_true,y_pred):
  lossx=K.sum(tf.math.multiply(K.square(y_true[:,:,:,:,0:1]-y_pred[:,:,:,:,0:1]),y_true[:,:,:,:,4:]),axis=[1,2,3,4])
  lossy=K.sum(tf.math.multiply(K.square(y_true[:,:,:,:,1:2]-y_pred[:,:,:,:,1:2]),y_true[:,:,:,:,4:]),axis=[1,2,3,4])
  loss1=lossx+lossy
  lossw=K.sum(tf.math.multiply(K.square(y_true[:,:,:,:,2:3]-y_pred[:,:,:,:,2:3]),y_true[:,:,:,:,4:]),axis=[1,2,3,4])
  lossh=K.sum(tf.math.multiply(K.square(y_true[:,:,:,:,3:4]-y_pred[:,:,:,:,3:4]),y_true[:,:,:,:,4:]),axis=[1,2,3,4])
  loss2=lossw+lossh
  loss_xy_wh=(loss1+loss2)*2.0
  #lossC=K.sum(tf.math.multiply(K.square(tf.math.subtract(y_true[:,:,:,4:],y_pred[:,:,:,4:])),y_true[:,:,:,4:]),axis=[1,2,3])
  #lossC2 =K.sum(tf.math.multiply(K.square(tf.math.subtract(y_true[:,:,:,4:],y_pred[:,:,:,4:])),(1-y_true[:,:,:,4:])),axis=[1,2,3])/16  
  #lossC=lossC+lossC2
  #conf_focal = tf.pow(y_true[:,:,:,:,4:] - y_true[:,:,:,:,4:], 2)
  conf_loss = K.sum( (
           y_true[:,:,:,:,4:] * log_loss(y_true[:,:,:,:,4:],y_pred[:,:,:,:,4:])*1.5
            +
            (1-y_true[:,:,:,:,4:]) *0.05* log_loss(y_true[:,:,:,:,4:],y_pred[:,:,:,:,4:])
    ),axis=[1,2,3,4])
    
  ciou = tf.expand_dims(bbox_ciou(y_pred[:,:,:,:,:4], y_true[:,:,:,:,:4]), axis=-1)
  ciou_loss = y_true[:,:,:,:,4:] * 2.0 * (1- ciou)

    
  total_loss=loss_xy_wh+K.sum(ciou_loss,axis=[1,2,3,4])+conf_loss
  return total_loss


# In[ ]:


model=my_model(clusters,input_shape=(256,256,3))


# In[ ]:


model.load_weights('../input/30k-wheats-modified/30k_yolov4_weights2.h5')


# In[ ]:


def get_boxes(box,boxes,scores):
  m=box.shape[0]
  for i in range(m):
    for j in range(m):
        for k in range(5):
            if(box[i][j][k][4]>=0.95):
                x=(i*(1024/m))+(box[i][j][k][0]*(1024/m))
                y=(j*(1024/m))+(box[i][j][k][1]*(1024/m))
                w=box[i][j][k][2]*1024
                h=box[i][j][k][3]*1024
                x1=int((x-w/2))
                y1=int((y-h/2))
                x2=int((x+w/2))
                y2=int((y+h/2))
                #print(box[i][j][k][4])
               #print(box[i][j][k][4],int(x1),y1,x2,y2)
                boxes.append([y1,x1,y2,x2])
                scores.append(box[i][j][k][4])
                #cv2.rectangle(img,(89,250),(202,363),(255,0,0),2)


# In[ ]:


def draw_boxes(address,model):
    img=cv2.imread(address)
    img1=cv2.resize(img,(256,256))
    img1=img1/255.0
    img1=img1[np.newaxis,:]
    box1,box2,box3=model.predict(img1)
    boxes=[]
    scores=[]
    get_boxes(box1[0],boxes,scores)
    get_boxes(box2[0],boxes,scores)
    get_boxes(box3[0],boxes,scores)
    boxes=np.array(boxes)
    scores=np.array(scores)
    boxes=tf.convert_to_tensor(boxes)
    scores=tf.convert_to_tensor(scores)
    boxes=tf.cast(boxes, tf.float32)
    scores=tf.cast(scores,tf.float32)
    selected_indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size=50, iou_threshold=0.1)
    selected_boxes = tf.gather(boxes, selected_indices)
    selected_scores=tf.gather(scores,selected_indices)
    return np.array(selected_scores),np.array(selected_boxes)


# In[ ]:


DIR_INPUT = '/kaggle/input/global-wheat-detection'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'


imagenames = os.listdir(DIR_TEST)


# In[ ]:


results=[]
for count, name in enumerate(imagenames):
    ids = name.split('.')[0]
    imagepath = '%s/%s.jpg'%(DIR_TEST,ids)
    img=cv2.imread(imagepath)
    img1=cv2.resize(img,(256,256))
    img1=img1/255.0
    img1=img1[np.newaxis,:]
    [box1,box2,box3]=model.predict(img1)
    boxes=[]
    scores=[]
    for boz in [box1,box2,box3]:
          box=boz[0]
          m=box.shape[0]
          for i in range(m):
            for j in range(m):
                for k in range(5):
                    if(box[i][j][k][4]>=0.1):
                        x=(i*(1024/m))+(box[i][j][k][0]*(1024/m))
                        y=(j*(1024/m))+(box[i][j][k][1]*(1024/m))
                        w=box[i][j][k][2]*1024
                        h=box[i][j][k][3]*1024
                        x1=int((x-w/2))
                        y1=int((y-h/2))
                        x2=int((x+w/2))
                        y2=int((y+h/2))
                        #print(box[i][j][k][4])
                       #print(box[i][j][k][4],int(x1),y1,x2,y2)
                        boxes.append([y1,x1,y2,x2])
                        scores.append(box[i][j][k][4])
    
      
     
    if(len(boxes)>0):
        boxes=np.array(boxes)
        scores=np.array(scores)
        boxes=tf.convert_to_tensor(boxes)
        scores=tf.convert_to_tensor(scores)
        boxes=tf.cast(boxes, tf.float32)
        scores=tf.cast(scores,tf.float32)
        selected_indices = tf.image.non_max_suppression(
            boxes, scores, max_output_size=50, iou_threshold=0.1)
        selected_boxes = tf.gather(boxes, selected_indices)
        selected_scores=tf.gather(scores,selected_indices)
        boxes=np.array(selected_boxes)
        scores=np.array(selected_scores)

        pred_strings = []
        m,n=boxes.shape
        for i in range(m):
            pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(np.clip(np.around(scores[i],4),0,1), np.clip(int(np.around(boxes[i][1])),0,1023), np.clip(int(np.around(boxes[i][0])),0,1023), np.clip(int(np.around(boxes[i][3]-boxes[i][1])),0,1023), np.clip(int(np.around(boxes[i][2]-boxes[i][0])),0,1023)))
        if(len(pred_strings)>0):
            result = {'image_id':ids,'PredictionString': " ".join(pred_strings)}
    else:
        result = {'image_id':ids,'PredictionString': " "}
        
    results.append(result)
    
    


# In[ ]:


results


# In[ ]:


test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

    


# In[ ]:


test_df.head()


# In[ ]:


test_df.to_csv('submission.csv', index=False)


# In[ ]:


def draw_boxes(box,ax):
  m=box.shape[0]
  for i in range(m):
    for j in range(m):
        for k in range(5):
            if(box[i][j][k][4]>0.875):
                x=(i*(1024/m))+(box[i][j][k][0]*(1024/m))
                y=(j*(1024/m))+(box[i][j][k][1]*(1024/m))
                w=box[i][j][k][2]*1024
                h=box[i][j][k][3]*1024
                x1=int((x-w/2))
                y1=int((y-h/2))
                x2=int((x+w/2))
                y2=int((y+h/2))
                #print(box[i][j][k][4])
               #print(box[i][j][k][4],int(x1),y1,x2,y2)
                rect = patches.Rectangle((x1,y1),w,h,linewidth=1,edgecolor='r',facecolor='none')
                boxes.append([y1,x1,y2,x2])
                scores.append(box[i][j][k][4])
                #cv2.rectangle(img,(89,250),(202,363),(255,0,0),2)
                ax.add_patch(rect)


# In[ ]:


img=cv2.imread('../input/global-wheat-detection/test/f5a1f0358.jpg')
img1=cv2.resize(img,(256,256))
img1=img1/255.0
img1=img1[np.newaxis,:]
box1,box2,box3=model.predict(img1)


# In[ ]:


fig,ax = plt.subplots(1)
ax.imshow(img[:,:,[2,1,0]])
boxes=[]
scores=[]
draw_boxes(box1[0],ax)
draw_boxes(box2[0],ax)
draw_boxes(box3[0],ax)
boxes=np.array(boxes)
scores=np.array(scores)
boxes=tf.convert_to_tensor(boxes)
scores=tf.convert_to_tensor(scores)
boxes=tf.cast(boxes, tf.float32)
scores=tf.cast(scores,tf.float32)
selected_indices = tf.image.non_max_suppression(
    boxes, scores, max_output_size=50, iou_threshold=0.2)
selected_boxes = tf.gather(boxes, selected_indices)
fig,ax = plt.subplots(1)
ax.imshow(img[:,:,[2,1,0]])
for i in selected_boxes:
  rect = patches.Rectangle((i[1],i[0]),i[3]-i[1],i[2]-i[0],linewidth=3,edgecolor='r',facecolor='none')
  ax.add_patch(rect)
fig.set_size_inches((12,12))
plt.show()


# In[ ]:


fig,ax = plt.subplots(1)
ax.imshow(img[:,:,[2,1,0]])
boxes=[]
scores=[]
draw_boxes(box1[0],ax)
draw_boxes(box2[0],ax)
draw_boxes(box3[0],ax)
boxes=np.array(boxes)
scores=np.array(scores)
boxes=tf.convert_to_tensor(boxes)
scores=tf.convert_to_tensor(scores)
boxes=tf.cast(boxes, tf.float32)
scores=tf.cast(scores,tf.float32)
selected_indices = tf.image.non_max_suppression(
    boxes, scores, max_output_size=50, iou_threshold=0.1)
selected_boxes = tf.gather(boxes, selected_indices)
fig,ax = plt.subplots(1)
ax.imshow(img[:,:,[2,1,0]])
for i in selected_boxes:
  rect = patches.Rectangle((i[1],i[0]),i[3]-i[1],i[2]-i[0],linewidth=3,edgecolor='r',facecolor='none')
  ax.add_patch(rect)
fig.set_size_inches((12,12))
plt.show()

