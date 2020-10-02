#!/usr/bin/env python
# coding: utf-8

# # Classifier of Hand Signs

# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile
from tensorflow.keras import utils
import os
tf.logging.set_verbosity(tf.logging.INFO)

print(os.listdir("../input/"))

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


ls ../input/signsv2/signs/SIGNS


# ## Data preprocessing 

# In[ ]:


train_dir='../input/signsv2/signs/SIGNS/train_signs'
dev_dir='../input/signsv2/signs/SIGNS/dev_signs'
test_dir='../input/signsv2/signs/SIGNS/test_signs'


# In[ ]:


def load_imgs(image_dir):
    """Load images from multiple folders"""
    sub_dirs = [os.path.join(image_dir, item)
                for item in gfile.ListDirectory(image_dir)]

    sub_dirs = sorted(item for item in sub_dirs
                    if gfile.IsDirectory(item))

    # load images from multiple folders
    imgs = []
    labels=[]
    for folder in sub_dirs:
        names = os.listdir(folder)
        filenames = [os.path.join(folder, f) for f in names if f.endswith('.jpg')]
        labels.append([int(filename.split('/')[-1][0]) for filename in filenames])
        for file in filenames:
            im = plt.imread(file)
            imgs.append(im)
    
    # Preprocessing
    imgs = np.asarray(imgs).astype(np.float32, casting='safe')
    imgs /=255.0
    
    # labels
    labels=np.hstack(labels)
    
    
    return imgs, labels


# In[ ]:


train_data,y_train=load_imgs(train_dir)
dev_data,y_dev=load_imgs(dev_dir)
test_data,y_test=load_imgs(test_dir)

print("Train shape: ", train_data.shape)
print("Dev shape: ", dev_data.shape)
print("Test shape: ", test_data.shape)


# In[ ]:


plt.imshow(test_data[0,:,:,:])


# ## Model MobileNetV2 

# ## Transfer learning model

# In[ ]:


def MobileNet_V2(features,labels, mode):
    """Transfer learning with MobileNetV2 as ConvBase  
    Args:
    features: feature vector of size [batch, last layer size]
    labels: labels or categories 
    lr: learning rate
    mode: 'infer', 'eval', 'train'
    """
    # Preatrained model MobileNetV2
    model=hub.Module('https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2')
    
    # Feature Vector
    outputs=model(features['x'])
    
    # Logits
    logits= tf.layers.dense(inputs=outputs, units=6)
    
    # Predictions
    predictions={
    'classes':tf.argmax(logits,axis=1),
    'probabilities':tf.nn.softmax(logits, name="softmax_tensor")}
    
    # Predict Mode
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # loss function 
    loss=tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
    
    # Train Mode
    if mode==tf.estimator.ModeKeys.TRAIN:
        optimizer=tf.train.AdamOptimizer(learning_rate=0.01)
        train_op=optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
    
    # Evaluation metrics
    eval_metric_ops={
        "accuracy":tf.metrics.accuracy(labels=labels,predictions=predictions['classes'])
    }
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss, eval_metric_ops=eval_metric_ops)
    


# ### Estimator 

# In[ ]:


# Create Estimator
classifier= tf.estimator.Estimator(model_fn=MobileNet_V2)

# log the values in the softmax tensor with labels probabilities 
tensor_to_log={'probabilities':'softmax_tensor'}
# Print every N iterations 
logging_hook=tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=50)


# ### Training the model 

# In[ ]:


with tf.Graph().as_default() as g:
    # Input function
    train_input_fn=tf.estimator.inputs.numpy_input_fn(x={'x':train_data},y=y_train, shuffle=True)
    for epoch in range(10):
    # Train
        classifier.train(input_fn=train_input_fn, hooks=[logging_hook], steps=50000)
    
    # Evaluate model
    eval_input_fn=tf.estimator.inputs.numpy_input_fn(x={'x':dev_data}, y=y_dev,shuffle=False)
    
    # Evaluate
    eval_results=classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


# ### Save model 

# In[ ]:


def serving_input_receiver_fn():
    inputs={
        'x':tf.placeholder(tf.float32, [None,224,224,3]), 
    }
    return tf.estimator.export.ServingInputReceiver(inputs,inputs)


# In[ ]:


classifier.export_savedmodel('model',serving_input_receiver_fn=serving_input_receiver_fn)


# ### Predict

# In[ ]:


predict_fn=tf.contrib.predictor.from_estimator(classifier,serving_input_receiver_fn=serving_input_receiver_fn)


# In[ ]:


# Predict image from the first example
y_pred=predict_fn({'x':np.expand_dims(test_data[0,:,:,:],axis=0)})


# In[ ]:


class_predicted=np.argmax(y_pred['probabilities'])
print("The class predicted was: ", class_predicted)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




