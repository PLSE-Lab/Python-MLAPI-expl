#!/usr/bin/env python
# coding: utf-8

# # Offline Predictions with AutoML models for Bengali Handwritten graphemes

# We have seen how to import data and train models in Google AutoML and export the models into the `saved_model` format of Tensorflow - among others (see [this Notebook](https://www.kaggle.com/wardenga/bengali-handwritten-graphemes-with-automl). In this Notebook we are going to import the `saved_model.pb` produced by AutoMl and make predictions. 

# In[ ]:


import tensorflow.compat.v1 as tf #modyficatin for tensorflow 2.1 might follow soon
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
import io
from matplotlib.image import imsave
import csv
import os
import time
import gc


# Load the model with `tf.saved_model.loader.load()` inside a `tf.Session`. Then we transform the data to an image (as in [this Notebook](https://www.kaggle.com/wardenga/bengali-handwritten-graphemes-with-automl)) since images are what we fed AutoML with. 
# 
# Note that the path fed to the loader has to be to the DIRECTORY that the `saved_model.pb` is contained in, not the file.

# In[ ]:


def make_predict_batch(img,export_path):
    """
    INPUT
        -`img` list of bytes representing the images to be classified
        
    OUTPUT
        -dataframe containing the probabilities of the labels and the la
        els as columnames
    """
    
    
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ['serve'], export_path)
        graph = tf.get_default_graph()
        
        feed_dict={'Placeholder:0':img}
        y_pred=sess.run(['Softmax:0','Tile:0'],feed_dict=feed_dict)
        
        if len(img)==1:
            labels=[label.decode() for label in y_pred[1]]
        else:
            labels=[label.decode() for label in y_pred[1][0]]
        
    return pd.DataFrame(data=y_pred[0],columns=labels)


# The actual prediction is made in the following Part of the above function (inside the `tf.Session`.
# 
# `
# feed_dict={'Placeholder:0':[imageBytearray.getvalue()]}
# y_pred=sess.run(['Softmax:0','Tile:0'],feed_dict=feed_dict)
# `
# 
# To understand How to adopt this for your pre-trained model we have to dive a bit into the structure of the model (see [this Blogpost](https://heartbeat.fritz.ai/automl-vision-edge-exporting-and-loading-tensorflow-saved-models-with-python-f4e8ce1b943a)). In fact we have to identify the input (here: 'Placeholder:0') and output nodes of the graph. Some trial and error can be involved here, especially since the last nodes in this example are not giving the actual prediction but the order of the labels, while the 'Softmax'-node actually gives the probabilities (You can look at the structure of the graph with the webapp [Netron](https://lutzroeder.github.io/netron/)). Lets look at an example prediction

# In[ ]:


i=0 
name=f'train_image_data_{i}.parquet'
test_img = pd.read_parquet('../input/bengaliai-cv19/'+name)[0:10]
test_img.head()


# In[ ]:


height=137
width=236
#we need the directory of the saved model
dir_path='../input/trained-models/Trained_Models/tf_saved_model-Bengaliai_vowel-2020-01-27T205839579Z'

images=test_img.iloc[:, 1:].values.reshape(-1, height, width)
image_id=test_img.image_id
imagebytes=[]
for i in range(test_img.shape[0]):
    imageBytearray=io.BytesIO()
    imsave(imageBytearray,images[i],format='png')
    imagebytes.append(imageBytearray.getvalue())

res=make_predict_batch(imagebytes,dir_path)
res['image_id']=image_id
res.head()


# To obtain the label run:

# In[ ]:


res.drop(['image_id'],axis=1).idxmax(axis=1)


# The following Function takes this into account and also formats a submission file following the requirements of the Bengali.Ai competition.

# In[ ]:


#walk the working directory to find the names of the directories
import os 
inputFolder = '../input/' 
for root, directories, filenames in os.walk(inputFolder): 
    for filename in filenames: print(os.path.join(root,filename))


# In[ ]:


def make_submit(images,height=137,width=236):
    """
    
    """
    consonant_path='../input/trained-models/Trained_Models/tf_saved_model-Bengaliai_consonant-2020-01-27T205840376Z'
    root_path='../input/trained-models/Trained_Models/tf_saved_model-Bengaliai_root-2020-01-27T205838805Z'
    vowel_path='../input/trained-models/Trained_Models/tf_saved_model-Bengaliai_vowel-2020-01-27T205839579Z'
    num=images.shape[0]
    #transform the images from a dataframe to a list of images and then bytes
    image_id=images.image_id
    images=images.iloc[:, 1:].values.reshape(-1, height, width)
    imagebytes=[]
    for i in range(num):
        imageBytearray=io.BytesIO()
        imsave(imageBytearray,images[i],format='png')
        imagebytes.append(imageBytearray.getvalue())
    
    #get the predictions from the three models - passing the bytes_list
    start_pred=time.time()
    prediction_root=make_predict_batch(imagebytes,export_path=root_path)
    prediction_consonant=make_predict_batch(imagebytes,export_path=consonant_path)
    prediction_vowel=make_predict_batch(imagebytes,export_path=vowel_path)
    end_pred=time.time()
    print('Prediction took {} seconds.'.format(end_pred-start_pred))
    
    start_sub=time.time()
    p0=prediction_root.idxmax(axis=1)
    p1=prediction_vowel.idxmax(axis=1)
    p2=prediction_consonant.idxmax(axis=1)
        
    row_id = []
    target = []
    for i in range(len(image_id)):
        row_id += [image_id.iloc[i]+'_grapheme_root', image_id.iloc[i]+'_vowel_diacritic',image_id.iloc[i]+'_consonant_diacritic']
        target += [p0[i], p1[i], p2[i]]
        
    submission_df = pd.DataFrame({'row_id': row_id, 'target': target})
    #submission_df.to_csv(name, index=False)
        
    end_sub=time.time()
    print('Writing the submission_df took {} seconds'.format(end_sub-start_sub))
    return submission_df


# Finally we can make the submission

# In[ ]:


with open('submission.csv','w') as sub:
    writer=csv.writer(sub)
    writer.writerow(['row_id','target'])

batchsize=1000

start = time.time()
for i in range(4):
    start1 = time.time()
    name=f'test_image_data_{i}.parquet'
    print('start with '+name+'...')
    test_img = pd.read_parquet('../input/bengaliai-cv19/'+name)
    
    print('starting prediction')
    start1 = time.time()
    #split into smaler filesl
    for r in range(np.ceil(test_img.shape[0]/batchsize).astype(int)):
            
        df=make_submit(test_img[r*batchsize:np.minimum((r+1)*batchsize,test_img.shape[0]+1)])
        df.to_csv('submission.csv',mode='a',index=False,header=False)
    
    end1 = time.time()
    print(end1 - start1)
    del test_img

end = time.time()
print(end - start)


# In[ ]:


df.head()

