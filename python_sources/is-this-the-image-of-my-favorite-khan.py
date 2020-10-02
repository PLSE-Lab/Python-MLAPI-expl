#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -I keras==2.1.6')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras import backend as K
import keras.preprocessing.image as keras_img
import matplotlib.pyplot as plt
import PIL.Image as Image
import os
from keras.models import load_model, Model
print(os.listdir("../input"))
from PIL import Image
from shutil import copyfile
copyfile(src = "../input/facenetkaggle/fr_utils.py", dst = "../working/fr_utils.py")
copyfile(src = "../input/facenetkaggle/inception_blocks_v2.py", dst = "../working/inception_blocks_v2.py")
from fr_utils import *
from inception_blocks_v2 import *
from keras.callbacks import EarlyStopping
import keras


# In[ ]:


img_size=96
def load_image(path):
    try:
        img = Image.open(path)
        img = img.resize((img_size,img_size), Image.ANTIALIAS)
    except:
        img=None
        return None
    return np.array(img)


def load_data():
    train_data_path = "../input/bollywood/q1_trainingdata/Q1_TrainingData/"
    classes = os.listdir(train_data_path)
    arr_df = []
    for i in range(len(classes)):
        class_path = classes[i]
        file_list = os.listdir(train_data_path + class_path)
        df = pd.DataFrame({"file":file_list})
        df["class"] = i
        df["class_name"] = classes[i]
        arr_df.append(df)

    train_data = pd.concat(arr_df)
    train_data = train_data.reset_index()
    train_data["img"] = train_data.apply(lambda x: load_image(train_data_path + x["class_name"] + "/" + x["file"]), axis=1)
    return train_data


# In[ ]:


K.set_image_data_format('channels_first')
def load_facenet_weights():
    global FRmodel
    facedir = "../input/facenetkaggle/all_weights/"
    fileNames = os.listdir(facedir)
    filePaths = [facedir + fname for fname in fileNames]
    paths = {}
    weights_dict = {}
    for n in range(len(fileNames)):
            paths[fileNames[n].replace('.csv', '')] = filePaths[n]
    load_weights_from_FaceNet(FRmodel, fileNames, filePaths)
    
def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    return loss


def img_to_fnencoding(img, model):
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding


def who_is_it(val_idx, val_img, encoding, true_class):
#encoding = img_to_fnencoding(val_img, FRmodel) 
    train_idx = list(set(list(train_data.index)) - set([val_idx]))

    train_data1 = train_data.loc[train_idx]
    train_data1["dist"] = train_data1["fn_encoding"].map(lambda x: np.linalg.norm(encoding- x))
    
    idx = train_data1.sort_values("dist").index[0]
    pos_data = train_data1[train_data1["class_name"]==true_class].sort_values("dist")
    neg_data = train_data1[train_data1["class_name"]!=true_class].sort_values("dist")
    pos_idx1, pos_idx2, pos_idx3 = pos_data[0:1].index[0], pos_data[1:2].index[0], pos_data[2:3].index[0]
    neg_idx1, neg_idx2, neg_idx3 = neg_data[0:1].index[0], neg_data[1:2].index[0], neg_data[2:3].index[0]
    print(train_data1.loc[idx, "class_name"],idx, pos_idx1, pos_idx2, pos_idx3, neg_idx1, neg_idx2, neg_idx3)
    return train_data1.loc[idx, "class_name"],idx, pos_idx1, pos_idx2, pos_idx3, neg_idx1, neg_idx2, neg_idx3


def batch_generator(batch_size = 32):
    while True:
        sample_review_idx = np.random.randint(0, review_idx.shape[0], batch_size)
        idx= review_idx[sample_review_idx]
        pos_col_idx = np.random.randint(1,4)
        neg_col_idx = np.random.randint(1,4)
        pos_idx = train_data.loc[idx,"pos_idx" + str(pos_col_idx)].values
        neg_idx = train_data.loc[idx,"neg_idx" + str(neg_col_idx)].values
        anc = train_data.loc[idx,"img"].apply(lambda x: np.around(np.transpose(x, (2,0,1))/255.0, decimals=12)).values
        neg = train_data.loc[neg_idx,"img"].apply(lambda x: np.around(np.transpose(x, (2,0,1))/255.0, decimals=12)).values
        pos = train_data.loc[pos_idx,"img"].apply(lambda x: np.around(np.transpose(x, (2,0,1))/255.0, decimals=12)).values

        arr = []
        for val in anc:
            arr.append(val)
        anc = np.array(arr)

        arr = []
        for val in neg:
            arr.append(val)
        neg = np.array(arr)

        arr = []
        for val in pos:
            arr.append(val)
        pos = np.array(arr)

        x_data = {'anchor': anc,
                  'anchorPositive': pos,
                  'anchorNegative': neg
                  }
        yield (x_data, np.zeros((batch_size, 2, 1)))
    
    
def triplet_loss_v2(y_true, y_pred):
    positive, negative = y_pred[:,0,0], y_pred[:,1,0]
    margin = K.constant(0.35)
    loss = K.mean(K.maximum(K.constant(0), positive - negative + margin))
    return loss

def euclidean_distance(vects):
    x, y = vects
    dist = K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
    return dist



# Setting layers non-trainable

def get_triplet_model():
    global FRmodel
    # Model Structure
    input_shape=(3, 96, 96)
    anchor = Input(shape=input_shape, name = 'anchor')
    anchorPositive = Input(shape=input_shape, name = 'anchorPositive')
    anchorNegative = Input(shape=input_shape, name = 'anchorNegative')

    anchorCode = FRmodel(anchor)
    anchorPosCode = FRmodel(anchorPositive)
    anchorNegCode = FRmodel(anchorNegative)


    positive_dist = Lambda(euclidean_distance, name='pos_dist')([anchorCode, anchorPosCode])
    negative_dist = Lambda(euclidean_distance, name='neg_dist')([anchorCode, anchorNegCode])
    stacked_dists = Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')([positive_dist, negative_dist])
    # Model
    tripletModel = Model([anchor, anchorPositive, anchorNegative], stacked_dists, name='triple_siamese')
    tripletModel.compile(optimizer = 'adadelta', loss = triplet_loss_v2, metrics = ['accuracy'])
    
    return tripletModel

def retrain_model():
    global FRmodel, tripletModel
    
    for layer in FRmodel.layers:
        layer.trainable = True
        
    for layer in FRmodel.layers[0:80]:
        layer.trainable = False
        
    gen = batch_generator(64)

    early_stopping = EarlyStopping(monitor='loss', patience=5, min_delta=0.00005)
    tripletModel.compile(optimizer = 'adadelta', loss = triplet_loss_v2, metrics = ['accuracy'])
    tripletModel.fit_generator(gen, epochs=100,steps_per_epoch=30,callbacks=[early_stopping])
    
    for layer in FRmodel.layers[0: 100]:
        layer.trainable  =  False
        
    gen = batch_generator(64)
    
    early_stopping = EarlyStopping(monitor='loss', patience=10, min_delta=0.0000001)
    tripletModel.compile(optimizer = 'adadelta', loss = triplet_loss_v2, metrics = ['accuracy'])
    tripletModel.fit_generator(gen, epochs=1,steps_per_epoch=30,callbacks=[early_stopping])

    #bst_model_path = "FRmodel"
    #FRmodel.save(bst_model_path)


# In[ ]:


def get_acc():
    global FRmodel
    train_data["fn_encoding"] = train_data["img"].map(lambda x: img_to_fnencoding(x, FRmodel))
    acc=0
    review_idx = []
    for val_idx in train_data.index.values:
        val_img = train_data.loc[val_idx, "img"]
        true_class = train_data.loc[val_idx, "class_name"]
        fn_encoding = train_data.loc[val_idx, "fn_encoding"]
        #plt.imshow(val_img)
        pred_class, pred_idx, pos_idx1, pos_idx2, pos_idx3, neg_idx1, neg_idx2, neg_idx3 = who_is_it(val_idx, val_img, fn_encoding, true_class)
        train_data.loc[val_idx,"pred_idx"] = pred_idx
        train_data.loc[val_idx,"pos_idx1"] = pos_idx1
        train_data.loc[val_idx,"pos_idx2"] = pos_idx2
        train_data.loc[val_idx,"pos_idx3"] = pos_idx3
        train_data.loc[val_idx,"neg_idx1"] = neg_idx1
        train_data.loc[val_idx,"neg_idx2"] = neg_idx2
        train_data.loc[val_idx,"neg_idx3"] = neg_idx3
        train_data.loc[val_idx,"pred_class_name"] = pred_class
        if true_class == pred_class:
            acc = acc+1
        else:
            review_idx.append(val_idx)
        #print(true_class, pred_class, acc)
    print(acc)
    review_idx = np.array(review_idx)
    return review_idx


def get_val_acc():
    global FRmodel
    val_data["fn_encoding"] = val_data["img"].map(lambda x: img_to_fnencoding(x, FRmodel))
    acc=0
    list_idx = []
    list_pred_idx = []
    for val_idx in val_data.index.values:
        val_img = val_data.loc[val_idx, "img"]
        true_class = val_data.loc[val_idx, "class_name"]
        fn_encoding = val_data.loc[val_idx, "fn_encoding"]
        #plt.imshow(val_img)
        pred_class, pred_idx, pos_idx1, pos_idx2, pos_idx3, neg_idx1, neg_idx2, neg_idx3 = who_is_it(val_idx, val_img, fn_encoding, true_class)
        if true_class == pred_class:
            acc = acc+1
        else:
            list_idx.append(val_idx)
            list_pred_idx.append(pred_idx)
        print(true_class, pred_class, acc)
    print(acc)
    return list_idx, list_pred_idx


# In[ ]:


train_data = load_data()


# In[ ]:


train_data.groupby("class").count()


# In[ ]:


cols = list(train_data.columns.values)
train_data = train_data[train_data["img"].isna()==False].reset_index()[cols]


# In[ ]:


fig, ax = plt.subplots(10,10, figsize=(20,20))
i,j = 0,0
for idx in range(0,100):
    img = train_data.loc[idx,"img"]
    ax[i,j].imshow(img)
    j=j+1
    if (j==10):
        j=0
        i=i+1


# In[ ]:


fig, ax = plt.subplots(10,10, figsize=(20,20))
i,j = 0,0
for idx in range(100,200):
    img = train_data.loc[idx,"img"]
    ax[i,j].imshow(img)
    j=j+1
    if (j==10):
        j=0
        i=i+1


# In[ ]:


fig, ax = plt.subplots(10,10, figsize=(20,20))
i,j = 0,0
for idx in range(200,300):
    img = train_data.loc[idx,"img"]
    ax[i,j].imshow(img)
    j=j+1
    if (j==10):
        j=0
        i=i+1


# In[ ]:


fig, ax = plt.subplots(10,10, figsize=(20,20))
i,j = 0,0
for idx in range(300,400):
    img = train_data.loc[idx,"img"]
    ax[i,j].imshow(img)
    j=j+1
    if (j==10):
        j=0
        i=i+1


# In[ ]:


fig, ax = plt.subplots(10,10, figsize=(20,20))
i,j = 0,0
for idx in range(400,500):
    img = train_data.loc[idx,"img"]
    ax[i,j].imshow(img)
    j=j+1
    if (j==10):
        j=0
        i=i+1


# In[ ]:


fig, ax = plt.subplots(10,10, figsize=(20,20))
i,j = 0,0
for idx in range(500,600):
    img = train_data.loc[idx,"img"]
    ax[i,j].imshow(img)
    j=j+1
    if (j==10):
        j=0
        i=i+1


# In[ ]:


fig, ax = plt.subplots(10,10, figsize=(20,20))
i,j = 0,0
for idx in range(600,700):
    img = train_data.loc[idx,"img"]
    ax[i,j].imshow(img)
    j=j+1
    if (j==10):
        j=0
        i=i+1


# In[ ]:


fig, ax = plt.subplots(10,10, figsize=(20,20))
i,j = 0,0
for idx in range(700,800):
    img = train_data.loc[idx,"img"]
    ax[i,j].imshow(img)
    j=j+1
    if (j==10):
        j=0
        i=i+1


# In[ ]:


fig, ax = plt.subplots(10,10, figsize=(20,20))
i,j = 0,0
for idx in range(900,1000):
    img = train_data.loc[idx,"img"]
    ax[i,j].imshow(img)
    j=j+1
    if (j==10):
        j=0
        i=i+1


# In[ ]:


fig, ax = plt.subplots(10,10, figsize=(20,20))
i,j = 0,0
for idx in range(1000,1100):
    img = train_data.loc[idx,"img"]
    ax[i,j].imshow(img)
    j=j+1
    if (j==10):
        j=0
        i=i+1


# In[ ]:


fig, ax = plt.subplots(10,10, figsize=(20,20))
i,j = 0,0
for idx in range(1100,1200):
    img = train_data.loc[idx,"img"]
    ax[i,j].imshow(img)
    j=j+1
    if (j==10):
        j=0
        i=i+1


# In[ ]:


train_data.columns


# In[ ]:


cols = list(train_data.columns.values) 
val_data_idx = np.random.randint(0, train_data.shape[0], 100)
val_data = train_data.loc[val_data_idx].reset_index()[cols]
train_data_idx = list(set(train_data.index) - set(val_data_idx))
train_data = train_data.loc[train_data_idx].reset_index()[cols]


# In[ ]:


FRmodel = load_model("../input/is-this-the-image-of-my-favorite-khan/KhanModel", custom_objects={"triplet_loss":triplet_loss})


# In[ ]:





# In[ ]:


if 1==2:
    FRmodel = faceRecoModel(input_shape=(3,img_size, img_size))
    load_facenet_weights()
    for layer in FRmodel.layers:
        layer.trainable = True
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])


# In[ ]:


review_idx = get_acc()


# In[ ]:


tripletModel = get_triplet_model()
retrain_model()
review_idx = get_acc()


# In[ ]:


#tripletModel = get_triplet_model(FRmodel)
retrain_model()
#review_idx = get_acc()


# In[ ]:


val_data = val_data[val_data["img"].isna()==False]
val_idx_incorrect, pred_idx_correct = get_val_acc()
#FRmodel = load_model(bst_model_path, custom_objects={"triplet_loss":triplet_loss}, compile=True)
#review_idx = get_acc()


# In[ ]:


len(val_idx_incorrect)


# # So now that we have great accuracy on train data we look good to validate it on test images!!!

# In[ ]:


fig, ax = plt.subplots(len(val_idx_incorrect),2, figsize=(20,20))
i,j = 0,0
for i in range(len(val_idx_incorrect)):
    img = val_data.loc[val_idx_incorrect[i],"img"]
    ax[i,0].imshow(np.array(img))
    img = train_data.loc[pred_idx_correct[i],"img"]
    ax[i,1].imshow(np.array(img))


# In[ ]:


FRmodel.save("KhanModel")


# In[ ]:



full_train_data = pd.concat([train_data[["class_name","fn_encoding"]], val_data[["class_name","fn_encoding"]]])
from IPython.display import HTML
import base64
full_train_data["str_encoding"] = full_train_data["fn_encoding"].map(lambda x: "^".join(list(x.flatten().astype(str))))
full_train_data[["class_name","str_encoding"]].to_csv("model_encoding.csv", index=False)


def create_download_link(title = "Download CSV file", filename='model_encoding.csv'):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

    


# In[ ]:


create_download_link()

