#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
import seaborn as sns
from tqdm import tqdm
import keras.backend as K
from keras.utils import to_categorical
from keras.layers import Conv2DTranspose
from keras.models import Sequential, Model
from keras.layers import CuDNNLSTM, Dense,Dropout,Conv1D, MaxPool1D, Reshape, UpSampling1D, Flatten,Softmax,Activation,Add,Reshape, Input, MaxPooling1D
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
import os
import math


# Here, I am exploring the dataset and doing some basic model testing to see how far I can take the accuracy.
# 
# Here are some of the criteria for my tests:
# 
# 1. Without data augmentation - Trying to see how far the score can be taken without creating fake data.
# 2. With data augmentation - To see how much data augmentation helps in this case.
# 
# For each of the criteria above, We shall use 3 models. Two of them were made randomly. The last one is from another kernel(https://www.kaggle.com/coni57/model-from-arxiv-1805-00794)
# For each of the models, We shall do class weighting, We shall use Adam Optimizer with learning rate scheduler and a batch size of 500 (source https://www.kaggle.com/coni57/model-from-arxiv-1805-00794).
# So total 6 conditions.
# We shall also see how precision, recall and F1 score changes with model and augmentation.
# 
# ****First, Let's load the mitbh dataset
# 

# In[ ]:


data_dir="../input/heartbeat"
mit_train_file = os.path.join(data_dir,"mitbih_train.csv")
rd_mit_ds = pd.read_csv(mit_train_file, header=None,names=(["data-"+str(i) for i in range(187)]+["target"]))


# Just printing the total number of samples.

# In[ ]:


print(len(rd_mit_ds))


# That's a good number of samples if class distribution is nearly even. Now let's divide the dataframe into feature and target. The last(187th,starting from 0) column is our Y or target.

# In[ ]:


train_mit_feat=rd_mit_ds.iloc[:,:187]
train_mit_target=rd_mit_ds["target"]


# We have the data labels. Let's explore the data distribution first.
# Before that, Let's replace the class numbers(0.0-4.0) with category labels (from dataset description).

# In[ ]:


dataset_category={ 0.0:'N', 1.0:'S', 2.0:'V', 3.0:'F', 4.0:'Q'}
rd_mit_ds["target_cat"] = rd_mit_ds["target"].map(dataset_category)
sns.countplot(x="target_cat",data=rd_mit_ds)


# Well, hope of getting a well distributed dataset is everyone's dream. But sometimes dream and reality lie in completely different dimensions.
# So,one class is quite big compared to others. 'N' probably represents normal. So most people have normal heartbeat(good). Even though it's a good thing that most people have normal heartbeat, it's not so good thing for people who play with data.
# 
# Just a percentage print to add to the injury 

# In[ ]:


cnt_mit_train=Counter(rd_mit_ds["target"])
s_lbl_mit=sum([val for key,val in cnt_mit_train.items()])
for key,value in cnt_mit_train.items():
    print(key,cnt_mit_train[key]/s_lbl_mit)


# Let's do it on the test data now

# In[ ]:


mit_test_file = os.path.join(data_dir,"mitbih_test.csv")
rd_mit_ds_test = pd.read_csv(mit_test_file, header=None,names=(["data-"+str(i) for i in range(187)]+["target"]))


# In[ ]:


print(len(rd_mit_ds_test))


# In[ ]:


test_mit_feat=rd_mit_ds_test.iloc[:,:187]
test_mit_target=rd_mit_ds_test["target"]


# In[ ]:


sns.countplot(x="target",data=rd_mit_ds_test)


# In[ ]:


cnt_mit_test=Counter(test_mit_target)
s_lbl_mit_test=sum([val for key,val in cnt_mit_test.items()])
for key,value in cnt_mit_test.items():
    print(int(key),cnt_mit_test[key]/s_lbl_mit_test)


# Interestingly, the distribution is exactly the same. So it was split from one big file may be.

# That's one unbalanced dataset(both train and test). So let's prepare some test scenerios. 
# We need to make a class. Which will - 
# 1. Take model as input
# 2. Take data as input(both train and test).
# 3. Apply class weights.
# 4. Perform training with provided EarlyStopping and ModelCheckpoint and Learningrate Scheduler.
# 5. Evaluate with test data.
# 6. Get the classification report on a pandas dataframe for later analysis.

# In[ ]:


class train_and_evaluate:
    def __init__(self,
                 name,
                 model,
                 train_feat,
                 train_label,
                 test_feat,
                 test_label,
                 batch_size=64,
                 do_class_weighting=False,
                 custom_class_weights=None,
                 model_optimizer="adam",
                 metrics=['accuracy'],
                 val_split=0.2):
        self.name=name
        self.model=model
        self.train_feat=train_feat
        self.test_feat=test_feat
        self.train_label=train_label
        self.test_label=test_label
        self.do_class_weighting=do_class_weighting
        self.custom_class_weights=custom_class_weights
        self.model_optimizer=model_optimizer
        self.metrics=metrics
        self.val_split=val_split
        self.batch_size=batch_size
        #data_reshaping
        self.train_X=np.reshape(self.train_feat.values,(len(self.train_feat),187,1))
        self.train_Y=to_categorical(self.train_label,num_classes=5)
        self.test_X=np.reshape(self.test_feat.values,(len(self.test_feat),187,1))
        self.test_Y=to_categorical(self.test_label,num_classes=5)
    def exp_decay(self,epoch):
        initial_lrate = 0.001
        k = 0.75
        t = self.train_X.shape[0]//(10000 * self.batch_size)  # every epoch we do n_obs/batch_size iteration
        lrate = initial_lrate * math.exp(-k*t)
        return lrate
    def train(self,
              eps=100):
        class_weights=None
        T_X,V_X,T_Y,V_Y=train_test_split(self.train_X,self.train_Y,test_size=self.val_split,random_state=42)
        if self.do_class_weighting==True:
            if self.custom_class_weights is not None:
                class_weights=self.class_weights
            else:
                class_weights=class_weight.compute_class_weight('balanced',
                                                np.unique(T_Y.argmax(axis=1)),
                                                T_Y.argmax(axis=1))
        es=EarlyStopping(patience=5)
        mcp=ModelCheckpoint(filepath="weights_{}.h5".format(self.name),save_best_only=True,save_weights_only=True)
        lrate = LearningRateScheduler(self.exp_decay)
        self.model.compile(loss="categorical_crossentropy",optimizer=self.model_optimizer,metrics=self.metrics)
        self.model.fit(T_X,T_Y,batch_size=self.batch_size,epochs=eps,verbose=1,validation_data=[V_X,V_Y],class_weight=class_weights,callbacks=[lrate,es,mcp])
    def evaluate(self):
        self.model.load_weights("weights_{}.h5".format(self.name))
        preds=self.model.predict(self.test_X)
        print(self.model.evaluate(self.test_X,self.test_Y))
        return confusion_matrix(self.test_label,preds.argmax(axis=1)),classification_report(self.test_label,preds.argmax(axis=1),output_dict=True)


# In[ ]:


#introducing model 1, small and cute Conv1D and LSTM based model.
def model_initial():
    mdl=Sequential()
    mdl.add(Conv1D(64,(3,),activation="relu",input_shape=(187,1),padding="same"))
    mdl.add(MaxPool1D())
    mdl.add(CuDNNLSTM(93))
    mdl.add(Dense(50,activation="relu"))
    mdl.add(Dense(5,activation="softmax"))
    return mdl
#Now comes model 2, no LSTM, Lot bigger but based on conv and pooling layers and some dense layers
def model_cnn():
    mdl=Sequential()
    mdl.add(Conv1D(32,(3,),activation="relu",input_shape=(187,1),padding="same"))
    mdl.add(Conv1D(32,(3,),activation="relu"))
    mdl.add(Conv1D(32,(3,),activation="relu"))
    mdl.add(MaxPool1D())
    mdl.add(Conv1D(32,(3,),activation="relu"))
    mdl.add(Conv1D(32,(3,),activation="relu"))
    mdl.add(MaxPool1D())
    mdl.add(Flatten())
    mdl.add(Dense(50,activation="relu"))
    mdl.add(Dense(5,activation="softmax"))
    return mdl
#The final, super big model from that nice notebook
def model_copied():
    inp = Input(shape=(187, 1))
    C = Conv1D(filters=32, kernel_size=5, strides=1)(inp)

    C11 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(C)
    A11 = Activation("relu")(C11)
    C12 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A11)
    S11 = Add()([C12, C])
    A12 = Activation("relu")(S11)
    M11 = MaxPooling1D(pool_size=5, strides=2)(A12)


    C21 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M11)
    A21 = Activation("relu")(C21)
    C22 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A21)
    S21 = Add()([C22, M11])
    A22 = Activation("relu")(S11)
    M21 = MaxPooling1D(pool_size=5, strides=2)(A22)


    C31 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M21)
    A31 = Activation("relu")(C31)
    C32 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A31)
    S31 = Add()([C32, M21])
    A32 = Activation("relu")(S31)
    M31 = MaxPooling1D(pool_size=5, strides=2)(A32)


    C41 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M31)
    A41 = Activation("relu")(C41)
    C42 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A41)
    S41 = Add()([C42, M31])
    A42 = Activation("relu")(S41)
    M41 = MaxPooling1D(pool_size=5, strides=2)(A42)


    C51 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M41)
    A51 = Activation("relu")(C51)
    C52 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A51)
    S51 = Add()([C52, M41])
    A52 = Activation("relu")(S51)
    M51 = MaxPooling1D(pool_size=5, strides=2)(A52)

    F1 = Flatten()(M51)

    D1 = Dense(32)(F1)
    A6 = Activation("relu")(D1)
    D2 = Dense(32)(A6)
    D3 = Dense(5)(D2)
    A7 = Softmax()(D3)

    model = Model(inputs=inp, outputs=A7)
    return model


# <H1>Case 1: No augmentation</H1>

# In[ ]:


global_conf_mat={}
global_class_rpt={}


# In[ ]:


import keras.backend as K
def get_dict(name):
    global global_class_rpt
    global global_conf_mat
    model_arr=[model_initial,model_cnn,model_copied]
    dict_conf_mat={}
    dict_classification_rpt={}
    for model in model_arr:
        print("current model: ",model.__name__)
        K.clear_session()
        model_to_use=model()
        trainer_obj  =  train_and_evaluate(name=model.__name__+"_class_wt_no_aug",
                                           model=model_to_use,
                                           train_feat=train_mit_feat,
                                           train_label=train_mit_target,
                                           test_feat=test_mit_feat,
                                           test_label=test_mit_target,
                                           batch_size=500,
                                           do_class_weighting=True,
                                           model_optimizer=Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999))
        trainer_obj.train()
        conf_mat,class_rpt=trainer_obj.evaluate()
        dict_conf_mat[model.__name__]=conf_mat
        dict_classification_rpt[model.__name__]=class_rpt
    global_conf_mat[name]=dict_conf_mat
    global_class_rpt[name]=dict_classification_rpt


# In[ ]:


get_dict("no_aug")


# In[ ]:


global_class_rpt


# In[ ]:


import pickle
with open("class_rpt_no_aug.pkl","wb") as f:
    pickle.dump(global_class_rpt,f)
with open("conf_mat_no_aug.pkl","wb") as f:
    pickle.dump(global_conf_mat,f)
    


# In[ ]:


df_new=pd.DataFrame(global_class_rpt)


# In[ ]:


total_lst=[]
for key_aug,value_model in global_class_rpt.items():
    if isinstance(value_model,dict):
        for key_model,value_class in value_model.items():
            if isinstance(value_class,dict):
                for key_class,value_params in value_class.items():
                    if isinstance(value_params,dict):
                        try:
                            lst_item=[key_aug,key_model,int(float(key_class)),value_params["precision"],value_params["recall"],value_params["f1-score"]]
                            total_lst.append(lst_item)
                        except:
                            pass


# In[ ]:


df_res=pd.DataFrame.from_records(total_lst,columns=["is_aug","model","class","precision","recall","f1_score"])


# In[ ]:


df_res.head(10)


# In[ ]:


df_res.to_csv("no_aug.csv")

