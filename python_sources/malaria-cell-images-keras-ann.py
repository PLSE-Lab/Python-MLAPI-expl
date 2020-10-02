#!/usr/bin/env python
# coding: utf-8

# # Malaria Cell Images Classification
# 
# ### Accuracy - around 95%

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


import random
random_seed=33
print("Random Seed - ",random_seed)


# In[ ]:


import gc
import time
def cleanup(t=10):
    gc.collect()
    time.sleep(t)


# In[ ]:


from PIL import Image
import os
from tqdm import tqdm_notebook

main_address='../input/cell_images/cell_images/Parasitized'
images_address=os.listdir(main_address)
images_address.remove('Thumbs.db')
parasitized_data=np.zeros((len(images_address),125,125,3),dtype=np.int16)

print("Importing Parasitized Data...")
for ind,img_address in tqdm_notebook(enumerate(images_address),total=len(images_address)):
    img=Image.open(main_address+'/'+img_address)
    img=img.resize((125,125),Image.ANTIALIAS)
    img=np.asarray(img)
    img=img.astype(np.int16)
    parasitized_data[ind]=img
print("Done Importing Parasitized Data!")
cleanup()


# In[ ]:


np.random.seed(random_seed)
random_10=np.random.randint(0,parasitized_data.shape[0],size=10)
fig=plt.figure(figsize=(10,5))
plt.title("Examples of Parasitized Body Cells",fontsize=20)
plt.axis('off')
for ind,rand in enumerate(random_10):
    ax_n=fig.add_subplot(2,5,ind+1)
    ax_n.imshow(parasitized_data[rand])
    ax_n.get_xaxis().set_visible(False)
    ax_n.get_yaxis().set_visible(False)
plt.show()


# In[ ]:


main_address='../input/cell_images/cell_images/Uninfected'
images_address=os.listdir(main_address)
images_address.remove('Thumbs.db')
uninfected_data=np.zeros((len(images_address),125,125,3),dtype=np.int16)

print("Importing Uninfected Data...")
for ind,img_address in tqdm_notebook(enumerate(images_address),total=len(images_address)):
    img=Image.open(main_address+'/'+img_address)
    img=img.resize((125,125),Image.ANTIALIAS)
    img=np.asarray(img)
    img=img.astype(np.int16)
    uninfected_data[ind]=img
print("Done Importing Uninfected Data!")
cleanup()


# In[ ]:


np.random.seed(random_seed)
random_10=np.random.randint(0,uninfected_data.shape[0],size=10)
fig=plt.figure(figsize=(10,5))
plt.title("Examples of Uninfected Body Cells",fontsize=20)
plt.axis('off')
for ind,rand in enumerate(random_10):
    ax_n=fig.add_subplot(2,5,ind+1)
    ax_n.imshow(uninfected_data[rand])
    ax_n.get_xaxis().set_visible(False)
    ax_n.get_yaxis().set_visible(False)
plt.show()


# In[ ]:


parasitized_indices=np.arange(parasitized_data.shape[0])
uninfected_indices=np.arange(uninfected_data.shape[0])

np.random.seed(random_seed)
np.random.shuffle(parasitized_indices)
np.random.seed(random_seed)
np.random.shuffle(uninfected_indices)

parasitized_data=parasitized_data[parasitized_indices]
uninfected_data=uninfected_data[uninfected_indices]


# In[ ]:


parasitized_train=parasitized_data[:int(3*parasitized_data.shape[0]/4)]
uninfected_train=uninfected_data[:int(3*uninfected_data.shape[0]/4)]

parasitized_test=parasitized_data[int(3*parasitized_data.shape[0]/4):]
uninfected_test=uninfected_data[int(3*uninfected_data.shape[0]/4):]


# In[ ]:


train=np.append(parasitized_train,uninfected_train,axis=0)
test=np.append(parasitized_test,uninfected_test,axis=0)


# In[ ]:


train_labels=np.array([1]*parasitized_train.shape[0]+[0]*uninfected_train.shape[0])
test_labels=np.array([1]*parasitized_test.shape[0]+[0]*uninfected_test.shape[0])


# In[ ]:


train_index=np.arange(train.shape[0])
np.random.seed(random_seed)
np.random.shuffle(train_index)
train=train[train_index]
train_labels=train_labels[train_index]

test_index=np.arange(test.shape[0])
np.random.seed(random_seed)
np.random.shuffle(test_index)
test=test[test_index]
test_labels=test_labels[test_index]


# In[ ]:


del(parasitized_data)
del(uninfected_data)
del(parasitized_train)
del(parasitized_test)
del(uninfected_train)
del(uninfected_test)
cleanup()


# In[ ]:


total=0
for i in np.arange(train.shape[0]):
    total+=np.mean(train[i])*125*125*3
for i in np.arange(test.shape[0]):
    total+=np.mean(test[i])*125*125*3
mean=total/(train.shape[0]*125*125*3+test.shape[0]*125*125*3)

sq_sum=0
for i in np.arange(train.shape[0]):
    sq_sum+=((np.sum((train[i]-mean)**2)))
for i in np.arange(test.shape[0]):
    sq_sum+=((np.sum((test[i]-mean)**2)))
std=np.sqrt(sq_sum/(train.shape[0]*125*125*3+test.shape[0]*125*125*3))

def scaler(x):
    return (x-mean)/std


# In[ ]:


from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,BatchNormalization,Dropout,Input
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau


# In[ ]:


def build_nn_trial(dropout_rates=[0.1],nodes_order=[256]):
    nn_model_input=Input(shape=(125,125,3))
    nn_model=Conv2D(filters=16,kernel_size=(5,5),activation='relu',padding='same')(nn_model_input)
    nn_model=Conv2D(filters=16,kernel_size=(5,5),activation='relu',padding='same')(nn_model)
    nn_model=MaxPooling2D(pool_size=(2,2),strides=3)(nn_model)
    nn_model=Conv2D(filters=32,kernel_size=(10,10),activation='relu')(nn_model)
    nn_model=Conv2D(filters=64,kernel_size=(15,15),activation='relu')(nn_model)
    nn_model=MaxPooling2D(pool_size=(2,2),strides=3)(nn_model)
    nn_model=Flatten()(nn_model)
    for ind in range(len(nodes_order)):
        nn_model=Dense(nodes_order[ind],activation='relu')(nn_model)
        nn_model=BatchNormalization()(nn_model)
        nn_model=Dropout(dropout_rates[ind])(nn_model)
    nn_model_output=Dense(1,activation='sigmoid')(nn_model)
    
    nn_model=Model(inputs=nn_model_input,outputs=nn_model_output)
    nn_model.compile(
        optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return nn_model


# In[ ]:


from sklearn.model_selection import train_test_split,StratifiedKFold

train_val_x,test_x,train_val_y,test_y=train_test_split(
    train,train_labels,stratify=train_labels,test_size=0.3,random_state=random_seed)
get_ipython().run_line_magic('time', 'test_x_norm=scaler(test_x)')
cleanup(5)


# In[ ]:


def train_nn(train_val_x,train_val_y,test_norm,nn_func,get_loss=False,params=None):
    verb=1
    if(get_loss==True):
        verb=0
    
    batch_size=64
    epochs=100
    kfolds=3
    
    test_preds=np.zeros((kfolds,test_norm.shape[0]))
    ind=0
    
    kfold=StratifiedKFold(n_splits=kfolds,random_state=random_seed)
    for _,index in kfold.split(train_val_x,train_val_y):
        
        early_stopping=EarlyStopping(monitor='val_acc',patience=15,mode='max',verbose=verb)
        model_check=ModelCheckpoint('best_model.hdf5',monitor='val_acc',verbose=verb,mode='max',save_best_only=True)
        reduceLR=ReduceLROnPlateau(monitor='val_acc',patience=10,verbose=verb,mode='max')
        
        if(get_loss==False):
            print("\n\n\nFold ",ind+1)
        sample_train=train_val_x[index]
        sample_labels=train_val_y[index]
        
        x1,x2,y1,y2=train_test_split(
            sample_train,sample_labels,test_size=0.25,stratify=sample_labels,random_state=random_seed)
        x1=scaler(x1)
        x2=scaler(x2)
        nn_model=nn_func(params[0],params[1])
        nn_model.fit(
            x1,y1,validation_data=(x2,y2),epochs=epochs,batch_size=batch_size,
            verbose=verb*2,callbacks=[early_stopping,model_check,reduceLR])
        nn_model=load_model('best_model.hdf5')
        
        del(sample_train)
        del(sample_labels)
        cleanup(10)
        
        if(get_loss==True):
            loss1=nn_model.evaluate(x1,y1,verbose=0,batch_size=batch_size)
            loss2=nn_model.evaluate(x2,y2,verbose=0,batch_size=batch_size)
            del(x1)
            del(y1)
            del(x2)
            del(y2)
            cleanup(5)
            return loss1,loss2
        
        del(x1)
        del(y1)
        del(x2)
        del(y2)
        preds=nn_model.predict(test_norm,batch_size=batch_size,verbose=1)
        preds=[x[0] for x in preds]
        test_preds[ind]=np.array(preds)
        ind+=1
        cleanup(5)
    return test_preds


# In[ ]:


get_ipython().run_line_magic('time', 'test_preds=train_nn(train_val_x,train_val_y,test_x_norm,build_nn_trial,get_loss=False,params=[[0.1,0.2,0.3],[512,256,128]])')


# In[ ]:


from sklearn.metrics import accuracy_score

def threshold_checker(y1,y_true,return_best_thresh=False):
    thrs=np.arange(0.2,0.8,0.01)
    thr_scores=np.zeros(thrs.shape[0])
    for ind,thr in enumerate(thrs):
        temp=[1 if x>=thr else 0 for x in y1]
        thr_scores[ind]=accuracy_score(y_true,temp)
    plt.plot(thrs,thr_scores)
    plt.title("Threshold Scores")
    plt.xlabel("Thresholds")
    plt.ylabel("Accuracy Scores")
    plt.show()
    
    print("Max Threshold Scores")
    max_index=list(thr_scores).index(max(thr_scores))
    max_thr=thrs[max_index]
    for i in range(-2,2):
        if((max_thr+0.01*i) in thrs):
            print("Threshold {thr:.2f} - {acc:.3f}%".format(thr=max_thr+0.01*i,acc=thr_scores[max_index+i]))
    
    if(return_best_thresh==True):
        return max_thr


# In[ ]:


avg_preds=np.mean(test_preds,axis=0)


# In[ ]:


import skimage.morphology as skm
from skimage.color import rgb2gray
def dark_spots(sample,title='',to_return=False):
    black_tophat=skm.black_tophat(sample)
    black_tophat_refined=skm.black_tophat(black_tophat)
    black_tophat_refined=skm.black_tophat(black_tophat_refined)
    if(to_return==True):
        return black_tophat_refined
    
    fig=plt.figure(figsize=(14,5))
    plt.title(title+'\n',fontsize=20)
    plt.axis('off')
    ax1=fig.add_subplot(131)
    ax1.imshow(sample)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title("Image",fontsize=15)
    ax2=fig.add_subplot(132)
    ax2.imshow(black_tophat)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title("Dark Spots",fontsize=15)
    ax3=fig.add_subplot(133)
    ax3.imshow(black_tophat_refined)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.set_title("Refined Dark Spots",fontsize=15)
    plt.show()


# In[ ]:


from skimage.feature import blob_dog,blob_log,blob_doh
def identify_blobs(sample,title='',to_return=False,threshold=2.5*1e-9):
    spots=dark_spots(sample,to_return=True)
    gray_spots=rgb2gray(spots)
    final_spots=skm.white_tophat(gray_spots,selem=skm.square(10))
    log=blob_doh(final_spots,threshold=threshold,min_sigma=10,max_sigma=50)
    if(to_return==True):
        return log
    
    fig=plt.figure(figsize=(12,7))
    plt.title(title,fontsize=20)
    plt.axis('off')
    ax1=fig.add_subplot(121)
    ax1.imshow(final_spots,cmap='gray')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title("Cleaned Grayscale Image",fontsize=15)
    ax2=fig.add_subplot(122)
    ax2.imshow(sample)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    for blob in log:
        y,x,r=blob
        c=plt.Circle((x,y),r,color='red',fill=False,linewidth=2)
        ax2.add_patch(c)
    ax2.set_title("Location of Blobs",fontsize=15)
    plt.show()


# In[ ]:


def blobs(sample,threshold=2.5*1e-9):
    blobs=identify_blobs(sample,to_return=True,threshold=threshold)
    if(len(blobs)==0):
        return 0
    return len(blobs)


# In[ ]:


def get_blob_features(data,threshold=2.5*1e-9):
    print("Extracting Blob Features...")
    blobs_cnt=np.zeros(data.shape[0],dtype=np.int16)
    for ind,sample in tqdm_notebook(enumerate(data),total=data.shape[0]):
        blobs_cnt[ind]=blobs(sample,threshold=threshold)
    print("Done Extracting Blob Features!")
    return blobs_cnt


# In[ ]:


test_blob_cnt=get_blob_features(test_x)
has_blob=[1 if x>0 else 0 for x in test_blob_cnt]


# In[ ]:


final_preds=(avg_preds*2+np.array(has_blob))/3
best_thr=threshold_checker(final_preds,test_y,return_best_thresh=True)


# In[ ]:


del(test_x)
del(test_x_norm)
del(test_y)
del(train_val_x)
del(train_val_y)
cleanup(10)


# In[ ]:


get_ipython().run_line_magic('time', 'test_norm=scaler(test)')
cleanup(5)
get_ipython().run_line_magic('time', 'test_preds=train_nn(train,train_labels,test_norm,build_nn_trial,get_loss=False,params=[[0.1,0.2,0.3],[512,256,128]])')
avg_preds=np.mean(test_preds,axis=0)


# In[ ]:


test_blob_cnt=get_blob_features(test)
has_blob=[1 if x>0 else 0 for x in test_blob_cnt]


# In[ ]:


del(test)
del(test_norm)
cleanup(5)


# In[ ]:


final_preds=(2*avg_preds+np.array(has_blob))/3
predictions=[1 if x>=best_thr else 0 for x in final_preds]
accuracy=accuracy_score(test_labels,predictions)
print("Final Accuracy : {acc:.3f}%".format(acc=accuracy*100))


# In[ ]:




