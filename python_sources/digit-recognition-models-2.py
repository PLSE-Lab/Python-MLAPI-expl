#!/usr/bin/env python
# coding: utf-8

# # &#x1F4D1; &nbsp;  Digit Recognition Models #2
# ## Links
# [SciPy. Multi-dimensional image processing](https://docs.scipy.org/doc/scipy/reference/ndimage.html)
# 
# [Keras. Deep Learning library for Theano and TensorFlow](https://keras.io/)
#  
# [TensorFlow. Deep MNIST for Experts](https://www.tensorflow.org/get_started/mnist/pros) & [Tensorflow. Deep MNIST Advanced Tutorial](http://docs.seldon.io/tensorflow-deep-mnist-example.html)
# 
# [Handwritten Digit Recognition using Convolutional Neural Networks in Python with Keras](http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)
# #### Other variants of this  project: [Digit Recognition Models](https://olgabelitskaya.github.io/kaggle_digits.html) & [Colaboratory Notebook](https://drive.google.com/open?id=1B1qh4ySXeJlWDMAXxAgHtS3jsyNdsmrn)
# #### P5: Build a Digit Recognition Program: [1](https://olgabelitskaya.github.io/MLE_ND_P5_V0_S1.html) & [2](https://olgabelitskaya.github.io/MLE_ND_P5_V0_S2.html) & [3](https://olgabelitskaya.github.io/MLE_ND_P5_V0_S3.html)
# ## Libraries
# 

# In[ ]:


import numpy as np,pandas as pd,pylab as pl
from time import time; from scipy import stats
import warnings; warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import label_propagation
from keras.utils import to_categorical
from keras.preprocessing import image as ksimage
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping 
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy,categorical_accuracy
from sklearn.metrics import accuracy_score,hamming_loss
from sklearn.metrics import classification_report,confusion_matrix
from keras.models import Sequential,Model
from keras.layers import Input,BatchNormalization,Flatten,Dropout
from keras.layers import Dense,LSTM,Activation,LeakyReLU,GlobalAveragePooling2D
from keras.layers import Conv2D,MaxPool2D,MaxPooling2D,GlobalMaxPooling2D
from keras.layers import UpSampling2D,Conv2DTranspose,DepthwiseConv2D
from keras import __version__
print('keras version:', __version__)


# In[ ]:


k=.48
wcnn='weights.best.digits.cnn.hdf5'
def hplot(history):
    pl.figure(figsize=(12,5))
    pl.plot(history.history['categorical_accuracy'][3:],'-o',label='train')
    pl.plot(history.history['val_categorical_accuracy'][3:],'-o',label='test')
    pl.legend(); pl.title('CNN Accuracy'); pl.show()
def top_3_categorical_accuracy(y_true,y_pred):
    return top_k_categorical_accuracy(y_true,y_pred,k=3)


# ## Datasets

# In[ ]:


df_train=pd.read_csv("../input/train.csv")
df_test=pd.read_csv("../input/test.csv")
print([df_train.shape,df_test.shape])
print(df_train.iloc[265,1:].values.reshape(28,28)[:,10])


# In[ ]:


images=["%s%s"%("pixel",pixel_no) for pixel_no in range(0,784)]
train_images=np.array(df_train[images])
train_images=(train_images.astype('float32')/255)**k
train_labels=df_train['label']
train_labels_cat=to_categorical(train_labels,num_classes=10)
test_images=np.array(df_test[images])
test_images=(test_images.astype('float32')/255)**k
#train_images = (train_images.astype('float32')/255)
[train_images.shape,train_labels_cat.shape,test_images.shape]


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(train_images,train_labels_cat, 
                 test_size=.2,random_state=32)
n=int(len(X_test)/2)
X_valid,y_valid=X_test[:n],y_test[:n]
X_test,y_test=X_test[n:],y_test[n:]
y_train_num=np.array([np.argmax(x) for x in y_train])
y_test_num=np.array([np.argmax(x) for x in y_test])
y_valid_num=np.array([np.argmax(x) for x in y_valid])
[X_train.shape,X_test.shape,X_valid.shape,y_train.shape,y_test.shape,y_valid.shape]


# ## Examples

# In[ ]:


fig,ax=pl.subplots(figsize=(12,2),nrows=1,ncols=10,
                   sharex=True,sharey=True)
ax=ax.flatten()
for i in range(10):
    image=train_images[i].reshape(28,28)
    ax[i].imshow(image,cmap=pl.cm.bone)
ax[0].set_xticks([]); ax[0].set_yticks([])
ax[4].set_title('Examples of symbols',fontsize=25)
pl.tight_layout(); pl.gcf(); pl.show()


# ## Models

# In[ ]:


# Model #1. Convolutional Neural Network. Keras
def cnn_model():
    model_input=Input(shape=(28,28,1))
    x=BatchNormalization()(model_input)    
    x=Conv2D(28,(5,5),padding='same')(x)
    x=LeakyReLU(alpha=.02)(x)
    x=MaxPooling2D(strides=(2,2))(x)
    x=Dropout(.25)(x)    
    x=Conv2D(96,(5,5))(x)
    x=LeakyReLU(alpha=.02)(x)
    x=MaxPooling2D(strides=(2,2))(x)
    x=Dropout(.25)(x) 
    x=GlobalMaxPooling2D()(x)    
    x=Dense(512)(x)
    x=LeakyReLU(alpha=.02)(x)
    x=Dropout(.5)(x)    
    y=Dense(10,activation='softmax')(x)    
    model=Model(input=model_input,output=y)    
    model.compile(loss='categorical_crossentropy',optimizer='nadam', 
                  metrics=[categorical_accuracy,top_3_categorical_accuracy])    
    return model
cnn_model=cnn_model()
cnn_model.summary()


# In[ ]:


checkpointer=ModelCheckpoint(filepath=wcnn,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=5,verbose=2,factor=.8)
estopping=EarlyStopping(monitor='val_loss',patience=20,verbose=2)
history=cnn_model.fit(X_train.reshape(-1,28,28,1),y_train, 
                      validation_data=(X_valid.reshape(-1,28,28,1),y_valid), 
                      epochs=100,batch_size=128,verbose=2, 
                      callbacks=[checkpointer,lr_reduction,estopping])


# In[ ]:


cnn_model.load_weights(wcnn)
hplot(history)
scores=cnn_model.evaluate(X_test.reshape(-1,28,28,1),y_test,verbose=0)
print("CNN Scores: ",(scores))
print("CNN Error: %.2f%%"%(100-scores[1]*100))


# In[ ]:


steps,epochs=1000,10
data_generator=ImageDataGenerator(featurewise_std_normalization=True,
                   zoom_range=.2,shear_range=.2,rotation_range=30,
                   height_shift_range=.2,width_shift_range=.2)
history=cnn_model.fit_generator(data_generator.flow(X_train.reshape(-1,28,28,1),
                                  y_train,batch_size=128),
              steps_per_epoch=steps,epochs=epochs,verbose=2,
              validation_data=(X_valid.reshape(-1,28,28,1),y_valid), 
              callbacks=[checkpointer,lr_reduction,estopping])


# In[ ]:


cnn_model.load_weights(wcnn)
scores=cnn_model.evaluate(X_test.reshape(-1,28,28,1),y_test,verbose=0)
print("CNN Scores: ",(scores))
print("CNN Error: %.2f%%"%(100-scores[1]*100))


# In[ ]:


Model #2. Multi-layer Perceptron. Keras
def mlp_model():
    model=Sequential()    
    model.add(Dense(784,activation='relu',input_shape=(784,)))
    model.add(Dropout(.25))
    model.add(Dense(392,activation='relu'))
    model.add(Dropout(.25))   
    model.add(Dense(196,activation='relu'))
    model.add(Dropout(.25))    
    model.add(Dense(10,activation='softmax'))
    model.compile(optimizer='nadam',loss='categorical_crossentropy', 
                  metrics=[categorical_accuracy,top_3_categorical_accuracy])
    return model
mlp_model=mlp_model()


# In[ ]:


checkpointer=ModelCheckpoint(filepath=wcnn,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=5,verbose=2,factor=.75)
estopping=EarlyStopping(monitor='val_loss',patience=20,verbose=2)
history=mlp_model.fit(X_train,y_train,validation_data=(X_valid,y_valid), 
                     epochs=100,batch_size=128,verbose=2,
                     callbacks=[checkpointer,lr_reduction,estopping])


# In[ ]:


hplot(history)
scores=mlp_model.evaluate(X_test,y_test)
print("\nMLP Scores: ",(scores))
print("MLP Error: %.2f%%"%(100-scores[1]*100))


# In[ ]:


# Model #3. Recurrent Neural Network. Keras
def rnn_model():
    model=Sequential()
    model.add(LSTM(196,return_sequences=True,input_shape=(1,784)))    
    model.add(LSTM(196,return_sequences=True))    
    model.add(LSTM(196))      
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='nadam', 
                  metrics=[categorical_accuracy,top_3_categorical_accuracy])    
    return model
rnn_model=rnn_model()


# In[ ]:


checkpointer=ModelCheckpoint(filepath=wcnn,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=5,verbose=2,factor=.75)
estopping=EarlyStopping(monitor='val_loss',patience=20,verbose=2)
history=rnn_model.fit(X_train.reshape(-1,1,X_train.shape[1]),y_train, 
                      epochs=100,batch_size=128,verbose=2, 
                      validation_data=(X_valid.reshape(-1,1,X_valid.shape[1]),y_valid),
                      callbacks=[checkpointer,lr_reduction,estopping])


# In[ ]:


hplot(history)
scores=rnn_model.evaluate(X_test.reshape(-1,1,X_test.shape[1]),y_test)
print("\nRNN Scores: ",(scores))
print("RNN Error: %.2f%%"%(100-scores[1]*100))


# In[ ]:


# Model #4. MLPClassifier. Scikit-learn
n_total=7000; X2=np.copy(X_train[:n_total])
y2=np.copy(y_train_num[:n_total]).astype('int64')
clf=MLPClassifier(hidden_layer_sizes=(784,),max_iter=100,alpha=1e-4,
                  solver='lbfgs',verbose=1,tol=1e-6,random_state=1,
                  learning_rate_init=7e-4,batch_size=128)
clf.fit(X2,y2)
print("MNIST. MLPClassifier. Train score: %f"%(clf.score(X2,y2)*100),'%')
print("MNIST. MLPClassifier. Test score: %f"%(clf.score(X_test,y_test_num)*100),'%')


# In[ ]:


# Model #5. LabelSpreading. Scikit-learn
n_total=5000; n_labeled=4000
X2=np.copy(X_train[:n_total])
y2=np.copy(y_train_num[:n_total]).astype('int64')
y2[n_labeled:]=-1
lp_model=label_propagation         .LabelSpreading(kernel='knn',n_neighbors=10,max_iter=20)
lp_model.fit(X2,y2)
predicted_labels=lp_model.transduction_[n_labeled:n_total]
true_labels=y_train_num[n_labeled:n_total]
print('Label Spreading: %d labeled & %d unlabeled points (%d total)'%
      (n_labeled,n_total-n_labeled,n_total))
print(classification_report(true_labels,predicted_labels))
print('Confusion matrix')
print(confusion_matrix(true_labels,predicted_labels,
                       labels=lp_model.classes_))
predict_entropies=stats.distributions.entropy(lp_model.label_distributions_.T)
uncertainty_index=np.argsort(predict_entropies)[-10:]


# ## Predictions

# In[ ]:


predict_labels=cnn_model.predict(test_images.reshape(-1,28,28,1))
predict_labels=predict_labels.argmax(axis=-1)


# In[ ]:


submission=pd.DataFrame({"ImageId":range(1,len(predict_labels)+1), 
                         "Label":predict_labels})
print(submission[0:10])
submission.to_csv('kaggle_digits_cnn.csv',index=False)


# In[ ]:


fig,ax=pl.subplots(figsize=(12,2),nrows=1,ncols=10,sharex=True,sharey=True)
ax=ax.flatten()
for i in range(10):
    image=test_images[i].reshape(28,28)
    ax[i].imshow(image,cmap=pl.cm.bone)
ax[0].set_xticks([]); ax[0].set_yticks([])
ax[4].set_title('Examples of digits. Test datapoints',fontsize=25)
pl.tight_layout(); pl.gcf(); pl.show()

