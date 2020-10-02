#!/usr/bin/env python
# coding: utf-8

# ## Cats vs Dogs ConvNet Keras

# ## [Please star/upvote in case u like it.]

# ## TABLE OF CONTENTS::

# [ **1 ) Importing Various Modules**](#content1)

# [ **2 ) Prepare the Data**](#content2)

# [ **3 ) Modelling**](#content3)

# [ **4 )  Evaluating the Model Performance**](#content4)

# [ **5 ) Visualizing Predictons on the Validation Set**](#content5)

# [ **6 ) Making Predictions on the Test Set**](#content6)

# [ **7 ) Saving Submissions onto a CSV**](#content7)

# <a id="content1"></a>
# ## 1 ) Importing Various Modules

# In[ ]:


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical

# specifically for cnn
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 
import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np         
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image
from tqdm import tqdm 


# <a id="content2"></a>
# ## 2 ) Prepare the Data

# ## 2.1) Making the functions to get the training ,testing and validation set from the Images

# I have created the training and validation sets. Much of this section is inspired from the [**sentdex** ](https://www.kaggle.com/sentdex/full-classification-example-with-convnet)work.

# In[ ]:


TRAIN_DIR = '../input/train'
TEST_DIR = '../input/test'
IMG_SIZE=100


# In[ ]:


def label_img(img):
    word_label = img.split('.')[0]
    return word_label


# In[ ]:


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label=label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),str(label)])
        
    shuffle(training_data)
    return training_data


# In[ ]:


train_data=create_train_data()
train_data=np.array(train_data)
print(train_data.shape)
X= np.array([i[0] for i in train_data]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y= np.array([i[1] for i in train_data])


# In[ ]:


def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img)])
        
    shuffle(testing_data)
    return testing_data


# ## 2.2 ) Visualizing some Random Images

# In this section I have visualized the dataset with just some random images. This confirms that dataset has been loaded correctly.

# In[ ]:


fig,ax=plt.subplots(6,2)
fig.set_size_inches(15,15)
for i in range(6):
    for j in range (2):
        l=rn.randint(0,len(Y))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Pet: '+Y[l])
        
plt.tight_layout()


# ## 2.3 ) One Hot Encoding

# Encoding the target as usual.

# In[ ]:


le=LabelEncoder()
Z=Y
Y=le.fit_transform(Y)
Y=to_categorical(Y)


# ## 2.4 ) Splitting into Training and Validation Sets

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


# ## 2.5 ) Setting the Random seeds

# Setting the seeds for ensuring reproducability. the seeds are of numpy,tensorflow and the python.

# In[ ]:


np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)


# <a id="content3"></a>
# ## 3 ) Modelling

# ## 3.1 ) Building the ConvNet Model

# In[ ]:


# # modelling starts using a CNN.

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (100,100,3)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
 
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 
model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dense(2, activation = "sigmoid"))
 


# In[ ]:


batch_size=128
epochs=20

from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=2,verbose=1,factor=0.1)


# ## 3.2 ) Data Augmentation to prevent Overfitting

# In[ ]:


datagen = ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.1,
    rotation_range=10,
    horizontal_flip=True)
datagen.fit(x_train)


# ## 3.3 ) Compiling the Keras Model

# In[ ]:


model.compile(optimizer=RMSprop(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])


# ## 3.4 ) Summary of the Model

# In[ ]:


model.summary()


# ## 3.5 ) Fitting on the Training set and making predcitons on the Validation set

# In[ ]:


History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size,callbacks=[red_lr])


# ####  The  final validation accuracy is close to 89 or 90 % which is quite decent.           
# Tuning in can definitely get an accuracy better than this,

# <a id="content4"></a>
# ## 4 ) Evaluating the Model Performance

# In[ ]:


plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


# In[ ]:


plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


# <a id="content5"></a>
# ## 5) Visualizing Predictons on the Validation Set

# **Just writing some functions to see some correct results and also misclassified images. This just makes it a bit more clearer and attractive.;)**

# In[ ]:


# getting predictions on val set.
pred=model.predict(x_test)
pred_pet=np.argmax(pred,axis=1)


# In[ ]:


pred_pet


# In[ ]:


# now storing some properly as well as misclassified indexes'.
i=0
prop_class=[]
mis_class=[]

for i in range(len(y_test)):
    if(np.argmax(y_test[i])==pred_pet[i]):
        prop_class.append(i)
    if(len(prop_class)==8):
        break

i=0
for i in range(len(y_test)):
    if(not np.argmax(y_test[i])==pred_pet[i]):
        mis_class.append(i)
    if(len(mis_class)==8):
        break


# #### PROPERLY CLASSIFIED IMAGES

# In[ ]:


warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(15,15)
for i in range (4):
    for j in range (2):
        ax[i,j].imshow(x_test[prop_class[count]])
        ax[i,j].set_title("Predicted Pet : "+str(le.inverse_transform([pred_pet[prop_class[count]]]))+"\n"+"Actual Pet : "+str(le.inverse_transform([np.argmax([y_test[prop_class[count]]])])))
        plt.tight_layout()
        count+=1


# #### MISCLASSIFIED PET IMAGES

# In[ ]:


warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(15,15)
for i in range (4):
    for j in range (2):
        ax[i,j].imshow(x_test[mis_class[count]])
        ax[i,j].set_title("Predicted Pet : "+str(le.inverse_transform([pred_pet[mis_class[count]]]))+"\n"+"Actual Pet : "+str(le.inverse_transform([np.argmax([y_test[mis_class[count]]])])))
        plt.tight_layout()
        count+=1


# <a id="content6"></a>
# ## 6 ) Making predictions on the Test Set

# In[ ]:


test_data=create_test_data()
test_data=np.array(test_data)
print(test_data.shape)
test= np.array([i[0] for i in test_data]).reshape(-1,IMG_SIZE,IMG_SIZE,3)


# In[ ]:


pred=model.predict(test)


# <a id="content7"></a>
# ## 7 ) Saving Submissions onto a CSV

# Generating a csv file to make predictions onto Kaggle.

# In[ ]:


imageid=[]
pred_prob=[]
for i in range(len(test)):
    imageid.append(i+1)
    pred_prob.append(pred[i,1])
   
d={'id':imageid,'label':pred_prob}
ans=pd.DataFrame(d)
ans.to_csv('predictions.csv',index=False)


# In[ ]:





# ## THE END!!!

# ## [Please star/upvote if u liked it.]

# In[ ]:




