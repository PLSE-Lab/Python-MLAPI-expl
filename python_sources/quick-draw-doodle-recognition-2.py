#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:steelblue; font-family:Ewert; font-size:150%;" class="font-effect-3d">Code Library, Style and Links</h1>
# 
# The previous notebook - [Quick, Draw! Doodle Recognition 1](https://www.kaggle.com/olgabelitskaya/quick-draw-doodle-recognition-1)

# In[ ]:


get_ipython().run_cell_magic('html', '', "<style>\n@import url('https://fonts.googleapis.com/css?family=Ewert|Roboto&effect=3d|ice|');\nspan {font-family:'Roboto'; color:black; text-shadow: 5px 5px 5px #aaa;}  \ndiv.output_area pre{font-family:'Roboto'; font-size:110%; color: steelblue;}      \n</style>")


# In[ ]:


import numpy as np,pandas as pd,keras as ks
import os,ast,warnings
import pylab as pl
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation,Dropout,Dense,Conv2D, MaxPooling2D, GlobalMaxPooling2D
warnings.filterwarnings('ignore')
pl.style.use('seaborn-whitegrid')
style_dict={'background-color':'gainsboro','color':'steelblue', 
            'border-color':'white','font-family':'Roboto'}
os.listdir("../input")


# In[ ]:


fpath='../input/quickdraw-doodle-recognition/train_simplified/'
files=os.listdir(fpath)
labels=[el.replace(" ","_")[:-4] for el in files]
print(sorted(labels)) # 340 labels - 17 sets


# In[ ]:


wpath='../input/quick-draw-model-weights-for-doodle-recognition/weights/'
weights=sorted(os.listdir(wpath))
print(weights) # files with weights for 17 label sets


# In[ ]:


I=64 # image size in pixels
T=20 # number of labels in one set


# In[ ]:


# https://stackoverflow.com/questions/25837544/get-all-points-of-a-straight-line-in-python
def get_line(x1,y1,x2,y2):
    steep=abs(y2-y1)>abs(x2-x1)
    if steep: x1,y1,x2,y2=y1,x1,y2,x2
    rev=False
    if x1>x2:
        x1,x2,y1,y2=x2,x1,y2,y1
        rev=True
    dx=x2-x1; dy=abs(y2-y1)
    error=int(dx/2)
    xy=[]; y=y1; ystep=None
    if y1<y2: ystep=1
    else: ystep=-1
    for x in range(x1,x2+1):
        if steep: xy.append([y,x])
        else: xy.append([x,y])
        error-=dy
        if error<0:
            y+=ystep
            error+=dx
    if rev: xy.reverse()
    return xy
def display_drawing():
    pl.figure(figsize=(10,10))
    pl.suptitle('Test Pictures')
    for i in range(20):
        picture=ast.literal_eval(test_data.drawing.values[i])
        for x,y in picture:
            pl.subplot(5,4,i+1)
            pl.plot(x,y,'-o',markersize=1,color='slategray')
            pl.xticks([]); pl.yticks([])
            pl.title(submission.iloc[i][1])
        pl.gca().invert_yaxis()
        pl.axis('equal');           
def get_image(data,k,I=I):
    img=np.zeros((280,280))
    picture=ast.literal_eval(data.values[k])
    for x,y in picture:
        for i in range(len(x)):
            img[y[i]+10][x[i]+10]=1
            if (i<len(x)-1):
                x1,y1,x2,y2=x[i],y[i],x[i+1],y[i+1]
            else:
                x1,y1,x2,y2=x[i],y[i],x[0],y[0]
            for [xl,yl] in get_line(x1,y1,x2,y2):
                img[yl+10][xl+10]=1                
    return resize(img,(I,I))    


# <h1 style="color:steelblue; font-family:Ewert; font-size:150%;" class="font-effect-3d">Test Data Exploration</h1>

# In[ ]:


ftest='../input/quickdraw-doodle-recognition/test_simplified.csv'
test_data=pd.read_csv(ftest,index_col='key_id')
test_data.tail(3).T.style.set_properties(**style_dict)


# In[ ]:


# creating test images in pixels
test_images=[]
test_images.extend([get_image(test_data.drawing,i) 
                    for i in range(len(test_data))])
test_images=np.array(test_images)
test_images.shape


# In[ ]:


pl.figure(figsize=(10,5))
pl.subplot(1,2,1); pl.imshow(test_images[0])
pl.subplot(1,2,2); pl.imshow(test_images[10000])
pl.suptitle('Key Points in the Test Pictures');


# <h1 style="color:steelblue; font-family:Ewert; font-size:150%;" class="font-effect-3d">The Model</h1>

# In[ ]:


def model():
    model=Sequential()  
    model.add(Conv2D(32,(5,5),padding='same',
                     input_shape=(I,I,1)))
    model.add(LeakyReLU(alpha=.02))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(.2))
    model.add(Conv2D(196,(5,5)))
    model.add(LeakyReLU(alpha=.02))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(.2))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dropout(.5)) 
    model.add(Dense(T))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    return model
model=model()


# <h1 style="color:steelblue; font-family:Ewert; font-size:200%;" class="font-effect-3d">Predictions</h1>

# In[ ]:


fn1='../input/quick-draw-model-weights-for-doodle-recognition/'+    'weights/weights.best.model001_020.hdf5'
model.load_weights(fn1)
test_predictions=model.predict(test_images.reshape(-1,I,I,1))
test_predictions.shape


# In[ ]:


# separated predictions for each label set
for w in weights[1:]:
    w=wpath+w
    model.load_weights(w)
    test_predictions2=model.predict(test_images.reshape(-1,I,I,1))
    test_predictions=np.concatenate((test_predictions,
                                     test_predictions2),axis=1)
test_predictions.shape


# In[ ]:


# 3 best guesses among all label sets
test_labels=[[labels[i] 
              for i in test_predictions[k].argsort()[-3:][::-1]] 
             for k in range(len(test_predictions))]
test_labels=[" ".join(test_labels[i]) 
             for i in range(len(test_labels))]
submission=pd.DataFrame({"key_id":test_data.index,
                         "word":test_labels})
submission.to_csv('qd_submission.csv',index=False)


# In[ ]:


display_drawing()

