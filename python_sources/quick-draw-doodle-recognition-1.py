#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:steelblue; font-family:Ewert; font-size:150%;" class="font-effect-fire-animation">Code Libraries, Style, & Links</h1>

# In[ ]:


get_ipython().run_cell_magic('html', '', "<style>\n@import url('https://fonts.googleapis.com/css?family=Ewert|Roboto&effect=3d|fire-animation');\nspan {font-family:'Roboto'; color:black; text-shadow:4px 4px 4px #aaa;}  \ndiv.output_area pre{font-family:'Roboto'; font-size:120%; color: steelblue;}      \n</style>")


# In[ ]:


import numpy as np,pandas as pd
import keras as ks,pylab as pl
import os,ast,h5py,warnings
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation,Dropout,Dense,Conv2D,MaxPooling2D,GlobalMaxPooling2D
warnings.filterwarnings('ignore')
pl.style.use('seaborn-whitegrid')
style_dict={'background-color':'gainsboro','color':'steelblue', 
            'border-color':'white','font-family':'Roboto'}
fpath='../input/quickdraw-doodle-recognition/train_simplified/'


# In[ ]:


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
    for k in range (5) :  
        pl.figure(figsize=(10,2))
        pl.suptitle(files[(S-1)*T+k])
        for i in range(5):
            picture=ast.literal_eval(data[labels[(S-1)*T+k]].values[i])
            for x,y in picture:
                pl.subplot(1,5,i+1)
                pl.plot(x,y,'-o',markersize=1,color='slategray')
                pl.xticks([]); pl.yticks([])
            pl.gca().invert_yaxis()
            pl.axis('equal');            
def get_image(data,k,I):
    img=np.zeros((280,280))
    picture=ast.literal_eval(data.iloc[k])
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


# <h1 style="color:steelblue; font-family:Ewert; font-size:150%;" class="font-effect-fire-animation">Data Exploration</h1>

# In[ ]:


data_alarm_clock=pd.read_csv(fpath+'alarm clock.csv',
                             index_col='key_id')
data_alarm_clock.tail(3).T.style.set_properties(**style_dict)


# In[ ]:


I=64 # image size in pixels
S=2 # number of the label set {1,...,10}->{1-34,...,307-340}
T=20 # number of labels in one set 
N=7000 # number of images with the same label in the training set
files=sorted(os.listdir(fpath))
print(files)


# In[ ]:


labels=[el.replace(" ","_")[:-4] for el in files]
data=pd.DataFrame(index=range(N),
                  columns=labels[(S-1)*T:S*T])
for i in range((S-1)*T,S*T):
    data[labels[i]]=    pd.read_csv(fpath+files[i],
                index_col='key_id').drawing.values[:N]
data.shape


# In[ ]:


display_drawing()


# In[ ]:


images=[]
for label in labels[(S-1)*T:S*T]:
    images.extend([get_image(data[label],i,I) 
                   for i in range(N)])    
images=np.array(images)
targets=np.array([[]+N*[k] for k in range((S-1)*T,S*T)],
                 dtype=np.uint8).reshape(N*T)
del data,data_alarm_clock 
images.shape,targets.shape


# In[ ]:


#with h5py.File('QuickDrawImages001-020.h5','w') as f:
#    f.create_dataset('images',data=images)
#    f.create_dataset('targets',data=targets)
#    f.close()


# In[ ]:


images=images.reshape(-1,I,I,1)
x_train,x_test,y_train,y_test=train_test_split(images,targets,
                 test_size=.2,random_state=1)
n=int(len(x_test)/2)
x_valid,y_valid=x_test[:n],y_test[:n]
x_test,y_test=x_test[n:],y_test[n:]
del images,targets
[x_train.shape,x_valid.shape,x_test.shape,
 y_train.shape,y_valid.shape,y_test.shape]


# In[ ]:


nn=np.random.randint(0,int(.8*T*N),3)
ll=labels[y_train[nn[0]]]+', '+labels[y_train[nn[1]]]+   ', '+labels[y_train[nn[2]]]
pl.figure(figsize=(10,2))
pl.subplot(1,3,1); pl.imshow(x_train[nn[0]].reshape(I,I))
pl.subplot(1,3,2); pl.imshow(x_train[nn[1]].reshape(I,I))
pl.subplot(1,3,3); pl.imshow(x_train[nn[2]].reshape(I,I))
pl.suptitle('Key Points to Lines: %s'%ll);


# <h1 style="color:steelblue; font-family:Ewert; font-size:150%;" class="font-effect-fire-animation">The Model</h1>

# In[ ]:


def model():
    model=Sequential()    
    model.add(Conv2D(32,(5,5),padding='same',
                     input_shape=x_train.shape[1:]))
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
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model
model=model()


# In[ ]:


fw='weights.best.model.021-040.hdf5'
checkpointer=ModelCheckpoint(filepath=fw,verbose=2,
                             save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',
                               patience=5,verbose=2,factor=.75)
history=model.fit(x_train,y_train-(S-1)*T,epochs=100,
                  batch_size=1024,verbose=2,
                  validation_data=(x_valid,y_valid-(S-1)*T),
                  callbacks=[checkpointer,lr_reduction])


# <h1 style="color:steelblue; font-family:Ewert; font-size:150%;" class="font-effect-fire-animation">Evaluation</h1>

# In[ ]:


model.load_weights(fw)
model.evaluate(x_test,y_test-(S-1)*T)


# In[ ]:


p_test=model.predict(x_test)
well_predicted=[]
for p in range(len(x_test)):
    if (np.argmax(p_test[p])==y_test[p]-(S-1)*T):
        well_predicted.append(labels[(S-1)*T+np.argmax(p_test[p])])
u=np.unique(well_predicted,return_counts=True)
pd.DataFrame({'labels':u[0],'correct predictions':u[1]}).sort_values('correct predictions',ascending=False).style.set_properties(**style_dict)


# <h1 style="color:steelblue; font-family:Ewert; font-size:150%;" class="font-effect-fire-animation">The Next Step</h1>
# The weights for each label set have saved in the special database and will be used for image recognition in the test data.<br/>
# The next notebook [Quick, Draw! Doodle Recognition 2](https://www.kaggle.com/olgabelitskaya/quick-draw-doodle-recognition-2)
