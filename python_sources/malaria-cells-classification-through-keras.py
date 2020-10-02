#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# Import Necessary Libraries. Here I will be using Keras library for making classifier. We will also requires numpy and Image for making image to array implementation. We will be using sequential model here in the notebook. 

# In[1]:


#Importing Necessary Libraries.
from PIL import Image
import numpy as np
import os
import cv2
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout


# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


get_ipython().system('ls')


# # Data Prepearaion

# Data Preperation: We will make data and labels list where data will be image to array implementatation which contains RGB values of each image. and label will be class of cells here I will be taking 0 and 1 for two classes

# In[4]:


data=[]
labels=[]
Parasitized=os.listdir("../input/cell_images/cell_images/Parasitized/")
print(Parasitized)
for a in Parasitized:
    try:
        image=cv2.imread("../input/cell_images/cell_images/Parasitized/"+a)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((50, 50))
        data.append(np.array(size_image))
        labels.append(0)
    except AttributeError:
        print("")
Uninfected=os.listdir("../input/cell_images/cell_images/Uninfected/")
for b in Uninfected:
    try:
        image=cv2.imread("../input/cell_images/cell_images/Uninfected/"+b)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((50, 50))
        data.append(np.array(size_image))
        labels.append(1)
    except AttributeError:
        print("")

    


# In[5]:


Cells=np.array(data)
labels=np.array(labels)
print(Cells.shape)
print(labels.shape)


# Saving and Loading the data we prepared so next time we can load it from saved .npy file.

# In[6]:


np.save("Cells",Cells)
np.save("labels",labels)


# In[7]:


Cells=np.load("Cells.npy")
labels=np.load("labels.npy")


# Do Train/Test Split of data and labels that prepared in early section. Classes are defined as the unique labels in the data. Here it will be 2 as Parasitized:0 and Uninfected:1, here 0 and 1 are the mapping in labels for these two classes

# In[8]:


s=np.arange(Cells.shape[0])
np.random.shuffle(s)
Cells=Cells[s]
labels=labels[s]


# In[9]:


num_classes=len(np.unique(labels))
len_data=len(Cells)


# In[10]:


(x_train,x_test)=Cells[(int)(0.1*len_data):],Cells[:(int)(0.1*len_data)]
x_train = x_train.astype('float32')/255 # As we are working on image data we are normalizing data by divinding 255.
x_test = x_test.astype('float32')/255
train_len=len(x_train)
test_len=len(x_test)


# In[11]:


(y_train,y_test)=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]
print(y_train)
print(y_test)
print(x_train.shape)
print(y_train.shape)


# # One hot encoding:

# Here the problem has two classes so last output layer of neural network will have 2 neurons one for each class, One hot encoding will help us to change labels in binary format.
# example:
#     2 can be represented as [1 0] if output layer has 2 neurons and [0 0 1 0] if output has 4 neurons/classes

# In[12]:


#Doing One hot encoding as classifier has multiple classes
y_train=keras.utils.to_categorical(y_train,2)
y_test=keras.utils.to_categorical(y_test,2)


# # Create Sequential Model:

# Here I will be using Relu{max(0,z)}, You can try tanh/sigmoid/Leaky Relu for finding performance on various activation functions.Our output layer will be softmax activation rather than sigmoid as we have more than one class to classify. softmax activation calculates e^value/sum(all_values_in_axis[0 or 1])

# In[13]:


#creating sequential model
model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2,activation="softmax"))#2 represent output layer neurons 
model.summary()


# compile the model with loss as categorical_crossentropy and using adam optimizer you can test result by trying RMSProp as well as Momentum

# In[14]:


# compile the model with loss as categorical_crossentropy and using adam optimizer you can test result by trying RMSProp as well as Momentum
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[15]:


#Fit the model with min batch size as 50[can tune batch size to some factor of 2^power ] 
model.fit(x_train,y_train,batch_size=32,epochs=20)


# # Check the accuracy on Test data:

# In[16]:


accuracy = model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test_Accuracy:-', accuracy[1])


# # Save the model weights:

# In[20]:


from keras.models import load_model
model.save('cells.h5')


# # Use of Model in simple application using tkinter.

# In[21]:


from keras.models import load_model
from PIL import Image
from PIL import Image
import numpy as np
import os
import cv2
def convert_to_array(img):
    im = cv2.imread(img)
    img_ = Image.fromarray(im, 'RGB')
    image = img_.resize((50, 50))
    return np.array(image)
def get_cell_name(label):
    if label==0:
        return "Paracitized"
    if label==1:
        return "Uninfected"
def predict_cell(file):
    model = load_model('cells.h5')
    print("Predicting Type of Cell Image.................................")
    ar=convert_to_array(file)
    ar=ar/255
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=model.predict(a,verbose=1)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    Cell=get_cell_name(label_index)
    return Cell,"The predicted Cell is a "+Cell+" with accuracy =    "+str(acc)


# In[19]:


"""from tkinter import Frame, Tk, BOTH, Text, Menu, END
from tkinter import filedialog 
from tkinter import messagebox as mbox

class Example(Frame):

    def __init__(self):
        super().__init__()   

        self.initUI()


    def initUI(self):

        self.master.title("File dialog")
        self.pack(fill=BOTH, expand=1)

        menubar = Menu(self.master)
        self.master.config(menu=menubar)

        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Open", command=self.onOpen)
        menubar.add_cascade(label="File", menu=fileMenu)        

        

    def onOpen(self):

        ftypes = [('Image', '*.png'), ('All files', '*')]
        dlg = filedialog.Open(self, filetypes = ftypes)
        fl = dlg.show()
        c,s=predict_cell(fl)
        root = Tk()
        T = Text(root, height=4, width=70)
        T.pack()
        T.insert(END, s)
        

def main():

    root = Tk()
    ex = Example()
    root.geometry("100x50+100+100")
    root.mainloop()  


if __name__ == '__main__':
    main()"""


# In[ ]:




