#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('writefile', 'mymodel.py', 'from keras.models import Sequential\nfrom keras.layers.normalization import BatchNormalization\nfrom keras.layers.convolutional import Conv2D\nfrom keras.layers.convolutional import MaxPooling2D\nfrom keras.layers.core import Activation\nfrom keras.layers.core import Flatten\nfrom keras.layers.core import Dropout\nfrom keras.layers.core import Dense\nfrom keras import backend as K\n\n\nclass MyModel:\n    @staticmethod\n    def build(width, height, depth, classes):\n        # initialize the model along with the input shape to be\n        # "channels last" and the channels dimension itself\n        model = Sequential()\n        inputShape = (height, width, depth)\n        chanDim = -1\n        if K.image_data_format() == "channels_first":\n            inputShape = (depth, height, width)\n            chanDim = 1\n\n        model.add(Conv2D(32, (3, 3), padding="same",\n                         input_shape=inputShape))\n        model.add(Activation("relu"))\n        model.add(BatchNormalization(axis=chanDim))\n        model.add(Conv2D(32, (3, 3), padding="same"))\n        model.add(Activation("relu"))\n        model.add(BatchNormalization(axis=chanDim))\n        model.add(MaxPooling2D(pool_size=(2, 2)))\n        model.add(Dropout(0.25))\n\n        model.add(Conv2D(64, (3, 3), padding="same"))\n        model.add(Activation("relu"))\n        model.add(BatchNormalization(axis=chanDim))\n        model.add(Conv2D(64, (3, 3), padding="same"))\n        model.add(Activation("relu"))\n        model.add(BatchNormalization(axis=chanDim))\n        model.add(MaxPooling2D(pool_size=(2, 2)))\n        model.add(Dropout(0.25))\n\n\n        model.add(Flatten())\n        model.add(Dense(512))\n        model.add(Activation("relu"))\n        model.add(BatchNormalization())\n        model.add(Dropout(0.5))\n\n        model.add(Dense(classes))\n        model.add(Activation("softmax"))\n        # return the constructed network architecture\n        return model')


# In[ ]:


from mymodel import MyModel
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


# In[ ]:


train_data = pd.read_csv("../input/digit-recognizer/train.csv")
test_data = pd.read_csv("../input/digit-recognizer/test.csv")
print(train_data.shape, test_data.shape)


# In[ ]:


X = np.array(train_data.drop("label", axis=1)).astype('float32')
y = np.array(train_data['label']).astype('float32')


# In[ ]:


plt.figure()
plt.imshow(X[0].reshape(28, 28))
plt.colorbar()
plt.grid(False)


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(y[i])
plt.show()


# In[ ]:


X=X.reshape(-1,28,28,1)
X=X/255.0
y=to_categorical(y)
(X_train,X_val,y_train,y_val)=train_test_split(X,y ,test_size=0.2,random_state=42)

X_train.shape, X_val.shape


# In[ ]:


X_test = np.array(test_data).astype('float32')
X_test = X_test / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

X_test.shape


# In[ ]:


aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, fill_mode="nearest")


# In[ ]:


print("[INFO] compiling model....")
opt = 'rmsprop'
model=MyModel.build(width=28,height=28,depth=1,classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])


# In[ ]:


print("[INFO] training network...")
history = model.fit_generator(aug.flow(X_train, y_train, batch_size=32),
validation_data=(X_val, y_val), steps_per_epoch=len(X_train) // 32,
epochs=50, verbose=1)


# In[ ]:


history.history['val_accuracy'][-1]


# In[ ]:



plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 50), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


# In[ ]:


predictions = model.predict_classes(X_test)

submit = pd.DataFrame(predictions,columns=["Label"])
submit["ImageId"] = pd.Series(range(1,(len(predictions)+1)))


# In[ ]:


submission = submit[["ImageId","Label"]]
submission.shape


# In[ ]:


submission.to_csv("submission3.csv",index=False)


# In[ ]:




