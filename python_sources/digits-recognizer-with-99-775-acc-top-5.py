# In my earlier notebook (https://www.kaggle.com/vishnu123/digits-recognizer-with-99-30-acc/comments) I scored 99.30% accuracy on validation
# dataset which after submitting for prediction take me to top 41% (not so interesting result). So, I thought let's try to fine tune my model a little bit and try to get 
# on top 10% atleast. 

# But, luckily after fine tuning with different parameters, optimization techniques and batch_size I stood on top 5% on leaderboard

# I also think with more experiment with network architecture and more fine tuning I can still go little higher.

# This notebook contains only code for more go to (https://www.kaggle.com/vishnu123/digits-recognizer-with-99-30-acc/comments)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Activation,Lambda,BatchNormalization,Dropout,Flatten,MaxPooling2D,Input,Dense,MaxPool2D
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import binary_crossentropy,mse
from tensorflow.keras import backend as K 
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")

Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)
X_train = X_train / 255.0
X_test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2,stratify = Y_train)


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
#model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
#model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


optimizer = tf.keras.optimizers.Adam()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.00001)
ckpt = ModelCheckpoint('cnn_model_adam.h5',
                            verbose=1, save_weights_only=True,save_best_only = True)
epochs = 50
batch_size = 86

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

train_datagen = ImageDataGenerator(featurewise_center=False,featurewise_std_normalization=False,
                                zca_whitening=False,rotation_range = 10,
                                width_shift_range = 0.1,height_shift_range = 0.1,
                                shear_range = 0.1,zoom_range = 0.1,horizontal_flip = False,
                                vertical_flip = False,fill_mode = 'nearest'
                           )
history1 = model.fit_generator(train_datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = 2, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction,ckpt])


sub = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
def predict_test(data,model):
    data = data / 255.0
    data = data.values.reshape(-1,28,28,1)
    return(model.predict(data))

res = predict_test(test,model1)

final_result = []
for i in range(len(res)):
    final_result.append(np.argmax(res[i]))
sub['Label'] = final_result
sub.to_csv("submission.csv")