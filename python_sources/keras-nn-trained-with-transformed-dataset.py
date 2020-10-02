import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import load_model
import skimage.transform as transform
from scipy import ndimage
from skimage import img_as_float, img_as_ubyte
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs




# sample 20,000 random rows , using 100
df = train.sample(n=1000)


# dimensions of our images.
img_width, img_height = 28, 28
# ============================================================================
# GENERATE EXTRA DATA BY APPLYTING TRANSFORMATIONS TO THE ORIGIGNAL DATASET
# 
#   erosion
#   dilation 
#   affine tranformation
#
#   and combinations
#
# ============================================================================
temp = []
for index, row in df.iterrows():
   rowData  = row.as_matrix();
   X_row    = rowData[1:785]
   Y_row    = rowData[0]
   X_row    = X_row.reshape(img_width, img_height)

   x_dilation = ndimage.grey_dilation(X_row, size=(2,2))
   x_erosion  = ndimage.grey_erosion(X_row,  size=(2,2))

   affine1 = transform.AffineTransform(shear=np.deg2rad(np.random.uniform(-20, 20)),
                                      rotation=np.deg2rad(np.random.uniform(-25, 25)))

   X_affine = img_as_ubyte(transform.warp(img_as_float(X_row / 255.0), affine1))


   affine2 = transform.AffineTransform(shear=np.deg2rad(np.random.uniform(-20, 20)),
                                      rotation=np.deg2rad(np.random.uniform(-25, 25)))

   X_affine_dilation = img_as_ubyte(transform.warp(img_as_float(x_dilation / 255.0), affine2))


   affine3 = transform.AffineTransform(shear=np.deg2rad(np.random.uniform(-20, 20)),
                                      rotation=np.deg2rad(np.random.uniform(-25, 25)))

   X_affine_erosion = img_as_ubyte(transform.warp(img_as_float(x_erosion / 255.0), affine3))


   # plt.figure(1)
   # plt.subplot(411)
   # plt.imshow(X_row, cmap='Greys', interpolation='nearest')
   #
   # plt.subplot(412)
   # plt.imshow(x_dilation, cmap='Greys', interpolation='nearest')
   #
   # plt.subplot(413)
   # plt.imshow(X_affine, cmap='Greys', interpolation='nearest')
   #
   # plt.subplot(414)
   # plt.imshow(x_erosion, cmap='Greys', interpolation='nearest')
   # plt.show()

   x_dilation = x_dilation.reshape(784)
   X_affine = X_affine.reshape(784)
   x_erosion = x_erosion.reshape(784)
   X_affine_dilation = X_affine_dilation.reshape(784)
   X_affine_erosion = X_affine_erosion.reshape(784)

   x_dilation = np.insert(x_dilation, 0,Y_row)
   X_affine = np.insert(X_affine, 0, Y_row)
   x_erosion = np.insert(x_erosion, 0, Y_row)
   X_affine_dilation = np.insert(X_affine_dilation, 0, Y_row)
   X_affine_erosion = np.insert(X_affine_erosion, 0, Y_row)


   temp.append(x_dilation)
   temp.append(X_affine)
   temp.append(x_erosion)
   temp.append(X_affine_dilation)
   temp.append(X_affine_erosion)


   # print index

df_transformation   = pd.DataFrame(temp, columns=df.columns)




# ============================================================================
# BUILD MODEL AND TRAIN
# ============================================================================


# size of pooling area for max pooling
pool_size = (2, 2)


batch_size = 256
nb_classes = 10
nb_epoch = 2


# merge transformed dataset with the original one
df = pd.concat([train, df_transformation])
df.sample(frac=1)

train = df.as_matrix()

X = train[:, 1:785]
Y = train[:, 0]
seed = 5

X = X.reshape(X.shape[0], img_width, img_height, 1)

input_shape = (img_width, img_height, 1)


X_train = X.astype('float32')
X_train /= 255



# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y, nb_classes)


# model = load_model('model_2_extra_data.h5')

model = Sequential()

model.add(Convolution2D(64, 3,3, border_mode='valid',   input_shape=input_shape))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Convolution2D(128, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))


model.add(Convolution2D(256, 2, 2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


model.fit(X_train, Y_train,
    batch_size=batch_size,
    nb_epoch=nb_epoch,
    verbose=1,
    validation_split=0.2
)


model.save('model_2_extra_data.h5')



# ============================================================================
# PREDICT
# ============================================================================

test = test.as_matrix()

# dimensions of our images.


X = test[:, 0:784]

X_test = X.reshape(X.shape[0], img_width, img_height, 1)


# model = load_model('model_2_extra_data.h5')


print("Generating test predictions...")
preds = model.predict_classes(X_test, batch_size=64, verbose=0)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "keras-extra.csv")




