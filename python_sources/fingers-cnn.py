from IPython.display import Image
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, MaxPool2D
from keras.models import Sequential
from PIL import Image
import keras
import os
import numpy as np

dir_train = "../input/fingers/fingers/train/"
dir_test = "../input/fingers/fingers/test/"

files_train = os.listdir(dir_train)
files_test = os.listdir(dir_test)

print("Total files in training directory : " + str(len(files_train)))
print("Total files in testing directory : " + str(len(files_test)))

X_train = []
Y_train = []

X_test = []
Y_test = []

i = 1

# Preprocess the images in the train directory
for file in files_train:
    path = os.path.join(dir_train, file)
    if "png" in path:
        # print(path)
        
        label = int(path.split("_")[1].split(".")[0])
        label = keras.utils.to_categorical(label, num_classes=6, dtype='int32')
        
        # print(label)
        
        # Convert the image RGB
        img = Image.open(path)
        # img = Image.open(path).convert("RGB")
        # img = Image.open(path).convert('LA')
        
        
        # Resize the image to input size accepted by VGG16
        # img = img.resize((224,224), Image.ANTIALIAS)
        
        # Image array
        img = np.array(img)
        img = np.reshape(img, (128, 128, -1)) 
        # print(img.shape)
        
        X_train.append(img)
        Y_train.append(label)
        
        # print(i)
        # i += 1

# Preprocess the images in the test directory
for file in files_test:
    path = os.path.join(dir_test, file)
    if "png" in path:
        # print(path)
        
        label = int(path.split("_")[1].split(".")[0])
        label = keras.utils.to_categorical(label, num_classes=6, dtype='int32')
        
        # print(label)
        
        # Convert the image RGB
        img = Image.open(path)
        # img = Image.open(path).convert("RGB")
        # img = Image.open(path).convert('LA')
        
        
        # Resize the image to input size accepted by VGG16
        # img = img.resize((224,224), Image.ANTIALIAS)
        
        # Image array
        img = np.array(img)
        img = np.reshape(img, (128, 128, -1)) 
        # print(img.shape)
        
        X_test.append(img)
        Y_test.append(label)
        
        # print(i)
        # i += 1
     
X_train = np.array(X_train)
Y_train = np.array(Y_train)     

X_test = np.array(X_test)
Y_test = np.array(Y_test)

# Save the numpy arrays
np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)

print(X_train.shape)
print(Y_train.shape)

print(X_test.shape)
print(Y_test.shape)

# Form the training and validation data out of the train folder
X_t, X_v, Y_t, Y_v = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)

# # Form the training and validation data out of the test folder
# X_t_t, X_v_v, Y_t_t, Y_v_v = train_test_split(X_test, Y_test, test_size=0.2, random_state=1)

# Form the simple CNN model
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = (128, 128, 1), activation = 'relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(256, (3,3), activation = 'relu'))
model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dropout(0.40))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.40))
model.add(Dense(6, activation = 'softmax'))

model.summary()

model.compile('SGD', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x = X_t, y = Y_t, batch_size = 128, epochs = 10, validation_data = (X_v, Y_v))

score = model.evaluate(X_test, Y_test)
print(score)

model.save('finger_cnn_model.h5')


# # Form the pretrained VGG16 model
# def VGG_model_16(use_imagenet=True):
#     # load pre-trained model graph, don't add final layer
#     model = keras.applications.VGG16(include_top=False, input_shape = (224,224,3) ,
#                                           weights='imagenet' if use_imagenet else None)
   
#     new_output = keras.layers.GlobalAveragePooling2D()(model.output)
#     new_output = keras.layers.Dense(6, activation='softmax')(new_output)
#     model = keras.engine.training.Model(model.inputs, new_output)
#     return model
    
# model = VGG_model_16()

# print(len(model.layers))
# # print(model.summary())

# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# model.fit(X_train, Y_train, epochs = 15, validation_data=(X_valid, Y_valid))