# Machine Learning - Exercise 4 Neural Network Learning
# Johnny Wang - johnny.wang@live.ca
# One Hidden Layer Nueral Net

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Lambda, Flatten
from keras.layers import Convolution2D, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras import backend as K


# Parameters
learning_rate = 0.001
epochs = 3
batch_size = 32
test_size = 0.15


# Load data into Python
data = pd.read_csv("../input/train.csv")
X = (data.iloc[:,1:].values).astype('float32')
y = data.iloc[:,0].values.astype('int32')
test = pd.read_csv("../input/test.csv")
X_test = test.values.astype('float32')
print(X.shape[0])
h = 28
w = 28

# View a traning and testing data for fun
plt.figure()
plt.imshow(X[5].reshape(h,w), cmap=plt.get_cmap('gray'))
print(y[5]);

plt.figure()
plt.imshow(X_test[7].reshape(h,w), cmap=plt.get_cmap('gray'))

# Reshape for visualizing purposes
X = X.reshape(X.shape[0], h,w, 1)
X_test = X_test.reshape(X_test.shape[0], h,w, 1)


# Convert y_train into one-hot vectors
y = to_categorical(y)
num_classes = y.shape[1]
print(num_classes)



# Split training data into validation and train and batch then
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=48)
train_batches = image.ImageDataGenerator().flow(X_train, y_train, batch_size=batch_size)
val_batches=image.ImageDataGenerator().flow(X_val, y_val, batch_size=batch_size)

# Define normalization
X_mean = X_train.mean().astype(np.float32)
X_std = X_train.std().astype(np.float32)
def normalize(x):
    return (x-X_mean)/X_std

# For the model
def compile_leNet5_model(height,width):
    model = Sequential([
        Lambda(normalize, input_shape=(height,width,1)),
        Convolution2D(6,(5,5),strides=(1, 1),padding='same', activation='relu'),
        AveragePooling2D(pool_size=(2, 2), strides=(1, 1)),
        Convolution2D(16,(5,5),strides=(1, 1), activation='relu'),
        AveragePooling2D(pool_size=(2, 2), strides=(1, 1)),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(10, activation='softmax'),
        ])
    model.compile(Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Make and train model
leNet5 = compile_leNet5_model(h,w)
print("input shape ",leNet5.input_shape)
print("output shape ",leNet5.output_shape)


h5_path = "model.lenet5"
checkpointer = ModelCheckpoint(h5_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
earlystopper = EarlyStopping(monitor='acc', patience=10, verbose=1)

# Train on original data
history=leNet5.fit_generator(generator=train_batches,steps_per_epoch=train_batches.n, epochs=epochs, validation_data=val_batches, validation_steps=val_batches.n, callbacks=[checkpointer,earlystopper])

for i in range(epochs):
    # Augment data for later training
    augment =image.ImageDataGenerator(rotation_range=15*i, width_shift_range=0.1*i, shear_range=0.3*i, height_shift_range=0.1*i, zoom_range=0.15*i)
    aug_batches = augment.flow(X,y,batch_size=batch_size)
    # Train model on augmented data
    history_aug=leNet5.fit_generator(generator=aug_batches,steps_per_epoch=aug_batches.n, epochs=epochs, validation_data=val_batches, validation_steps=val_batches.n, callbacks=[checkpointer,earlystopper])

leNet5.load_weights(h5_path)

# Plot Loss functions
loss_values = history.history['loss']
val_loss_values = history.history['val_loss']
loss_values_aug = history_aug.history['loss']
val_loss_values_aug = history_aug.history['val_loss']

t = range(1,epochs+1)
fig, axs = plt.subplots(2, 1)
axs[0].plot(t, loss_values,t, val_loss_values)
axs[0].set_ylabel('Loss')
axs[0].legend(['train','valid'])
axs[0].grid(True)
axs[1].plot(t, loss_values_aug,t, val_loss_values_aug)
axs[1].legend(['train','valid'])
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epochs')
axs[1].grid(True)
plt.show()

# Submitting to Kaggle
predictions = leNet5.predict_classes(X_test, verbose=0)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("submission.csv", index=False, header=True)

