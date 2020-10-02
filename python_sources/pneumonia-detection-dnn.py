import numpy as np
import pandas as pd
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Conv2D, Flatten, MaxPooling2D, ZeroPadding2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


train_dir = '../input/chest-xray-pneumonia/chest_xray/train'
val_dir = '../input/chest-xray-pneumonia/chest_xray/val'
test_dir = '../input/chest-xray-pneumonia/chest_xray/test'

classes = ["NORMAL", "PNEUMONIA"]
data_gen = ImageDataGenerator(1./255)
train_batches = data_gen.flow_from_directory(train_dir, target_size = (64, 64), classes= classes, class_mode= 'categorical', shuffle= True)
val_batches = data_gen.flow_from_directory(val_dir, target_size = (64, 64), classes= classes, class_mode= 'categorical', shuffle= True)
test_batches = data_gen.flow_from_directory(test_dir, target_size = (64, 64), classes= classes, class_mode= 'categorical', shuffle= False)


#This is a Convolutional Artificial Neural Network
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same",
                 input_shape=(64,64,3)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(1024,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(rate=0.4))
model.add(Dense(2, activation="softmax"))


model.compile(optimizer = Adam(lr= 0.00001), 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

history = model.fit_generator(train_batches, epochs=3, validation_data=val_batches, verbose= 2)


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

#model.save_weights("model.h5")

prediction = model.predict_generator(generator=test_batches, verbose= 2)

# create csv
predictions = prediction>0.5
submission = pd.DataFrame(zip(predictions), columns=['Prediction'])
print(submission)
submission.to_csv('submission.csv', index=False)

model.save_weights("weights.h5")
