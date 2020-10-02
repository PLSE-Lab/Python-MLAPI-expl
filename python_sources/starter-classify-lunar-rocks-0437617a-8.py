
import numpy as np 
import tensorflow as tf 
import pandas as pd 
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout,InputLayer,MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.models import model_from_yaml

image_size = 56
num_classes = 2

data_generator = ImageDataGenerator(rescale=1./255,
	preprocessing_function=preprocess_input, 
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip = True,
    validation_split=0.20) 

train_generator = data_generator.flow_from_directory(
        directory = '/kaggle/input/DataSet/Train Images',
        target_size=(image_size, image_size),
        shuffle=True,
        batch_size=30,
        class_mode='binary',
        subset="training"
        )
val_generator = data_generator.flow_from_directory(
        directory = '/kaggle/input/DataSet/Train Images',
        target_size=(image_size, image_size),
        shuffle=True,
        batch_size=30,
        class_mode='binary',
        subset="validation")

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size=3,activation='relu',padding = "same",input_shape = [image_size,image_size,3]))
model.add(Conv2D(filters = 32, kernel_size=3,padding = "same", activation='relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size=3, activation='relu',padding = "same"))
model.add(Conv2D(filters = 64, kernel_size=3,padding = "same", activation='relu'))
model.add(MaxPool2D(pool_size = (2,2), strides = 2))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer = optimizer ,
              metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


model.fit_generator(
	train_generator,
	epochs=1,
	validation_data= val_generator,
	steps_per_epoch=320,
	verbose = 2,
	callbacks = [learning_rate_reduction]
	)
model.save("model.h5")


"""
test_data_file = pd.read_csv("/kaggle/input/DataSet/test.csv")
test_File = test_data_file["Image_File"]
test_paths = ["/kaggle/input/DataSet/Test Images/"+i for i in test_File]
predictions = []
cnt = 0
#imgs = imageio.imread(test_paths[0])
for i in test_paths:
    print(i)
    for i in test_paths:
	imgs = imageio.imread(i)
	#imgs = [load_img(i,target_size=(image_size,image_size,3))]
	img_array = np.array([img_to_array(imgs)])
	pred = model.predict(img_array)
	#print(pred)
	if(pred[0][0] > pred[0][1]):
		predictions.append([test_File[cnt],"Large"])
		#print("Large")
	else:
		predictions.append([test_File[cnt],"Small"])
		#print("Small")
	cnt += 1
	#print(str(cnt)+"\t\t"+predictions[cnt])


res = pd.DataFrame(predictions)
res.columns = ["Image_File","Class"]
res.to_csv("prediction_results.csv",index = False)"""