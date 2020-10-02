# %% [code]
import numpy as np 
import pandas as pd 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

num_classes = 10
img_width = 28
img_height = 28
num_channels = 1

num_epochs = 30
batch_size = 128
steps_per_epoch = 210

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

x_train = train.drop(["label"], axis = 1)
y_train = train["label"]

x_train = x_train.values.reshape(x_train.shape[0], img_width, img_height, num_channels)
x_test = test.values.reshape(test.shape[0], img_width, img_height, num_channels)
y_train = to_categorical(y_train, num_classes)

x_train = x_train / 255
x_test = x_test / 255

print(x_train.shape)

model = Sequential()

# C1 Convolutional Layer
model.add(Conv2D(12, kernel_size = (5, 5), padding = "same", strides = (1, 1), input_shape = (img_width, img_height, num_channels)))
model.add(Activation("relu"))
model.add(BatchNormalization())

# S2 Pooling Layer
model.add(MaxPooling2D(pool_size = (2, 2), padding = "valid", strides = (2, 2)))

# C3 Convolutional Layer
model.add(Conv2D(32, kernel_size = (5, 5), padding = "valid", strides = (1, 1)))
model.add(Activation("relu"))
model.add(BatchNormalization())

# S4 Pooling Layer
model.add(MaxPooling2D(pool_size = (2, 2), padding = "valid", strides = (2, 2)))

# FC Layer
model.add(Flatten())
model.add(Dense(units = 120))
model.add(Activation("relu"))

# FC Layer
model.add(Dense(units = 84))
model.add(Activation("relu"))

# Output Layer
model.add(Dense(units = 10))
model.add(Activation("softmax"))

model.compile(loss = categorical_crossentropy,
	optimizer = "SGD",
	metrics = ["accuracy"])

train_aug = ImageDataGenerator(
	rotation_range = 8,
	zoom_range = 0.1,
	width_shift_range = 0.08,
	height_shift_range = 0.08
)

train_aug.fit(x_train)

hist = model.fit_generator(train_aug.flow(x_train, y_train, batch_size = batch_size),
	epochs = num_epochs, 
	verbose = 1)

evaluate = model.evaluate(x_train, y_train)
print("loss: ", evaluate[0])
print("accuracy: ", evaluate[1])

prediction = model.predict_classes(x_test)

final_prediction = pd.DataFrame({"ImageId": list(range(1, (len(prediction) + 1))),
								 "Label": prediction})

final_submission = final_prediction.to_csv("submission.csv", index = "False", header = "True")

df = pd.read_csv("submission.csv")

df = df.drop(df.columns[0], axis = 1)
df = df.to_csv("submission_final.csv", index = False)