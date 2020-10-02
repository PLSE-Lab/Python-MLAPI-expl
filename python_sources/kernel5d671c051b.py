import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm 

"../input/skin-cancer-malignant-vs-benign/train"
TRAIN_DIR = "../input/skin-cancer-malignant-vs-benign/train"
TEST_DIR = "../input/skin-cancer-malignant-vs-benign/test" 
CATEGORIES = ["benign","malignant"]

for category in CATEGORIES:
    path=os.path.join(TRAIN_DIR,category) #path to benign and malignant
    for img in tqdm(os.listdir(path)):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.figure()
        plt.imshow(img_array)
        plt.colorbar()
        plt.grid(False)
        plt.show()
        break
    break
print(img_array.shape)

img_size = 224
new_array = cv2.resize(img_array, (img_size,img_size))
plt.imshow(new_array,cmap = 'gray')
plt.show()

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(TRAIN_DIR,category) #path to benign and malignant
        class_num=CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE) # give numerical values 0,1 to data data not to be string like benign / malignant
                new_array = cv2.resize(img_array, (img_size,img_size))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()

testing_data = []
def process_test_data():
    for category in CATEGORIES:
        path=os.path.join(TEST_DIR,category) #path to benign and malignant
        class_num=CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                #img_num = img.split('.')[0]
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size,img_size))
                testing_data.append([new_array,class_num])
            except Exception as e:
                pass
#run the function now

process_test_data()

import random
random.shuffle(training_data)
random.shuffle(testing_data)

X=[]
y=[]

test_x=[]
test_y=[]

for features, label in training_data:
    X.append(features)
    y.append(label)
    
X= np.array(X).reshape(-1,img_size,img_size,1)
y=np.array(y)


for features, label in testing_data:
    test_x.append(features)
    test_y.append(label)
    
test_x= np.array(test_x).reshape(-1,img_size,img_size,3)
test_y=np.array(test_y)


#print(test_x#labels
#print(test_y) # Features

import pickle 

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X= pickle.load(pickle_in)


X[0]
      
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout , Activation , Flatten ,Conv2D , MaxPooling2D
import pickle 
#from tensorflow.keras.callbacks import TensorBoard
import time 

#NAME="BenignVsMalignant_224-{}".format(int(time.time()))
#tensorboard =  TensorBoard(log_dir="logs\\model\\".format(NAME))

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X=X/255.0

model = Sequential()

model.add(Conv2D(64, (3,3) , input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3,3) , input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
#model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid')) 

model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=['accuracy']
             )
model.fit(X, y, batch_size=32, epochs=2,validation_split=0.1)

      
predictions = model.predict([test_x])

print(predictions)
print(np.argmax(predictions[9]))

plt.imshow(test_x[0])
plt.show()