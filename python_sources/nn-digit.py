import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

from sklearn.preprocessing import StandardScaler

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

target = train.iloc[:,0]
train = train.drop('label', axis=1)

train_images_count = train.shape[0]
test_images_count = test.shape[0]
features_count = train.shape[1]

img_index = np.random.randint(train_images_count)
plt.imshow(train.iloc[img_index,:].reshape(28, 28), cmap=plt.cm.gray_r)

scaler = StandardScaler()
x_train = scaler.fit_transform(train)
y_train = np_utils.to_categorical(target).astype(np.uint8)
x_test = scaler.fit_transform(test)
class_count = y_train.shape[1]

def nn_model():
    model = Sequential()
    model.add(Dense(features_count, input_dim=features_count, activation='relu', init='normal'))
#    model.add(Dense(np.log2(features_count), input_dim=features_count, activation='relu', init='normal'))
#    model.add(Dense(np.log2(features_count), activation='relu'))
    model.add(Dense(class_count, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    
model = nn_model()
model.fit(x_train, y_train, batch_size=100, nb_epoch=50, verbose=2)
scores = model.evaluate(x_train, y_train)

pred = model.predict(x_test)
pred = np_utils.categorical_probas_to_classes(pred)

result = pd.DataFrame({
                       'ImageID' : [x+1 for x in range(test.shape[0])],
                       'Label' : pred
                       })
result.to_csv('./result.csv', index=False)