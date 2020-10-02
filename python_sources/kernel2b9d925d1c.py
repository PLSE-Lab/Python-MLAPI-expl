import numpy as np
import pandas as pd

import seaborn 
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils  import np_utils
#from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import csv

train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")

print(train.head())

train_y = train["label"].values
train_x = train_x = train.drop("label",axis = 1)

print(train_x.shape, train_y.shape) #(42000, 784) (42000, )

print(train_x.isnull().any().describe())
# There are no missing value

# Show the training datas value distribution
seaborn.countplot(train_y)
plt.show()
plt.savefig('value_distribution.png')

# 28 * 28 = 784 pixels
train_x = train_x.values.reshape(len(train_x),28,28, 1)
test = test.values.reshape(len(test),28,28, 1)


print(train_x.shape, test.shape) #(42000, 28, 28, 1) (28000, 28, 28, 1)


#creat CNN model
print('Creating CNN model...')
tensor_in = Input((28, 28, 1))
tensor_out = tensor_in
tensor_out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(tensor_out)
tensor_out = Dropout(0.5)(tensor_out)
tensor_out = MaxPooling2D((2,2), padding='same')(tensor_out)
tensor_out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(tensor_out)
tensor_out = Dropout(0.5)(tensor_out)
tensor_out = MaxPooling2D((2,2), padding='same')(tensor_out)
tensor_out = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = MaxPooling2D((2,2), padding='same')(tensor_out)

tensor_out = Flatten()(tensor_out)
tensor_out = Dropout(0.5)(tensor_out)
tensor_out = Dense(10, name='digit', activation='softmax')(tensor_out)

from keras.models import Model
model = Model(inputs=tensor_in, outputs=tensor_out)
model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
model.summary()


train_y = np_utils.to_categorical(train_y)

from sklearn.model_selection import train_test_split
s_train_x, s_test_x, s_train_y, s_test_y = train_test_split(train_x, train_y, test_size=0.20, random_state=7)
mhistory = model.fit(s_train_x, s_train_y, validation_data=(s_test_x, s_test_y), epochs=100, batch_size=128, verbose=2)

'''
plt.plot(mhistory.history['acc'])
plt.plot(mhistory.history['val_acc'])
plt.legend(['training', 'validation'], loc='lower right')
plt.show()
plt.savefig('train_history.png')
'''

pred_result = model.predict(test)
pred_result = np.array(pred_result)

y_pred_final = []
for i in pred_result:
    y_pred_final.append(np.argmax(i))


results = pd.Series(y_pred_final,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv('submission.csv', index = False)
'''
from IPython.display import HTML
def create_download_link(title = "Download CSV file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

# create a link to download the dataframe which was saved with .to_csv method
create_download_link(filename='submission.csv')
'''