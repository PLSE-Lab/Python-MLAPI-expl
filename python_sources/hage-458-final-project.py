# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, Input
from keras import layers, callbacks
from keras import backend as K
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import itertools
from glob import glob
from PIL import Image
from xgboost import XGBClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
pd.set_option("display.max_columns",20)
basedir = "../input/"
#Create dictionary to easily retreive which images are in which of the two directories
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(basedir, '*', '*.jpg'))}

train_dir = os.path.join(basedir, 'ham10000_images_part_1')
test_dir = os.path.join(basedir, 'ham10000_images_part_2')

# Any results you write to the current directory are saved as output.
# Read the metadat for file names and labels
hmnist = pd.read_csv("../input/HAM10000_metadata.csv")
hmnist['path'] = hmnist.image_id.map(imageid_path_dict.get)
hmnist['dx_code'] = pd.Categorical(hmnist.dx).codes

imageslist = list()
for img in hmnist.path[:5]:
    imageslist.append(imread(img))
    
#  Clever trick for adding the np arrays directly into pandas for easier splitting and modeling, found at https://www.kaggle.com/sid321axn/step-wise-approach-cnn-model-77-0344-accuracy
## Resize down because of processing limitations on a 450x600 image
hmnist['image'] = hmnist.path.map(lambda x: np.asarray(Image.open(x).resize((75,100))))
hmnist[['male', 'female', 'unknown']] = pd.get_dummies(hmnist.sex)
hmnist[['scalp', 'ear', 'face', 'back', 'trunk', 'chest',
       'upper extremity', 'abdomen', 'unknown', 'lower extremity',
       'genital', 'neck', 'hand', 'foot', 'acral']] = pd.get_dummies(hmnist.localization)
       
gencols = ['male', 'female', 'unknown']
loccols = ['scalp', 'ear', 'face', 'back', 'trunk', 'chest',
       'upper extremity', 'abdomen', 'unknown', 'lower extremity',
       'genital', 'neck', 'hand', 'foot', 'acral']
#create test holdout that won't be touched until the model is completed 
holdout = hmnist.sample(n = 100)
hmnist = hmnist.drop(holdout.index)

# split into train/validation sets
x_train_full, x_test_full, y_train, y_test = train_test_split(hmnist, hmnist['dx_code'], test_size=0.20)
x_train = np.asarray(x_train_full['image'].tolist())
x_test = np.asarray(x_test_full['image'].tolist())
y_train = to_categorical(y_train, num_classes = 7)
y_test = to_categorical(y_test, num_classes = 7)

# set up plotting function
def makeplots(h):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(h.history['acc'])+1),h.history['acc'])
    axs[0].plot(range(1,len(h.history['val_acc'])+1),h.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(h.history['acc'])+1),len(h.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(h.history['loss'])+1),h.history['loss'])
    axs[1].plot(range(1,len(h.history['val_loss'])+1),h.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(h.history['loss'])+1),len(h.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


### Sample images
n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         hmnist.sort_values(['dx']).groupby('dx')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
# Data generator and flow from directory
## Scale for the CNN and, since there aren't many training images, take advantage of the data generator and flip/zoom/rotate funcs
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = False,
    vertical_flip = False)




epochs = 25
batch_size = 32
test_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow(
    # train_dir,
    x_train,
    y_train,
    # target_size = (75, 100, 3),
    batch_size = batch_size,
   )
test_generator = test_datagen.flow(
    x_test,
    y_test,
    batch_size = batch_size)
# Model
# Set some callbacks to improve the model and stop when it isn't learning anything new

callbacks_list = [
    callbacks.EarlyStopping(
        monitor = ['acc'],
        patience = 5,),
    callbacks.ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.1,
        patience = 2)
]
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (100, 75, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(7, activation = 'softmax'))
model.compile(optimizer = 'rmsprop',
            loss = 'categorical_crossentropy',
            metrics = ['accuracy'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=250,
    epochs=50,
    validation_data=test_generator,
    validation_steps=50,
    )
    
loss, accuracy = model.evaluate(x_test, y_test, verbose = 1)

makeplots(history)



# Deeper model
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (100, 75, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64,(3, 3), activation = 'relu'))
# model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
# model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(256, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(512, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(1028, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
# model.add(layers.Dense(2056, activation = 'relu'))
model.add(layers.Dense(1028, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(7, activation = 'softmax'))
model.compile(optimizer = 'rmsprop',
            loss = 'categorical_crossentropy',
            metrics = ['accuracy'])
early_stopping = callbacks.EarlyStopping(monitor = 'acc', patience = 5)
red_lr = callbacks.ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.1,
        patience = 2)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=250,
    epochs=50,
    validation_data=test_generator,
    validation_steps=50,
    callbacks = [early_stopping, red_lr])
    
loss, accuracy = model.evaluate(x_test, y_test, verbose = 1)

makeplots(history)
accuracy
plot_model(model, to_file = 'model1.png', show_shapes = True, show_layer_names = True)

## Try predictions wtih xgboost
xgb = XGBClassifier()
relcols = gencols + loccols + ['age']
X = x_train_full[relcols]
y = x_train_full.dx_code
X_test = x_test_full[relcols].values
Y_test = x_test_full.dx_code.values
xgb.fit(X.values, y.values)
testpreds = xgb.predict(X_test)
accuracy = accuracy_score(testpreds, Y_test)

### using keras Model instead of sequential
## Trying multibranch inception style 
img_in = Input(shape = (100, 75, 3),
            dtype = 'float32',
            name = 'imgs')
branch_a = layers.Conv2D(128, 1, activation = 'relu', strides = 2)(img_in)

branch_b = layers.Conv2D(128, 1, activation = 'relu')(img_in)
branch_b = layers.Conv2D(128, 3, activation = 'relu', strides = 2)(branch_b)

branch_c = layers.AveragePooling2D(3, strides = 2)(img_in)
branch_c = layers.Conv2D(128, 3, activation = 'relu')(branch_c)

branch_d = layers.Conv2D(128, 1, activation = 'relu')(img_in)
branch_d = layers.Conv2D(128, 3, activation = 'relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation = 'relu', strides = 2)(branch_d)

output = layers.concatenate(
    [branch_a, branch_b, branch_c, branch_d], axis = -1)

branch_a = layers.Conv2D(64, 1, activation='relu', strides=2, padding = 'same')(img_in)
branch_b = layers.Conv2D(64, 1, activation='relu')(img_in)
branch_b = layers.Conv2D(64, 3, activation='relu', strides=2, padding = 'same')(branch_b)
branch_c = layers.AveragePooling2D(3, strides=2, padding = 'same')(img_in)
branch_c = layers.Conv2D(64, 3, activation='relu', padding = 'same')(branch_c)
branch_d = layers.Conv2D(64, 1, activation='relu')(img_in)
branch_d = layers.Conv2D(64, 3, activation='relu', padding = 'same')(branch_d)
branch_d = layers.Conv2D(64, 3, activation='relu', strides=2, padding = 'same')(branch_d)
branch_e = layers.MaxPooling2D(3, strides=2, padding = 'same')(img_in)
branch_e = layers.Conv2D(64, 3, activation='relu', padding = 'same')(branch_e)
branch_f = layers.Conv2D(64, 1, activation='relu')(img_in)
branch_f = layers.Conv2D(64, 3, activation='relu', padding = 'same')(branch_f)
branch_f = layers.Conv2D(64, 3, activation='relu', padding = 'same')(branch_f)
branch_f = layers.Conv2D(64, 3, activation='relu', padding = 'same')(branch_f)
branch_f = layers.Conv2D(128, 3, activation='relu', strides=2, padding = 'same')(branch_f)
output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=1)
output = layers.Flatten()(output)
diagnosis = layers.Dense(7, activation = 'softmax')(output)
model = Model(inputs = img_in, outputs = diagnosis)
model.compile(optimizer = 'rmsprop',
            loss = 'categorical_crossentropy',
            metrics = ['acc'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch=250,
    epochs=50,
    validation_data=test_generator,
    validation_steps=50,
    callbacks = [early_stopping, red_lr])
loss, accuracy = model.evaluate(x_test, y_test, verbose = 1)
accuracy
makeplots(history)


modpreds = model.predict(x_test)
modpredsl = list()
for i in modpreds:
    # modpreds[i] = 
    modpredsl.append(np.where(i == np.max(i))[0][0])

modpredsl = np.asarray(modpredsl)

clf = LogisticRegression(penalty = 'l2', )
X = X.fillna(0)
y = y.fillna(0)
clf.fit(X.values,y.values)
X_test = pd.DataFrame(X_test).fillna(0).values
logpreds = clf.predict(X_test)

ensemble = np.around(.3 * testpreds + .6 * modpredsl + .1 * logpreds, decimals = 0)
accuracy_score(ensemble, Y_test)


# def multi_input_datagen(x1, x2, y, batch_size):
#     genX1 = gen.flow(x1, y, batch_size = batch_size)
#     genX2 = gen.flow(x2, y, batch_size = batch_size)
#     while True:
#         X1i = genX1.next()
#         X2i = genX2.next()
#         yield [Xl1[0], X2i[0]], X1i[1]
# new_train_datagen = ImageDataGenerator(
#     rescale = 1./255,
#     rotation_range = 40,
#     shear_range = 0.2,
#     zoom_range = 0.2,
#     horizontal_flip = True,
#     vertical_flip = True)




# epochs = 25
# batch_size = 32
# test_datagen = ImageDataGenerator(rescale = 1./255)
# train_generator = train_datagen.flow(
#     # train_dir,
#     x_train,
#     y_train,
#     # target_size = (75, 100, 3),
#     batch_size = batch_size,
#   )
# test_generator = test_datagen.flow(
#     x_test,
#     y_test,
#     batch_size = batch_size)
    
#     #####

# ##
# subj = x_train_full[relcols]
# agemean = subj.age.mean()
# agesd = subj.age.std()
# subj.age -= agemean
# subj.age /= agesd

# subjective_inputs = Input(shape = subj.shape,
#                     dtype = 'float32',
#                     name = 'subject')
# subj_in = layers.Dense(1056, activation = 'relu')(subjective_inputs)                
# subj_in = layers.Dense(256, activation = 'relu')(subj_in)                
# subj_in = layers.Dense(128, activation = 'relu')(subj_in)           
# # subj_in = layers.Flatten()(subj_in)
# # subj_in = layers.Dense(7, activation = 'softmax')(subj_in)
# img_train = np.asarray(x_train_full.image.tolist())
# img_test = np.asarray(x_test_full.image.tolist())

# # img_train = img_train.reshape(7932, 100*75, -1)
# img_input = Input(shape = (100, 75, 3),
#                     dtype = 'float32',
#                     name = 'images')
#                     # (x_train)
# img_in = layers.Conv2D(32,(3, 3), activation = 'relu', input_shape = (100, 75, 3))(img_input)
# img_in = layers.MaxPooling2D(2, 2)(img_in)
# img_in = layers.Conv2D(64,(3, 3), activation = 'relu')(img_in)
# img_in = layers.MaxPooling2D(2, 2)(img_in)
# img_in = layers.Conv2D(128,(3, 3), activation = 'relu')(img_in)
# img_in = layers.MaxPooling2D(2, 2)(img_in)

# import keras.backend as K
# diagnosis = layers.concatenate([K.reshape(img_in, (-1, 7932, 128)), subj_in], axis = -1)
# diagnosis = layers.Dense(512, activation = 'relu')(diagnosis)
# diagnosis = layers.Dense(7, activation = 'softmax')(diagnosis)
# model = Model(inputs = [img_input, subjective_inputs], outputs = diagnosis)
# # img_in = layers.Flatten()(img_in)
# # img_in = layers.Dense(7, activation = 'softmax')(img_in)


# diagnosis = layers.concatenate([subj_in, img_in], axis = -1)
# model = Model(inputs = [img_input, subjective_inputs], outputs = diagnosis)
# model.compile(optimizer = 'rmsprop',
#                 loss = 'categorical_crossentropy',
#                 metrics = ['acc'])
# model.fit([img_train,subj.values], x_train_full.dx_code.values)


# gender_train = np.asarray(x_train_full[gencols].values)
# gender_test = np.asarray(x_test_full[gencols].values)
# age_train = np.asarray(x_train_full['age'].tolist())
# age_test = np.asarray(x_test_full['age'].tolist())
# locs_train = np.asarray(x_train_full[loccols].values)
# locs_test = np.asarray(x_test_full[loccols].values)


# gender_input = Input(shape = (7932,3),
#                 dtype = 'float32',
#                 name = 'gender')

# gen_in = layers.Dense(32, activation = 'relu')(gender_input)

                
# age_input = Input(shape = (7932,1),
#                 dtype = 'float32',
#                 name = 'age')

# age_in = layers.Dense(32, activation = 'relu')(age_input)                

# loc_input = Input(shape = (7932, 15),
#                 dtype = 'float32',
#                 name = 'localization')

# loc_in = layers.Dense(32, activation = 'relu')(loc_input)                


# img_input = Input(shape = (100, 75, 3),
#                     dtype = 'float32',
#                     name = 'images')
#                     # (x_train)
# img_in = layers.Conv2D(32,(3, 3), activation = 'relu', input_shape = (100, 75, 3))(img_input)
# img_in = layers.MaxPooling2D(2, strides = 2)(img_in)
# img_in = layers.Conv2D(64,(3, 3), activation = 'relu')(img_in)
# img_in = layers.MaxPooling2D(2, strides = 2)(img_in)
# img_in = layers.Conv2D(128,(3, 3), activation = 'relu')(img_in)
# img_in = layers.MaxPooling2D(2, strides = 2)(img_in)
# img_in = layers.Flatten()(img_in)
# img_in = layers.Dense(7, activation = 'softmax')(img_in)
# img_in.shape
# concatenated = layers.Concatenate(axis = -1)([img_input, loc_input, age_input, gender_input] )
# diagnosis = layers.Dense(7, activation = 'softmax')(concatenated)
                    
# model = Model(inputs = [img_input, loc_input, age_input, gender_input], outputs = [img_in, loc_in, age_in, gen_in])
# model.add(layers.Dense(7, activation = 'softmax'))
# model.compile(optimizer = 'rmsprop',
#                 loss = 'categorical_crossentropy',
#                 metrics = ['acc'])

# plot_model(model, to_file = 'multiinput.png', show_shapes = True)

# model.fit([img_train, locs_train, age_train, gender_train], y_train)

