# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import print_function

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os, sys
import inspect, json, pickle

#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle

from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from copy import deepcopy

from math import ceil
import keras
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, AveragePooling2D, Input, Flatten
from keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model

class hqmap_training_data():

    def __init__(self):
        ''' defines some paths and reads x data in pickle format.
            prerequisite is x data in pickle format... find a better alternative? '''        
        self.inpath = Path(r"/kaggle/input/a-surprisingly-difficult-image-dataset-heroquest/")
        self.model_path = Path(r"/kaggle/models/")
        self.maps_path = Path(r"/kaggle/maps/")
        self.x_data_package_path = self.inpath / 'x_data.pck'
        self.df_x = pd.read_pickle(self.x_data_package_path).fillna(0)

        self.prepare_x_vectors()
        self.prepare_y_vectors()
        self.training_data()
        
        
    def prepare_x_vectors(self):
        ''' prepare several X vectors that can be used for the neural nets
        '''
        
        # X_big contains the 78x78 pixel picture showing the near surrounding
        # of the target square
        X_big = self.df_x['pic'].values
        X_big = np.stack(X_big)
        self.X_big = X_big.reshape((len(self.df_x), 78, 78, 1))
        
        # x_data contains the 34x34 pixel picture of the center squre only
        # making it ideal to identify one-square-items but not multi square
        # items.
        X_data = self.df_x['center_pic'].values
        X_data = np.stack(X_data)
        self.X_data = X_data.reshape((len(self.df_x), 34, 34, 1))
        
        # get auxiliary data if possible. Currently that's min and max color
        # of the center of the small image
        minc = np.array(self.df_x['min_color'].tolist())
        maxc = np.array(self.df_x['max_color'].tolist())
        self.X_aux = np.hstack([minc, maxc])

    
    def prepare_y_vectors(self):
        '''take the data from the door vector and the small items vector 
        and make an machine readable categorical vector from it.
        
        We have several Y categories: 
            Furniture and Monster symbols for one,
            Doors as a second (as they need to be distinguished by direction)
            Rooms as a third (Because they maybe need another picture frame)
            and maybe later minisymbols as a fourth
        '''
        
        self.load_class_dict()
        
        df_y = deepcopy(self.df_x[['filename', 'x', 'y', 'real_Y', 'door_Y',  'room_y']])
        df_y['real_Y'] = df_y['real_Y'].apply(str).apply(str.strip)

        df_y.loc[df_y['real_Y']=="other", 'real_Y'] = 0
        df_y.loc[:,'real_Y_numeric'] = df_y['real_Y'].apply(self.Y_dict_forw.get, 0)
        df_y['real_Y_numeric'] = df_y['real_Y_numeric'].fillna(0)
        self.df_y = df_y

        self.Y = to_categorical(df_y['real_Y_numeric'], self.num_classes).astype(int)

        df_y.loc[:,'door_Y_numeric'] = df_y['door_Y'].apply(self.Y_dict_forw.get, 0)
        df_y['door_Y_numeric'] = df_y['door_Y_numeric'].fillna(0)
        self.Y_door = to_categorical(df_y['door_Y_numeric'], self.num_classes).astype(int)
        
        df_y.loc[:,'room_Y_numeric'] = df_y['room_y'].apply(self.Y_dict_forw.get, 0)
        df_y['room_Y_numeric'] = df_y['room_Y_numeric'].fillna(0)
        self.Y_room = to_categorical(df_y['room_Y_numeric'], self.num_classes).astype(int)
        


    def load_class_dict(self, makenew = False):
        ''' loads the dict of classes that was saved to the model path.
        if it has changed, the model will have to be retrained. 
        Also loads of X values to be able to reuse it later
        in prediction
        
        In both cases, if the file is not found in the expected place, it will
        be regenerated - with the consequence of a need for retraining.
        '''
        json_path = self.model_path / "translation.json"
        
        if json_path.exists() and makenew == False:
            with open(json_path, 'r') as f:
                Y_dict_back = json.load(f)
        else:
            Y_dict_back = self.make_new_class_dict()
        Y_dict_forw = {ouptput_class:i for i, ouptput_class in Y_dict_back.items()}
        
        self.num_classes = len(Y_dict_back)
        self.Y_dict_back = Y_dict_back
        self.Y_dict_forw = Y_dict_forw
        

    def training_data(self, makenew = False):
        mean_path = self.model_path /'x_mean.pickle'
        if mean_path.exists() and makenew == False:
            with open(str(mean_path), 'rb') as f:
                x_mean = pickle.load(f) 
                self.prepare_training_data(x_mean)
        else:
            self.prepare_training_data()


    def prepare_training_data(self, x_mean = []):
        ''' splits x and y data in training- and test datasets, 
        calculates a mean over the training set and preserves it.
        '''
        x_train, x_test, self.y_train, self.y_test = train_test_split(self.X_data, self.Y, test_size = 0.20,  
                                                    random_state =np.random.randint(0, 1000),
                                                    shuffle = True)
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        print('y_train shape:', self.y_train.shape)
        
        # Normalize data.
        self.x_train = x_train.astype('float32') / 255
        self.x_test = x_test.astype('float32') / 255
        
        
        # subtract pixel mean 
        #if len(x_mean) > 0:
        #    self.x_mean = np.mean(x_train, axis=0)
        #else:
        #    self.x_mean = x_mean
        #self.x_train -= self.x_mean
        #self.x_test -= self.x_mean
        #self.save_x_mean()
        
        #self.X_all = (self.X_data.astype('float32') / 255) - self.x_mean
        

    def find_underrepresented_labels(self, howmany = 10):
        labeldict = {}
        labellist = []
        for label in self.df_x['real_Y'].unique():
            labeldict[label] = sum(self.df_x['real_Y'] == label)
            labellist.append(sum(self.df_x['real_Y'] == label))
        for key, value in sorted(labeldict.items(), key=lambda item: item[1],reverse=False)[:howmany]:
            print(key, ": ", value)        
        print("average: ", np.floor(np.mean(labellist)))
        print("10th percentile: ", np.floor(np.percentile(labellist, 10)))

    
    def make_new_class_dict(self):
        ''' generates a new translation dict if needed. Attention: If this
        dict is changed, the neural net needs a new model and new training!'''
        Y_classes = set()
        for y in list(self.df_x[:]['real_Y']):
            Y_classes.add(str(y))
        for y in list(self.df_x[:]['door_Y']):
            Y_classes.add(str(y))
        for y in list(self.df_x[:]['room_y']):
            Y_classes.add(str(y))
        #if 0 in Y_classes:
        #    Y_classes.remove(0)
        Y_dict_back = {i:ouptput_class for i, ouptput_class in enumerate(sorted(Y_classes))}
        
        # save new translation dict 
        if not self.model_path.exists():
            os.makedirs(self.model_path)

        json_path = self.model_path / "translation.json"
        with open(json_path, 'w') as f:
            json.dump(Y_dict_back, f)        
        
        return Y_dict_back
 
    
    def load_new_y_labels(self):
        '''Reads a hand-edited y vector excel table and replace the labels in the data df
        No new lines can be added by this - there has to be the same number of
        lines like in the original dataframe
        '''
        self.df_y = pd.read_excel(str(self.inpath / "y_vector.xlsx")) \
                        .fillna(value = 0)
        self.df_x['real_Y'] = self.df_y['real_Y']
        self.df_x['door_Y'] = self.df_y['door_Y']
        self.df_x['room_y'] = self.df_y['room_y']
 
    
    def save_y_data(self):
        self.df_y.to_excel(str(self.inpath / "y_vector.xlsx"))


    def save(self):
        ''' saves x data and the mean of the training data for later reuse '''
        self.df_x.to_pickle(str(self.x_data_package_path))
        self.save_x_mean()

        
    def save_x_mean(self):
        ''' If someone tries to enhance the training results by applying 
        standardization, the mean must be saved to have it available when
        the model is used to predict unknown maps outside the training 
        set'''
        mean_path = self.model_path /'x_mean.pickle'
        with open(mean_path, 'wb') as f:
            pickle.dump(self.x_mean, f)  


    def plot_images(self, mask=[], start_x=None, end_x=None, 
                     big = False, 
                    label = "real_Y", images_per_row = 7):
        ''' print outs some chosen images and the label that was chosen along
        with it; plus their index.
        
        Takes either a mask of len(n_samples) or a start and end integer.
        
        The switch "big" decides, which picture is shown.
        
        label determines which column is used as label,
        
        images per row decides how many images are squeezed in each output row.'''
        
        if big:
            col = 'pic'
        else:
            col = 'center_pic'
            
        if label not in self.df_x.columns:
            label = 'real_Y'
        
        if len(mask) > 0:
            images = self.df_x.loc[mask, col].values
            indexes = list(self.df_x.loc[mask, col].index)
            labels = self.df_x.loc[mask, label].values
        else:
            if (isinstance(start_x, type(None)) or isinstance(end_x, type(None))):
                return
            images = self.df_x.loc[start_x:end_x, col].values
            indexes = list(self.df_x.loc[start_x:end_x, col].index)
            labels = self.df_x.loc[start_x:end_x, label].values

        no_of_images = len(images)

        for i in range(0, no_of_images, images_per_row):
            fig, axes = plt.subplots(nrows=1, ncols=images_per_row, 
                                     figsize = (images_per_row*0.975, 5))
            for j in range(0, images_per_row):
                if i+j >= no_of_images:
                    break
                axes[j].autoscale_on = True
                axes[j].matshow(images[i+j], cmap = plt.cm.binary_r)
                axes[j].annotate(labels[i+j], rotation=15, xy=(0.1, 1.1),
                                xytext=(0.1, 1.1), textcoords='axes fraction')
                axes[j].set_xticks([])
                axes[j].set_yticks([])
            plt.show()
            spaces =  ' '* (7 - len(str(indexes[i])))
            print('  ' + spaces.join([str(idx) for idx in indexes[i:i+images_per_row]]))


    ''' Add_files '''
    @staticmethod
    def crop(infile, width, height, xoff, yoff, startxoff, startyoff):
        ''' cuts the incoming picture in pieces and gives back 
        small image snippets plus their x and y positions'''
        im = Image.open(infile)
        imgwidth, imgheight = im.size
        for i in range(int(imgheight/yoff)):
            for j in range(int(imgwidth/xoff)):
                box = (int(j * xoff) + startxoff, 
                       int(i * yoff)  + startyoff, 
                       int(j * xoff + width), 
                       int(i * yoff + height))
                yield [im.crop(box), j, i]
    
    
    @staticmethod                
    def cut_map(self, maps_path, file, snippets_path):
        ''' cuts down a file that adheres to the stanard png size of 33.33 pixel per square,
        if a square has content, it is saved and the function notes some metadata.'''
        im_list = []
        color_pick_box = (28, 28, 50, 50)
        center_image_box = (22, 22, 56, 56)
        for item in self.crop(str(maps_path / file), 56, 56, 33.33, 33.33, -22, -22):
            #item[0] is the image, item[1] is x, item[2] is y
            metadata = {}
            colors = item[0].crop(color_pick_box).getextrema()
            metadata['min_color'] = list(np.array(colors)[:,0])
            metadata['max_color'] = list(np.array(colors)[:,1])
            if metadata['min_color'] == metadata['max_color']:
                continue
            else:
                metadata['filename'] = str("snippets\\" + file + str(item[1]) + "_" + str(item[2]) + ".png")
                metadata['x'] = item[1]
                metadata['y'] = item[2]
                im = item[0]
                im = im.convert('LA')
    
                npic = np.array(im) 
                npic = npic[:,:,0] # keep grey channel, discard alpha channel
                metadata['pic'] = npic
                
                center_im = im.crop(center_image_box)
                colors = center_im.getextrema()
                min_c = sum(list(np.array(colors)[:,0]))/3
                max_c = sum(list(np.array(colors)[:,1]))/3
                
                #center_im = ImageOps.autocontrast(center_im)
                #size = center_im.size
                npic = np.array(center_im) 
                npic = npic[:,:,0] # keep grey channel, discard alpha channel
                npic = npic - min_c
                npic = npic * 255 / (max_c - min_c)
                center_im = Image.fromarray(npic).convert('LA')
                metadata['center_pic'] = npic
                #im.save(metadata['filename'])
                #center_im.save(metadata['filename'][:-4] + 'cnt.png')
                im_list.append(metadata)
        return im_list

    def add_images(self):
        ''' goes to the maps folder, checks for new map png's, crops the 
        map into learnable pieces and adds them to the x vector.
        This was used to generate the dataset. It will probably work 
        only on your laptop at home, not in kaggle.'''
        
        metadata = []
        maps =  list(filter(lambda s: ".png" in s, os.listdir(self.maps_path)))

        for inmap in maps:
            metadata.extend(self.cut_map(self.maps_path, self.inmap, self.snippets_path))

        metadata_filename = str(self.inpath / "metadata.xlsx")
        df_more_pics = pd.DataFrame(metadata)
        df_more_pics.to_excel(metadata_filename)
        
        self.df_x = self.df_x.append(df_more_pics, ignore_index = True, sort = False)
        self.df_y = self.df_y.append(df_more_pics[['x', 'y', 'filename', 'center_pic']], ignore_index = True, sort = False)
        self.df_y.to_excel(str(self.inpath / "y_vector.xlsx"), index = False)

        self.save()

##########################################################################################
##########################################################################################
# Second part of the script: Train a neural net

hqmap = hqmap_training_data()
print(hqmap.num_classes, ' classes')
input_shape = hqmap.x_train[0].shape


def augmentation():
    return ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=True,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.15,
        height_shift_range=0.15,

        shear_range=0.00,  # set range for random shear
        zoom_range=0.00,  # set range for random zoom

        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

# This model comes nearly 1:1 from the keras example page
# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------

n = 5
depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, 2)


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, num_classes):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 100:
        lr *= 0.5e-3
    elif epoch > 80:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 50:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


model = resnet_v2(input_shape=input_shape, depth=depth, num_classes = hqmap.num_classes)
model.compile(loss='categorical_crossentropy',
              optimizer = Adagrad(),
              #optimizer=Adam(lr=lr_schedule(0)),
              metrics=['categorical_accuracy', 'accuracy'])
model.summary()
print(model_type)


x_train = hqmap.x_train
x_test = hqmap.x_test

y_train = hqmap.y_train
y_test = hqmap.y_test

params = [
          {'batch_size': 8,      'epochs': 2,     'lr': 1e-3,    'max_epoch' : 1000,},
          {'batch_size': 16,     'epochs': 3,     'lr': 1e-4,    'max_epoch' : 500,}, 
          {'batch_size': 60,     'epochs': 20,     'lr': 1e-5,    'max_epoch' : 500,},
          {'batch_size': 120,     'epochs': 150,  'lr': 1e-5,    'max_epoch' : 100,},
         ]

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)


# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_categorical_accuracy',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]



# This will do preprocessing and realtime data augmentation:
datagen = augmentation()
datagen.fit(x_train)

# Fit the model on the batches generated by datagen.flow().
for param in params:
    print("Batch size is now: ", param['batch_size'])
    #x_train, y_train = shuffle(x_train, y_train)
    
    model.fit_generator(datagen.flow(x_train, y_train, batch_size = param['batch_size']),
                    validation_data = (x_test, y_test),
                    epochs = param['epochs'], 
                    steps_per_epoch = min(ceil(x_train.shape[0] / (param['batch_size'])), param['max_epoch']),
                    shuffle = True,
                    verbose = 1, 
                    workers = 8,
                    callbacks = callbacks,)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])