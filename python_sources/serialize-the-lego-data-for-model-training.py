'''
Read the images from the Lego Brick dataset by Joost Hazelzet, split into
training and validation sets, and serialize. Since I intend to do some
continual learning studies with the data (meaning that I will start with
only a few classes and then iteratively add in more classes), I have chosen 
not to mix the classes into a large array, but to save them as separate datasets
within a large HDF5 file.

Note: the contents of the "train" and "valid" directories appear to be identical,
therefore I will only use the contents of "train", using 80% for my new training
set and the remaining 20% for my validation set.
'''

import os
import random
import numpy as np
from skimage.io import imread
from skimage.color import rgba2rgb
import h5py

import os
import re
def get_imgs(data_dir):
    '''
    returns a file path generator for all files under data_dir
    '''
    fileiter = (os.path.join(root, f)
                for root, _, files in os.walk(data_dir)
                for f in files)
    return fileiter


WRITE = True  # option to write the numpy array of images to an hdf5 file

input_data_dir = '../../DATA/LEGO_brick_images/' # THE PATH TO YOUR LEGO DATA
lego_data_file = 'lego_np_data.h5'  # the output file for the WRITE option

class_dict = { 1  : {'id' : '2357',  'name' : 'Brick corner 1x2x2'},
               2  : {'id' : '3003',  'name' : 'Brick 2x2'},
               3  : {'id' : '3004',  'name' : 'Brick 1x2'},
               4  : {'id' : '3005',  'name' : 'Brick 1x1'},
               5  : {'id' : '3022',  'name' : 'Plate 2x2'},
               6  : {'id' : '3023',  'name' : 'Plate 1x2'},
               7  : {'id' : '3024',  'name' : 'Plate 1x1'},
               8  : {'id' : '3040',  'name' : 'Roof Tile 1x2x45deg'},
               9  : {'id' : '3069',  'name' : 'Flat Tile 1x2'},
               10 : {'id' : '3673',  'name' : 'Peg 2M'},
               11 : {'id' : '3713',  'name' : 'Bush for Cross Axle'},
               12 : {'id' : '3794',  'name' : 'Plate 1X2 with 1 Knob'},
               13 : {'id' : '6632',  'name' : 'Technic Lever 3M'},
               14 : {'id' : '11214', 'name' : 'Bush 3M friction with Cross axle'},
               15 : {'id' : '18651', 'name' : 'Cross Axle 2M with Snap friction'},
               16 : {'id' : '32123', 'name' : 'half Bush'} }

              
# get a list of all filenames in input_images_dir and read as np arrays
filetype = '.png'
training_imgs = []
validation_imgs = []
for nb in [x for x in class_dict]:
    id = class_dict[nb]['id']
    name = class_dict[nb]['name']
    input_images_dir = input_data_dir + 'train/' + id + ' ' + name
    print(input_images_dir)
    fileiter = get_imgs(input_images_dir)
    pngs = [ f for f in fileiter if os.path.splitext(f)[1] == filetype ]
    random.Random(13).shuffle(pngs) # randomize filename order
    
    nb_imgs = len(pngs)
    nb_train = round(nb_imgs * 0.8)  # number of training images for this class
    nb_val  = nb_imgs - nb_train     # number of validation images for this class

    # append each image
    current_trains = []
    current_vals = []
    for ii in range(nb_imgs):
        img = imread(pngs[ii], as_gray='True')
        img = rgba2rgb(img) # remove the alpha channel
        if ii <= nb_train:
            current_trains.append(img)
        else:
            current_vals.append(img)
    training_imgs.append(current_trains)
    validation_imgs.append(current_vals)


# check that they were all read
print("training images")
for i in range(len(training_imgs)):
    print(len(training_imgs[i]))
print("validation images")
for i in range(len(validation_imgs)):
    print(len(validation_imgs[i])) 


if WRITE:
    # write data array to HDF5 file
    print('writing data file (HDF5 format)...')
    h5f = h5py.File(lego_data_file, 'w')
    
    h5f.create_dataset('training-1',   data=training_imgs[0])
    h5f.create_dataset('training-2',   data=training_imgs[1])
    h5f.create_dataset('training-3',   data=training_imgs[2])
    h5f.create_dataset('training-4',   data=training_imgs[3])
    h5f.create_dataset('training-5',   data=training_imgs[4])
    h5f.create_dataset('training-6',   data=training_imgs[5])
    h5f.create_dataset('training-7',   data=training_imgs[6])
    h5f.create_dataset('training-8',   data=training_imgs[7])
    h5f.create_dataset('training-9',   data=training_imgs[8])
    h5f.create_dataset('training-10',   data=training_imgs[9])
    h5f.create_dataset('training-11',   data=training_imgs[10])
    h5f.create_dataset('training-12',   data=training_imgs[11])
    h5f.create_dataset('training-13',   data=training_imgs[12])
    h5f.create_dataset('training-14',   data=training_imgs[13])
    h5f.create_dataset('training-15',   data=training_imgs[14])
    h5f.create_dataset('training-16',   data=training_imgs[15])
    
    h5f.create_dataset('validation-1', data=validation_imgs[0])
    h5f.create_dataset('validation-2', data=validation_imgs[1])
    h5f.create_dataset('validation-3', data=validation_imgs[2])
    h5f.create_dataset('validation-4', data=validation_imgs[3])
    h5f.create_dataset('validation-5', data=validation_imgs[4])
    h5f.create_dataset('validation-6', data=validation_imgs[5])
    h5f.create_dataset('validation-7', data=validation_imgs[6])
    h5f.create_dataset('validation-8', data=validation_imgs[7])
    h5f.create_dataset('validation-9', data=validation_imgs[8])
    h5f.create_dataset('validation-10', data=validation_imgs[9])
    h5f.create_dataset('validation-11', data=validation_imgs[10])
    h5f.create_dataset('validation-12', data=validation_imgs[11])
    h5f.create_dataset('validation-13', data=validation_imgs[12])
    h5f.create_dataset('validation-14', data=validation_imgs[13])
    h5f.create_dataset('validation-15', data=validation_imgs[14])
    h5f.create_dataset('validation-16', data=validation_imgs[15])
    
    h5f.close()
    print('data saved to disk.')          
    
    
#################################################3
'''
The class arrays can be read from the HDF5 file like this:
'''

# with h5py.File(lego_data_file, 'r') as hf:
#     print("READING FILE: ", lego_data_file)
    
#     train_1 = hf['training-1'][:]
#     print("train_1:  ", train_1.shape)
#     train_2 = hf['training-2'][:]
#     print("train_2:  ", train_2.shape)    
#     train_3 = hf['training-3'][:]
#     print("train_3:  ", train_3.shape)
#     train_4 = hf['training-4'][:]
#     print("train_4:  ", train_4.shape)
#     train_5 = hf['training-5'][:]
#     print("train_5:  ", train_5.shape)
#     train_6 = hf['training-6'][:]
#     print("train_6:  ", train_6.shape)
#     train_7 = hf['training-7'][:]
#     print("train_7:  ", train_7.shape)
#     train_8 = hf['training-8'][:]
#     print("train_8:  ", train_8.shape)
#     train_9 = hf['training-9'][:]
#     print("train_9:  ", train_9.shape)
#     train_10 = hf['training-10'][:]
#     print("train_10: ", train_10.shape)
#     train_11 = hf['training-11'][:]
#     print("train_11: ", train_11.shape)
#     train_12 = hf['training-12'][:]
#     print("train_12: ", train_12.shape)
#     train_13 = hf['training-13'][:]
#     print("train_13: ", train_13.shape)
#     train_14 = hf['training-14'][:]
#     print("train_14: ", train_14.shape)
#     train_15 = hf['training-15'][:]
#     print("train_15: ", train_15.shape)
#     train_16 = hf['training-16'][:]
#     print("train_16: ", train_16.shape)

#     validation_1 = hf['validation-1'][:]
#     print("validation_1:  ", validation_1.shape)
#     validation_2 = hf['validation-2'][:]
#     print("validation_2:  ", validation_2.shape)    
#     validation_3 = hf['validation-3'][:]
#     print("validation_3:  ", validation_3.shape)
#     validation_4 = hf['validation-4'][:]
#     print("validation_4:  ", validation_4.shape)
#     validation_5 = hf['validation-5'][:]
#     print("validation_5:  ", validation_5.shape)
#     validation_6 = hf['validation-6'][:]
#     print("validation_6:  ", validation_6.shape)
#     validation_7 = hf['validation-7'][:]
#     print("validation_7:  ", validation_7.shape)
#     validation_8 = hf['validation-8'][:]
#     print("validation_8:  ", validation_8.shape)
#     validation_9 = hf['validation-9'][:]
#     print("validation_9:  ", validation_9.shape)
#     validation_10 = hf['validation-10'][:]
#     print("validation_10: ", validation_10.shape)
#     validation_11 = hf['validation-11'][:]
#     print("validation_11: ", validation_11.shape)
#     validation_12 = hf['validation-12'][:]
#     print("validation_12: ", validation_12.shape)
#     validation_13 = hf['validation-13'][:]
#     print("validation_13: ", validation_13.shape)
#     validation_14 = hf['validation-14'][:]
#     print("validation_14: ", validation_14.shape)
#     validation_15 = hf['validation-15'][:]
#     print("validation_15: ", validation_15.shape)
#     validation_16 = hf['validation-16'][:]
#     print("validation_16: ", validation_16.shape)
