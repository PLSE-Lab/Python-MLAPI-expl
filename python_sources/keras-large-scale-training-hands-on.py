#!/usr/bin/env python
# coding: utf-8

#  <div style="text-align:center"><span style="color:teal; font-family:Georgia; font-size:2em;">How to Train you Keragon !</span></div>
#   <div style="text-align:center"><span style="color:brown; font-family:Georgia; font-size:1.5em;">Session 2: Keragon In Action</span></div>
# <p> </p>
# *__Abstract-__ The real world application of custom generator is bit complex procedure. In this secion we will see how to build a traing and validation work flow using custom generator. To take the advantage of custom generator we also need to pre process the images and save them as numpy batches. Then keras model can easily read those numpy batches and train the model.*
#   
#   **Keyword: Image Classification, Keras, Tensorflow, Keras Image Generators**

# **Please Read the [Session 1][1] to have a better understanding. For some reason when I commint the notebook it is not able to find the output data locaiton so I Put the code in markdown. To run it please un-quote the code block convert them in to code from markdown.**
# 
# [1]: https://www.kaggle.com/datapsycho/training-large-scale-data-with-keras-and-tf

# # Regular Imports

# ### preprocession import
# ```python
# import os
# from keras.preprocessing.image import load_img
# import numpy as np
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# from itertools import islice
# from keras.preprocessing.image import ImageDataGenerator, img_to_array
# from sklearn.utils import shuffle as skf
# import keras
# ```

# 
# ### model build import
# ```python
# from keras import layers
# from keras import models
# from keras import optimizers
# from keras import backend as K
# ```

# ```python
# # run this code first time to create directories to save numpy bathhes
# # os.mkdir("../output/xray_project/train/")
# # os.mkdir("../output/xray_project/val/")
# ```

# ## Image Pre-Processing:
# So after successfull execution of _[previous workflow][1]_ now its time to train the Keragon. Though its not a big data set. Lets first have a look on the PrePixer build by DataPsycho. That's the workflow for PrePixer:
# - Load the image file path into category list (`con_dir, prod_dir`)
# - Attach a `ImageDataGenerator`
# - Combine all the files in to a list
# - Mix the files randomly
# - Conver the list in to small list of list
# - Convert each list of images in to `.npy` files
# - Continue the execution Until the last batch
# 
# [1]: https://www.kaggle.com/datapsycho/training-large-scale-data-with-keras-and-tf

# ```python
# class PrePixer(object):
#   def __init__(self, cons_dir, prod_dir, mode, chunk_size):
#     self.cons_dir = cons_dir
#     self.prod_dir = prod_dir
#     self.mode = mode
#     self.chunk_size = chunk_size
# 
#     self.data_gen = ImageDataGenerator(
#       rescale=1./255,
#       rotation_range=40,
#       width_shift_range=0.2,
#       height_shift_range=0.2,
#       shear_range=0.2,
#       zoom_range=0.2,
#       horizontal_flip=True)
# 
#   # crate a list of batches based on the total file and desired batch size
#   def chunk(self, it, size):
#     it = iter(it)
#     return iter(lambda: tuple(islice(it, size)), ())
# 
#   def pre_pixer(self):
#     # select the file directory
#     _cons_dir = self.cons_dir
#     _prod_dir = self.prod_dir
#     # list all the image files
#     cons_list = os.listdir(_cons_dir)
#     prod_list = os.listdir(_prod_dir)
#     # select the file bese on minimum to have 50-50 proportion
#     min_len = min([len(cons_list), len(prod_list)])
#     cons_list = [{"file_name":item, "label":0} for item in cons_list if ".jpeg" in item][: int(min_len)]
#     prod_list = [{"file_name":item, "label":1} for item in prod_list if ".jpeg" in item][: int(min_len)]
#     file_list = cons_list + prod_list
#     total_len = len(file_list)
#     # Shuffle the file for proper mix of both class
#     file_list = skf(file_list, random_state=13)
# 
#     # create list of list based on batch size parameter
#     chunks = list(self.chunk(file_list, self.chunk_size))
# 
#     for idx, meta_files in enumerate(chunks, 1):
#       X_file = []
#       y_file = []
#       batch_size_i = len(meta_files)
#     #take each image and convert to numpy array
#       for item in meta_files:
#         label, file_name = item.get("label"), item.get("file_name")
#         if label ==0:
#           y_file.append([label])
#           img_path_i = os.path.join(_cons_dir, file_name)
#           img_i = load_img(img_path_i, target_size=(128, 128))
#           img_i = img_to_array(img_i)
#           exp_img = np.expand_dims(img_i, axis=0)
#           X_file.append(exp_img)
#         else:
#           y_file.append([label])
#           img_path_i = os.path.join(_prod_dir, file_name)
#           img_i = load_img(img_path_i, target_size=(128, 128))
#           img_i = img_to_array(img_i)
#           exp_img = np.expand_dims(img_i, axis=0)
#           X_file.append(exp_img)
#       # a batch of numpy array
#       X_file = np.concatenate(X_file)
#       y_file = np.concatenate([y_file])
#      # for validaton we do not need to augment the file so we will save them directly
#       if self.mode == "val":
#         dump_loc = "../output/xray_project/val/"
#         x_file_name = "image_file_{}.npy".format(idx)
#         y_file_name = "label_file_{}.npy".format(idx)
#         np.save(os.path.join(dump_loc, x_file_name), X_file)
#         np.save(os.path.join(dump_loc, y_file_name), y_file)
#         print("Done batch for validation {}/{}".format(idx, len(chunks)))
#       # for train we will have a augmented file of each image
#       elif self.mode == "train":
#         image_flow = self.data_gen.flow((X_file, y_file), batch_size=batch_size_i)
#         aug_x_set = []
#         aug_y_set = []
#         for aug_x, aug_y in image_flow:
#           aug_x_set.append(aug_x)
#           aug_y_set.append(aug_y)
#           break
#         aug_y_set = np.concatenate(aug_y_set)
#         aug_x_set = np.concatenate(aug_x_set)
#         X_file = np.concatenate([X_file/255.0] + [aug_x_set])
#         y_file = np.concatenate([y_file] + [aug_y_set])
#         dump_loc = "../output/xray_project/train/"
#         x_file_name = "image_file_{}.npy".format(idx)
#         y_file_name = "label_file_{}.npy".format(idx)
#         np.save(os.path.join(dump_loc, x_file_name), X_file)
#         np.save(os.path.join(dump_loc, y_file_name), y_file)
#         print("Done batch for train {}/{}".format(idx, len(chunks)))
#       else:
#         raise Exception("Only <val> or <train> is available as parameter")
#     return print("DB Created")
# ```

# Now its time to run the bathces and save to file in to directory. For training batches a augmented image is considered. Whcich means the final sample size for training will be dubled.

# ```python
# con_dir = "../input/chest_xray/chest_xray/train/NORMAL"
# prod_dir = "../input/chest_xray/chest_xray/train/PNEUMONIA/"
# pxer = PrePixer(con_dir, prod_dir, mode="train", chunk_size=128)
# pxer.pre_pixer()
# ```

# ```python
# con_dir = "../input/chest_xray/chest_xray/val/NORMAL"
# prod_dir = "../input/chest_xray/chest_xray/val/PNEUMONIA/"
# pxer = PrePixer(con_dir, prod_dir, mode="val", chunk_size=64)
# pxer.pre_pixer()
# ```

# ## The Custom Generator
# Then it's time to build the custom generator for that particular data. The custom generator has been explained in detail in [previous post][1].
# 
# [1]: https://www.kaggle.com/datapsycho/training-large-scale-data-with-keras-and-tf

# ```python
# class DataGenerator(keras.utils.Sequence):
#     def __init__(self, list_IDs, mode):
#         'Initialization'
#         self.list_IDs = list_IDs
#         self.on_epoch_end()
#         self.mode = mode
#         
#     def __len__(self):
#         return int(len(self.list_IDs))
# 
#     def __getitem__(self, index):
#         'Generate one batch of data'
#         indexes = self.indexes[index:(index+1)]
#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]
#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)
#         return X, y
# 
#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))
#         
#     def load_file(self, id_list, location):
#         list_IDs_temp = id_list
#         data_loc = location
#         for ID in list_IDs_temp:
#             x_file_path = os.path.join(data_loc, ID)
#             y_file_path = os.path.join(data_loc, image_label_map.get(ID))
#             # Store sample
#             X = np.load(x_file_path)
#             # Store class
#             y = np.load(y_file_path)
#         return X, y        
# 
#     def __data_generation(self, list_IDs_temp):
#         """Generates data containing batch_size samples"""
#         if self.mode == "train":
#             data_loc = "../output/xray_project/train/"
#             # Generate data
#             X, y = self.load_file(list_IDs_temp, data_loc)
#             return X, y
#         elif self.mode == "val":
#             data_loc = "../output/xray_project/val/"
#             X, y = self.load_file(list_IDs_temp, data_loc)
#             return X, y
#  ```
#         

# **Here he load all the batch file location for the training.**

# ```python
# # ====================
# # train set
# # ====================
# all_files_loc ="../output/xray_project/train/"
# all_files = os.listdir(all_files_loc)
# image_label_map = {
#         "image_file_{}.npy".format(i+1): "label_file_{}.npy".format(i+1)
#         for i in range(int(len(all_files)/2))}
# partition = [item for item in all_files if "image_file" in item]
# 
# # ====================
# # validation set
# # ====================
# 
# all_val_files_loc = "../output/xray_project/val/"
# all_val_files = os.listdir(all_val_files_loc)
# val_image_label_map = {
#         "image_file_{}.npy".format(i+1): "label_file_{}.npy".format(i+1)
#         for i in range(int(len(all_val_files)/2))}
# val_partition = [item for item in all_val_files if "image_file" in item]
# ```

# ```python
# training_generator = DataGenerator(partition, "train")
# validation_generator = DataGenerator(val_partition, "val")
# ```

# # The Model
# That a VGG like basic model for training image with 2 class.

# ```python
# #=====================
# # Defining a model
# #=====================
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
# ```

# ```python
# # Model execution.
# hst = model.fit_generator(generator=training_generator, 
#                            epochs=5, 
#                            validation_data=validation_generator,
#                           # for gpu training only though I am not using gpu
#                            use_multiprocessing=True,
#                            max_queue_size=32)
# ```
