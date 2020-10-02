import multiprocessing
import math

import sys
import sklearn

import tensorflow as tf
from tensorflow import keras
import datetime
import glob
import numpy as np
import pandas as pd
import os
import ast

import skimage.io
import cv2

from contextlib import ExitStack

from tqdm.notebook import tqdm

np.random.seed(42)

import matplotlib as mpl
import matplotlib.image as mpimg


PATH = '../input/global-wheat-detection'
TRAIN_IMAGES_PATH = os.path.join(PATH,'train')
TEST_IMAGES_PATH = os.path.join(PATH,'test')
TRAIN_EXT = 'jpg'
TEST_EXT = 'jpg'
OUT_IMG_SIZE = 416
TFREC_DIR = '/kaggle/working'
NUM_SLIDES_IN_TFRECORD = 900
TFRECORD_FILE_NAME = 'record'
PROCESSING_VERBOSE = 2


class Time:
    """
    Class for displaying elapsed time.
    """

    def __init__(self):
        self.start = datetime.datetime.now()

    def elapsed_display(self):
        time_elapsed = self.elapsed()
        print("Time elapsed: " + str(time_elapsed))

    def elapsed(self):
        self.end = datetime.datetime.now()
        time_elapsed = self.end - self.start
        return time_elapsed
    
def get_image_names_with_shape():
    result = list()
    img_paths = [os.path.join(TRAIN_IMAGES_PATH,img) 
                 for img in glob.glob1(TRAIN_IMAGES_PATH,f'*.{TRAIN_EXT}')]
    for img_path in tqdm(img_paths):
        img_shape, image_id = cv2.imread(img_path).shape, os.path.basename(img_path).replace(f'.{TRAIN_EXT}','')
        result.append([img_shape[0], img_shape[1], image_id])

    return pd.DataFrame(result, columns = ['img_width', 'img_height', 'image_id'])
    
def prepare_for_records(data=pd.DataFrame, data_type='train'):
    if data_type!='test':
        data[['x_min','y_min', 'bwidth', 'bheight']] = pd.DataFrame([ast.literal_eval(x) 
                                                                     for x in data.bbox.tolist()], 
                                                                    index= data.index)
    #     img_shapes = get_image_names_with_shape()
    #     data = data.merge(img_shapes, how='inner', on=['image_id'])

        data['x_max'] = (data['x_min'] + data['bwidth'])/data['width']
        data['y_max'] = (data['y_min'] + data['bheight'])/data['height']
        data['x_min'] = data['x_min']/data['width']
        data['y_min'] = data['y_min']/data['height']

        unique_img_data = data[['image_id', 'source', 'width', 'height']].drop_duplicates()
        unique_img_data = unique_img_data.reset_index().drop('index', axis=1)
    else:
        imgs = glob.glob1(TEST_IMAGES_PATH,f'*.{TEST_EXT}')
        imgs = [img_id.replace(f'.{TEST_EXT}','') for img_id in imgs]
        data = pd.DataFrame(imgs, columns=['image_id'])
        unique_img_data = data
    
    return data, unique_img_data

def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def image_example(data, image_features, data_type): 
    if data_type!='test':
        height = image_features['height']
        width = image_features['width']
        filename = tf.compat.as_bytes(image_features['image_id'])
        source = tf.compat.as_bytes(image_features['source'])

        img_path = os.path.join(TRAIN_IMAGES_PATH, image_features['image_id']+"."+TRAIN_EXT)
        img = np.array(mpimg.imread(img_path))
    #     img = tf.image.resize(img, (OUT_IMG_SIZE, OUT_IMG_SIZE)) #cv2.resize(img, (OUT_IMG_SIZE, OUT_IMG_SIZE))
        img_data = tf.io.encode_jpeg(img)

        encoded_image_data = img_data
        image_format = tf.compat.as_bytes(TRAIN_EXT) 

        num_bounding_boxes = len(data)

        xmins = data['x_min'].values 
        xmaxs = data['x_max'].values

        ymins = data['y_min'].values 
        ymaxs = data['y_max'].values

        classes_text = [b'Wheat'] * num_bounding_boxes
        classes = [0] * num_bounding_boxes 

        tf_example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': int64_feature(height),
          'image/width': int64_feature(width),
          'image/filename': bytes_feature(filename),
          'image/source_id': bytes_feature(source),
          'image/encoded': bytes_feature(encoded_image_data),
          'image/format': bytes_feature(image_format),
          'image/object/bbox/xmin': float_list_feature(xmins),
          'image/object/bbox/xmax': float_list_feature(xmaxs),
          'image/object/bbox/ymin': float_list_feature(ymins),
          'image/object/bbox/ymax': float_list_feature(ymaxs),
          'image/object/class/text': bytes_list_feature(classes_text),
          'image/object/class/label': int64_list_feature(classes),
        }))   
    else:
        filename = tf.compat.as_bytes(image_features['image_id'])
        img_path = os.path.join(TEST_IMAGES_PATH, image_features['image_id']+"."+TEST_EXT)
        img = np.array(mpimg.imread(img_path))
        img_data = tf.io.encode_jpeg(img)

        encoded_image_data = img_data
        image_format = tf.compat.as_bytes(TEST_EXT)
        
        tf_example = tf.train.Example(features=tf.train.Features(feature={
          'image/filename': bytes_feature(filename),
          'image/encoded': bytes_feature(encoded_image_data),
          'image/format': bytes_feature(image_format),
        })) 

    return tf_example

def write_data_to_tfrecords(data, unique_img_data, num_list, filename_prefix, num_process, data_type):
    n_shards = math.ceil(len(num_list)/NUM_SLIDES_IN_TFRECORD)
    os.makedirs(TFREC_DIR, exist_ok=True)
    
    paths = [TFREC_DIR+"/"+"{}-{}.tfrecord-{:03d}-{:05d}-of-{:05d}".format(filename_prefix, 
                                                                              TFRECORD_FILE_NAME, num_process, 
                                                                              index, n_shards)
             for index in range(n_shards)]
    
    with ExitStack() as stack:
        writers = [stack.enter_context(tf.io.TFRecordWriter(path))
                   for path in paths]
        for num in num_list:
            image_features = unique_img_data.loc[num]
            subdata = data[data['image_id'] == image_features['image_id']]
            
            if data_type=='test':
                _dir = os.path.join(TEST_IMAGES_PATH, image_features['image_id']+"."+TEST_EXT)
            else:
                _dir = os.path.join(TRAIN_IMAGES_PATH, image_features['image_id']+"."+TRAIN_EXT)
            
            if os.path.exists(_dir):            
                ind = num_list.index(num)
                shard = math.floor(ind/NUM_SLIDES_IN_TFRECORD)
                writers[shard].write(image_example(subdata, image_features, data_type).SerializeToString())
    
    return num_list
            
def multiprocess_write_data_to_tfrecords(data,
    unique_img_data,
    num_list,
    filename_prefix='record',
    data_type = 'train'
    ):
    timer = Time()
    
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)

    num_images = len(num_list)
    
    if num_processes > num_images:
        num_processes = num_images
    images_per_process = num_images / num_processes

    if PROCESSING_VERBOSE>0:
        print ('Number of processes: ' + str(num_processes))
        print ('Number of training images: ' + str(num_images))

    tasks = []
    
    for num_process in range(1, num_processes + 1):
        start_index = (num_process - 1) * images_per_process + 1
        end_index = num_process * images_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        
        sublist = num_list[start_index - 1:end_index]
        tasks.append((data, unique_img_data, sublist, filename_prefix, num_process, data_type))
        if PROCESSING_VERBOSE>1:
            print ('Task #' + str(num_process) + ': Process images ' \
                + str(sublist))
        
    results = []
    for t in tasks:
        results.append(pool.apply_async(write_data_to_tfrecords, t))        

    for result in results:
        (image_nums) = result.get()
        if PROCESSING_VERBOSE>1:
            print ('Done processing: %s' % image_nums)

    if PROCESSING_VERBOSE>0:
        print ('Time to generate records (multiprocess): %s\n' \
            % str(timer.elapsed()))