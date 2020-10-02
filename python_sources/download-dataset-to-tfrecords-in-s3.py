# -*- coding: utf-8 -*-
# !/usr/bin/python

import sys, os, multiprocessing, csv
from urllib import request
from PIL import Image
from io import BytesIO
import boto3
import tensorflow as tf
from itertools import zip_longest

# NOTES
# The script needs to pull your S3 credentials for the Boto library, so you need to download the script to run it 
# This will take a lot of network bandwidth, so it's preferable to run on an AWS instance

# CREDIT
# I used the kernel from kaggle user ~anokas~ as a starting point, added everything related to tfrecords

# CONFIGURATION
tfrecord_batch_size = 500 # number of images to store in each tfrecord file
n_processes = None # num threads to run the script on, None defaults to n_cpus
input_path = "path/to/input/csv"
tfrecord_temp_dir = "tfrecord_temp/" # name of directory to store tfrecord files temporarily until they are uploaded to S3
dest_dir = "train_data/" # names of your test and train "directories" in S3
bucket_name = 'my-bucket' # the name of your S3 bucket

# HELPER FUNCTIONS

# creates a tfrecord file from a batch of keys and urls
def create_and_upload_tfrecord_from_batch(enumerated_batch):
    index, tfrecord_batch = enumerated_batch
    if index in tfrecord_indices_processed:
        print('already completed tfrecord', index, '- skipping')
    else:
        try:
            examples = []
            for image_metadata in tfrecord_batch:
                try:
                    if image_metadata is not None: # the last batch is padded with None if undersized
                        url = image_metadata[1]
                        label = image_metadata[2] if len(image_metadata) > 2 else -1 # label is -1 if running on test data
                            
                        jpeg_string, shape = load_image_data_from_url(url)
                        example = create_example_from_jpeg_string(jpeg_string, shape, label)
                        examples.append(example)
                except Exception as e:
                    print('Failed to create example for image with url:', url)

            tfrecord_filename = save_examples_as_tfrecord(examples, index)
            
            upload_tfrecord(tfrecord_filename)
            
            cleanup_files(tfrecord_filename)

        except KeyboardInterrupt:
            raise

# deletes temporary file(s) in a directory
def cleanup_files(*args):
    for arg in args:
        if isinstance(arg, str):
            if os.path.exists(arg):
                os.remove(arg)
        elif isinstance(arg, list):
            for subarg in arg:
                if os.path.exists(subarg):
                    os.remove(subarg)

# uploads the tfrecord file 
def upload_tfrecord(filename):
    dest_filekey = dest_dir + os.path.basename(filename) # remove the tfrecord's current path from the S3 filekey
    bucket.upload_file(filename, dest_filekey)

    print('uploaded', filename, 'to', dest_filekey)

# save a list of tfrecord examples to a single tfrecord file
def save_examples_as_tfrecord(examples, index):
    filename = tfrecord_temp_dir + str(index) + '.tfrecord'

    writer = tf.python_io.TFRecordWriter(filename)

    for example in examples:
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()

    return filename

# create tfrecord example from image data
def create_example_from_jpeg_string(jpeg_string, shape, label):
    feature = {
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[jpeg_string])),
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=shape))
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))

# download image data from url
def load_image_data_from_url(url):
    # download image, save as rgb jpeg
    response = request.urlopen(url)
    image_data = response.read()
    pil_image = Image.open(BytesIO(image_data))
    pil_image_rgb = pil_image.convert('RGB')
    with BytesIO() as output:
        pil_image_rgb.save(output, format="JPEG", quality=90)
        contents = output.getvalue()
        return contents, pil_image_rgb.size

# batch images together to save as tfrecords
def get_image_batches(input_path):
    image_metadata_list = parse_data(input_path)
    print("num images:", (len(image_metadata_list)))

    batches = list(batcher(tfrecord_batch_size, image_metadata_list))

    return batches

def batcher(n, iterable):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=None, *args)

def parse_data(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    image_metadata_list = [ line for line in csvreader ]
    return image_metadata_list[1:]  # Chop off header


# MAIN THREAD

# filesystem setup
if not os.path.exists(tfrecord_temp_dir):
    os.makedirs(tfrecord_temp_dir)

# S3 setup
bucket = boto3.resource('s3').Bucket(bucket_name)

# find a list of already-completed tfrecords, so we don't have to redo those
tfrecord_filekeys = [ obj.key for obj in list(bucket.objects.filter(Prefix=dest_dir).all()) ]
tfrecord_indices_processed = [ int(tfrecord_filekey.split('/')[1].split('.')[0]) for tfrecord_filekey in tfrecord_filekeys ]
print('found', len(tfrecord_indices_processed), 'pre-existing tfrecord files')

# get batches
tfrecord_batches = get_image_batches(input_path)
print(len(tfrecord_batches), 'batches to generate tfrecords for')

# add indices for naming the tfrecord files
tfrecord_batches = list(enumerate(tfrecord_batches, 1))

pool = multiprocessing.Pool(processes=n_processes)
pool.map(create_and_upload_tfrecord_from_batch, tfrecord_batches)
pool.close()
pool.terminate()



# bonus code: reading the image data from the tfrecord
def read_tfrecord_images(filename):
    record_iterator = tf.python_io.tf_record_iterator(path=filename)
    
    for record_string in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(record_string)

        img_byte_string = example.features.feature['image'].bytes_list.value[0]
        shape = example.features.feature['shape'].int64_list.value    

        tempBuff = BytesIO()
        tempBuff.write(img_byte_string)
        tempBuff.seek(0)

        image_data = np.array(Image.open(tempBuff).convert("RGB"))