#!/usr/bin/env python
# coding: utf-8

# # Bengali.AI - Starter kernel with ResNet
# 
# The following notebook can be used as a starter kernel for the [Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19) competition. The kernel includes steps for preprocessing the competition data, training separate ResNet models for each classification problem, and generating a submission file for the competition.
# 
# The Bengali.AI competition is a multi-label classification problem to identify handwritten characters (graphemes) from the Bengali language. Each example in the data is an image of a single, handwritten Bengali character. For each character, the objective of the competition is to classify the grapheme's root (a vowel, consonant, or conjuct) as well as the vowel diacritic (accent on vowel characters) and consonant diacritic (accent on consonant characters).
# 
# This kernel uses Google Cloud services for model training and data storage. The [ResNet container](https://aihub.cloud.google.com/u/0/p/products%2F4b08be38-7a6c-41b8-9d13-bfaa11cf199f) from AI Hub is used to train models with the [AI Platform Training](https://cloud.google.com/ai-platform/training/docs) service. In order to run this kernel, you'll need to create an account on Google Cloud Platform and enable the AI Platform and Cloud Storage APIs. Refer to the instructions in the [GCP set up](#gcp) section for more information.
# 
# The notebook includes the following sections:
#  - [GCP set up](#gcp)
#  - [Preprocess competition data](#preprocess)
#  - [Model training with AI Hub ResNet container](#training)
#  - [Generate predictions](#prediction)
#  - [Create submission file](#submission)
#  - [Inspect predictions](#inspect)
#  
# (Disclosure: I work with the team at GCP building these ML containers for AI Hub)

# ## GCP set up <a id='gcp'></a>
# 
# This tutorial uses billable components of Google Cloud Platform (GCP):
# 
# * Cloud AI Platform
# * Cloud Storage
# 
# Learn about [Cloud AI Platform
# pricing](https://cloud.google.com/ml-engine/docs/pricing) and [Cloud Storage
# pricing](https://cloud.google.com/storage/pricing), and use the [Pricing
# Calculator](https://cloud.google.com/products/calculator/)
# to generate a cost estimate based on your projected usage.
# 
# ### Set up your GCP project
# 
# 1. [Select or create a GCP project.](https://console.cloud.google.com/cloud-resource-manager). When you first create an account, you get a $300 free credit towards your compute/storage costs.
# 
# 2. [Make sure that billing is enabled for your project.](https://cloud.google.com/billing/docs/how-to/modify-project)
# 
# 3. [Enable the AI Platform APIs and Compute Engine APIs.](https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com,compute_component)
# 
# ### Authenticate your GCP account
# 
# To run this notebook in the Kaggle kernels environment, follow the steps below to authenticate your GCP account:
# 
# 1. [Create a GCP service account.](https://cloud.google.com/iam/docs/creating-managing-service-accounts#creating)
# 
# 2. [Create a service account key](https://cloud.google.com/iam/docs/creating-managing-service-account-keys#creating_service_account_keys)
# 
# 3. Authenticate your GCP service account by uploading the key and running the cell below.

# In[ ]:


import os

# Location of service account credentials for the Google Cloud SDK
GCLOUD_KEY = '/kaggle/input/YOUR-KEY.json'

# Authenticate with the Google Cloud SDK using a service account
if os.path.exists(GCLOUD_KEY):
    get_ipython().system('gcloud auth activate-service-account --key-file {GCLOUD_KEY}')


# ### Create a Cloud Storage bucket
# 
# Set the name of your Cloud Storage bucket below. It must be unique across all
# Cloud Storage buckets.
# 
# When defining the `REGION` variable, make sure to [choose a region where Cloud
# AI Platform services are
# available](https://cloud.google.com/ml-engine/docs/tensorflow/regions). You may
# not use a Multi-Regional Storage bucket for training with AI Platform.

# In[ ]:


BUCKET_NAME = "YOUR-BUCKET-NAME"
REGION = 'us-central1'


# If your bucket doesn't already exist, run the following cell to create your Cloud Storage bucket.

# In[ ]:


# !gsutil mb -l $REGION gs://$BUCKET_NAME


# ## Preprocess competition data <a id='preprocess'></a>
# 
# Convert the competition image data to [TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord). TFRecords is a serialized data format that helps to improve the efficiency of reading data during model training and prediction.

# In[ ]:


# Import dependencies
import pandas as pd
import tensorflow as tf  # TF version 2.1.0
import numpy as np
import os
import pyarrow.parquet as pq
import re
import sys
from datetime import datetime
import sklearn
from sklearn import metrics

# Set constant variables for location of GCS bucket and test data
GCS_BUCKET = 'gs://' + BUCKET_NAME
TEST_DATA = '/kaggle/input/bengaliai-cv19/test*.parquet'

# Set to True to run the data preprocessing cells in the notebook
PREPROCESS_DATA = False
# Set to True to run the ResNet model training cells in the notebook
RUN_TRAINING = False
# Set to True to inspect the train set accuracy from the generated predictions
# To generate predictions on the training data, set TEST_DATA='/kaggle/input/bengaliai-cv19/train*.parquet'
INSPECT_PREDICTIONS = False


# In[ ]:


# Import training and test labels
train_labels = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

# Remove the grapheme character (included for informational purposes only)
del train_labels['grapheme']

# Add 1 to all labels to reserve class=0 as background
# This step is required for training with the AI Hub ResNet container
train_labels.iloc[:, 1:4] = train_labels.iloc[:, 1:4] + 1


# In[ ]:


# Create empty array of labels for test set
test_labels = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')
test_labels = test_labels['image_id'].unique()
test_labels = pd.DataFrame(np.concatenate(
    (np.expand_dims(test_labels, 1),
     np.full(shape=(len(test_labels), 3), fill_value=-1)),
    axis=1))
test_labels.columns = ['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']


# In[ ]:


# Get training and test image files
train_image_files = tf.io.gfile.glob('/kaggle/input/bengaliai-cv19/train_image_data*')
test_image_files = tf.io.gfile.glob('/kaggle/input/bengaliai-cv19/test_image_data*')


# In[ ]:


# Import and reshape image data
def reshape_image(data, ids, labels):

    # Reshape to unflattened image size (137 by 236)
    data = tf.reshape(data, [137, 236])
    
    # Convert data to RGB format by duplicating the greyscale values across 3 channels
    data = tf.expand_dims(data, 2)
    data = tf.tile(data, [1, 1, 3])
    
    # Reshape labels and ids
    if labels is not None:
        labels = tf.squeeze(labels, 0)
    if ids is not None:
        ids = tf.squeeze(ids, 0)
    
    return data, ids, labels


# In[ ]:


# Convert images arrays to encoded JPG
def convert_to_encoded_jpg(data, ids, labels):
    data = tf.io.encode_jpeg(image=data, format='rgb', quality=100)
    return data, ids, labels


# In[ ]:


# Convert integer to TFRecord feature
def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# Convert string to TFRecord feature
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# Write a batch of images and labels to a TFRecord file
def write_tfrecord(filename, image_data, image_ids, image_labels, class_column):

    writer = tf.io.TFRecordWriter(filename + '.tfrecord')

    for i in range(len(image_data)):

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': _int64_feature(137),
            'image/width': _int64_feature(236),
            'image/colorspace': _bytes_feature(b'RGB'),
            'image/channels': _int64_feature(3),
            'image/class/label': _int64_feature(image_labels[i, class_column]),
            'image/id': _bytes_feature(image_ids.numpy()[i]),
            'image/format': _bytes_feature(b'JPEG'),
            'image/filename': _bytes_feature(b''),
            'image/encoded': _bytes_feature(image_data.numpy()[i])
        }))
        writer.write(example.SerializeToString())
    writer.close()


# In[ ]:


# Preprocess image data and convert to a set of TFRecord files
def convert_to_tfrecords(file, labels, filename_pattern, class_column, create_val_set=False):
    
    # Import parquet image file
    data = pq.read_table(file).to_pandas()
    data_ids = data['image_id']
    del data['image_id']

    labels = labels.loc[labels['image_id'].isin(data_ids.values)].iloc[:, 1:4].values
    labels = labels.astype(int)

    # Create tf.data.Dataset pipeline to preprocess images
    dataset = tf.data.Dataset.from_tensor_slices((data.values, data_ids.values, labels))
    dataset = dataset.batch(1)
    dataset = dataset.map(lambda data, ids, labels: reshape_image(data, ids, labels))
    dataset = dataset.map(lambda data, ids, labels: convert_to_encoded_jpg(data, ids, labels))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(10000)

    # Iterate through each batch of images, labels and create a TFRecord file
    j = 0
    for batch_images, batch_ids, batch_labels in dataset:
        filename = filename_pattern + '-' + str(j)
        if create_val_set and j == 0:
            filename = filename.replace('train', 'validation')
        write_tfrecord(
            filename=filename,
            image_data=batch_images,
            image_ids=batch_ids,
            image_labels=batch_labels,
            class_column=class_column)
        j += 1
        print('Wrote TFRecord file {}'.format(filename))


# In[ ]:


# Create local directories to store preprocessed image data
if PREPROCESS_DATA:
    if not os.path.exists('/kaggle/tfrecord'):
        os.mkdir('/kaggle/tfrecord')
        os.mkdir('/kaggle/tfrecord/grapheme_root')
        os.mkdir('/kaggle/tfrecord/vowel_diacritic')
        os.mkdir('/kaggle/tfrecord/consonant_diacritic')


# In[ ]:


# For each classification problem (grapheme root, vowel diacritic, and consonant diacritic) preprocess
# the image data and set the label to the appropriate value.
if PREPROCESS_DATA:
    for i in range(len(train_image_files)):

        # Preprocess data with the label set to the grapheme root
        convert_to_tfrecords(
            file=train_image_files[i],
            labels=train_labels,
            filename_pattern='/kaggle/tfrecord/grapheme_root/train-{}'.format(i),
            class_column=0,
            # Create validation set from batch of first 10K examples
            create_val_set=(True if i == 0 else False))
        
        # Preprocess data with the label set to the vowel diacritic
        convert_to_tfrecords(
            file=train_image_files[i],
            labels=train_labels,
            filename_pattern='/kaggle/tfrecord/vowel_diacritic/train-{}'.format(i),
            class_column=1,
            # Create validation set from batch of first 10K examples
            create_val_set=(True if i == 0 else False))

        # Preprocess data with the label set to the consonant diacritic
        convert_to_tfrecords(
            file=train_image_files[i],
            labels=train_labels,
            filename_pattern='/kaggle/tfrecord/consonant_diacritic/train-{}'.format(i),
            class_column=2,
            # Create validation set from batch of first 10K examples
            create_val_set=(True if i == 0 else False))
    


# In[ ]:


# Copy the preprocessed TFRecords data to Cloud Storage
if PREPROCESS_DATA:

    # Copy the preprocessed TFRecord files to a GCS bucket (internet must be enabled)
    get_ipython().system('gsutil -m cp -r /kaggle/tfrecord {GCS_BUCKET}/tfrecord')
    
    # Remove local copies of preprocessed data
    get_ipython().system('rm -r /kaggle/tfrecord')

    # Inspect the contents of the GCS bucket
    get_ipython().system('gsutil ls {GCS_BUCKET}/tfrecord')


# ## Model training with AI Hub ResNet container <a id='training'></a>
# 
# This kernel uses the [ResNet container](https://aihub.cloud.google.com/u/0/p/products%2F4b08be38-7a6c-41b8-9d13-bfaa11cf199f) from AI Hub to train 3 separate image classifiers. The ResNet container can be used with the AI Platform training service to run distributed training jobs with hardware accelerators. The component does not currently offer multi-label classification so it's necessary to train 3 separate models.

# In[ ]:


# AI Platform Training job params
REGION='us-central1'
SCALE_TIER='CUSTOM'
MASTER_MACHINE_TYPE='standard_v100'  # Machine that includes one NVIDIA Tesla V100 GPU

# Container registry location of the ResNet container
RESNET_CONTAINER = 'gcr.io/aihub-c2t-containers/kfp-components/oob_algorithm/resnet'                    '@sha256:ea935b6bbf83055afb4ccc30d42947673cb6aa43be20d4aa173509e684b2d3b8'


# In[ ]:


# Training for grapheme root classification problem

# Model params
DATA = GCS_BUCKET + '/tfrecord/grapheme_root'
GR_OUTPUT_LOCATION = GCS_BUCKET + '/resnet_output/grapheme_root'
NUMBER_OF_CLASSES=168
RESNET_DEPTH=200
TRAINING_STEPS=100000
BATCH_SIZE=8
LEARNING_RATE=0.1
MOMENTUM=0.9

# Training job params
JOB_NAME="resnet_kaggle_bengali_grapheme_root"

if RUN_TRAINING:
    # Submit training job to AI Platform Training
    get_ipython().system('gcloud ai-platform jobs submit training {JOB_NAME}         --master-image-uri {RESNET_CONTAINER}         --region {REGION}         --scale-tier {SCALE_TIER}         --master-machine-type {MASTER_MACHINE_TYPE}         --         --data {DATA}         --output-location {GR_OUTPUT_LOCATION}         --number-of-classes {NUMBER_OF_CLASSES}         --use-cache False         --resnet-depth {RESNET_DEPTH}         --training-steps {TRAINING_STEPS}         --batch-size {BATCH_SIZE}         --learning-rate {LEARNING_RATE}         --momentum {MOMENTUM}')


# In[ ]:


# Training for consonant diacritic classification problem

# Model params
DATA = GCS_BUCKET + '/tfrecord/consonant_diacritic'
CD_OUTPUT_LOCATION = GCS_BUCKET + '/resnet_output/consonant_diacritic'
NUMBER_OF_CLASSES=7
RESNET_DEPTH=200
TRAINING_STEPS=100000
BATCH_SIZE=8
LEARNING_RATE=0.1
MOMENTUM=0.9

# Training job params
JOB_NAME="resnet_kaggle_bengali_consonant_diacritic"

if RUN_TRAINING:
    # Submit training job to AI Platform Training
    get_ipython().system('gcloud ai-platform jobs submit training {JOB_NAME}         --master-image-uri {RESNET_CONTAINER}         --region {REGION}         --scale-tier {SCALE_TIER}         --master-machine-type {MASTER_MACHINE_TYPE}         --         --data {DATA}         --output-location {CD_OUTPUT_LOCATION}         --number-of-classes {NUMBER_OF_CLASSES}         --use-cache False         --resnet-depth {RESNET_DEPTH}         --training-steps {TRAINING_STEPS}         --batch-size {BATCH_SIZE}         --learning-rate {LEARNING_RATE}         --momentum {MOMENTUM}')


# In[ ]:


# Training for vowel diacritic classification problem

# Model params
DATA = GCS_BUCKET + '/tfrecord/vowel_diacritic'
VD_OUTPUT_LOCATION = GCS_BUCKET + '/resnet_output/vowel_diacritic'
NUMBER_OF_CLASSES=11
RESNET_DEPTH=200
TRAINING_STEPS=100000
BATCH_SIZE=8
LEARNING_RATE=0.1
MOMENTUM=0.9

# Training job params
JOB_NAME="resnet_kaggle_bengali_vowel_diacritic"

if RUN_TRAINING:
    # Submit training job to AI Platform Training
    get_ipython().system('gcloud ai-platform jobs submit training {JOB_NAME}         --master-image-uri {RESNET_CONTAINER}         --region {REGION}         --scale-tier {SCALE_TIER}         --master-machine-type {MASTER_MACHINE_TYPE}         --         --data {DATA}         --output-location {VD_OUTPUT_LOCATION}         --number-of-classes {NUMBER_OF_CLASSES}         --use-cache False         --resnet-depth {RESNET_DEPTH}         --training-steps {TRAINING_STEPS}         --batch-size {BATCH_SIZE}         --learning-rate {LEARNING_RATE}         --momentum {MOMENTUM}')


# After each training run a TensorFlow model folder (`saved_model.pb`, `variables`) is exported to the GCS bucket specified by the `--output-location` runtime argument. These exported files can be used to restore the state of the ResNet model and generate inferences.

# In[ ]:


# Copy all exported models into a single folder that can be uploaded to the Kaggle kernel
if RUN_TRAINING:
    # Find timestamped folder of exported model
    gr_saved_model_dir = get_ipython().getoutput('gsutil ls {GR_OUTPUT_LOCATION}/export')
    cd_saved_model_dir = get_ipython().getoutput('gsutil ls {CD_OUTPUT_LOCATION}/export')
    vd_saved_model_dir = get_ipython().getoutput('gsutil ls {VD_OUTPUT_LOCATION}/export')
    # Copy exported model folders to single GCS location
    get_ipython().system('gsutil cp -r {os.path.dirname(gr_saved_model_dir[-1])} {GCS_BUCKET}/assets/grapheme_root')
    get_ipython().system('gsutil cp -r {os.path.dirname(cd_saved_model_dir[-1])} {GCS_BUCKET}/assets/consonant_diacritic')
    get_ipython().system('gsutil cp -r {os.path.dirname(vd_saved_model_dir[-1])} {GCS_BUCKET}/assets/vowel_diacritic')


# ## Generate predictions <a id='prediction'></a>
# 
# To generate a submission file, we'll need to import the 3 trained ResNet models and generate inferences from each one. These models cannot be loaded into the same TensorFlow graph, since the variable names for the tensors and operations in each model are identical.
# 
# One way to work around this is to use the granular level controls for graphs and sessions provided with TensorFlow version 1. Another way to work around this while using the eager execution and saved_model functions from TF version 2, is to generate the predictions as separate processes.
# 
# Below I've written a Python script (`generate_predictions.py`) that loads one of the trained ResNet models, generates predictions for the test data, and writes these predictions to a local .CSV file. The module accepts the following command line arguments:
#  - `--data-dir`: File pattern for the test data.
#  - `--model-dir`: Local folder path to a trained ResNet model.
#  - `--prediction-batch-size`: Batch size for the prediction requests.
#  - `--prediction-file-name`: File name to save the predictions. Should end with .csv.
# 
# Run the code cells below to generate inferences from the trained ResNet models, and submit predictions to Kaggle.
# 
# ***Make sure that GPU is enabled in the Kaggle kernel before generating inferences***

# In[ ]:


prediction_script = """
# Import dependencies
import argparse
import sys
import pandas as pd
import tensorflow as tf  # TF version 2.0.1
import pyarrow.parquet as pq

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-dir',
    type=str,
    required=True,
    help='File pattern for the test data.')
parser.add_argument(
    '--model-dir',
    type=str,
    required=True,
    help='Local folder path to the TensorFlow SavedModel.')
parser.add_argument(
    '--prediction-batch-size',
    type=int,
    default=1024,
    required=False,
    help='Batch size for the prediction requests.')
parser.add_argument(
    '--prediction-file-name',
    type=str,
    required=True,
    help="File name to save the predictions. Should end with '.csv'.")


# Import trained model
def import_model(model_dir):
    model = tf.saved_model.load(export_dir=model_dir)
    predict_fn = model.signatures["serving_default"]
    return model, predict_fn


# Import test image file
def import_data(file):
    data = pq.read_table(file, buffer_size=1).to_pandas()
    data_ids = data['image_id'].values
    del data['image_id']
    # Create tf.data.Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((data.values, data_ids))
    return dataset


def reshape_data(data, ids):
    # Reshape to unflattened image size (137 by 236)
    data = tf.reshape(data, [tf.shape(data)[0], 137, 236])
    # Duplicate greyscale values across 3 channels for RGB format
    data = tf.expand_dims(data, 3)
    data = tf.tile(data, [1, 1, 1, 3])
    return data, ids


# Convert Numpy arrays to encoded JPEG images
def convert_images(data, ids):
    # Convert array to encoded JPEG
    data = tf.map_fn(
        lambda x: tf.io.encode_jpeg(image=x, format='rgb', quality=100),
        data,
        dtype=tf.string)
    return data, ids


# Get predictions for a batch of test data
def get_predicted_classes(predict_fn, images):
    predictions = predict_fn(images)
    return predictions['classes'].numpy().tolist()


# Iterate through the test data and generate predictions for each example
def generate_predictions(file_pattern, model_dir, prediction_batch_size):

    ids = []
    predictions = []

    image_files = tf.io.gfile.glob(file_pattern)
    _, predict_fn = import_model(model_dir)

    for i in range(len(image_files)):

        dataset = import_data(image_files[i])
        dataset = dataset.batch(prediction_batch_size)
        dataset = dataset.map(
            lambda data, ids: reshape_data(data, ids)
            )
        dataset = dataset.map(
            lambda data, ids: convert_images(data, ids)
            )

        print('Generating inferences for file: ' + str(image_files[i]))
        j = 0

        for batch_data, batch_ids in dataset:

            # If GPU device is available, assign the inference operations to
            # the GPU device
            try:
                with tf.device('/GPU:0'):
                    predicted_classes = get_predicted_classes(
                        predict_fn, batch_data)
            except:
                predicted_classes = get_predicted_classes(
                    predict_fn, batch_data)
            ids.extend(batch_ids.numpy().tolist())
            predictions.extend(predicted_classes)

            print('Generated inferences for batch: ' + str(j))
            j += 1

    try:
        ids = [id.decode('utf-8') for id in ids]
    except:
        pass

    return ids, predictions


# Write the predicted classes into a .CSV file
def write_predictions(ids, predictions, filename):
    pred_file = pd.DataFrame(ids)
    pred_file['predicted_class'] = predictions
    pred_file.columns = ['id', 'predicted_class']
    pred_file.to_csv(filename, index=False)


def main():

    # Parse command line arguments
    args, _ = parser.parse_known_args(sys.argv[1:])

    # Generate predictions
    print(args)
    ids, predictions = generate_predictions(
        file_pattern=args.data_dir,
        model_dir=args.model_dir,
        prediction_batch_size=args.prediction_batch_size)

    # Save predictions
    write_predictions(
        ids=ids,
        predictions=predictions,
        filename=args.prediction_file_name)


if __name__ == '__main__':
    main()

"""

if RUN_TRAINING:
    # Write prediction_script into Python module
    with open('/kaggle/generate_predictions.py', 'w') as f:
        f.write(prediction_script)

    # Copy prediction script to GCS bucket
    get_ipython().system('gsutil cp /kaggle/generate_predictions.py {GCS_BUCKET}/assets/generate_predictions.py')


# ## Upload model files and prediction script to Kaggle kernel
# 
# Once the models have been trained, the exported model files and `generate_predictions.py` script will have to be uploaded to the `/kaggle/input` folder in the Kaggle kernel. Uploading these assets to the `/kaggle/input` folder will make these assets accessible when the kernel is submitted.
# 
# 1. Download the assets locally: Run the following command on your local machine to download the trained ResNet models and the `generate_predictions.py` script to your local machine.
# 

# In[ ]:


print("gsutil -m cp -r {}/assets YOUR/LOCAL/PATH".format(GCS_BUCKET))


# 2. Use the Kaggle UI to upload this folder to the kernel: Navigate to the 'Edit' page for the kernel, click on the `+ Add Data` button, click `Upload`, then drag and drop the kernel assets folder and click `Create`.

# In[ ]:


# Replace 'assets' with the name of your Kaggle dataset folder
GRAPHEME_ROOT_MODEL = '/kaggle/input/assets/grapheme_root'
CONSONANT_DIACRITIC_MODEL = '/kaggle/input/assets/consonant_diacritic'
VOWEL_DIACRITIC_MODEL = '/kaggle/input/assets/vowel_diacritic'
PREDICTION_SCRIPT = '/kaggle/input/assets/generate_predictions.py'
PREDICTION_BATCH_SIZE = 1024


# In[ ]:


# Generate predictions for grapheme root
get_ipython().system('python3 {PREDICTION_SCRIPT}     --data-dir "{TEST_DATA}"     --model-dir {GRAPHEME_ROOT_MODEL}     --prediction-batch-size {PREDICTION_BATCH_SIZE}     --prediction-file-name grapheme_root_predictions.csv')


# In[ ]:


# Generate predictions for consonant diacritic
get_ipython().system('python3 {PREDICTION_SCRIPT}     --data-dir "{TEST_DATA}"     --model-dir {CONSONANT_DIACRITIC_MODEL}     --prediction-batch-size {PREDICTION_BATCH_SIZE}     --prediction-file-name consonant_diacritic_predictions.csv')


# In[ ]:


# Generate predictions for vowel diacritic
get_ipython().system('python3 {PREDICTION_SCRIPT}     --data-dir "{TEST_DATA}"     --model-dir {VOWEL_DIACRITIC_MODEL}     --prediction-batch-size {PREDICTION_BATCH_SIZE}     --prediction-file-name vowel_diacritic_predictions.csv')


# In[ ]:


# Import the predictions for each classification problem
gr_pred = pd.read_csv('grapheme_root_predictions.csv')
ids = gr_pred.id.values
grapheme_root_pred = gr_pred.predicted_class.values
del gr_pred

cd_pred = pd.read_csv('consonant_diacritic_predictions.csv')
assert np.array_equal(cd_pred.id.values, ids)
consonant_diacritic_pred = cd_pred.predicted_class.values
del cd_pred

vd_pred = pd.read_csv('vowel_diacritic_predictions.csv')
assert np.array_equal(vd_pred.id.values, ids)
vowel_diacritic_pred = vd_pred.predicted_class.values
del vd_pred


# In[ ]:


# Remove the local files storing the predictions
get_ipython().system('rm grapheme_root_predictions.csv')
get_ipython().system('rm consonant_diacritic_predictions.csv')
get_ipython().system('rm vowel_diacritic_predictions.csv')


# ## Create submission file <a id='submission'></a>

# In[ ]:


# Create dataframe with predictions and ids
submission = pd.DataFrame(columns =['row_id', 'target'])
submission = submission.append(pd.DataFrame(
    list(zip([id + '_consonant_diacritic' for id in ids], consonant_diacritic_pred)), 
    columns =['row_id', 'target']))
submission = submission.append(pd.DataFrame(
    list(zip([id + '_grapheme_root' for id in ids], grapheme_root_pred)), 
    columns =['row_id', 'target']))
submission = submission.append(pd.DataFrame(
    list(zip([id + '_vowel_diacritic' for id in ids], vowel_diacritic_pred)), 
    columns =['row_id', 'target']))

# Sort data by Id number
submission['row_id_number'] = [int(re.search('[0-9]+', row_id).group()) for row_id in submission['row_id'].tolist()]
submission = submission.sort_values(['row_id_number', 'row_id'])
del submission['row_id_number']

# Create submission.csv file
submission.to_csv('submission.csv', index=False)


# ## Inspect the predictions <a id='inspect'></a>
# 
# The following cells can be used to visually inspect the predictions and examine the training set error. To calculate the train set accuracy, set the `TEST_DATA` var to `/kaggle/input/bengaliai-cv19/train*.parquet`.

# In[ ]:


# Inspect submission file
if INSPECT_PREDICTIONS:
    with open('submission.csv', 'r') as f:
        i = 0
        for line in f:
            print(line)
            i += 1
            if i > 30:
                break


# In[ ]:


if INSPECT_PREDICTIONS:
    # Sanity check the predictions by calculating training set error for each classification problem
    train_labels = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')
    test_labels = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

    predictions = pd.DataFrame(ids)
    predictions['cd_pred'] = consonant_diacritic_pred
    predictions['gr_pred'] = grapheme_root_pred
    predictions['vd_pred'] = vowel_diacritic_pred
    predictions.columns = ['id', 'cd_pred', 'gr_pred', 'vd_pred']

    # Sort data by Id number
    predictions['id_number'] = [int(re.search('[0-9]+', id).group()) for id in predictions['id'].tolist()]
    predictions = predictions.sort_values(['id_number', 'id'])
    del predictions['id_number']

    print(predictions.head())


# In[ ]:


if INSPECT_PREDICTIONS and re.search('train', TEST_DATA):

    train_labels_subset = train_labels[train_labels.image_id.isin(predictions.id.tolist())]

    assert np.array_equal(predictions.id.values, train_labels_subset.image_id.values)

    train_labels_subset.head()


# In[ ]:


if INSPECT_PREDICTIONS and re.search('train', TEST_DATA):

    cd_accuracy = sklearn.metrics.accuracy_score(
        y_true=train_labels_subset.consonant_diacritic.values,
        y_pred=predictions.cd_pred.values)
    gr_accuracy = sklearn.metrics.accuracy_score(
        y_true=train_labels_subset.grapheme_root.values,
        y_pred=predictions.gr_pred.values)
    vd_accuracy = sklearn.metrics.accuracy_score(
        y_true=train_labels_subset.vowel_diacritic.values,
        y_pred=predictions.vd_pred.values)

    print('Consonant Diacritic accuracy = {}'.format(cd_accuracy))
    print('Grapheme Root accuracy = {}'.format(gr_accuracy))
    print('Vowel Diacritic accuracy = {}'.format(vd_accuracy))

