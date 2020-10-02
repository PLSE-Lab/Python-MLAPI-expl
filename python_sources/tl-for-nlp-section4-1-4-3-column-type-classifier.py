#!/usr/bin/env python
# coding: utf-8

# # Preliminaries
# Write requirements to file, anytime you run it, in case you have to go back and recover dependencies.
# 
# Latest known such requirements are hosted for each notebook in the companion github repo, and can be pulled down and installed here if needed. Companion github repo is located at https://github.com/azunre/transfer-learning-for-nlp

# In[ ]:


get_ipython().system('pip freeze > kaggle_image_requirements.txt')


# # Obtain Sample Tabular Datasets

#  Let's first download a simple OpenML dataset of baseball stats...

# In[ ]:


get_ipython().system('wget https://www.openml.org/data/get_csv/3622/dataset_189_baseball.arff # simple OpenML dataset')


# Load and take a peek at the dataset

# In[ ]:


import pandas as pd
raw_baseball_data = pd.read_csv('dataset_189_baseball.arff', dtype=str) # .arff format is apprently mostly equivalent for .csv for our purposes (look up Weka if really curious)

# expand default pandas display options to make things more clearly visible when printed
pd.set_option('display.max_colwidth', 300)

print(raw_baseball_data.head())


# Let's also get another dataset, for reasons that will be very clear later. Without getting into too many details, we suffice it to mention here that this dataset will be used to expand our SIMOn classifier beyond the set of classes the pretrained model we will use was designed to detect.
# 
# This is a dataset of public library data in British Columbia, obtained from https://catalogue.data.gov.bc.ca/dataset/bc-public-libraries-statistics-2002-present and attached to this notebook, just to ensure it is available for readers of the book easily.
# 
# Let's display its path to make sure we know where it is.

# In[ ]:


get_ipython().system('ls ../input/20022018-bc-public-libraries-open-data-v182 # see BC public libdary data 2002-2018 path')


# Let's load this dataset and take a peek

# In[ ]:


raw_data = pd.read_csv('../input/20022018-bc-public-libraries-open-data-v182/2002-2018-bc-public-libraries-open-data-csv-v18.2.csv', dtype=str)
print(raw_data.head())


# Let's select a pair of columns to focus on (of type 'int' and 'percent')

# In[ ]:


COLUMNS = ["PCT_ELEC_IN_TOT_VOLS","TOT_AV_VOLS"] # lots of columns in this data set, let's just focus on these two
raw_library_data = raw_data[COLUMNS]
print(raw_library_data)


# # Preprocess Tabular Datasets

# Install SIMOn as a first step.

# In[ ]:


get_ipython().system('pip install git+https://github.com/algorine/simon')


# Make the required imports

# In[ ]:


from Simon import Simon # SIMOn model class 
from Simon.Encoder import Encoder # SIMOn data encoder class


# Now, let's download a basic pretrained SIMOn model

# In[ ]:


get_ipython().system('wget https://raw.githubusercontent.com/algorine/simon/master/Simon/scripts/pretrained_models/Base.pkl # pretrained SIMOn model configuration - Encoder, etc')
get_ipython().system('wget https://raw.githubusercontent.com/algorine/simon/master/Simon/scripts/pretrained_models/text-class.17-0.04.hdf5 # corresponding model weights')
get_ipython().system('ls')


# Next, we want to load the SIMOn model
# 
# First load configurations and encoder

# In[ ]:


checkpoint_dir = "" # model weight are at the current level
execution_config = "Base.pkl" # name of our pretrained model configuration that was downloaded

Classifier = Simon(encoder={}) # create text classifier instance for loading encoder from model configs
config = Classifier.load_config(execution_config, checkpoint_dir) # load model config
encoder = config['encoder'] # get encoder
checkpoint = config['checkpoint'] # get checkpoint name

print(checkpoint) # the name of the checkpoint is stored in config as well, here we double-check it to make sure we got the right one


# Next, we define some hyperparameters to model data

# In[ ]:


max_len = 20 # maximum length of each tabular cell
max_cells = encoder.cur_max_cells # 500, maximum number of cells in a column
print(max_cells)


# What are the pretrained categories?

# In[ ]:


Categories = encoder.categories
category_count = len(Categories) # number of handled categories
print(encoder.categories)


# # Encode Data as Numbers

# We "standardize" the data into the expected length, i.e.:
# 
# *truncate columns that are too long and replicate cells in shorter columns such that all columns are of length max_cells*
# 
# This is necessary because convolutional neural networks require a fixed length input
# 
# We also convert the data from a DataFrame into a Numpy array, and transpose it such that the first dimension corresponds to sample columns, as per convention
# 
# All these steps are accomplished within the function *encodeDataFrame*
# 
# 
# Handle baseball data first:

# In[ ]:


X_baseball = encoder.encodeDataFrame(raw_baseball_data) # encode data (standardization, transposition, conversion to Numpy array)

print(X_baseball[0].shape)
print(X_baseball.shape) # display shape of encoded data
print(X_baseball[0]) # display encoded first column


# Handle library data next:

# In[ ]:


X_library = encoder.encodeDataFrame(raw_library_data) # encode data (standardization, transposition, conversion to Numpy array)

print(X_library[0])
print(X_library[0].shape)
print(X_library.shape)


# Generate some representative simulated data for illustrative purposes. The downloaded pretrained model was first trained on data of this sort, before transferring to a smaller set of hand-labeled examples

# In[ ]:


from Simon.DataGenerator import DataGenerator # Simulated/Fake data generation utility (using the library Faker)

# define appropriate parameters for the simulated data
data_cols = 5 # number of columns to generate
data_count = 10 # number if cells/rows per column

try_reuse_data = False # don't reuse data
simulated_data, header = DataGenerator.gen_test_data((data_count, data_cols), try_reuse_data)

print("SIMULATED DATA:") # display results
print(simulated_data)
print("SIMULATED DATA HEADER:")
print(header)


# # Generate Model and Predict Tabular Data Column Types
# 
# Generate model, load weights into it, compile it...

# In[ ]:


model = Classifier.generate_model(max_len, max_cells, category_count) # generate model

Classifier.load_weights(checkpoint, None, model, checkpoint_dir) # load weights

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) # compile model


# Define decision threshold, display model architecture

# In[ ]:


p_threshold = 0.5 # probability threshold for deciding membership of class

model.summary() # display model architecture


# In[ ]:


# alternative visualizationm method
from keras.utils.vis_utils import plot_model
plot_model(model,to_file='model.png') # visualize model architecture (for display here)
plot_model(model,to_file='model.svg') # higher resolution version for later

from IPython.display import Image # display image in notebook
Image(retina=True, filename='model.png') 


# Make prediction for the OpenML baseball dataset

# In[ ]:


get_ipython().system('ls')


# In[ ]:


y = model.predict(X_baseball) # predict classes
result = encoder.reverse_label_encode(y,p_threshold) # reverse encode labels
print("Recall that the column headers were:") # display the output
print(list(raw_baseball_data))
print("The predicted classes and probabilities are respectively:")
print(result)


# Make a prediction on the BC library, and let's look at it

# In[ ]:


X = encoder.encodeDataFrame(raw_library_data) # encode data using original frame
y = model.predict(X) # predict classes

result = encoder.reverse_label_encode(y,p_threshold) # reverse encode labels

print("Recall that the column headers were:")
print(list(raw_library_data))
print("The predicted class/probability:")
print(result)


# Gets integer column right, but percent is recognized as "text". This is because that class was not trained for in the current model... 
# 
# Let's try fine-tuning the float category to detect percentages... 
# 
# The percent column here is rather large - ~1200 rows, which can be broken up into >50 columns if each one is length 20  

# In[ ]:


# let's recall what the raw library data columns look like
print(raw_library_data)
print(raw_library_data.shape)


# In[ ]:


# turn into two lists
percent_value_list = raw_library_data['PCT_ELEC_IN_TOT_VOLS'].values.tolist()
int_value_list = raw_library_data['TOT_AV_VOLS'].values.tolist()


# In[ ]:


# Break it up into individual sample columns of size 20 cells each
original_length = raw_data.shape[0] # original length, 1207
chunk_size = 20 # length of each newly generated column
header_list = list(range(2*original_length//chunk_size)) # list of indices of new columns
new_raw_data = pd.DataFrame(columns = header_list) # initialize new DataFrame to hold new data
for i in range(original_length//chunk_size): # populate new DataFrame
    new_raw_data[i] = percent_value_list[i:i+chunk_size] # percent
    new_raw_data[original_length//chunk_size+i] = int_value_list[i:i+chunk_size] # integer
print(new_raw_data.head())


# In[ ]:


# let's create a corresponding header for our training data
header = [("percent",),]*(original_length//chunk_size)
header.extend([("int",),]*(original_length//chunk_size))
print(header)


# In[ ]:


# new categories
print(encoder.categories)


# In[ ]:


import numpy as np

# but first grab last layer weights for initialization
print("WEIGHTS::")
old_weights = model.layers[8].get_weights()
print(old_weights[1]) # biases
print("Layer Name::")
print(model.layers[8].name) # name of last layer
print("SHAPE::")
print(model.layers[8].get_weights()[0].shape) # shape of weight matrix in last layer


# In[ ]:


# find old weight index for closest category - text
old_category_index = encoder.categories.index('text')

# update the encoder with new category list, sort it alphabetically, find index of new category
encoder.categories.append("percent")
encoder.categories.sort()
new_category_index = encoder.categories.index('percent')

print(encoder.categories)


# In[ ]:


# perform the initialization
# the most similar class is text, as our experiment above showed, so we use that as a proxy for % when initializing
new_weights = np.copy(old_weights) # important to perform the copy operation

# initialize new weights
new_weights[0] = np.insert(new_weights[0], new_category_index, old_weights[0][:,old_category_index], axis=1) # weights
new_weights[1] = np.insert(new_weights[1], new_category_index, 0) # biases

new_weights[0].shape


# In[ ]:


# rebuild model using new category_count in last layer
model = Classifier.generate_transfer_model(max_len, max_cells, category_count, category_count+1, checkpoint, checkpoint_dir) # generate model


# only the last layer should be trainable - this was already done by the function above, but we repeat for illustrative purposes
for layer in model.layers:
    layer.trainable = False
model.layers[-1].trainable = True

model.layers[8].set_weights(new_weights)


# In[ ]:


# compile model
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

# see updated model summary, pay attention to dimension of output layer
print(model.summary())


# In[ ]:


# encode data (standardization, transposition, conversion to Numpy array)
X = encoder.encodeDataFrame(new_raw_data)
print("Encoded data shape")
print(X.shape)

# encode labels
y = encoder.label_encode(header)

# visualize
print(y)


# In[ ]:


# Prepare data in the expected format -> 60/30/10 train/validation/test data split
data = Classifier.setup_test_sets(X, y)


# In[ ]:


import time

# Train
batch_size = 4
nb_epoch = 10
start = time.time()
history = Classifier.train_model(batch_size, checkpoint_dir, model, nb_epoch, data) # train model
end = time.time()
print("Time for training is %f sec"%(end-start))


# Visualize convergence

# In[ ]:


import matplotlib.pyplot as plt

df_history = pd.DataFrame(history.history)

fig,ax = plt.subplots()
plt.plot(range(df_history.shape[0]),df_history['val_acc'],'bs--',label='validation')
plt.plot(range(df_history.shape[0]),df_history['acc'],'r^--',label='training')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('SIMOn Percent Transfer Classification Training')
plt.legend(loc='best')
plt.grid()
plt.show()
# Save figures
fig.savefig('SIMOnConvergence.eps', format='eps')
fig.savefig('SIMOnConvergence.pdf', format='pdf')
fig.savefig('SIMOnConvergence.png', format='png')
fig.savefig('SIMOnConvergence.svg', format='svg')


# In[ ]:


# Test trained model on test set
y = model.predict(data.X_test) # predict classes
result = encoder.reverse_label_encode(y,p_threshold) # reverse encode labels
print("The predicted classes and probabilities are respectively:")
print(result)
print("True labels/probabilities, for comparision:")
print(encoder.reverse_label_encode(data.y_test,p_threshold))


# In[ ]:


X = encoder.encodeDataFrame(raw_library_data) # repeat test on the raw two columns, as a final check
y = model.predict(X) # predict classes

result = encoder.reverse_label_encode(y,p_threshold) # reverse encode labels

print("The predicted class/probability:")
print(result)


# In[ ]:


# Make figures downloadable to local system in interactive mode
from IPython.display import HTML
def create_download_link(title = "Download file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

create_download_link(filename='SIMOnConvergence.svg')

