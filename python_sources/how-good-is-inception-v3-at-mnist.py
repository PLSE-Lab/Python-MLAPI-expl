#!/usr/bin/env python
# coding: utf-8

# # Important Notes
# 
#  1.  The code below will **NOT** run on Kaggle.
# - If you are interested in downloading the source code, it is [available on GitHub here](https://github.com/scoliann/Kaggle-MNIST-Inception-CNN).
# 
# ## Inspiration
# 
# Inception is a massive CNN built by Google that can be retrained for any classification task.  I have successfully retrained Inception to tell the difference between images of men and women with 86.5% accuracy.  These results led me to wonder how useful Inception would be in the classic MNIST competition.  Could this model achieve results better than my previous score, 98.357%?
# 
# ## Previous Work
#   - [Gender Classification Project](https://github.com/scoliann/GenderClassifierCNN)
#   - [Previous MNIST Attempt](https://www.kaggle.com/scolianni/tensorflow-convolutional-neural-network-for-mnist)
# 
# ## About Inception
# 
# The Inception model is a CNN built by Google to compete in the ImageNet competition. Inception is therefore natively trained to classify input images into one of 1,000 categories used in ImageNet. Transfer Learning is a method by which one retrains part of a pre-existing machine learning model for a new purpose. In this case, we will be retraining the bottleneck layer of Inception. The "bottleneck" layer is the neural network layer before the final softmax layer in the CNN. Only one layer of the Inception model needs to be retrained, as the previous layers have already learned useful, generalizable functions such as edge detection.
# 
# ## Results
# 
# After experimenting with a number of hyperparameter settings for retraining Inception, I eventually reached a model that achieved 95.314% accuracy on the test set. This was not as high as I was hoping for, as my previous model achieved a 98.357% classification accuracy.
# 
# ## Possible Improvements
# 
# - Hyperparameter configurations that I did not have time to thoroughly test (eg. a smaller learning rate, etc.) may have resulted in better classification accuracy for the Inception model.
# - Retraining Inception is done with the retrain.py file in TensorFlow. This file does not include dropout or learning rate decay. Modifying the retrain.py file to include these additions (and/or others) would likely improve the classification accuracy.
# 
# # The Code

# ## Python Modules

# In[ ]:


'''
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from natsort import natsorted
import pandas as pd
'''


# ## Read in Test Set

# In[ ]:


'''
# Read in photos for each class and encode
testSetFolder = 'testSet'
testSetPhotos = [join(testSetFolder, f) for f in listdir(testSetFolder) if isfile(join(testSetFolder, f))]
sortedTestSetPhotos = natsorted(testSetPhotos)
encodedTestSetPhotos = [tf.gfile.FastGFile(photo, 'rb').read() for photo in sortedTestSetPhotos]
X = encodedTestSetPhotos
'''


# ## Read in TensorFlow Graph File

# In[ ]:


'''
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	_ = tf.import_graph_def(graph_def, name='')
'''


# ## Generate Predictions on the Test Data

# In[ ]:


'''
# Make predictions
predictionList = []
with tf.Session() as sess:

	# Feed the image_data as input to the graph and get first prediction
	softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

	# Iterate over all images and make predictions
	imageCounter = 0
	for image_data in X:

		# Print image coutner to terminal
		imageCounter += 1
		print('On Image ' + str(imageCounter) + '/' + str(len(X)))

		# Make a prediction
		predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    
		# Sort to show labels of first prediction in order of confidence
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

		# Get the predicted class and add it to list of predictions
		prediction = label_lines[top_k[0]]
		predictionList.append(prediction)
'''


# ## Output Predictions to Submission File

# In[ ]:


'''
# Create Submission CSV file
results = pd.DataFrame({'ImageId': pd.Series(range(1, len(predictionList) + 1)), 'Label': pd.Series(predictionList)})
results.to_csv('results.csv', index=False)
'''

