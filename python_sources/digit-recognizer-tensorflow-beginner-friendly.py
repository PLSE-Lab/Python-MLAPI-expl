#!/usr/bin/env python
# coding: utf-8

# <div style="border: thin solid red; padding-left: 15px; padding-right:15px; padding-top:15px; padding-bottom:15px"><h3><span style="color: #0000ff;"><strong>Hey Everyone</strong></span></h3>
# <p>This kernel is for anyone who is just getting started with Machine Learning using Python/Tensorflow and wants to get a basic idea about computer vision and how to implement it for Digit recognition.<br />I'll try to implement this kernel by dividing it into a couple of sections to make it easier to understand.</p>
# <p><span style="text-decoration: underline;"><strong>Sections</strong></span>:<br />&nbsp; &nbsp;1. Importing required libraries<br />&nbsp; &nbsp;2. Importing Image Data and re-arranging it.<br />&nbsp; &nbsp;3. Building Convolutional Neural Network using tensorflow.<br />&nbsp; &nbsp;4. Making Predictions on Test Data.<br />&nbsp; &nbsp;5. Exporting the results for submission in competition.</p></div>

# <h2>1. Importing required libraries</h2>

# In[ ]:


import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print(os.listdir('/kaggle/input/digit-recognizer'))


# <div style="border: thin solid red; padding-left: 15px; padding-right:15px; padding-top:15px; padding-bottom:15px">
# <h2>2. Importing Image Data and re-arranging it:</h2>
# For this competition the data is not given in form of images, rather we have a 'csv' file where each row represents one image.
# <br>Every row has 785 columns i.e 784 columns for pixel values and 1 column for label representing what number it is.
# <br><br><b><u>Note:</b></u> 784 pixels translates to 28x28 image.<br><br>First of all, we will import 'train.csv' using pandas and see how many rows and columns are there.<br>
# </div>

# In[ ]:


# Import 'train.csv' using pandas and see how many rows and columns are there.
all_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv', header=0)
print(all_data.shape)
all_data.head()


# <div style="border: thin solid red; padding-left: 15px; padding-right:15px; padding-top:15px; padding-bottom:15px">
# So, there are 42000 rows or images and every row has 785 columns.
# <br>Also, we can see the pixel values (0-255) for 784 pixel columns.
# <br><br>Now, we need to separate target variable from the data.
# <br>Here, Target variable = 'label' column which tells us what number is there in the image.
# </div>

# In[ ]:


# Assign 'label' column to target variable (y_all_data)
y_all_data = all_data['label']

# Remove 'label' column from rest of the data 
X_all_data = all_data.drop('label', axis=1)

# Convert dataframe into numpy array 
X_all_data = np.asarray(X_all_data)
y_all_data = np.asarray(y_all_data)

#Normalize Data
X_all_data = X_all_data/255

# No. of rows and columns per image
in_shape = X_all_data[0].shape

print(in_shape)


# <div style="border: thin solid red; padding-left: 15px; padding-right:15px; padding-top:15px; padding-bottom:15px">
# Now, we have separated target variable(label) from rest of the data and printed the shape of a single image.
# <br>The 'in_shape' variable will be useful to us when we need to feed this data into tensorflow neural network.
# 
# <br><br>Let's print one image from this data to see how it looks.
# </div>

# In[ ]:


# Re-shape image into 28x28 pixels
disp_image = X_all_data[1].reshape(28,28)

# Display image using matplotlib
plt.imshow(disp_image)
print("Label: ",y_all_data[1])


# <div style="border: thin solid red; padding-left: 15px; padding-right:15px; padding-top:15px; padding-bottom:15px">
# Looks like this particular image is for the number zero, which is correctly reflected in the label.
# </div>

# <div style="border: thin solid red; padding-left: 15px; padding-right:15px; padding-top:15px; padding-bottom:15px">
# <h2>3. Building Convolutional Neural Network using tensorflow</h2>
# <p>Let's start building neural network using tensorflow. <br /><br /> This neural network contains following 8 layers: <br />&nbsp; &nbsp;1. <span style="text-decoration: underline;">Reshape</span>: This layer converts 784 pixels into a 28x28 matrix which more accuarately represents an image. <br />&nbsp; &nbsp;2. <span style="text-decoration: underline;">Convolution</span>: In simple terms, a convolution layer is used to highlight "features" in an image. Convolutions can be used to highlight edges of objects within an image for better classification. Since this neural network utilizes convolutions, hence the name Convolutional Neural Network (CNN). <br />&nbsp; &nbsp;3. <span style="text-decoration: underline;">MaxPool</span>: This layer can be used to compress/reduce the size of image while retaining most of the information, which makes the training process faster. <br />&nbsp; &nbsp;4. <span style="text-decoration: underline;">Convolution</span>: Same as convolution layer above. <br />&nbsp; &nbsp;5. <span style="text-decoration: underline;">MaxPool</span>: Same as maxpool layer above. <br />&nbsp; &nbsp;6. <span style="text-decoration: underline;">Flatten</span>: This layer converts 2-D output from preceeding layers back into 1-D shape. <br />&nbsp; &nbsp;7. <span style="text-decoration: underline;">Dense</span>: This particular layer is a hidden layer in our neural net. <br />&nbsp; &nbsp;8. <span style="text-decoration: underline;">Dense</span>: Final/Output layer of the network. This layer contains 10 nodes/neurons representing 10 possible prediction values (0-9). <br><br>Once the structure of the model is defined, we need to compile it using parameters: optimizer, loss, metrics. <br /><br />Finally, we'll start training the model using 'model.fit()' and passing: <br />&nbsp; &nbsp; 1. <span style="text-decoration: underline;">Training data</span>: X_all_data &amp; y_all_data <br />&nbsp; &nbsp; 2. <span style="text-decoration: underline;">Epochs</span>: No. of times entire training data is passed to the neural network for training. <br />&nbsp; &nbsp; 3. <span style="text-decoration: underline;">Callbacks</span>: A function defined in top of the cell which defines that the model should stop training once desired accuracy is acheived.</p>
# </div>

# In[ ]:


# Function to stop training once desired accuracy is acheived
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.999):
            print('99.9% Accuracy reached')
            self.model.stop_training = False
            
mycallback = myCallback()  

# Start building model
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Reshape((28,28,1), input_shape=in_shape),
                                    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')
]
)

# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

#model.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=10, callbacks=[mycallback])

# Start Training Process
model.fit(X_all_data,y_all_data, epochs=50, callbacks=[mycallback])


# <div style="border: thin solid red; padding-left: 15px; padding-right:15px; padding-top:15px; padding-bottom:15px">
# <p>This model will be trained using input data for 50 epochs. <br />Once the training is complete, we can see that the training accuracy is ~0.99. <br /><br /><span style="text-decoration: underline;"><strong>Note</strong></span>: It is recommended to divide input data into training and validation sets. Training set is used to train the model and validation set can be used to test the accuracy of the model on "unseen" data. <br />This technique is useful to prevent overfitting where a model has good accuracy on training data but doesn't perform well on test data. <br /> For this notebook, I've used entire data for training to keep things easy to understand.</p>
# </div>

# <div style="border: thin solid red; padding-left: 15px; padding-right:15px; padding-top:15px; padding-bottom:15px">
# <h2>4. Making Predictions on Test Data</h2>
# <p>Once the neural network is trained, we can start making predictions on the test data provided in dataset. </p>
# <p>For this competition, the submission file must contain two columns: <br />&nbsp; &nbsp;1. <span style="text-decoration: underline;">ImageID</span>: This represents the serial no. of image in test dataset. <br />&nbsp; &nbsp;2. <span style="text-decoration: underline;">Label</span>: Prediction made by model for test image. <br /><br /> As we make predictions, we'll keep storing these two values in two separate arrays.</p>
# </div>

# In[ ]:


# Making Prediction
all_test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv', header=0)
final_test = np.asarray(all_test_data)
count=1
label = []
id = []
for test in final_test:
    #print(test.reshape(1,784))
    label.append(np.argmax(model.predict(test.reshape(1,784), batch_size=1000)))
    id.append(count)
    count += 1


# <div style="border: thin solid red; padding-left: 15px; padding-right:15px; padding-top:15px; padding-bottom:15px">
# <h2>5. Exporting the results for submission in competition</h2>
# Finally, we need to convert label and image id arrays into a format which can be submitted to the competition.
# <br> We'll store these two arrays in 'submission.csv' file using code given below.
# <br> You need to Commit the kernel by clicking blue "Commit" button on top right corner. It'll take a couple of minutes to run this entire notebook.
# <br><br> Once, the notebook is completely commited, you can see the 'submission.csv' file in the 'Output' section of the notebook.
# <br>Submit this file by either manually uploading the file in competition or by clicking "Submit to competition" button.
# </div>

# In[ ]:


#Save to csv

submission = pd.DataFrame()
submission['ImageId'] = id
submission['Label'] = label

submission.to_csv('submission.csv', index=False)


# <div style="border: thin solid red; padding-left: 15px; padding-right:15px; padding-top:15px; padding-bottom:15px">
# <h2>Keep Learning !</h2>
# </div>
