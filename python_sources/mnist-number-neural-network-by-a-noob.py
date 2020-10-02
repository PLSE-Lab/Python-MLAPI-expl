#Author:   Michael Wheeler
#Project:  MNIST number neural netowrk
#Reason:   This is a basic neural network I made from a combination of googling and tutorials
#          so that I could learn a bit about neural networks and show it to a few of my friends.

#Sources that helped me:
# https://www.tensorflow.org/tutorials/keras/basic_classification

#Imports:  This is where external libraries are imported for my use.
#------------------------------------------------------------------------
import numpy as np #Used for my arrays
import csv         #improted because csv's are what we are reading
import matplotlib.pyplot as plt #This lets us display that first image!

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

#from IPython.display import Image, display
import os  #Importing our input directory I think? 
print(os.listdir("../input"))
#------------------------------------------------------------------------

#Functions:  This is where functions are defined so I may call them later
#------------------------------------------------------------------------
#View first image of given path
def viewOne(view_path):
    with open(view_path, 'r') as csv_file:
        for data in csv.reader(csv_file):
            # The first column is the label
            label = data[0]

            # The rest of columns are pixels
            pixels = data[1:]
    
            # Make those columns into a array of 8-bits pixels
            # This array will be of 1D with length 784
            # The pixel intensity values are integers from 0 to 255
            pixels = np.array(pixels, dtype='uint8')
    
            # Reshape the array into 28 x 28 array (2-dimensional array)  
            pixels = pixels.reshape((28, 28))
    
            # Plot
            plt.title('Label is {label}'.format(label=label))
            plt.imshow(pixels, cmap='gray')
            plt.show()
    
            break # This stops the loop, I just want to see one
    

#------------------------------------------------------------------------

#Main Script:  This is where the magic mostly happens
#------------------------------------------------------------------------

#Establish File Paths
train_path = "../input/mnist_train.csv"
test_path =  "../input/mnist_test.csv"

#View first image in train set
print("Here is an image of our first TRAIN number: ")
viewOne(train_path)

#Read train csv into a numpy array
reader = csv.reader(open(train_path), delimiter=",")   #reads our csv file in
temp = list(reader)                                    #turns this csv into a list
train_data = np.array(temp).astype("float")            #turns this lsit into a numpy array
train_x = train_data[0:,1:]                            #gets all but the first column of the array, this is 784 pixels each for 60K sample images
train_y = train_data[0:,0]                             #This is the number that those pixels resemble
train_x /= 255                                         #makes pixes whiteness on a scale of 0-1 instead of 0-255

#Read test csv into a numpy array, mostly same as previous segment
reader = csv.reader(open(test_path), delimiter=",")
temp = list(reader)
test_data = np.array(temp).astype("float") 
test_x = test_data[0:,1:]
test_y = test_data[0:,0]  
test_x /= 255  #makes pixes whiteness on a scale of 0-1 instead of 0-255

print(train_x.shape)  #Prints the structure of the data. In this case (#rows, #columns)
print(train_y.shape)  #Prints the structure of the data. In this case (#rows, #columns)

#Create Neural Network Model
model = keras.models.Sequential([
    keras.layers.Dense(784),                            #Input Layer: Takes in our input numbers, pushes them to neighboring nodes in next layer
                                                        #             as they travel to the next node the value is multiplied by the "weight" of the
                                                        #             edge connecting the input node to the following node.  So each following node
                                                        #             connected to each input node may receive different values.

    keras.layers.Dense(128, activation=tf.nn.relu),     #Middle Layer: Hidden lays can make room for more complex pattern recoginition.  Although I think they increase
                                                        #              the training requirements of our neural netowrk.   
                                                        
    keras.layers.Dense(10, activation=tf.nn.softmax)    #Output Layer:  Inputs from edges of last middle layer are then sent to last layer resulting in
                                                        #               probabilities to give us our outputs.  

#Compile Model so we may run it
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Train Model: By running train x values in our neural network.  The output probabilities of our network
#       are then compared to the real y values and tell our network how close it was to being correct.
#       A higher probability for the correct number means less "Loss".  Probabilities above 0 for incorrect
#       numbers mean more "Loss."   "Loss" will help us determine how much change our AI needs in "back propogation".
#       "back propogation" is the process which the AI corrects itself.
model.fit(train_x, train_y, epochs=5)

#Test Model: Very similar to fitting our model, the evaluation will run a test set of inputs "test_x" and compare
#            the probabilities it outputs to a set of answers "test_y".  However in this run it will be recording
#            it's accuracy.  I believe it will not make further modifications to it's system to account for "Loss"
#            either here.  In short I think this is just to test how accurate the current model is and it is no longer
#            "learning."
test_loss, test_acc = model.evaluate(test_x, test_y)

#Prints out Loss and Accuracy from the evaluation above, honestly no idea what loss in this context means yet
print("... \n")
print("-----------------------------------")
print("LOSS: ", test_loss)  
print("ACCURACY: ",  test_acc)
print("-----------------------------------\n")

predictions = model.predict(test_x)  #spits out the probabilities the model generates givent he pixels of an image.
                                     #This is very similar to the model.evaluate portion but it is not given the answers
                                     #to compare.  This is for when you have data you want it to estimate, but don't know the
                                     #answers.  Or you just want to see what it spits out since we do actually have the answers
                                     #for the test_x data set.  (test_y)
                                     
print("Our models probabilities for the first number: ", predictions[0], "\n")
print("The first TEST number was actually: ", test_y[0])
viewOne(test_path)
print("The probability our model output for 7 was: ", predictions[0][7])  #0 is lowest, 1 is highest

#------------------------------------------------------------------------
