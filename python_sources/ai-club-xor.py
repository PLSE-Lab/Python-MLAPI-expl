#!/usr/bin/env python
# coding: utf-8

# Hi!  Welcome to you first project!
# 
# This page may look a little confusing, but don't worry.  All you need to worry about right now is this little box of text :)
# 
# Kaggle works by letting you write code in the browser and then displaying the results of running that code in underneath it.  This makes it very useful for projects like this because you can open your project on any computer.
# 
# To start off, we are going to import some libraries.

# In[ ]:


#Numpy is a library that lets you create lists of numbers and manipulate them in all sorts of cool ways!
import numpy 

#Keras is the library that we will be using to make the ML models
import keras
print("This code just ran!")


# Great!  Now that we've imported keras, we need to run the code so that the program knows keras and numpy are included.
# Try clicking on the box with the code in it and pressing the blue play button to run the code.
# 
# Remember, whenever you see a code block, run it!

# OK, now that we have all of our libraries, it's time to get started with the machine learning program!
# 
# As you know, to train an ML program we need data to train it on.  Right now I've just gone ahead and created the data for you, but in the future you'll be finding datasets of you own to use!

# In[ ]:


#This is the data we will be using to train the model.
Xs = numpy.array([ #Input data
    [0, 0], 
    [0, 1], 
    [1, 0], 
    [1, 1]
])

Ys = numpy.array([ #Expected output data
    [0],
    [1],
    [1],
    [0]
])


# Once we have created the model, we will be training it on the dataset.  When we give the model an input, it will calculate what it thinks the output will be.  We can then see how correct it was and try to teach it to do better in the future.
# 
# (Just like taking a few thousand practice tests)

# Now it's time to make the ML model!
# 
# Firstly, we need to define what kind of model it is.  For this program we will be using something called a Sequential model.  These models are used for most machine learning projects and are made up of layers of neurons.
# 
# To start off, let's create a variable named model and set it to the keras.sequential object

# In[ ]:


model = keras.models.Sequential()


# Now that we have a model, we need to give it some layers!
# 
# Layers are how we work with ml models in keras.  As an example, adding a dense layer will create a bunch of neurons that are all connected to each other

# In[ ]:


#Time to add our first layer!  The input_dim tells the network how many inputs we'll be giving it.
#The units is the number of neurons in the layer 
#Activation tells the layer what activation function to use! (things like sin, cos, tanh, sigmoid, ect)
model.add(keras.layers.Dense(input_dim=2, units=4, activation="relu"))


# Great! Having one layer is fantastic, but if we want to be able to solve complex problems then we need more.
# 
# Let's add two more layers, just to make the program a little better.

# In[ ]:


model.add(keras.layers.Dense(units=4, activation="relu")) #We only need to specify number of inputs in the first layer
model.add(keras.layers.Dense(units=1, activation="sigmoid"))


# Great!  Now we've finished the model, right?
# 
# **WRONG**
# 
# We still need to tell the program that the model is finished and give the model it's activation and optimization functions!
# 
# To do this, we compile the model

# In[ ]:


model.compile(optimizer="sgd", loss="mean_squared_error")


# Now that we have our model, we can tell keras to show us a summary of it!

# In[ ]:


print(model.summary())


# So, now we have our model, let's see how good it is.
# 
# We'll use keras' predict function to see what the model thinks.

# In[ ]:


print(model.predict(Xs))


# That doesn't look too good!  It should be outputting 0, 1, 1, 0
#  
# Let's try training the model a little, to see if we can get in to learn anything!

# In[ ]:


#Train the model by calling the fit function (the epochs is the # of times the model will train on the data)
model.fit(Xs, Ys, epochs=1)
print(model.predict(Xs))


# Still not very good.  You should see that the loss value is around ~0.2, or quite bad.
# 
# Well, let's train it some more! (this part of the code can take a little while to run)

# In[ ]:


for i in range(10):
    model.fit(Xs, Ys, epochs=2000, verbose=0) #Setting the verbos to 0 means that keras won't tell us the loss.  That's ok!
    
    #Now get the model to tell us what it thinks!
    print(model.predict(Xs))
    print("================")


# Fantastic!  You've trained a Machine Learning Model!
