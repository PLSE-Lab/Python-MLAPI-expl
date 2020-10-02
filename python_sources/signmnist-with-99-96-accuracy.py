#!/usr/bin/env python
# coding: utf-8

# # SignMNIST

# ## DataSet Visualization

# So, we will start by visualizing the input data. Let's import all the required libraries and get started.
# I'll be using the following libraries to provide you some insights on this dataset:
# * ***Pandas*** to import the csv as a dataframe and perform the required operations.
# * ***Matplotlib*** to give some graphical data for better understanding.
# * ***Seaborn*** just to have a fancy look [XD XD].
# 

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Now, I'll read the dataset into a Pandas Dataframe and show you the information related to the data which contains things like **Total no. of rows**, **Total no. of cloumns**, **types of values**, etc..

# In[ ]:


df = pd.read_csv("../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")
df_test = pd.read_csv("../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")


# In[ ]:


df.head()


# In[ ]:


# You can see the Total number of rows and columns here
df.info()


# In[ ]:


df.describe()


# So, the SignMNIST dataset consists of 27455 rows and each row in turn has 785 columns.
# Out of the 785 columns, the first column is to identify the sign or we can say it contains all the *labels*.  
# 
# To have a better understanding, I'll show you exactly the number of examples each label contains. This can be done in the following manner: 
# 

# In[ ]:


count = df['label'].value_counts()
print(count)


# In[ ]:


# Also, just for verification
count.sum()


# Now, Using the Matplotlib Library of Python, you can see the graphical distribution of the given data for each label.
# 
# This can be done in the following manner:

# In[ ]:


fig1 = plt.figure(figsize=(5,3),dpi=100)
axes = fig1.add_axes([1,1,1,1])
axes.set_ylim([0, 1300])
axes.set_xlabel('classes')
axes.set_ylabel('No. of Examples available')
axes.set_xlim([0,24])
for i in range(24):
    axes.axvline(i)
axes.bar(count.index,count.values,color='purple',ls='--') 


# Clearly, the dataset has no data for the class label **9**. This is because in order to express the word 'J', you need to provide rotation to your hand whose data can't really be captured in 2-D arrays. Hence, the empty bar in the graph above.
# 
# Now, just to be a bit fancy, I'll plot the same thing via Seaborn too. Seaborn is another library of Python which is built on top of Matplotlib. It provides some fancy customizations (like palettes and all) and also is easier to use, hence, you can try your hands on that.

# In[ ]:


# I like the palette 'magma' too much. You can go with others like 'coolwarm','hsl' or 'husl'
sns.countplot(x=df['label'], palette='magma')


# **Have a look at few of the examples in the dataset.**

# In[ ]:


image_labels = df_test['label']
del df['label']
fig, axes = plt.subplots(3,4,figsize=(10,10),dpi=150)
k = 1
for i in range(3):
    for j in range(4):
        axes[i,j].imshow(df.values[k].reshape(28,28))
        axes[i,j].set_title("image "+str(k))
        k+=1
        plt.tight_layout()


# # Building the Model

# ## Importing Libraries

# First, I'll import all the necessary libraries I'm gonna need to get started.

# In[ ]:


import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd


# ## Importing the dataset

# I'm using the csv library as I used it when I was learning via Coursera. Also, You can use a combination of 'os', 'numpy' and 'pandas' to achieve the same as well. 

# In[ ]:


def get_data(filename):
    with open(filename) as training_file:
        csv_reader = csv.reader(training_file, delimiter=',')
        first_line = True
        temp_images = []
        temp_labels = []
        for row in csv_reader:
            if first_line:
                # print("Ignoring first line")
                first_line = False
            else:
                temp_labels.append(row[0])
                image_data = row[1:785]
                image_data_as_array = np.array_split(image_data, 28)
                temp_images.append(image_data_as_array)
        images = np.array(temp_images).astype('float')
        labels = np.array(temp_labels).astype('float')
        #print(labels)
    return images, labels

path_sign_mnist_train = f"{getcwd()}/../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)


# So, in short, what I've done is the following: 
# * Read the CSV file via the csv.reader() method provided by csv library.
# * Created to simple arrays which contain 1. Labels 2. Image Data
# * Appended the data from the CSV file in the respective arrays
# * Converted the arrays to numpy arrays and changed the datatype to float(from String)

# ## Applying Image Augmentation
# Simply, followed the most commonly used Image Augmentation techniques like:
# * height/width shit which in simple terms is Cropping the image
# * rescaling to normalize the data and centre is between 0 and 1
# * rotation of images by 10% and zooming the images by 20%
# 
# Normalization has been done for both Training as well as Validation Data.

# In[ ]:


training_images = np.expand_dims(training_images, axis=-1)
testing_images = np.expand_dims(testing_images, axis=-1)

# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.1,
                                   zoom_range=0.2)

validation_datagen = ImageDataGenerator(rescale=1.0/255)


# ## The Model

# This is how i made my model. 
# 
# You can freely experiment with it by increasing/decreasing layers, adding more features like Activity Regularization, Callbacks, Strides, etc etc..  
# I did not include Batch normalization as I didn't understand that concept very well. Surely after developing a better understanding, I'll include it. Or you guys can suggest me somethings anytime.
# 
# I mean, there's infinite scope of improvement in this discipline, no model can be termed perfect. 

# In[ ]:


# Define the model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
ankitz = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.75, min_lr=0.00001)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, (4, 4), activation='relu', input_shape=(28, 28, 1),padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',bias_regularizer=regularizers.l2(1e-4),),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same',bias_regularizer=regularizers.l2(1e-4)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(1,1), padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(384, activation=tf.nn.relu, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(25, activation=tf.nn.softmax,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.l2(1e-4))])
model.summary()


# ## Compiling the model

# So, I'm gonna use a batch size of 128 and Adam as the optimizer. In future, Ill surely improve by going much deeper, But as of now I'm also learning and so, do suggest the improvements in the comments.
# 
# **Some of the compiling definitions I used are: **
# * ADAM(Adaptive Moment) Optimizer
# * No. of epochs is 30
# * A callback has been used which will reduce the learning rate on plateaus by a factor of 0.75
# * loss used is 'sparse_categorical_crossentropy'.  
# (You can use 'categorical_crossentropy' as well but with a few changes in the above code.)

# In[ ]:


# Compile Model. 
model.compile(optimizer=tf.optimizers.Adam(),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Train the Model
history = model.fit_generator(train_datagen.flow(training_images, training_labels, batch_size=128),
                              epochs = 30,
                              validation_data=validation_datagen.flow(testing_images, testing_labels),
                             callbacks = [ankitz])

print("\n\nThe accuracy for the model is: "+str(model.evaluate(testing_images, testing_labels, verbose=1)[1]*100)+"%")


# I'll be honest here. The accuracy isn't fixed even for the identical process repeated twice. This is because your machine need not face the exact same complications every single time it runs. Sometimes, all goes well and you obtain the desired results. But, this truly is "random". So, the value above is one of the entire range of accuracy along with some(+-) error.  
# Also, the accuracy for the above model, when i tested it four time came out to be 99.9%, 99.2%, 99.7% and 99.9%. So, I'd say the accuracy will mostly come out to be greater than 99%.
# If not, as i said above, feel free to experiment..

# # Analysis of results

# So, first, we will start by simply comparing and visualizing the loss and accuracy over training and validation respectively.  
# 
# I'm doin this using the Matplotlib Library. The variable 'history' will provide you with all the necessary values to plot and analyze the results.

# In[ ]:


# Plot the chart for accuracy and loss on both training and validation
get_ipython().run_line_magic('matplotlib', 'inline')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# ### So, both the graphs came out pretty smooth I'd say. You can further make it better by having fun with the model.

# In[ ]:


predictions = model.predict_classes(testing_images)
print(predictions[:15])
print(image_labels.values[:15])


# ### Clearly, first 15 of our results have been correctly Classified. Below is the text report showing the main classification metrics.**

# In[ ]:


from sklearn.metrics import classification_report
classes_report = ["Class " + str(i) for i in range(25) if i != 9]
print(classification_report(image_labels, predictions, target_names = classes_report))


# # Thank You!!
# Thanks a lot guys. This is my **first** (basically my first activity XD) on Kaggle. I hope I was able to provide you with insights of the problem. If you have any constructive criticism, please let me know in comments. 
# Also, I don't know if there is any option to message other people on kaggle, if there is, feel free to ask any doubt and also to share your knowledge with me.
