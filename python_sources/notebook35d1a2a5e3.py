#!/usr/bin/env python
# coding: utf-8

# # Starter's Pack for Invasive Species Detection
# 
# #### Hello everyone! This is Chris, the main researcher behind the dataset featured in this competition!
# 
# First of all, I we want to send a huge amount of thanks to Kaggle for making this nonprofit competition a real thing!
# 
# I am here to help everyone getting a head start in the competition! We are going to be building a pretty simple yet very powerful classificator based entirely on the blog entry ["Building powerful image classification models using very little data"](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) by F. Chollet.
# 
# We are going to make use of Keras (Theano as backend) and the VGG16 pretrained weights. We will build our own fully connected layers to place on top of VGG16 and later fine-tune the model for a few epochs (so we get some juicy extra AUC points in the leaderboard). The training is pretty light, so it should run in an average Intel I5 laptop's CPU overnight!
# 
# To keep the structure just like in the blog entry, this notebook is compiling together what in the blog are 2 different scripts. We are doing so sequentially, so it is easier to go to the source blog and get better understanding since the code will look the same there. First, we are going to predict features for our training and validation set for the last convolutional layers and save those to disk. Then we are going to use those predicted features to train our little fully connected model that will tell us whether there is or not an invasive species.
# 

# In[ ]:





# 

# In[ ]:


train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 500
nb_epoch = 50


# 

# In[ ]:





# 

# In[ ]:


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy','rb'))
    train_labels = np.array([0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array([0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.SGD(lr=1e-5, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              nb_epoch=nb_epoch, batch_size=32,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


# In[ ]:





# In[ ]:





# 

# In[ ]:





# 

# In[ ]:





# 

# In[ ]:




