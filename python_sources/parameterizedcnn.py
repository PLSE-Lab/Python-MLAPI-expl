## This Python script is for creating a very generic CNN model.##
## On Passing the required constructor parameter it will return the CNN model for readily used into other projects.##

from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model

import numpy as np 

### Parameters Info for method generate_model()###
## input_shape: tuple of 3-dimension -  (width, height, depth) of the image ##
## hyperparameters: It is a dict datatype with the values are per the format mentioned at line# 18 (default_parameters variable). 
##                  The size of list for each dict key-value corresponds to the number of CNN layer we need for our model##
## Classes: It is value of total number of classes to be classified using this CNN model ##
## Output : return the Model object
class ParameterizedCNN:
    default_parameters = {"filters":[16,32,64], "filter_size":[3,3,3], "pool_size":[2,2,2],"padding": ["same","same","same"], "drop_out":[0.3,0.4,0.5],"dense":256}
    @staticmethod
    def generate_model(input_shape, hyperparameters=default_parameters, classes=2):
        chanDim = -1
        input_layer = Input(shape=input_shape)
        x=input_layer
        
        loop_i=0
        # loop to build CNN layers for the model
        for f in hyperparameters["filters"]:
            fs = hyperparameters["filter_size"][loop_i]
            ps = hyperparameters["pool_size"][loop_i]
            png = hyperparameters["padding"][loop_i]
            drpot = hyperparameters["drop_out"][loop_i]
            x=Conv2D(f,(fs,fs),padding=png)(x)
            x=Activation("relu")(x)
            x=BatchNormalization(axis=chanDim)(x)
            x=Dropout(drpot)(x)
            x=MaxPooling2D(pool_size=(ps,ps))(x)
            loop_i+=1
        
        # Dense Layer 1
        x = Flatten()(x)
        x = Dense(hyperparameters["dense"])(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(hyperparameters["drop_out"][0])(x)
        
        x = Dense(classes)(x)
        result = Activation("softmax")(x)
        
        model = Model(input_layer,result, name="ParameterizedCNN")
        
        return model