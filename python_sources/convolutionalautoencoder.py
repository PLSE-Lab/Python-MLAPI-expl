## This utility script is implmenting a static function to construct the AutoEncoder model for images.
# The first argument of "create" method will take a tuple of shape = (width, height, depth) of the image.
# By default the encoder will perform 2 layers of convolutional in this network, but, the same can be inhanced by appending to the "filters" argument of this method.
# the output of the encoder is low dimension space, controlled by parameter "ldim" of the"create" method. 
# to call this utility : use <<CNAutoEnc.create(input_shape=(32,32,3))>>

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K

import numpy as np

class CNAutoEnc:
    @staticmethod
    def create(input_shape, filters=(32,64), ldim=16):
        chanDim = -1
        
        #Encoder Module
        input_layer = Input(shape=input_shape)
        x= input_layer
        
        #filtersize (3,3)
        for f in filters:
            x=Conv2D(f, (3,3), strides=2, padding='same')(x)
            x=LeakyReLU(alpha=0.2)(x)
            x= BatchNormalization(axis=chanDim)(x)
        
        vsize = K.int_shape(x)
        x=Flatten()(x)
        latent= Dense(ldim)(x)
        encoder = Model(input_layer, latent, name="encoder")
        print(encoder.summary())
        
        #decoder module
        latent_input = Input(shape=(ldim,))
        x=Dense(np.prod(vsize[1:]))(latent_input)
        x=Reshape((vsize[1],vsize[2],vsize[3]))(x)
        
        #loop into filters to reconstruct the higher dim space
        for f in filters[::-1]:
            x=Conv2DTranspose(f,(3,3), strides=2, padding="same")(x)
            x=LeakyReLU(alpha=0.2)(x)
            x=BatchNormalization(axis=chanDim)(x)
        
        #apply the Conv2dTranspose layer again to get into the actual depth of the Image, as the current size is 32 according to above operations
        x=Conv2DTranspose(input_shape[2],(3,3), padding="same")(x)        
        #generate the output
        output=Activation("sigmoid")(x)
        #encapsulate Decoder
        decoder = Model(latent_input, output, name="decoder")
        print(decoder.summary())
        #encapsulate total Autoencoder
        autoencoder = Model(input_layer,decoder(encoder(input_layer)))
        print(autoencoder.summary())
        return (encoder, decoder, autoencoder)
            