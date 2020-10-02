#!/usr/bin/env python
# coding: utf-8

# # KERAS - TRAINING WITH FLOAT16 - Test kernel 2
# 
# ## Introduction
# 
# Due to the big size of the images and the required detail for a good model to work in the Human Protein Atlas Image Classification Challenge, one of the possibilities of reducing the amount of memory needed is training with `float16` precision.    
# 
# In Keras, this should be done simply by setting `K.set_floatx('float16')`, however a few other things must be done to avoid `nan` values and to use the BatchNormalization layer using Tensorflow backend, which requires `float32` in all cases. (This Kernel was not tested for other backends and they may work differently)   
# 
# Also, after fixing the normalization layer, it will be necessary to fix the optimizer for conflicting types. 
# 
# ### Warning:
# 
# Although this kernel shows that it's possible to find a batch size that works in `float16` while not in `float32`, it seems slower and occupies more memory than [Test Kernel 1](https://www.kaggle.com/danmoller/keras-training-with-float16-test-kernel-1), which on its side doesn't seem to favor `float16`. 
# 
# But these tests are very fresh and if you find a bug or can solve the issue, please let us know :)   
# 
# 
# ## Differences between Test Kernel 1 and Test Kernel 2
# 
# In this kernel 2, we change the batch normalization layer to use `float16`. 
#   
#  - This skips using Tensorflow's fused batch normalization and uses a regular batch normalization    
#  - This also skips the need of changing the optimizer for different formats   
#  
#  In [Test Kernel 1](https://www.kaggle.com/danmoller/keras-training-with-float16-test-kernel-1), we use the original batch normalization operations from Keras (they are faster and occupy less memory)    
# 
# 
# ## Getting started - Setting to float16 and avoiding NaNs
# 
# In order to do this, simply run these commands before anything else.    
# The purpose of setting `epsilon` to a bigger value is because the default value is too little for `float16` and will cause `nan` loss values during training. 
# 
# **The epsilon value is not optimized, you may want to test it with 1e-3 or other values**

# In[ ]:


import keras.backend as K
K.set_floatx('float16')
K.set_epsilon(1e-4) #default is 1e-7

isTestMode = False #use this for training short epochs to quickly see the results
epochs = 2 
batchSize = 168


# ## Fixing the BatchNormalization layer
# 
# Because of Tensorflow's requirement of using `float32` in batch normalization, the setting above will break some things because Keras will send `float16` values to Tensorflow.
# 
# Thus, we will create custom weight initializers and a custom BatchNormalization layer:

# In[ ]:


from keras.layers import BatchNormalization, Layer
from keras.initializers import Initializer
from keras.backend.tensorflow_backend import tf, _regular_normalize_batch_in_training


#custom initializers to force float32
class Ones32(Initializer):
    def __call__(self, shape, dtype=None):
        return K.constant(1, shape=shape, dtype='float32')

class Zeros32(Initializer):
    def __call__(self, shape, dtype=None):
        return K.constant(0, shape=shape, dtype='float32')
    


class BatchNormalizationF16(Layer):

    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(BatchNormalizationF16, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = (
            initializers.get(moving_variance_initializer))
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                return tf.nn.batch_normalization(#K.batch_normalization(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    #axis=self.axis,
                    self.epsilon)#epsilon=self.epsilon)
            else:
                return tf.nn.batch_normalization(#K.batch_normalization(
                    inputs,
                    self.moving_mean,
                    self.moving_variance,
                    self.beta,
                    self.gamma,
                    #axis=self.axis,
                    self.epsilon)#epsilon=self.epsilon)

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return normalize_inference()

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = _regular_normalize_batch_in_training(#K.normalize_batch_in_training(
            inputs, self.gamma, self.beta, reduction_axes,
            epsilon=self.epsilon)

        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs)[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)],
                        inputs)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer':
                initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer':
                initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(BatchNormalizationF16, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


# ## Fixing the optimizers
# 
# In this test kernel 2, we won't need to fix the optimizers, since everything is `float16`, there will be no mixing in the optimizer and Keras will handle everything properly.

# In[ ]:


from keras.optimizers import SGD


# # Training
# 
# Now we're testing our changes in a simple model (this model is not really well thought for this competition, just an example). 
# 
# Lets create the model after a few definitions for loading data and organizing it in images with 4 channels (RGBY). Don't forget to convert your data to `float16`.

# In[ ]:


competitionFolder = '../input/' #human-protein-atlas-image-classification/'
trainFolder = competitionFolder + 'train/'
testFolder = competitionFolder + 'test/'
nClasses = 28
side=512
originalSide = 512
cropSide = 256


# In[ ]:


import numpy as np
from PIL import Image
import random

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

def loadClasses():
    trainFile = competitionFolder + 'train.csv'
    filesAndClasses = list()
    
    with open(trainFile, 'r') as f:
        _ = next(f)
        for row in f:
            fields = row.split(',')
            file = fields[0]
            
            classesNp = np.zeros((nClasses,), dtype='float16')
            classes = fields[1].split(' ')
            for c in classes: classesNp[int(c)] = 1
                
            filesAndClasses.append((trainFolder + file, classesNp))
    return filesAndClasses

def loadImage(file):
    colors = ['_red.png', '_green.png', '_blue.png', '_yellow.png']
    images = [Image.open(file + color) for color in colors]
    return np.stack(images, axis=-1)

#flips a batch of images, flipMode is an integer in range(8)
def flip(x, flipMode):
    if flipMode in [4,5,6,7]:
        x = np.swapaxes(x,1,2)
    if flipMode in [1,3,5,7]:
        x = np.flip(x,1)
    if flipMode in [2,3,6,7]:
        x = np.flip(x,2)
        
    return x

def inspect(x, name):
    print(name + ": ", 'shape:', x.shape, 'min:', x.min(), 'max:',x.max())
    
def plotChannels(img, minVal = 0, maxVal = 255):
    fig, ax = plt.subplots(1, img.shape[-1], figsize=(20,10))
    for i in range(img.shape[-1]):
        ax[i].imshow(img[:,:,i], vmin = minVal, vmax= maxVal)
        
    plt.show()
    
def competitionMetric(true,pred):
    pred = K.cast(K.greater(pred,0.5), K.floatx())
    
    groundPositives = K.sum(true, axis=0) + K.epsilon()
    correctPositives = K.sum(true * pred, axis=0)
    predictedPositives = K.sum(pred, axis=0) + K.epsilon()

    precision = correctPositives / predictedPositives
    recall = correctPositives / groundPositives

    m = (2 * precision * recall) / (precision + recall + K.epsilon())

    return K.mean(m)


# ## Data generator with cropping and flipping
# 
# In order to train even faster, we're creating a data generator that crops from the 512x512 images.   
# It was not studied if this crop may hide areas that contain the target proteins, but it's very probable that the protein be present in this crop if it's big enough.
# 
# Here we will be training with crops of size 256x256
# 
# 

# In[ ]:


from keras.utils import Sequence
from random import shuffle

#works with channels last
class ImageLoader(Sequence):
    
    #class creator, use generationMode = 'predict' for returning only images without labels
        #when using 'predict', pass only a list of files, not files and classes
    def __init__(self, filesAndClasses, batchSize, generationMode = 'train'):
        
        self.filesAndClasses = filesAndClasses
        self.batchSize = batchSize
        self.generationMode = generationMode
        
        assert generationMode in ['train', 'predict']
            

    #gets the number of batches this generator returns
    def __len__(self):
        l,rem = divmod(len(self.filesAndClasses), self.batchSize)
        return (l + (1 if rem > 0 else 0))
    
    #shuffles data on epoch end
    def on_epoch_end(self):
        if self.generationMode == 'train':
            shuffle(self.filesAndClasses)
        
    #gets a batch with index = i
    def __getitem__(self, i):
        
        #x are images   
        #y are labels
        
        pairs = self.filesAndClasses[i*self.batchSize:(i+1)*self.batchSize]
        if self.generationMode == 'train':
            files, classes = zip(*pairs) 
            y = np.stack(classes, axis=0)
            x = [loadImage(f) for f in files]
        elif self.generationMode == 'predict':
            files = pairs
            x = [loadImage(f) for f in files]
        else:
            raise Exception("ImageLoader does not support 'generationMode' of type " + self.generationMode)
    
        x = np.stack(x, axis=0)
        
        #cropping and flipping when training
        if self.generationMode == 'train':
            
            startH = random.randint(0,side - cropSide)
            startW = random.randint(0,side - cropSide)
            
            x = x[:, startW:startW + cropSide, startH:startH + cropSide]
            
            flipMode = random.randint(0,7) #see flip functoin defined above
            x = flip(x, flipMode)

        if self.generationMode == 'predict':
            return x
        else:
            return x, y
        


# ## Loading data and creating generator
# 
# Let's load the data an make a quick inspection of the generator. 

# In[ ]:


valBatchSize = batchSize // 4

trainFiles = loadClasses()

#creating a fold for validation
fold = 0
testLen = len(trainFiles)//5
testStart = fold * testLen
testEnd = testStart + testLen

valFiles = trainFiles[testStart:testEnd]
trainFiles = trainFiles[0:testStart] + trainFiles[testEnd:]

if isTestMode:
    valFiles = valFiles[:5*valBatchSize]
    trainFiles = trainFiles[:5*batchSize]

#creating train and val generators
trainGenerator = ImageLoader(trainFiles, batchSize)
valGenerator = ImageLoader(valFiles, valBatchSize)

#quick check
trainFileList, trainLabels = zip(*trainFiles)
predictGenerator = ImageLoader(trainFileList, batchSize, generationMode='predict')
for i in range(5):
    originalX = predictGenerator[i]
    x,y = trainGenerator[i]
    inspect(x,'images')
    inspect(y,'labels')
    print('unique y: ', np.unique(y))
    print('original')
    plotChannels(originalX[0])
    print('cropped')
    plotChannels(x[0])


# ## Model Creator
# 
# Here, a simple model using the custom layer and optimizer, similar to a ResNet. 
# 
# 

# In[ ]:


from keras.layers import *
from keras.models import Model

def modelCreator(convFilters, denseFilters):
                
    ######################################## definitions for layers ###########################3
    
    def denseBN(inputs, filters, activation, name):
        out = Dense(filters, name = name, use_bias=False)(inputs)
        out = BatchNormalizationF16(name = name + "BN")(out)
        out = Activation(activation, name = name + "ACT")(out)
        
        return out
    
    def convBN(inputs, filters, kernelSize, activation, name):
        out = Conv2D(filters, kernelSize, name=name, padding='same', use_bias=False)(inputs)
        out = BatchNormalizationF16(name = name + "BN")(out)
        out = Activation(activation, name = name + 'ACT')(out)
            
        return out
    
    ##################################### block definitions ####################################

    def downBlock(i, filters, inputs):
        
        name = str(i)
        out = inputs
        
        #make maxpooling and resnet connection if not first block
        if (i != 0):
            
            out = MaxPooling2D(poolSizes[i], name='Down' + name)(out)
            connection = convBN(out, filters, 3, activation = 'linear', name = 'resConnDownB' + name)


        out = convBN(out,filters, 3, activation='relu', name = 'downConvA' + name)
        out = convBN(out, filters, 3, activation='relu', name = 'downConvB' + name )
        out = convBN(out, filters, 3, activation='relu', name = 'downConvC' + name )
        
        #resnet connection
        if i != 0:
            out = convBN(out, filters, 3, activation = 'linear', name = 'resConnDownA' + name)
            out = Add(name='resAddDown' + name)([out,connection])
            out = BatchNormalizationF16(name = 'resNormDown' + name)(out)
            out = Activation('relu', name = 'downAct' + name)(out)
        
        return out
    
    ####################################### model creation #############################################
    
        
    poolSizes = [0,4,4,4,4]
    
    #notice we are training with 256x256 and validating with 512x512, thus None as size
    inp = Input((None,None,4))
    out = BatchNormalizationF16(name='initNorm')(inp)

    for i,filts in enumerate(convFilters):
        out = downBlock(i,filts,out)
    
    out = GlobalMaxPooling2D(name='globalPool')(out)
    
    for i, filts in enumerate(denseFilters):
        out = denseBN(out, filts, 'relu', name = 'dense' + str(i))
    
    out = denseBN(out, nClasses, activation='sigmoid', name="FinalDense")

    model = Model(inp,out)
                       
    return model


# ## Creating, compiling and fitting
# 
# Let's do it. 
# 
# **Warning:** the loss function selected may not be the best for this competition. 

# In[ ]:


model = modelCreator(convFilters =  [20,40,90,130,200],
                     denseFilters = [100,50,30])

#confirm dtype is float16
print("type is: ", K.dtype(model.get_layer('downConvA0').kernel))

#use a regular SGD
model.compile(optimizer = SGD(lr=0.01,momentum=.9), loss = 'categorical_crossentropy', metrics=[competitionMetric])

model.fit_generator(trainGenerator,len(trainGenerator), 
                    validation_data = valGenerator, validation_steps = len(valGenerator),
                   epochs = epochs, workers=5, max_queue_size=10)


# # Saving and Loading
# 
# In order to save and load the model using custom layers and optimizers, one needs to create a custom objects dictionary to tell Keras how to recreate these objects.
# 
# So we should include our layer, initializers, custom metric and optimizer in this object.

# In[ ]:


from keras.models import load_model

customObjects = {
    'BatchNormalizationF16': BatchNormalizationF16,
    'competitionMetric': competitionMetric,
    'Ones32': Ones32,
    'Zeros32': Zeros32
}

model.save('savedModel')
loadedModel = load_model('savedModel', customObjects)

#training starts from where it ended, including otimizer state
loadedModel.fit_generator(trainGenerator,len(trainGenerator), 
                    validation_data = valGenerator, validation_steps = len(valGenerator),
                   epochs = epochs, workers=5, max_queue_size=10)


# # Comparing with float32
# 
# Let's try to train the same generator with same batch size on a model with precision `float32` and see the GPU return an "out of memory (OOM)" error.
# 
# Even though, take a look at the warning at the beginning of this kernel and compare it with [Test Kernel 1](https://www.kaggle.com/danmoller/keras-training-with-float16-test-kernel-1)

# In[ ]:


dtype='float32'
K.set_floatx(dtype)


model = modelCreator(convFilters =  [20,40,90,130,200],
                     denseFilters = [100,50,30])

#confirm dtype is float32
print("type is: ", K.dtype(model.get_layer("downConvA0").kernel))

#use a regular SGD
model.compile(optimizer = SGD(lr=0.01,momentum=.9), loss = 'categorical_crossentropy', metrics=[competitionMetric])

model.fit_generator(trainGenerator,len(trainGenerator), 
                    validation_data = valGenerator, validation_steps = len(valGenerator),
                   epochs = epochs, workers=5, max_queue_size=10)

