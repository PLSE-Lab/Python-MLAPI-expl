
# ----------------------------------DISCRIMINATOR------------------------------------------

# here we create our discriminator model for the GAN
from keras.layers import LeakyReLU, Convolution2D, MaxPool2D, ZeroPadding2D, Flatten, Dense,  \
    Dropout, Input, BatchNormalization
from keras.models import Sequential
from keras.regularizers import l2
from keras import Model
from keras.optimizers import Adam


class Discriminator:
    def __init__(self):
        # defining hyperparamters

        self.image_shape = (224, 224, 3)
        self.conv2d_kernel_size_big = (5, 5)
        self.conv2d_kernel_size_small = (3, 3)
        self.strides = (1, 1)
        self.max_pool_size = (2, 2)
     
        self.conv2d_1_kernels = 16
        self.conv2d_2_kernels = 32
        self.conv2d_3_kernels = 64
        self.conv2d_4_kernels = 128
        self.conv2d_5_kernels = 256

        self.dropout_rate = 0.3
        self.dropout_rate_big = 0.7
        self.dense_1_units = 2048
        self.dense_2_units = 1

        self.final_activation = 'sigmoid'

    def createDiscriminator(self):
        print('Creating Discriminator...')

        model = Sequential()

        # ---------------
        model.add(Convolution2D(filters=self.conv2d_1_kernels, kernel_size=self.conv2d_kernel_size_big,
                                strides=self.strides, input_shape=self.image_shape ))

        model.add(LeakyReLU())    
        model.add(MaxPool2D(pool_size=self.max_pool_size))
        model.add(Dropout(rate=self.dropout_rate))
        print(model.output_shape)
        
        # ----------------
        model.add(Convolution2D(filters=self.conv2d_2_kernels, kernel_size=self.conv2d_kernel_size_small,
                                strides=self.strides))
         
        model.add(LeakyReLU())    
        model.add(MaxPool2D(pool_size=self.max_pool_size))
        model.add(Dropout(rate=self.dropout_rate))
        print(model.output_shape)

        # ----------------
        model.add(Convolution2D(filters=self.conv2d_3_kernels, kernel_size=self.conv2d_kernel_size_small,
                                strides=self.strides ))
                                
        model.add(LeakyReLU())    
        model.add(MaxPool2D(pool_size=self.max_pool_size))
        model.add(Dropout(rate=self.dropout_rate))
        print(model.output_shape)
        
        # --------------------
        model.add(Convolution2D(filters=self.conv2d_4_kernels, kernel_size=self.conv2d_kernel_size_small,
                                strides=self.strides ))
                                
                            
        model.add(LeakyReLU())    
        model.add(MaxPool2D(pool_size=self.max_pool_size))
        model.add(Dropout(rate=self.dropout_rate))
        print(model.output_shape)
        
        # ---------------------
        model.add(Convolution2D(filters=self.conv2d_5_kernels, kernel_size=self.conv2d_kernel_size_small,
                                strides=self.strides))
                                
        model.add(LeakyReLU())    
        model.add(MaxPool2D(pool_size=self.max_pool_size))
        # model.add(Dropout(rate=self.dropout_rate_big))
        print(model.output_shape)

        # ---------------------------
        model.add(Flatten())
        model.add(Dropout(rate=self.dropout_rate_big))
        print(model.output_shape)
        
        # ----------------------------
        model.add(Dense(units=self.dense_1_units))
        model.add(LeakyReLU())  
        # model.add(Dropout(rate=self.dropout_rate))
        print(model.output_shape)

        # ---------------
        model.add(Dense(units=self.dense_2_units, activation=self.final_activation))
        print(model.output_shape)

        print(model.summary())

        image = Input(shape=self.image_shape)
        prediction = model(image)
        return Model(image, prediction)

    
    
# ------------------------------------GENERATOR--------------------------------------------

# here we create our generator!
# Input : a 232x232x3 image
# Output : a 224x224x3 generated image

# I won't use deconvolution because input size is bigger, we will proceed with a normal CNN
# the basic architecture is
# Conv->Relu->BatchNorm->Dropout
# Dense->OutputImage

# won't be using a MaxPool layer too!

class Generator:
    def __init__(self):
        # defining hyperparamters
        self.input_shape = (232, 232, 3)
        self.output_shape = (224, 224, 3)
        self.kernel_size = (3, 3)
        self.strides = (1, 1)
        self.dropout_rate = 0.3

        self.conv2d_1_kernels = 128
        self.conv2d_2_kernels = 64
        self.conv2d_3_kernels = 32
        self.conv2d_4_kernels = 3

        self.dense_units = 224 * 224 * 3
        self.final_activation = 'sigmoid'

    def createGenerator(self):
        model = Sequential()

        # ----------------------
        model.add(Convolution2D(filters=self.conv2d_1_kernels, kernel_size=self.kernel_size, strides=self.strides,
                                input_shape=self.input_shape))
        print(model.output_shape)

        model.add(LeakyReLU())
        print(model.output_shape)

        model.add(BatchNormalization())
        print(model.output_shape)

        model.add(Dropout(rate=self.dropout_rate))
        print(model.output_shape)

        # ----------------------
        model.add(Convolution2D(filters=self.conv2d_2_kernels, kernel_size=self.kernel_size, strides=self.strides))
        print(model.output_shape)

        model.add(LeakyReLU())
        print(model.output_shape)

        model.add(BatchNormalization())
        print(model.output_shape)

        model.add(Dropout(rate=self.dropout_rate))
        print(model.output_shape)

        # ----------------------
        model.add(Convolution2D(filters=self.conv2d_3_kernels, kernel_size=self.kernel_size, strides=self.strides))
        print(model.output_shape)

        model.add(LeakyReLU())
        print(model.output_shape)

        model.add(BatchNormalization())
        print(model.output_shape)

        model.add(Dropout(rate=self.dropout_rate))
        print(model.output_shape)

        # ----------------------
        model.add(Convolution2D(filters=self.conv2d_4_kernels, kernel_size=self.kernel_size, 
                                strides=self.strides,
                                activation = self.final_activation))
        print(model.output_shape)

        print(model.summary())

        image = Input(shape=self.input_shape)
        transformed = model(image)

        return Model(image, transformed)
        
        
        
# ----------------------------------GAN----------------------------------

from keras.optimizers import Adam
import cv2
import numpy as np

class GAN:
    def __init__(self):
        self.loss = 'binary_crossentropy'
        self.optimizer = Adam(0.0002, 0.5)
        self.input_image_shape = (232, 232, 3)
        self.output_image_shape = (224, 224, 3)

        d = Discriminator()
        self.discriminator = d.createDiscriminator()
        self.discriminator.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

        g = Generator()
        self.generator = g.createGenerator()
        self.generator.compile(optimizer=self.optimizer, loss=self.loss)

        input_image = Input(shape=self.input_image_shape)
        gen_img = self.generator(input_image)

        self.discriminator.trainable = False
        validity = self.discriminator(gen_img)

        self.combined = Model(input_image, validity)
        self.combined.compile(optimizer=self.optimizer, loss=self.loss)

    def train(self, dogs, cats, resized_dogs, ep):
        # first rescale the pixel values between 0 to 1
        dogs = dogs/255
        cats = cats/255
        resized_dogs = resized_dogs/255

        #  ------------------training the discriminator ---------------------

        y_dogs = [0] * dogs.shape[0]
        y_cats = [1] * cats.shape[0]

        # print('---------Training discriminator---------')
        d_loss_dogs = self.discriminator.train_on_batch(dogs, y_dogs)
        d_loss_cats = self.discriminator.train_on_batch(cats, y_cats)
        
        # print('-------Done!---------')

        d_loss = 0.5 * np.add(d_loss_dogs, d_loss_cats)

        # ----------------training the generator --------------------

        y_target = [1] * resized_dogs.shape[0]

        # print('----------Training generator----------')
        g_loss = self.combined.train_on_batch(resized_dogs, y_target)
        # print('----------Done!--------')

        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (ep, d_loss[0], 100 * d_loss[1], g_loss))

    def save_models(self):
        self.generator.save('generator.h5')
        self.discriminator.save('discriminator.h5')
        self.combined.save('combined.h5')

        
        
# if __name__ == '__main__':

#     batch_size = 32
#     size = 12500
#     epochs = 400
#     path = '../input/data/Data/'
#     adam = Adam(lr = 0.001)
    
#     d = Discriminator()
#     discriminator = d.createDiscriminator()
#     discriminator.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
#     for epoch in range(epochs):
#         idx = np.random.randint(0, size, batch_size)
    
#         dogs_data = []
#         cats_data = []
    
#         for i in idx.__iter__():
#             path_cat = path + 'cat.' + str(i) + '.jpg'
#             cat = (cv2.imread(path_cat) - 127.5)/127.5
#             cats_data.append(cat)

#             path_dog = path + 'dog.' + str(i) + '.jpg'
#             dog = (cv2.imread(path_dog)-127.5)/127.5
#             dogs_data.append(dog)

#         dogs_data = np.array(dogs_data)
#         cats_data = np.array(cats_data)
        
#         y_dogs = [1]* batch_size
#         y_cats = [0]* batch_size
        
#         X = np.concatenate((dogs_data,cats_data))
#         X = np.reshape(X, (X.shape[0], 224 * 224 * 3))
        
#         Y = np.concatenate((y_dogs,y_cats))
        
#         data = np.column_stack((X,Y))
        
#         np.random.shuffle(data)
        
#         X = data[:,:-1]
#         Y = data[:,-1:]
        
#         X = np.reshape(X,(X.shape[0],224,224,3))
        
#         print(discriminator.train_on_batch(X,Y))
        
        
#     exit(0)
        
        
    
if __name__ == '__main__':
    
    batch_size = 32
    size = 12500
    epochs = 6000
    gan = GAN()
    path = '../input/data/Data/'
    # exit(0)
    
    for epoch in range(epochs + 1):
        idx = np.random.randint(0, size, batch_size)
        idx2 = np.random.randint(0, size, batch_size)

        dogs_data = []
        cats_data = []
        resized_dogs_data = []

        for i in idx.__iter__():
            path_cat = path + 'cat.' + str(i) + '.jpg'
            cat = cv2.imread(path_cat)
            cats_data.append(cat)

            path_dog = path + 'dog.' + str(i) + '.jpg'
            dog = cv2.imread(path_dog)
            dogs_data.append(dog)

        for i in idx2.__iter__():
            path_dog = path + 'dog.' + str(i) + '.jpg'
            dog = cv2.imread(path_dog)
            resized_dog = cv2.resize(dog, (232, 232))
            resized_dogs_data.append(resized_dog)

        dogs_data = np.array(dogs_data)
        cats_data = np.array(cats_data)
        resized_dogs_data = np.array(resized_dogs_data)

        gan.train(dogs_data, cats_data, resized_dogs_data, epoch)

    gan.save_models()

   

