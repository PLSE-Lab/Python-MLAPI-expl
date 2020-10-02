#!/usr/bin/env python
# coding: utf-8

# **Breast Cancer Detection**
# ![](https://blogs.nvidia.com/wp-content/uploads/2018/01/AI_Mammographie.jpg)

# ***Domain Background*** : 
# 	Breast Cancer is the most common type of cancer in woman worldwide accounting for 20% of all cases.
#     
# >     In 2012 it resulted in 1.68 million new cases and 522,000 deaths.
#     
# One of the major problems is that women often neglect the symptoms, which could cause more adverse effects on them thus lowering the survival chances. In developed countries, the survival rate is although high, but it is an area of concern in the developing countries where the 5-year survival rates are poor. In India, there are about one million cases every year and the five-year survival of stage IV breast cancer is about 10%. Therefore it is very important to detect the signs as early as possible. 
#     
# >     Invasive ductal carcinoma (IDC) is the most common form of breast cancer.
#    
#    About 80% of all breast cancers are invasive ductal carcinomas. Doctors often do the biopsy or a scan if they detect signs of IDC. The cost of testing for breast cancer sets one back with $5000, which is a very big amount for poor families and also manual identification of presence and extent of breast cancer by a pathologist is critical. Therefore automation of detection of breast cancer using Histopathology images could reduce cost and time as well as improve the accuracy of the test. This is an active research field lot of research papers and articles are present online one that I like is -(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5453426/) as they used deep learning approach to study on histology images and achieved the sensitivity of 95 which is greater than many pathologists (~90). This shows the power of automation and how it could help in the detection of breast cancer.
# 
# 

# **IMPORT FILES**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# In[ ]:


print(os.listdir("../input")) #to check the name of the directory inside which we have our files


# **Data Exploration**

# 
# In data exploration we will first check the name of the files.

# > **Code Conclusion :**  There are only png extentions which are present in alphabets therefore it means that we have only one image extention files with *.png* extentions. Therefore we will load only that.

# In[ ]:


from glob import glob
Data = glob('../input/IDC_regular_ps50_idx5/**/*.png', recursive=True)  #we extract only png files


# In[ ]:


Data


# In[ ]:


from PIL import Image
Image.open(Data[0]).size


# > **Code Conclusion **: We have total of 277524 image files

# Next Step is that we will check whether the dimentions of all the images are same or different

# In[ ]:



from PIL import Image #adds support for opening, manipulating, and saving many different image file formats
from tqdm import tqdm #adds progress bar for the loops
dimentions=list()
x=1
for images in tqdm(Data):
    dim = Image.open(images)
    size= dim.size
    if size not in dimentions:
        dimentions.append(size)
        x+=1
    if(x>3): #going through all the images will take up lot of memory, so therefore we will check until we get three different dimentions.
        break
print(dimentions)


# > ***Code Conclusion : *** We can see that the dimentions of images are not equal therefore we would make it all equal  to work bettter with our network.

# ***Data Extraction and Visualization***

# In[ ]:


import cv2 #used for computer vision tasks such as reading image from file, changing color channels etc
import matplotlib.pyplot as plt #for plotting various graph, images etc.
def view_images(image): #function to view an image
    image_cv = cv2.imread(image) #reads an image
    plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)); #displays an image
view_images(Data[18])


# > ***Code Conclusion :*** We can see that images are very small, though they are cropped images, its hard for human eye to understand them without using some high costly machines. 

# Now lets look at the color ranges that our images have

# In[ ]:


def hist_plot(image): #to plot histogram of pixel values present in an image VS intensities
    img = cv2.imread(image)
    plt.subplot(2, 2,1)
    view_images(image)
    plt.subplot(2, 2,2)
    plt.hist(img.ravel()) 
    plt.xlabel('Pixel Values')
    plt.ylabel('Intensity')
hist_plot(Data[169])
    


# > ***Code Conclusion :*** From the above image we can conclude that brighter region is more than the darken region in our image.  

# Next step is we need to extract the class names in which each files belong from its file names. We will save it in output.csv file.

# In[ ]:


from tqdm import tqdm
import csv #to open and write csv files
Data_output=list()
Data_output.append(["Classes"])
for file_name in tqdm(Data):
    Data_output.append([file_name[-10:-4]])
with open("output.csv", "w") as f:
    writer = csv.writer(f)
    for val in Data_output:
        writer.writerows([val])


# Below code reads the data from output.csv and displays it

# In[ ]:


from IPython.display import display # Allows the use of display() for DataFrames
data_output = pd.read_csv("output.csv")
display(data_output.head(5))
print(data_output.shape)


# > *Class1* represents** IDC(+)** and* Class0* represents** IDC(-)**

# In[ ]:


class1 = data_output[(data_output["Classes"]=="class1" )].shape[0]
class0 = data_output[(data_output["Classes"]=="class0" )].shape[0]
objects=["class1","class0"]
y_pos = np.arange(len(objects))
count=[class1,class0]
plt.bar(y_pos, count, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of images')
plt.title('Class distribution')
 
plt.show()


# > ***Code Conclusion :*** We can see that we have an unbalanced class and which is a common problem when we have medical data, therefore this is one another problem that we have to deal with later.

# In[ ]:


percent_class1=class1/len(Data)
percent_class0=class0/len(Data)
print("Total Class1 images :",class1)
print("Total Class0 images :",class0)
print("Percent of class 0 images : ", percent_class0*100)
print("Percent of class 1 images : ", percent_class1*100)


# > ***Data Processing  *** 

# We will first shuffle are images to remove any patterns if present and then load them.

# In[ ]:


from sklearn.utils import shuffle #to shuffle the data
Data,data_output= shuffle(Data,data_output)


# In[ ]:



from tqdm import tqdm
data=list()
for img in tqdm(Data):
    image_ar = cv2.imread(img)
    data.append(cv2.resize(image_ar,(50,50),interpolation=cv2.INTER_CUBIC))


# We would encode our output data which is present as Class1 and Class0 to 1 and 0.

# In[ ]:


data_output=data_output.replace(to_replace="class0",value=0)
data_output=data_output.replace(to_replace="class1",value=1)


# In the next step we will OneHot encode our data to better work with neural networks.

# In[ ]:


from keras.utils import to_categorical #to hot encode the output labels
data_output_encoded =to_categorical(data_output, num_classes=2)
print(data_output_encoded.shape)


# Now we will split our data into training set and testing set.

# In[ ]:


from sklearn.model_selection import train_test_split
data=np.array(data)
X_train, X_test, Y_train, Y_test = train_test_split(data, data_output_encoded, test_size=0.3)
print("Number of train files",len(X_train))
print("Number of test files",len(X_test))
print("Number of train_target files",len(Y_train))
print("Number of  test_target  files",len(Y_test))


# We have a large dataset and we will work with neural networks, therefore for better debugging we will use only a part of data, considering limited RAM and non GPU processor, this will not cost us much as we would also be using under sampling methods and image argumentation to deal with class imbalances and moderate data.

# In[ ]:


X_train=X_train[0:70000]
Y_train=Y_train[0:70000]
X_test=X_test[0:30000]
Y_test=Y_test[0:30000]


# We will now do undersampling, to treat our data for class imbalances. The Code inspiration for undersampling is taken from a notebook - https://www.kaggle.com/paultimothymooney/predict-idc-in-breast-cancer-histology-images

# In[ ]:


from keras.utils import to_categorical #to hot encode the data
from imblearn.under_sampling import RandomUnderSampler #For performing undersampling

X_train_shape = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
X_test_shape = X_test.shape[1]*X_test.shape[2]*X_test.shape[3]
X_train_Flat = X_train.reshape(X_train.shape[0], X_train_shape)
X_test_Flat = X_test.reshape(X_test.shape[0], X_test_shape)

random_US = RandomUnderSampler(ratio='auto') #Constructor of the class to perform undersampling
X_train_RUS, Y_train_RUS = random_US.fit_sample(X_train_Flat, Y_train) #resamples the dataset
X_test_RUS, Y_test_RUS = random_US.fit_sample(X_test_Flat, Y_test) #resamples the dataset
del(X_train_Flat,X_test_Flat)

class1=1
class0=0

for i in range(0,len(Y_train_RUS)): 
    if(Y_train_RUS[i]==1):
        class1+=1
for i in range(0,len(Y_train_RUS)): 
    if(Y_train_RUS[i]==0):
        class0+=1
#For Plotting the distribution of classes
classes=["class1","class0"]
y_pos = np.arange(len(classes))
count=[class1,class0]
plt.bar(y_pos, count, color = 'green', align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
plt.ylabel('Number of images')
plt.title('Class distribution')
 
plt.show()


#hot encoding them
Y_train_encoded = to_categorical(Y_train_RUS, num_classes = 2)
Y_test_encoded = to_categorical(Y_test_RUS, num_classes = 2)

del(Y_train_RUS,Y_test_RUS)

for i in range(len(X_train_RUS)):
    X_train_RUS_Reshaped = X_train_RUS.reshape(len(X_train_RUS),50,50,3)
del(X_train_RUS)

for i in range(len(X_test_RUS)):
    X_test_RUS_Reshaped = X_test_RUS.reshape(len(X_test_RUS),50,50,3)
del(X_test_RUS)


# We also need a validation set inorder to check overfitting. We can do two things either split test set further into valid set or split train se into valid set.

# We will go for spliting testing set into validation set.

# In[ ]:


X_test, X_valid, Y_test, Y_valid = train_test_split(X_test_RUS_Reshaped, Y_test_encoded, test_size=0.2,shuffle=True)


# In[ ]:


print("Number of train files",len(X_train_RUS_Reshaped))
print("Number of valid files",len(X_valid))
print("Number of train_target files",len(Y_train_encoded))
print("Number of  valid_target  files",len(Y_valid))
print("Number of test files",len(X_test))
print("Number of  test_target  files",len(Y_test))


# In[ ]:


from sklearn.utils import shuffle
X_train,Y_train= shuffle(X_train_RUS_Reshaped,Y_train_encoded)


# > We need to now preprocess our image file. We change pixels range from 0-255 to 0-1.

# In[ ]:


display(Y_train_encoded.shape)
display(Y_test.shape)
display(Y_valid.shape)


# In[ ]:


print("Training Data Shape:", X_train.shape)
print("Validation Data Shape:", X_valid.shape)
print("Testing Data Shape:", X_test.shape)
print("Training Label Data Shape:", Y_train.shape)
print("Validation Label Data Shape:", Y_valid.shape)
print("Testing Label Data Shape:", Y_test.shape)


# Now we have our three sets of train, valid and test. We will now create our benchmark model.

# In[ ]:


import itertools #create iterators for effective looping
#Plotting the confusion matrix for checking the accuracy of the model
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# # The training part starts from here on.

# > ***BENCHMARK MODEL: *** A simple CNN model

# > ***Now we will plot the confusion matrix :***

# ## Below is the model trained using data augmentation.

# 
# ***Image Argumentation***

# We will now add image argumentation to our data, so that it may be set for wider range of domain

# We will also rescale our image pixels, from range of 0-255.0 to 0-1.

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator  #For Image argumentaton
datagen = ImageDataGenerator(
        shear_range=0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        rescale=1/255.0,
        horizontal_flip=True,
        vertical_flip=True)


# In[ ]:


predictions_arg = [np.argmax(argum_model.predict(np.expand_dims(feature, axis=0))) for feature in tqdm(X_test_e)]


# > ***Now we will plot the confusion matrix :***

# In[ ]:


from sklearn.metrics import confusion_matrix
class_names=['IDC(-)','IDC(+)']
cnf_matrix_Arg=confusion_matrix(np.argmax(Y_test, axis=1), np.array(predictions_arg))
plot_confusion_matrix(cnf_matrix_Arg, classes=class_names)


# # XifengGuo

# In[ ]:


import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import initializers, layers


class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config


def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        inputs_expand = K.expand_dims(inputs, 1)

        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            c = tf.nn.softmax(b, dim=1)

            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]

            if i < self.routings - 1:
                b += K.batch_dot(outputs, inputs_hat, [2, 3])


        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
  
    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) +         0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


# In[ ]:


import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
# from utils import combine_images
from PIL import Image
# from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

K.set_image_data_format('channels_last')


x = layers.Input(shape=(256,256,3))
conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
digitcaps = CapsuleLayer(num_capsule=2, dim_capsule=16, routings=3, name='digitcaps')(primarycaps)
out_caps = Length(name='capsnet')(digitcaps)

eval_model = models.Model(x, out_caps)


# In[ ]:


eval_model.summary()


# In[ ]:


import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

K.set_image_data_format('channels_last')


# def CapsNet(input_shape, n_class, routings):
#     """
#     A Capsule Network on MNIST.
#     :param input_shape: data shape, 3d, [width, height, channels]
#     :param n_class: number of classes
#     :param routings: number of routing iterations
#     :return: Two Keras Models, the first one used for training, and the second one for evaluation.
#             `eval_model` can also be used for training.
#     """
#     x = layers.Input(shape=input_shape)

#     # Layer 1: Just a conventional Conv2D layer
#     conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

#     # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
#     primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

#     # Layer 3: Capsule layer. Routing algorithm works here.
#     digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
#                              name='digitcaps')(primarycaps)

#     # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
#     # If using tensorflow, this will not be necessary. :)
#     out_caps = Length(name='capsnet')(digitcaps)

#     # Decoder network.
#     y = layers.Input(shape=(n_class,))
#     masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
#     masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

#     # Shared Decoder model 
#     # Models for training and evaluation (prediction)
#     train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
#     eval_model = models.Model(x, [out_caps, decoder(masked)])

#     # manipulate model
#     noise = layers.Input(shape=(n_class, 16))
#     noised_digitcaps = layers.Add()([digitcaps, noise])
#     masked_noised_y = Mask()([noised_digitcaps, y])
#     manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
#     return train_model, eval_model, manipulate_model


# def margin_loss(y_true, y_pred):
#     """
#     Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
#     :param y_true: [None, n_classes]
#     :param y_pred: [None, num_capsule]
#     :return: a scalar loss value.
#     """
#     L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
#         0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

#     return K.mean(K.sum(L, 1))



# In[ ]:


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()


def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)





    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks
    model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(manipulate_model, (x_test, y_test), args)
        test(model=eval_model, data=(x_test, y_test), args=args)


# # The Capsule Network starts from here on.

# I have by default added the data augmentation. The augmentor for `CapsNet` is called `datagen_caps`.

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator  #For Image argumentaton
datagen_caps = ImageDataGenerator(
        shear_range=0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        rescale=1/255.0,
        horizontal_flip=True,
        vertical_flip=True)


# The following are the functions and parameters that are further required for construction of `Capsule Network`.

# In[ ]:


from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils

from keras.models import Model



# the squashing function.
# we use 0.5 in stead of 1 in hinton's paper.
# if 1, the norm of vector will be zoomed out.
# if 0.5, the norm will be zoomed in while original norm is less than 0.5
# and be zoomed out while original norm is greater than 0.5.
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x



# define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)



def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)



# ## Capsule Layer
# 
# ** This is the custom layer of capsule network. **

# ## The entire capsule network

# In[ ]:


class Capsule(layers.Layer):
    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


# In[ ]:


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


# In[ ]:


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Input, Reshape, Lambda
from keras.models import Sequential, Model


input_image = Input(shape=(50, 50, 3))
x = Conv2D(64, (3, 3), activation='relu')(input_image)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = AveragePooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = Conv2D(128, (3, 3), activation='relu')(x)


"""now we reshape it as (batch_size, input_num_capsule, input_dim_capsule)
then connect a Capsule layer.

the output of final model is the lengths of 10 Capsule, whose dim=16.

the length of Capsule is the proba,
so the problem becomes a 10 two-classification problem.
"""

# x = Reshape((-1, 128))(x)
primarycaps = PrimaryCap(x, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
capsule = Capsule(2, 16, 3, True)(primarycaps)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
model1 = Model(inputs=input_image, outputs=output)

# we use a margin loss
model1.compile(loss=margin_loss, optimizer='AdaDelta', metrics=['accuracy'])
model1.summary()


# In[ ]:


term = keras.callbacks.TerminateOnNaN()


# In[ ]:


# batch_size=64
batch_size=32 #comment this line and uncomment the above line if you want the batch size of 64
epochs=22
model1.fit_generator(datagen_caps.flow(X_train, Y_train, batch_size), 
          validation_data=(X_valid_e, Y_valid), steps_per_epoch=len(X_train) / batch_size, callbacks = [term],
          epochs=epochs, verbose=1)


# In[ ]:


from keras.utils import plot_model
plot_model(model1, to_file='model1.png', show_shapes = True, show_layer_names = True)


# ****<img src='model1.png'/>

# In[ ]:


from IPython.display import SVG
from keras.utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))


# In[ ]:


train_acc1 = model1.history.history['acc']
val_acc1 = model1.history.history['val_acc']
plt.plot(train_acc1)
plt.plot(val_acc1, color = 'green')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# In[ ]:


train_loss1 = model1.history.history['loss']
val_loss1 = model1.history.history['val_loss']
plt.plot(train_loss1)
plt.plot(val_loss1, color = 'green')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# ## More sophisticated model
# 
# With 2 capsule layers

# In[ ]:


import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations
from tensorflow.keras import utils
from tensorflow.keras.models import Model

# the squashing function.
# we use 0.5 in stead of 1 in hinton's paper.
# if 1, the norm of vector will be zoomed out.
# if 0.5, the norm will be zoomed in while original norm is less than 0.5
# and be zoomed out while original norm is greater than 0.5.
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x



# define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)



def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)




class Capsule(Layer):
    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


# In[ ]:


from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input, Reshape, Lambda, Activation, BatchNormalization
from tensorflow.keras.models import Sequential, Model

custom_model = Sequential()
# custom_model.add(256, (5, 5), input_shape=(50, 50, 3))
custom_model.add(Conv2D(256, (5, 5), input_shape=(50, 50, 3)))
custom_model.add(BatchNormalization())
custom_model.add(Conv2D(128, (3, 3), activation='relu'))
# custom_model.add(BatchNormalization())
custom_model.add(Reshape((-1, 128)))
custom_model.add(Capsule(64, 8, 3, True))
custom_model.add(Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2))))
custom_model.add(Dense(8, activation='softmax'))


# In[ ]:


# import tensorflow as tf

import keras.backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.models import Model

# the squashing function.
# we use 0.5 in stead of 1 in hinton's paper.
# if 1, the norm of vector will be zoomed out.
# if 0.5, the norm will be zoomed in while original norm is less than 0.5
# and be zoomed out while original norm is greater than 0.5.
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x



# define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)



def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)




class Capsule(Layer):
    """A Capsule Implement with Pure Keras
    There are two vesions of Capsule.
    One is like dense layer (for the fixed-shape input),
    and the other is like timedistributed dense (for various length input).

    The input shape of Capsule must be (batch_size,
                                        input_num_capsule,
                                        input_dim_capsule
                                       )
    and the output shape is (batch_size,
                             num_capsule,
                             dim_capsule
                            )

    Capsule Implement is from https://github.com/bojone/Capsule/
    Capsule Paper: https://arxiv.org/abs/1710.09829
    """

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        """Following the routing algorithm from Hinton's paper,
        but replace b = b + <u,v> with b = <u,v>.

        This change can improve the feature representation of Capsule.

        However, you can replace
            b = K.batch_dot(outputs, hat_inputs, [2, 3])
        with
            b += K.batch_dot(outputs, hat_inputs, [2, 3])
        to realize a standard routing.
        """

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


# In[ ]:


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Input, Reshape, Lambda, Activation, BatchNormalization
from keras.models import Sequential, Model


input_image = Input(shape=(50, 50, 3))
x = Conv2D(256, (5, 5), activation='relu')(input_image)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
# x = Conv2D(64, (3, 3), activation='relu')(x)
# x = AveragePooling2D((2, 2))(x)
# x = Conv2D(128, (3, 3), activation='relu')(x)
# x = Conv2D(128, (3, 3), activation='relu')(x)
# x = Reshape((-1, 128))(x)
primarycaps = PrimaryCap(x, dim_capsule=8, n_channels=64, kernel_size=9, strides=2, padding='valid')
# capsule = Capsule(64, 8, 3, True)(primarycaps)
capsule = Capsule(48, 16, 3, True)(primarycaps)
capsule = Capsule(32, 24, 3, False)(capsule)# This is more than one capsule layer
capsule = Capsule(16, 32, 3, False)(capsule)# This is more than one capsule layer
capsule = Capsule(2, 48, 3, False)(capsule)# This is more than one capsule layer
# output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
out_caps = Length(name='capsnet')(capsule)
model2 = Model(inputs=input_image, outputs=out_caps)



"""now we reshape it as (batch_size, input_num_capsule, input_dim_capsule)
then connect a Capsule layer.

the output of final model is the lengths of 10 Capsule, whose dim=16.

the length of Capsule is the proba,
so the problem becomes a 10 two-classification problem.
"""


# we use a margin loss
model2.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])
model2.summary()


# In[ ]:


import keras
term = keras.callbacks.TerminateOnNaN()

batch_size=32
epochs=22
# epochs=50 #In cases if you try and are not getting the satisfactory accuracy compared to other models then it s advisable to train
#it for longer period of time by increasing the number of epochs.
model2.fit_generator(datagen.flow(X_train, Y_train, batch_size), 
          validation_data=(X_valid, Y_valid), steps_per_epoch=len(X_train) / batch_size,callbacks = [term],
          epochs=epochs, verbose=1)


# In[ ]:


from keras.utils import plot_model
plot_model(model2, to_file='model2.png', show_shapes = True, show_layer_names = True)


# In[ ]:


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Input, Reshape, Lambda, Activation, BatchNormalization
from keras.models import Sequential, Model

custom_model = Sequential()
custom_model.add(Conv2D(64, (7,7), input_shape=(50, 50, 3)))
custom_model.add(Activation('elu'))
custom_model.add(BatchNormalization())
custom_model.add(Conv2D(64, (7,7)))
custom_model.add(Activation('elu'))
custom_model.add(BatchNormalization())
custom_model.add(MaxPooling2D(pool_size=(2,2)))
custom_model.add(Dropout(0.2))

custom_model.add(Conv2D(128, (5,5)))
custom_model.add(Activation('elu'))
custom_model.add(BatchNormalization())
custom_model.add(Conv2D(128, (5,5)))
custom_model.add(Activation('elu'))
custom_model.add(BatchNormalization())
custom_model.add(MaxPooling2D(pool_size=(2,2)))
custom_model.add(Dropout(0.3))
 
custom_model.add(Conv2D(256, (3,3)))
custom_model.add(Activation('elu'))
custom_model.add(BatchNormalization())
custom_model.add(Conv2D(256, (3,3)))
custom_model.add(Activation('elu'))
custom_model.add(BatchNormalization())
# custom_model.add(MaxPooling2D(pool_size=(2,2)))
custom_model.add(Dropout(0.4))
 
custom_model.add(Flatten())
custom_model.add(Dense(328, activation='relu'))
custom_model.add(Dense(128, activation='relu'))
custom_model.add(Dense(8, activation='softmax'))

custom_model.summary()


# In[ ]:


import keras
term = keras.callbacks.TerminateOnNaN()

batch_size=32
epochs=22
# epochs=50 #In cases if you try and are not getting the satisfactory accuracy compared to other models then it s advisable to train
#it for longer period of time by increasing the number of epochs.
custom_model.fit_generator(datagen.flow(X_train, Y_train, batch_size), 
          validation_data=(X_valid, Y_valid), steps_per_epoch=len(X_train) / batch_size,callbacks = [term],
          epochs=epochs, verbose=1)


# In[ ]:


from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
import tensorflow as tf
import tensorflow.keras.backend as K
weight_decay = 1e-4
_EPSILON = 10e-8

a = 0.2625
epochs = 175
iterations = 1
learning_rate = 0.0002


acc_list = list()
loss_list = list()
val_acc_list = list()
val_loss_list = list()


for i in range(iterations):
  

  def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

  _EPSILON = 10e-8
  def customentropy(target, output, from_logits=False):

    """Categorical crossentropy between an output tensor and a target tensor.
    # Arguments
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        target: A tensor of the same shape as `output`.
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    # scale preds so that the class probas of each sample sum to 1





    output /= tf.reduce_sum(output,
                            axis=len(output.get_shape()) - 1,
                            keepdims=True)
        # manual computation of crossentropy
    epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    return tf.reduce_sum(target * (tf.math.exp(1.0)-tf.math.exp(tf.math.pow(output, a))),
                               axis=len(output.get_shape()) - 1)
    
  datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
  datagen.fit(x_train)
 
  #training
  batch_size = 64
 
  opt_rms = tf.keras.optimizers.Adam(lr=learning_rate,decay=1e-6)
  custom_model.compile(loss=customentropy, optimizer=opt_rms, metrics=['accuracy'])

  custom_model.fit_generator(datagen.flow(x_train, y_train,
                                 batch_size=batch_size),
                                 steps_per_epoch=x_train.shape[0] // batch_size, 
                                 epochs=epochs,
                                 #verbose=1, 
                                 validation_data=(x_test, y_test),
                                 workers=1)  
  
  acc_list.append(custom_model.history.history['acc'])
  val_acc_list.append(custom_model.history.history['val_acc'])
  loss_list.append(custom_model.history.history['loss'])
  val_loss_list.append(custom_model.history.history['val_loss'])


# <img src='model2.png' />

# It was observed that the model, after certain epochs starts attaining the nan loss. Hence we applied the below call back in order to terminate as soon as it starts attaining nan loss.

# In[ ]:





# In[ ]:



import keras
term = keras.callbacks.TerminateOnNaN()

batch_size=32
epochs=22
# epochs=50 #In cases if you try and are not getting the satisfactory accuracy compared to other models then it s advisable to train
#it for longer period of time by increasing the number of epochs.
model2.fit_generator(datagen_caps.flow(X_train, Y_train, batch_size), 
          validation_data=(X_valid_e, Y_valid), steps_per_epoch=len(X_train) / batch_size,callbacks = [term],
          epochs=epochs, verbose=1)


# Through below 2 cells you can print the accuracy and loss of the model.

# In[ ]:


train_acc = model2.history.history['acc']
val_acc = model2.history.history['val_acc']

plt.plot(train_acc)
plt.plot(val_acc, color='green')
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# In[ ]:


train_loss = model2.history.history['loss']
val_loss = model2.history.history['val_loss']

plt.plot(train_loss)
plt.plot(val_loss, color='green')
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# In[ ]:


import numpy
print('The highest accuracy it achieved during the validation was ' , numpy.amax(numpy.array(model2.history.history['val_acc'])))


# Below two cells can plot the confusion matrix for the `model2` capsule network.

# In[ ]:


predictions_caps2 = [np.argmax(model2.predict(np.expand_dims(feature, axis=0))) for feature in tqdm(X_test_e)]


# In[ ]:


from sklearn.metrics import confusion_matrix
class_names=['IDC(-)','IDC(+)']
cnf_matrix_caps2=confusion_matrix(np.argmax(Y_test, axis=1), np.array(predictions_caps2))
plot_confusion_matrix(cnf_matrix_caps2, classes=class_names)


# In[ ]:


from sklearn.metrics import confusion_matrix
class_names=['IDC(-)','IDC(+)']
nf_matrix_caps2=confusion_matrix(np.argmax(Y_test, axis=1), np.array(predictions_transfer))
plot_confusion_matrix(cnf_matrix_transfer, classes=class_names)


# ## Comparision between the models 
# 
# ** Graph of all four models' validation accuracy and loss.**

# In[ ]:


train_acc_conv = argum_model.history.history['acc'][:20]
train_acc_caps = model2.history.history['acc']
train_acc_trans = model_transfer.history.history['acc']
# [1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 
plt.plot(train_acc_conv)
plt.plot(train_acc_caps, color='green')
plt.plot(train_acc_trans, color='red')


# plt.plot(train_acc1, color='red')
# plt.plot(val_acc1, color = 'orange')
plt.title('Training Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['ConvNet', 'CapsNet', 'VGG 19'], loc='upper left')
plt.show()


# In[ ]:


train_loss_conv = argum_model.history.history['loss'][:20]
train_loss_caps = model2.history.history['loss']
train_loss_trans = model_transfer.history.history['loss']
# [1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 
plt.plot(train_loss_conv)
plt.plot(train_loss_caps, color='green')
plt.plot(train_loss_trans, color='red')


# plt.plot(train_acc1, color='red')
# plt.plot(val_acc1, color = 'orange')
plt.title('Training Loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['ConvNet', 'CapsNet', 'VGG 19'], loc='upper left')
plt.show()


# In[ ]:


val_acc_conv = argum_model.history.history['val_acc']
val_acc_caps = model2.history.history['val_acc']
val_acc_trans = model_transfer.history.history['val_acc']

plt.plot(val_acc_conv)
plt.plot(val_acc_caps, color='green')
plt.plot(val_acc_trans, color='red')


# plt.plot(train_acc1, color='red')
# plt.plot(val_acc1, color = 'orange')
plt.title('Validation Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['ConvNet', 'CapsNet', 'VGG 19'], loc='upper left')
plt.show()


# In[ ]:


val_acc_simp = argum_model.history.history['val_loss']
val_acc_augm = model2.history.history['val_loss']
val_acc_caps = model_transfer.history.history['val_loss']

plt.plot(val_acc_simp)
plt.plot(val_acc_augm, color='green')
plt.plot(val_acc_caps, color='red')


# plt.plot(train_acc1, color='red')
# plt.plot(val_acc1, color = 'orange')
plt.title('Validation Loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['ConvNet', 'CapsNet', 'VGG 19'], loc='upper left')
plt.show()


# ### However, I am willing to add some more analysis and visualization in a sooner time.

# ## Sensitivity between the three models.

# In[ ]:


from sklearn.metrics import confusion_matrix
class_names=['IDC(-)','IDC(+)']
cnf_matrix_caps2=confusion_matrix(np.argmax(Y_test, axis=1), np.array(predictions_caps2))
plot_confusion_matrix(cnf_matrix_caps2, classes=class_names)


# In[ ]:


from sklearn.metrics import confusion_matrix
class_names=['IDC(-)','IDC(+)']
cnf_matrix_transfer=confusion_matrix(np.argmax(Y_test, axis=1), np.array(predictions_transfer))
plot_confusion_matrix(cnf_matrix_transfer, classes=class_names)


# In[ ]:


from sklearn.metrics import confusion_matrix
class_names=['IDC(-)','IDC(+)']
cnf_matrix_Arg=confusion_matrix(np.argmax(Y_test, axis=1), np.array(predictions_arg))
plot_confusion_matrix(cnf_matrix_Arg, classes=class_names)


# In[ ]:


# tp=0
# for i in range(0,len(Y_test)): #Number of positive cases
#     if(np.argmax(Y_test[i])==1):
#         tp+=1
#Senstivity of models
# confusion_bench_s=cnf_matrix_bench[1][1]/tp *100 
confusion_Arg_sens=cnf_matrix_Arg[0][0]/(cnf_matrix_Arg[0][0]+cnf_matrix_Arg[0][1])
cnf_matrix_caps2_sens=cnf_matrix_caps2[0][0]/(cnf_matrix_caps2[0][0]+cnf_matrix_caps2[0][1])
cnf_matrix_transfer_sens=cnf_matrix_transfer[0][0]/(cnf_matrix_transfer[0][0]+cnf_matrix_transfer[0][1])

classes=["ConvNet","CapsNet", "VGG19"]
objects=["ConvNet","CapsNet", "VGG19"]
y_pos = np.arange(len(classes))
count=[confusion_Arg_sens, cnf_matrix_caps2_sens, cnf_matrix_transfer_sens]
plt.bar(y_pos, count, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Percentage')
plt.title('Sensitivity')

plt.show()


# In[ ]:


cnf_matrix_Arg


# In[ ]:


cnf_matrix_Arg[0][0]+cnf_matrix_Arg[0][1]


# In[ ]:


cnf_matrix_caps2


# In[ ]:


[confusion_Arg_sens, cnf_matrix_caps2_sens, cnf_matrix_transfer_sens]


# ## Specificity between the three models.

# In[ ]:


# tp=0
# tn=0
# for i in range(0,len(Y_test)):  #Number of postive cases
#     if(np.argmax(Y_test[i])==1): 
#         tp+=1
# for i in range(0,len(Y_test)): #number of negative cases
#     if(np.argmax(Y_test[i])==0):
#         tn+=1
confusion_Arg_spec=cnf_matrix_Arg[1][1]/(cnf_matrix_Arg[1][0]+cnf_matrix_Arg[1][1])
cnf_matrix_caps2_spec=cnf_matrix_caps2[1][1]/(cnf_matrix_caps2[1][0]+cnf_matrix_caps2[1][1])
cnf_matrix_transfer_spec=cnf_matrix_transfer[1][1]/(cnf_matrix_transfer[1][1]+cnf_matrix_transfer[1][1])

classes=["ConvNet","CapsNet", "VGG19"]
objects=["ConvNet","CapsNet", "VGG19"]
y_pos = np.arange(len(classes))
count=[confusion_Arg_spec, cnf_matrix_caps2_spec, cnf_matrix_transfer_spec]
plt.bar(y_pos, count, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Percentage')
plt.title('Specificity')

plt.show()


# In[ ]:


[confusion_Arg_spec, cnf_matrix_caps2_spec, cnf_matrix_transfer_spec]


# ## Precision

# In[ ]:


confusion_Arg_prec=cnf_matrix_Arg[0][0]/(cnf_matrix_Arg[0][0]+cnf_matrix_Arg[1][0])
cnf_matrix_caps2_prec=cnf_matrix_caps2[0][0]/(cnf_matrix_caps2[0][0]+cnf_matrix_caps2[1][0])
cnf_matrix_transfer_prec=cnf_matrix_transfer[0][0]/(cnf_matrix_transfer[0][0]+cnf_matrix_transfer[1][0])

classes=["ConvNet","CapsNet", "VGG19"]
objects=["ConvNet","CapsNet", "VGG19"]
y_pos = np.arange(len(classes))
count=[confusion_Arg_prec, cnf_matrix_caps2_prec, cnf_matrix_transfer_prec]
plt.bar(y_pos, count, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Percentage')
plt.title('Precision')

plt.show()


# In[ ]:


[confusion_Arg_prec, cnf_matrix_caps2_prec, cnf_matrix_transfer_prec]


# ## F-measure

# In[ ]:


confusion_Arg_fmea=(2*confusion_Arg_prec*confusion_Arg_sens)/(confusion_Arg_prec+confusion_Arg_sens)
cnf_matrix_caps2_fmea=(2*cnf_matrix_caps2_prec*cnf_matrix_caps2_sens)/(cnf_matrix_caps2_prec+cnf_matrix_caps2_sens)
cnf_matrix_transfer_fmea=(2*cnf_matrix_transfer_prec*cnf_matrix_transfer_prec)/(cnf_matrix_transfer_prec+cnf_matrix_transfer_prec)

classes=["ConvNet","CapsNet", "VGG19"]
objects=["ConvNet","CapsNet", "VGG19"]
y_pos = np.arange(len(classes))
count=[confusion_Arg_fmea, cnf_matrix_caps2_fmea, cnf_matrix_transfer_fmea]
plt.bar(y_pos, count, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Percentage')
plt.title('F-measure')

plt.show()


# In[ ]:


[confusion_Arg_fmea, cnf_matrix_caps2_fmea, cnf_matrix_transfer_fmea]


# # End Remarks
# 
# It was observed that the CapsNet performed better than all of the models. 
# 
# It had performed well on the train data, validation data and the test data as well as seen by the confusion matrix.
# 
# Also, it trained in less no. of epochs and the remained stable for the rest of the epochs. While, somewhat unstable nature was shown by augmented model. 
# 
# And the transfer learning model's accuracy saturated at around 80%.
# 
# Further, I am willing to make this a more complex, hence better model that may consists of more than just two capsule layers.
# 
