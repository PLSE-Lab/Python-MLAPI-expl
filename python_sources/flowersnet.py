import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random
import os
import PIL

#!pip install tensorflow==1.13.1

import tensorflow as tf

print(tf.__version__)

#print(os.listdir("../input"))
#TODO: 
# - unfreese only some latest layers
# - try different applications
# - analyze errors on each class
# - clearify augmentation


IMAGES_PATH = '../input/train/train/'
LABELS_PATH = '../input/train_labels.csv'
TEST_IMAGES_PATH = '../input/test/test/'
SUMBISSION_FILE_PATH = 'Sumbission.csv'
REPRODUCIBLE = True
IMAGE_WIDTH = 299#224
IMAGE_HEIGHT = 299#224
COLOR_CHANNELS = 3
INPUT_IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
INPUT_IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, COLOR_CHANNELS)

RANDOM_STATE = 33 if REPRODUCIBLE else None


def unrandomize():
    
    os.environ['PYTHONHASHSEED'] = '0'
#    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # numpy
    np.random.seed(RANDOM_STATE)
    
    # core Python
    random.seed(RANDOM_STATE)
    
    # TensorFlow
    tf.set_random_seed(RANDOM_STATE)
    
    from keras import backend as K
    
    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/
#    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#    session = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#    K.set_session(session)
    
    print('unrandomized')

if (REPRODUCIBLE == True):
    unrandomize()
    
from keras import backend as K
from keras import layers, initializers, optimizers, regularizers
from keras import datasets, models, callbacks, applications, utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

def load_labels(path):
    labels = pd.read_csv(path)
    # add file name column
    labels['File'] = labels['Id'].apply(lambda x: f"{str(x).zfill(4)}.jpg")
    return labels
    
def get_pixel(image, i, j):
    # Inside image bounds?
    width, height = image.size
    if i > width or j > height:
      return None

    # Get Pixel
    pixel = image.getpixel((i, j))
    return pixel
    
def create_image(i, j):
  image = PIL.Image.new("RGB", (i, j), "white")
  return image    
    
# Create a Grayscale version of the image
def convert_grayscale(image):
  # Get size
  width, height = image.size

  # Create new Image and a Pixel Map
  new = create_image(width, height)
  pixels = new.load()

  # Transform to grayscale
  for i in range(width):
    for j in range(height):
      # Get Pixel
      pixel = get_pixel(image, i, j)

      # Get R, G, B values (This are int from 0 to 255)
      red =   pixel[0]
      green = pixel[1]
      blue =  pixel[2]

      # Transform to grayscale
      gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)

      # Set Pixel in new image
      pixels[i, j] = (int(gray), int(gray), int(gray))

    # Return new image
    return new    

def load_image(folder_path, filename, size):
    path = os.path.join(folder_path, filename)
    img = PIL.Image.open(path).resize(size)
    #img = convert_grayscale(img)
    return np.asarray(img)[:, :, :COLOR_CHANNELS]

# anomaly_ids = [4, 11, 32, 78, 79, 107, 167, 178,
# 200, 257, 261, 288, 374, 451, 484, 711, 725,
# 820, 843, 974, 994, 1037, 1217, 1269, 1271, 1365,
# 1368, 1378, 1427, 1587, 1610, 1659, 1759, 1871, 2119,
# 2198, 2203, 2308, 2366, 2453, 2460, 2461, 2520, 2525,
# 2543, 2461, 2520, 2525, 2552, 2606, 2635, 2666, 2764,
# 2922, 2990, 2977, 2937, 2651, 2912, 2720, 2765, 2821]  

anomaly_ids = [
    #none
    4, 6, 11, 30, 32, 78, 79, 107, 167, 178,
    200, 231, 257, 261, 288, 374, 435, 451, 462, 484, 561, 613, 711, 725, 750,
    820, 843, 940, 974, 994, 974, 1037, 1169, 1207, 1217, 1269, 1271, 1306, 1337, 1365,
    1368, 1378, 1427, 1437, 1468, 1587, 1610, 1659, 1685, 1722, 1759, 1830, 1834, 1859, 1871, 1893, 2011,  2119, 2135,
    2198, 2203, 2308, 2366, 2370, 2453, 2460, 2461, 2520, 2525,
    2543, 2461, 2520, 2525, 2552, 2606, 2616, 2621, 2635, 2666, 2708, 2733, 2753, 2764, 2829,
    2922, 2990, 2977, 2937, 2651, 2912, 2720, 2765, 2821, 2947, 2977, 2195,
    
    #bad
    #422, 478, 819, 971, 1408, 1587, 1833, 2016, 2404, 2449, 2539, 2573, 2696, 2824, 2953
     ]    

def load_images(images_path, filenames, img_size):
    images = filenames.apply(lambda filename: load_image(images_path, filename, img_size))
    return np.stack(images).astype(np.float32)#/255.0


labels_dataset = load_labels(LABELS_PATH)
#labels_dataset = labels_dataset.head(100)
labels_list = np.unique(labels_dataset['Category'].values)
index_to_label_map = dict(enumerate(labels_list))
label_to_index_map = {y:x for x,y in index_to_label_map.items()}

print(f"Full dataset: {labels_dataset.shape}")
labels_dataset = labels_dataset[~labels_dataset['Id'].isin(anomaly_ids)]
print(f"Dataset without anomalies: {labels_dataset.shape}")

x = load_images(IMAGES_PATH, labels_dataset['File'], INPUT_IMAGE_SIZE)

# convert labels to integers
labels_dataset['Category_Id'] = labels_dataset['Category'].apply(lambda label: label_to_index_map[label])

y = labels_dataset['Category_Id'].values

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.15, stratify=y, shuffle=True, random_state=RANDOM_STATE)

print(f"labels_dataset:\n{labels_dataset}")
print(f"labels_list:\n{labels_list}")
print(f"index_to_label_map:\n{index_to_label_map}")
print(f"label_to_index_map:\n{label_to_index_map}")

print(f"x: {x.shape}")

print(f"y({y.shape}):\n:{y}")

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")

print(f"X_val: {X_val.shape}")
print(f"y_val: {y_val.shape}")

#add black/white
#rescale=1./255,
AUG_BATCH_SIZE = 32
train_datagen = ImageDataGenerator(zoom_range=0.5, 
                                   rotation_range=45,
                                   width_shift_range=0.5, 
                                   height_shift_range=0.4, 
                                   shear_range=0.4, 
                                   horizontal_flip=True, 
                                   vertical_flip=True, 
                                   fill_mode='nearest',
                                   preprocessing_function=preprocess_input)
train_generator = train_datagen.flow(X_train, y_train, batch_size=AUG_BATCH_SIZE, seed=RANDOM_STATE, shuffle=True)  
#rescale=1./255
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_generator = val_datagen.flow(X_val, y_val, batch_size=AUG_BATCH_SIZE, seed=RANDOM_STATE, shuffle=True)


#pretrained model without top layer

def train_model(optimizer = 'adam', pooling_type = 'avg', base_epochs = 1, epochs=30):
    
#    base_model = applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='max')
    # base_model = applications.xception.Xception(
    #     include_top=False, 
    #     weights='imagenet', 
    #     input_shape=INPUT_IMAGE_SHAPE, 
    #     pooling=pooling_type, 
    #     classes=len(labels_list))
    
    # base_model = applications.ResNet50(
    #     include_top=False,
    #     weights='imagenet', 
    #     input_shape= INPUT_IMAGE_SHAPE,
    #     pooling=pooling_type)

    base_model = InceptionResNetV2(
        include_top=False,
        weights='imagenet', 
        input_shape= INPUT_IMAGE_SHAPE,
        pooling=pooling_type)
        
    #fix weights
    base_model.trainable = False

    #and our model become simple
    inp = layers.Input(INPUT_IMAGE_SHAPE)
    resnet = base_model(inp)

    fc = layers.Dense(len(labels_list))(resnet)
    fc = layers.Activation('softmax')(fc)

    model = models.Model(inp, fc)
    
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
        
    #train only dense layer on top
    model.fit_generator(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=len(y_train)//AUG_BATCH_SIZE,
        validation_steps=len(y_val)//AUG_BATCH_SIZE,
        epochs=base_epochs,
        callbacks=[
#            callbacks.ModelCheckpoint('weights.h5', verbose=1, save_best_only=True, save_weights_only=True),
            callbacks.ReduceLROnPlateau(patience=2, verbose=1),
            callbacks.EarlyStopping(patience=4, verbose=1)
          ],
          shuffle=True
        )
        
    #unfreeze all weights and train 
    base_model.trainable = True
    
    # freeze first 20% of layers
    # 33%: 0.19 loss
    # 20%: 0.26
    # 66%: 0.11 loss
    for layer in base_model.layers[len(base_model.layers)*8//10:]:
        layer.trainable = False
    
    model.compile(
        optimizer=optimizer, 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    history = model.fit_generator(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=len(y_train)//AUG_BATCH_SIZE,
        validation_steps=len(y_val)//AUG_BATCH_SIZE,
        epochs=epochs,
        callbacks=[
            callbacks.ModelCheckpoint('best-flower-weights.h5', verbose=1, save_best_only=True, save_weights_only=True),
            callbacks.ModelCheckpoint('best-flower-model.h5', verbose=1, save_best_only=True, save_weights_only=False),
            callbacks.ReduceLROnPlateau(patience=3, verbose=1),
            callbacks.EarlyStopping(patience=5, verbose=1)
              #tensorboard
          ],
          shuffle=True
          )        
          
    return model, history

from tensorflow.python.keras.callbacks import TensorBoard        
    
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
#from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args

#dim_batch_size = Integer(low=1, high=16, name='batch_size')
#dim_optimizer = Categorical(categories=['Adam', 'Nadam', 'RMSprop'], name='optimizer')
#dim_pooling_type = Categorical(categories=['max', 'avg'], name='pooling_type')
#dim_base_epochs = Integer(low=1, high=10, name='base_epochs')

dim_batch_size = Integer(low=1, high=200, name='batch_size')
dim_optimizer = Categorical(categories=['Adam', 'Nadam', 'RMSprop'], name='optimizer')
dim_pooling_type = Categorical(categories=['max', 'avg'], name='pooling_type')
dim_base_epochs = Integer(low=1, high=3, name='base_epochs')
dim_base_learning_rate = Integer(low=1e-3, high=3, name='learning_rate')



#default_parameters = [16, 'Adam', 'avg', 1]

#dimensions = [dim_batch_size,
#              dim_optimizer,
#              dim_pooling_type,
#              dim_base_epochs]
              
dim_base_learning_rate = Real(low=1e-6, high=1e-4, name='learning_rate')
default_parameters = [15e-5]
dimensions = [dim_base_learning_rate]
              

path_best_model = 'best_model.keras'
best_accuracy = 0.0

@use_named_args(dimensions=dimensions)
def fitness(learning_rate):
    # Print the hyper-parameters.
    print('learning_rate:', learning_rate)

    model, history = train_model(optimizer=optimizers.Adam(learning_rate), epochs=1)
    
        # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    accuracy = history.history['val_acc'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    global best_accuracy

    # If the classification accuracy of the saved model is improved ...
    if accuracy > best_accuracy:
        # Save the new model to harddisk.
        model.save(path_best_model)
        
        # Update the classification accuracy.
        best_accuracy = accuracy

    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    
    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return -accuracy

#fitness(x=[16, optimizers.Adam(15e-5), 'avg', 1])
#optimizers.Adam(15e-5)
model, history = train_model(optimizers.Adam(15e-5), 'avg', base_epochs=1, epochs=30)

# serialize model to JSON
model_json = model.to_json()
with open("flowers_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("flowers_model.h5")
print("Saved model to disk")

model.save('flowers_full_model.h5') 


# from keras.models import model_from_json
# # load json and create model
# json_file = open('flowers_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("flowers_model.h5")
# print("Loaded model from disk")

# model = loaded_model
# search_result = gp_minimize(func=fitness,
#                             dimensions=dimensions,
#                             acq_func='EI', # Expected Improvement.
#                             n_calls=5,
#                             x0=default_parameters)
# space = search_result.space

# print(f'Best Accuracy: {-search_result.fun}')
# params = space.point_to_dict(search_result.x)
# print(f'Hyper Params:\n {search_result.x}')

#dim_names = ['batch_size', 'optimizer', 'pooling_type', 'base_epochs']
#fig, ax = plot_objective(result=search_result)#, dimensions=dim_names)
#fig, ax = plot_evaluations(result=search_result, dimensions=dim_names)

#print("clear session")
#K.clear_session()

#print('del model')
#del model

#load best weights
model.load_weights("./best-flower-weights.h5")

X = []

print('loading test data')

testpath = '../input/test/test/'

for filename in os.listdir(testpath):
    img_path = os.path.join(os.path.abspath(testpath), filename)
    img = PIL.Image.open(img_path)
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    X.append(np.asarray(img)[:, :, :3]) #remove alpha channel    
    
print("overall images:", len(X))

X = np.stack(X).astype(np.float32)#/255.0

print('preprocess input')

X = preprocess_input(X)

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator



#print('loading best model')

#model = load_model('best-flower-model.h5')

#print('best model loaded')

predictions = model.predict(X)
print(predictions)

y_classes = predictions.argmax(axis=-1)
predicted_labels = np.array([index_to_label_map[key] for key in y_classes])
print(predicted_labels)

import os.path

ids = []
for filename in os.listdir(testpath):
    ids.append(int(os.path.splitext(filename)[0]))

submission = pd.DataFrame({'Id':ids,'Category':predicted_labels})
filename = 'Submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)

#https://ai.googleblog.com/2016/08/improving-inception-and-image.html
