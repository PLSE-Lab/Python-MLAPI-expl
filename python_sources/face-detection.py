#!/usr/bin/env python
# coding: utf-8

# # Face Detection
# 
# Hello! In this task you will create your own deep face detector.
# 
# First of all, we need import some useful stuff.

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('pylab', 'inline')
get_ipython().run_line_magic('cd', "'/kaggle/input/face-detection-dataset/face-detection'")


# Do you have modern Nvidia [GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit)? There is your video-card model in [list](https://developer.nvidia.com/cuda-gpus) and CUDA capability >= 3.0?
# 
# - Yes. You can use it for fast deep learning! In this work we recommend you use tensorflow backend with GPU. Read [installation notes](https://www.tensorflow.org/install/) with attention to gpu section, install all requirements and then install GPU version `tensorflow-gpu`.
# - No. CPU is enough for this task, but you have to use only simple model. Read [installation notes](https://www.tensorflow.org/install/) and install CPU version `tensorflow`.
# 
# Of course, also you should install `keras`, `matplotlib`, `numpy` and `scikit-image`.

# In[ ]:


from keras import backend as K


# In[ ]:


from matplotlib import pyplot as plt
import numpy as np
from skimage import transform


# In[ ]:


from get_data import load_dataset, unpack


# In this task we use processed [FDDB dataset](http://vis-www.cs.umass.edu/fddb/). Processing defined in file [./prepare_data.ipynb](prepare_data.ipynb) and consists of:
# 
# 1. Extract bboxes from dataset. In base dataset face defined by [ellipsis](http://vis-www.cs.umass.edu/fddb/samples/) that not very useful for basic neural network learning.
# 2. Remove images with big and small faces on one shoot.
# 3. Re-size images to bounding boxes (bboxes) have same size 32 +/- pixels.
# 
# Each image in train, validation and test datasets have shape (176, 176, 3), but part of this image is black background. Interesting image aligned at top left corner.
# 
# Bounding boxes define face in image and consist of 5 integer numbers: [image_index, min_row, min_col, max_row, max_col]. Bounding box width and height are 32 +/- 8 pixels wide.
# 
# `train_bboxes` and `val_bboxes` is a list of bboxes.
# 
# `train_shapes` and `val_shapes` is a list of interesting image shapes.

# In[ ]:


# First run will download 30 MB data from github

train_images, train_bboxes, train_shapes = load_dataset("data", "train")
val_images, val_bboxes, val_shapes = load_dataset("data", "val")


# ## Prepare data (1 point)
# 
# For learning we should extract positive and negative samples from image.
# Positive and negative samples counts should be similar.
# Every samples should have same size.

# In[ ]:


from graph import visualize_bboxes
visualize_bboxes(images=train_images,
                 true_bboxes=train_bboxes
                )


# Every image can represent multiple faces, so we should extract all faces from every images and crop them to `SAMPLE_SHAPE`. This set of extracted images are named `positive`.
# 
# Then we chould extract `negative` set. This images should have `SAMPLE_SHAPE` size. Pseudocode for extracting:
# 
#     negative_collection := []
# 
#     for i in range(negative_bbox_count):
#         Select random image.
#         image_shape := image_shapes[image_index]
#         image_true_bboxes := true_bboxes[true_bboxes[:, 0] == image_index, 1:]
#         
#         for j in TRY_COUNT: # TRY_COUNT is a magic constant, for example, 100
#             Generate new_bbox within image_shape.
#             
#             if new_bbox is negative bbox for image_true_bboxes:
#                 Extract from image, new_bbox and resize to SAMPLE_SIZE negative_sample.
#                 Add negative sample to negative_collection.
#                 Break # for j in TRY_COUNT

# In[ ]:


SAMPLE_SHAPE = (32, 32, 3)


# In[ ]:


from scores import iou_score # https://en.wikipedia.org/wiki/Jaccard_index

def is_negative_bbox(new_bbox, true_bboxes, eps=1e-1):
    """Check if new bbox not in true bbox list.
    
    There bbox is 4 ints [min_row, min_col, max_row, max_col] without image index."""
    for bbox in true_bboxes:
        if iou_score(new_bbox, bbox) >= eps:
            return False
    return True


# In[ ]:


# Write this function
def gen_negative_bbox(image_shape, bbox_size, true_bboxes):
    """Generate negative bbox for image."""
    tries = 1000
    for i in range(tries):
        corner_x = np.random.randint(max(image_shape[0] - bbox_size[0], 1))
        corner_y = np.random.randint(max(image_shape[1] - bbox_size[1], 1))
        new_bbox = [corner_x, corner_y, corner_x + bbox_size[0], corner_y + bbox_size[1]]
        if is_negative_bbox(new_bbox,true_bboxes):
            return new_bbox
    return None

def get_positive_negative(images, true_bboxes, image_shapes, negative_bbox_count=None):
    """Retrieve positive and negative samples from image."""
    positive = []
    negative = []
    image_count = image_shapes.shape[0]
    
    if negative_bbox_count is None:
        negative_bbox_count = len(true_bboxes)
    
    # Pay attention to the fact that most part of image may be black -
    # extract negative samples only from part [0:image_shape[0], 0:image_shape[1]]
    
    # Write code here
    # ...
    w, h, c = SAMPLE_SHAPE
    for true_bbox in true_bboxes:        
        image_index = true_bbox[0]
        pos_img = images[image_index][true_bbox[1]:true_bbox[1]+w, true_bbox[2]:true_bbox[2]+h, :]
        positive.append(pos_img)
    
    print(negative_bbox_count)
    for i in range(negative_bbox_count):
        image_index = np.random.choice(len(images))
        image_shape = image_shapes[image_index]
        image_true_bboxes = true_bboxes[true_bboxes[:, 0] == image_index, 1:]
        for j in range(100):
            bbox_size = (w, h)
            new_bbox = gen_negative_bbox(image_shape, bbox_size, image_true_bboxes)
            if new_bbox is None: continue
            neg_img = images[image_index][new_bbox[0]:new_bbox[2], new_bbox[1]:new_bbox[3]]
            negative.append(neg_img)
            break
        print(i)
    return positive, negative


# In[ ]:


def get_samples(images, true_bboxes, image_shapes):
    """Usefull samples for learning.
    
    X - positive and negative samples.
    Y - one hot encoded list of zeros and ones. One is positive marker.
    """
    positive, negative = get_positive_negative(images=images, true_bboxes=true_bboxes, 
                                               image_shapes=image_shapes)
    X = positive
    Y = [[0, 1]] * len(positive)
    
    X.extend(negative)
    Y.extend([[1, 0]] * len(negative))
    
    return np.array(X), np.array(Y)


# Now we can extract samples from images.

# In[ ]:


X_train, Y_train = get_samples(train_images, train_bboxes, train_shapes)
X_val, Y_val = get_samples(val_images, val_bboxes, val_shapes)


# In[ ]:


out_file = '/kaggle/working/{}.npy'
np.save(out_file.format('X_train'), X_train)
np.save(out_file.format('Y_train'), Y_train)
np.save(out_file.format('X_val'), X_val)
np.save(out_file.format('Y_val'), Y_val)


# In[ ]:


out_file = '/kaggle/working/{}.npy'
X_train = np.load(out_file.format('X_train'))
Y_train = np.load(out_file.format('Y_train'))
X_val = np.load(out_file.format('X_val'))
Y_val = np.load(out_file.format('Y_val'))


# In[ ]:


# There we should see faces
from graph import visualize_samples
visualize_samples(X_train[Y_train[:, 1] == 1])


# In[ ]:


# There we shouldn't see faces
visualize_samples(X_train[Y_train[:, 1] == 0])


# ## Classifier training (3 points)
# 
# First of all, we should train face classifier that checks if face represented on sample.

# In[ ]:


BATCH_SIZE = 64
K.clear_session()


# ### Image augmentation
# 
# Important thing in deep learning is augmentation. Sometimes, if your model are complex and cool, you can increase quality by using good augmentation.
# 
# Keras provide good [images preprocessing and augmentation](https://keras.io/preprocessing/image/). This preprocessing executes online (on the fly) while learning.
# 
# Of course, if you want using samplewise and featurewise center and std normalization you should run this transformation on predict stage. But you will use this classifier to fully convolution detector, in this case such transformation quite complicated, and we don't recommend use them in classifier.
# 
# For heavy augmentation you can use library [imgaug](https://github.com/aleju/imgaug). If you need, you can use this library in offline manner (simple way) and online manner (hard way). However, hard way is not so hard: you only have to write [python generator](https://wiki.python.org/moin/Generators), which returns image batches, and pass it to [fit_generator](https://keras.io/models/model/#fit_generator)

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator # Usefull thing. Read the doc.

datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.1,
                            )
datagen.fit(X_train)


# ### Fitting classifier
# 
# For fitting you can use one of Keras optimizer algorithms. [Good overview](http://ruder.io/optimizing-gradient-descent/)
# 
# To choose best learning rate strategy you should read about EarlyStopping and ReduceLROnPlateau or LearningRateScheduler on [callbacks](https://keras.io/callbacks/) page of keras documentation, it's very useful in deep learning.
# 
# If you repeat architecture from some paper, you can find information about good optimizer algorithm and learning rate strategy in this paper. For example, every [keras application](https://keras.io/applications/) has link to paper, that describes suitable learning procedure for this specific architecture.

# In[ ]:


import os.path
from keras.optimizers import Adam
# Very usefull, pay attention
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from graph import plot_history

model_path = '/kaggle/working/FDmodel.hdf5'
if os.path.isfile(model_path):
    os.remove(model_path)
    
def fit(model, datagen, X_train, Y_train, X_val, Y_val, class_weight=None, epochs=50, lr=0.001, verbose=False):
    """Fit model.
    
    You can edit this function anyhow.
    """
    
    if verbose:
        model.summary()

    model.compile(optimizer=Adam(lr=lr), # You can use another optimizer
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                                  validation_data=(datagen.standardize(X_val), Y_val),
                                  epochs=epochs, steps_per_epoch=len(X_train) // BATCH_SIZE,
                                  callbacks=[ModelCheckpoint(model_path, save_best_only=True)],
                                  class_weight=class_weight,
            
                                 )  # starts training
    
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


# #### (first point out of three)
# 
# ![lenet architecture](lenet_architecture.png)
# LeCun, Y., Bottou, L., Bengio, Y. and Haffner, P., 1998. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), pp.2278-2324.
# 
# Of course, you can use any another architecture, if want. Main thing is classification quality of your model.
# 
# Acceptable validation accuracy for this task is 0.92.

# In[ ]:


from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Activation, Input, Dropout, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

def generate_model(sample_shape):
    # Classification model
    # You can start from LeNet architecture
    x = inputs = Input(shape=sample_shape)

    # Write code here
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)

    # This creates a model
    predictions = Dense(2, activation='softmax')(x)
    return Model(inputs=inputs, outputs=predictions)

model = generate_model(SAMPLE_SHAPE)


# In[ ]:


model.summary()


# #### Fit the model (second point out of three)
# 
# If you doesn't have fast video-card suitable for deep learning, you can first check neural network modifications with small value of parameter `epochs`, for example, 10, and then after selecting best model increase this parameter.
# Fitting on CPU can be long, we suggest do it at bedtime.
# 
# Don't forget change model name.

# In[ ]:


# Attention: Windows implementation may cause an error here. In that case use model_name=None.
fit(model=model, datagen=datagen, X_train=X_train.astype('float32'), X_val=X_val.astype('float32'), Y_train=Y_train, Y_val=Y_val)


# #### (third point out of three)
# 
# After learning model weights saves in folder `data/checkpoints/`.
# Use `model.load_weights(fname)` to load best weights.
# 
# If you use Windows and Model Checkpoint doesn't work on your configuration, you should implement [your own Callback](https://keras.io/callbacks/#create-a-callback) to save best weights in memory and then load it back.

# In[ ]:


def get_checkpoint():
    return model_path

model.load_weights(get_checkpoint())


# ## Detection
# 
# If you have prepared classification architecture with high validation score, you can use this architecture for detection.
# 
# Convert classification architecture to fully convolution neural network (FCNN), that returns heatmap of activation.
# 
# ### Detector model or sliding window (1 point)
# 
# Now you should replace fully-connected layers with $1 \times 1$ convolution layers.
# 
# Every fully connected layer perform operation $f(Wx + b)$, where $f(\cdot)$ is nonlinear activation function, $x$ is layer input, $W$ and $b$ is layer weights. This operation can be emulated with $1 \times 1$ convolution with activation function $f(\cdot)$, that perform exactly same operation $f(Wx + b)$.
# 
# If there is `Flatten` layer with $n \times k$ input size before fully connected layers, convolution should have same $n \times k$ input size.
# Multiple fully connected layers can be replaced with convolution layers sequence.
# 
# After replace all fully connected layers with convolution layers, we get fully convolution network. If input shape is equal to input size of previous network, output will have size $1 \times 1$. But if we increase input shape, output shape automatically will be increased. For example, if convolution step of previous network strides 4 pixels, increase input size with 100 pixels along all axis makes increase outputsize with 25 values along all axis. We got activation map of classifier without necessary extract samples from image and multiple calculate low-level features.
# 
# In total:
# 1. $1 \times 1$ convolution layer is equivalent of fully connected layer.
# 2. $1 \times 1$ convolution layers can be used to get activation map of classification network in "sliding window" manner.
# 
# We propose replace last fully connected layer with softmax actiovation to convolution layer with linear activation.It will be usefull to find good treshold. Of course, you can use softmax activation.
# 
# #### Example of replace cnn head:
# 
# ##### Head before convert
# 
# ![before replace image](before_convert.png)
# 
# ##### Head after convert
# 
# ![before replace image](after_convert.png)
# 
# On this images displayed only head. `InputLayer` should be replaced with convolution part exit.
# Before convert network head takes fifty $8 \times 8$ feature maps and returns two values: probability of negative and positive classes. This output can be considered as activation map with size $1 \times 1$.
# 
# If input have size $8 \times 8$, output after convert would have $1 \times 1$ size, but input size is $44 \times 44$.
# After convert network head returns one $37 \times 37$ activation map.

# In[ ]:


# FCNN

IMAGE_SHAPE = (176, 176, 3)

def generate_fcnn_model(image_shape):
    """After model compilation input size cannot be changed.
    
    So, we need create a function to have ability to change size later.
    """
    x = inputs = Input(image_shape)

    # Write code here
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (8, 8), activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Dropout(0.25)(x)

    # This creates a model
    predictions = Conv2D(2, (1, 1), activation='linear')(x)
    return Model(inputs=inputs, outputs=predictions)

fcnn_model = generate_fcnn_model(IMAGE_SHAPE)


# In[ ]:


fcnn_model.summary()


# #### (1 point)
# 
# Then you should write function that copy weights from classification model to fully convolution model.
# Convolution weights may be copied without modification, fully-connected layer weights should be reshaped before copy.
# 
# Pay attention to last layer.

# In[ ]:


def copy_weights(base_model, fcnn_model):
    """Set FCNN weights from base model.
    """
    
    new_fcnn_weights = []
    prev_fcnn_weights = fcnn_model.get_weights()
    prev_base_weights = base_model.get_weights()
    
    # Write code here
    for prev_fcnn_weight, prev_base_weight in zip(prev_fcnn_weights, prev_base_weights):
        new_fcnn_weights.append(prev_base_weight.reshape(prev_fcnn_weight.shape))
        
    fcnn_model.set_weights(new_fcnn_weights)

copy_weights(base_model=model, fcnn_model=fcnn_model)


# ### Model visualization

# In[ ]:


from graph import visualize_heatmap


# In[ ]:


predictions = fcnn_model.predict(np.array(val_images))
visualize_heatmap(val_images, predictions[:, :, :, 1])


# ### Detector (1 point)
# 
# First detector part is getting bboxes and decision function.
# Greater decision function indicates better detector confidence.
# 
# This function should return pred_bboxes and decision_function:
# 
# - `pred bboxes` is list of 5 int tuples like `true bboxes`: `[image_index, min_row, min_col, max_row, max_col]`.
# - `decision function` is confidence of detector for every pred bbox: list of float values, `len(decision function) == len(pred bboxes)` 
#  
# We propose resize image to `IMAGE_SHAPE` size, find faces on resized image with `SAMPLE_SHAPE` size and then resize them back.

# In[ ]:


# Detection
from skimage.feature import peak_local_max

def get_bboxes_and_decision_function(fcnn_model, images, image_shapes):      
    cropped_images = np.array([transform.resize(image, IMAGE_SHAPE, mode="reflect")  if image.shape != IMAGE_SHAPE else image for image in images])
    pred_bboxes, decision_function = [], []
   
    # Predict
    predictions = fcnn_model.predict(cropped_images)

    # Write code here
    for i in range(len(predictions)):
        img_shape = image_shapes[i]
        local_max_list = peak_local_max(predictions[i][:,:,1], num_peaks=5, min_distance=3, exclude_border=False)
        for local_max_orig in local_max_list:
            local_max = ((local_max_orig + 2)*176/37).astype(int)
            
            if local_max[0] < img_shape[0] and local_max[1] < img_shape[1]:
                bbox = [i] + [local_max[0]-16,local_max[1]-16,local_max[0]+16,local_max[1]+16]
                
                pred_bboxes.append(bbox)
                decision_function.append(predictions[i, local_max_orig[0], local_max_orig[1], 1])
        
    return pred_bboxes, decision_function


# #### Detector visualization

# In[ ]:


pred_bboxes, decision_function = get_bboxes_and_decision_function(fcnn_model=fcnn_model, images=val_images, image_shapes=val_shapes)

visualize_bboxes(images=val_images,
                 pred_bboxes=pred_bboxes,
                 true_bboxes=val_bboxes,
                 decision_function=decision_function
                )


# ## Detector score (1 point)
# 
# Write [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) graph.
# 
# You can use function `best_match` to extract matching between prediction and ground truth, false positive and false negative samples. Pseudo-code for calculation precision and recall graph:
#     
#     # Initialization for first step threshold := -inf
#     tn := 0 # We haven't any positive sample
#     fn := |false_negative| # But some faces wasn't been detected
#     tp := |true_bboxes| # All true bboxes have been caught
#     fp := |false_positive| # But also some false positive samples have been caught
#     
#     Sort decision_function and pred_bboxes with order defined by decision_function
#     y_true := List of answers for "Is the bbox have matching in y_true?" for every bbox in pred_bboxes
#     
#     for y_on_this_step in y_true:
#         # Now we increase threshold, so some predicted bboxes makes positive.
#         # If y_t is True then the bbox is true positive else bbox is false positive
#         # So we should
#         Update tp, tn, fp, fn with attention to y_on_this_step
#         
#         Add precision and recall point calculated by formula through tp, tn, fp, fn on this step
#         Threshold for this point is decision function on this step.

# In[ ]:


from scores import best_match
from graph import plot_precision_recall

def precision_recall_curve(pred_bboxes, true_bboxes, decision_function):
    precision, recall, thresholds = [], [], []
    
    # Write code here
    threshold = min(decision_function) - 1
    max_th = max(decision_function) + 1
    num_steps = 100
    th_step = (max_th - threshold) / num_steps

    sorted_boxes = [[x] + y for y, x in sorted(zip(pred_bboxes, decision_function), key=lambda pair : pair[1])]
    
    for step in range(num_steps):        
        pred_bboxes_th = [x[1:] for x in sorted_boxes if x[0] > threshold]
        if len(pred_bboxes_th) > 0:
            matched, false_negative, false_positive = best_match(pred_bboxes_th, true_bboxes, decision_function)
        else:
            break
        
        prec = len(matched) / (len(matched) + len(false_positive))
        rec = len(matched) / (len(matched) + len(false_negative))
        
        thresholds.append(threshold)
        recall.append(rec)
        precision.append(prec)
        threshold += th_step
        
    return precision, recall, thresholds


# In[ ]:


precision, recall, thresholds = precision_recall_curve(pred_bboxes=pred_bboxes, true_bboxes=val_bboxes, decision_function=decision_function)
plot_precision_recall(precision=precision, recall=recall)


# ### Threshold (1 point)
# 
# Next step in detector creating is select threshold for decision_function.
# Every possible threshold presents point on recall-precision graph.
# 
# Select threshold for `recall=0.85`.

# In[ ]:


def get_threshold(thresholds, recall):
    return thresholds[np.argmax(np.asarray(recall) <= 0.85)] # Write this code

THRESHOLD = get_threshold(thresholds, recall)


# In[ ]:


def detect(fcnn_model, images, image_shapes, threshold, return_decision=True):
    """Get bboxes with decision_function not less then threshold."""
    pred_bboxes, decision_function = get_bboxes_and_decision_function(fcnn_model, images, image_shapes)   
    result, result_decision = [], []
    
    # Write code here
    for i in range(len(pred_bboxes)):
        if decision_function[i] >= threshold:
            result.append(pred_bboxes[i])
            result_decision.append(decision_function[i])
    
    if return_decision:
        return result, result_decision
    else:
        return result


# In[ ]:


pred_bboxes, decision_function = detect(fcnn_model=fcnn_model, images=val_images, image_shapes=val_shapes, threshold=THRESHOLD, return_decision=True)

visualize_bboxes(images=val_images,
                 pred_bboxes=pred_bboxes,
                 true_bboxes=val_bboxes,
                 decision_function=decision_function
                )

precision, recall, thresholds = precision_recall_curve(pred_bboxes=pred_bboxes, true_bboxes=val_bboxes, decision_function=decision_function)
plot_precision_recall(precision=precision, recall=recall)


# ## Test dataset (1 point)
# 
# Last detector preparation step is testing.
# 
# Attention: to avoid over-fitting, after testing algorithm you should run [./prepare_data.ipynb](prepare_data.ipynb), and start all fitting from beginning.
# 
# Detection score (in graph header) should be 0.85 or greater.

# In[ ]:


test_images, test_bboxes, test_shapes = load_dataset("data", "test")

# We test get_bboxes_and_decision_function becouse we want pay attention to all recall values
pred_bboxes, decision_function = get_bboxes_and_decision_function(fcnn_model=fcnn_model, images=test_images, image_shapes=test_shapes)

visualize_bboxes(images=test_images,
                 pred_bboxes=pred_bboxes,
                 true_bboxes=test_bboxes,
                 decision_function=decision_function
                )

precision, recall, threshold = precision_recall_curve(pred_bboxes=pred_bboxes, true_bboxes=test_bboxes, decision_function=decision_function)
plot_precision_recall(precision=precision, recall=recall)


# ## Optional tasks
# 
# ### Real image dataset
# 
# Test your algorithm on original (not scaled) data.
# Visualize bboxes and plot precision-recall curve.

# In[ ]:


# First run will download 523 MB data from github

original_images, original_bboxes, original_shapes = load_dataset("data", "original")


# In[ ]:


# Write code here
# ...


# ## Hard negative mining
# 
# Upgrade the score with [hard negative mining](https://www.reddit.com/r/computervision/comments/2ggc5l/what_is_hard_negative_mining_and_how_is_it/).
# 
# A hard negative is when you take that falsely detected patch, and explicitly create a negative example out of that patch, and add that negative to your training set. When you retrain your classifier, it should perform better with this extra knowledge, and not make as many false positives.

# In[ ]:


# Write this function
def hard_negative(train_images, image_shapes, train_bboxes, X_val, Y_val, base_model, fcnn_model):
    pass


# In[ ]:


hard_negative(train_images=train_images, image_shapes=train_shapes, train_bboxes=train_bboxes, X_val=X_val, Y_val=Y_val, base_model=model, fcnn_model=fcnn_model)


# In[ ]:


model.load_weights("data/checkpoints/...")


# In[ ]:


copy_weights(base_model=model, fcnn_model=fcnn_model)

pred_bboxes, decision_function = get_bboxes_and_decision_function(fcnn_model=fcnn_model, images=val_images, image_shapes=val_shapes)

visualize_bboxes(images=val_images,
                 pred_bboxes=pred_bboxes,
                 true_bboxes=val_bboxes,
                 decision_function=decision_function
                )

precision, recall, thresholds = precision_recall_curve(pred_bboxes=pred_bboxes, true_bboxes=val_bboxes, decision_function=decision_function)
plot_precision_recall(precision=precision, recall=recall)


# ### Multi scale detector
# 
# Write and test detector with [pyramid representation][pyramid].
# [pyramid]: https://en.wikipedia.org/wiki/Pyramid_(image_processing)
# 
# 1. Resize images to predefined scales.
# 2. Run detector with different scales.
# 3. Apply non-maximum supression to detection on different scales.
# 
# References:
# 1. [E. H. Adelson,C. H. Anderson, J. R. Bergen, P. J. Burt, J. M. Ogden: Pyramid methods in image processing](http://persci.mit.edu/pub_pdfs/RCA84.pdf)
# 2. [PETER J. BURT, EDWARD H. ADELSON: The Laplacian Pyramid as a Compact Image Code](http://persci.mit.edu/pub_pdfs/pyramid83.pdf)

# In[ ]:


def multiscale_detector(fcnn_model, images, image_shapes):
    return []


# ### Next  step
# 
# Next steps in deep learning detection are R-CNN, Faster R-CNN and SSD architectures.
# This architecture realization is quite complex.
# For this reason the task doesn't cover them, but you can find the articles in the internet.
