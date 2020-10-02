### Install imutils
!pip install imutils

#### Import packages
import os
import numpy as np
import shutil
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

########################################## CONFIG ##############################################

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = "/kaggle/working/finetuningkeras/dataset"

# define the names of the training, testing, and validation
# directories
TRAIN = "training"
TEST = "evaluation"
VAL = "validation"

# initialize the list of class label names
CLASSES = ["FAKE", "REAL"]

# set the batch size when fine-tuning
BATCH_SIZE = 32

# set the path to the serialized model after training
MODEL_PATH = os.path.sep.join(["/kaggle/working/finetuningkeras","output", "Deepfake.model"])

# define the path to the output training history plots
UNFROZEN_PLOT_PATH = os.path.sep.join(["/kaggle/working/finetuningkeras","output", "unfrozen.png"])
WARMUP_PLOT_PATH = os.path.sep.join(["/kaggle/working/finetuningkeras","output", "warmup.png"])

########################################################################################################

################################ CREATING FOLDERS AND DATA STRUCTURE ###################################
# # Creating Train / Val / Test folders (One time use)
root_dir = '/kaggle/working/finetuningkeras/dataset'
real = '/REAL'
fake = '/FAKE'

os.makedirs(root_dir +'/training' + real)
os.makedirs(root_dir +'/training' + fake)
os.makedirs(root_dir +'/validation' + real)
os.makedirs(root_dir +'/validation' + fake)
os.makedirs(root_dir +'/evaluation' + real)
os.makedirs(root_dir +'/evaluation' + fake)
os.makedirs('/kaggle/working/finetuningkeras/real_fake/FAKE')
os.makedirs('/kaggle/working/finetuningkeras/real_fake/REAL')
os.makedirs('/kaggle/working/FAKE_frames') 
os.makedirs('/kaggle/working/REAL_frames') 
os.makedirs('/kaggle/working/finetuningkeras/output')

# %% [code]
file = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json'
img_path = '/kaggle/input/deepfake-detection-challenge/train_sample_videos'
data_path = '/kaggle/working/finetuningkeras/real_fake'

df = pd.read_json(file)
df = df.T

# %% [code]
label = df[['label']]

#print(label)

for index, row in label.iterrows():
    print(os.path.join(img_path, index))
    print(os.path.join(data_path, row['label'],index))
    shutil.copy(os.path.join(img_path, index),os.path.join(data_path, row['label'],index))
    #print(index,row['label']) 

# %% [code]
v_count = 0
for filename in os.listdir(os.path.join('/kaggle/working/finetuningkeras/real_fake/FAKE')):
    while v_count <25:
        if filename.endswith(".mp4"): 
            print(filename)
            #print(os.path.join('C:\\Users\\Mcd\\Desktop\\Recordings\\lane1\\'+str(filename)))
            # Read the video from specified path 
            cam = cv2.VideoCapture(os.path.join('/kaggle/working/finetuningkeras/real_fake/FAKE/'+str(filename))) 
            print(cam.isOpened())
            print('This is video number...'+str(v_count))
            v_count = v_count + 1

            # frame 
            currentframe = 0

            while(True): 

                # reading from frame 
                ret,frame = cam.read() 

                if ret: 
                    # if video is still left continue creating images 
                    name = '/kaggle/working/FAKE_frames/' + filename + '_frame' + str(currentframe) + '.jpg'
                    print ('Creating...' + name) 

                    # writing the extracted images 
                    cv2.imwrite(name, frame) 

                    # increasing counter so that it will 
                    # show how many frames are created 
                    currentframe += 1
                else: 
                    break

            # Release all space and windows once done 
            cam.release() 
            cv2.destroyAllWindows() 

            continue
        else:
            continue

# %% [code]
v_count = 0
for filename in os.listdir(os.path.join('/kaggle/working/finetuningkeras/real_fake/REAL')):
    while v_count <25:
        if filename.endswith(".mp4"): 
            print(filename)
            #print(os.path.join('C:\\Users\\Mcd\\Desktop\\Recordings\\lane1\\'+str(filename)))
            # Read the video from specified path 
            cam = cv2.VideoCapture(os.path.join('/kaggle/working/finetuningkeras/real_fake/REAL/'+str(filename))) 
            print(cam.isOpened())
            print('This is video number...'+str(v_count))
            v_count = v_count + 1

            # frame 
            currentframe = 0

            while(True): 

                # reading from frame 
                ret,frame = cam.read() 

                if ret: 
                    # if video is still left continue creating images 
                    name = '/kaggle/working/REAL_frames/' + filename + '_frame' + str(currentframe) + '.jpg'
                    print ('Creating...' + name) 

                    # writing the extracted images 
                    cv2.imwrite(name, frame) 

                    # increasing counter so that it will 
                    # show how many frames are created 
                    currentframe += 1
                else: 
                    break

            # Release all space and windows once done 
            cam.release() 
            cv2.destroyAllWindows() 

            continue
        else:
            continue


# Creating partitions of the data after shuffeling
fake_src = '/kaggle/working/FAKE_frames' # Folder to copy images from
root_dir = '/kaggle/working/finetuningkeras/dataset'


f_allFileNames = os.listdir(fake_src)
np.random.shuffle(f_allFileNames)
f_train_FileNames, f_val_FileNames, f_test_FileNames = np.split(np.array(f_allFileNames),
                                                          [int(len(f_allFileNames)*0.7), int(len(f_allFileNames)*0.85)])


f_train_FileNames = [fake_src+'/'+ name for name in f_train_FileNames.tolist()]
f_val_FileNames = [fake_src+'/' + name for name in f_val_FileNames.tolist()]
f_test_FileNames = [fake_src+'/' + name for name in f_test_FileNames.tolist()]

print('Total images: ', len(f_allFileNames))
print('Training: ', len(f_train_FileNames))
print('Validation: ', len(f_val_FileNames))
print('Testing: ', len(f_test_FileNames))

# %% [code]
# Copy-pasting images
for name in f_train_FileNames:
    print(name)
    file = name.split('/')[4:5][0]
    print(file)
    print(os.path.join(root_dir+'/training/FAKE/'+file))
    shutil.move(name,os.path.join(root_dir+'/training/FAKE/'+file))

# %% [code]
for name in f_val_FileNames:
    print(name)
    file = name.split('/')[4:5][0]
    print(os.path.join(root_dir+'/validation/FAKE/'+file))
    shutil.move(name,os.path.join(root_dir+'/validation/FAKE/'+file))
    #print(root_dir+'/validation'+name)
    #shutil.copy(name, "Data2/val"+currentCls)

# %% [code]
for name in f_test_FileNames:
    print(name)
    file = name.split('/')[4:5][0]
    print(os.path.join(root_dir+'/evaluation/FAKE/'+file))
    shutil.move(name,os.path.join(root_dir+'/evaluation/FAKE/'+file))

# %% [code]
# Creating partitions of the data after shuffeling
real_src = '/kaggle/working/REAL_frames' # Folder to copy images from

r_allFileNames = os.listdir(real_src)
np.random.shuffle(r_allFileNames)
r_train_FileNames, r_val_FileNames, r_test_FileNames = np.split(np.array(r_allFileNames),
                                                          [int(len(r_allFileNames)*0.7), int(len(r_allFileNames)*0.85)])


r_train_FileNames = [real_src+'/'+ name for name in r_train_FileNames.tolist()]
r_val_FileNames = [real_src+'/' + name for name in r_val_FileNames.tolist()]
r_test_FileNames = [real_src+'/' + name for name in r_test_FileNames.tolist()]

print('Total images: ', len(r_allFileNames))
print('Training: ', len(r_train_FileNames))
print('Validation: ', len(r_val_FileNames))
print('Testing: ', len(r_test_FileNames))

# %% [code]
# Copy-pasting images
for name in r_train_FileNames:
    print(name)
    file = name.split('/')[4:5][0]
    #print(file)
    print(os.path.join(root_dir+'/training/REAL/'+file))
    shutil.move(name,os.path.join(root_dir+'/training/REAL/'+file))

# %% [code]
for name in r_val_FileNames:
    print(name)
    file = name.split('/')[4:5][0]
    print(os.path.join(root_dir+'/validation/FAKE/'+file))
    shutil.move(name,os.path.join(root_dir+'/validation/REAL/'+file))
    #print(root_dir+'/validation'+name)
    #shutil.copy(name, "Data2/val"+currentCls)

# %% [code]
for name in r_test_FileNames:
    print(name)
    file = name.split('/')[4:5][0]
    print(os.path.join(root_dir+'/evaluation/FAKE/'+file))
    shutil.move(name,os.path.join(root_dir+'/evaluation/REAL/'+file))

# %% [code]
#os.listdir('/kaggle/working/finetuningkeras/dataset/')

####################### IMAGE CLASSIFICATION #############################################

def plot_training(H, N, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)

# derive the paths to the training, validation, and testing
# directories
trainPath = os.path.sep.join([BASE_PATH, TRAIN])
valPath = os.path.sep.join([BASE_PATH, VAL])
testPath = os.path.sep.join([BASE_PATH, TEST])

# determine the total number of image paths in training, validation,
# and testing directories
totalTrain = len(list(paths.list_images(trainPath)))
totalVal = len(list(paths.list_images(valPath)))
totalTest = len(list(paths.list_images(testPath)))

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# initialize the validation/testing data augmentation object (which
# we'll be adding mean subtraction to)
valAug = ImageDataGenerator()

# define the ImageNet mean subtraction (in RGB order) and set the
# the mean subtraction value for each of the data augmentation
# objects
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	trainPath,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=True,
	batch_size=BATCH_SIZE)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	valPath,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=BATCH_SIZE)

# initialize the testing generator
testGen = valAug.flow_from_directory(
	testPath,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=BATCH_SIZE)

# load the VGG16 network, ensuring the head FC layer sets are left
# off
baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(CLASSES), activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model (this needs to be done after our setting our
# layers to being non-trainable
print("[INFO] compiling model...")
opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network for a few epochs (all other layers
# are frozen) -- this will allow the new FC layers to start to become
# initialized with actual "learned" values versus pure random
print("[INFO] training head...")
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // BATCH_SIZE,
	validation_data=valGen,
	validation_steps=totalVal // BATCH_SIZE,
	epochs=50)

# reset the testing generator and evaluate the network after
# fine-tuning just the network head
print("[INFO] evaluating after fine-tuning network head...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
	steps=(totalTest // BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))
plot_training(H, 50, WARMUP_PLOT_PATH)

# reset our data generators
trainGen.reset()
valGen.reset()

# now that the head FC layers have been trained/initialized, lets
# unfreeze the final set of CONV layers and make them trainable
for layer in baseModel.layers[15:]:
	layer.trainable = True

# loop over the layers in the model and show which ones are trainable
# or not
for layer in baseModel.layers:
	print("{}: {}".format(layer, layer.trainable))

# for the changes to the model to take affect we need to recompile
# the model, this time using SGD with a *very* small learning rate
print("[INFO] re-compiling model...")
opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the model again, this time fine-tuning *both* the final set
# of CONV layers along with our set of FC layers
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // BATCH_SIZE,
	validation_data=valGen,
	validation_steps=totalVal // BATCH_SIZE,
	epochs=20)

# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating after fine-tuning network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
	steps=(totalTest // BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))
plot_training(H, 20, UNFROZEN_PLOT_PATH)

# serialize the model to disk
print("[INFO] serializing network...")
model.save(MODEL_PATH)