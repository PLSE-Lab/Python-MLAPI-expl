from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from imutils import paths
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# %% [code]

import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# %% [code]
imagePaths = list(paths.list_images("/kaggle/working/dataset"))
data = []
labels = []


for imagePath in imagePaths:

    label = imagePath.split(os.path.sep)[-2]


    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))

    
    data.append(image)
    labels.append(label)


data = np.array(data) / 255.0
labels = np.array(labels)


lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


(trainX, testX, trainY, testY) = train_test_split(data, labels,	test_size=0.20, stratify=labels, random_state=42)


print()
print()
print(len(trainX))
print()
print(len(trainY))
print()
print(len(testX))
print()
print(len(testY))




(trainX, valX, trainY, valY) = train_test_split(trainX, trainY, test_size=0.20, stratify=trainY, random_state=22)







print()
print('train length after val split = ', len(trainX))
print()

print()
print('validation length after val split = ', len(valX))
print()

print()
print(testY)

print()


# %% [code]

INIT_LR = 1e-3
EPOCHS = 100
BS = 128

# %% [code]

trainAug = ImageDataGenerator(
    rotation_range=15,
    fill_mode="nearest")

# %% [code]
baseModel = VGG19(weights=None, include_top=False,	input_tensor=Input(shape=(128, 128, 3)))


headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)





print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

# %% [code]
print("[INFO] training head...")
H = model.fit_generator(
    trainAug.flow(trainX, trainY, batch_size=BS),
    validation_data=(valX, valY),
    validation_steps=len(valX) // BS,
    epochs=EPOCHS)


print("[INFO] evaluating network...")

predIdxs = model.predict(testX, batch_size=BS)
print()
print()
print()
print(predIdxs)
print()
print()
print()
predIdxs = np.argmax(predIdxs, axis=1)

print()
print()
print()
print(predIdxs)
print()
print()

print(classification_report(testY.argmax(axis=1), predIdxs,target_names=lb.classes_))

class_names = ['covid','normal']

cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

#plot confusion matrix

plot_confusion_matrix(conf_mat=cm, figsize=(12, 12), class_names = class_names, show_normed=False) 	
plt.show()


# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('vgg19-train-test-graph.png')
plt.show()

# Plot training & validation accuracy values
plt.figure()
plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('vgg19-train-accuracy-graph.png')
plt.show()

# Plot training & validation loss values
plt.figure()
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('vgg19-train-loss-graph.png')
plt.show()


# %% [code]
#roc curve

print("testY==",testY)
print("predIdxs==",predIdxs)
testY = np.argmax(testY, axis=1)

fpr, tpr, thresholds = roc_curve(testY, predIdxs )
roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt

plt.figure()
lw = 1
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('vgg19-roc-graph.png')
plt.show()


#saving the model
print("[INFO] saving COVID-19 detector model...")
model.save('vgg19_model.h5')