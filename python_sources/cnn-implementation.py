
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import itertools

train_data = pd.read_csv('../input/train.csv')
train_labels = to_categorical(train_data.iloc[:,0], num_classes=10, dtype='float32')
train_flat_data = train_data.iloc[:,1:]
train_images = train_flat_data.values.reshape(-1, 28, 28, 1)

test_data = pd.read_csv('../input/test.csv')
test_images = test_data.values.reshape(-1, 28, 28, 1)
plt.imshow(test_images[1,:,:,0])

train_x, val_x, train_y, val_y = train_test_split(train_images, train_labels, shuffle=True, test_size=0.05)

train_x = train_x/255.
val_x = val_x/255.

model = Sequential([
    Conv2D(32, (4,4), input_shape=(28,28,1), activation='relu',padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.2),
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(10, activation='softmax')
])

print(train_images.shape)
print(train_labels.shape)

optimizer = Adam(lr=0.001, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

datagen = ImageDataGenerator(
        height_shift_range=0.1,
        width_shift_range=0.1,
        zoom_range=0.1,
        rotation_range=10,
    )

datagen.fit(train_x)
history = model.fit_generator(datagen.flow(train_x, train_y, batch_size=64), validation_data=(val_x, val_y), steps_per_epoch = len(train_x) // 64, epochs=50, verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(train_loss)
plt.plot(val_loss)
plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


pred_val = model.predict(val_x)
pred_val_vals = np.argmax(pred_val, axis=1)
val_y_vals = np.argmax(val_y, axis=1)
cm =confusion_matrix(val_y_vals, pred_val_vals)
plot_confusion_matrix(cm , range(10))



# submission

results = model.predict(test_images)
results = np.argmax(results, axis=1)
p_results = pd.Series(results, name='Label')
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),p_results],axis = 1)
submission.to_csv("cnn_v7.csv",index=False)

print(submission.head())