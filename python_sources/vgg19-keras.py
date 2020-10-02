# VGG19 on Keras
# A simple Exercise
# VGG-19
from keras.layers import Input
from keras.layers import Conv2D, Dense, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import plot_model

# layer 1
inputs = Input(shape=(256, 256, 3), name='inputs')
conv1_1 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1_1')(inputs)
conv1_2 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1_2')(conv1_1)
pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1_2)

# layer2
conv2_1 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2_1')(pool1)
conv2_2 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2_2')(conv2_1)
pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2_2)

# layer3
conv3_1 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_1')(pool2)
conv3_2 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_2')(conv3_1)
conv3_3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_3')(conv3_2)
conv3_4 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_4')(conv3_3)
pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3_4)

# layer4
conv4_1 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_1')(pool3)
conv4_2 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_2')(conv4_1)
conv4_3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_3')(conv4_2)
conv4_4 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_4')(conv4_3)
pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4_4)

# layer5
conv5_1 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_1')(pool4)
conv5_2 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_2')(conv5_1)
conv5_3 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_3')(conv5_2)
conv5_4 = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_4')(conv5_3)
pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5_4)
global_average_pooling2d_1 = GlobalAveragePooling2D(name='global_average_pooling2d_5')(pool5)

# FC6 layer
FC6 = Dense(1024, activation='relu', name='dense_1')(global_average_pooling2d_1)

# FC7 layer
predictions = Dense(10, activation='softmax', name='dense_2')(FC6)

model = Model(inputs=inputs, outputs=predictions)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
# model.fit()
