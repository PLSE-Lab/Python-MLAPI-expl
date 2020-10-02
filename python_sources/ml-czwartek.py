from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statistics import mean

train = pd.read_csv("../input/train.csv")
X = train.values[:,1:].astype('float32')

X = X.reshape(len(X), 28, 28, 1)

cut = int(0.05*len(X))
print("training on {} examples".format(cut))

x_train = X[:cut]
x_test = X[cut:cut*2]

x_train /= 255
x_test /= 255

noise_factor = 0.3
x_train_noisy = np.clip(x_train + np.random.normal(scale=noise_factor, size=x_train.shape), 0, 1)
x_test_noisy = np.clip(x_test + np.random.normal(scale=noise_factor, size=x_test.shape), 0, 1)

epochs = 40

# --- MODEL --- #

input_img = Input(shape=(28, 28, 1), name='input_img')

conv_1 = Conv2D(32, (3, 3), activation='relu', padding='same')
x = conv_1(input_img)
pool_1 = MaxPooling2D((2, 2), padding='same')
x = pool_1(x)

conv_2 = Conv2D(32, (3, 3), activation='relu', padding='same')
x = conv_2(x)
pool_2 = MaxPooling2D((2, 2), padding='same')
x = pool_2(x)

conv_3 = Conv2D(16, (3, 3), activation='relu', padding='same')
x = conv_3(x)
pool_3 = MaxPooling2D((2, 2), padding='same')
x = pool_3(x)

flatten_1 = Conv2D(8, (3, 3), activation='relu', padding='same',
                   activity_regularizer=regularizers.l1(1e-6))
encoded = flatten_1(x)
# <- BOTTLENECK (4, 4, 8)
conv_4 = Conv2D(16, (3, 3), activation='relu', padding='same')
x = conv_4(encoded)
upsampling_1 = UpSampling2D((2, 2), name='upsampling_1')
x = upsampling_1(x)

conv_5 = Conv2D(32, (3, 3), activation='relu', padding='same')
x = conv_5(x)
upsampling_2 = UpSampling2D((2, 2), name='upsampling_2')
x = upsampling_2(x)

conv_6 = Conv2D(32, (3, 3), activation='relu', padding='valid')  # padding trimming
x = conv_6(x)
upsampling_3 = UpSampling2D((2, 2))
x = upsampling_3(x)

flatten_2 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')
decoded = flatten_2(x)

# AUTOENCODER
autoencoder = Model(input_img, decoded)
# ENCODER
encoder = Model(input_img, encoded)
# DECODER
encoded_input = Input(shape=(4, 4, 8), name='encoded_input')
x = conv_4(encoded_input)
x = upsampling_1(x)
x = conv_5(x)
x = upsampling_2(x)
x = conv_6(x)
x = upsampling_3(x)
decoded = flatten_2(x)

decoder = Model(encoded_input, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train, epochs=epochs, batch_size=256, validation_data=(x_test_noisy, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

print(encoded_imgs.mean())

# plotting results
n = 10
plt.figure(figsize=(10, 4))
for i in range(n):
    # original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()

    # noisy
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()

    # reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
plt.savefig('imgs.png')
# plotting representations
n = 10
plt.figure(figsize=(10, 2))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(encoded_imgs[i].reshape(4, -1).T)
    plt.gray()
plt.savefig('encodings.png')

