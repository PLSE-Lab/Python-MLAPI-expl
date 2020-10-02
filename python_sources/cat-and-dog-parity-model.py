import os
import tensorflow as tf
from glob import glob
import numpy as np
import cv2
import pickle
from PIL import Image

img_size = 150
k = 4

base_dir = "/kaggle/input/catdogfiltered/cats_and_dogs_filtered"
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, "cats")

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, "dogs")

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, "cats")

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, "dogs")

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(150, 150, 3), kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(k + 1, activation="sigmoid"),
    ]
)

# model = tf.keras.models.load_model("parity.h5")
model.compile(
    loss=tf.keras.losses.MSE,
    optimizer=tf.keras.optimizers.Adam(lr=1e-4),
    metrics=["acc"],
)

### data generator ###
# def rand(a=0, b=1):
#     return np.random.rand()*(b-a) + a

# def get_random_data(image, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
#     '''random preprocessing for real-time data augmentation'''
#     # numpy array: BGR, 0-255
#     # height, width, channel
#     image_data = cv2.resize(image, input_shape)
#     h, w = input_shape
#     if not random:
#         return image_data

#     # resize image
#     new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
#     scale = rand(.75, 1.25)
#     if new_ar < 1:
#         nh = int(scale*h)
#         nw = int(nh*new_ar)
#     else:
#         nw = int(scale*w)
#         nh = int(nw/new_ar)

#     # resize
#     image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
#     # convert into PIL Image object
#     image = Image.fromarray(image[:, :, ::-1])

#     # place image
#     dx = int(rand(0, w-nw))
#     dy = int(rand(0, h-nh))
#     new_image = Image.new('RGB', (w,h), (128,128,128))
#     new_image.paste(image, (dx, dy))
#     # convert into numpy array: BGR, 0-255
#     image = np.asarray(new_image)[:, :, ::-1]

#     # horizontal flip (faster than cv2.flip())
#     h_flip = rand() < 0.5
#     if h_flip:
#         image = image[:, ::-1]

#     # vertical flip
#     v_flip = rand() < 0.5
#     if v_flip:
#         image = image[::-1]

#     # rotation augment
#     is_rot = False
#     if is_rot:
#         right = rand() < 0.5
#         if right:
#             image = image.transpose(1, 0, 2)[:, ::-1]
#         else:
#             image = image.transpose(1, 0, 2)[::-1]

#     # distort image
#     img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     H = img_hsv[:, :, 0].astype(np.float32)
#     S = img_hsv[:, :, 1].astype(np.float32)
#     V = img_hsv[:, :, 2].astype(np.float32)

#     hue = rand(-hue, hue) * 179
#     H += hue
#     np.clip(H, a_min=0, a_max=179, out=H)

#     sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
#     S *= sat
#     np.clip(S, a_min=0, a_max=255, out=S)

#     val = rand(1, val) if rand()<.5 else 1/rand(1, val)
#     V *= val
#     np.clip(V, a_min=0, a_max=255, out=V)

#     img_hsv[:, :, 0] = H.astype(np.uint8)
#     img_hsv[:, :, 1] = S.astype(np.uint8)
#     img_hsv[:, :, 2] = V.astype(np.uint8)

#     # convert into numpy array: RGB, 0-1
#     image_data = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
#     return image_data

def get_X(data_list: np.array, random: bool):
    newimg = np.zeros((img_size, img_size, 3), np.uint8)
    assert k == 4
    for x in range(2):
        for y in range(2):
            i = x * 2 + y
            data = data_list[i]
            img = cv2.imread(data)
#             img = get_random_data(img, (img_size // 2, img_size // 2), random)
            img = cv2.resize(img, (img_size // 2, img_size // 2))
            newimg[
                x * img_size // 2 : (x + 1) * img_size // 2,
                y * img_size // 2 : (y + 1) * img_size // 2,
            ] = img
#     cv2.imwrite("tmp.jpg", newimg)
    return tf.keras.preprocessing.image.img_to_array(newimg) / 255.0


def data_generator(cats_dir: str, dogs_dir: str, batch_size, random=True):
    cats_list = glob(os.path.join(cats_dir, "*.jpg"))
    dogs_list = glob(os.path.join(dogs_dir, "*.jpg"))
    while True:
        x_data = []
        y_data = []
        for _ in range(batch_size):
            cat_num = np.random.randint(0, k + 1)
            dog_num = k - cat_num
            data_list = np.hstack(
                [
                    np.random.choice(cats_list, cat_num),
                    np.random.choice(dogs_list, dog_num),
                ]
            )
#             np.random.shuffle(data_list)
            x = get_X(data_list, random)
            y = np.eye(k + 1, k + 1)[dog_num]
            x_data.append(x)
            y_data.append(y)
        yield np.array(x_data), np.array(y_data)


train_generator = data_generator(train_cats_dir, train_dogs_dir, 64)
validation_generator = data_generator(validation_cats_dir, validation_dogs_dir, 64)
#######################
next(train_generator)
# train_generator.debug()
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=15, verbose=1
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", min_delta=0, patience=50, verbose=1
)
class myCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        acc = logs.get('acc')
        print("batch: %d loss: %.4f acc:%.4f" %(batch+1,loss,acc))
#     def on_epoch_end(self, epoch, logs={}):
#         print("\nepoch_end\n")
#         train_generator.on_epoch_end()
#         validation_generator.on_epoch_end()
        
history = model.fit(
    train_generator,
    steps_per_epoch=250,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=25,
    verbose=2,
    callbacks=[reduce_lr, early_stopping,myCallback()],
)

model.save("parity.h5")

with open(f"parity_model_history_k{k}.pck", "wb") as file:
    pickle.dump(history.history, file)
