#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import io
import bson
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import concurrent.futures
from multiprocessing import cpu_count


# In[ ]:


num_images = 800000
im_size = 16
num_cpus = cpu_count()


# In[ ]:


def imread(buf):
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_ANYCOLOR)

def img2feat(im):
    x = cv2.resize(im, (im_size, im_size), interpolation=cv2.INTER_AREA)
    return np.float32(x) / 255

X = np.empty((num_images, im_size, im_size, 3), dtype=np.float32)
y = []

def load_image(pic, target, bar):
    picture = imread(pic)
    x = img2feat(picture)
    bar.update()
    
    return x, target

bar = tqdm_notebook(total=num_images)
with open('../input/train.bson', 'rb') as f,         concurrent.futures.ThreadPoolExecutor(num_cpus) as executor:

    data = bson.decode_file_iter(f)
    delayed_load = []

    i = 0
    try:
        for c, d in enumerate(data):
            target = d['category_id']
            for e, pic in enumerate(d['imgs']):
                delayed_load.append(executor.submit(load_image, pic['picture'], target, bar))
                
                i = i + 1

                if i >= num_images:
                    raise IndexError()

    except IndexError:
        pass;
    
    for i, future in enumerate(concurrent.futures.as_completed(delayed_load)):
        x, target = future.result()
        
        X[i] = x
        y.append(target)


# In[ ]:


X.shape, len(y)


# In[ ]:


y = pd.Series(y)

num_classes =500 
valid_targets = set(y.value_counts().index[:num_classes-1].tolist())
valid_y = y.isin(valid_targets)

y[~valid_y] = -1

max_acc = valid_y.mean()
print(max_acc)


# In[ ]:


y, rev_labels = pd.factorize(y)


# In[ ]:


from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
#from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(16, 3 , activation='relu', padding='same', input_shape=X.shape[1:]))
#model.add(Conv2D(16, 2, activation='relu', padding='same'))
model.add(MaxPooling2D(2))
model.add(Dropout(0.25))

model.add(Conv2D(32, 3, activation='relu', padding='same'))
#model.add(Conv2D(32, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(num_classes, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


opt = Adam(lr=0.001)

model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()


model.fit(X, y, validation_split=0.1, epochs=5)


model.save_weights('model.h5')


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='_id')

most_frequent_guess =1000018296
submission['category_id'] = most_frequent_guess 

num_images_test = 950000
with open('../input/test.bson', 'rb') as f,          concurrent.futures.ThreadPoolExecutor(num_cpus) as executor:

    data = bson.decode_file_iter(f)

    future_load = []

    for i,d in enumerate(data):
        if i >= num_images_test:
              break
        future_load.append(executor.submit(load_image, d['imgs'][0]['picture'], d['_id'], bar))
        
        print("Starting future processing")
    for future in concurrent.futures.as_completed(future_load):
        x, _id = future.result()
        
        y_cat = rev_labels[np.argmax(model.predict(x[None])[0])]
        if y_cat == -1:
            y_cat = most_frequent_guess

        bar.update()
        submission.loc[_id, 'category_id'] = y_cat
print('Finished')


# In[ ]:


submission.to_csv('new_submission.csv.gz', compression='gzip')

