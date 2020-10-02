import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.models import Model
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.model_selection import learning_curve, GridSearchCV

from tqdm import tqdm_notebook as tqdm
import os
import urllib
import requests
import tarfile
import glob
from collections import defaultdict
import cv2



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

def hog_feature_extraction(data):
    data_hog = []
    hog = cv2.HOGDescriptor(_winSize=(32, 32), 
                            _blockSize=(16, 16),
                            _blockStride=(8, 8),
                            _cellSize=(8, 8), 
                            _nbins=9)
    for image in data:
        data_hog.append(np.hstack(hog.compute(np.transpose(image, (1, 2, 0))
                                              .astype('uint8'))))
    return data_hog

def acc(y_correct, y_pred):
    acc = 0
    for i in range(len(y_pred)):
        acc += y_pred[i] == y_correct[i]
    
    return acc / len(y_pred)




'''

url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
filename = url.rsplit("/", maxsplit=1)[-1]
r = requests.get(url, stream=True)
total_size = int(r.headers.get('content-length', 0)); 

with open(filename, 'wb') as f:
    for data in tqdm(r.iter_content(), total=total_size, unit='B', unit_scale=True):
        f.write(data)


with tarfile.open(filename) as tar_file:
    directory = os.path.commonprefix(tar_file.getnames())
    tar_file.extractall()
'''


'''
# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

'''


directory = "../input/"

X_train, y_train = [], []
for training_file in glob.glob("%s/data_batch_*" % directory):
    batch = unpickle(training_file) 
    X_train.append(batch[b'data'])
    y_train.append(batch[b'labels'])
    
    
X_train = np.concatenate(X_train).reshape(-1, 3, 32, 32).astype(np.float32)
y_train = np.concatenate(y_train).astype(np.int32)

test_file = unpickle("%s/test_batch" % directory)
X_test = test_file[b'data'].reshape(-1, 3, 32, 32).astype(np.float32)
y_test = np.array(test_file[b'labels'], dtype=np.int32)





classes = defaultdict(list)
for img, class_ in zip(X_train, y_train):
    classes[class_].append(img)

fig, axes = plt.subplots(10, 10, figsize=(6, 6))
for j in range(10):
    for k in range(10):
        images = classes[j]
        i = np.random.choice(range(len(images)))
        axes[j][k].set_axis_off()
        axes[j][k].imshow(np.transpose(images[i], (1, 2, 0)).astype('uint8'), 
                          interpolation='nearest')





svc = LinearSVC()



X_train[:, 0, :, :] -= 103.939
X_train[:, 1, :, :] -= 116.779
X_train[:, 2, :, :] -= 123.68

X_test[:, 0, :, :] -= 103.939
X_test[:, 1, :, :] -= 116.779
X_test[:, 2, :, :] -= 123.68



X_train = X_train.reshape(50000, 32, 32, 3)
X_test = X_test.reshape(10000, 32, 32, 3)
print(X_train.shape)
print(X_test.shape)




from keras.applications.vgg16 import VGG16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
model = Model(inputs=[base_model.input], outputs=[base_model.get_layer('block5_pool').output])

cnn_codes_train = model.predict(X_train)
cnn_codes_test = model.predict(X_test)





print(cnn_codes_train.shape)
print(cnn_codes_test.shape)





cnn_codes_train = cnn_codes_train.reshape(50000, -1)
cnn_codes_test = cnn_codes_test.reshape(10000, -1)





pca = PCA(n_components=2)
fit = pca.fit(cnn_codes_train)
cnn_codes_2 = fit.transform(cnn_codes_train)





df_cnn_codes = pd.DataFrame(cnn_codes_2)
df_cnn_codes.columns = ['PC1', 'PC2']
df_cnn_codes.head()





df_cnn_codes.plot(
    kind='scatter',
    x='PC2',
    y='PC1',
    figsize=(16,8))





cnn_codes_train = cnn_codes_train[:10000]
y_train = y_train[:10000]




svc_params = {'kernel': ['rbf'], 'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100]}
clf = GridSearchCV(estimator=SVC(), param_grid=svc_params, cv=3, n_jobs=-1, 
                   scoring='accuracy', verbose=10)
clf.fit(cnn_codes_train, y_train)





y_pred = clf.predict(cnn_codes_test)
print("For params: %s we got results: acc: %.2f, F1: %.2f" % (str(clf.best_params_), acc(y_test, y_pred), f1_score(y_test, y_pred, average='micro')))

print("Final accuracy: %f, F1: %f" % (acc(y_test, y_pred), f1_score(y_test, y_pred, average='micro')))
