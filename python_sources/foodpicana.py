# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import h5py
import skimage
import matplotlib.pyplot as plt
import sys
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#File Format
#f=h5py.File('../input/food_c101_n1000_r384x384x3.h5','r')
f=h5py.File('../input/food_c101_n10099_r64x64x3.h5','r')
print(list(f.keys()))
print(len(f["category"]))
print(len(f["category_names"]))
print(len(f["images"]))
# Print Sample Pictures
print([int(i) for i in f["category"][0]])
print(f["images"][0].shape)
fig=plt.figure(figsize=(20,20))
n=25
col=5
for i in range(n):
    ax=fig.add_subplot(n/col,col,i+1)
    #ax.set_title(f["category_names"][i].decode())
    ax.imshow(f["images"][i])
plt.savefig("./sample_show_64x64")
#sys.exit(0)
#Keras
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras import optimizers
#model = ResNet50(weights=None,input_shape=(384,384,3),classes=101)
model = VGG16(weights=None,input_shape=(64,64,3),classes=101)
x=np.array(f["images"])/255.
y=np.array([[int(i) for i in f["category"][j]] for j in range(len(f["category"]))])
#model.compile(loss='categorical_crossentropy',optimizer=optimizers.rmsprop(lr=0.0001, decay=1e-6))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.00008, beta_1=0.9, beta_2=0.97, epsilon=1e-7))
from sklearn.cross_validation import train_test_split
train_x,test_x, train_y, test_y = train_test_split(x,y,test_size = 0.2)
model.fit(train_x[:128],train_y[:128],batch_size=128,epochs=150,shuffle=False)
#print(model.evaluate(test_x,test_y))
test_x=train_x[:50]
test_y=train_y[:50]
pred_y=model.predict(test_x)
zero_y=np.zeros(pred_y.shape)
argmax_lst=np.argmax(pred_y,axis=1)
for i in range(len(argmax_lst)):
    zero_y[i][argmax_lst[i]]=1
pred_y=zero_y
from sklearn.metrics import f1_score,accuracy_score
print("Acc-Score:",accuracy_score(np.array(test_y),np.array(pred_y)))
#print("F-score:",f1_score(test_y,pred_y))

