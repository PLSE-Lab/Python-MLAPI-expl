# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from time import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
import cv2


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

SZ=56
bin_n = 16 # Number of bins
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def unique_rows(data):
    uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])
    
t0 = time()
print("read data,resize, skew, blur and threshold it...")
data = pd.read_csv('../input/train.csv')
print("shape data:",data.shape) 
test = pd.read_csv('../input/test.csv')
print("shape test:",test.shape) 

labels = np.array(data.pop('label')) # Remove the labels as a numpy array from the dataframe

x = np.array(data,dtype = 'uint8') 
t = np.array(test,dtype = 'uint8') 

len_test_data=len(t)
len_train_data=len(x)

merged_data=np.concatenate((x,t),axis=0)


skew = []
for row in merged_data:
    image = row.reshape(28 , 28)
    rsimage = cv2.resize(image, (0,0), fx=2, fy=2)     
    skewed_image = deskew(rsimage)
    gray = cv2.GaussianBlur(skewed_image, (3, 3), 0)
    thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,0)  
    skew.append(thresh.flatten())

newdata = np.array(skew ).astype(int)

print ("read data, skew, blur and threshold done in [sec]: %0.3fs" % (time() - t0))
newdata[newdata>0]=1


t0 = time()
print ("data preperation starts...")

ohe = OneHotEncoder()
labels3 = np.array(ohe.fit_transform(labels.reshape(-1, 1)).todense(),dtype = 'uint8') 

testlabels_concat=np.append(labels3,labels[:, np.newaxis], axis=1)
labels_concat_unique=unique_rows(testlabels_concat)
print("shape labels_concat_unique: ",labels_concat_unique.shape)
print ("data preperation takes[sec]: %0.3fs" % (time() - t0))



a = []
for i in list(range(len_train_data)):      
    pred_rev=-1
    pred=labels3[i,:]
    for row in labels_concat_unique:
        if np.allclose(row[:10],pred)==True:
            pred_rev=row[10]
            a.append(pred_rev)
bn=np.array(a)
print("shape reversed labels:",bn.shape)            

if np.allclose(bn,labels)==True:
    print("reversing succesfull")
