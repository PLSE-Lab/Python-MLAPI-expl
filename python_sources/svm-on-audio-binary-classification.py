#Libraries used
import os
import re
import numpy as np
import pandas as pd
import scipy.io.wavfile as sw
import python_speech_features as psf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#Importing directory
RootDir=os.chdir("E:\\Mission Data Scientist\\INSOFE\\Internship\\Cats Dogs Audio dataset\\cats_dogs")

file_names = os.listdir(RootDir)#list of file names
final_dataset = pd.DataFrame()#Blank initial dataset 

#Feature Extraction
for i in file_names:
    rate,signal = sw.read(i)
    features = psf.base.mfcc(signal)
    features = psf.base.fbank(features)
    features = psf.base.logfbank(features[1])
    features = psf.base.lifter(features,L=22)
    features = psf.base.delta(features,N=13)
    features = pd.DataFrame(features)
    features["Target"] = i
    final_dataset = final_dataset.append(features)#rbind(final_dataset,features)


#Correcting indexing
index = 26
for i in range(0,len(final_dataset)):
    final_dataset.iloc[i,index] = final_dataset.iloc[i,index].replace('.wav', '')
    final_dataset.iloc[i,index] = re.sub(r'[0-9]+', '',final_dataset.iloc[i,index])
    final_dataset.iloc[i,index] = final_dataset.iloc[i,index].replace('_', '')
    final_dataset.iloc[i,index] = final_dataset.iloc[i,index].replace('barking', '0')
    final_dataset.iloc[i,index] = final_dataset.iloc[i,index].replace('cat', '1')
    final_dataset.iloc[i,index] = final_dataset.iloc[i,index].replace('dog', '0')
    final_dataset.iloc[i,index] = final_dataset.iloc[i,index].replace('00', '0')

#Finalize dataset with the attributes and target
fd=final_dataset
fd = fd.rename(columns = {'y' : 'target'})
y=fd.iloc[:,-1]
X=fd.iloc[:,0:26]

#Spliting into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1)
#del y_train

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

type(y_train)
X_train=pd.DataFrame(X_train)
y_train=pd.DataFrame(y_train)
X_test=pd.DataFrame(X_test)

preprocessed_dataset = pd.DataFrame()
preprocessed_dataset = preprocessed_dataset.append(X_train,y_train)
preprocessed_dataset = preprocessed_dataset.append(X_train,X_train)


#Model Building

#SVM
model = svm.SVC(kernel = 'rbf', C = 1)
model1 = model.fit(X_train,y_train)
model1.score(X_train,y_train)
predicted=model.predict(X_test)
accuracy_score(y_test,predicted)
