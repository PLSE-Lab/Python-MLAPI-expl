# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import theano.gpuarray as cuda
import random
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
def normInput(x):return (x-meanPx)/stdPx
def createFCModel():
	model=Sequential([
			Lambda(normInput,input_shape=(1,28,28)),
			Flatten(),
			Dense(512,activation='softmax'),
			Dense(10,activation='softmax')])
	model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
	return model
def ts(imDatGen):#typicalSteps
	if imDatGen.n%imDatGen.batch_size==0:return imDatGen.n//imDatGen.batch_size
	else:return imDatGen.n//imDatGen.batch_size+1
def valAccDesc(model,name,maxLog,trainFlow,testFlow,saveVgt=False):
	trainMore=True
	epoch=0
	accLog=[]
	valAccLog=[]
	winner=0
	batch_size=512
	while trainMore:
		print("Running epoch: %d on %s" % (epoch,name))
		hstr=model.fit_generator(trainFlow,steps_per_epoch=ts(trainFlow),epochs=1,
							  validation_data=testFlow,validation_steps=ts(testFlow))
		latest_weights_filename = 'fc%d.h5' % epoch
		if saveVgt:model.save_weights(results_path+'\\'+latest_weights_filename)
		accLog.append(hstr.history['acc'][-1])
		valAccLog.append(hstr.history['val_acc'][-1])
		if len(valAccLog)>maxLog:
			accLog.pop(0)
			valAccLog.pop(0)
		if len(valAccLog)>=maxLog and not max(valAccLog)>valAccLog[0]:
			trainMore=False
			maxVal=max(valAccLog)
			winner=valAccLog.index(maxVal)
		epoch+=1
	print('done training %s in %d epochs. Winner is on %d with valAcc %.4f '%(name,epoch,epoch-maxLog,maxVal))
	WinnerWeightsFilename='%s%d.h5'%(name,winner)
	if saveVgt:fcModel.load_weights(results_path+'\\'+WinnerWeightsFilename)
	return[name,maxVal]
bachSize=256
ensemble=2
wait2improve=2
xTrain=np.genfromtxt('../input/train.csv', delimiter=',')
xTrain=xTrain[1:]#drop index row
random.shuffle(xTrain)
v=len(xTrain)//10#validation size
xVal=xTrain[:v]
xTrain=xTrain[v:]
yTrain=xTrain[:,0]#get Y
xTrain=xTrain[:,1:]#get X
xTrain=np.reshape(xTrain,(-1,28,28))
plt.imshow(xTrain[random.randint(0,len(xTrain))])
yVal=xVal[:,0]#get Y
xVal=xVal[:,1:]#get X
xVal=np.reshape(xVal,(-1,28,28))
plt.imshow(xVal[random.randint(0,len(xVal))])
xTrain=np.expand_dims(xTrain,1)
xVal=np.expand_dims(xVal,1)
yTrain=to_categorical(yTrain)
yVal=to_categorical(yVal)
meanPx=xTrain.mean().astype(np.float32)
stdPx=xTrain.mean().astype(np.float32)
gen=image.ImageDataGenerator()
genA=image.ImageDataGenerator(rotation_range=8,width_shift_range=.08,height_shift_range=.08,zoom_range=.08)#sensitivity test
trainFlow=genA.flow(xTrain,yTrain,batch_size=bachSize)
valFlow=gen.flow(xVal,yVal,batch_size=bachSize,shuffle=False)
xTest=np.genfromtxt('../input/test.csv', delimiter=',')
xTest=xTest[1:]#drop index row
xTest=np.reshape(xTest,(-1,28,28))
xTest=np.expand_dims(xTest,1)
testFlow=gen.flow(xTest,batch_size=bachSize,shuffle=False)
evals=[]
for i in range(ensemble):
	fc=createFCModel()
	valAccDesc(fc,'model',wait2improve,trainFlow,valFlow)
	preds=fc.predict_generator(testFlow,ts(testFlow),verbose=1)
	evals.append(preds)
evalsnp=np.array(evals)
evalsnp.mean(axis=0)
preds=np.argmax(evalsnp[0],axis=1)
subm=pd.DataFrame(preds,index=[i for i in range(1,len(preds)+1)],columns=['Label'])
submission_file_name = 'kernel%d.csv'% 1
print('writing submission %s'%submission_file_name)
subm.to_csv(submission_file_name,index_label='ImageId')
