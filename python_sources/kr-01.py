# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import keras as kr
import time
import datetime
import sys
import os.path
import random
import csv
import pickle
import pandas as pd
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

dfTraining = pd.read_csv('../input/train.csv')
#dfTraining = pd.read_csv('train.csv')
nTraining = len(dfTraining.index)
xTraining = (dfTraining.iloc[:,1:].values).astype('float32')
xTraining /= xTraining.max()
xTraining -= xTraining.mean()
yLabels = dfTraining.iloc[:,0].values.astype('int32')
yTraining=np.zeros((nTraining,10),dtype=np.bool_)
yTraining[np.arange(0,nTraining),yLabels]=1

def displayProgress(processName,i,n,timeStart,extraInfo=''):
    i+=sys.float_info.min
    dTime = round(time.time()-timeStart)
    tpi = dTime/i #time per iteration    
    ds = processName.upper() +'   {:.3f}'.format(100*i/n)+' % done\n'\
    + repr(round(i)) + ' out of ' + repr(n) + '\n'\
    + 'time elapsed:       '+str(datetime.timedelta(seconds=round(dTime)))\
    + '   time remaining: '+str(datetime.timedelta(seconds=round((n-i)*tpi)))+'\n'\
    + 'time per iteration: '+str(datetime.timedelta(seconds=round(tpi)))\
    + '   total time '+str(datetime.timedelta(seconds=round(n*tpi)))+'\n'\
    + extraInfo +'\n\n'
    return ds

def runNN(ax,ay,ab,at,an,ant,ds):
    model = kr.models.Sequential()
    model.add(kr.layers.Dense(at[0],activation='softmax',input_dim=ax.shape[1]))
    model.add(kr.layers.Dropout(0.5))
    at = at[1:]
    for it in at:
        model.add(kr.layers.Dense(it,activation='softmax'))
        model.add(kr.layers.Dropout(0.5))
    model.add(kr.layers.Dense(10,activation='sigmoid'))
    model.compile(loss='categorical_crossentropy'
                  ,optimizer='adam'
                  ,metrics=['accuracy'])
    i=0
    ts = time.time()
    while True:
        if ant == 'TIME':
            print('\x1b[J')
            ds.append(displayProgress('training individual NN',
                                      time.time()-ts,an,ts,'i: '+repr(i)))
            print(*ds)
            ds.pop()
            if time.time()-ts>=an:
                break
        elif ant == 'COUNT':
            print('\x1b[J')
            ds.append(displayProgress('training individual NN',i,an,ts,''))
            print(*ds)
            ds.pop()
            if i>=an:
                break
        else:
            raise(AttributeError('ant: '+repr(ant)))
        model.fit(ax,ay,batch_size=ab,epochs=1,verbose=0)
        i += 1
    return model.evaluate(ax,ay)[1],model
    
    

f,m=runNN(xTraining,yTraining,100,[70],3600-100,'TIME',[])
dfTesting = pd.read_csv('../input/test.csv')
#dfTesting = pd.read_csv('test.csv')
xTesting = (dfTesting.iloc[:,:].values).astype('float32')
xTesting /= xTesting.max()
xTesting -= xTesting.mean()

y=m.predict_on_batch(xTesting)

y = y.argmax(axis=1)

pd.DataFrame({"ImageId": list(range(1,len(y)+1)), "Label": y})\
.to_csv('drKR_sbmsn_t1.csv', index=False, header=True)