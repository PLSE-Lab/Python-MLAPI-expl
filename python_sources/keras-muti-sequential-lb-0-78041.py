#!/usr/bin/env python
# coding: utf-8

# Keras multi sequential model i.e 14 models merged to main model to produce output

# In[ ]:


from sklearn import *
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.layers import Dropout,Merge
from sklearn.preprocessing import LabelEncoder
train = pd.read_csv("../input/training_variants")
test = pd.read_csv("../input/test_variants")
trainx = pd.read_csv("../input/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
testx = pd.read_csv("../input/test_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
train = pd.merge(train, trainx, how='left', on='ID')
y = train['Class'].values
train = train.drop('Class', axis=1)
test = pd.merge(test, testx, how='left', on='ID')
pid = test['ID'].values
all_data=np.concatenate((train,test),axis=0)
all_data=pd.DataFrame(all_data)
all_data.columns=['ID','Gene','Variation','Text']
sent=all_data['Text']
vect=TfidfVectorizer(stop_words='english')
sent_vectors=vect.fit_transform(sent)
svd=TruncatedSVD(200)
sent_vectors1=svd.fit_transform(sent_vectors)
sent_vectors2=svd.fit_transform(sent_vectors)
sent_vectors3=svd.fit_transform(sent_vectors)
sent_vectors4=svd.fit_transform(sent_vectors)
sent_vectors5=svd.fit_transform(sent_vectors)
sent_vectors6=svd.fit_transform(sent_vectors)
sent_vectors7=svd.fit_transform(sent_vectors)
sent_vectors8=svd.fit_transform(sent_vectors)
sent_vectors9=svd.fit_transform(sent_vectors)
sent_vectors10=svd.fit_transform(sent_vectors)
sent_vectors11=svd.fit_transform(sent_vectors)
sent_vectors12=svd.fit_transform(sent_vectors)
sent_vectors13=svd.fit_transform(sent_vectors)
sent_vectors14=svd.fit_transform(sent_vectors)
def baseline_model():
    model=Sequential()
    model.add(Dense(1024,input_dim=200,init='normal',activation='relu'))
    model.add(Dropout(0.25))
    lower_model=Sequential()
    lower_model.add(Dense(1024,input_dim=200,init='normal',activation='relu'))
    lower_model.add(Dropout(0.25))
    lower_model1=Sequential()
    lower_model1.add(Dense(1024,input_dim=200,init='normal',activation='relu'))
    lower_model1.add(Dropout(0.25))
    lower_model2=Sequential()
    lower_model2.add(Dense(1024,input_dim=200,init='normal',activation='relu'))
    lower_model2.add(Dropout(0.25))
    lower_model3=Sequential()
    lower_model3.add(Dense(1024,input_dim=200,init='normal',activation='relu'))
    lower_model3.add(Dropout(0.25))
    lower_model4=Sequential()
    lower_model4.add(Dense(1024,input_dim=200,init='normal',activation='relu'))
    lower_model4.add(Dropout(0.25))
    lower_model5=Sequential()
    lower_model5.add(Dense(1024,input_dim=200,init='normal',activation='relu'))
    lower_model5.add(Dropout(0.25))
    lower_model6=Sequential()
    lower_model6.add(Dense(1024,input_dim=200,init='normal',activation='relu'))
    lower_model6.add(Dropout(0.25))
    lower_model7=Sequential()
    lower_model7.add(Dense(1024,input_dim=200,init='normal',activation='relu'))
    lower_model7.add(Dropout(0.25))
    lower_model8=Sequential()
    lower_model8.add(Dense(1024,input_dim=200,init='normal',activation='relu'))
    lower_model8.add(Dropout(0.25))
    lower_model9=Sequential()
    lower_model9.add(Dense(1024,input_dim=200,init='normal',activation='relu'))
    lower_model9.add(Dropout(0.25))
    lower_model10=Sequential()
    lower_model10.add(Dense(1024,input_dim=200,init='normal',activation='relu'))
    lower_model10.add(Dropout(0.25))
    lower_model11=Sequential()
    lower_model11.add(Dense(1024,input_dim=200,init='normal',activation='relu'))
    lower_model11.add(Dropout(0.25))
    lower_model12=Sequential()
    lower_model12.add(Dense(1024,input_dim=200,init='normal',activation='relu'))
    lower_model12.add(Dropout(0.25))
    merged_model=Merge([model,lower_model,lower_model1,lower_model2,lower_model3,lower_model4,lower_model5,lower_model6,lower_model7,lower_model8,lower_model9,lower_model10,lower_model11,lower_model12],mode='concat')
    final_model=Sequential()
    final_model.add(merged_model)
    final_model.add(Dense(9,init='normal',activation='softmax'))
    final_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return final_model
encoder=LabelEncoder()
encoder.fit(y)
encoded_y=encoder.transform(y)
dummy_y=np_utils.to_categorical(encoded_y)
print(dummy_y.shape)


# In[ ]:


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
submission = pd.read_csv('../input/stage2_sample_submission.csv')
stage1_test = pd.read_csv('../input/test_variants')
stage2_test = pd.read_csv('../input/stage2_test_variants.csv')
stage1_solution = pd.read_csv('../input/stage1_solution_filtered.csv')

stage1_solution = stage1_solution.merge(stage1_test, how = 'left', on = 'ID')

stage2_test.merge(
        stage1_solution.drop('ID', axis = 1), 
        how = 'left', 
        on = ['Gene', 'Variation'])\
    .drop(['Gene', 'Variation'], axis = 1)\
    .fillna(1)\
    .to_csv('submission.csv', index = False)


# In[ ]:




