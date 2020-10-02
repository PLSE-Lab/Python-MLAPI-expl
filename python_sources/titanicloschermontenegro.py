#!/usr/bin/env python
# coding: utf-8

# Juan Montenegro
# 
# Ivan Loscher

# In[ ]:


import numpy as np
import pandas as pd 
from keras.layers import Input, Dense, BatchNormalization, Add, GaussianNoise, Dropout, Activation
from keras.models import Model, Sequential
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Wrapper
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras import regularizers
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import os
print(os.listdir("../input"))
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

file_train=pd.read_csv('/kaggle/input/titanic/train.csv')
file_test=pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


#auxiliares
precisiones_globales=[]
epochs = 50
def precision(model, registrar=False):
    y_pred = model.predict(train_dfX)
    train_auc = roc_auc_score(train_dfY, y_pred)
    y_pred = model.predict(val_dfX)
    val_auc = roc_auc_score(val_dfY, y_pred)
    print('Train AUC: ', train_auc)
    print('Vali AUC: ', val_auc)
    if registrar:
        precisiones_globales.append([train_auc,val_auc])


# In[ ]:


train_df_raw = file_train
train_df_raw.head()
#vemos que ticket, fare, cabin no son importantes
#vemos que sex es male o female, hay que convertirlo a 1 y 0
#eso en realidad me parece algo complicado, preferiria lo siguiente, vamos a droppear name, fare, ticket, cabin, embarked y no se si age porque salven a los carajitos
#SibSp es si tiene familia, o sea sibling spouse y parch es parent children


# In[ ]:


test_df_raw = file_test
test_df_raw.head()
#aqui vemos que esta el id de pasajero, name, ticket, fare, cabin y embarked de nuevo, podemos droppearlos tambien supongo


# Preprocesamiento de datos

# In[ ]:


#entonces vamos a hacer esos cambios
#mantenemos los viejos no modificados
train = file_train
test = file_test
#ojo, run all y one hot encoding para female y male
file_train = file_train.replace(["male", "female"], [0,1]) 
file_train = file_train.drop(['Name','Ticket','Fare','Cabin','Embarked'],axis=1)
file_train = file_train.fillna(0)
train_df_raw = file_train
train_df_raw.head()


# In[ ]:


file_test = file_test.replace(["male", "female"], [0,1])
file_test = file_test.drop(['Name','Ticket','Fare','Cabin','Embarked'], axis=1)
file_test = file_test.fillna(0)
test_df_raw = file_test
test_df_raw.head()
#y ya con estos cambios creo que podemos empezar a hacer el modelo


# In[ ]:


#variables X y Y con las que se van a entrenar al modelo
train_dfY = file_train["Survived"]
train_dfX = file_train[["PassengerId","Pclass","Sex","Age","SibSp","Parch"]]
test_df = file_test[["PassengerId","Pclass","Sex","Age","SibSp","Parch"]]
submission = test_df[['PassengerId']].copy()
#normalizando valores
sc = StandardScaler()
train_dfX = sc.fit_transform(train_dfX)
test_df = sc.transform(test_df)


# In[ ]:


#separando entrenamiento de validacion
train_dfX, val_dfX,train_dfY, val_dfY = train_test_split(train_dfX,train_dfY , test_size=0.1, stratify=train_dfY)
print("Entrenamiento: ",train_dfX.shape)
print("Validacion : ",val_dfX.shape)


# Arquitectura del modelo

# In[ ]:


#creamos el modelo
#el modelo posee 6 capas desde el input layer hasta el output final, el input layer posee 150 nodos, la capa 1 100, la 2 tiene 50, la 3 tiene 10 la 4 tiene 2 y el output tiene 1.
model = Sequential()
#input
model.add(Dense(150, input_shape = (6,))) #6 por entramiento y validacion
model.add(Activation("relu"))
#el input layer y el hidden layer 1 utilizan relu como funcion de activacion y las demas utilizan entre relu y sigmoide, a medida que baja la cantidad, se usa sigmoide.
#escondidas
model.add(Dense(100, input_shape = (150,)))
model.add(Activation("relu"))
model.add(Dense(50, input_shape = (100,)))
model.add(Activation("sigmoid"))
model.add(Dense(10, input_shape = (50,)))
model.add(Activation("sigmoid"))
model.add(Dense(2, input_shape = (10,)))
model.add(Activation("sigmoid"))
#nos fuimos guiando con otros modelos y resultados previos para ayudar a determinar un numero de capas y nodos que fuese optimo
#output
model.add(Dense(1, input_shape = (2,)))
model.add(Activation("sigmoid"))
#ademas los resultados obtenidos nos parecieron suficientemente buenos como para no tener que regularizar.
#compile para clasificacion
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


# Entrenamiento del modelo

# In[ ]:


#entrenamiento del modelo
 train_history = model.fit(train_dfX, train_dfY, batch_size=50, epochs=epochs, validation_data=(val_dfX, val_dfY))
#estos son los valores que mejor resultado dieron


# Resultado de modelo

# In[ ]:


precision(model,True)
#You might need a way of handling missing values, such as pandas.DataFrame.fillna or sklearn.preprocessing.Imputer. See our Missing Values tutorial for more details.
#creo que por eso daba loss: nan


# In[ ]:


y_test = np.round(model.predict(test_df))
y_test = y_test.astype(np.int32)
submission['Survived'] = y_test
submission = pd.DataFrame(submission)
submission.to_csv('submission.csv', index=False)

