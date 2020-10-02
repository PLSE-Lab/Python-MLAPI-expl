from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D,Dense,Dropout
from keras.models import Sequential
from keras.applications.xception import preprocess_input
import numpy as np
import pandas as pd

from sklearn.datasets import load_files
from keras.utils import np_utils
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


model = Xception(weights='../input/pesos-xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5',include_top=False)

def ler_imagem(img_path):
     img = image.load_img(img_path, target_size=(224, 224))
     x = image.img_to_array(img)
     x = np.expand_dims(x, axis=0)
     x = preprocess_input(x)
     x = model.predict(x)
     x = np.reshape(x, (7,7,2048))
     return x
 
def carregar_dataset(img_paths):
     imagens = []
     for arq in img_paths:
        imagens.append(ler_imagem(arq))         
     return np.asarray(imagens)
     

def load_dataset_dogs(path,n_class=120):
    data = load_files('../input/dogs-k5/kfold_4/' + path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), n_class)
    return (carregar_dataset(dog_files), dog_targets)
    
    
x, y = load_dataset_dogs(path = 'k4')
x1, y1 = load_dataset_dogs(path = 't4')
#x2, y2 = load_dataset_dogs(path = 'valid')

np.savez_compressed('dogs_features_train_k4',x_train = x, y_train = y)
np.savez_compressed('dogs_features_test_k4',x_test = x1, y_test = y1)
#np.savez_compressed('dogs_features_valid',x_train = x2, y_train = y2)

rede = Sequential()
rede.add(GlobalAveragePooling2D(input_shape=(7,7,2048)))
rede.add(Dense(1024,activation='relu'))
rede.add(Dropout(0.2))

rede.add(Dense(512,activation='relu'))
rede.add(Dropout(0.1))
rede.add(Dense(256,activation='relu'))
rede.add(Dense(120,activation='softmax'))
rede.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
rede.summary()
rede.fit(x,y,epochs=50, batch_size=200)

rede.save_weights('dogs_k4_model.h5')


def decode_preds(idx):
    base = pd.read_csv('../input/dicionario/dicionario.csv')
    #base = base.sort_values(['idx'])
    return base.loc[base.idx==idx]

def predict(rede, img_path):   
    img = image.load_img('../input/image/image/'+img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x) 
    x = model.predict(x)
    raca = int(rede.predict_classes(x)[0])
    perc = rede.predict_proba(x)
    perc = perc.reshape(120)
    nome_raca = decode_preds(raca).values
    return [raca,nome_raca[0][1],perc[raca]]
pred = predict(rede,img_path='cairn.jpg')
print(pred)

from sklearn.metrics import accuracy_score
predicao = rede.predict_classes(x1)
classes = [np.argmax(y, axis=None, out=None) for y in y1]
accc= accuracy_score(classes, predicao)
print(accc)
