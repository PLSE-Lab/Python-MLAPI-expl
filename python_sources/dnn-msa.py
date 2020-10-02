#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 11:34:55 2019

@author: victorcumer
"""

import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.data import loadlocal_mnist
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.

"""
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        RBM OBJECT       RBM OBJECT      RBM OBJECT
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
""" 
class RBM_np():
    def __init__(self, p, q):
        self.w = np.zeros((p, q), dtype=np.float32)
        self.a = np.zeros(p, dtype=np.float32)
        self.b = np.zeros(q, dtype=np.float32)
     

"""
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        ALPHADIGIT FUNCTIONS       ALPHADIGIT FUNCTIONS      
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
"""   
def lire_alpha_digit(file):
    X = pd.DataFrame(scipy.io.loadmat(file)['dat'])   
    IMG = pd.DataFrame(np.zeros((1404, 20*16), dtype=np.int8))
    ligne = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            IMG.iloc[ligne,:] = np.concatenate([X.iloc[i,j][k] for k in range(20)])
            ligne += 1
    return IMG
    
    
    
"""
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        RBM FUNCTIONS       RBM FUNCTIONS      RBM FUNCTIONS  
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
"""
def init_RBM_np(p, q):
    rbm = RBM_np(p, q)
    rbm.w = np.random.normal(loc = 0, scale = 0.1, size = rbm.w.shape) # scale is the standard deviation
    return rbm


def entree_sortie_np(rbm, X):
    B = np.array([rbm.b]*X.shape[0])
    return 1/(1+np.exp(-np.dot(X, rbm.w) - B))


def sortie_entree_np(rbm, H):
    A = np.array([rbm.a]*H.shape[0])
    return 1/(1+np.exp(-np.dot(H, rbm.w.transpose()) - A))  


def train_RBM_np(X, rbm, batch_size, nb_iter, lr):

    EQM = []
    for i in range(nb_iter):
        if type(X)!='pandas.core.frame.DataFrame':
            X = pd.DataFrame(X)
        X_shuffled = X.sample(frac=1).reset_index(drop=True).values
        for j in range(0, X_shuffled.shape[0], batch_size):

            v = X_shuffled[j:min(j+batch_size, X_shuffled.shape[0]),:]

            phv = entree_sortie_np(rbm, v)

            h = np.random.random(phv.shape)<=phv

            pvh = sortie_entree_np(rbm, h)
            v1 = np.random.random(pvh.shape)<=pvh
            phv1 = entree_sortie_np(rbm, v1)

            dw = (np.dot(v.transpose(), phv) - np.dot(v1.transpose(), phv1))/v.shape[0]

            da = (v.sum(0) - v1.sum(0))/v.shape[0]
            db = (phv.sum(0) - phv1.sum(0))/v.shape[0]

            rbm.w = rbm.w + lr*dw
            rbm.a = rbm.a + lr*da
            rbm.b = rbm.b + lr*db

        
        # affichage de l'erreur quadratique
        sortie = entree_sortie_np(rbm, X.values)
        new_entree = sortie_entree_np(rbm, sortie)
        erreur = np.mean(np.mean(X.values - new_entree)**2)
        EQM.append(erreur)
    plt.plot(range(nb_iter), EQM)
    plt.legend(['EQM'])
    plt.title("évolution de l'EQM en fonction du nombre d'itérations (pré-entrainement)")
    return rbm

def generer_img(rbm, nb_gibbs, nb_images, shape):
    for i in range(nb_images):
        tirage = np.random.rand(1, p)
        image = (tirage < 0.5)
        for n in range(nb_gibbs):
            proba = entree_sortie_np(rbm, image)
            tirage = np.random.rand(proba.shape[0], proba.shape[1])
            sortie = (tirage < proba)
            proba = sortie_entree_np(rbm, sortie)
            tirage = np.random.rand(proba.shape[0], proba.shape[1])
            image = (tirage < proba)
        # afficher image 
        image_rsh = np.reshape(image, shape)
        plt.figure()
        plt.imshow(image_rsh)


"""
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        DBN FUNCTIONS       DBN FUNCTIONS      DBN FUNCTIONS  
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
"""
        
def init_DBN(nb_couches, neurones):     # neurones = [(p1, q1), (p2, q2), ...]
    dbn = []
    for i in range(nb_couches):
        dbn.append(init_RBM_np(neurones[i][0], neurones[i][1]))
    return dbn

def train_DBN(X, dbn, batch_size, nb_iter, lr, neurones):
    dnn = []
    donnees = X
    for i in range(len(dbn)):
        dnn.append(train_RBM_np(donnees, dbn[i], batch_size, nb_iter, lr))
        donnees = entree_sortie_np(dnn[i], donnees)
        plt.legend(neurones)
        print(i)
    return dnn

def generer_img_DBN(dnn, nb_gibbs, nb_images, shape):
    for i in range(nb_images):
        tirage = np.random.rand(1, dnn[len(dnn)-1].w.shape[0])
        image = (tirage < 0.5)
        
        for n in range(nb_gibbs):
            proba = entree_sortie_np(dnn[len(dnn)-1], image)
            tirage = np.random.rand(proba.shape[0], proba.shape[1])
            sortie = (tirage < proba)
            proba = sortie_entree_np(dnn[len(dnn)-1], sortie)
            tirage = np.random.rand(proba.shape[0], proba.shape[1])
            image = (tirage < proba)
        
        for l in range(len(dnn)-2, -1, -1):
            proba = sortie_entree_np(dnn[l], image)
            tirage = np.random.rand(proba.shape[0], proba.shape[1])
            image = (tirage < proba)
        # afficher image 
        image_rsh = np.reshape(image, shape)
        plt.figure()
        plt.imshow(image_rsh)


"""
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        DNN FUNCTIONS       DNN FUNCTIONS      DNN FUNCTIONS  
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
"""

def calcul_softmax(rbm, X):
    B = np.array([rbm.b]*X.shape[0])
    numerat = np.exp(np.dot(X, rbm.w) + B)
    denum = numerat.sum(1)
    ploud = np.zeros(numerat.shape)
    for i in range(ploud.shape[1]):
        ploud[:,i] = denum
    return numerat/ploud


def entree_sortie_reseau(dnn, X):
    sorties_couches = []
    sorties_couches.append(entree_sortie_np(dnn[0], X))
    for i in range(1, len(dnn)-1):
        sorties_couches.append(entree_sortie_np(dnn[i], sorties_couches[-1]))
    sorties_couches.append(calcul_softmax(dnn[-1], sorties_couches[-1]))
    return sorties_couches
    
def copy_dnn(dnn):
    new_dnn = init_DBN(len(dnn), [dnn[i].w.shape for i in range(len(dnn))])
    for i in range(len(dnn)):
        new_dnn[i].a = dnn[i].a.copy()
        new_dnn[i].b = dnn[i].b.copy()
        new_dnn[i].w = dnn[i].w.copy()
    return new_dnn

def retropropagation(dnn, X_train, Y_train, nb_iter, lr, batch_size, pre_trained):
    if type(X_train) == 'pandas.core.frame.DataFrame':
            X_train = X_train.values
    entrop_crois = []
    for ploud in range(nb_iter):
        indices = np.arange(0,X_train.shape[0],1)
        np.random.shuffle(indices)
        for j in range(0, X_train.shape[0], batch_size):
            new_dnn = copy_dnn(dnn)
            batch_ind = indices[j:min(j+batch_size, X_train.shape[0])]
            data = X_train[batch_ind,:]
            sorties_couches = entree_sortie_reseau(dnn, data)
            
            ## début dernière couche
            matrice_c = sorties_couches[-1] - Y_train[batch_ind]
            der_w = np.dot(sorties_couches[len(dnn)-2].transpose(), matrice_c)/data.shape[0]
            der_b = matrice_c.sum(0)/data.shape[0]
            new_dnn[-1].w = new_dnn[len(dnn)-1].w - lr*der_w #/batch
            new_dnn[-1].b = new_dnn[len(dnn)-1].b - lr*der_b
            ## fin dernière couche
            
            for couche in range(len(dnn)-2, -1, -1):
                
                if couche == 0:
                    inpute = data
                else:
                    inpute = sorties_couches[couche-1]
                
                h_mult = sorties_couches[couche]*(1-sorties_couches[couche])
                matrice_c = np.dot(matrice_c, dnn[couche+1].w.transpose())*h_mult                
                der_w = np.dot(inpute.transpose(), matrice_c)/data.shape[0]
                der_b = matrice_c.sum(0)/data.shape[0]
                new_dnn[couche].w = new_dnn[couche].w - lr*der_w
                new_dnn[couche].b = new_dnn[couche].b - lr*der_b
            
            dnn = copy_dnn(new_dnn)
        sorties_couches = entree_sortie_reseau(dnn, X_train)
        classif = -np.log10(sorties_couches[-1])[Y_train==1]
        erreur = classif.sum()
        entrop_crois.append(erreur)
    f = plt.figure(figsize=(10, 7))
    plt.plot(range(nb_iter), entrop_crois)
    plt.legend(['Entropie croisée'])
    plt.title("Évolution de l'entropie croisée au cours des iterations")
    plt.xlabel("nombre d'itérations")
    plt.ylabel('entropie croisée')
    f.savefig('retropropagation_{}.png'.format(pre_trained))
    return dnn
            
def test_DNN(my_dnn, X_test, Y_test, pre_trained):
    Y_pred = entree_sortie_reseau(my_dnn, X_test)[-1]
    classif = -np.log10(Y_pred)[Y_test==1]
    erreur = classif.sum()
    print("entropie croisée :", erreur)
    for i in range(Y_pred.shape[0]):
        for j in range(Y_pred.shape[1]):
            if Y_pred[i,j] == max(Y_pred[i,:]):
                Y_pred[i,j] = 1
            else:
                Y_pred[i,j] = 0
    print(accuracy_score(Y_test, Y_pred))
    matrice_de_confusion(Y_test, Y_pred, erreur, pre_trained)
    
def accuracy_score(y_test, y_pred):
    result = (y_test != y_pred)
    return "Accuracy : {}%".format(round((1 - (result.sum()//2)/y_test.shape[0])*100*100)/100)
    
def matrice_de_confusion(y_test, y_pred, erreur, pre_trained):
    y_t = []
    y_p = []
    for i in range(y_test.shape[0]):
        for j in range(y_test.shape[1]):
            if y_test[i,j] == 1:
                y_t.append(j)
            if y_pred[i,j] == 1:
                y_p.append(j)
    img = plt.figure(figsize=(7, 7))
    plt.imshow(confusion_matrix(y_t, y_p))  
    plt.title("MATRICE DE CONFUSION {} \n {}\n Entropie croisée : {}".format(pre_trained, accuracy_score(y_test, y_pred), round(erreur*100)/100))
    img.savefig('confusion_mat_{}.png'.format(pre_trained))

"""
######################################################################################################
###     ALPHADIGITS    ALPHADIGITS    ALPHADIGITS    ALPHADIGITS    ALPHADIGITS    ALPHADIGITS
######################################################################################################
"""
"""
filepath = '../input/data-dnn/binaryalphadigs.mat'
X = lire_alpha_digit(filepath)
p = X.shape[1]

image_x = 20
image_y = 16
shape = (image_x, image_y)
        
neurones = [(p, 50),(50, 20),(20, 10)]
nb_couches = len(neurones)

m = 10
X_train = X.iloc[10*39:(10+m)*39,:]
Y_train = np.zeros((39*m, 10))
for i in range(10):
    Y_train[39*i:39*(i+1),i] = 1

nb_iter = 150
lr = 0.05
batch_size = 10

indices_train = []
indices_test = []

for i in range(10):
    indices_train += range(39*i,39*(i+1)-10,1)
    indices_test += range(39*(i+1)-10, 39*(i+1),1)
    
data_x_train = X_train.values[np.array(indices_train),:]
data_y_train = Y_train[indices_train, :]
data_x_test = X_train.values[indices_test, :]
data_y_test = Y_train[indices_test, :]
        
dbn = init_DBN(nb_couches, neurones)
dnn = retropropagation(dbn, data_x_train, data_y_train, nb_iter, lr, batch_size, "normal")
test_DNN(dnn, data_x_test, data_y_test)

#michel = init_DBN(nb_couches, neurones)
#michelle = train_DBN(data_x_train, michel, batch_size, nb_iter, lr, neurones)
#claude = retropropagation(michelle, data_x_train, data_y_train, nb_iter, lr, batch_size, "pre_trained")
#test_DNN(claude, data_x_test, data_y_test)
"""
    
"""
######################################################################################################
###     MNIST    MNIST    MNIST    MNIST    MNIST    MNIST    MNIST    MNIST    MNIST    MNIST
######################################################################################################
"""
def tranform_to_dummy(matrice):
    dummy_matrice = np.zeros((matrice.shape[0], max(matrice)+1))
    for i in range(len(matrice)):
        dummy_matrice[i,matrice[i]]=1
    return dummy_matrice

X, y = loadlocal_mnist(
        images_path='../input/train-images.idx3-ubyte', 
        labels_path='../input/train-labels.idx1-ubyte')

X = np.array(X>=100, dtype=np.int8)
y = tranform_to_dummy(y)

data_x_test, data_y_test = loadlocal_mnist(
        images_path='../input/t10k-images.idx3-ubyte', 
        labels_path='../input/t10k-labels.idx1-ubyte')
data_x_test = np.array(data_x_test>=100, dtype=np.int8)    
data_y_test = tranform_to_dummy(data_y_test)

"""
taille_image = int(np.sqrt(X.shape[1]))
shape = (taille_image, taille_image)
        

data_x_train = X[:900,:]
data_y_train = y[:900,:]
data_x_test = X[900:1000,:]
data_y_test = y[900:1000,:]

p = X.shape[1]

nb_iter_retro = 200
nb_iter_rbm = 200
lr = 0.1
batch_size = 100

neurones = [(p, 360),(360, 100),(100, 10)]

nb_couches = len(neurones)
        
dbn = init_DBN(nb_couches, neurones)
dnn = retropropagation(dbn, data_x_train, data_y_train, nb_iter, lr, batch_size, "normal")
test_DNN(dnn, data_x_test, data_y_test)
"""

"""
######################################################################################################
###     ANALYSE    ANALYSE    ANALYSE    ANALYSE    ANALYSE    ANALYSE    ANALYSE    ANALYSE
######################################################################################################
"""

##### EN FONCTION DU NOMBRE DE COUCHES
"""
# on travaille avec 3000 données d'entrainement 
data_x_train, x_dont_care, data_y_train, y_dont_care = train_test_split(X, y, train_size=0.05, random_state=42)


p = data_x_train.shape[1]
nb_iter_retro = 200
nb_iter_rbm = 100
lr = 0.1
batch_size = 150

for i in range(7):
    neurones = []
    neurones.append((p, 200))
    for j in range(i):
        neurones.append((200, 200))
    neurones.append((200, 10))
    
    print(neurones)
    
    dbn_alone = init_DBN(len(neurones), neurones)
    dbn_pretrained = copy_dnn(dbn_alone)
    
    dnn_alone = retropropagation(dbn_alone, data_x_train, data_y_train, nb_iter_retro, lr, batch_size, "pre-trained{}".format(i))
    test_DNN(dnn_alone, data_x_test, data_y_test, "normal{}".format(i))
    
    dnn_pretrained = train_DBN(data_x_train, dbn_pretrained, batch_size, nb_iter_rbm, lr, neurones)
    dnn_pretrained_trained = retropropagation(dbn_pretrained, data_x_train, data_y_train, nb_iter_retro, lr, batch_size, "pre-trained{}".format(i))
    test_DNN(dnn_pretrained_trained, data_x_test, data_y_test, "pre-trained{}".format(i))

"""

##### EN FONCTION DU NOMBRE DE NEURONES PAR COUCHE
"""
# on travaille avec 3000 données d'entrainement
data_x_train, x_dont_care, data_y_train, y_dont_care = train_test_split(X, y, train_size=0.05, random_state=42)

p = data_x_train.shape[1]
nb_iter_retro = 200
nb_iter_rbm = 100
lr = 0.1
batch_size = 150

for i in range(1,8):
    neurones = [(p, 100*i), (100*i, 100*i), (100*i, 10)]
    print(neurones)
    
    dbn_alone = init_DBN(len(neurones), neurones)
    dbn_pretrained = copy_dnn(dbn_alone)
    
    dnn_alone = retropropagation(dbn_alone, data_x_train, data_y_train, nb_iter_retro, lr, batch_size, "normal{}".format(i))
    test_DNN(dnn_alone, data_x_test, data_y_test, "normal{}".format(i))
    
    dnn_pretrained = train_DBN(data_x_train, dbn_pretrained, batch_size, nb_iter_rbm, lr, neurones)
    dnn_pretrained_trained = retropropagation(dnn_pretrained, data_x_train, data_y_train, nb_iter_retro, lr, batch_size, "pre-trained{}".format(i))
    test_DNN(dnn_pretrained_trained, data_x_test, data_y_test, "pre-trained{}".format(i))
"""


##### EN FONCTION DU NOMBRE DE DONNÉES TRAIN

"""
nb_train_data = [round(1000*100/60000)/100, round(100*3000/60000)/100, round(100*7000/60000)/100, round(100*15000/60000)/100, round(100*30000/60000)/100, round(100*45000/60000)/100, 1]

for i in range(len(nb_train_data)):
    
    data_x_train, x_dont_care, data_y_train, y_dont_care = train_test_split(X, y, train_size=nb_train_data[i], random_state=42)
    
    p = data_x_train.shape[1]
    nb_iter_retro = 200
    nb_iter_rbm = 100
    lr = 0.1
    batch_size = 150
    
    neurones = [(p, 200), (200, 200), (200, 10)]
    print(neurones)
    print(len(data_x_train))
    
    dbn_alone = init_DBN(len(neurones), neurones)
    dbn_pretrained = copy_dnn(dbn_alone)
    
    dnn_alone = retropropagation(dbn_alone, data_x_train, data_y_train, nb_iter_retro, lr, batch_size, "normal{}".format(i))
    test_DNN(dnn_alone, data_x_test, data_y_test, "normal{}".format(i))
    
    dnn_pretrained = train_DBN(data_x_train, dbn_pretrained, batch_size, nb_iter_rbm, lr, neurones)
    dnn_pretrained_trained = retropropagation(dnn_pretrained, data_x_train, data_y_train, nb_iter_retro, lr, batch_size, "pre-trained{}".format(i))
    test_DNN(dnn_pretrained_trained, data_x_test, data_y_test, "pre-trained{}".format(i))
"""

"""
data_x_train, x_dont_care, data_y_train, y_dont_care = train_test_split(X, y, train_size=0.75, random_state=42)
#data_x_train = X
#data_y_train = y
i = "45000"

p = data_x_train.shape[1]
nb_iter_retro = 200
nb_iter_rbm = 100
lr = 0.1
batch_size = 150

neurones = [(p, 200), (200, 200), (200, 10)]
print(neurones)
print(len(data_x_train))

dbn_alone = init_DBN(len(neurones), neurones)
dbn_pretrained = copy_dnn(dbn_alone)

dnn_alone = retropropagation(dbn_alone, data_x_train, data_y_train, nb_iter_retro, lr, batch_size, "normal{}".format(i))
test_DNN(dnn_alone, data_x_test, data_y_test, "normal{}".format(i))

dnn_pretrained = train_DBN(data_x_train, dbn_pretrained, batch_size, nb_iter_rbm, lr, neurones)
dnn_pretrained_trained = retropropagation(dnn_pretrained, data_x_train, data_y_train, nb_iter_retro, lr, batch_size, "pre-trained{}".format(i))
test_DNN(dnn_pretrained_trained, data_x_test, data_y_test, "pre-trained{}".format(i))
"""

##### HYPERPARAMÈTRES

##### BATCH_SIZE
"""
# on travaille avec 3000 données d'entrainement 
data_x_train, x_dont_care, data_y_train, y_dont_care = train_test_split(X, y, train_size=0.05, random_state=42)


p = data_x_train.shape[1]
nb_iter_retro = 200
nb_iter_rbm = 100
lr = 0.1
batch_size = [50, 100, 150, 200, 250]

for i in range(len(batch_size)):
    neurones = [(p, 200), (200, 200), (200, 10)]
    print(neurones)
    print("nb_train_data :", data_x_train.shape)
    print(batch_size[i])
    dbn_alone = init_DBN(len(neurones), neurones)
    dbn_pretrained = copy_dnn(dbn_alone)
    
    dnn_alone = retropropagation(dbn_alone, data_x_train, data_y_train, nb_iter_retro, lr, batch_size[i], "pre-trained_bs{}".format(batch_size[i]))
    test_DNN(dnn_alone, data_x_test, data_y_test, "normal_bs{}".format(batch_size[i]))
    
    dnn_pretrained = train_DBN(data_x_train, dbn_pretrained, batch_size[i], nb_iter_rbm, lr, neurones)
    dnn_pretrained_trained = retropropagation(dnn_pretrained, data_x_train, data_y_train, nb_iter_retro, lr, batch_size[i], "pre-trained_bs{}".format(batch_size[i]))
    test_DNN(dnn_pretrained_trained, data_x_test, data_y_test, "pre-trained_bs{}".format(batch_size[i]))
"""
##### LEARNING RATE

"""
# on travaille avec 3000 données d'entrainement 
data_x_train, x_dont_care, data_y_train, y_dont_care = train_test_split(X, y, train_size=0.05, random_state=42)


p = data_x_train.shape[1]
nb_iter_retro = 200
nb_iter_rbm = 100
lr = [0.01, 0.03, 0.06, 0.1, 0.13, 0.16]
batch_size = 150

for i in range(len(lr)):
    neurones = [(p, 200), (200, 200), (200, 10)]
    print(neurones)
    print("nb_train_data :", data_x_train.shape)
    print(lr[i])
    dbn_alone = init_DBN(len(neurones), neurones)
    dbn_pretrained = copy_dnn(dbn_alone)
    
    dnn_alone = retropropagation(dbn_alone, data_x_train, data_y_train, nb_iter_retro, lr[i], batch_size, "pre-trained_lr{}".format(lr[i]))
    test_DNN(dnn_alone, data_x_test, data_y_test, "normal_lr{}".format(lr[i]))
    
    dnn_pretrained = train_DBN(data_x_train, dbn_pretrained, batch_size, nb_iter_rbm, lr[i], neurones)
    dnn_pretrained_trained = retropropagation(dnn_pretrained, data_x_train, data_y_train, nb_iter_retro, lr[i], batch_size, "pre-trained_lr{}".format(lr[i]))
    test_DNN(dnn_pretrained_trained, data_x_test, data_y_test, "pre-trained_lr{}".format(lr[i]))

"""

##### LEARNING RATE

"""
# on travaille avec 3000 données d'entrainement 
data_x_train, x_dont_care, data_y_train, y_dont_care = train_test_split(X, y, train_size=0.05, random_state=42)


p = data_x_train.shape[1]
nb_iter_retro = [100, 150, 200, 250, 300]
nb_iter_rbm = 100
lr = 0.1
batch_size = 150

for i in range(len(nb_iter_retro)):
    neurones = [(p, 200), (200, 200), (200, 10)]
    print(neurones)
    print("nb_train_data :", data_x_train.shape)
    print(nb_iter_retro[i])
    dbn_alone = init_DBN(len(neurones), neurones)
    dbn_pretrained = copy_dnn(dbn_alone)
    
    dnn_alone = retropropagation(dbn_alone, data_x_train, data_y_train, nb_iter_retro[i], lr, batch_size, "pre-trained_epoch{}".format(nb_iter_retro[i]))
    test_DNN(dnn_alone, data_x_test, data_y_test, "normal_epoch{}".format(nb_iter_retro[i]))
    
    dnn_pretrained = train_DBN(data_x_train, dbn_pretrained, batch_size, nb_iter_rbm, lr, neurones)
    dnn_pretrained_trained = retropropagation(dnn_pretrained, data_x_train, data_y_train, nb_iter_retro[i], lr, batch_size, "pre-trained_epoch{}".format(nb_iter_retro[i]))
    test_DNN(dnn_pretrained_trained, data_x_test, data_y_test, "pre-trained_epoch{}".format(nb_iter_retro[i]))

"""

##### BEST CONFIG PRE_TRAINED
"""
train_size_pretrained = round(100*45000/60000)/100
data_x_train, x_dont_care, data_y_train, y_dont_care = train_test_split(X, y, train_size=train_size_pretrained, random_state=42)


p = data_x_train.shape[1]
nb_iter_retro = 300
nb_iter_rbm = 100
lr = 0.06
batch_size = 200


neurones = [(p, 500), (500, 400), (400, 300), (300, 200), (200, 200), (200, 100), (100, 10)]
print(neurones)
print("nb_train_data :", data_x_train.shape)

dbn_pretrained = init_DBN(len(neurones), neurones)

dnn_pretrained = train_DBN(data_x_train, dbn_pretrained, batch_size, nb_iter_rbm, lr, neurones)
dnn_pretrained_trained = retropropagation(dnn_pretrained, data_x_train, data_y_train, nb_iter_retro, lr, batch_size, "pre-trained_best")
test_DNN(dnn_pretrained_trained, data_x_test, data_y_test, "pre-trained_best")
"""

        
##### BEST CONFIG NORMAL
"""
p = X.shape[1]
nb_iter_retro = 250
nb_iter_rbm = 100
lr = 0.16
batch_size = 50

neurones = [(p, 500), (500, 400), (400, 10)]
print(neurones)
print("nb_train_data :", X.shape)

dbn = init_DBN(len(neurones), neurones)

dnn = retropropagation(dbn, X, y, nb_iter_retro, lr, batch_size, "normal_best")
test_DNN(dnn, data_x_test, data_y_test, "normal_best")
"""

"""
######################################################################################################
###     ANALYSE    ANALYSE    ANALYSE    ANALYSE    ANALYSE    ANALYSE    ANALYSE    ANALYSE
######################################################################################################
"""

##### EN FONCTION DU NOMBRE DE COUCHES

def comparaison_par_nb_couches():
    # on travaille avec 3000 données d'entrainement 
    data_x_train, x_dont_care, data_y_train, y_dont_care = train_test_split(X, y, train_size=0.05, random_state=42)
    
    
    p = data_x_train.shape[1]
    nb_iter_retro = 200
    nb_iter_rbm = 100
    lr = 0.1
    batch_size = 150
    
    for i in range(7):
        neurones = []
        neurones.append((p, 200))
        for j in range(i):
            neurones.append((200, 200))
        neurones.append((200, 10))
        
        print(neurones)
        
        dbn_alone = init_DBN(len(neurones), neurones)
        dbn_pretrained = copy_dnn(dbn_alone)
        
        dnn_alone = retropropagation(dbn_alone, data_x_train, data_y_train, nb_iter_retro, lr, batch_size, "pre-trained{}".format(i))
        test_DNN(dnn_alone, data_x_test, data_y_test, "normal{}".format(i))
        
        dnn_pretrained = train_DBN(data_x_train, dbn_pretrained, batch_size, nb_iter_rbm, lr, neurones)
        dnn_pretrained_trained = retropropagation(dnn_pretrained, data_x_train, data_y_train, nb_iter_retro, lr, batch_size, "pre-trained{}".format(i))
        test_DNN(dnn_pretrained_trained, data_x_test, data_y_test, "pre-trained{}".format(i))
    


##### EN FONCTION DU NOMBRE DE NEURONES PAR COUCHE
def comparaison_par_nb_neurones():
    # on travaille avec 3000 données d'entrainement
    data_x_train, x_dont_care, data_y_train, y_dont_care = train_test_split(X, y, train_size=0.05, random_state=42)
    
    p = data_x_train.shape[1]
    nb_iter_retro = 200
    nb_iter_rbm = 100
    lr = 0.1
    batch_size = 150
    
    for i in range(1,8):
        neurones = [(p, 100*i), (100*i, 100*i), (100*i, 10)]
        print(neurones)
        
        dbn_alone = init_DBN(len(neurones), neurones)
        dbn_pretrained = copy_dnn(dbn_alone)
        
        dnn_alone = retropropagation(dbn_alone, data_x_train, data_y_train, nb_iter_retro, lr, batch_size, "normal{}".format(i))
        test_DNN(dnn_alone, data_x_test, data_y_test, "normal{}".format(i))
        
        dnn_pretrained = train_DBN(data_x_train, dbn_pretrained, batch_size, nb_iter_rbm, lr, neurones)
        dnn_pretrained_trained = retropropagation(dnn_pretrained, data_x_train, data_y_train, nb_iter_retro, lr, batch_size, "pre-trained{}".format(i))
        test_DNN(dnn_pretrained_trained, data_x_test, data_y_test, "pre-trained{}".format(i))
    


##### EN FONCTION DU NOMBRE DE DONNÉES TRAIN

def comparaison_par_nb_train():
    nb_train_data = [round(1000*100/60000)/100, round(100*3000/60000)/100, round(100*10000/60000)/100, round(100*30000/60000)/100, round(100*45000/60000)/100, 1]
    
    for i in range(len(nb_train_data)):
        
        data_x_train, x_dont_care, data_y_train, y_dont_care = train_test_split(X, y, train_size=nb_train_data[i], random_state=42)
        
        p = data_x_train.shape[1]
        nb_iter_retro = 200
        nb_iter_rbm = 100
        lr = 0.1
        batch_size = 150
        
        neurones = [(p, 200), (200, 200), (200, 10)]
        print(neurones)
        print(len(data_x_train))
        
        dbn_alone = init_DBN(len(neurones), neurones)
        dbn_pretrained = copy_dnn(dbn_alone)
        
        dnn_alone = retropropagation(dbn_alone, data_x_train, data_y_train, nb_iter_retro, lr, batch_size, "normal{}".format(i))
        test_DNN(dnn_alone, data_x_test, data_y_test, "normal{}".format(i))
        
        dnn_pretrained = train_DBN(data_x_train, dbn_pretrained, batch_size, nb_iter_rbm, lr, neurones)
        dnn_pretrained_trained = retropropagation(dnn_pretrained, data_x_train, data_y_train, nb_iter_retro, lr, batch_size, "pre-trained{}".format(i))
        test_DNN(dnn_pretrained_trained, data_x_test, data_y_test, "pre-trained{}".format(i))


##### HYPERPARAMÈTRES

##### BATCH_SIZE
def test_batch_size():
    # on travaille avec 3000 données d'entrainement 
    data_x_train, x_dont_care, data_y_train, y_dont_care = train_test_split(X, y, train_size=0.05, random_state=42)
    
    
    p = data_x_train.shape[1]
    nb_iter_retro = 200
    nb_iter_rbm = 100
    lr = 0.1
    batch_size = [50, 100, 150, 200, 250]
    
    for i in range(len(batch_size)):
        neurones = [(p, 200), (200, 200), (200, 10)]
        print(neurones)
        print("nb_train_data :", data_x_train.shape)
        print(batch_size[i])
        dbn_alone = init_DBN(len(neurones), neurones)
        dbn_pretrained = copy_dnn(dbn_alone)
        
        dnn_alone = retropropagation(dbn_alone, data_x_train, data_y_train, nb_iter_retro, lr, batch_size[i], "pre-trained_bs{}".format(batch_size[i]))
        test_DNN(dnn_alone, data_x_test, data_y_test, "normal_bs{}".format(batch_size[i]))
        
        dnn_pretrained = train_DBN(data_x_train, dbn_pretrained, batch_size[i], nb_iter_rbm, lr, neurones)
        dnn_pretrained_trained = retropropagation(dnn_pretrained, data_x_train, data_y_train, nb_iter_retro, lr, batch_size[i], "pre-trained_bs{}".format(batch_size[i]))
        test_DNN(dnn_pretrained_trained, data_x_test, data_y_test, "pre-trained_bs{}".format(batch_size[i]))

##### LEARNING RATE

def test_lr():
    # on travaille avec 3000 données d'entrainement 
    data_x_train, x_dont_care, data_y_train, y_dont_care = train_test_split(X, y, train_size=0.05, random_state=42)
    
    
    p = data_x_train.shape[1]
    nb_iter_retro = 200
    nb_iter_rbm = 100
    lr = [0.01, 0.03, 0.06, 0.1, 0.13, 0.16]
    batch_size = 150
    
    for i in range(len(lr)):
        neurones = [(p, 200), (200, 200), (200, 10)]
        print(neurones)
        print("nb_train_data :", data_x_train.shape)
        print(lr[i])
        dbn_alone = init_DBN(len(neurones), neurones)
        dbn_pretrained = copy_dnn(dbn_alone)
        
        dnn_alone = retropropagation(dbn_alone, data_x_train, data_y_train, nb_iter_retro, lr[i], batch_size, "pre-trained_lr{}".format(lr[i]))
        test_DNN(dnn_alone, data_x_test, data_y_test, "normal_lr{}".format(lr[i]))
        
        dnn_pretrained = train_DBN(data_x_train, dbn_pretrained, batch_size, nb_iter_rbm, lr[i], neurones)
        dnn_pretrained_trained = retropropagation(dnn_pretrained, data_x_train, data_y_train, nb_iter_retro, lr[i], batch_size, "pre-trained_lr{}".format(lr[i]))
        test_DNN(dnn_pretrained_trained, data_x_test, data_y_test, "pre-trained_lr{}".format(lr[i]))


##### NB ITÉRATIONS RÉTROPROPAGATION

def test_nb_iter_retro():
    # on travaille avec 3000 données d'entrainement 
    data_x_train, x_dont_care, data_y_train, y_dont_care = train_test_split(X, y, train_size=0.05, random_state=42)
    
    
    p = data_x_train.shape[1]
    nb_iter_retro = [100, 150, 200, 250, 300]
    nb_iter_rbm = 100
    lr = 0.1
    batch_size = 150
    
    for i in range(len(nb_iter_retro)):
        neurones = [(p, 200), (200, 200), (200, 10)]
        print(neurones)
        print("nb_train_data :", data_x_train.shape)
        print(nb_iter_retro[i])
        dbn_alone = init_DBN(len(neurones), neurones)
        dbn_pretrained = copy_dnn(dbn_alone)
        
        dnn_alone = retropropagation(dbn_alone, data_x_train, data_y_train, nb_iter_retro[i], lr, batch_size, "pre-trained_epoch{}".format(nb_iter_retro[i]))
        test_DNN(dnn_alone, data_x_test, data_y_test, "normal_epoch{}".format(nb_iter_retro[i]))
        
        dnn_pretrained = train_DBN(data_x_train, dbn_pretrained, batch_size, nb_iter_rbm, lr, neurones)
        dnn_pretrained_trained = retropropagation(dnn_pretrained, data_x_train, data_y_train, nb_iter_retro[i], lr, batch_size, "pre-trained_epoch{}".format(nb_iter_retro[i]))
        test_DNN(dnn_pretrained_trained, data_x_test, data_y_test, "pre-trained_epoch{}".format(nb_iter_retro[i]))



##### BEST CONFIG PRE_TRAINED
def best_pre_trained():
    train_size_pretrained = round(100*45000/60000)/100
    data_x_train, x_dont_care, data_y_train, y_dont_care = train_test_split(X, y, train_size=train_size_pretrained, random_state=42)
    
    
    p = data_x_train.shape[1]
    nb_iter_retro = 300
    nb_iter_rbm = 100
    lr = 0.06
    batch_size = 200
    
    
    neurones = [(p, 500), (500, 400), (400, 300), (300, 200), (200, 200), (200, 100), (100, 10)]
    print(neurones)
    print("nb_train_data :", data_x_train.shape)
    
    dbn_pretrained = init_DBN(len(neurones), neurones)
    
    dnn_pretrained = train_DBN(data_x_train, dbn_pretrained, batch_size, nb_iter_rbm, lr, neurones)
    dnn_pretrained_trained = retropropagation(dnn_pretrained, data_x_train, data_y_train, nb_iter_retro, lr, batch_size, "pre-trained_best")
    test_DNN(dnn_pretrained_trained, data_x_test, data_y_test, "pre-trained_best")


        
##### BEST CONFIG NORMAL
def best_normal():
    p = X.shape[1]
    nb_iter_retro = 250
    #nb_iter_rbm = 100
    lr = 0.16
    batch_size = 75
    
    neurones = [(p, 300), (300, 200), (200, 10)]
    print(neurones)
    print("nb_train_data :", X.shape)
    
    dbn = init_DBN(len(neurones), neurones)
    
    dnn = retropropagation(dbn, X, y, nb_iter_retro, lr, batch_size, "normal_best")
    test_DNN(dnn, data_x_test, data_y_test, "normal_best")
    
    
"""
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        LANCER LES FONCTIONS SUIVANTES POUR TESTER 
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
"""
# =============================================================================
# comparaison_par_nb_couches()
# comparaison_par_nb_neurones()
# comparaison_par_nb_train()
# test_batch_size()
# test_lr()
# test_nb_iter_retro()
# best_pre_trained()
best_normal()
# =============================================================================
