#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn. feature_selection import SelectKBest,mutual_info_classif
from sklearn.neural_network import MLPClassifier


# In[ ]:


df = pd.read_csv("../input/health-care-data-set-on-heart-attack-possibility/heart.csv")


# In[ ]:


df.head(20)


# In[ ]:


df = df.loc[(df['trestbps'] >=100) & (df['trestbps'] <=180)]
df2 = df['age'].replace(np.arange(0,38),value = 0)
df2 = df2.replace(np.arange(38,53),value = 1)
df2 = df2.replace(np.arange(53,67),value = 2)
df2 = df2.replace(np.arange(67,100),value = 3)
df['age'] = df2


# In[ ]:


df2 = df['trestbps'].replace(np.arange(100,118),0)
df2 = df2.replace(np.arange(118,143),1)
df2 = df2.replace(np.arange(143,159),2)
df2 = df2.replace(np.arange(159,180),3)
df['trestbps'] = df2


# In[ ]:


temp = df['trestbps'].map(str) + '_' +df['exang'].map(str) 
encoder = LabelEncoder()
temp = encoder.fit_transform(temp)
df['exang_trestbps'] = temp


# In[ ]:


df.describe()


# In[ ]:


corr = df.corr("pearson")
corrs = []
for i in df.columns:
    temp = []
    for j in df.columns:
        temp.append(corr[i][j])
    corrs.append(np.array(temp))
corrs = np.array(corrs)
fig,ax = plt.subplots(1,1,figsize = (12,10))
sns.heatmap(corrs,annot = True,xticklabels = corr.keys(),yticklabels = corr.keys(),ax = ax)


# In[ ]:


class Logistic_Regression:
    def __init__(self,epochs,learning_rate,beta,regularization,err_thr,func):
        self.__epochs = epochs
        self.__alpha = learning_rate
        self.__err_thr = err_thr
        self.__act = func
        self.__rparam = regularization 
        self.__parameters = {}
        self.__beta = beta
        self.__losses = []
        self.__val_losses = []
    
    def __initialize_parameters(self,x_h):
        np.random.seed(1)
        self.__parameters['W'] = np.random.randn(1,x_h) * 0.01
        self.__parameters['b'] = 0
        self.__parameters['Vdw'] = np.zeros((1,x_h))
        self.__parameters['Vdb'] = 0
    
    def __compute_cost(self,A,Y):
        m = Y.shape[1]
        temp = (self.__rparam/2) * np.matmul(self.__parameters['W'],self.__parameters['W'].T) / m
        val = np.sum(np.multiply(Y,np.log(A)) + np.multiply((1-Y),np.log(1-A))) * -1/m
        val += temp[0,0]
        
        
        return val
        
    def __activation(self,X):
        val = 0
        if(self.__act == "sigmoid"):
            t = 1 + np.exp(-1 * X)
            val = 1/t
        elif(self.__act == "relu"):
            val = np.maximum(0,X)
        else:
            val = np.tanh(X)
        return val

    def __activation_derivative(self,X):
        val = 0
        if(self.__act == "sigmoid"):
            val = np.multiply(X,1-X)
        elif self.__act == "relu":
            val = X>0
        else:
            val = 1 - X**2
        return val
    
    def __linear_forward(self,X):
        Z = np.matmul(self.__parameters['W'],X) + self.__parameters['b']
        A = self.__activation(Z)
        
        return A
    def __linear_backward(self,A,X,Y):
        m = X.shape[0]
        
        dZ = A-Y
        
        dW = (self.__rparam * self.__parameters['W']) + (np.matmul(dZ,X.T) * 1/m)
        db = np.sum(dZ,axis = 1) * 1/m
        Vdw = (self.__beta * self.__parameters['Vdw']) + (1 - self.__beta)*dW
        Vdb = (self.__beta * self.__parameters['Vdb']) + (1 - self.__beta)*db
        
        grads = {'Vdw':Vdw,'Vdb':Vdb}
        return grads
    
    def __update_parameters(self,grads):
        self.__parameters['W'] -= self.__alpha * grads['Vdw']
        self.__parameters['b'] -= self.__alpha * grads['Vdb']
        self.__parameters['Vdw'] = grads['Vdw']
        self.__parameters['Vdb'] = grads['Vdb']
    
    def fit(self,X,Y,X_val,Y_val,print_cost = True):
        self.__initialize_parameters(X.shape[0])
        #print(self.__parameters)
        
        for i in range(self.__epochs):
            A = self.__linear_forward(X)
            
            cost = self.__compute_cost(A,Y)
            self.__losses.append(cost)
            Aval = self.__linear_forward(X_val)
            val_cost = self.__compute_cost(Aval,Y_val)
            self.__val_losses.append(val_cost)
            if(cost < self.__err_thr):
                print("error threshold reached")
                break
            grads = self.__linear_backward(A,X,Y)
            
            self.__update_parameters(grads)
            
            if(print_cost):
                print("epoch {0}: loss-> {1},validation loss-> {2}".format(i,cost,val_cost))
    
    def training_losses(self):
        return self.__losses
    
    def validation_losses(self):
        return self.__val_losses
    
    def predict(self,X,chance = False):
        A = self.__linear_forward(X)
        
        pred = [int(i>0.5) for i in A[0]]
        if(not chance):
            return np.array(pred)
        else:
            return np.array(pred),A[0]
    
    def confusion_matrix(self,X,Y,classes,show_table = True):
        pred = self.predict(X)    
        cmat = np.zeros((classes,classes))
        print(cmat.shape)
        for j in range(len(pred)):
            cmat[pred[j]][Y[j]]+=1
    
        if(show_table):
            fig,ax = plt.subplots(1,1,figsize = (10,8))
            sns.heatmap(cmat,annot = True,xticklabels = range(classes),yticklabels = range(classes),ax = ax);
        else:
            return cmat


# In[ ]:


class MLP:
    def __init__(self,layer_dims,func,loss_type,epochs,
                 regularization = 0.001,learning_rate = 0.1,beta = 0.0):
        self.__layer_dims = layer_dims
        self.__layers = len(layer_dims) - 1
        self.__func = func
        self.__epochs = epochs
        self.__alpha = learning_rate
        self.__beta = beta
        self.__rparam = regularization
        self.__loss_type = loss_type
        self.__parameters = {}
        self.__losses = []
        self._val_losses = []
    
    def __initialize_parameters(self):
    
        for i in range(1,self.__layers+1):
            self.__parameters['W'+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1])*0.01
            self.__parameters['b'+str(i)] = np.zeros((layer_dims[i],1))
            self.__parameters['VdW' + str(i)] = np.zeros((layer_dims[i],1))
            self.__parameters['Vdb' + str(i)] = np.zeros((layer_dims[i],1))
            #print(parameters['W'+str(i)].shape,",",parameters['b'+str(i)].shape )
    
    def __compute_cost(self,A,Y):
        loss = 0
        
        temp = 0
        for i in range(1,self.__layers+1):
            temp += (self.__rparam /2) * np.sum(np.sum(self.__parameters['W' + str(i)] ** 2, 
                                                       axis = 1, keepdims = True))
        m = Y.shape[1]
        if(self.__loss_type == "logistic"):
            temp = temp / m
            loss = np.sum(np.multiply(np.log(A),Y) + np.multiply(np.log(1-A),(1-Y))) * -1/m
            loss += temp
        elif(self.__loss_type == "cross entropy"):
            loss = np.sum(np.sum(np.multiply(np.log(A),Y),axis = 1,keepdims = True)) * -1/m
            loss += temp/m
        elif(self.__loss_type == "mse"):
            loss = np.sum((Y-A)**2) * 1/m
            loss += temp/m
        return loss
    
    def __activation(self,X,func = "tanh"):
        val = 0
        if(func == "sigmoid"):
            t = 1 + np.exp(-1 * X)
            val = 1/t
        elif(func == "softmax"):
            t1 = np.sum(np.exp(X),axis = 0,keepdims = True)
            val = np.exp(X) / t1
        elif(func == "relu"):
            val = np.maximum(0,X)
        else:
            val = np.tanh(X)
        return val

    def __activation_derivative(self,X,func = "tanh"):
        val = 0
        if(func == "sigmoid"):
            val = np.multiply(X,1-X)
        elif func == "relu":
            val = X>0
        else:
            val = 1 - X**2
        return val
    
    def __forward_propagation(self,X):
        cache = {}
        fin = X
        for i in range(1,self.__layers+1):
            zi = np.matmul(self.__parameters["W"+str(i)],fin) + self.__parameters["b" + str(i)]
            ai = self.__activation(zi,self.__func[i-1])
            fin = ai
            cache["Z" + str(i)] = zi
            cache["A" + str(i)] = ai
        cache["A0"] = X
    
        return fin,cache
    
    def __backward_propagation(self,Y,A,cache):
        m = Y.shape[1]
        grads = {}
    
        dA = -1 * (np.divide(Y,A) - np.divide((1-Y),(1-A)))
    
        for i in range(self.__layers,0,-1):
            if(self.__func[i-1] == "softmax"):
                dzi = A-Y
            else:
                dzi = np.multiply(dA,self.__activation_derivative(cache["A" + str(i)],self.__func[i-1]))
        
            dwi = (self.__rparam * self.__parameters['W'+str(i)]) + np.matmul(dzi,cache["A" + str(i-1)].T) * 1/m 
            Vdwi = self.__beta * (self.__parameters['VdW' + str(i)]) + (1-self.__beta) * dwi
            dbi = np.sum(dzi,axis = 1,keepdims = True) * 1/m
            Vdbi = self.__beta * (self.__parameters['Vdb' + str(i)]) + (1-self.__beta) * dbi

        
            dA = np.matmul(self.__parameters["W" + str(i)].T,dzi)
        
            grads["VdW" + str(i)] = Vdwi
            grads["Vdb" + str(i)] = Vdbi
    
        return grads
    
    def __update_parameters(self,grads):
        for i in range(1,self.__layers+1):
            self.__parameters["W" + str(i)] -= (self.__alpha * grads["VdW" + str(i)]) 
            self.__parameters["b" + str(i)] -= (self.__alpha * grads["Vdb" + str(i)])
            self.__parameters['VdW' + str(i)] = grads["VdW" + str(i)]
            self.__parameters['Vdb' + str(i)] = grads["Vdb" + str(i)]
            
    def fit (self,X,Y,X_val,Y_val,print_cost = True):
        self.__initialize_parameters()
        
    
        for i in range(self.__epochs):
            A,cache = self.__forward_propagation(X)
            Aval,val_cache = self.__forward_propagation(X_val)
        
            cost = self.__compute_cost(A,Y)
            val_cost = self.__compute_cost(Aval,Y_val)
            self.__losses.append(cost)
            self._val_losses.append(val_cost)
        
            grads = self.__backward_propagation(Y,A,cache)
        
            self.__update_parameters(grads)
        
            if(print_cost and i%1000 == 0): 
                print("epoch {0}: loss {1},validation_loss {2}".format(i,cost,val_cost))
    
    def Weights(self):
        return self.__parameters
    
    def predict(self,X,chances = False):
        A,cache = self.__forward_propagation(X)

        pred = []
        if(func[self.__layers-1] == "softmax"):
            pred = [int(a[1] > a[0]) for a in A.T]
        else:
            pred = [int(i>0.5) for i in A[0]]
        if(chances):
            return np.array(pred),A
        return np.array(pred)
    
    def training_losses(self):
        return self.__losses
    def validation_losses(self):
        return self._val_losses
    def __to_numeric(self,Y):
        Y_n = []
        for i in Y:
            Y_n.append(np.argmax(i))
        return Y_n
    
    def confusion_matrix(self,X,Y,classes,show_table = True):
        pred = self.predict(X)
        Y_n = Y[0]
        if(Y.shape[0] > 1):
            Y_n = self.__to_numeric(Y.T)
    
        cmat = np.zeros((classes,classes))
        for j in range(len(pred)):
            cmat[pred[j]][Y_n[j]]+=1
    
        if(show_table):
            fig,ax = plt.subplots(1,1,figsize = (6,4))
            sns.heatmap(cmat,annot = True,xticklabels = range(classes),yticklabels = range(classes),ax = ax);
        else:
            return cmat
    def score(self,X,Y):
        cmat = self.confusion_matrix(X,Y,2,False)

        recall = cmat[1,1] / (cmat[1,1] + cmat[1,0])
        precision = cmat[1,1] / (cmat[0,1] + cmat[1,1])
        f1 = (2*recall*precision) / (recall + precision)
        print(f1,precision,recall)


# **The Given Configuration gives the best result in heart disease chance identification using 2-layer Neural network**
# #layer_dims = 13,16,8,2 \n
# #learning_rate = 0.01 \n
# #epochs = 28000 \n
# #func = "tanh","relu","softmax" \n
# #categorical output data \n
# #with softmax,precision obtained  = 0.8 \n
# #with binary classification and sigmoid activation ,precision obtained = 0.8 \n
# 
# #learning losses are as given below : \n
# 
# #epoch 0: loss 0.6931485676515335,validation_loss 0.693144098510043
# #epoch 1000: loss 0.6887017123988481,validation_loss 0.6974813547273234
# #epoch 2000: loss 0.6886745301956823,validation_loss 0.6975035213871978
# #epoch 3000: loss 0.6885836409704611,validation_loss 0.6973784363529545
# #epoch 4000: loss 0.6880411478759397,validation_loss 0.6967235006020699
# #epoch 5000: loss 0.6768335025712596,validation_loss 0.6847894715047003
# #epoch 6000: loss 0.39612337914583884,validation_loss 0.3903921853547465
# #epoch 7000: loss 0.34890486288833983,validation_loss 0.29703991344674785
# #epoch 8000: loss 0.34798941791835686,validation_loss 0.2950970475817546
# #epoch 9000: loss 0.347175283772169,validation_loss 0.29653197227789085
# #epoch 10000: loss 0.3456832946151523,validation_loss 0.30017665638344937
# #epoch 11000: loss 0.3418326485401162,validation_loss 0.3061799398299396
# #epoch 12000: loss 0.33601893498916086,validation_loss 0.3133779449464917
# #epoch 13000: loss 0.32733439288860544,validation_loss 0.31598487121667734
# #epoch 14000: loss 0.31521338175795094,validation_loss 0.31369064883205455
# #epoch 15000: loss 0.29969110969458085,validation_loss 0.270702233443918
# #epoch 16000: loss 0.2813077420036701,validation_loss 0.22839025650880446
# #epoch 17000: loss 0.2590510873294439,validation_loss 0.1832658942216211
# #epoch 18000: loss 0.23872765202909813,validation_loss 0.1604170508980663
# #epoch 19000: loss 0.21673753662651696,validation_loss 0.15780967629505582
# #epoch 20000: loss 0.1955953437948535,validation_loss 0.1721391450175061
# #epoch 21000: loss 0.17523453746847187,validation_loss 0.19372259753133697
# #epoch 22000: loss 0.15722203124502623,validation_loss 0.22126037123680972
# #epoch 23000: loss 0.1389943107200719,validation_loss 0.22853247381147768
# #epoch 24000: loss 0.12369653763094877,validation_loss 0.22576547358014826
# #epoch 25000: loss 0.1065589747634406,validation_loss 0.21411620541363036
# #epoch 26000: loss 0.09304460864597647,validation_loss 0.20239256375625403
# #epoch 27000: loss 0.08178586174949458,validation_loss 0.19118345954115026
# 
# #the confusion matrix for softmax function is as given below:
# """data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAiYAAAHWCAYAAABDtE
# LCAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29md
# HdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAZA0lEQVR4nO3df
# cylZX0n8O9vRoymdpc2GIGZUXSdrRarQrajxpjQplVgUZpIFDfKlpjMipqobXzZbiNxE5PaP8xKsNJJa5HYYmnsKrFgY9YSoJY3EaYwaHfUpjwwQ
# tUtFCHKzHPtH3Owj888r+O5Oee65/Mhdzgv97nONWSG5zff33Xdd7XWAgAwD7bMegIAAE9QmAAAc0NhAgDMDYUJADA3FCYAwNxQmA
# AAc0NhAgActaraWlVfq6ovrPBeVdUlVbW/qvZW1enrjacwAQB+Gu9Kcs8q752VZOfk2J3kE+sNpjABAI5KVW1P8p+T/NEqp5yb5Ip22E1Jjq+
# qk9YaU2ECAByt/5XkfUkWV3l/W5J7lzxfmLy2qqdMZ16re/y733LNe5iBp5/8qllPAY5ZB390Xz2Z3zfEz9qnPvM//Lccbr88YU9rbc8TT6rqnCQPtta+WlV
# nrDLMSv8d1pzr4IUJANCfSRGyZ41TXpnkdVV1dpKnJfl3VfXp1tqbl5yzkGTHkufbk9y/1vdq5QBA7xYPTf9YR2vtv7fWtrfWTklyfpIvLytKkuTq
# JBdMdue8PMlDrbUDa40rMQEApqaq3pYkrbXLklyT5Owk+5M8muTCdT/f2rBLQKwxgdmwxgRm50lfY/LAN6b+s/a4Z/3Ck/preIJWDgAwN7RyAKB3i
# 6vt1u2PwgQAOtfaeAoTrRwAYG5ITACgdyNq5UhMAIC5ITEBgN6NaI2JwgQAereBK7X2QisHAJgbEhMA6N2IWjkSEwBgbkhMAKB3I9ourDABgM658isAwAAkJgDQuxG1ciQmAMDckJgAQO+sMQEAmD6JCQD0bkSXpFeYAEDvtHIAAKZPYgIAvbNdGABg+iQmANC7Ea0xUZgAQO+0cgAApk9iAgCda2081zGRmAAAc0NiAgC9s/gVAJgbFr8CAEyfxAQAejeiVo7EBACYGxITAOjd4ni2CytMAKB3WjkAANMnMQGA3tkuDAAwfRITAOidNSYAANMnMQGA3o1ojYnCBAB6N6LCRCsHAJgbEhMA6Fxr47nyq8QEAJgbEhMA6J01JgDA3GiL0z/WUVVPq6pbqurOqrq7qj60wjlnVNVDVXXH5PjgeuNKTACAo/HDJL/aWnukqo5LcmNVXdtau2nZeTe01s7Z6KAKEwDo3QxaOa21luSRydPjJkf7acfVygEAjlBVu6vqtiXH7hXO2VpVdyR5MMmXWms3rzDUKybtnmur6tT1vldiAgC9G+BeOa21PUn2rHPOoSQvrarjk/zvqnpRa+2uJafcnuQ5k3bP2Uk+l2TnWmNKTACgd4uL0z82obX2L0muS3Lmstcfbq09Mnl8TZLjquqEtcZSmAAAm1ZVz5wkJamqpyf5tSRfX3bOiVVVk8e7crju+N5a42rlAEDvBmjlbMBJST5VVVtzuOC4qrX2hap6W5K01i5Lcl6Si6rqYJLHkpw/WTS7KoUJALBprbW9SU5b4fXLljy+NMmlmxlXYQIAvXPlVwCA6ZOYAEDvRpSYKEwAoHezWfw6CK0cAGBuSEwAoHcjauVITACAuSExAYDejWiNicIEAHqnlQMAMH0SEwDo3YhaORITAGBuSEwAoHcjWmOiMAGA3o2oMNHKAQDmhsQEAHrX2qxnMDUSEwBgbkhMAKB31pgAAEyfxAQAejeixERhAgC9c+VXAIDpk5gAQO9G1MqRmAAAc0NiAgC9G9EF1hQmANA7rRwAgOmTmABA7yQmAADTJzEBgN6N6AJrChMA6FxbHM+uHK0cAGBuSEwAoHcWvwIATJ/EBAB6N6LFrxITAGBuSEwAoHcj2pWjMAGA3ln8CgAwfRITAOidxAQAYPokJgDQu2bxKwAwL7RyAACmT2HCqg4dOpTzfvMdeft7L571VOCY8ZpXn5G777o+X993Y9733nfMejr0YrFN/5gRhQmr+vRffD7PO+XZs54GHDO2bNmSSz724Zzz2jfnl17yK3njG38jL3zhzllPC1ZUVU+rqluq6s6quruqPrTCOVVVl1TV/qraW1WnrzfuuoVJVb2gqt4/Gfhjk8cvPNpfCH34zoP/nOu/ckte/9rXzHoqcMzY9cun5Zvf/Md8+9v/lMcffzxXXfX5vM6fQTaiLU7/WN8Pk/xqa+0lSV6a5Myqevmyc85KsnNy7E7yifUGXbMwqar3J/lMkkpyS5JbJ4+vrKoPbGTW9OkjH/vD/Nbb35oqoRo8WU7edmLuXbj/x88X7juQk08+cYYzohszaOW0wx6ZPD1uciz/4LlJrpice1OS46vqpLXGXW9XzluTnNpae3zpi1X10SR3J/m9dWdOd67725vz8z93fE59wc7ccvveWU8HjhlVdcRrbUTbQBmfqtqa5KtJnp/k4621m5edsi3JvUueL0xeO7DamOv9dXgxyckrvH7S5L3VJrq7qm6rqtv+6Ior1/kK5s3X9u7LdTfelFe//r/mvRf/Xm756p15/4d+f9bTgtG7b+FAdmz/t//lbt92Ug4ceGCGM6IXbXFx6sfSn+WTY/cR39vaodbaS5NsT7Krql607JQjq+0jU5WfsF5i8u4k/6eq/m/+reJ5dg5XRu9c7UOttT1J9iTJ49/9lnK/M++56MK856ILkyS33L43l1/52Xzk4vfNeFYwfrfedkee//zn5pRTduS++76TN7zh3LzlAjtzmI2lP8s3cO6/VNV1Sc5McteStxaS7FjyfHuS+7OGNQuT1toXq+o/JtmVw9FLTb7k1tbaoY1MFoCNOXToUN717t/NNX/1Z9m6ZUsu/9SfZ9++f5j1tOjBDLb3VtUzkzw+KUqenuTXknxk2WlXJ3lnVX0mycuSPNRaW7WNkyQ1dP9SYgKz8fSTXzXrKcAx6+CP7luphTGYH3z4gqn/rP2Z/3HFmr+Gqnpxkk8l2ZrDS0Ouaq39z6p6W5K01i6rwwunLs3hJOXRJBe21m5ba1yXpAeA3m1se+90v7K1vUlOW+H1y5Y8bkk21Y9UmABA72Z4pdZpc5EKAGBuSEwAoHfuLgwAMH0SEwDo3YjWmChMAKB3M9iVMxStHABgbkhMAKB3I2rlSEwAgLkhMQGAzrURbRdWmABA77RyAACmT2ICAL2TmAAATJ/EBAB65wJrAADTJzEBgN6NaI2JwgQAOtdGVJho5QAAc0NiAgC9k5gAAEyfxAQAeudeOQDA3NDKAQCYPokJAPROYgIAMH0SEwDoXGvjSUwUJgDQO60cAIDpk5gAQO8kJgAA0ycxAYDOubswAMAAJCYA0LsRJSYKEwDo3Xju4aeVAwDMD4kJAHTO4lcAgAFITACgdyNKTBQmANA7i18BAKZPYgIAnbP4FQBgABITAOjdiNaYKEwAoHNaOQDAMa2qdlTV31TVPVV1d1W9a4Vzzqiqh6rqjsnxwfXGlZgAQO9m08o5mOS3W2u3V9XPJvlqVX2ptbZv2Xk3tNbO2eigEhMAYNNaawdaa7dPHv9rknuSbPtpx1WYAEDn2uL0j82oqlOSnJbk5hXefkVV3VlV11bVqeuNpZUDAL0boJVTVbuT7F7y0p7W2p4VzntGks8meXdr7eFlb9+e5DmttUeq6uwkn0uyc63vVZgAAEeYFCFHFCJLVdVxOVyU/Glr7S9XGOPhJY+vqao/qKoTWmvfXW1MhQkAdG6zrZdpqKpK8sdJ7mmtfXSVc05M8kBrrVXVrhxeQvK9tcZVmAAAR+OVSd6S5O+r6o7Ja7+T5NlJ0lq7LMl5SS6qqoNJHktyfmttzYuuKEwAoHczSExaazcmqXXOuTTJpZsZ164cAGBuSEwAoHOzWGMyFIUJAHRuTIWJVg4AMDckJgDQOYkJAMAAJCYA0Lu25q7drihMAKBzWjkAAAOQmABA59rieFo5EhMAYG5ITACgc2NaY6IwAYDOtRHtytHKAQDmhsQEADo3plaOxAQAmBsSEwDonO3CAAADkJgAQOdam/UMpkdhAgCd08oBABiAxAQAOicxAQAYgMQEADpn8SsAMDe0cgAABiAxAYDOubswAMAAJCYA0Lkx3V1YYQIAnVvUygEAmD6JCQB0zuJXAIABSEwAoHMusAYAMACJCQB0zr1yAIC5oZUDADAAiQkAdM4F1gAABiAxAYDOjekCawoTAOjcmHblaOUAAHNDYgIAnbP4FQBgABITAOjcmBa/SkwAoHOtTf9YT1XtqKq/qap7quruqnrXCudUVV1SVfuram9Vnb7euBITAOBoHEzy262126vqZ5N8taq+1Frbt+Scs5LsnBwvS/KJyb9XpTABgM7NYvFra+1AkgOTx/9aVfck2ZZkaWFybpIrWmstyU1VdXxVnTT57IoGL0yefvKrhv4KYAWP3X/DrKcAdKyqdifZveSlPa21Pauce0qS05LcvOytbUnuXfJ8YfLa7AoTAGBYQyx+nRQhKxYiS1XVM5J8Nsm7W2sPL397paHXGs/iVwDgqFTVcTlclPxpa+0vVzhlIcmOJc+3J7l/rTEVJgDQucVWUz/WU1WV5I+T3NNa++gqp12d5ILJ7pyXJ3lorfUliVYOAHRvRrfKeWWStyT5+6q6Y/La7yR5dpK01i5Lck2Ss5PsT/JokgvXG1RhAgBsWmvtxqy8hmTpOS3JOzYzrsIEADrnXjkAAAOQmABA58Z0rxyFCQB0bnHWE5girRwAYG5ITACgc23tzTFdkZgAAHNDYgIAnVuc0RXWhqAwAYDOLWrlAABMn8QEADpn8SsAwAAkJgDQORdYAwAYgMQEADo3pjUmChMA6JxWDgDAACQmANA5iQkAwAAkJgDQOYtfAYC5sTieukQrBwCYHxITAOicuwsDAAxAYgIAnWuznsAUKUwAoHOuYwIAMACJCQB0brEsfgUAmDqJCQB0bkyLXyUmAMDckJgAQOfGtCtHYQIAnXOvHACAAUhMAKBz7pUDADAAiQkAdG5M24UVJgDQOYtfAQAGIDEBgM6N6TomEhMAYG5ITACgcxa/AgBzw+JXAIABSEwAoHMWvwIAx7Sq+mRVPVhVd63y/hlV9VBV3TE5PriRcSUmANC5GSUmlye5NMkVa5xzQ2vtnM0MKjEBADattXZ9ku9Pe1yFCQB0rtX0jyl5RVXdWVXXVtWpG/mAVg4AdG6IVk5V7U6ye8lLe1prezYxxO1JntNae6Sqzk7yuSQ71/uQwgQAOMKkCNlMIbL88w8veXxNVf1BVZ3QWvvuWp9TmABA5+Zxu3BVnZjkgdZaq6pdObx85HvrfU5hAgBsWlVdmeSMJCdU1UKSi5MclySttcuSnJfkoqo6mOSxJOe31ta9er7CBAA6N4t75bTW3rTO+5fm8HbiTVGYAEDn3CsHAGAAEhMA6Nw8Ln49WhITAGBuSEwAoHNjSkwUJgDQuVnsyhmKVg4AMDckJgDQOduFAQAGIDEBgM6NafGrxAQAmBsSEwDo3Jh25ShMAKBziyMqTbRyAIC5ITEBgM5Z/AoAMACJCQB0bjwrTBQmANA9rRwAgAFITACgc+6VAwAwAIkJAHRuTBdYU5gAQOfGU5Zo5QAAc0RiAgCds10YAGAAEhMA6JzFrwDA3BhPWaKVAwDMEYkJAHTO4lcAgAFITACgc2Na/CoxAQDmhsQEADo3nrxEYQIA3bP4FQBgABITAOhcG1EzR2ICAMwNiQkAdG5Ma0wUJgDQOdcxAQAYgMQEADo3nrxEYgIAzBGJCQB0zhoTRu81rz4jd991fb6+78a8773vmPV04Jhy6NChnPeb78jb33vxrKdCJxYHOGZFYcIRtmzZkks+9uGc89o355de8it54xt/Iy984c5ZTwuOGZ/+i8/neac8e9bTgDVV1Ser6sGqumuV96uqLqmq/VW1t6pO38i4ChOOsOuXT8s3v/mP+fa3/ymPP/54rrrq83nda18z62nBMeE7D/5zrv/KLXm9P3NsQhvgnw24PMmZa7x/VpKdk2N3kk9sZFCFCUc4eduJuXfh/h8/X7jvQE4++cQZzgiOHR/52B/mt97+1lT53zPzrbV2fZLvr3HKuUmuaIfdlOT4qjppvXGP+nd+VV14tJ9lvlXVEa+1Np6FVTCvrvvbm/PzP3d8Tn2B1imbM6drTLYluXfJ84XJa2v6aXblfCjJn6z0RlXtzuHYJrX132fLlp/5Kb6GJ9t9CweyY/vJP36+fdtJOXDggRnOCI4NX9u7L9fdeFNu+Ltb88MfPZ4f/ODRvP9Dv5+PXPy+WU+NY9DSn+UTe1prezYzxAqvrfu33DULk6rau8aXPWu1z00mvidJnvLUbf6q3Zlbb7sjz3/+c3PKKTty333fyRvecG7ecoGdOTC091x0Yd5z0eEw+pbb9+byKz+rKGFDhri78NKf5UdpIcmOJc+3J7l/lXN/bL3E5FlJXpPk/y17vZJ8ZTOzox+HDh3Ku979u7nmr/4sW7dsyeWf+vPs2/cPs54WAKuY05v4XZ3knVX1mSQvS/JQa+3Aeh9arzD5QpJntNbuWP5GVV13NLOkD9d+8cu59otfnvU04Ji16/QXZ9fpL571NGBVVXVlkjOSnFBVC0kuTnJckrTWLktyTZKzk+xP8miSDa1NXbMwaa29dY33/stGvgAAGNbiDDYotNbetM77Lcmm1wHYjwYAzA33ygGAzo1pl4nCBAA65yZ+AAADkJgAQOeGuI7JrEhMAIC5ITEBgM7N6QXWjorCBAA6Z/ErAMAAJCYA0DmLXwEABiAxAYDOjWnxq8QEAJgbEhMA6Fybwd2Fh6IwAYDO2S4MADAAiQkAdM7iVwCAAUhMAKBzY7rAmsIEADpn8SsAwAAkJgDQuTFdx0RiAgDMDYkJAHRuTNuFFSYA0Lkx7crRygEA5obEBAA6Z7swAMAAJCYA0DnbhQEABiAxAYDOjWmNicIEADpnuzAAwAAkJgDQuUWLXwEApk9iAgCdG09eojABgO6NaVeOVg4AMDckJgDQOYkJAMAAJCYA0Lkx3StHYQIAndPKAQAYgMQEADrnXjkAAAOQmABA58a0+FViAgAclao6s6q+UVX7q+oDK7x/RlU9VFV3TI4PrjemxAQAOjeLXTlVtTXJx5P8epKFJLdW1dWttX3LTr2htXbORsdVmABA52bUytmVZH9r7VtJUlWfSXJukuWFyaZo5QAAR6iq3VV125Jj97JTtiW5d8nzhclry72iqu6sqmur6tT1vldiAgCdG6KV01rbk2TPGqfUSh9b9vz2JM9prT1SVWcn+VySnWt9r8QEADgaC0l2LHm+Pcn9S09orT3cWntk8viaJMdV1QlrDaowAYDOtQH+2YBbk+ysqudW1VOTnJ/k6qUnVNWJVVWTx7tyuO743lqDauUAQOcWZ7D4tbV2sKremeSvk2xN8snW2t1V9bbJ+5clOS/JRVV1MMljSc5v66zUraFX8j7lqdvGc9UX6Mhj998w6ynAMeu4E5630vqLwbzoWS+f+s/aux646Un9NTxBYgIAnXOvHACAAUhMAKBzs1hjMhSFCQB0TisHAGAAEhMA6NyYWjkSEwBgbkhMAKBz1pgAAAxAYgIAnRvTGhOFCQB0TisHAGAAEhMA6Fxri7OewtRITACAuSExAYDOLY5ojYnCBAA610a0K0crBwCYGxITAOjcmFo5EhMAYG5ITACgc2NaY6IwAYDOjemS9Fo5AMDckJgAQOfcKwcAYAASEwDo3JgWv0pMAIC5ITEBgM6N6QJrChMA6JxWDgDAACQmANA5F1gDABiAxAQAOjemNSYKEwDo3Jh25WjlAABzQ2ICAJ0bUytHYgIAzA2JCQB0bkzbhRUmANC5ZvErAMD0SUwAoHNjauVITACAuSExAYDO2S4MADAAiQkAdG5Mu3IUJgDQOa0cAOCYV1VnVtU3qmp/VX1ghferqi6ZvL+3qk5fb0yJCQB0bhaJSVVtTfLxJL+eZCHJrVV1dWtt35LTzkqyc3K8LMknJv9elcQEADgau5Lsb619q7X2oySfSXLusnPOTXJFO+ymJMdX1UlrDaowAYDOtQGODdiW5N4lzxcmr232nJ8weCvn4I/uq6G/g+FU1e7W2p5ZzwOONf7ssRlD/Kytqt1Jdi95ac+y35MrfefymmYj5/wEiQnr2b3+KcAA/Nljplpre1pr/2nJsbxQXkiyY8nz7UnuP4pzfoLCBAA4Grcm2VlVz62qpyY5P8nVy865OskFk905L0/yUGvtwFqD2pUDAGxaa+1gVb0zyV8n2Zrkk621u6vqbZP3L0tyTZKzk+xP8miSC9cbt8Z0URamT58bZsOfPY5VChMAYG5YYwIAzA2FCSta7zLDwDCq6pNV9WBV3TXrucAsKEw4wpLLDJ+V5BeTvKmqfnG2s4JjxuVJzpz1JGBWFCasZCOXGQYG0Fq7Psn3Zz0PmBWFCSvZ9CWEAWAaFCasZNOXEAaAaVCYsJJNX0IYAKZBYcJKNnKZYQCYOoUJR2itHUzyxGWG70lyVWvt7tnOCo4NVXVlkr9L8gtVtVBVb531nODJ5MqvAMDckJgAAHNDYQIAzA2FCQAwNxQmAMDcUJgAAHNDYQIAzA2FCQAwNxQmAMDc+P8NKr0tKW+BcgAAAABJRU5ErkJggg==

# In[ ]:


def heart_disease_classifier(X_orig,Y_orig,t_size,layer_dims,learning_rate,C,epochs,beta,func):
    
    
    X = (X_orig - np.mean(X_orig,axis = 0,keepdims = True))/np.std(X_orig,axis= 0,keepdims = True)
    Y = np.zeros((Y_orig.shape[1],2))

    for i in range(Y_orig.shape[1]):
         Y[i,Y_orig[0,i]] = 1

    X_temp,X_test,Y_temp,Y_test = train_test_split(X,Y,stratify = Y,random_state = 42,test_size = t_size)
    X_train,X_val,Y_train,Y_val = train_test_split(X_temp,Y_temp,stratify = Y_temp,random_state = 42,test_size = 0.015)
       
    X_train = X_train.T
    Y_train = Y_train.T
    X_test = X_test.T
    Y_test = Y_test.T
    X_val = X_val.T
    Y_val = Y_val.T
   
    layer_dims.insert(0,X_train.shape[0])
    layer_dims.append(Y_train.shape[0])
   
    model = MLP(layer_dims,func,'cross entropy',epochs,C,learning_rate,beta)
    model.fit(X_train,Y_train,X_val,Y_val,print_cost = False)
    
    #model.confusion_matrix(X_test,Y_test,2,True)
    
    model.score(X_test,Y_test)
    return model,X_test,Y_test


# In[ ]:


Y_orig = df["target"].values.reshape(1,-1)
df = df.drop(columns = 'target')

for i in range(5,15):
    X_orig = np.array(df)
    selector = SelectKBest(mutual_info_classif,k = i)
    selector.fit(X_orig,Y_orig[0])
    X_new = selector.transform(X_orig)
    col = df.columns
    temp= pd.DataFrame(selector.inverse_transform(X_new),columns = col)
    selected_features = temp.columns[temp[col].var() != 0]
    X_orig = X_new
    #print(selected_features)
    test_size = 0.05
    layer_dims = [17,8]
    learning_rate = 0.015
    C = 0.001
    epochs = 10000
    beta = 0.2
    func = ["relu","relu","softmax"]
    print('for K = {0}, score is :'.format(i),end = ' ')
    model,Xtest,Ytest = heart_disease_classifier(X_orig,Y_orig,test_size,layer_dims,learning_rate,C,epochs,beta,func)
    


# In[ ]:


#Y_orig = df["target"].values.reshape(1,-1)
#df = df.drop(columns = 'target')
for i in range(5,15):
    X_orig = np.array(df)
    selector = SelectKBest(mutual_info_classif,k = i)
    selector.fit(X_orig,Y_orig[0])
    X_new = selector.transform(X_orig)
    col = df.columns
    temp= pd.DataFrame(selector.inverse_transform(X_new),columns = col)
    selected_features = temp.columns[temp[col].var() != 0]
    print(selected_features)
    X_orig = X_new
    X = (X_orig - np.mean(X_orig,axis = 0,keepdims = True))/np.std(X_orig,axis= 0,keepdims = True)

    Y = Y_orig.T

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,stratify = Y,random_state = 42,test_size = 0.025)
    X_train = X_train.T
    Y_train = Y_train.T
    X_test = X_test.T
    Y_test = Y_test.T
    
    
    model2 = LogisticRegression().fit(X_train.T,Y_train[0])
    print('Logistic Regression,K = {0}, score = {1}'.format(i,model2.score(X_test.T,Y_test[0])))
    model3 = RandomForestClassifier().fit(X_train.T,Y_train[0])
    
    print('Random Forest,K = {0}, score = {1}'.format(i,model3.score(X_test.T,Y_test[0])))
    model4 = MLPClassifier((16,8),'relu',max_iter = 10000).fit(X_train.T,Y_train[0])
    print('Neural Network,K = {0}, score = {1}'.format(i,model4.score(X_test.T,Y_test[0])))


# In[ ]:


fig,ax = plt.subplots(1,2,figsize = (20,5))
ax[0].plot(np.arange(len(model.training_losses())),model.training_losses(),'r')
ax[1].plot(np.arange(len(model.validation_losses())),model.validation_losses(),'b')


# In[ ]:


print(model.predict(X_test))
print(Y_test[0])
print(model2.predict(X_test.T))


# In[ ]:


model2.score(X_test.T,Y_test[0]) * 100


# In[ ]:





# In[ ]:


chances = [((str(i[0] * 100) +","+ str(i[1]*100)).format('%0.4f') + '%') for i in predict(X_test,3,parameters,func,True)[1].T]
print(chances)


# In[ ]:


model.predict(X_test,True)


# In[ ]:


print(Y_test.T)


# In[ ]:


import inspect
import os

def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)

write_agent_to_file(heart_disease_classifier, "heart.py")


# In[ ]:


open("./Multi-LayerNeuralNetwork.py",'r').read()

