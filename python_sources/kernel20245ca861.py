import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
import cmath

Ktr=0;Kte=0;  # Ktr denotes training features extracted from hundreds of subnetwork nodes. 
##############  Kte denotes testing features extracted from hundreds of subnetwork nodes.

import scipy.io
mat = scipy.io.loadmat('../input/spatialpyramidfeatures4scene15.mat')

type(mat)
#prnt(mat)

for key in mat.keys(): 
    print (key)
    
#import keras
import numpy

labels_raw = mat.get('labelMat')
images_raw = mat.get('featureMat')

labels_trans = numpy.transpose(labels_raw)
images_trans = numpy.transpose(images_raw)
#print(numpy.array_str(y[242]))#.tostring())
##int(numpy.array_str(y[242]), 2)
#print(type(y))
#print(type(y[242]))
##y[1].astype(str)
len(labels_trans)
len(images_trans)
print(images_trans[0])
print(images_trans[1])
print(images_trans[2])

def find(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]
    
labels = []
images = [] 
#a = [1, 2, 3, 1, 2, 3, 1, 2, 3]

for x in range(0, len(labels_trans)):
    labels.append(find(labels_trans[x], lambda x: x == 1))
    images.append(images_trans[x])

images
images
type(labels)
len(images)

import pandas as pd
df = pd.DataFrame(images)
print(df)

labels_int = [] 

for i in range(len(labels)):
    for j in range(len(labels[i])):
        print(labels[i][j], end=' ')
        labels_int.append(labels[i][j])
    print()
    
#results = list(map(int, labels))
df.insert(0,"Labels", labels_int)
print(df)
type(labels_int[0])

train_per_image = 100
#########################sample_G

nclass = df['Labels'].max()
type(int(nclass))

fdatabase_label = df.iloc[:,0]

tr_idx=[]
ts_idx=[]
import random

tr_fea = pd.DataFrame()
ts_fea = pd.DataFrame()
for jj in range(nclass+1):
    print(jj)
    idx_label = find((fdatabase_label), lambda x: x == jj)
    num = len(idx_label)
    tr_num = train_per_image
    idx_label_random = idx_label
    random.shuffle(idx_label_random)
    tr_idx = idx_label_random[:100]
    ts_idx = idx_label_random[100:] ######################################## Do randomize at the last Use random to randomize images and labels
    #tr_idx = idx_label_random[:100]
    #ts_idx = idx_label_random[100:]
    #len(tr_idx)
    #len(ts_idx)
    tr_fea_classwise = df.iloc[tr_idx,:]
    ts_fea_classwise = df.iloc[ts_idx,:]
    tr_fea = tr_fea.append(tr_fea_classwise, ignore_index = True)
    ts_fea = ts_fea.append(ts_fea_classwise, ignore_index = True)
    
Training = tr_fea
Testing = ts_fea

class settings:
    def __init__(self, xmax, xmin, ymax, ymin, yrange, xrange):
        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin
        self.yrange = yrange
        self.xrange = xrange
        pass


def mapminmax(x, ymin=-1.0, ymax=1.0):
    return create(x, ymin, ymax)


def create(x, ymin, ymax):
    xrows = x.shape[0]
    xmin = x.min(1)
    xmax = x.max(1)

    xrange = xmax - xmin
    yrows = xrows
    yrange = ymax - ymin

    gain = yrange / xrange

    fix = np.nonzero(~np.isfinite(xrange) | (xrange == 0))

    if(not all(fix)):
        None
    else:
        gain[fix] = 1
        xmin[fix] = ymin

    return [mapminmax_apply(x, xrange, xmin, yrange, ymin),
            settings(xmax=xmax, xmin=xmin, ymax=ymax, ymin=ymin, yrange=yrange, xrange=xrange)]


def mapminmax_apply(x, xrange, xmin, yrange, ymin):
    gain = yrange / xrange

    fix = np.nonzero(~np.isfinite(xrange) | (xrange == 0))
    if(not all(fix)):
        None
    else:
        gain[fix] = 1
        xmin[fix] = ymin

    cd = np.multiply((np.ones((x.shape[0], x.shape[1]))), xmin.values.reshape(x.shape[0], 1))
    a = x - cd

    b = np.multiply((np.ones((x.shape[0], x.shape[1]))), gain.values.reshape(x.shape[0], 1))
    return np.multiply(a, b) + ymin


class MapMinMaxApplier(object):
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept
    def __call__(self, x):
        return x * self.slope + self.intercept
    def reverse(self, y):
        return (y-self.intercept) / self.slope
    
def mapminmax_rev(x, ymin=-1, ymax=+1):
    x = np.asanyarray(x)
    xmax = x.max(axis=-1)
    xmin = x.min(axis=-1)
    if (xmax==xmin).any():
        raise ValueError("some rows have no variation")
    slope = ((ymax-ymin) / (xmax - xmin))[:,np.newaxis]
    intercept = (-xmin*(ymax-ymin)/(xmax-xmin))[:,np.newaxis] + ymin
    ps = MapMinMaxApplier(slope, intercept)
    return ps(x), ps



import sys
import time
import math
numpy.set_printoptions(threshold=sys.maxsize)

def LayerFirst(train_data, test_data, Elm_Type, NumberofHiddenNeurons, ActivationFunction, C, kkkk, sn, name):
    
    # Macro Defination
    REGRESSION = 0
    CLASSIFIER = 1
    fdafe = 0
    
    # Load training dataset
    ############################# instead of tr_fea put train_data - done
    T = pd.DataFrame(train_data.iloc[:,0]).apply(np.conj).T
    P = train_data.iloc[:,1:len(train_data.columns)].apply(np.conj).T
    OrgP = P
    train_data = None
    
    # Load testing dataset
    ############################# instead of ts_fea put test_data - done
    TV_T = pd.DataFrame(test_data.iloc[:,0]).apply(np.conj).T
    TV_P = test_data.iloc[:,1:len(test_data.columns)].apply(np.conj).T
    test_data = None
    
    NumberofTrainingData = len(P.columns)
    NumberofTestingData = len(TV_P.columns)
    NumberofInputNeurons = len(P)
    
    #Labels Generation for Classification Task
    if Elm_Type!=REGRESSION:
        #Preprocessing the data of classification
        sorted_target = pd.concat([T, TV_T], axis=1, ignore_index=True).reindex(TV_T.index)
        (sorted_target.values).sort()
        #len(sorted_target.columns)
        #sorted_target[101]
        #print(sorted_target.to_string())
        #sorted_target = sorted_target.sort_values(by = 'Labels', axis=1)
        #len(sorted_target.columns)
        #sorted_target[101]
        #print(sorted_target.to_string())
        label = numpy.zeros(shape=(1,15)).astype(int)
        label = pd.DataFrame(label)
        label[0][0] = sorted_target[0][0]
        
        j = 0
        for i in range(0, NumberofTrainingData+NumberofTestingData):
            if sorted_target[i][0] != label[j][0]:
                j = j+1
                label[j][0] = sorted_target[i][0]
        number_class = j + 1
        NumberofOutputNeurons = number_class

        #Preprossesing targets of training
        temp_T = pd.DataFrame(numpy.zeros(shape=(NumberofOutputNeurons,NumberofTrainingData)).astype(int))
        #print(temp_T)
        for i in range(0, NumberofTrainingData):
            for j in range(0, number_class):
                if label[j][0] == T[i][0]:
                    break
            temp_T[i][j] = 1
            
        T = (temp_T*2-1)
        
        #Preprossesing targets of testing
        temp_TV_T = pd.DataFrame(numpy.zeros(shape=(NumberofOutputNeurons,NumberofTestingData)).astype(int))
        #print(temp_T)
        for i in range(0, NumberofTestingData):
            for j in range(0, number_class):
                if label[j][0] == TV_T[i][0]:
                    break
            temp_TV_T[i][j] = 1
            
        TV_T = (temp_TV_T*2-1)   #End of Elm_Type
        
    #Training Part
    NumberofCategory = len(T)
    
    start_time = time.time()
    #print("--- %s seconds ---" % (time.time() - start_time))
    saveT = T
    
    #Calculate weights & biases
    for subnetwork in range(1, sn):
        for j in range(1, kkkk):
            if j == 1:
                count = 1
            else:
                count = 1
            
            for nxh in range(0, count):   ##################### We changed 1 to 0 here for looping prob
                if j == 1:
                #Random generate input weights InputWeight (a_f) and biases
                #(b_f) of the initial subnetwork node  based on equation (2)
                    BiasofHiddenNeurons1 = pd.DataFrame(np.random.uniform(0,1,100)) ############## Replace 100 by NumberofHiddenNeurons 
                    BiasofHiddenNeurons1 = pd.DataFrame(scipy.linalg.orth(BiasofHiddenNeurons1))
                    InputWeight1= pd.DataFrame(np.random.uniform(0,1,size=(100, NumberofInputNeurons))*2-1) ############## Replace 100 by NumberofHiddenNeurons
                    if NumberofHiddenNeurons > NumberofInputNeurons:
                        InputWeight1 = pd.DataFrame(scipy.linalg.orth(InputWeight1))
                    else:
                        InputWeight1 = pd.DataFrame(scipy.linalg.orth(InputWeight1.apply(np.conj).T)).apply(np.conj).T
                        
                        
                    tempH=InputWeight1 @ P ####################################### * is for normal multiplication here, but @ is matrix multiplication. Check if not right
                    ind = pd.DataFrame(numpy.ones(shape=(1,NumberofTrainingData)).astype(int))
                    BiasMatrix = pd.concat([BiasofHiddenNeurons1[:100]]*1500, axis=1, ignore_index=True) #Extend the bias matrix BiasofHiddenNeurons to match the demention of H
                    tempH=tempH+BiasMatrix
                    
                    # initial subnetwork node generation End #
                else:
                    # update a_f and b_f based on equation (4)-(5)
                    if ActivationFunction == 'sig' or ActivationFunction == 'sigmoid':
                        #PP1=(-log((1./PP)-1))  
                        None        #####################################negative log not possible in python
                    if ActivationFunction == 'sin' or ActivationFunction == 'sine':
                        arcsin_complex = lambda t: cmath.asin(t)
                        vfunc = np.vectorize(arcsin_complex)
                        PP1 = vfunc(PP)
                        PP1 = pd.DataFrame(PP)

                    PP=None
                    PP1=PP1.apply(np.real)    # Get error feedback g^{-1}(u_j(P_{c-1})) in euqation (5) 
                    P=P_save
                    H= None
                    a_1 = (pd.DataFrame(np.eye(len(P))) / C + (P @ (P.apply(np.conj).T)))
                    b_1 = (P @ (PP1.apply(np.conj).T))
                    InputWeight1= (pd.DataFrame(np.linalg.solve(a_1, b_1))).apply(np.conj).T   ##############  Check if error comes
                    # input weights calculation in equation (5)

                    fdafe=0;
                    tempH=InputWeight1 @ P
                    
                    YYM_H=InputWeight1 @ pd.concat([P, TV_P], axis=1, ignore_index=True).reindex(TV_P.index) ###########assign if not workonh
                    BB1=PP1.shape  
                    BB2 = pd.DataFrame.sum(pd.DataFrame.sum(tempH-PP1))

                    PP1=None
                   
                    BBP=BB2/BB1[1]   #%%%%%%%%%%%% biases calculation in equation (5) 
                    
                    tempH = (tempH.apply(np.conj).T - BBP.T).apply(np.conj).T  
                    YYM_tempH = (YYM_H.apply(np.conj).T - BBP.T).apply(np.conj).T
                    YYM_H = None
                    
                #Calculation equation (4)-(5) completed %%%%%%
                #Calculate subspace features in equation (6) %%%%%%%
                # Sine

                H = tempH.apply(np.sin)
                tempH = None    #   Release the temparary array for calculation of hidden neuron output matrix H
                BiasMatrix = None
                 
                #Save subspace features $H_i$ in harddisk 
                if j>1:
                    YYM_H = YYM_tempH.apply(np.sin)
                    YYM_H, temp_fea = mapminmax(YYM_H, -1, 1)
                    temp_fea = None
                    H = YYM_H.iloc[:,0:NumberofTrainingData]   # %%%%%% Training features go to the second layer 
                    s2 = "feature_%d.mat" % (subnetwork) # %%%% All the features from Training and Testing are save in the harddisk
                    
                    s2=name + s2
                    scipy.io.savemat(s2,{'struct':YYM_H.values})
                
                # subspace features save End
                P_save = P
                P = H
                H = None
                FT = pd.DataFrame(numpy.zeros(shape=(3,17766)).astype(int))
                #print(FT)
                E1 = T
                
                for i in range(0, 2):
                    Y2 = E1
                    tempH = None
                    # get u_n(y) in equation (3) 
                    if fdafe == 0:
                        if ActivationFunction == 'sig' or ActivationFunction == 'sigmoid':
                            Y22, PS_subnetwork = mapminmax(Y2, 0.01, 0.99)
                        if ActivationFunction == 'sin' or ActivationFunction == 'sine':
                            Y22, PS_subnetwork = mapminmax(Y2, -1, 1)
                            Y22_temp, ps = mapminmax_rev(Y2)
                    else:
                        Y22 = mapminmax_apply(Y2, PS_subnetwork.xrange, PS_subnetwork.xmin, PS_subnetwork.yrange, PS_subnetwork.ymin) ##################### -1 is ymin
                    
                    Y2 = Y22
                    
                    if ActivationFunction == 'sig' or ActivationFunction == 'sigmoid':
                        a = (1./Y2)-1
                        Y4 = -log(a)  ################### log of negative nos dont work in python because of complex nos
                    if ActivationFunction == 'sin' or ActivationFunction == 'sine':
                        #x = np.array([1, 2, 3, 4, 5])
                        arcsin_complex = lambda t: cmath.asin(t)
                        vfunc = np.vectorize(arcsin_complex)
                        Y4 = vfunc(Y2)
                        Y4 = pd.DataFrame(Y4)

                        #Y4 = Y2.apply(np.arcsin)
                    
                    Y4 = Y4.apply(np.real)
                    
                    if fdafe == 0:
                        a = (pd.DataFrame(np.eye(len(P))) / C + (P @ (P.apply(np.conj).T)))
                        b = (P @ (Y4.apply(np.conj).T))
                        YYM = pd.DataFrame(np.linalg.solve(a, b))   ################################### Check this step maybe wrong (\-matlab) operator in python 
                        YJX = ((YYM.apply(np.conj).T) @ P).apply(np.conj).T
                    else:
                        a = Y4.apply(np.conj).T
                        eye = pd.DataFrame(np.eye(len(YYM)))
                        YYM_conj = YYM.apply(np.conj).T
                        a_ling = eye/C+YYM @ YYM_conj
                        ling_solve = np.linalg.solve(a_ling , YYM)
                        ling_solve_df = pd.DataFrame(ling_solve).apply(np.conj).T
                        PP = a @ ling_solve_df
                        PP = PP.apply(np.conj).T  # get error feedback 
                        
                        YJX = (PP.apply(np.conj).T) @ YYM
                        
                    BB1 = Y4.shape
                    BB2 = pd.DataFrame(pd.DataFrame.sum(YJX - (Y4.apply(np.conj).T))).T
                    BB = BB2/BB1[1]
                    BB = BB[0]
                    
                    BB = pd.DataFrame(np.full((1500,15), BB[0]))
                    
                    GXZ111 = (YJX.apply(np.conj).T - BB.T).apply(np.conj).T       ############################# Values check not trusted
                    
                    if ActivationFunction == 'sig' or ActivationFunction == 'sigmoid':
                        None #GXZ2=1./(1+exp(-GXZ111')); ########################### Do the conversion
                    if ActivationFunction == 'sin' or ActivationFunction == 'sine':
                        GXZ2=(GXZ111.apply(np.conj).T).apply(np.sin)
                    
                    FYY = pd.DataFrame(ps.reverse(GXZ2.values)).apply(np.conj).T     ############## Using 2 minmax of different writers. Probably wrong here
                    
                    if i==0:
                        FT1 = FYY.apply(np.conj).T   ################### FT{i} take 2 dataframes
                        E1 = E1 - FT1
                    if i==1:
                        FT2 = FYY.apply(np.conj).T
                        E1 = E1 - FT2
                    
                    if i==0:
                        fdafe=1
            
            PP = PP+P
        
        T=E1
        P=OrgP
        fdafe=0
    return NumberofTrainingData

def eval_kernel(feaset_1, feaset_2, kernel, kparam):
    if(feaset_1.shape[1] != feaset_2.shape[1]):
        print('Error in shape(eval_kernel)')
    (L1, dim) = feaset_1.shape
    (L2, dim) = feaset_2.shape
    if kernel == 'plus':
        K = feaset_1.apply(np.conj).T + feaset_2.apply(np.conj).T
    if kernel == 'linear':
        K = feaset_1 @ feaset_2.apply(np.conj).T
    else:
        print('Unknown Kernel')

    return K
    ############################################################ 
    ############################################################ Other cases needed to be updated

def featurecomb(Ktr, Kte, name, sn, NumberofTrainingData):
    for loop in range(1, sn):
        s2 = "feature_%d.mat" % (loop) 
        s2 = name + s2
        feature_load = scipy.io.loadmat(s2)
        YYM_H = feature_load.get('struct')
        YYM_H = pd.DataFrame(YYM_H)
        H_train1 = YYM_H.iloc[:,0:NumberofTrainingData]
        H_test1 = YYM_H.iloc[:,NumberofTrainingData:]
        
        Ktr_temp = eval_kernel(H_train1.apply(np.conj).T, H_train1.apply(np.conj).T, 'linear', 1)
        Ktr = Ktr + Ktr_temp

        H_test1 = H_test1.apply(np.conj).T
        H_test1 = H_test1.reset_index(drop=True)
        
        Kte_temp = eval_kernel(H_test1, H_train1.apply(np.conj).T, 'linear', 1)
        Kte = Kte + Kte_temp
        YYM_H = None
        
    return Ktr,Kte

def lastLayer(TrainingData_File, TestingData_File, Elm_Type, ActivationFunction, kkk, C):

    # Macro Defination
    REGRESSION = 0
    CLASSIFIER = 1
    
    # Load training dataset
    train_data = TrainingData_File
    T = pd.DataFrame(train_data.iloc[:,0]).apply(np.conj).T
    P = train_data.iloc[:,1:len(train_data.columns)].apply(np.conj).T
    train_data = None
    
    # Load testing dataset
    test_data = TestingData_File
    TV_T = pd.DataFrame(test_data.iloc[:,0]).apply(np.conj).T
    TV_P = test_data.iloc[:,1:len(test_data.columns)].apply(np.conj).T
    test_data = None
    
    NumberofTrainingData = len(P.columns)
    NumberofTestingData = len(TV_P.columns)
    
    #Labels Generation for Classification Task
    if Elm_Type!=REGRESSION:
        #Preprocessing the data of classification
        sorted_target = pd.concat([T, TV_T], axis=1, ignore_index=True).reindex(TV_T.index)
        (sorted_target.values).sort()
        label = numpy.zeros(shape=(1,15)).astype(int)
        label = pd.DataFrame(label)
        label[0][0] = sorted_target[0][0]
        ############################################################ Check for loops range
        j = 0
        for i in range(0, NumberofTrainingData+NumberofTestingData):
            if sorted_target[i][0] != label[j][0]:
                j = j+1
                label[j][0] = sorted_target[i][0]
        number_class = j + 1
        NumberofOutputNeurons = number_class

        #Preprossesing targets of training
        temp_T = pd.DataFrame(numpy.zeros(shape=(NumberofOutputNeurons,NumberofTrainingData)).astype(int))
        #print(temp_T)
        for i in range(0, NumberofTrainingData):
            for j in range(0, number_class):
                if label[j][0] == T[i][0]:
                    break
            temp_T[i][j] = 1
            
        T = (temp_T*2-1)
        
        #Preprossesing targets of testing
        temp_TV_T = pd.DataFrame(numpy.zeros(shape=(NumberofOutputNeurons,NumberofTestingData)).astype(int))
        #print(temp_T)
        for i in range(0, NumberofTestingData):
            for j in range(0, number_class):
                if label[j][0] == TV_T[i][0]:
                    break
            temp_TV_T[i][j] = 1
            
        TV_T = (temp_TV_T*2-1)   #End of Elm_Type

    # Training Part %%%%%%%%%%%%%%%%%

    D_YYM_i = pd.DataFrame()
    shape1 = T.shape
    Y = pd.DataFrame(numpy.zeros(shape=T.shape).astype(int))
    E1=T

    for i in range (1, kkk):
        Y2=E1

        # Get $e_{c-1}$ in equation (9) %%%%%%%%
        
        if ActivationFunction == 'sig' or ActivationFunction == 'sigmoid':
            Y22, PS_i = mapminmax(Y2, 0.01, 0.99)
            Y22_temp, ps = mapminmax_rev(Y2)
        if ActivationFunction == 'sin' or ActivationFunction == 'sine':
            Y22, PS_i = mapminmax(Y2, 0, 1)
            Y22_temp, ps = mapminmax_rev(Y2)

        Y2=Y22.apply(np.conj).T

        if ActivationFunction == 'sig' or ActivationFunction == 'sigmoid':
            a = (1./Y2)-1
            Y4 = -log(a)  ################### log of negative nos dont work in python because of complex nos
        if ActivationFunction == 'sin' or ActivationFunction == 'sine':
            #arcsin_complex = lambda t: cmath.asin(t)
            #vfunc = np.vectorize(arcsin_complex)
            #Y4 = vfunc(Y2)
            #Y4 = pd.DataFrame(Y4).apply(np.conj).T
            Y4 = pd.DataFrame(Y2.apply(np.arcsin)).apply(np.conj).T
        # End %%%%%%%%

        #Get input weights of a subnetwork node $a^c_p$ in euqation (9) %%%%
        P = P.reset_index(drop=True)
        a = (pd.DataFrame(np.eye(len(P))) / C + P @ (P.apply(np.conj).T))
        b = (P @ (Y4.apply(np.conj).T))
        YYM = pd.DataFrame(np.linalg.solve(a, b))

        D_YYM_i = YYM ################################ Check dataframe if erroe

        YJX = P.apply(np.conj).T @ YYM

        BB1=Y4.shape  
        BB2 = pd.DataFrame.sum(pd.DataFrame.sum(YJX-Y4.apply(np.conj).T))
        BB =BB2/BB1[1]

        GXZ111 = P.apply(np.conj).T @ YYM - BB
        
        if ActivationFunction == 'sig' or ActivationFunction == 'sigmoid':
            None  ################### log of negative nos dont work in python because of complex nos
        if ActivationFunction == 'sin' or ActivationFunction == 'sine':
            GXZ2 = (GXZ111.apply(np.conj).T).apply(np.sin)

        FYY = pd.DataFrame(ps.reverse(GXZ2.values))

        # End %%%%%%
        # updated training error %%%%

        FT1=FYY
        E1=E1-FT1   ################ Check on this
        # End 

        Y=Y+FYY  #Total output 

        # Training accuracy calculation 
        if Elm_Type == CLASSIFIER:
            MissClassificationRate_Training=0
            for i in range(0, len(T.columns)):
                x, label_index_expected = T.iloc[:,i].max(0), T.iloc[:,i].argmax(0)
                x, label_index_actual = Y.iloc[:,i].max(0), Y.iloc[:,i].argmax(0)
            if label_index_actual!=label_index_expected:
                MissClassificationRate_Training=MissClassificationRate_Training+1

            TrainingAccuracy=1-MissClassificationRate_Training/len(T.columns)

        # End 

    # Training Part End 

    
    #  Testing part begian 
    temp = pd.DataFrame(numpy.zeros(shape=TV_T.shape).astype(int))
    TY2= temp*TV_T;

    # Get output for testing samples
    for i in range(1,kkk):
        TV_P = TV_P.reset_index(drop=True)
        GXZ1= (D_YYM_i.apply(np.conj).T) @ TV_P - BB 
        if ActivationFunction == 'sig' or ActivationFunction == 'sigmoid':
            None  ################### log of negative nos dont work in python because of complex nos
        if ActivationFunction == 'sin' or ActivationFunction == 'sine':
            GXZ2 = (GXZ1.apply(np.conj).T).apply(np.sin)
        
        FYY = pd.DataFrame(ps.reverse((GXZ2.apply(np.conj).T).values))
        
        TY2=TY2+FYY

    # End 
    # Testing accuracy calculation 
    if Elm_Type == CLASSIFIER:
        MissClassificationRate_Testing=0
        for i in range(0, len(TV_T.columns)):
            x, label_index_expected = TV_T.iloc[:,i].max(0), TV_T.iloc[:,i].argmax(0)
            x, label_index_actual = TY2.iloc[:,i].max(0), TY2.iloc[:,i].argmax(0)
            if label_index_actual!=label_index_expected:
                MissClassificationRate_Testing=MissClassificationRate_Testing+1

        TestingAccuracy=1-MissClassificationRate_Testing/len(TV_T.columns)

    # End 

    # Training Part End 
    return TrainingAccuracy, TestingAccuracy

###################################### main
for x in range(10):
    name = "scene15_channel_%d" %x
    num_subnetwork_node = 4
    dimension = 100
    C1 = 2**8
    
    NumberofTrainingData = LayerFirst(Training, Testing, 1, dimension, 'sine', C1, 3, num_subnetwork_node, name)

    (Ktr, Kte) = featurecomb(Ktr, Kte, name, 4, NumberofTrainingData)

Training =  pd.concat([Training.iloc[:,0], Ktr], axis=1, ignore_index=True).reindex(Ktr.index)
Testing = pd.concat([Testing.iloc[:,0], Kte], axis=1, ignore_index=True).reindex(Kte.index)

C2 = 2 ** 12
(train_accuracy11, test_accuracy) = lastLayer(Training, Testing, 1, 'sin', 2, C2)

print(test_accuracy*100)