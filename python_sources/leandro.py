import numpy as np
import pandas as pd
import csv

umbral = 0.5
tasa_de_aprendizaje = 0.1

pesos = []
for i in range(0,784):
    pesos.append(0)

some_list = [1,2,3,4]
print (map(lambda a: a*2, some_list))

x = []
conjunto_de_entrenamiento = [((1, 0, 0), 1), ((1, 0, 1), 1), ((1, 1, 0), 1), ((1, 1, 1), 0)]

with open("../input/train.csv",newline='') as train:
     reader = csv.reader(train)
     next(reader)
     #for row in reader:
         #x.append((map(int, row[1:]),int(row[0])))
         #x.append((((int(i) for i in row[1:])),int(row[0])))
         #x.append((map(lambda a:int(a),row[1:])))
         #,int(row[0])))
        #lista = lista.append(row[
     #your_list = list(reader)

#print (x)

    #for row in spamreader:
     #   print(.join(row))
        
        
        
        
#dataset = pd.read_csv("../input/train.csv")
#target = dataset[[0]].values.ravel()
#train = dataset.iloc[:,22:].values
#datatest = pd.read_csv("../input/test.csv")
#test = datatest.iloc[:,21:].values

#rf.fit(train, target)

#print(model.feature_importances_)

#np.savetxt('resultado.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')


