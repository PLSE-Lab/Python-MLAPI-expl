import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time
import os
# np.random.seed(998244353)
# torch.manual_seed(998244353)

# Configurations
OLD_INDEX = ['Pclass','Sex','Age','UknAge','SibSp','Parch','Fare','Embarked','Survived']
NEW_INDEX = ['Age', 'UknAge', 'Fare',
             'Pclass_0', 'Pclass_1', 'Pclass_2',
             'Sex_0', 'Sex_1',
             'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8',
             'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Parch_9',
             'Embarked_0', 'Embarked_1', 'Embarked_2',
             'Survived'
            ]
MAP_Sex = {'male':0,'female':1}
MAP_Embarked = {'C':0,'Q':1,'S':2}
ONE_HOT = [[1,0],[0,1]]
FEATURES = 26
ACTIVATION = {'sigmoid':nn.Sigmoid(),
              'tanh':nn.Tanh(),
              'ReLU':nn.ReLU(),
              'softplus':nn.Softplus(),
              'LeakyReLU':nn.LeakyReLU(),
              'logsigmoid':nn.LogSigmoid(),
              'PReLU':nn.PReLU(),
              'ReLU6':nn.ReLU6(),
              'softsign':nn.Softsign()
             }
EPOCH = 500
INPUT_PATH = "../input/titanic/"
OUTPUT_PATH = "/kaggle/working/"

def preprocess( data ):
    # Data Cleaning
    data = pd.DataFrame(data,columns=OLD_INDEX)
    data['UknAge'] = data['UknAge'].fillna(0)
    data['Survived'] = data['Survived'].fillna(0)
    #### print(data[data['Age'].isnull()])
    data.loc[data['Age'].isnull(),'UknAge'] = 1
    data['Age'] = data['Age'].fillna(0)
    #### print(data[data['Fare'].isnull()])
    data['Fare'] = data['Fare'].fillna(14.4)
    #### print(data[data['Embarked'].isnull()])
    data['Embarked'] = data['Embarked'].fillna('C')
    #### One-hot Encoding
    data['Pclass'] -= 1
    data['Sex'] = data['Sex'].map(MAP_Sex)
    data['Embarked'] = data['Embarked'].map(MAP_Embarked)
    data = pd.get_dummies(data,columns=['Pclass','Sex','SibSp','Parch','Embarked'])
    data = pd.DataFrame(data,columns=NEW_INDEX)
    data = data.fillna(0)
    #### Normalization
    for col in NEW_INDEX:
        maximum = data[col].max()
        if maximum > 0:
            data[col] /= maximum
    #### To List
    temp = np.array(data)
    data = [[torch.FloatTensor(temp[j][:FEATURES]),
             torch.FloatTensor(ONE_HOT[int(temp[j][FEATURES])])] for j in range(temp.shape[0])]
    return data

class FCDNN( nn.Module ):

    def __init__( self, nodes, activation ):
        super(FCDNN,self).__init__()
        self.layers = nn.Sequential()
        self.name = 'FCDNN'
        for i in range(len(nodes)-1):
            self.layers.add_module('fc-{0}'.format(i),nn.Linear(nodes[i],nodes[i+1]))
            self.layers.add_module('activation-{0}'.format(i),ACTIVATION[activation])
            self.name += str(nodes[i])+'-'
        self.name += str(nodes[len(nodes)-1])+','+activation+')'
        self.layers.add_module('softmax',nn.Softmax(0))

    def forward( self, x ):
        x = self.layers(x)
        return x

    def train( self, loss_func, optimizer, eta, decay, train_data, validate_data, epoch, batch_num, batch_size ):
        print('['+self.name+' {0}*{1}*{2}'.format(epoch,batch_num,batch_size)+'] with optimizer ['+str(type(optimizer))+']:')
        n = len(train_data)
        m = len(validate_data)
        for E in range(epoch):
            np.random.shuffle(train_data)
            p = 0
            for T in range(batch_num):
                optimizer.zero_grad()
                for j in range(batch_size):
                    loss = loss_func(self.forward(train_data[p][0]),train_data[p][1])
                    loss.backward()
                    p = (p+1)%n
                optimizer.step()
            eta *= decay
            if eta < 1e-6:
                break
            for p in optimizer.param_groups:
                p['lr'] *= decay
        with torch.no_grad():
            L = 0.
            R = 0.
            for j in train_data:
                t = self.forward(j[0])
                L += loss_func(t,j[1])
                R += float((t[0]<t[1]) != (j[1][0]<j[1][1]))
            print("\tTraining Loss = %.6f"%(L/n))
            print("\tTraining Error Rate = %.4f%%"%(R/n*100))
            L = 0.
            R = 0.
            for j in validate_data:
                t = self.forward(j[0])
                L += loss_func(t,j[1])
                R += float((t[0]<t[1]) != (j[1][0]<j[1][1]))
            print("\tValidation Loss = %.6f"%(L/m))
            print("\tValidation Error Rate = %.4f%%"%(R/m*100))

    def evaluate( self, validate_data, target_data ):
        with torch.no_grad():
            R = 0.
            for j in validate_data:
                t = self.forward(j[0])
                R += float((t[0]<t[1]) != (j[1][0]<j[1][1]))
            R /= len(validate_data)
            weight = math.log(math.sqrt((1-R)/R))
            prediction = torch.zeros(len(target_data),2)
            for i in range(len(target_data)):
                prediction[i] = self.forward(target_data[i][0])

            # Special Limitation
            if R > 0.2:
                print("Discarded!")
                return torch.zeros(len(target_data),2)
            return prediction*weight

def output( prediction, filedir ):
    submission = []
    for i in range(892,1310):
        submission.append([i,int(prediction[i-892][0]<prediction[i-892][1])])
    submission = pd.DataFrame(submission)
    submission.columns = ['PassengerId','Survived']
    submission.to_csv(filedir,index=0)

def RandomFCDNN( train_data, validate_data ):
    layers = [FEATURES]
    L = np.random.randint(1,5)
    for j in range(L):
        layers.append(2 ** np.random.randint(1,7))
    layers.append(2)
    A = np.random.choice(list(ACTIVATION))
    E = 0.0001
    W = np.random.randint(1,8)
    B = 1
    D = np.random.choice([1,2,4,5,8,10,16,20,25,32])
    N = FCDNN(layers,A)
    O = np.random.choice([optim.SGD(N.parameters(),lr=E,momentum=np.random.random()*0.9),
                          optim.Adam(N.parameters(),lr=E),
                          optim.Adagrad(N.parameters(),lr=E),
                          optim.RMSprop(N.parameters(),lr=E,momentum=np.random.random()*0.9)
                         ])
    N.train(nn.MSELoss(),O,E,B,train_data,validate_data,EPOCH*W,800//D,D)
    return N.evaluate(validate_data,target_data)



origin_data = preprocess(pd.read_csv(INPUT_PATH+"train.csv"))
target_data = preprocess(pd.read_csv(INPUT_PATH+"test.csv"))
# predictions = evaluate_gender_submission(validate_data,target_data)
predictions = torch.zeros(len(target_data),2)

begin = time.time()

while 1:
    np.random.shuffle(origin_data)
    train_data = origin_data[:800].copy()
    validate_data = origin_data[800:].copy()

    start = time.time()
    predictions += RandomFCDNN(train_data,validate_data)
    output(predictions,OUTPUT_PATH+"submission.csv")
    end = time.time()
    print('Time cost: %.4f s, total time cost: %.4f s'%(end-start,end-begin))