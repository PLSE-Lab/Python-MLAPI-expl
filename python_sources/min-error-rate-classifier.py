import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m
import numpy.linalg as md
from mpl_toolkits.mplot3d import Axes3D

p_train = pd.read_csv('../input/Testset.txt', header=None, sep=',', dtype='float64')
train_arr = p_train.values
len_train = train_arr[:, 0].size

class_1 = []
class_2 = []

data=pd.read_csv('../input/Dataset_csv.csv',sep=',')
columnName = ["x1","x2","class"]

data.columns = columnName
classA = [data[data["class"]==1.0]["x1"],data[data["class"]==1.0]["x2"]]
classA = np.array(classA)
classB = [data[data["class"]==2.0]["x1"],data[data["class"]==2.0]["x2"]]
classB = np.array(classB)
#print(classA)

m1 =  np.mean(classA, axis=1)
m2 =  np.mean(classB, axis=1)
#print(m2)

sg1 = np.cov(classA)
sg2 = np.cov(classB)

row = []
row1 = []

def pdf(u,s,train_arr):
    s_i = np.linalg.inv(s)
    p = pow(6.2832,2)
    sig= md.det(s)
    p = p*sig
    sqt  = m.sqrt(p)
    dim = 1/sqt
    sub = train_arr - u
    sub_t  =np.array(sub).T   
    e  = np.dot(sub,s_i)
    e_1  =np.dot(e,sub_t)
    e_111  = -0.5 * e_1
    exp  = m.exp(e_111)
    w1= exp*dim
    return w1
        
for i in range(len_train):   
     w1 = pdf(m1,sg1,train_arr[i,:])
     w2 = pdf(m2,sg2,train_arr[i,:])

     if(w1>w2):
        class_1.append(train_arr[i,:])
      
     else:
        class_2.append(train_arr[i,:])


class_1 = np.array(class_1)
class_2 = np.array(class_2)
print(class_1)

plt.scatter(class_1[:, 0],class_1[:, 1], color='red',label='class_1', marker='o')
plt.scatter(class_2[:, 0], class_2[:, 1], color='blue',label='class_2', marker='^')
plt.legend()
plt.show()


x = np.linspace(0,12,300)
y = np.linspace(0,12,300)

X, Y = np.meshgrid(x,y)
Z = np.zeros_like(X)
D= np.zeros_like(X)
len_x = len(x)
len_y = len(y)
for i in range(len_x):
    for j in range(len_y):  
       w1 = pdf(m1,sg1,np.array([x[i],y[j]]))
       w2 = pdf(m2,sg2,np.array([x[i],y[j]]))
       if(w1>w2):
        Z[j][i] = w1
      
       else:
         Z[j][i] = w2
       D[j][i] = w1-w2

       
fig = plt.figure()
graph = Axes3D(fig)
graph.scatter(class_1[:,0], class_1[:,1],  c='r', marker='o')
graph.scatter(class_2[:,0], class_2[:,1],  c='b', marker='^')

graph.plot_surface(X, Y, Z, rstride=8, cstride=8, linewidth=1, antialiased=True, cmap='ocean', alpha=0.3)
graph.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap='ocean', alpha=0.3)
graph.contour3D(X, Y, D, zdir='z', offset=-0.15, cmap='ocean')
graph.set_zlim(-0.15, 0.5)
graph.set_zticks(np.linspace(0,0.3,7))
graph.view_init(azim=240)
graph.set_xlabel('X')
graph.set_ylabel('Y')
graph.set_zlabel('Probability Density')
graph.legend()

plt.show()














