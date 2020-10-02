import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

qd_f = pd.read_csv('../input/uci-iris/iris.data', sep=",", header = None)
qd_arr = qd_f.values

#print(qd_f)

k = qd_arr[:,0].size
print(k)

classA = []
classB = []

for i in range(k):
    if(i<50):
        classA.append(qd_arr[i,:])
        
    elif(i>100):
        classB.append(qd_arr[i,:])

classA = np.array(classA)
classB = np.array(classB) 


a_x = classA[:,0]
a_y = classA[:,2]
b_x = classB[:,0]
b_y = classB[:,2]


plt.scatter(a_x, a_y, color = 'red', marker = '^')
plt.scatter(b_x, b_y, color = 'blue', marker = '*')

plt.show()

d = classA[:,0].size
m = classB[:,0].size

def nClassify(X1,X2):

     arr1=(X1*X1,X2*X2,X1*X2,X1,X2,1)
     arr1 = np.array(arr1)
     return arr1
 
    
for i in range(d):
         print(nClassify(classA[i,0],classA[i,2]))
        
        
for i in range(m):
         print(nClassify(classB[i,0],classA[i,2]))
         
                  
w_one = np.ones((1, 6),dtype=int)

alpha_list = [];
one_at_a_time1 = [];

def one_at_a_time_func(w, alpha):
    check = 0;
    count = 0;

    while check != 6:
        count = count + 1;
        check = 0;
        for i in range(6):
            g = np.dot(d[i, :], w.transpose());
            if g <= 0:
                w = w + alpha * y[i, :];
            else:
                check = check + 1;
        if check == 6:
            break;
    return count;
         

