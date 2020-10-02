# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import scipy.linalg as la

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statistics
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

import pandas as pd
url = "../input/Iris.csv"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
# Any results you write to the current directory are saved as output.
# from sklearn.preprocessing import StandardScaler
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
sl = df.loc['1':, features[0]].values     #Sepal Length
sw = df.loc['1':, features[1]].values     #Sepal Width
pl = df.loc['1':, features[2]].values     #Petal Length
pw = df.loc['1':, features[3]].values     #Petal Width

# Separating out the target
y = df.loc[:,['target']].values

# Finding Sizes
sls = sl.size-1
sws = sw.size-1
pls = pl.size-1
pws = pw.size-1

# Converting into floats to perform Mathematics
for i in range (0,sls+1):
    sl[i]=float(sl[i])
    sw[i]=float(sw[i])
    pl[i]=float(pl[i])
    pw[i]=float(pw[i])


# Finding Mean
slm=statistics.mean(sl)
swm=statistics.mean(sw)
plm=statistics.mean(pl)
pwm=statistics.mean(pw)
# print(sl[sls])
# print(sl[0]*10)
# print('SL Mean = ', slm, 'SW Mean = ', swm, 'PL Mean = ', plm, 'PW Mean = ', pwm)

# New Data by subtracting the mean
sln=sl-slm
swn=sw-swm
pln=pl-plm
pwn=pw-pwm

# print(sln)
# print(x[150,0])
# print(x[150,1])
# print(x[150,2])
# print(x[150,3])

print('SLn Type= ', type(sln))
sln = list(sln)
swn = list(swn)
pln = list(pln)
pwn = list(pwn)
print('Now SLn Type = ', type(sln))
# newfts = [sln, swn, pln, pwm]

# Covariance matrix
cov_mat = np.stack((sln, swn, pln, pwn))  
cm=np.cov(cov_mat)
print('Covariance matrix of new data:')
print(cm)
print(type(cm))

# Eigen values & Eigen Vectors
eigvals, eigvecs = la.eig(cm)
print(eigvals)
print(eigvecs)
print(type(eigvals))
print(type(eigvecs))
print('Breaking eigen vectors:')
eigvec0=eigvecs[:,0]
eigvec1=eigvecs[:,1]
eigvec2=eigvecs[:,2]
eigvec3=eigvecs[:,3]
print(eigvec0)
print(eigvec1)
print(eigvec2)
print(eigvec3)
eigvecb=[eigvec0, eigvec1, eigvec2, eigvec3]
print(type(eigvecb))
eigvals=eigvals.real
meigvals=max(eigvals)
eigvals = eigvals.tolist()



# meigvals=meigvals.real
es=[10, 8, 100, 3, 4000, 5]
el1=len(eigvals)
el=len(es)
# maxpos=0
mps=0
def minimum(eigvals, el1): 
    mps = eigvals.index(max(eigvals)) 
    print("The maximum is at position", mps)

print("Maximum eigen value = ", meigvals)
# minimum(es, el)
minimum(eigvals, el1)
mev=eigvecb[mps]
print("Maximum eigen vector is: ", mev)
print(eigvals)
print(type(eigvals))
print(type(es))
print("New features:")
print(sln[3])
# print(type(newfts))
# print(newfts[0,2])
# print(newfts[2,2])
# print(newfts[3,2])

# data1=mev*newfts
# print("New data:")
# print(data1)



# frame = pd.DataFrame(data=data, columns=["SLN1","SWN1","PLN1","PWN1"])

# data = list(data)
# print(data[0,3]*2)
# print(type(data))


# eigvals=eigvals.sort()
# print(type(eigvals))
print(eigvals)
print(eigvecb[0])
print(eigvecb[1])
print(eigvecb[2])
print(eigvecb[3])

eigvecb.sort(key=lambda x: x[1])
eigvals.sort()
print(eigvals)
print(eigvecb)
eigvecbnew = np.array([eigvecb])
print(type(eigvecb))
print(type(eigvecbnew))

# New Data
data = np.array([sln, swn, pln,pwn])
# print(data[0,3]*2)
print(type(data))
to=4
md = mev.dot(data)
print(md.size)
print(data.size)
print(md)

# print(type(data[0,3]*2))
# eigvecb.sort()
# eigvecb=eigvecb.sort()
# Fazoooooooooooooooooool
# print(x[:, 0])
# Standardizing the features
# x = StandardScaler().fit_transform(x)
# print(x)
