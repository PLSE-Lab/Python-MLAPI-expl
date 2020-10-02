#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle


file_list = []
class_list = []

DATADIR = '/kaggle/input/trainocr'

# All the categories you want your neural network to detect
CATEGORIES = ["A", "B", "C", "D", "E",
	      "F", "G", "H", "I", "J",
	      "K", "L", "M", "N", "P", "Q","R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


# The size of the images that your neural network will use
IMG_SIZE = 30
ims=2

for category in CATEGORIES :
    path = os.path.join(DATADIR, category)
    path = os.path.join(path, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))


training_data = []
data={}


def create_training_data():
    for category in CATEGORIES :
        path = os.path.join(DATADIR, category)
        path = os.path.join(path, category)
		#class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                gray = cv2.cvtColor(new_array, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                edged = cv2.Canny(blur, 10, 100)
                training_data.append([(edged.flatten()).astype(int), category])
                #data.update({'IMAGE ARRAY' : new_array, 'LETTER': category})
            except Exception as e:
                pass
            
create_training_data()
print(len(training_data))


# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# c=0
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         c+=1
#         print(os.path.join(dirname, filename))
# print(c)

# Any results you write to the current directory are saved as output.


# In[ ]:



a=[]
for j in range(0,len(training_data)):
    a.append([])
    for i in range(0,len(training_data[j][0])):
        a[j].append(training_data[j][0][i])
        if i == len(training_data[j][0])-1:
            a[j].append(training_data[j][1][0])
       
        
print(len(a))


# In[ ]:


import pandas as pd
df2=pd.DataFrame(a)
df2.to_csv('trainingdataset.csv', index=False) 
dataset = pd.read_csv('trainingdataset.csv')


# In[ ]:



X = dataset.iloc[:, 0:899].values
y = dataset.iloc[:, 900].values
print((X))


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])])

dataset = np.array(ct.fit_transform(y), dtype=np.float)

print(dataset)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# In[ ]:


print((y))


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors =7, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# In[ ]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=25)
kmeans.fit(X_train, y_train)
kmeans.labels_


# In[ ]:


y_pred = kmeans.predict(X_test)


# In[ ]:


print(set(y_pred))


# In[ ]:


dataset['target'] = dataset['900']


# In[ ]:


del dataset['900']


# In[ ]:


dataset.head()


# In[ ]:


labels=np.array([kmeans.labels_])
LABEL_COLOR_MAP = {0:'A', 1:'B', 2:'C', 3:'D', 4: 'E', 5:'F', 6: 'G', 7:'H', 8:'I', 9: 'J', 10:'K', 11: 'L', 12: 'M', 13: 'N', 14: 'P', 15: 'Q', 16: 'R',17: 'S', 18:'T', 19:'U', 20:'V', 21:'W', 22:'X', 23:'Y', 24: 'Z'}
label_color = [LABEL_COLOR_MAP[l] for l in labels]
plt.scatter(y_pred, y_test, s=100, c=label_color)
plt.show()


# In[ ]:


plt.scatter(y_pred==0, y_test, s=100, c='red', label ='A')
plt.scatter(y_pred==1, y_test, s=100, c='blue', label ='B')
plt.scatter(y_pred==2, y_test, s=100, c='green', label ='C')
plt.scatter(y_pred==3, y_test, s=100, c='cyan', label ='D')
plt.scatter(y_pred==4, y_test, s=100, c='magenta', label ='E')
plt.scatter(y_pred==5, y_test, s=100, c='red', label ='F')
plt.scatter(y_pred==6, y_test, s=100, c='blue', label ='G')
plt.scatter(y_pred==7, y_test, s=100, c='green', label ='H')
plt.scatter(y_pred==8, y_test, s=100, c='cyan', label ='I')
plt.scatter(y_pred==9, y_test, s=100, c='purple', label ='J')
plt.scatter(y_pred==10, y_test, s=100, c='indigo', label ='K')
plt.scatter(y_pred==11, y_test, s=100, c='wheat', label ='L')
plt.scatter(y_pred==12, y_test, s=100, c='brown', label ='M')
plt.scatter(y_pred==13, y_test, s=100, c='darkslategray', label ='N')
plt.scatter(y_pred==14, y_test, s=100, c='olive', label ='P')
plt.scatter(y_pred==15, y_test, s=100, c='orange', label ='Q')
plt.scatter(y_pred==16, y_test, s=100, c='pink', label ='R')
plt.scatter(y_pred==17, y_test, s=100, c='chocolate', label ='S')
plt.scatter(y_pred==18, y_test, s=100, c='black', label ='T')
plt.scatter(y_pred==19, y_test, s=100, c='yellow', label ='U')
plt.scatter(y_pred==20, y_test, s=100, c='lime', label ='V')
plt.scatter(y_pred==21, y_test, s=100, c='darkseagreen', label ='W')
plt.scatter(y_pred==22, y_test, s=100, c='rosybrown', label ='X')
plt.scatter(y_pred==23, y_test, s=100, c='darkorange', label ='Y')
plt.scatter(y_pred[y_pred==24], y_test, s=100, c='slateblue', label ='Z')
plt.show()


# In[ ]:


plt.scatter(y_pred, y_test, s=100, c='slateblue', label ='Z')
plt.show()


# In[ ]:


from matplotlib import cm
cmap = cm.get_cmap('gnuplot'),
scatter = pd.plotting.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[ ]:


print(cm)


# In[ ]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
  
# actual = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0] 
# predicted = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0] 
results = confusion_matrix(y_test, y_pred)
  
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',accuracy_score(y_test, y_pred))
print('Report : ')
print(classification_report(y_test, y_pred))


# In[ ]:


from matplotlib.colors import ListedColormap
plt.scatter(y_test, y_pred)
plt.show()


# In[ ]:


a=[]
for i in range(0,4560):
    a.append(i)


# In[ ]:


print(len(y_test))


# In[ ]:



fig, ax = plt.subplots()
myplot = ax.scatter( y_pred,y_test, c=25, cmap=plt.cm.Reds)

plt.show()


# In[ ]:


fig, ax = plt.subplots()
myplot = ax.scatter( y_train, y_test, c=a, cmap=plt.cm.Reds, vmin=0, vmax = 599)

plt.show()


# In[ ]:


IMG_SIZE=30
z=0
z = cv2.imread('/kaggle/input/ocr-testing-set/Z/Z/4.jpg')
new_array = cv2.resize(z, (IMG_SIZE, IMG_SIZE))
gray = cv2.cvtColor(new_array, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blur, 10, 100)
z_test=(edged.flatten()).astype(int)
# cv2.imshow("img",z)
# cv2.waitKey(0)


# In[ ]:


a_test=[]
for j in range(0,1):
    a_test.append([])
    for i in range(0,len(z_test)):
        a_test[j].append(z_test[i])


# In[ ]:


print(a_test)


# In[ ]:


import pandas as pd
df_test=pd.DataFrame(a_test)
df_test.to_csv('trainingdataset_test.csv', index=False) 
dataset_test = pd.read_csv('trainingdataset_test.csv')


# In[ ]:


print(df_test)


# In[ ]:


testing_z = dataset_test.iloc[:,0:899].values


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
testing_z = sc.fit_transform(testing_z)


# In[ ]:


print((testing_z.shape))


# In[ ]:


z_pred2 = classifier.predict(testing_z)


# In[ ]:


yz_test = [13]


# In[ ]:


print(z_pred2)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yz_test,z_pred2)


# In[ ]:


fig, ax = plt.subplots()
myplot = ax.scatter( z_pred2,yz_test, c=yz_test, cmap=plt.cm.Reds, vmin=0, vmax = 1)

plt.show()


# In[ ]:


print(len(y_train))


# In[ ]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
# X1= np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 899].max() + 1, step = 0.01))
# plt.contourf(X1,  classifier.predict(np.array([X1.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

