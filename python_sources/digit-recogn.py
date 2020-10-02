#!/usr/bin/env python
# coding: utf-8

# # Disclaimer
# 
# The data in this notebook is mostly copied from https://www.kaggle.com/archaeocharlie/a-beginner-s-approach-to-classification . I intended to do modification later to the tutorial, so please permit me for using it.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Library Import
# For starter import any machine libary we wanted to use. SKLearn is good choice for beginner, the question is what the algorithm we interested to test. Here's what we are going to need:
# 1. At least a classification algorithm (SVM or Decision Tree is a Good Choice)
# 2. Matplotlib
# 3. Preprocessing tools
# 4. Train test split And since we have been import numpy and panda no need to import them.

# In[ ]:


import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm, tree


# # Load Data
# load data dengan menggunakan library pandas .
# 1.  Gunakan pandas [read_csv](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) untuk membaca train.csv ke dalam [DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html).
# 2.  Memisahkan* images* dan *labels* untuk *supervised learning*.
# 3.  Melakukan [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) untuk membagi data menjadi 2 set, satu untuk *training* dan satu lagi untuk *testing*.

# In[ ]:


# Load Data
# Baca file .csv ke dalam DataFrame
labeled_images = pd.read_csv('../input/train.csv')       
# iloc (index location) , menyeleksi berdasarkan posisi
# labeled_images.iloc[0:5000,1:], yang terseleksi kedalam variabel images baris ke-0 sampai 4999 dan kolom ke-1 sampai kolom terakhir
images = labeled_images.iloc[0:5000,1:]
# labeled_images.iloc[0:5000,:1], yang terseleksi kedalam variabel labels baris ke-0 sampai 4999 dan kolom ke-0
labels = labeled_images.iloc[0:5000,:1]
# split array ke dalam himpunan bagian random train dan test
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)


# In[ ]:


print(train_labels.head())
#print(train_labels.index[1])
#print(train_labels.iloc[0])
#print(train_labels.iloc[0].values)


# # Q1
# Notice in the above we used _images.iloc?, can you confirm on the documentation? what is the role?
# 
# **Jawab :**
# 
# Menggunakan [iloc](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iloc.html) untuk menyeleksi berdasarkan posisi.
# 
# *  `labeled_images.iloc[0:5000,1:]` , data yang terseleksi kedalam variabel *images* dari baris ke-0 sampai 4999 dan kolom ke-1 sampai kolom terakhir.
# *   `labeled_images.iloc[0:5000,:1]` , data yang terseleksi kedalam variabel *labels* dari baris ke-0 sampai 4999 dan kolom ke-0.

# # Melihat gambar dari setiap kelas angka
# *  *Image* disini masih berupa 1 dimensi, kemudian diubah menjadi 2 dimensi dengan menggunakan [numpy array ](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html)dan [reshape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html)
# *  Plot gambar dengan menggunakan [pyplot.plot](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) dari library [matplotlib](https://matplotlib.org/api/index.html)

# In[ ]:


# now we gonna load the second image, reshape it as matrix than display it
i=1
# select i row 
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])


# # Q2
# Now plot an image for each image class
# 
# **Jawab :**
# *  Plot gambar dari setiap kelas , kelas digit punya 10 kelas, yaitu 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 dan 9.

# In[ ]:


# np.unique (untuk return elemen unik dari sebuah array yang sudah di sort)
number = np.unique(train_labels)
print (number)
# indeks manual
indeks = [12,4,18,2,42,9,1,0,11,7]
# indeks otomatis
indexnum = [0,0,0,0,0,0,0,0,0,0]
# mencari indeks dari setiap kelas angka 
# dengan looping sebanyak baris dari train_labels
for i in range (len(train_labels)):
    # variabel labelnum untuk menyimpan kelas angka, dan untuk menemukan index nya menggunakan i
    # (iloc[i] mencari baris ke i dan .label untuk mencari di dalam kolom label)
    labelnum = train_labels.iloc[i].label
    # isi yang berada di dalam indexnum akan direplace dengan lokasi indeks dari setiap kelas angka dari 0-9
    indexnum[labelnum] = i
    
for i in indexnum:
    plt.figure()
    img=train_images.iloc[i].values
    img=img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(train_labels.iloc[i,0])


# Now plot the histogram within img

# In[ ]:


#train_images.iloc[i].describe()
i=1
print(type(train_images))
print(type(train_labels))
plt.hist(train_images.iloc[i])


# # Q3
# Can you check in what class does this histogram represent?. How many class are there in total for this digit data?. How about the histogram for other classes
# 
# **Jawab :**
# * [Histogram](https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist) di atas menunjukan bahwa x nilai pixel dan y frekuensi dari  1 sampel kelas angka yaitu angka 6.

# In[ ]:


labelcount = train_labels["label"]
print(labelcount.value_counts())


# In[ ]:


# create histogram for each class (data merged per class)
# Todo
#print(train_labels.iloc[:5])
#data1 = train_images.iloc[1]
#data2 = train_images.iloc[3]
#data1 = np.array(data1)
#data2 = np.array(data2)
#data3 = np.append(data1,data2)
#print(len(data3))
#plt.hist(data3)


# In[ ]:


label = [[],[],[],[],[],[],[],[],[],[]]
for j in range(10):
    for i in range(len(train_images)):
        if (train_labels.iloc[i].label == j):
            data = train_images.iloc[i]
            data = np.array(data)
            label[j] = np.append(label[j],data)
            
    plt.figure(j)
    plt.hist(label[j])
    plt.title(j)
        


# ## Train the model
# 1. Gunakan modul [sklearn.svm](http://scikit-learn.org/stable/modules/svm.html) untuk membuat [vector classifier](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
# 2. Melakukan fitting dengan menggunakan metode [fit](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.fit).
# 3. Melihat indikasi akurasi (0-1) dengan menggunakan metode [score](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.score).

# In[ ]:


# Define model
clf = svm.SVC()
# Fit: Capture patterns from provided data.
clf.fit(train_images, train_labels.values.ravel())
# Determine how accurate the model's
clf.score(test_images,test_labels)


# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(random_state=0)
tree.fit(train_images, train_labels)
test_predict = tree.predict(test_images)
print(mean_absolute_error(test_labels, test_predict))


# # Q4
# In above, did you see score() function?, open SVM.score() dokumentation at SKLearn,what does it's role?. Does it the same as MAE discussed in class previously?.Ascertain it through running the MAE. Now does score() and mae() prooduce the same results?.
# 
# **Jawab :**
# 
# 1. Fungsi [score()](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.score) pada svm.SVC menghasilkan rata rata akurasi dari test images dan test labels yang diberikan. hasil dari score adalah **0.1**. Rata-rata akurasi semakin mendekati angka 1 semakin baik. sedangkan
#  
# 2. Fungsi [mean_absolute_error()](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) menghasilkan rata rata error dari test labels dan prediksi dari test images. hasil dari mean absolute error adalah **1.033**. rata rata error akan semakin baik jika nilainya mendekati angka 0.

# In[ ]:


print(train_labels.values.ravel())
print(np.unique(test_labels)) # to see class number


# # Improving Performance
# Did you noticed, that the performance is so miniscule in range of ~0.1. Before doing any improvement, we need to analyze what are causes of the problem?. But allow me to reveal one such factor. It was due to pixel length in [0, 255]. Let's see if we capped it into [0,1] how the performance are going to improved.

# In[ ]:


test_images[test_images>0]=1
train_images[train_images>0]=1
img=train_images.iloc[i].values.reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])


# In[ ]:


# now plot again the histogram
plt.hist(train_images.iloc[i])


# # Retrain the model
# Using the now adjusted data, let's retrain our model to see the improvement

# In[ ]:


clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
scoresvm = clf.score(test_images,test_labels)
print (scoresvm)


# # Q5
# Based on this finding, Can you explain why if the value is capped into [0,1] it improved the performance significantly?. Perharps you need to do several self designed test to see why.
# 
# **Jawab :**
# 
# Menyederhanakan gambar dengan membuatnya benar benar hitam dan putih (1,0), sebelum nilai pixel di ubah menjadi 1 dan 0 , gambar mengandung warna abu abu menjadikan gambar agak sedikit tidak jelas atau blur, tetapi ketika nilai pixel hanya 0 dan 1 , gambar menjadi jelas dikarenakan warnanya hanya hitam dan putih,  dengan demikian nilai akurasinya meningkat. 

# # Prediction labelling
# In Kaggle competition, we don't usually submit the end test data performance on Kaggle. But what to be submitted is CSV of the prediction label stored in a file.

# In[ ]:


# Test again to data test
test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:5000])


# In[ ]:


# separate code section to view the results
print(results)
print(len(results))


# In[ ]:


df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)


# In[ ]:


#check if the file created successfully
print(os.listdir("."))


# # Data Download
# 
# We have the file, can listed it but how we are take it from sever. Thus we also need to code the download link.

# In[ ]:


# from https://www.kaggle.com/rtatman/download-a-csv-file-from-a-kernel

# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(df)


# # Q6
# Alhamdulillah, we have completed our experiment. Here's things to do for your next task:
# * What is the overfitting factor of SVM algorithm?. Previously on decision tree regression, the factor was max_leaf nodes. Do similar expriment using SVM by seeking SVM documentation!
# *  Apply Decision Tree Classifier on this dataset, seek the best overfitting factor, then compare it with results of SVM.
# * Apply Decision Tree Regressor on this dataset, seek the best overfitting factor, then compare it with results of SVM & Decision Tree Classifier. Provides the results in table/chart. I suspect they are basically the same thing.
# * Apply Decision Tree Classifier on the same dataset, use the best overfitting factor & value. But use the unnormalized dataset, before the value normalized to [0,1]
# 
# 

# *  Untuk mendapatkan nilai gamma dan c yang optimal , saya menggunakan [grid search](http://http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html ) .
# *  Grid search digunakan untuk pencarian menyeluruh atas nilai parameter yang ditentukan untuk estimator.
# 

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Set the parameters by cross-validation
parameters = {'gamma': [0.01, 0.001, 0.0001],'C': [1, 10, 100,1000]}

# Create a classifier object with the classifier and parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid = parameters)
clf.fit(train_images,train_labels.values.ravel())


# In[ ]:


print('Best C:',clf.best_estimator_.C) 
print('Best Gamma:',clf.best_estimator_.gamma)


# In[ ]:


#final svm
best_c = 10
best_gamma = 0.01
clf_final = svm.SVC(C=best_c,gamma=best_gamma)
clf_final.fit(train_images, train_labels.values.ravel())
finalsvm = clf_final.score(test_images,test_labels)
print(clf_final.score(test_images,test_labels))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
def get_mae_train_classifie(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_train = model.predict(train_X)
    mae = mean_absolute_error(train_y, preds_train)
    return(mae)

def get_mae_test_classifie(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


maee_train=[]
maee_test= []
leaf_nodes=[5,25,50,70,100,300,500,1000,3000,5000,7000]
for max_leaf_nodes in leaf_nodes:
    my_maetrain = get_mae_train_classifie(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    my_maetest = get_mae_test_classifie(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    maee_train.append(my_maetrain)
    maee_test.append(my_maetest)

plt.figure()
plt.plot(leaf_nodes,maee_test,color="red",label='Validation')
plt.plot(leaf_nodes,maee_train,color="blue",label='Training')
plt.xlabel("Tree Depth")
plt.ylabel("MAE")
plt.title("Decision Tree Classifier")
plt.legend()
plt.show()

print (maee_train)
print (maee_test)


# In[ ]:


#final model dtclassifier

best_tree_size=1000
treeclassifie = DecisionTreeClassifier(max_leaf_nodes=best_tree_size, random_state=0)
treeclassifie.fit(train_images,train_labels)
scoredtc = treeclassifie.score(test_images,test_labels)
print ("DTC = ",scoredtc)
print ("SVM = ",finalsvm)


# In[ ]:


# Decision Tree Regressor
def get_mae_train_regress(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
    model.fit(train_X, train_y)
    preds_train = model.predict(train_X)
    mae = mean_absolute_error(train_y, preds_train)
    return(mae)

def get_mae_test_regress(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

maee_train=[]
maee_test= []
leaf_nodes=[5,25,50,70,100,300,500,1000,3000,5000,7000]
for max_leaf_nodes in leaf_nodes:
    my_maetrain = get_mae_train_regress(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    my_maetest = get_mae_test_regress(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    maee_train.append(my_maetrain)
    maee_test.append(my_maetest)

plt.figure()
plt.plot(leaf_nodes,maee_test,color="red",label='Validation')
plt.plot(leaf_nodes,maee_train,color="blue",label='Training')
plt.xlabel("Tree Depth")
plt.ylabel("MAE")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

print (maee_train)
print (maee_test)


# In[ ]:


#final model dtregressor
best_tree_size=1000
treeregres = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)
treeregres.fit(train_images,train_labels)
scoredtr = treeregres.score(test_images,test_labels)
print ("DTR = ",scoredtr)
print ("DTC = ",scoredtc)
print ("SVM = ",finalsvm)


# 
# No | Modul | Score
# --- | --- | ---
# 1 | Decision Tree Regressor| 0.586
# 2 | Decision Tree Classifier | 0.779
# 3 | SVM | 0.94

# In[ ]:


# Decision Tree Classifier
# labeled_images.iloc[0:5000,1:], yang terseleksi kedalam variabel images baris ke-0 sampai 4999 dan kolom ke-1 sampai kolom terakhir
images = labeled_images.iloc[0:5000,1:]
# labeled_images.iloc[0:5000,:1], yang terseleksi kedalam variabel labels baris ke-0 sampai 4999 dan kolom ke-0
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
best_tree_size=1000
tree = DecisionTreeClassifier(max_leaf_nodes=best_tree_size,random_state=0)
tree.fit(train_images, train_labels)
test_predict = tree.predict(test_images)
#print(mean_absolute_error(test_labels, test_predict))
scoredtr_ver2 = tree.score(test_images,test_labels)
print ("DTC = ",scoredtr_ver2)

