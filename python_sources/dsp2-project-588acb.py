#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
from sklearn.externals.six.moves import xrange
from sklearn.cluster import KMeans
import time
import keras


# In[ ]:


def get_mnist_reduced(mnist, number_of_examples): #number of examples is 1000 in case of train and 100 in case of test
    mnist_reduced = pd.DataFrame()
    for i in range(10):
        mnist_reduced = pd.concat([mnist_reduced, mnist[(mnist.iloc[:, 0] == i).values][0:number_of_examples]], axis = 0)
#    return mnist_reduced.iloc[:, 1:].values.reshape(-1, 28, 28), mnist_reduced.iloc[:, 0].values
    return mnist_reduced.iloc[:, 1:].values, mnist_reduced.iloc[:, 0].values


# In[ ]:


def load_data():
    train_mnist = pd.read_csv('../input/mnist-original/mnist_train.csv')
    X_train, y_train = get_mnist_reduced(train_mnist, 1000)
    test_mnist = pd.read_csv('../input/mnist-original/mnist_test.csv')
    X_test, y_test = get_mnist_reduced(test_mnist, 100)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data()
   #print("Y_train classes values are:\n", y_train.value_counts()) #remove .values from get_mnist_reduced to make this works
    #print("Y_test classes values are:\n", y_test.value_counts())
classes=list(dict.fromkeys(y_test))
     


# In[ ]:


print(classes)


# In[ ]:


def plot_image(img, label = None):
    plt.axis('off')
    plt.imshow(img.reshape(28, 28), cmap = 'gray')
    if label is not None:
        plt.title("number is " + str(label))
    plot_image(X_train[1100], y_train[1100])


# # **Confusion Matrix Function only call** confusion_matrix(y_test,y_pred,classes,t)

# In[ ]:


def plot_confusion_matrix(cm, labels=[],title="confusion matrix"):
    cmap=plt.cm.Greens
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


def confusion_matrix(y_test,y_pred,classes,t):
        conf = np.zeros([len(classes),len(classes)])
        confnorm = np.zeros([len(classes),len(classes)])
        for i in range(y_test.shape[0]):
            j = y_test[i]
            k = y_pred[i]
            conf[j,k] = conf[j,k] + 1
        for i in range(len(classes)):
            confnorm[i,:] = conf[i,:] / np.sum(conf[i,:]) 
            #print(confnorm)  
        plt.figure()
        plot_confusion_matrix(confnorm, labels=classes,title=t)


# In[ ]:


DCT_features_train = dct(X_train)[:, :120] #get DCT features
DCT_features_test = dct(X_test)[:, :120] #get DCT features


# In[ ]:


pca_first_120 = PCA(n_components = 120)#to calculate 120 features
pca_first_120.fit(X_train)
pca_train_120 = pca_first_120.transform(X_train)
pca_test_120 = pca_first_120.transform(X_test)


# In[ ]:


pca_variance_bigger_than_95 = PCA(n_components = .95)
pca_variance_bigger_than_95.fit(X_train)
pca_train_variance_95 = pca_variance_bigger_than_95.transform(X_train)
pca_test_variance_95 = pca_variance_bigger_than_95.transform(X_test)


# In[ ]:


data_transformations = {'NO': [X_train, X_test],
                            'DCT': [DCT_features_train, DCT_features_test],
                              'PCA120': [pca_train_120, pca_test_120], 
                                'PCA95%': [pca_train_variance_95, pca_test_variance_95]}


# In[ ]:


x1, x2 = data_transformations['NO']
print(x1.shape, x2.shape)


# # **GMM CLASSIFIER** **Functions**
# 

# In[ ]:


#GMM 
def GMM_Classifier(n_components,classes_n,Features_Train,class_margin):
    G=[]
    for i in range (classes_n):  
        G_temp=GMM(n_components=n_components,n_init=10,max_iter=5000,covariance_type='full').        fit(Features_Train[i*class_margin:i*class_margin+class_margin-1])
        G.append(G_temp.means_)
    G=np.array(G)
    return G                                            #return means
#Predict
def predict_label(test_features,label_set,model):
    Y_predict=np.zeros_like(label_set)
    for i in range (Y_predict.shape[0]):
        Y_predict[i]=find_class(test_features[i],model)
    return Y_predict
#class decision
def find_class(x,y):                                   #finding label
    min_d=np.ones(y.shape[0])*100000000.0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            temp=np.linalg.norm(x-y[i][j])
            if temp<min_d[i]:
                min_d[i]=temp
    min_class_idx=np.argmin(min_d)
    return min_class_idx



# In[ ]:


n_components=[1,2,4];
n_samples_per_class=np.count_nonzero(y_train == 1)
n_classes=len(np.unique(y_train))
for key in data_transformations.keys():
    X_train, X_test = data_transformations[key]
    for n_component in n_components:
        tic = time.time()
        classifier=GMM_Classifier(n_component,n_classes,X_train,n_samples_per_class)
        y_pred=predict_label(X_test,y_test,classifier)
        toc = time.time()
        confusion_matrix(y_test,y_pred,classes,"confusion matrix Trans.:{} - n:{}".format(key,n_component))
        print("accuracy ={:.2f}% for {} Transformation no. of GMM Components={}"
        .format(accuracy_score(y_test,y_pred)*100,key,n_component))
        print("elapsed time =",round(toc-tic,2),"sec")


# In[ ]:


#this show for non transforming data, it will be shown also below but without linear kernel 
""""
for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    t0 = time.time()
    classifier = SVC(kernel = kernel)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    confusion_matrix(y_test,y_pred,classes,"confusion matrix Trans.:{} - Kernal:{}".format(key,kernel))
    print("after time: {}s, Accuracy score is:{}, for Kernel:{}".format(time.time() - t0,
                                                                accuracy_score(y_train, y_pred), kernel))
"""


# In[ ]:



for key in data_transformations.keys():
    for kernel in ['poly', 'rbf', 'sigmoid']:
        t0 = time.time()
        classifier = SVC(kernel = kernel)
        X_train, X_test = data_transformations[key]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        confusion_matrix(y_test,y_pred,classes,"confusion matrix Trans.:{} - Kernal:{}".format(key,kernel))
        print("after time: {}, Accuracy score is: {}, for Kernel: {} and data transformation: {}".format(time.time() - t0,
                                                                accuracy_score(y_test, y_pred), kernel, key))


# In[ ]:


n_digits = len(np.unique(y_test))
print(n_digits)

# Initialize KMeans model
kmeans = KMeans(n_clusters = n_digits)

# Fit the model to the training data
kmeans.fit(X_train)
KMeans( init='k-means++',
       max_iter=300, n_clusters=10,
        n_init=10, random_state=None,tol=0.0,
        verbose=0)


# In[ ]:


def infer_cluster_labels(kmeans, actual_labels):
    """
    Associates most probable label with each cluster in KMeans model
    returns: dictionary of clusters assigned to each label
    """

    inferred_labels = {}

    for i in range(kmeans.n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]

        #print(labels)
        #print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))
        
    return inferred_labels  

def infer_data_labels(X_labels, cluster_labels):
    """
    Determines label for each array, depending on the cluster it has been assigned to.
    returns: predicted labels for each array
    """
    
    # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
    
    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
                
    return predicted_labels
# test the infer_cluster_labels() and infer_data_labels() functions
cluster_labels = infer_cluster_labels(kmeans, y_train)
X_clusters = kmeans.predict(X_train)
predicted_labels = infer_data_labels(X_clusters, cluster_labels)
print (predicted_labels)
print (y_train)


# In[ ]:


from sklearn import metrics
def calculate_metrics(estimator, data, labels):
    # Calculate and print metrics
    print('Number of Clusters: {}'.format(estimator.n_clusters))
    print('Inertia: {}'.format(estimator.inertia_))
    print('Homogeneity: {}'.format(metrics.homogeneity_score(labels, estimator.labels_)))


# In[ ]:


clusters = [10, 20, 40, 80, 160]

# test different numbers of clusters
for n_clusters in clusters:
    estimator = KMeans(n_clusters = n_clusters)
    estimator.fit(X_train)
    
    # print cluster metrics
    calculate_metrics(estimator, X_train, y_train)
    
    # determine predicted labels
    cluster_labels = infer_cluster_labels(estimator, y_train)
    predicted_Y = infer_data_labels(estimator.labels_, cluster_labels)
    
    # calculate and print accuracy
    print('Accuracy: {}\n'.format(metrics.accuracy_score(y_train, predicted_Y)))


# In[ ]:


n_clusters=[10,20,40,80,160];
n_samples_per_class=np.count_nonzero(y_train == 1)
n_classes=len(np.unique(y_train))
for key in data_transformations.keys():
    X_train, X_test = data_transformations[key]
    for n_cluster in n_clusters:
        tic = time.time()
        estimator = KMeans(n_clusters = n_cluster)
        estimator.fit(X_train)
    
    # print cluster metrics
        #calculate_metrics(estimator, X_train, y_train)
    
    # determine predicted labels
        cluster_labels = infer_cluster_labels(estimator, y_train)
        predicted_Y = infer_data_labels(estimator.labels_, cluster_labels)
        metrics.accuracy_score(y_train, predicted_Y)
        toc = time.time()
        confusion_matrix(y_test,predicted_Y,classes,"confusion matrix Trans.:{} - n:{}".format(key,n_cluster))
        #print("accuracy ={:.2f}% for {} Transformation no. of kmean_clusters={}"
        #.format(accuracy_score(y_test,y_pred)*100,key,n_cluster))
        print("elapsed time =",round(toc-tic,2),"sec")
        print('Accuracy: {}\n'.format(metrics.accuracy_score(y_train, predicted_Y)))

