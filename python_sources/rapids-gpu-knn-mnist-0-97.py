#!/usr/bin/env python
# coding: utf-8

# # kNN has 600x SPEEDUP! using GPU with RAPIDS cuML!
# 
# ![1-24-20-header2.jpg](attachment:1-24-20-header2.jpg)
# 
# In order to predict one test image in Kaggle's MNIST competition using kNN, we must multiply the test image which is a vector of length 784 by all 42,000 training vectors (of length 784) in the training set. This is 33 million multiplies. To predict all test images, we must do this for all 28,000 test images for a total of 1 trillion multiplies! 
# 
# A 3GHz single core CPU does 3 billion multiplies per second and therefore takes 300 seconds or 5 minutes to infer all test images. In comparison, a GPU with 1500 CUDA cores operating at 1.5GHz can do 2.2 trillion multiplies per second and therefore takes 0.5 seconds to infer all the test images. That's 750 times faster!
#   
# ![1-24-20-dot3.jpg](attachment:1-24-20-dot3.jpg)
#   
# In this kernel, we witness RAPIDS cuML's kNN utilize Kaggle's GPU (Nvidia Tesla P100 with 3500 CUDA cores) to infer all test images in an incredible 2.5 seconds. For comparision, Scikit-learn's kNN uses Kaggle's CPU (Intel Xeon with 2 cores) and infers all test images in 1500 seconds (25 minutes). We witness a 600x speedup using RAPIDS cuML!
# 
# You can learn more about Nvidia's RAPIDS library [here][1]. The RAPIDS library allows us to perform all our data science on GPUs. The library cuDF provides Pandas functionality on GPU, and cuML provides Scikit-learn functionality on GPU. Other libraries like cuGraph, cuSignal, cuSpatial provide additional machine learning tools.
# 
# [1]: https://rapids.ai/

# In[ ]:


get_ipython().run_cell_magic('time', '', '# INSTALL RAPIDS FROM KAGGLE DATASET. TAKES 1 MINUTE :-)\nimport sys\n!cp ../input/rapids/rapids.0.11.0 /opt/conda/envs/rapids.tar.gz\n!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz\nsys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path\n!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[ ]:


# LOAD LIBRARIES
import cudf, cuml
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, KFold
from cuml.neighbors import KNeighborsClassifier, NearestNeighbors
import matplotlib.pyplot as plt
print('cuML version',cuml.__version__)


# # Load Data

# In[ ]:


# LOAD TRAINING DATA
train = cudf.read_csv('../input/digit-recognizer/train.csv')
print('train shape =', train.shape )
train.head()


# In[ ]:


# VISUALIZE DATA
samples = train.iloc[5000:5030,1:].to_pandas().values
plt.figure(figsize=(15,4.5))
for i in range(30):  
    plt.subplot(3, 10, i+1)
    plt.imshow(samples[i].reshape((28,28)),cmap=plt.cm.binary)
    plt.axis('off')
plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.show()


# # Grid Search kNN for optimal k
# Here we perform grid search with a 20% holdout set to find the best parameter `k`. Alternatively, we could use full KFold cross validation shown below. We find that `k=3` achieves the best validation accuracy.

# In[ ]:


get_ipython().run_cell_magic('time', '', "# CREATE 20% VALIDATION SET\nX_train, X_test, y_train, y_test = train_test_split(train.iloc[:,1:], train.iloc[:,0],\\\n        test_size=0.2, random_state=42)\n\n# GRID SEARCH FOR OPTIMAL K\naccs = []\nfor k in range(3,22):\n    knn = KNeighborsClassifier(n_neighbors=k)\n    knn.fit(X_train, y_train)\n    # Better to use knn.predict() but cuML v0.11.0 has bug\n    # y_hat = knn.predict(X_test)\n    y_hat_p = knn.predict_proba(X_test)\n    acc = (y_hat_p.to_pandas().values.argmax(axis=1)==y_test.to_array() ).sum()/y_test.shape[0]\n    #print(k,acc)\n    print(k,', ',end='')\n    accs.append(acc)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# COMPUTE NEIGHBORS\nrow = 5; col = 7; sft = 10\nknn = NearestNeighbors(n_neighbors=col)\nknn.fit(X_train)\ndistances, indicies = knn.kneighbors(X_test)\n# DISPLAY NEIGHBORS\ndisplayV = X_test.to_pandas().iloc[sft:row+sft].values\ndisplayT = X_train.to_pandas().iloc[indicies[sft:row+sft].to_pandas().values.flatten()].values\nplt.figure(figsize=(15,row*1.5))\nfor i in range(row):\n    plt.subplot(row,col+1,(col+1)*i+1)\n    plt.imshow(displayV[i].reshape((28,28)),cmap=plt.cm.binary)\n    if i==0: plt.title('Unknown\\nDigit')\n    for j in range(col):\n        plt.subplot(row, col+1, (col+1)*i+j+2)\n        plt.imshow(displayT[col*i+j].reshape((28,28)),cmap=plt.cm.binary)\n        if i==0: plt.title('Known\\nNeighbor '+str(j+1))\n        plt.axis('off')\nplt.subplots_adjust(wspace=-0.1, hspace=-0.1)\nplt.show()")


# In[ ]:


# PLOT GRID SEARCH RESULTS
plt.figure(figsize=(15,5))
plt.plot(range(3,22),accs)
plt.title('MNIST kNN k value versus validation acc')
plt.show()


# # KFold Grid Search
# For a more accurate grid search we could use KFold instead of a single holdout set. If we do this, we find it gives similar results as above.

# In[ ]:


get_ipython().run_cell_magic('time', '', "# GRID SEARCH USING CROSS VALIDATION\nfor k in range(3,6):\n    print('k =',k)\n    oof = np.zeros(len(train))\n    skf = KFold(n_splits=5, shuffle=True, random_state=42)\n    for i, (idxT, idxV) in enumerate( skf.split(train.iloc[:,1:], train.iloc[:,0]) ):\n        knn = KNeighborsClassifier(n_neighbors=k)\n        knn.fit(train.iloc[idxT,1:], train.iloc[idxT,0])\n        # Better to use knn.predict() but cuML v0.11.0 has bug\n        # y_hat = knn.predict(train.iloc[idxV,1:])\n        y_hat_p = knn.predict_proba(train.iloc[idxV,1:])\n        oof[idxV] =  y_hat_p.to_pandas().values.argmax(axis=1)\n        acc = ( oof[idxV]==train.iloc[idxV,0].to_array() ).sum()/len(idxV)\n        print(' fold =',i,'acc =',acc)\n    acc = ( oof==train.iloc[:,0].to_array() ).sum()/len(train)\n    print(' OOF with k =',k,'ACC =',acc)")


# # Predict Test
# Below we witness GPU RAPIDS kNN infer the entire Kaggle test dataset of 28,000 images against a training set of 48,000 images in an incredible 2.5 seconds (using Kaggle's Nvidia Tesla P100)! For comparison, a CPU (Intel Xeon 2 core) takes 600 times longer!

# In[ ]:


# LOAD TEST DATA
test = cudf.read_csv('../input/digit-recognizer/test.csv')
print('test shape =', test.shape )
test.head()


# In[ ]:


# FIT KNN MODEL
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train.iloc[:,1:785], train.iloc[:,0])


# In[ ]:


get_ipython().run_cell_magic('time', '', '# PREDICT TEST DATA\n# Better to use knn.predict() but cuML v0.11.0 has bug\n# y_hat = knn.predict(test)\ny_hat_p = knn.predict_proba(test)\ny_hat = y_hat_p.to_pandas().values.argmax(axis=1)')


# In[ ]:


# SAVE PREDICTIONS TO CSV
sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
sub.Label = y_hat
sub.to_csv('submission_cuML.csv',index=False)
sub.head()


# In[ ]:


# PLOT PREDICTION HISTOGRAM
plt.hist(sub.Label)
plt.title('Distribution of test predictions')
plt.show()


# # Submit to Kaggle
# When we submit to Kaggle, we see that our LB score 0.968 (test accuracy) matches our CV score 0.967 (cross validation accuracy). We achieve a prediction accuracy of 97% with only 2.5 seconds of work! Amazing!
#   
# ![1-24-20-cuML.png](attachment:1-24-20-cuML.png)

# # Comparison to CPU kNN
# For comparison, we see that using CPU kNN takes 1500 seconds (25 minutes) on an Intel Xeon 2 core processor. That's 600x slower than GPU kNN! Below we infer 1000 of the 28,000 test images and it takes 54 seconds.

# In[ ]:


# TRAIN SKLEARN KNN MODEL
from sklearn.neighbors import KNeighborsClassifier as K2
knn = K2(n_neighbors=3,n_jobs=2)
knn.fit(train.iloc[:,1:].to_pandas(), train.iloc[:,0].to_pandas())


# In[ ]:


get_ipython().run_cell_magic('time', '', "# PREDICT 1/28 OF ALL TEST IMAGES WITH CPU\ny_hat = knn.predict(test.iloc[:1000,:].to_pandas())\nprint('Here we only infer 1000 out of 28,000 test images on CPU')")


# In[ ]:





# In[ ]:




