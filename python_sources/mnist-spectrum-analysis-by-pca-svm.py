#!/usr/bin/env python
# coding: utf-8

# ## PCA, Spectrum Analysis and SVM
# ## Background
# Almost everyone predicts digit by CNN which easily gives more than 99% accuracy. It is still curious, however, how good performance of other methods such as Support Vector Machine especially if images are converted by Principal Component Analysis.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv", dtype = "int16")
test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv", dtype = "int16")
sample_submission = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv", dtype = "int16")
train_y = train_df.iloc[:,0]
all_df = np.concatenate((train_df.values[:,1:], test_df.values))
n_train = train_y.shape[0]
n_test = test_df.shape[0]


# In[ ]:


n_train, n_test


# ## 1. Variances and Eigenvalues
# To check effectiveness of PCA, variances of original data and eigenvalues shall be compared.

# In[ ]:


Cov_mat = np.cov(all_df, rowvar = False, bias = True)


# In[ ]:


variances = np.sort(np.diagonal(Cov_mat))[::-1]
variances_pca = np.sort(np.linalg.eigvalsh(Cov_mat))[::-1]


# In[ ]:


cumsum_var = np.cumsum(variances)
cumsum_var_pca =  np.cumsum(variances_pca)


# In[ ]:


fig, ax = plt.subplots(figsize = (8,5))
ax = plt.plot(cumsum_var/cumsum_var[-1], label = "original", lw = 2)
ax = plt.plot(cumsum_var_pca/cumsum_var_pca[-1], label = "pca", lw = 2)
plt.legend(fontsize=14)
plt.ylabel("Fraction of total variance",fontsize=14)
plt.xlabel("Dimensions",fontsize=14)
plt.show()


# In[ ]:


ori_90_percent = np.where(cumsum_var/cumsum_var[-1] > 0.9)[0][0]
pca_90_percent = np.where(cumsum_var_pca/cumsum_var_pca[-1] > 0.9)[0][0]
ori_95_percent = np.where(cumsum_var/cumsum_var[-1] > 0.95)[0][0]
pca_95_percent = np.where(cumsum_var_pca/cumsum_var_pca[-1] > 0.95)[0][0]
print("Before PCA, 90% of total variances is in", ori_90_percent, "dimensions")
print(" After PCA, 90% of total variances is in", pca_90_percent, "dimensions")
print("Before PCA, 95% of total variances is in", ori_95_percent, "dimensions")
print(" After PCA, 95% of total variances is in", pca_95_percent, "dimensions")


# ### Result
# By PCA, the numbers of dimension can be reduced from 784 to 86. It is really significant reduction if the prediction still gives high accuracy. Before machine learning, let's see what is going on in detail when images are transformed into principal components (PCA).

# ## 2. Spectrum Analysis by PCA
# First, to calculate eigenvalues and eigenvectors.

# In[ ]:


eigenvalues, eigenvectors = np.linalg.eigh(Cov_mat)


# ### 2.1 Mode Vectors
# 24 eigenvectors (mode vectors) are plotted in 28x28 images. All images consist of those eigenvectors times eigenvalues.
# Btw, mode1 looks like "0"!. So, the digit "0" must have strong amplitude in mode1. Lets see the results.

# In[ ]:


fix1, ax1 = plt.subplots(3,8, figsize = (16,7))
for i in range(0,24):
    y = i % 8
    x = int(i / 8)
    ax1[x, y].imshow(np.reshape(eigenvectors[:,-(i+1)], (28,28)), cmap = "gray")
    ax1[x, y].set_title("mode" + str((i+1)))  
    ax1[x, y].set_xticks([], [])
    ax1[x, y].set_yticks([], [])
plt.show()


# This function randomly choose 3 samples of specified value. Then, it plots original image and amplitudes of each spectra

# ### 2.2 Spectrum Analysis

# In[ ]:


def spectrum_plot_multiple(value):
    sample = np.random.choice(np.where(train_y == value)[0],3)
    range_x = 20
    fig3, ax3 = plt.subplots(3,2, figsize = (15,16))
    for i in range(3):
        
        ax3[i,0].imshow(np.reshape(all_df[sample[i],:],(28,28)), cmap = "gray")
        ax3[i,0].set_title("Original Image", fontsize=14)
        ax3[i,1].bar(range(1,range_x+1),np.flip(np.dot(all_df[sample[i],:], eigenvectors)[-range_x:]))
        ax3[i,1].set_xlabel("Modes", fontsize=14)
        ax3[i,1].set_ylabel("Amplitude", fontsize=14)
    
    plt.show()    


# In[ ]:


spectrum_plot_multiple(value = 0)


# As anticipated, digit 0 has strong response (netative response) in the 1st mode. 

# In[ ]:


spectrum_plot_multiple(value = 1)


# In[ ]:


spectrum_plot_multiple(value = 2)


# In[ ]:


spectrum_plot_multiple(value = 4)


# In[ ]:


spectrum_plot_multiple(value = 9)


# ### 2.3 Projection to eigenvectors
# Image can be re-constructed by following function "pca_projection". According to the result of projection in 86 and 153 dimensions, it is still possible to see which digit they are. Moreover, almost no difference between 86 and 153. Parhaps, just 5% of variance does not have significant impact at least for human's perceptions.

# In[ ]:


def pca_projection(mode, sample):
    
    U = eigenvectors[:,-mode:]
    U2 = np.dot(all_df[sample,:], U)
    U.shape, U2.shape, (U2*U).shape
    P = np.sum(U2*U, axis = 1)
    return np.reshape(P,(28,28))

def plot_projections(sample):
    PR1 = pca_projection(mode=86, sample = sample)
    PR2 = pca_projection(mode=153, sample = sample)
    fig2, ax2 = plt.subplots(1,3, figsize=(10,4))
    ax2[0].imshow(PR1, cmap = "gray")
    ax2[0].set_title("PCA 86 dimensions")  
    ax2[1].imshow(PR1, cmap = "gray")
    ax2[1].set_title("PCA 153 dimensions")  
    ax2[2].imshow(np.reshape(all_df[sample,:], (28,28)), cmap = "gray")
    ax2[2].set_title("Original Image")  
    
    for i in range(3):
        ax2[i].set_xticks([], [])
        ax2[i].set_yticks([], [])
    
    plt.show()


# In[ ]:


plot_projections(sample = 55)


# In[ ]:


plot_projections(sample = 1001)


# In[ ]:


plot_projections(sample = 9000)


# ### 2.4 Distribution 
# Following figure is scatterplot of each digit along with two principal components (1st and 2nd modes). It looks like decision boundary may be curved. But it is still hard to see.

# In[ ]:


all_pca = np.dot(all_df, eigenvectors[:,-86:])


# In[ ]:


pca_df = pd.DataFrame(np.flip(all_pca[:n_train], axis = 1))
pca_df["label"] = train_y


# In[ ]:


plt.figure(figsize=(7,7))
sns.scatterplot(data = pca_df.iloc[4000:7000,:], x=0, y=1, hue = "label", palette = 'Set1', alpha = 0.8)


# But when they are plotted separately, distribution of each digit looks like concentrated like cluster. It must be interesting to further sub-categorize a digit by k-means cluster analysis.

# In[ ]:


sns.relplot(data = pca_df.iloc[4000:15000,:], x=0, y=1, col = "label", hue = "label", palette = 'Set1', alpha = 0.4, col_wrap = 4)


# ## 3. Support Vector Machine
# ### 3.1 Model selection
# Because distributions of each digits are consentrated and those shapes are round, quadratic kernel would be appropriate for classification by SVM.

# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(all_pca[:n_train], train_y.values, test_size = 0.03)


# In[ ]:


X_train.shape, y_train.shape, X_val.shape, y_val.shape


# In[ ]:


model_svc = SVC(kernel = "poly", C = 1.0, coef0 = 1.0, degree = 2)


# In[ ]:


time1 = time.time()
model_svc.fit(X_train, y_train)
time2 = time.time()
print("Training by SVM (quadratic kernel) took only", int(time2 - time1), "sec" )


# ### 3.2 Training error
# Surprisingly, quadratic kernel perfectly classified train data. That's why training finished very fast. In this case, further parameter tuning is unnecessary if validation accuracy is also high.

# In[ ]:


pred_train = model_svc.predict(X_train)
print("train error = ", np.sum(pred_train != y_train)/y_train.shape[0])


# ### 3.3 Validation error
# Validation error is very small although a bit higher than training error. Therefore, SVM model is chosen.

# In[ ]:


pred_val = model_svc.predict(X_val)
print("validation error = ", np.sum(pred_val != y_val)/y_val.shape[0])


# In[ ]:


confusion_matrix(y_val, pred_val)


# ### Submission

# In[ ]:


pred_test = model_svc.predict(all_pca[n_train:])


# In[ ]:


test_submission = sample_submission
test_submission["Label"] = pred_test


# In[ ]:


test_submission.to_csv('submission.csv',index=False)


# ## 4. Conclusion
# While CNN gets very high accuracy, in my submission it was 0.994, with longer training time, SVM recorded 0.978 after PCA. Considering this short training time (about 20sec), SVM is still good classifier. However, when training data is not clearly separable, it would take much longer time to converge. Furthermore, it must be interesting to analyse some digits by k-means cluster analysis.

# In[ ]:




