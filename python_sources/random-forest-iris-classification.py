#!/usr/bin/env python
# coding: utf-8

# # RANDOM FOREST

# In[ ]:





# # LOAD THE DEPENDANCIES

# ## Pandas

# In[ ]:


import pandas as pd
from pandas import set_option
from pandas.plotting import scatter_matrix


# ## Numpy

# In[ ]:


import numpy as np
from numpy import set_printoptions


# ## Matplotlib & Seaborn

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set()


# ## sklearn

# In[ ]:


from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA 
from sklearn.manifold import Isomap


# ## System

# In[ ]:


import os
import sys
import pprint


# ## notebook widgets

# In[ ]:


import ipywidgets as widgets
from IPython.display import Image
from IPython.display import display, Math, Latex
from IPython.core.interactiveshell import InteractiveShell  


# # FUNCTIONS

# ## Label Encoding

# In[ ]:


def label_encoding(dataset,input_headers):
    
    for i in input_headers:
        
        the_data_type=dataset[i].dtype.name
        if (the_data_type=='object'):
            lable_enc=preprocessing.LabelEncoder()
            lable_enc.fit(dataset[i])
            labels=lable_enc.classes_   #this is an array
            labels=list(labels) #converting the labels array to a list
            print(labels)
            dataset[i]=lable_enc.transform(dataset[i])

            return labels
    
        else:
            c=list(np.unique(dataset[i]))
            return [str(x) for x in c]


# ## Feature Scaling

# In[ ]:


def feature_scaling(X_train,X_test):
    
    sc_X=StandardScaler()
    X_train=sc_X.fit_transform(X=X_train,y=None)
    X_test=sc_X.fit_transform(X=X_test,y=None)

    print(sc_X.fit(X_train))
    print(X_train[0:5])
    
    
    
    return X_train, X_test


# ## Visualization

# ### Plot the data space (scatter)

# In[ ]:


def plot_of_data_space(dataset,data,labels,input_headers):
    
    xx_1=pd.DataFrame(data[:,0]) 
    xx_2=pd.DataFrame(data[:,1]) 
    y=pd.DataFrame(labels)
    
    plt.figure(figsize=(15,10)) 
    b=plt.scatter(xx_1[y==0],xx_2[y==0],color='b') 
    r=plt.scatter(xx_1[y==1],xx_2[y==1],color='r')
    g=plt.scatter(xx_1[y==2],xx_2[y==2],color='g') 
    bl=plt.scatter(xx_1[y==3],xx_2[y==3],color='black')
    
    plt.xlabel(input_headers[0])
    plt.ylabel(input_headers[1])

    plt.grid(b=True)
    plt.legend((b,r,g,bl),tuple(np.unique(labels)))
    plt.show()


# ### Feature Distributions (histograms)

# In[ ]:


def feature_distributions(df,target_header,*args):
    """Histrogram plots of the input variables for each target class"""
    
    data=df.drop(target_header,axis=1,inplace=False)

    num_plot_rows=len(data.columns)

    print (classes)
    
    label_encoder = preprocessing.LabelEncoder()
    df[target_header]=label_encoder.fit_transform(df[target_header])
    labels=label_encoder.classes_   #this is an array
    labels=list(labels) #converting the labels array to a list
    print (labels)

    fig = plt.figure(figsize = (20,num_plot_rows*4))
    j = 0

    ax=[]
    colors=['b','r','g','black']
    for i in data.columns:
        plt.subplot(num_plot_rows, 4, j+1)
        j += 1
        for k in range(len(labels)):
    #         print(k)
            a=sns.distplot(data[i][df[target_header]==k], color=colors[k], label = str(labels[k])+classes[k]);
            ax.append(a)
        plt.legend(loc='best')
    
    print('Feature Distribution Plots: \n')
#     fig.suptitle(target_header+ 'Feature Distribution Plots',fontsize=16)
    fig.tight_layout()
    # fig.subplots_adjust(top=0.95)
    plt.show()


# ## Preprocessing: Splitting the dataset

# In[ ]:


def split_the_dataset(dataset,input_headers,target_header):
    
    X=dataset[input_headers]
    y=dataset[target_header]
    
    X.head()
    
    return X,y


# ## Replacing Zeros

# In[ ]:


def replacing_zeros(dataset,the_headers):
    """Function used to replace zeros with the mean"""

    for header in the_headers:
        dataset[header]=dataset[header].replace(0,np.nan)
        mean=int(dataset[header].mean(skipna=True))
        dataset[header]=dataset[header].replace(np.nan,mean)
        
    return dataset


# ## Feature Correlations

# In[ ]:


def correlation_matrix(dataset,input_headers,target_header):
    """Correlation matrix (matrix,heatmap and pairplot)"""
    
    correlation_threshold=.7
    
#     dataset.drop([target_header[0]],axis=1,inplace=True)
    feature_matrix=dataset[input_headers+target_header]
    corr=feature_matrix.corr().abs()
    corr
    
    plt.figure(figsize=(10,10))
    corr_plot=sns.heatmap(corr,cmap="Reds",annot=True)
    
    corr_to_target=corr[target_header[0]].sort_values(ascending=False)
    print(f'Correlations with respect to target:\n{corr_to_target}\n')
    
#     high_corr_drop=[i for i in ctt.index if any (ctt.iloc[i]>.50)]
    high_corr_drop=[]
    for x in range(len(corr_to_target)):
        if (corr_to_target.iloc[x]>correlation_threshold):
            high_corr_drop.append(corr_to_target.index[x])
    print(f'Recommended features to drop due to high correlation (greater than {correlation_threshold}) to target variable:\n{high_corr_drop}')
    
    
#     corr_pair=sns.pairplot(dataset,hue=target_header[0])
    plt.show()
    
#     return corr_plot  #corr_pair 
    


# ## Principal Component Analysis (PCA)

# In[ ]:


def pca(dataset,input_headers,target_header,*args):
    """Dimensionality reduction via PCA. This function is called when the there are more than 2 predictor variables in the dataset."""
    
    feature_matrix=dataset[input_headers]
    model = PCA(n_components=2)            # 2. Instantiate the model with hyperparameters
    model.fit(feature_matrix)  # 3. Fit to data. Notice y is not specified!
    X_2D = model.transform(feature_matrix)         # 4. Transform the data to two dimensions

    dataset['PCA1'] = X_2D[:, 0]
    dataset['PCA2'] = X_2D[:, 1]
    
#     f,ax=plt.subplots(figsize=(8, 8))
#     cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
#     sns.scatterplot("PCA1", "PCA2", hue=target_header[0],style=target_header[0],
#                     data=dataset,palette="Set1",label = None,legend=False,ax=ax).set_title("PCA Variables per Target Class");
    ax=sns.lmplot("PCA1", "PCA2", hue=target_header[0], data=dataset, fit_reg=False);
    plt.title("PCA Variables per Target Class")
    plt.legend(title=target_header[0], loc='lower right', labels=classes)


# # MAIN PROGRAM

# ## Get Data

# In[ ]:


if __name__ == "__main__":
    
    location='../input/Iris.csv'

    dataset=pd.read_csv(location)
    dataset.info()


# In[ ]:


dataset.head()


# In[ ]:


dataset.describe()


# ## Selecting inputs and targets

# In[ ]:


target_header=['Species']
input_headers=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

target_label=label_encoding(dataset,target_header)
classes=target_label
test_label=label_encoding(dataset,input_headers)

dataset=dataset[input_headers+target_header]
X,y=split_the_dataset(dataset,input_headers,target_header)

X.head()


# ## Data Visualizations

# ### Data space

# In[ ]:


if (X.values.shape[1]==2):
    plot_of_data_space(dataset,X.values,y.values,input_headers)
else:
    pca(dataset,input_headers,target_header,classes)


# ### Feature distributions

# In[ ]:


feature_distributions(dataset,target_header[0],classes)


# In[ ]:


X.head()


# ## Correlation Matrix

# In[ ]:


correlation_matrix(dataset,input_headers,target_header)


# ## Splitting the Train-Test data

# In[ ]:


test_data_size=.2
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=test_data_size,random_state=42)


# In[ ]:


print(f'Xtrain sample count: {Xtrain.shape[0]}')
print(f'ytrain sample count: {ytrain.shape[0]}')
print(f'Xtest sample count: {Xtest.shape[0]}')
print(f'ytest sample count: {ytest.shape[0]}')


# ## Scale the data

# In[ ]:


#Scale the data    
Xtrain, Xtest=feature_scaling(Xtrain,Xtest)


# ## Random Forest Model

# In[ ]:


model=RandomForestClassifier(n_estimators=100, criterion='gini',max_depth=None,
                             random_state=42)


# ### Fit model to training data

# In[ ]:


ytrain=ytrain.values.reshape(ytrain.size,)
model.fit(Xtrain,ytrain)


# ### Model prediction on test data

# In[ ]:


y_model=model.predict(Xtest)
y_model


# In[ ]:


y_model_prob=model.predict_proba(Xtest)
y_model_prob[0:5]


# ### Model score & performance

# In[ ]:


accur=accuracy_score(ytest,y_model)
recall=recall_score(ytest, y_model,average=None)
precision=precision_score(ytest, y_model,average=None)
print (f'MODEL RESULTS WITH DATASET SPLIT AT {(1-test_data_size)*100:.1f}% TRAINING DATA AND {test_data_size*100:.1f}% TEST DATA\n')
print(f'Model Accuracy:{accur*100:.1f}%\n')

for i,k in enumerate (classes):
    print(f'{k} Recall:{recall[i]*100:.1f}%\n')
    print(f'{k} Precision:{precision[i]*100:.1f}%\n')


# ### Confusion Matrix

# In[ ]:


cm=confusion_matrix(ytest, y_model)
cm=pd.DataFrame(data=cm,columns=classes,index=classes)
sns.heatmap(cm,square=True,annot=True,cbar=True)
plt.title('CONFUSION MATRIX')
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()


# ### Cross Validation

# In[ ]:


y=y.values.reshape(y.size,)   # reshape y to a 1-d array
k_fold=10
score=cross_val_score(model,X,y,cv=k_fold)
score.mean();

print(f'Cross Validation Scores:\n{score}\n')
print(f'Mean Score:{score.mean()*100:.2f}%\nStandard Deviation:{score.std():.2f}')

sns.boxplot(x=score,orient='v')
plt.title(f'{k_fold} Fold Cross Validation Results')
plt.ylabel('Model Accuracy')
plt.show()

