#!/usr/bin/env python
# coding: utf-8

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


# # Land Classification - Analysis and Approach to Classifying the Land piece
# 

# # (Update Version- 
# # Containing the Outlier Treatment Approach Using Tukey Method , and the Use of the Concpet of Robust Scaling)

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# #  *Exploratory Data Analysis*

# 1. ***Loading the Training File***

# In[ ]:


data=pd.read_csv('../input/land_train.csv')


# In[ ]:


test_data=pd.read_csv('../input/land_test.csv')


# In[ ]:


print("The Columns in the Training Dataset: {}".format(data.columns.values))


# In[ ]:


print("The Columns in the Testing Dataset: {}".format(test_data.columns.values))


# * ***Printing the first 3 rows from the datafram***

# In[ ]:


print(data.head(3))


# In[ ]:


print("The Types of the Features in the dataframe are {} ".format(data.info()))


# * ***Missing Values are a big problem if not treated earlier , lets check if there are any in this dataset***

# In[ ]:


if(data.isnull().values.any()==True):
    print("Missing Values Found.\n")
else:
    print("No Missing Values found")


# # *Analyzing the Features*
# <br>**Consists of** 
# <br>         *Univariate EDA *
# <br>          *BIvariate EDA* 
# <br>          *Multivariate EDA* 

# # *Univariate EDA*

# *In case of eda of continuous variables, we need to understand the central tendency and spread of the variable.These are measured using methods such as Boxplot,Histogram/Distribution Plot, Violin Plot etc.*

# In[ ]:


#function for plotting the distribution plot , Violin Plot and Box Plot of the Feature Columns
def univariate(df,col,vartype,hue =None):
    sns.set(style="whitegrid")

    fig, ax=plt.subplots(nrows =1,ncols=3,figsize=(10,4))
    ax[0].set_title("Distribution Plot")
    sns.distplot(df[col],ax=ax[0])
    ax[1].set_title("Violin Plot")
    sns.violinplot(data =df, x=col,ax=ax[1], inner="quartile")
    ax[2].set_title("Box Plot")
    sns.boxplot(data =df, x=col,ax=ax[2],orient='v')
plt.show()


# In[ ]:


#X1
univariate(df=data,col='X1',vartype=0)


# In[ ]:


#X2
univariate(df=data,col='X2',vartype=0)


# In[ ]:


#X3
univariate(df=data,col='X3',vartype=0)


# In[ ]:


#X4
univariate(df=data,col='X4',vartype=0)


# In[ ]:


univariate(df=data,col='X5',vartype=0)


# In[ ]:


univariate(df=data,col='X6',vartype=0)


# In[ ]:


univariate(df=data,col='I1',vartype=0)


# In[ ]:


univariate(df=data,col='I3',vartype=0)


# In[ ]:


univariate(df=data,col='I4',vartype=0)


# In[ ]:


#I5
univariate(df=data,col='I5',vartype=0)


# In[ ]:


#I6
univariate(df=data,col='I6',vartype=0)


# # *Bivariate EDA of Features*

# ## *Plots of X1 against X2*:

# In[ ]:


#JOINT PLOT OF X1 and X2
sns.jointplot(x="X1", y="X2", data=data, size=7)


# ### *This shows that a strong positive correlation exists b/w X1 and X2*

# In[ ]:


#using seaborn's FacetGrid to color the scatterplot by Target class
#TARGET CLASS WISE JOINT PLOTS OF X1 and X2
sns.FacetGrid(data, hue="target", size=7)    .map(plt.scatter, "X1","X2")    .add_legend()


# ### *This shows that irrespective what class we have , these two X1 and X2 are strong positively correlated with eah other*

# In[ ]:


#Similarly we can use seborn's pairplot to show bivariate relations b/w features
sns.pairplot(data, hue="target", size=3)


# # *Multivariate EDA of Features*

# *Construct a **Coorelation matrix** of the int64 and float64 feature types*

# In[ ]:


import seaborn as sns
# data['target']=labels
f, ax = plt.subplots(figsize=(10, 8))
corr = data.corr()
sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


# # *Understanding the Relevance of the Feature  in Dataset*

# ## *1. Mutual Classif Info Test:*
# <br> *This estimates mutual information for a  target variable.*

# In[ ]:


from sklearn.feature_selection import RFECV, RFE, mutual_info_classif
from sklearn.svm import SVR


# In[ ]:


# X1_train=np.array(sm_X).delete([1],axis)
# print(data.shape)
p=data.drop(['target'],axis=1)
print(p.shape)
p=np.array(p)
X1=p
Y1=data['target']
mutual_info_classif(X1, Y1)


# *Using mutual information to select which features to include for classification ,  We can clearly see that I1 ,I3,l4 <br>   provide similar mutual information.*

# ## *2-> PCA -Principal Component Analysis*

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA()
X1[: ,:6] = pca.fit_transform(X1[: ,:6])
pca.explained_variance_ratio_


# In[ ]:


pca = PCA(0.95).fit(X1[: ,:6])
print(' Out of X1 to X6 ,only %d components explain 95%% of the variation in data' % pca.n_components_)


# *X5 and X6 have relatively less exlpaination of target variable as<br> compared to X1 which explains atound 79 percent of variation alone.*

# In[ ]:


pca2=PCA()
X1[: ,6:] = pca2.fit_transform(X1[: ,6:])
pca2.explained_variance_ratio_


# In[ ]:


pca2=PCA(0.95).fit(X1[: ,6:])
print(' Out of I1 to I6 ,only %d components explain 95%% of the variation in data' % pca2.n_components_)


# In[ ]:


#selector=RFECV(SVR(kernel="linear"), step=1, cv=5)
#selector.fit(X1_train,Y1_train)


# *We can clearly see from above that the I6 can easily be removed* <br> *from the dataset as it has explains very less variance of the target.*

# ## *Dropping I6 from Dataset*

# In[ ]:


labels=data['target']
data =data.drop(['I6','target'],axis=1)
print(data.head())
print(labels.head())


# In[ ]:


test_data=test_data.drop(['I6'],axis=1)


# # *What to do with the Outliers: Lets Treat them !*

# # *Using Tukey Method :*

# In[ ]:


from collections import Counter
def detect_outliers(df,n,features):
    outlier_indices = []
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        # select observations containing more than 2 outliers
        outlier_indices = Counter(outlier_indices)        
#         print((outlier_indices.items))
        multiple_outliers = list( k for k, v in outlier_indices.items() if v >= n )
#         print(multiple_outliers.values())
        return multiple_outliers 


# ## *Since From the boxplot in the Univariate EDA of features ,  we saw that many points in X1,X2,X3,X4,X5,X6 were outside the whiskers* ,

# ## *So Dropping Rows which have atleast one outlier point*

# In[ ]:



Outliers_to_drop = detect_outliers(data,1,["X1",'X2','X3','X4','X5','X6'])
# print(Outliers_to_drop)
print("No. of Rows with Atleast One Outlier: {}".format(len(data.loc[Outliers_to_drop]))) # Show the outliers rows
# Drop outliers
data= data.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
labels=labels.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


# In[ ]:


print(data.shape)


# In[ ]:


print(labels.shape)


# # *Lets generate a class map : class vs number of samples*

# In[ ]:


#generates a class map of the samples
def get_class_map(labels):
    class_map={}
    for i in labels:
        if str(i) not in class_map:
            class_map[str(i)]=1
        else:
            class_map[str(i)]+=1
#     print(class_map)
    return class_map

p=get_class_map(labels.values)
print(p)
q=[i for i in p.keys()]
# p=get_class_map(labels.values)
sizes = [i for i in p.values()]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
plt.pie(sizes, labels=q, colors=colors,
        autopct='%1.1f%%', shadow=True)
plt.show()


# # *An Example of Class Imbalance Problem*

# *There are two approaches -><br>
# either you over sample it (i.e. you increase the number of samples in minority class)<br>
# . or you under sample it ( decrease the samples in the majority class to make it equal)
# *

# *I don't prefer under sample as it some times removes the information in the data*

# *I will be using SMOTE-Synthetic Minority OverSampling Method<br>
# and ADASYN -Adaptive Synthetic Oversanpling<br>
# for balancing dataset and see if which one is better here*

# In[ ]:


from imblearn.over_sampling import SMOTE,ADASYN
#function for generating additional samples to balance the classes
def class_balancer(dataset,labels):
    sm_X,sm_Y=SMOTE().fit_resample(dataset,labels)
    ad_X,ad_Y=ADASYN().fit_resample(dataset,labels)
    return sm_X,sm_Y,ad_X,ad_Y


# In[ ]:


sm_X,sm_Y,ad_X,ad_Y=class_balancer(data,labels)
print("SMOTE's class Balanced Dataset:{}".format(get_class_map(sm_Y)))
print("ADASYN's class Balanced Dataset:{}".format(get_class_map(ad_Y)))
# print(type(sm_Y))
p=get_class_map(get_class_map(sm_Y))
sizes = [i for i in p.values()]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
plt.title('SMOTEs plot of class map')
plt.pie(sizes, labels=q, colors=colors,
        autopct='%1.1f%%', shadow=True)
plt.show()


# ## *Finally My Dataset looks like :*

# In[ ]:


X1_train=sm_X
Y1_train=sm_Y
print(X1_train.shape)
print(Y1_train.shape)


# In[ ]:


# from sklearn.utils import shuffle
# X1,Y1=shuffle(X1_train,Y1_train,random_state=0)
# print(X1)


# # *Preprocessing the Input:*

# In[ ]:


for i in data.columns:
    print(i+str("\t[")+str(data[i].min())+"\t"+str(data[i].max())+"]")


# *As the ranges of different features are so different , we need to standardize them,otherwise it would be very difficult for the neural network to learn the information*

# *Note : Earlier I used StandardScaler , but Now I have used the Concept of Robust Scaling , Robust Scaling is very effective for highly tail ended distributions and the distributions which might have good number of Outliers*

# In[ ]:


# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# print(X1_train.shape)
# X1_train=scaler.fit_transform(X1_train)
# print(X1_train.shape)


# In[ ]:


from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
print(X1_train.shape)
X1_train=scaler.fit_transform(X1_train)
print(X1_train.shape)


# In[ ]:


X1_test=np.array(test_data)
X1_test=scaler.fit_transform(X1_test)


# In[ ]:


X1_train


# # *Building the Architecture for Training Dataset*

# *Part 1 -> Lets enode the classes into one hot enocoding* 

# In[ ]:


def one_hot_encode(x, n_classes):
    return np.eye(n_classes)[x-1]


Y1_train=Y1_train.reshape((-1,1))
encoded_Y1_train=one_hot_encode(Y1_train,4)
encoded_Y1_train=(encoded_Y1_train.reshape((-1,4)))
print(encoded_Y1_train)


# Part 2-> Setting the Default hyperparameters in our architecture:-
# <br>
# default_test_split,
# epochs,
# number_of_classes,
# batch_size,
# number of hidden layer neurons in first hidden layer
# number of hidden layer neurons in seond hidden layer

# In[ ]:


# various hyperparameter variables default values
default_validation_split=0.25
epochs=15
number_of_classes=4
batch_size=100
hidden_neurons_1=20
hidden_neurons_2=20
input_size=11
print(X1_train.shape)
print(encoded_Y1_train.shape)


# In[ ]:


# Shuffling and Splitting


# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_validate,encoded_Y_train,encoded_Y_validate = train_test_split(X1_train,encoded_Y1_train,test_size=default_validation_split,shuffle=True)
print("Training X_train size : {} ".format(X_train.shape))
print("Training Y_train size : {} ".format(encoded_Y_train.shape))
print("Validating X_test size : {} ".format(X_validate.shape))
print("Validating Y_test size : {} ".format(encoded_Y_validate.shape))
print(encoded_Y_validate)


# Three layered architecture i.e. having two hidden layers .

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.callbacks import EarlyStopping

#Three layered architecture i.e. having two hidden layers .
landClassifier = Sequential()
landClassifier.add(Dense( hidden_neurons_1,  kernel_initializer = 'uniform', activation='sigmoid' , input_dim=11))
# landClassifier.add(Dropout(0.2))
landClassifier.add(Dense(hidden_neurons_2, kernel_initializer = 'uniform', activation='sigmoid'))
landClassifier.add(Dense(4, kernel_initializer = 'uniform',activation='softmax'))

landClassifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'],)
    
landClassifier.summary()


# #  *Fitting the Architecture/Model to our Dataset*

# In[ ]:


# from sklearn.utils import shuffle
# X1_train, encoded_Y1_train = shuffle(X1_train , encoded_Y1_train,random_state=0)
# print(X1_train)


# In[ ]:


# early_stopping = EarlyStopping(monitor='val_loss', patience=2)
# X1_train=np.delete(X1_train, [X1_train.shape[1]-1], axis=1)
output=landClassifier.fit(X_train,encoded_Y_train, 
                   batch_size=batch_size,
                   epochs=epochs,
                   validation_data=(X_validate,encoded_Y_validate),
                   shuffle=False
#                    callbacks=[early_stopping]
                  )


# # *With Default Hyperparameters , We get around 93 percent Validation Accuracy which is not that bad .*

# ## *Analyzing the Training Results:*

# ## *Plot of Loss Accuracy and Accuracy versus Epoch*

# In[ ]:


accs=output.history['acc']
val_accs=output.history['val_acc']
x_axis=[i+1 for i in range(epochs)]
plt.plot(x_axis,accs)
plt.plot(x_axis,val_accs)
plt.show()


# ## *Confusion Matrix*

# In[ ]:


#Creating Confusion Matrix for the Whole Dataset Based on the Trained Classififer
from sklearn.metrics import classification_report
y_trained_by_model=np.argmax(landClassifier.predict(X1_train),axis=1)
y_trained_by_model=y_trained_by_model+1
print(classification_report(Y1_train, y_trained_by_model))


# # *Lets see if we can improve it by Tuning  Our hyperparameter*s
# # *Approach->  Using ( Grid Search )*

# 1.Batch_Size_Grid: Contains the Batch_Size Options<br>
# 2.Validation_Split_Grid: Contains the Validation Split Options

# In[ ]:


historys=[]
batch_size=[]
validation=[]
batch_size_grid=[50,100,150]
validation_grid=[0.1,0.15,0.20]
for i in batch_size_grid:
    for j in validation_grid:
        h=landClassifier.fit(X_train,encoded_Y_train, 
                   batch_size=int(i),
                   epochs=epochs,
                   validation_split=float(j),
                   shuffle=True,
#                    callbacks=[early_stopping]
                  )
        historys.append(h.history['val_acc'])
        batch_size.append(i)
        validation.append(j)


# In[ ]:


history_array=[np.array(i) for i in historys]
valid_acc_history=[np.mean(i,axis=0) for i in history_array]
print(len(valid_acc_history))
print("Validation Accuracies obtained in various grid points: \n{} ".format(valid_acc_history))


# # *Best Validation Accuracy 96% comes out to be at Combination  7 which is with Hyperparameters<br> Validation Split = 0.1 <br> Batch Size =150* 

# # *Hence We increased our Validation Accuracy from  93%  to   96%     which is pretty well*

# # *Predicting using the trained Model*

# In[ ]:


print(X1_test.shape)


# In[ ]:


print(X1_test)


# In[ ]:


y_predicted=np.argmax(landClassifier.predict(X1_test),axis=1)


# In[ ]:


print(y_predicted)


# In[ ]:


y_predicted=y_predicted+1


# In[ ]:


print(y_predicted)


# In[ ]:


get_class_map((y_predicted))


# In[ ]:


test_data['target']=pd.DataFrame(y_predicted)


# In[ ]:


print(test_data)


# In[ ]:


temp=pd.read_csv('../input/land_test.csv')
df=pd.DataFrame(data=test_data)
df['I6']=temp['I6']
l=df['target']
df=df.drop(['target'],axis=1)
df['target']=l
print(df)
df.to_csv('labelled_land_test.csv',index=False)


# In[ ]:




