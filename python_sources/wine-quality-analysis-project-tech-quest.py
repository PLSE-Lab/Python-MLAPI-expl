#!/usr/bin/env python
# coding: utf-8

# 
# ![wine.png](attachment:wine.png)

# # **Project Problem Description:**
# 
# ### A wine bottling company has a lab where they test wine quality.
# ### The company wants to be able to predict the marketability of any new wine sample they get from a vineyard before they commit to marketing it.
# ### Create a machine learning model to predict the quality score of a given wine sample
#  

# 
# #  Input Features
# *    fixed acidity
# *    volatile acidity
# *    citric acid
# *     residual sugar
# *     chlorides
# *    free sulfur dioxide
# *    total sulfur dioxide
# *  density
# *    pH
# *     sulphates
# *     alcohol
#    
# Output variable:
#    Quality (score between 0 and 10)
# 

#   

# # **THE PROCESSES INVOLVED IN THIS PROJECT ARE LISTED BELOW**
# 
# <br> 
# 1. [IMPORTING THE DATA AND IMPORT MODULES USED IN THIS PROJECT](#1-bullet) <br><br>
# 
# 2. [CHECKING FOR MISSING VALUES AND PROCESSING/EXPLORING DATA](#2-bullet)<br><br>
# 
# 3. [GRAPHICAL ANALYSIS ](#3-bullet) <br><br>
# 
# 4. [ CORRELATION ANALYSIS](#4-bullet) <br><br> 
#    
# 5. [DATA FITING AND TRANSFORMATION](#5-bullet) <br><br>
#     
# 6. [DATA MODELLING: PREDICTING QUALITY](#6-bullet) <br><br>
#     
# <br>

#   

# # 1. IMPORTING THE DATA AND IMPORT MODULES USED IN THIS PROJECT <a class="anchor" id="1-bullet"></a>

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv(r'../input/winedataset/02_WineDataset.csv')
pd.options.display.max_columns = None #Display all columns
pd.options.display.max_rows = None #Display all rows
data.head()


#    

# # 2. CHECKING FOR MISSING VALUES AND PROCESSING/EXPLORING DATA 
# <a class="anchor" id="2-bullet"></a>

# In[ ]:


data.describe()


# In[ ]:


data.isnull().any().any() #Check is there is any NULL value in the data set


# In[ ]:


data.rename(columns={'fixed acidity': 'fixed_acidity',
                     'citric acid':'citric_acid',
                     'volatile acidity':'volatile_acidity',
                     'residual sugar':'residual_sugar',
                     'free sulfur dioxide':'free_sulfur_dioxide',
                     'total sulfur dioxide':'total_sulfur_dioxide'},
            inplace=True)


# In[ ]:


data.head()


# In[ ]:


data['quality'].unique()


# In[ ]:


data.quality.value_counts().sort_index()


#    

# # 3. GRAPHICAL ANALYSIS 
# <a class="anchor" id="3-bullet"></a>

# In[ ]:


plt.figure(figsize=(10,15))

for pl,col in enumerate(list(data.columns.values)):
    plt.subplot(4,3,pl+1)
    sns.set()
    sns.boxplot(col,data=data)
    plt.tight_layout()


# # From the various box plots above we see that most of the data is right skewed. Focusing on Fixed Acidity we see that 50% of the data is roughly between 6.5 and 7.5. 

#    

# In[ ]:


sns.catplot(x='quality', data=data, kind='count');
plt.title('Distribution of the Quality');


# # We see in the graph above that the quality of wine that appears most in the data set is 6

#    

# # **4. CORRELATION ANALYSIS:**
# 
# *     PAIR PLOT
# *     BOX PLOTS,
# *     VIOLIN PLOTS
# *     SWARM LOTS 
# *     LM PLOTS
# *     SCATTER DIAGRAMS 
# *     DATA DISTRIBUTION PLOTS
# *    DATA ANALYSIS
# <a class="anchor" id="4-bullet"></a>

#   

# In[ ]:


data.corr()['quality'].sort_values()


#   

# In[ ]:


data_Cor = data.drop(['fixed_acidity', 'volatile_acidity', 'density', 'residual_sugar', 'chlorides','total_sulfur_dioxide'], axis=1)


# In[ ]:


sns.pairplot(data_Cor,hue = 'quality');


# # Focusing on the features in the data set that give a postive correlation we see both distribution of single variables on the diagonal with respect to the quality (denoted by the various colours) and relationships between two variables with respect to the quality(denoted by the various colours)

#    

# In[ ]:


plt.figure(figsize=(14,8))
ax = sns.heatmap(data_Cor.corr(), annot = True, cmap='RdPu')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.1, top - 0.1);


#    

# # From the correlation analysis done above we see that the attributes in the dataset that have a positive correlation with the quality are :
# 
# *  pH  ---                    0.0195
# *  sulphates  ---                0.038485
# *  free_sulfur_dioxide  ---      0.055463
# *  citric_acid     ---           0.085532
# *  alcohol      ---              0.444319
# 

# # Below we will analyse each attribute 

#   

# In[ ]:


data['pH'].describe()


# In[ ]:


data['sulphates'].describe()


# In[ ]:


data['free_sulfur_dioxide'].describe()


# In[ ]:


data['alcohol'].describe()


#   

# In[ ]:


data.iloc[:,:11].head() #Removing the quality column


# In[ ]:


plt.figure(figsize=(10,15))

for pl,col in enumerate(list(data.iloc[:,:11].columns.values)):
    plt.subplot(4,3,pl+1)
    sns.violinplot(y= data[col],x='quality',data=data, scale='count')
    plt.title(f'quality/{col}')
    plt.tight_layout()
    
    


# In[ ]:


#This plots a 2d scatter plot with a regression line. Easily showing the correlation, distribution, and outliers!

for col in (data.iloc[:,:11].columns.values):
 
    sns.lmplot(x='quality',y=col,data=data, fit_reg=False)
  
    plt.title(f'quality/{col}');
    plt.ylabel(col);
    plt.show();
    plt.tight_layout();
    plt.close() 
    
    sns.lmplot(x='quality',y=col,data=data)
  
    plt.title(f'quality/{col}');
    plt.ylabel(col);
    plt.show();
    plt.tight_layout();
    plt.close() 
    
    print('   ')
 


# # From both the violin plot and the lm plot above we can easily see the correlation between the different attributes and the quality

#  

# In[ ]:


condition = [(data['quality']>6),(data['quality']<=4)]#Setting the condition for good and bad ratings

rating = ['good','bad']


# In[ ]:


data['rating'] = np.select(condition,rating,default='average')
data.rating.value_counts()


# In[ ]:


data.head(25)


# In[ ]:


#This cell takes roughly about 15mins to an hour+ to run depending on the specifications of your workstation

for col in data.iloc[:,:11].columns.values:
 
    
    sns.set()
    sns.violinplot(y= col ,x='rating',data=data, scale='count')
    plt.title(f'rating/{col}');
    plt.ylabel(col);
    plt.show();
    plt.tight_layout();
    plt.close() 
    
    sns.set()
    sns.swarmplot(x='rating',y=col,data=data)
    plt.title(f'rating/{col}');
    plt.ylabel(col);
    plt.show();
    plt.tight_layout();
    plt.close() 
    
    print('   ')
    


# # The swarm and violin plot shows how much of the data is distributed into the 'good', 'average', and 'bad' rating

#   

# In[ ]:


data[[('rating'),('quality')]].head(25)


# In[ ]:


data.groupby('rating')['quality'].value_counts()


# # Based on the ratio of good,average, and bad wine samples in the data set and the total number of samples being 6,495; the percentage of these ratings are as follows:
# 
# * good --- 20%
# * average --- 76%
# * bad --- 4%

#    

# # 5. DATA FITTING AND TRANSFORMATION
# <a class="anchor" id="5-bullet"></a>

# In[ ]:


#This changes the quality from numbers to ratings between good and bad

bins = (2, 4, 9)
group_names = ['bad', 'good']
data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)


# In[ ]:


data.head(25)


# In[ ]:


data[[('rating'),('quality')]].head(25)


# In[ ]:


#This basically maps all good values to 1 and all bad values to 0 in the quality column

dfL = np.array(data['quality'])

dfL = pd.DataFrame(dfL)

data['quality'] = dfL.apply(lambda x: x.map({'good':1,'bad':0})) 



# In[ ]:


data.head(30)


#   

# In[ ]:


data[[('rating'),('quality')]].head(25)


# In[ ]:


#Setting the values of X and Y

X =  data[['alcohol','density','sulphates','pH','free_sulfur_dioxide','citric_acid']]
y =  data['quality']


#   

# *  X_tr is training data for x
# *  X_t is testing data for x
# *  y_tr is training data for y
# *  y_t is testing data for y

# In[ ]:


X_tr,X_t,y_tr,y_t = train_test_split(X,y)


# In[ ]:


X_tr.shape, X_t.shape


# In[ ]:


y_tr.shape, y_t.shape


# In[ ]:


stds= StandardScaler()

X_tr= stds.fit_transform(X_tr)
X_t = stds.fit_transform(X_t)


# # 6. **DATA MODELLING: PREDICTING QUALITY:**
# *  LOGISTIC REGRESSION
# *  RANDOM FOREST
# *  GAUSSIAN NORMAL DISTRIBUTION
# *  SUPPORT VECTOR CLASSIFIER
# *  DESCISION TREE
# *  STOCHASTIC GRADIENT DESCENT
# 
# <a class="anchor" id="6-bullet"></a>

# In[ ]:


#The functions below will be used to measure the accuracy of the model

def generateClassificationReport_Tr(y_true,y_pred):
    '''Train data accuracy tester'''
    print(classification_report(y_true,y_pred));
    print(confusion_matrix(y_true,y_pred));
    print('\n\nTrain Accuracy is: ',
          round(100*accuracy_score(y_true,y_pred),3),'%\n');
    
def generateClassificationReport_T(y_true,y_pred):
    '''Test data accuracy tester'''
    print(classification_report(y_true,y_pred));
    print(confusion_matrix(y_true,y_pred));
    print('\n\nTest Accuracy is: ',
          round(100*accuracy_score(y_true,y_pred),3),'%\n');


#   

# # Both the train and test data will be passed to the models. This is done to check how accurate the models are. 

#   

# # **LOGISTIC REGRESSION**

# In[ ]:


#LOGISTIC REGRESSION

logr = LogisticRegression(max_iter=1000);
logr.fit(X_tr,y_tr);


# In[ ]:


#TRAIN DATA

ytr_pred = logr.predict(X_tr)
generateClassificationReport_Tr(y_tr,ytr_pred)


# In[ ]:


#TEST DATA

yt_pred = logr.predict(X_t)
generateClassificationReport_T(y_t,yt_pred)


#        

# # **RANDOM FOREST**

# In[ ]:


#RANDOM FOREST

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_tr,y_tr);


# In[ ]:


#TRAIN DATA

ytr_pred = rfc.predict(X_tr)
generateClassificationReport_Tr(y_tr,ytr_pred)


# In[ ]:


#TEST DATA

yt_pred = rfc.predict(X_t);
generateClassificationReport_T(y_t,yt_pred);


#   

#  # **GAUSSIAN NORMAL DISTRIBUTION**

# In[ ]:


#GAUSSIAN NORMAL DISTRIBUTION 

gnd = GaussianNB()
gnd.fit(X_tr,y_tr);


# In[ ]:


#TRAIN DATA

ytr_pred = gnd.predict(X_tr)
generateClassificationReport_Tr(y_tr,ytr_pred)


# In[ ]:


#TEST DATA

yt_pred = gnd.predict(X_t)
generateClassificationReport_T(y_t,yt_pred)


#    

# # **SUPPORT VECTOR CLASSIFIER**

# In[ ]:


#SUPPORT VECTOR CLASSIFIER

svc = SVC()
svc.fit(X_tr,y_tr);


# In[ ]:


#TRAIN DATA

ytr_pred = svc.predict(X_tr)
generateClassificationReport_Tr(y_tr,ytr_pred)


# In[ ]:


#TEST DATA

yt_pred = svc.predict(X_t)
generateClassificationReport_T(y_t,yt_pred)


#    

# # **DESCISION TREE**

# In[ ]:


#DESCISION TREE

dtc = DecisionTreeClassifier()
dtc.fit(X_tr,y_tr);


# In[ ]:


#TRAIN DATA

ytr_pred = dtc.predict(X_tr)
generateClassificationReport_Tr(y_tr,ytr_pred)


# In[ ]:


#TEST DATA

yt_pred = dtc.predict(X_t)
generateClassificationReport_T(y_t,yt_pred)


#   

# # **STOCHASTIC GRADIENT DESCENT**

# In[ ]:


#STOCHASTIC GRADIENT DESCENT

sgd = SGDClassifier()
sgd.fit(X_tr, y_tr);


# In[ ]:


#TRAIN DATA

ytr_pred = sgd.predict(X_tr)
generateClassificationReport_Tr(y_tr,ytr_pred)


# In[ ]:


#TEST DATA

yt_pred = sgd.predict(X_t)
generateClassificationReport_T(y_t,yt_pred)


#   

# # **CONCLUSION**
# 
# 
# ## We see that alot of the models we used gave an accuracy really close to 100% but the Random Forest Classifier yielded the best results with an acuracy of 97.231%

# In[ ]:




