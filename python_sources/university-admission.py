#!/usr/bin/env python
# coding: utf-8

# This data set has the information on the GRE,TOEFL,CGPA and other details of students seeking Post graduation admission at Universities.We will try to exprole the data and see what we can understand from it.The deeper question would be are college degrees revalent in the era of Nano degrees?This is a work in process and I will be updating the kernel in coming days.If you like my work please do vote.

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


# **Importing the modules needed for the analysis **

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os
import warnings
warnings.filterwarnings('ignore') 
#plt.style.use('seaborn')
plt.style.use('fivethirtyeight')


# **Importing the data into kernel**

# In[ ]:


admt=pd.read_csv('../input/Admission_Predict.csv')


# In[ ]:


admt.head()


# In[ ]:


admt.isnull().sum()


# We can see that there are no missing values in the data set.

# **Summary Of Dataset**

# In[ ]:


print('Rows     :',admt.shape[0])
print('Columns  :',admt.shape[1])
print('\nFeatures :\n     :',admt.columns.tolist())
print('\nMissing values    :',admt.isnull().values.sum())
print('\nUnique values :  \n',admt.nunique())


# **Renaming the columns to make our lives easy**

# In[ ]:


admt.columns.to_frame().T


# **Method to display the columns in the data set **

# In[ ]:


admt.count().to_frame().T


# Above method can be used to find out the rows of values in the data set.

# In[ ]:


print("There are",len(admt.columns),"columns:")
for x in admt.columns:
    sys.stdout.write(str(x)+", ")                                                      #admt.columns also works 


# Looking at the column names we can see that we can make the names of the colums shorter.

# In[ ]:


admt.rename(columns={'Serial No.':'Srno','GRE Score':'GRE','TOEFL Score':'TOEFL','University Rating':'UnivRating','Chance of Admit ':'Chance'},inplace=True)


# In[ ]:


admt.head()


# We can see that the name of the columns have be changed as per our convience.We can see that first  column is serial number it will not have any effect on the chance of admission to the University.We better drop the column of serial number from the data set.

# In[ ]:


admt.columns


# In[ ]:


admt.drop('Srno', axis=1, inplace=True)
admt.head()


# We can see that the column for serial number is droped or removed from the dataset.

# **Lets explore the data **

# In[ ]:


admt.describe().plot(kind = "area",fontsize=27, figsize = (20,8), table = True,colormap="rainbow")
plt.xlabel('Statistics',)
plt.ylabel('Value')
plt.title("General Statistics of Admissions")


# We can display describe dunction in pictorial way.In most cases the describle table is sufficient for us the get valuable information about the data.

# In[ ]:


plt.figure(1, figsize=(10,6))
plt.subplot(1,4, 1)
plt.boxplot(admt['GRE'])
plt.title('GRE Score')

plt.subplot(1,4,2)
plt.boxplot(admt['TOEFL'])
plt.title('TOEFL Score')

plt.subplot(1,4,3)
plt.boxplot(admt['UnivRating'])
plt.title('University Rating')

plt.subplot(1,4,4)
plt.boxplot(admt['CGPA'])
plt.title('CGPA')

plt.show()


# Above box plot shows us the min,median and max values for GRE,TOEFL,University rating and CGPA for the dataset.

# ** Finding out correlations between the features and the chance of admission to the university**

# In[ ]:


fig=plt.gcf()
fig.set_size_inches(10,10)
fig=sns.heatmap(admt.corr(),annot=True,cmap='inferno',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)


# Heat map gives a good pictorial representation of the correlation of features to our target value chance of admit to university.Looking at the heat map to get the correlation can sometimes be condusing.One way out would be the to get the correlation values against target(Chance) as shown below.

# In[ ]:


#correlations_data = admt.corr()['Chance'].sort_values(ascending=False)
cor=admt.corr()['Chance']
# Print the correlations
print(cor)


# We can see that CGPA,GRE,TOEFL,University Ranking has the highest correlation with the chance of admission to the university.The other parameters like SOP,LOR and Research has less impact on the chance of admission.We can dropt he column Srno from our dataframe as it doesnt have any impact on the chance of admission.

# **Plotting the distribution of the data**

# In[ ]:


admt[['GRE','TOEFL','UnivRating','CGPA']].hist(figsize=(10,8),bins=10,color='#ffd700',linewidth='1',edgecolor='k')
plt.tight_layout()
plt.show()


# In[ ]:


category = ['GRE','TOEFL','UnivRating','SOP','LOR ','CGPA','Research','Chance']
color = ['yellowgreen','gold','lightskyblue','pink','red','purple','orange','gray']
start = True
for i in np.arange(4):
    
    if start == True:
        fig = plt.figure(figsize=(14,8))
        start = False
        
    plt.subplot2grid((4,2),(i,0))
    admt[category[2*i]].hist(color=color[2*i],bins=10)
    plt.title(category[2*i])
    plt.subplot2grid((4,2),(i,1))
    admt[category[2*i+1]].hist(color=color[2*i+1],bins=10)
    plt.title(category[2*i+1])
    
plt.subplots_adjust(hspace = 0.7, wspace = 0.2)    
plt.show()


# In[ ]:


print('Mean CGPA Score is :',int(admt[admt['CGPA']<=500].CGPA.mean()))
print('Mean GRE Score is :',int(admt[admt['GRE']<=500].GRE.mean()))
print('Mean TOEFL Score is :',int(admt[admt['TOEFL']<=500].TOEFL.mean()))
print('Mean University rating is :',int(admt[admt['UnivRating']<=500].UnivRating.mean()))


# Target of an aspirant would be get more than the mean scores displayed above.

# **How important is Research to get an Admission?**

# In[ ]:


a=len(admt[admt.Research==1])
b=len(admt[admt.Research==0])
print('Total number of students',a+b)
print('Students having Research:',len(admt[admt.Research==1]))
print('Students not having Research:',len(admt[admt.Research==0]))


# In[ ]:


y=np.array([len(admt[admt.Research==1]),len(admt[admt.Research==0])])
x=['Having Research','Not having Research']
ax=plt.bar(x,y,width=0.5,color='red',edgecolor='k',align='center',linewidth=2)
#plt.xlabel('',fontsize=20)
plt.ylabel('Student Count',fontsize=20)
#ax.tick_params(labelsize=20)
plt.title('Student Research',fontsize=25)
plt.grid()
plt.ioff()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
admt['Research'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Students Research')
ax[0].set_ylabel('Student Count')
sns.countplot('Research',data=admt,ax=ax[1])
ax[1].set_title('Students Research')
plt.show()


# We can see that 55% Students have done Research.It possible only the better student could get a chance for doing research.Doing research does add practical knowledge and increases the student skill of working with groups or teams.

# In[ ]:


sns.scatterplot(data=admt,x='GRE',y='TOEFL',hue='Research')


# We can see that stutents who have done research do have good TOEFL and GRE Score.

# **Chance of admission**

# We are assuming here that students with 0.7 chance of admission have secured admission.We create another column in oour dataset named Admit.The value of Admit=1 if Chance>0.7 and Admit=0 if Chance<0.7.

# In[ ]:


def modiffy(row):
    if row['Chance'] >0.7 :
        return 1
    else :
        return 0
admt['Admit'] = admt.apply(modiffy,axis=1)
admttemp = admt.drop(['Chance'], axis=1)
#sns.pairplot(admttemp,hue='Admit')
sns.scatterplot(data=admttemp,x='GRE',y='TOEFL',hue='Admit')
del admttemp


# We can clearly see that students with higher GRE and TOEFL scores have very high chance of getting an university admission.

# In[ ]:


sns.factorplot('Research','Admit',data=admt)
plt.show()


# Yes your chance of Admission increases if you do Research.

# **What should be your Scores for 0.9 % Chance of Admission?**

# In[ ]:


admt_sort=admt.sort_values(by=admt.columns[-1],ascending=False)
admt_sort.head()
#admt.head()
#admttemp.head()


# We can see that the maximum Chance of admission is 0.97.Lets find out the scores needed for 90 % chance of admission.

# In[ ]:


admt_sort[(admt_sort['Chance']>0.90)].mean().reset_index()


# For having a 90% Chance to get admission one should have GRE=333.61,TOEFL=116.28,CGPA=9.53 .If you get scores more than this then your chances of admission are very good.

# **Violin plots to reinforce our earlier learning**

# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot("Research","GRE",hue="Admit", data=admt,split=True)
plt.subplot(2,2,2)
sns.violinplot("Research","TOEFL",hue="Admit", data=admt,split=True)
plt.subplot(2,2,3)
sns.violinplot("Research","CGPA",hue="Admit", data=admt,split=True)
plt.subplot(2,2,4)
sns.violinplot("Research","UnivRating",hue="Admit", data=admt,split=True)
#ax[0].set_title('Pclass and Age vs Survived')
#ax[0].set_yticks(range(0,110,10))
#sns.violinplot("Sex","Age", hue="Survived", data=data,split=True,ax=ax[1])
#ax[1].set_title('Sex and Age vs Survived')
#ax[1].set_yticks(range(0,110,10))
plt.ioff()
plt.show()


# We can clearly see that the student with research have higher chance of admission and their overall all GRE,TOEFL and CPGA scores are also high.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
admt['Admit'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Admitted to University')
ax[0].set_ylabel('')
sns.countplot('Admit',data=admt,ax=ax[1])
ax[1].set_title('Admitted to University')
plt.show()


# We can see that 59% of the student have high chance of Admission.

# **Lets start with machine learning **

# In[ ]:


from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix


# **Lets Predict the Chance of Admission **

# In[ ]:


admt.head()


# In[ ]:


X=admt.iloc[:,:-2].values
X[0]


# In[ ]:


y=admt.iloc[:,-2].values # or we can use y=data.iloc[:,3].values
y[0]


# Splitting the data into training and test data using test size of 0.05

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.05,random_state=0)


# **1.Linear regression **

# In[ ]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error
reg=linear_model.LinearRegression()
reg.fit(X_train,y_train)
pred1=reg.predict(X_test)
print("Mean Squared Error: ",mean_squared_error(y_test,pred1))
#('Accuracy for Linear Regression is ',metrics.accuracy_score(y_pred,y_test))


# The test values from the dataset are :

# In[ ]:


y_test


# The Values predicted by Linear regression are :

# In[ ]:


pred1


# Predicting the chance for a use case.We give the input to the algorithm in the form of a list as shown below.

# In[ ]:


Score=['337','118','4','4.5','4.5','9.65','1']
Score=pd.DataFrame(Score).T
chance=reg.predict(Score)
chance


# So the algrothim predicts the value as 0.95 against the actual value 0.92

# ### Pictorial representation of correlation between the actual and predicted values

# In[ ]:


plt.figure(figsize=(12,8))
y=pred1
y1=y_test
x=np.arange(1, 21, 1)
x1=np.arange(0,21,2)
plt.plot(x,y,color='r',marker='o',label='Predicted')
plt.plot(x,y1,color='g',label='Actual')
plt.xticks(x1)
plt.gca().legend(('Predicted','Test'))
plt.xlabel('Cases',fontsize=20)
plt.ylabel('Chance of Admission',fontsize=20)
plt.title('Chance Predicted Vs Actual Values',fontsize=25)
plt.grid()
plt.ioff()


# We can see from the above plot that we have fairly good correlation.

# **2.Decision Tree **

# In[ ]:


admt.head()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators = 1000,random_state = 123)
columns = ['Admit']
admt.drop(columns, inplace=True, axis=1)
X = admt.drop('Chance',axis = 1)
y = admt['Chance']
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = .25,random_state = 123)
rf_model = RandomForestRegressor(n_estimators = 1000,random_state = 123)
rf_model.fit(X_train,y_train)
feature_importance = pd.DataFrame(sorted(zip(rf_model.feature_importances_, X.columns)), columns=['Value','Feature'])
plt.figure(figsize=(10, 6))
sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False))
plt.xlabel('Value',fontsize=20)
plt.ylabel('Feature',fontsize=20)
plt.title('Random Forest Feature Importance',fontsize=25)
plt.grid()
plt.ioff()
plt.tight_layout()


# We can see CGPA,GRE,TOEFL and SOP arte most important features in the data set.

# **3.Logistic Regression: **
# It is used to predict binary results.In this case we have crerated the column Admit which tells us detail of Whether the candidate has got admission(1) or not (0).We have seen from the decision tree algorithm that CGPA and the GRE Score has the highest influence on the chance of admission.So while making a Logistic Regression we will use the values of CGPA and GRE score to predict the Admission to the University 

# **3.1 Generating Array of Features and Target Values**

# In[ ]:


admt.head()


# In[ ]:


X=admt_sort.iloc[:,[0,5]].values    # O represents GRE Score and 5 represnts CGPA 
y=admt_sort.iloc[:,8].values        # 8 tells us if the Candidate got Admission or not 


# In[ ]:


from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
Y = lab_enc.fit_transform(y)


# **3.2 Splitting the dataset to Train and Test Set**

# In[ ]:


from sklearn.model_selection import train_test_split   #cross_validation doesnt work any more
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0) 
#y_train


# **3.3 Feature Scaling**

# In[ ]:


from sklearn.preprocessing import StandardScaler 
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
#X_train


# **3.4 Fitting Logistic Regression into Training set**

# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# **3.5 Predicting the test set results**

# In[ ]:


y_pred=classifier.predict(X_test)


# **3.6 Making the confusion matrix**

# In[ ]:


from sklearn.metrics import confusion_matrix  #Class has capital at the begining function starts with small letters 
cm=confusion_matrix(y_test,y_pred)
import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Test for Test Dataset")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.show()


# Correct predictions =28+39=64 
# 
# Wrong predictions =6+7=13
# 
# Accuracy =(64/77)*100 =83.11 %

# **3.7 Visualizing the Training Set Results**

# In[ ]:


from matplotlib.colors import ListedColormap
X_set,y_set=X_train,y_train
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                 np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
            alpha=0.75,cmap=ListedColormap(('yellow','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
               c=ListedColormap(('red','green'))(i),label=j)
plt.title('Predicting University Admission')
plt.xlabel('GRE Score')
plt.ylabel('CGPA')
plt.legend()
plt.show()


# **3.8 Visualizing the Test Set Results**

# In[ ]:


from matplotlib.colors import ListedColormap
X_set,y_set=X_test,y_test
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                 np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
            alpha=0.75,cmap=ListedColormap(('yellow','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
               c=ListedColormap(('red','green'))(i),label=j)
plt.title('Predicting University Admission')
plt.xlabel('GRE Score')
plt.ylabel('CGPA')
plt.legend()
plt.show()


# The Yellow region is the area of people who failed to get admission.Red dots represent the students who failed to get admission.
# Green Dotd and Green Area represent the people who Managed to get admission.
# 
# 0-Not Admitted 
# 
# 1-Admitted 
# 

# 4. K Means Classification 

# **4.1 Fitting K Nearest  Neighbor to Training set**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier_4=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier_4.fit(X_train,y_train)


# **4.2 Predicting the test set results**

# In[ ]:


y_pred_4=classifier_4.predict(X_test)


# **4.3 Making the confusion matrix**

# In[ ]:


from sklearn.metrics import confusion_matrix  #Class has capital at the begining function starts with small letters 
cm=confusion_matrix(y_test,y_pred_4)
import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Test for Test Dataset")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.show()


# Correct predictions =24+41=65
# 
# Wrong predictions =6+9=15
# 
# Accuracy = (65/180)*100 =81. 25 %

# **4.4Visualizing the Training Set Results**

# In[ ]:


from matplotlib.colors import ListedColormap
X_set,y_set=X_train,y_train
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                 np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier_4.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
            alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
               c=ListedColormap(('red','green'))(i),label=j)
plt.title('K-NN (Training set)')
plt.xlabel('GRE Score')
plt.ylabel('CGPA')
plt.legend()
plt.show()


# **4.5.Visualizing the Test Set Results**

# In[ ]:


from matplotlib.colors import ListedColormap
X_set,y_set=X_test,y_test
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                 np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier_4.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
            alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
               c=ListedColormap(('red','green'))(i),label=j)
plt.title('K-NN (Test set)')
plt.xlabel('GRE Score')
plt.ylabel('CGPA')
plt.legend()
plt.show()


# **5. K Means Clustering: **
# K means is an unsupervised clustering algorithm.We use it here to see how the students will be getting clustered based on their GRE and CGPA Scores.

# **5.1 Generating the Array of Features **

# In[ ]:


X=admt_sort.iloc[:,[0,5]].values 
#X


# **5.2 Using Elbow method to find the optiminal cluster number**

# In[ ]:


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# From the Elbow plot we can see that four could be the optiminal number of cluster for this analysis.

# **5.3. Applying K means to the Dataset**

# In[ ]:


kmeans=KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)


# **5.4 Visualizing the clusters**

# In[ ]:


plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Must Improve') 
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='greenyellow',label='Excellent')  
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='yellow',label='Good')   
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='green',label='Outstanding')  #cyan
#plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='burlywood',label='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='magenta',label='Centroids')
plt.title('Cluster of Students')
plt.xlabel('GRE Score')
plt.ylabel('CGPA')
plt.legend()
plt.show


# Based on the cluster I have catogerised the students into four catogeries.
# 
# 1.Outstanding - GRE> 327 + and CGPA > 8.5
# 
# 2.Ecxcellent -GRE> 317 + and CGPA > 7.7
# 
# 3.Good -GRE> 306 + and CGPA > 7.3
# 
# 4.Must Improve -GRE> 290 + and CGPA > 6.7

# **6. Artificial Neural Network (ANN)**

# **6.1 Generating the Array of Features and Target Values **

# In[ ]:


admt_sort.head()


# In[ ]:


X=admt_sort.iloc[:,0:8].values

y=admt_sort.iloc[:,8].values
#X
y


# **6.2 Splitting the dataset to Train and Test Set**

# In[ ]:


from sklearn.model_selection import train_test_split   #cross_validation doesnt work any more
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) 
#X_train


# **6.3 Feature Scaling**

# In[ ]:


from sklearn.preprocessing import StandardScaler 
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
#X_train


# **6.4 Importing the Keras Library**

# In[ ]:


import keras 
from keras.models import Sequential 
from keras.layers import Dense 


# **6.5 Initialising the ANN**

# In[ ]:


classifier_6=Sequential()


# **6.6 Adding the input layer and the first hidden layer**

# In[ ]:


classifier_6.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=8))


# **6.7 Adding the second hidden Layer**

# In[ ]:


classifier_6.add(Dense(output_dim=5,init='uniform',activation='relu'))


# **6.8 Adding the output layer**

# In[ ]:


classifier_6.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))


# **6.9 Compliling the ANN**

# In[ ]:


classifier_6.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# **6.10 Fitting the ANN to training set**

# In[ ]:


classifier_6.fit(X_train, y_train,batch_size=10,nb_epoch=100)


# **6.11 Predicting the test set results**

# In[ ]:


y_pred=classifier_6.predict(X_test)
y_pred=(y_pred>0.7)
y_pred


# **6.12 Making the confusion matrix**

# In[ ]:


from sklearn.metrics import confusion_matrix  #Class has capital at the begining function starts with small letters 
cm=confusion_matrix(y_test,y_pred)
import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Test for Test Dataset")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.show()


# Correct prediction=30+42=72
# 
# Wrong predictions =8+0=8
# 
# Accuracy of Prediction =(72/80)*100 =90 %

# # Predicting the chance of Admission using ANN

# In[ ]:


admt.loc[admt['Chance']>=0.5,['Chance']]=1
admt.loc[admt['Chance']<0.5,['Chance']]=0


# ## Feature scaling of the data 

# In[ ]:


admt['GRE']=admt['GRE']/admt['GRE'].max()
admt['TOEFL']=admt['TOEFL']/admt['TOEFL'].max()
admt['UnivRating']=admt['UnivRating']/admt['UnivRating'].max()
admt['SOP']=admt['SOP']/admt['SOP'].max()
admt['LOR ']=admt['LOR ']/admt['LOR '].max()
admt['CGPA']=admt['CGPA']/admt['CGPA'].max()


# In[ ]:


import keras

X=admt[['GRE','TOEFL','UnivRating','SOP','LOR ','CGPA','Research']]
# labels y are one-hot encoded, so it appears as two classes 
y = keras.utils.to_categorical(np.array(admt["Chance"]))


# ## Splitting the data set 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=100)


# ## Defining a model 

# In[ ]:


from keras.models import Sequential
from keras.layers.core import Dense, Activation


model = Sequential()
model.add(Dense(128, input_dim=7))
model.add(Activation('sigmoid'))
model.add(Dense(32))
model.add(Activation('sigmoid'))
model.add(Dense(2))
model.add(Activation('sigmoid'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# ## Training the model 

# In[ ]:


model.fit(X_train, y_train, epochs=1000, batch_size=100, verbose=0)


# In[ ]:


score = model.evaluate(X_test, y_test)
print(score)


# In[ ]:


y_pred=model.predict(X_test)
y_pred

