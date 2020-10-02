#!/usr/bin/env python
# coding: utf-8

# **WELCOME!! WELCOME!! WELCOME!!**                                                                Accuracy : 76.55
# 
# **ITS FOR BEGINNERS TITANIC DATASET PROBLEM WHERE WE'LL UNDERSTAND EVERY ASPECT OF DATASET AND DEALING WITH PROBLEMS**
# 
# Solving a problem in datascience is not the only thing that makes it SUPER. 
# 
# > WE CANT SOLVE THE EQUATION UNLESS WE HAVE ALL THE VARIABLES          
#                                                                 by **Tony Stark** 
#                                                             
# So How are we gonna do it :
# 
# **Data Pre Processing: Bringing data into right format**
# 
# **Data Visualization : To understand how data looks like so that we could identify the right model for it.**
# 
# **Tesing : To understand whether the model is doing good job or not**
# 
# **Applying the model : At last applying the model**
# 
# 
# Lets Inport the Dependencies first :) 

# In[ ]:


import pandas as pd        # For all the Preprocessing Stuff
import numpy as np      # For matrix MAth
import seaborn as sns     # For data visualization
import matplotlib.pyplot as plt   #Matplotlib is a heleper function for SNS a tool for data visualization
from sklearn.preprocessing import MinMaxScaler     # Used for feature Scaling, dont worry we'll dig in


# **Time for Loading the dataset so that we can do some ACTION **

# In[ ]:


Train_Data = pd.read_csv('../input/titanic/train.csv')
Test_Data = pd.read_csv('../input/titanic/test.csv')

#Lets See How Data Looks Like 
print(Train_Data.head())


# Man We really need to bring it in right format as we cant pass it directly to the model it's a mess.

# In[ ]:


#Lets first see the information of the dataset.
print(Train_Data.describe())          # It gives count of every feature, mean ,std,min value ,max value and so on.......


# In[ ]:


# Lets Remove un necessary attributes that might not affect much to you model

# Name is not required that's for sure
Train_Data.drop(['Name'],axis = 1,inplace = True)
# Ticket Number is also not required 
Train_Data.drop(['Ticket'],axis = 1,inplace = True)
print(Train_Data.head())


# In[ ]:


# Step 1 : Lets check that wheteher we have some null values or not in our Training set
print(Train_Data.isnull().any())
#Looks like in Age and Cabin and Embarked feature we do have some null values


# Lets First deal with this null values because they can really Blow up you calculations.
# But How do we deal with them ????
# 

# In[ ]:


print(Train_Data[Train_Data['Cabin'].isnull()].index) #This shows that 687 entries and the following index are NULL for Cabin.
#Total number of entries in dataset is 891 (shown in describe()) almost everything is NULL so you can remove this columns no harm would be there
Train_Data.drop(['Cabin'],axis = 1,inplace = True)
print(Train_Data.head())


# In[ ]:


print(Train_Data[Train_Data['Embarked'].isnull()].index) # This shows that 2 entries and the following index are NULL for Embarked.
# We can simply forward or backward fill Embarked 
Train_Data['Embarked'].fillna(method = 'ffill',inplace = True)
print(Train_Data.isnull().any())


# In[ ]:


print(Train_Data[Train_Data['Age'].isnull()].index)    # This shows that 177 entries and the following index are NULL for age.
#Now how to fill the null values of Age ??
#First lets bring Sex and Embarked into right form
Train_Data['Sex'] = pd.factorize(Train_Data['Sex'])[0]
Train_Data['Embarked'] = pd.factorize(Train_Data['Embarked'])[0]
print(Train_Data.head())
Diag = sns.PairGrid(Train_Data)
Diag.map(plt.scatter)
plt.show()


# If we see this Pair grid in detail we can see that the Age attribute and Pclass or Sex they are straight so we could fill the age attribute something like say if Sex == M then put there the average of only mail Age category or we could do it Pclass wise. I am gonna do it Sex wise here :)

# In[ ]:


def Fillme(Value,Male_Avg,Female_Avg):         # Helper Function 
    Sex = Value[0]        # Value[0] is the sex attribute
    Age = Value[1]        # Value[1] is the Age attribute
    
    if pd.isnull(Age):
        if Sex == 0:      # If its Male else Female
            return Male_Avg
        else:
            return Female_Avg
    else:
        return Age
    


# In[ ]:


Male_Avg = np.mean(Train_Data[Train_Data['Sex'] == 0].Age)               # 0 is male 1 is female by factorize function
Female_Avg = np.mean(Train_Data[Train_Data['Sex'] == 1].Age) 

# Now lets fill them 
Train_Data['Age'] = Train_Data[['Sex','Age']].apply(lambda x : Fillme(x,Male_Avg,Female_Avg) , axis = 1)  # Fill me is a helper Function That is defined above'
print(Train_Data.isnull().any())
print('\n\n\n')
print(Train_Data.head())


# Now we got rid of NULL Values now Lets bring everyone of them on same Scale. 
# Scale ???    Lets understand:
# >                 A     B
# >                 5KG   5000G
# >                 7KG   7000G
# >                 
#   if we see this dataset here two columns A and B are exactly the same but they look different as they are in different scale. one is in KG and other is in Grams. So feature scaling means bringing everything on the same scale. Its really important for the algorithm that uses distance metric like SVM,Logistic regression. IT ALSO HELPS GRADIENT DESCENT TO CONVERGE FAST

# In[ ]:


Target_Variable = np.array(Train_Data.Survived)
Train_Data.drop(['Survived'],axis = 1,inplace = True)      # Dropping target Variable
# Now first converting the Dataset into Data Matrix
Data_Matrix = Train_Data.values
Scaler = MinMaxScaler()
Data_Matrix = Scaler.fit_transform(Data_Matrix)         # Now Every variable is on the same scale
print(Data_Matrix)


# At this Point our First Step Data Preprocessing is Done.  FINALLY........
# 
# Lets move to Data Visualization step to see how data actually looks like:
# But we have a very big feature space how are we gonna do it ??
# 
# Simple by bringing them into lower dimension by PCA
# 
# **For Classification problem we can plot upto 4 dimensional feature space where 1 space is target which can be shown by diferent colour
# For Regression we can plot upto 3 dimension where 1 dimension is your Real target Variable**

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D           # Used for 3D Visualization of the data set
from sklearn.decomposition import PCA             # PCA Decomposition

pca = PCA(n_components=3)     # We are choosing the first three component
pca.fit(Data_Matrix)
#print(pca.explained_variance_ratio_)       # Lets see how much variation we covered. This Shows that the Principal components 1,2 and 3 
# encodes [0.36751082 0.26115811 0.14993309] amount of variation.  
Components = pca.fit_transform(Data_Matrix)

Matrix = np.hstack((Components,np.array(Target_Variable).reshape(-1,1)))   # Stacking the Target Variable with matrix

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')        # Creating a subplot for 3d view

Dict = {}
for instance in Matrix:                                # this loop simply stacks up entries where target is Zero and where its One
    if(int(instance[-1]) in Dict):
        Dict[instance[-1]].append(instance[:-1])
    else:
        Dict[int(instance[-1])] = []

for K,V in Dict.items():
    Array = np.array(V)
    if K == 0:
        ax.scatter(Array[:,0],Array[:,1],Array[:,-1],c='blue',depthshade = False,marker = '*')
    else:
        ax.scatter(Array[:,0],Array[:,1],Array[:,-1],c='red',depthshade = False,marker = 'D')

plt.show()


# This is How Dataset Looks Like Blue and Red are our classes and they are definately Non Linear. So we should choose our Model according to that..........
# 
# Here Ends the Second Step i.e. Data Visualization.
# 
# **Now Lets apply the model and Check the Performance**

# In[ ]:


from sklearn.model_selection import learning_curve    # For Checking whether model gets Overfitted,Underfitted or WHAT

def LearningCurve(Classifier,Datamatrix,Label):               # IT PLOTS THE LEARNING CURVE
    train_sizes, train_scores, validation_scores = learning_curve(
    estimator = Classifier,
    X = Datamatrix,
    y = Label, train_sizes = np.linspace(0.1,1.0,10), cv = 5,
    scoring = 'accuracy',shuffle = True)
    train_scores_mean = train_scores.mean(axis = 1)     # Mean of Rows of training score
    validation_scores_mean = validation_scores.mean(axis = 1)      # Mean of rows of Testing score
    Train_Score =  pd.Series(train_scores_mean, index = train_sizes)
    validation_Score = pd.Series(validation_scores_mean, index = train_sizes)

    plt.plot((np.array(Train_Score)), "r-+", linewidth=2, label="train")          # Red COLOR IS Training
    plt.plot((np.array(validation_Score)), "b-", linewidth=3, label="val")        # Blue Color is Validation


# In[ ]:


from sklearn.model_selection import cross_val_score        # For Cross Validation
from sklearn.svm import SVC         # We are using Support vector machine woth gaussain kernel
from sklearn.neural_network import MLPClassifier      # Lets See Neural Network

clf = SVC(gamma = 'scale',kernel = 'rbf')

print(cross_val_score(clf,Data_Matrix,Target_Variable,cv = 5))
LearningCurve(clf,Data_Matrix,Target_Variable)           # Lets See How Well our SVM Classifier is Doing
plt.show()


# Now Lets Do Prediction on Test Data. First Lets Quickly Preprocess it.....

# In[ ]:


Test_Data.drop(['Name','Ticket','Cabin'],axis = 1,inplace = True)
Test_Data.Embarked = pd.factorize(Test_Data.Embarked)[0]
Test_Data.Sex = pd.factorize(Test_Data.Sex)[0]

Male_Avg = np.mean(Test_Data[Test_Data['Sex'] == 0].Age)               # 0 is male 1 is female by factorize function
Female_Avg = np.mean(Test_Data[Test_Data['Sex'] == 1].Age) 

Test_Data['Age'] = Test_Data[['Sex','Age']].apply(lambda x : Fillme(x,Male_Avg,Female_Avg) , axis = 1)  # Fill me is a helper Function That is defined above'
Test_Data.Fare.fillna(np.mean(Test_Data.Fare),inplace = True)
Test_Matrix = Test_Data.values
Scaler = MinMaxScaler(feature_range = (0, 1))
Test_Matrix = Scaler.fit_transform(Test_Matrix)


# In[ ]:


clf.fit(Data_Matrix,Target_Variable)   #Fitting the Model
Prediction = clf.predict(Test_Matrix)
print(Prediction)

