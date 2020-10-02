#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##################Problem statement#######################
#Titanic: Machine Learning from Disaster
#some group of people were more likely to escape in the Titanic sinking.
#we need to find out what sorts of people are likely to survive.
#Lets use simple logistic regression and see if it works.

#I have referred to some other kernels and whatever I have written below are as per my understanding.
#Reference links:https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/Random%20Forest%20Tutorial.ipynb
#Thanks to WillKoehrsen who had explained Random Forest algorithm in detail.

#I have used one of the most widely used algorithm in machine learning which is Random Forest


# In[ ]:


##################Approach##############################
#1. understanding the problem statement
#2. understanding the data
#3. Data preparation and missing value imputation
#4. Exploratory data analysis
#5. Feature engineering.
#6. Model building
#7. Training the model to get the correct parameters.
#8. Testing the model


# In[ ]:


#import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sklearn.model_selection
import sklearn.svm as svm
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
import os


# In[ ]:


############# 1. Understanding the data ##############
#read the files
train_df=pd.read_csv("../input/train.csv")
test_df=pd.read_csv("../input/test.csv")


# In[ ]:


#check the shape of the data
print('Train data shape',train_df.shape) #
print('Test data shape',test_df.shape)


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


#There are some categorical variables like Pclass,Sex,Age,SibSp,Parch,cabin,embarked.
#Continuous variable in the dataset is fare.


# In[ ]:


################## 2.Data preparation and missing value imputation ######################


# In[ ]:


#lets check for the missing values in the data 
train_df.isnull().sum()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


#There are missing values in the data for the age,cabin and Embarked in the training dataset.
#There are missing values in the data for age,fare and cabin in the test dataset.
#lets impute them with relevant values


# In[ ]:


########################## first impute the field Age with relevant values #################################3


# In[ ]:


train_df['Age'].dropna(inplace=True)
sns.distplot(train_df['Age'])


# In[ ]:


#It is not a perfect normal distribution. the Age is right skewed.

#we have to use either mean or median to impute
print('The mean age is %.2f' %(train_df['Age'].mean(skipna=True)))
print('The median age is %.2f' %(train_df['Age'].median(skipna=True)))

#Both the values are almost same. Since it is right skewed the mean might give a biased result.
#so lets take the median. 


# In[ ]:


#To be more specific on the missing value imputation lets use the below approach.
#Find out the median of the Age at the granularity of pclass and Sex. Then do the imputation.


# In[ ]:


""" 
Output of the below code can be achieved by the code in the next line as well.

#find the median values specific to Sex and class
print('pclass1:Sex_Male : ',train_df[(train_df.Pclass==1) & (train_df.Sex=='male')]['Age'].median())
print('pclass2:Sex_Male : ',train_df[(train_df.Pclass==2) & (train_df.Sex=='male')]['Age'].median())
print('pclass3:Sex_Male : ',train_df[(train_df.Pclass==3) & (train_df.Sex=='male')]['Age'].median())

print('pclass1:Sex_feMale : ',train_df[(train_df.Pclass==1) & (train_df.Sex=='female')]['Age'].median())
print('pclass2:Sex_feMale : ',train_df[(train_df.Pclass==2) & (train_df.Sex=='female')]['Age'].median())
print('pclass3:Sex_feMale : ',train_df[(train_df.Pclass==3) & (train_df.Sex=='female')]['Age'].median())"""


# In[ ]:


data=train_df[['Pclass','Sex','Age']]
data.groupby(['Pclass','Sex']).median()


# In[ ]:


#lets impute the values
def impute_age(cols):
    age=cols[0]
    sex=cols[1]
    pclass=cols[2]
    if pd.isnull(age):
        if sex=='male':
            if pclass==1:
                return 40
            elif pclass==2:
                return 30
            elif pclass==3:
                return 25
            else:
                print('pclass not in 1,2,3',pclass)
                return np.nan
        elif sex=='female':
            if pclass==1:
                return 35
            elif pclass==2:
                return 28
            elif pclass==3:
                return 21.5
            else:
                print('pclass not in 1,2,3',pclass)
                return np.nan
    
        else:
            print('Sex not in male or female',sex)
    else:
        return age
    
    


# In[ ]:


#Lets copy the train_df and test_df into train_data and test_data
train_data=train_df.copy()
test_data=test_df.copy()


# In[ ]:


train_data['Age']=train_data[['Age','Sex','Pclass']].apply(impute_age,axis=1)


# In[ ]:


test_data['Age']=test_data[['Age','Sex','Pclass']].apply(impute_age,axis=1)


# In[ ]:


################################# next lets see the missing values in the cabin ########################################
print('missing "Cabin" records is %.2f%%' %((train_df['Cabin'].isnull().sum()/train_df.shape[0])*100))


# In[ ]:


#since majority of the values are missing in this field it is better to remove that field.
train_data.drop('Cabin',axis=1,inplace=True)
test_data.drop('Cabin',axis=1,inplace=True)


# In[ ]:


"""
This function can also be used to drop the field 'Cabin

def dropCabin(train,test,drop='Cabin'):
    for i in [train,test]:
        for z in drop:
            del i[z]
    return train,test"""


# In[ ]:


######################### next lets see the missing values in the Embarked ####################################
#as seen above only two records have missing values.So lets impute them with the most frequently occuring value
train_data['Embarked'].value_counts()


# In[ ]:


sns.countplot(data=train_data,x='Embarked')


# In[ ]:


#embarked as 'S' is the most frequently occuring one.
train_data['Embarked'].value_counts().idxmax()


# In[ ]:


#lets write a function to impute the null values in the embarked.


# In[ ]:


def embarkedImpute(train,test):
    for i in [train,test]:
        i['Embarked']=i['Embarked'].fillna('S')
    return train,test


# In[ ]:


train_data,test_data=embarkedImpute(train_data,test_data)


# In[ ]:


#this is another way of imputing missing values for embarked.
#train_data['Embarked'].fillna(train_df['Embarked'].value_counts().idxmax(),inplace=True)
#test_data['Embarked'].fillna(test_df['Embarked'].value_counts().idxmax(),inplace=True)


# In[ ]:


train_data.isnull().sum()


# In[ ]:


#There is one missing value in the field Fare.


# In[ ]:



sns.distplot(train_df['Fare'])


# In[ ]:


#Lets impute the missing value in the Fare with the mean.
test_data['Fare'].fillna(train_data['Fare'].mean(),inplace=True)


# In[ ]:


test_data.isnull().sum()


# In[ ]:


#we are done with the missing value imputation. There are no other data preparation needed for now.


# In[ ]:


############################# Exploratory Data analysis ############################################
#This step will help you in the feature engineering.
#EDA will show some hidden patterns in the data.
#Insights from EDA can be used to create new features in the dataset.


# In[ ]:


#lets define some hypothesis and we can confirm that using the exploratory data analysis

#H1: Passengers who are on the premium class are likely to escape compared to other lower class passengers.
#H2: Female passengers are likely to survive compared to male.
#H3: Passengers who are kids are likely to escape
#H4: Passengers travelling with family are more likely to escape than solo passengers.
#H5: Passengers who paid more fare are more likely to escape. This may be because the high class tickets are more costly.

#lets check above are true or not and also try to add more features.


# In[ ]:


#H1: Passengers who are on the premium class are likely to escape compared to other lower class passengers.
sns.barplot('Pclass','Survived',data=train_data)


# In[ ]:


train_data['Survived'].groupby(train_data['Pclass']).mean()


# In[ ]:


sns.countplot(train_data['Pclass'],hue=train_data['Survived'])


# In[ ]:


#The conclusion is that passengers in the class 1 are likely to survive compared to 2 and 3.
#You can notice that the most of the passengers in the class 3 were not able to survive.


# In[ ]:


#H2: Female passengers are likely to survive compared to male.
sns.barplot('Sex','Survived',data=train_data)


# In[ ]:


train_data['Sex'].value_counts(normalize=True)


# In[ ]:


train_data['Survived'].groupby(train_data['Sex']).mean()


# In[ ]:


#The conclusion is that female passengers are more likely to survive compared to the Male passengers.


# In[ ]:


#H3: Passengers who are kids are likely to escape
#Lets explore the Age field.


# In[ ]:


#check the distribution of survived and not by age.

plt.figure(figsize=(30,8))
ax=sns.distplot(train_data["Age"][train_data.Survived == 1])
sns.distplot(train_data['Age'][train_data.Survived==0])
plt.legend(['Survived','Died'])
ax.set(xlabel='Age')
plt.xlim(0,85)
plt.show()


# In[ ]:


#lets plot a bar chart
plt.figure(figsize=(50,8))
avg_survival_byage = train_data[["Age", "Survived"]].groupby(['Age'], as_index=False).mean()
g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color="LightSeaGreen")
plt.show()


# In[ ]:


#lets categorize the field 'Age' into different buckets
train_data['Survived'].groupby(pd.qcut(train_data['Age'],9)).mean()


# In[ ]:


pd.qcut(train_data['Age'],9).value_counts()


# In[ ]:


#The conclusion is that There is a higher chance of escape if you are below age 16.


# In[ ]:


#H4: Passengers travelling with family are more likely to escape than solo passengers.


# In[ ]:


#lets check the field sibsp
sns.barplot('SibSp','Survived',data=train_data)

#Avg survival by age
#plt.figure(figsize=(20,8))
#avg_survival_bysibsp = train_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()
#g=sns.barplot(x='SibSp',y='Survived',data=avg_survival_bysibsp)


# In[ ]:


#lets check the field parch
sns.barplot('Parch','Survived',data=train_data)


# In[ ]:


#individuals travellling alone are more likely to die.


# In[ ]:


train_data['Survived'].groupby(train_data['SibSp']).mean()


# In[ ]:


train_data['Survived'].groupby(train_data['Parch']).mean()


# In[ ]:


#The conclusion is that people who travel with a family are more likely to escape


# In[ ]:


#H5: Passengers who paid more fare are more likely to escape. This may be because the high class tickets are more costly.


# In[ ]:


######################### now lets explore the fare ###############################
plt.figure(figsize=(20,8))
ax=sns.kdeplot(train_data['Fare'][train_data.Survived==1],shade=True)
sns.kdeplot(train_data['Fare'][train_data.Survived==0],shade=True)
plt.legend(['Survived','Died'])
ax.set(xlabel='Fare')
plt.show()


# In[ ]:


pd.qcut(train_data['Fare'],3).value_counts()


# In[ ]:


train_data['Survived'].groupby(pd.qcut(train_data['Fare'],6)).mean()


# In[ ]:


#The conclusion is that the people who paid more fare are more likely to escape.


# In[ ]:


#Now we have checked all the hypothesis which were defined as per our knowledge and assumptions.
#But there can be still some some insights hidden in the data.
#so Lets explore some other fields also in the dataset.


# In[ ]:


#now lets look at the  field embarked
sns.barplot('Embarked','Survived',data=train_data)


# In[ ]:


#Those people who embarked from C are more likely to escape


# In[ ]:


train_data['Embarked'].value_counts(normalize=True)


# In[ ]:


#Lets check the Name Title and see if we can use that in the model
train_data['Name'].apply(lambda x:x.split(',')[1]).apply(lambda x:x.split()[0]).value_counts()


# In[ ]:


train_data['Survived'].groupby(train_data['Name'].apply(lambda x:x.split(',')[1]).apply(lambda x:x.split()[0])).mean()


# In[ ]:


sns.barplot(train_data['Name'].apply(lambda x:x.split(',')[1]).apply(lambda x:x.split()[0]),'Survived',data=train_data)


# In[ ]:


#There are some passengers with specific title are more likely to survive.
#If you have watched the movie titanic you might remember the below things.
    #The captain of the ship did not survive.
    #There was a priest who was ready to sacrifice his life by giving his lifejacket to some one else.
    #My assumption may not be right but if you look at the data you can see that passengers with title as Capt. and Rev are not survived.
#The title of the name can be a valid field which is usefull.


# In[ ]:


# There may be more scope for the exploratory data analysis and feature engineering with expert knowledge and imagination.
# For now I am giving a pause for the EDA and moving onto the feature Engineering.


# In[ ]:


#Below are the stories which we can use in feature creation.
#Use helper functions wherever needed to do feature engineering


# In[ ]:


# story1. passengers travelling alone are less likely to survive.
# story2. passengers with certain name title are more likely to survive.
# story3. passengers travelling in the first class are more likely to escape. 
# story4. female passengers are more likely to escape 
# story5. people who paid less fare are less likely to escape
# story6. people who embarked from C are more likely to escape


# In[ ]:


#lets ensure we have relevant variables to support the above stories


# In[ ]:


# story1 : passengers travelling alone are less likely to survive.

#lets use a function to do this task.

def fam_size(train,test):
    for i in [train,test]:
        i['Fam_Size']=np.where((i['SibSp']+i['Parch'])==0,'Solo',
                               np.where((i['SibSp']+i['Parch'])<=3,'SmallFamily',
                               'BigFamily'))
        del i['SibSp']
        del i['Parch']
    return train,test


# In[ ]:


# story2. passengers with certain title are more likely to survive.
def NameTitle(train,test):
    for i in [train,test]:
        i['Name_Title']=i['Name'].apply(lambda x:x.split(',')[1]).apply(lambda x:x.split()[0])
        del i['Name']
    return train,test


# In[ ]:


# story3. passengers travelling in the first class are more likely to escape. 
# we can use the field Pclass in the model.Since it is a categorial variable we should create dummy variables for that.


# In[ ]:


# story4. female passengers are more likely to escape
# we can use the field sex in the model.Since it is a categorial variable we should create dummy variables for that.


# In[ ]:


# story5. people who paid less fare are less likely to escape 
# we can use the field fare in the model.


# In[ ]:


# story6. people who embarked from C are more likely to escape
# we can use the field embarked in the model.


# In[ ]:


#add the family size variable to the dataset
train_data,test_data=fam_size(train_data,test_data)


# In[ ]:


#add the name_title to the dataset
train_data,test_data=NameTitle(train_data,test_data)


# In[ ]:


train_data.head()


# In[ ]:


#create dummy variables to support the stories 1,2,3,4 and 6.


# In[ ]:


#train_data=pd.get_dummies(train_data,columns=["Pclass","Embarked",'Sex','Fam_Size','Name_Title'])
#test_data=pd.get_dummies(test_data,columns=['Pclass','Embarked','Sex','Fam_Size','Name_Title'])


# In[ ]:


#Lets use a function to create dummy variables
def dummies(train,test,columns=['Pclass','Embarked','Sex','Fam_Size','Name_Title']):
    for column in columns:
        train[column]=train[column].apply(lambda x:str(x))
        test[column]=test[column].apply(lambda x:str(x))
        good_cols=[column+'_'+i for i in train[column].unique() if i in test[column].unique()]
        train=pd.concat((train,pd.get_dummies(train[column],prefix=column)[good_cols]),axis=1)
        test=pd.concat((test,pd.get_dummies(test[column],prefix=column)[good_cols]),axis=1)
        del train[column]
        del test[column]
    return train,test


# In[ ]:


print('Train_DataShape',train_data.shape)
print('Test_DataShape',test_data.shape)


# In[ ]:


train_data,test_data=dummies(train_data,test_data,columns=['Pclass','Embarked','Sex','Fam_Size','Name_Title'])


# In[ ]:


print('Train_DataShape',train_data.shape)
print('Test_DataShape',test_data.shape)


# In[ ]:


#train_data = train_data.rename(columns = {'Name_Title_Don.': 'Name_Title_Dona.'})


# In[ ]:


#There are some fields which are not usefulin train dataset.lets remove them
train_data.drop('PassengerId',axis=1,inplace=True)
train_data.drop('Ticket',axis=1,inplace=True)
"""train_data.drop('Parch',axis=1,inplace=True)
train_data.drop('Ticket',axis=1,inplace=True)
train_data.drop('Age',axis=1,inplace=True)
train_data.drop('Sex_male',axis=1,inplace=True)
train_data.drop('Pclass_3',axis=1,inplace=True)
train_data.drop('Embarked_Q',axis=1,inplace=True)"""


# In[ ]:


#There are some fields which are not useful test dataset.lets remove them
test_data.drop('PassengerId',axis=1,inplace=True)
test_data.drop('Ticket',axis=1,inplace=True)
"""test_data.drop('PassengerId',axis=1,inplace=True)
test_data.drop('Parch',axis=1,inplace=True)
test_data.drop('Ticket',axis=1,inplace=True)
test_data.drop('Age',axis=1,inplace=True)
test_data.drop('Sex_male',axis=1,inplace=True)
test_data.drop('Pclass_3',axis=1,inplace=True)
test_data.drop('Embarked_Q',axis=1,inplace=True)"""


# In[ ]:


train_data.info()


# In[ ]:


#select only those fields which are needed for modelling.
cols=["Age","Fare","Pclass_1","Pclass_2","Pclass_3","Embarked_C","Embarked_Q","Embarked_S",
      "Sex_female","Sex_male","Fam_Size_BigFamily","Fam_Size_SmallFamily","Fam_Size_Solo","Name_Title_Mr.",
      "Name_Title_Mrs.","Name_Title_Miss.","Name_Title_Master.","Name_Title_Rev.","Name_Title_Dr.","Name_Title_Ms.",
      "Name_Title_Col.","Survived"]


# In[ ]:


#Lets give a pause to the feature engineering. Definitely there are more scope of feature engineering.Good features will
#definitely help us to get more accuracy in the predictions.


# In[ ]:


################### Modeling ##################################


# In[ ]:


#The problem statement we have is a classic binary classification problem.
#There are different methods used to solve binary classification problems.for exaple logistic regression.
#In this notebook I have used randomforest algorithm for modeling.


# In[ ]:


#what is random forest?

#Random forest creates an ensemble model out of hundreds of thoudsands of trees to reduce the variance.Each tree is trained on
#a random set of the observations and for each split of a node only a subset of the features are used for making a split. When
#making predictions the random forest averages the predictions for each of the individual decision trees for each data point 
#in order to arrive at a final classification.

#Suppose out training dataset is represented by T and suppose dataset has M features

#T={(X1,y1),(X2,y2),...,(Xn,yn)}
#Xi is input vector {Xi1,Xi2,.....,XiM}
#yi is the label (or output or class)

#Random forest algorithm is based on promarily two methods
#1)Bagging
#2)Random Subspace method

#Suppose we decide to have S number of trees in our forest then we first create S datasets of "same size as original" 
#created from the random resampling of data in T with replacement(n times for each dataset). 

#This will result in {T1,T2,...,TS} datasets.

#Each of these is called bootstrap dataset.

#Due to with replacement every dataset Ti can have duplicate data records and Ti can be missing several data records
#from the original datasets.This is calld Bootstraping.

#Now RF creates S trees and uses m(=sqrt(M) or =floor(lnM+1)) random subfeatures out of M possible features to create any tree.
#This is called random subspace method

#so for each Ti bootstrap dataset you create a tree Ki.

#If you want to classify some input data D={x1,x2,...xM} you let it pass though each tree and produce S outputs
#(one for each tree). Which can be denoted by Y={y1,y2,...ys}. Final prediction is a majority vote on this set.


# In[ ]:


#What is OOB (Out of Bag error)?
#consider each observation on the training set as a test observation.

#Since each tree is a bootstrap sample each observation in the training set can be used as a test observation by the trees
#that do not have this observation in the bootstrap sample.

#All these trees predict on this observation and you get an error on this observation by calculating the aggregation of errors
#on this observation.


# In[ ]:


RSEED=150


# In[ ]:


#Create a train and test dataset
from sklearn.model_selection import train_test_split

#extract the labels
labels=np.array(train_data.pop('Survived'))

#lets keep 30% examples in the training data
train,test,train_labels,test_labels=train_test_split(train_data,labels,
                                                    stratify=labels,
                                                    test_size=0.3,
                                                    random_state=RSEED)


# In[ ]:


print('train shape',train.shape)
print('test shape',test.shape)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
#create a model with 100 trees
model=RandomForestClassifier(n_estimators=100,
                            random_state=RSEED,
                            max_features='sqrt',
                            n_jobs=-1,verbose=1)

#fit on training data
model.fit(train,train_labels)


# In[ ]:


#now lets check how many nodes are there for each tree on average and maximum depth of each tree. 
#There are 100 trees in the forest
n_nodes=[]
max_depths=[]

for ind_tree in model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
    
print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average max depth {np.mean(max_depths)}')


# In[ ]:


#lets check the random forest results
train_rf_predictions=model.predict(train)
train_rf_probs=model.predict_proba(train)[:,1]


# In[ ]:


rf_predictions=model.predict(test)
rf_probs=model.predict_proba(test)[:,1]


# In[ ]:


#now lets evaluate the model by checking the recall,precision and the ROC curve.
#A classification algorithm is evaluated on the basis of its capability to distinguish between the different classes.
#There are different measures to do this evaluation.


# In[ ]:


#what is recall?
#what proportion of the actual positives are identified correctly.
#Recall=TP/(TP+FN)

#what is precision?
#what proportion of positive idetifications are actually cocorrect?
#precision=TP/(TP+FP)


# In[ ]:


#what is ROC curve and AUC?
#ROC : Receiver operating characteristic is a graph showing the performance of a classification model at all classification
#thresholds

#This curve plots two parameters:
#1) True Positive Rate
#2) False Positive Rate

#ROC is a probability curve and AUC represent degree or measure of separability.It tells how much a model is capable of
#distinguishing between classes. 

#Higher the AUC better the model is at distinguishing.


# In[ ]:


#create a function for model evaluation.
#write a function that calculates the number of metrics for the baseline (guessing most common lablel in training data.),the 
#testing predictions and the training predictions


# In[ ]:


from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve


# In[ ]:


def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves');


# In[ ]:


from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)


# In[ ]:


evaluate_model(rf_predictions,rf_probs,train_rf_predictions,train_rf_probs)


# In[ ]:


cm = confusion_matrix(test_labels,rf_predictions)
plot_confusion_matrix(cm, classes = ['Not Survived', 'Survived'],
                      title = 'Confusion Matrix')


# In[ ]:


# Features for feature importances
features = list(train.columns)


# In[ ]:


#Now lets check the feature importances.
fi_model=pd.DataFrame({'feature':features,
                      'importance':model.feature_importances_}).\
                        sort_values('importance', ascending = False)


# In[ ]:


#how does the feature importance is calculated?
######## Feature importance ############
#In decision tree every node is a condition how to split values in a single feature, so that similar values of dependent
#variable end up in the same set after the split.The condition is based on impurity, which in case of classification is Gini impurity
#or information gain.For regression tree it is variance.

#So when training a tree we can compute how much each feature contributes to decreasing the weighted impurity.In random forest
#we are talking about averaging the decrese in impurity over trees.


# In[ ]:


#Random Forest Optimization through Random Search
#we need to find out the best hyperparameters so that the performance is the best.
#Randomly select the combinations of hyperparameters from a grid,evaluate them using cross validation on the training data
#and return the values that perform the best.
from sklearn.model_selection import RandomizedSearchCV

# Hyperparameter grid
param_grid = {
    'n_estimators': np.linspace(10, 200).astype(int),
    'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
    'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),
    'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
}

# Estimator for use in random search
estimator = RandomForestClassifier(random_state = RSEED)

# Create the random search model
rs = RandomizedSearchCV(estimator, param_grid, n_jobs = -1, 
                        scoring = 'roc_auc', cv = 3, 
                        n_iter = 10, verbose = 1, random_state=RSEED)

# Fit 
rs.fit(train, train_labels)


# In[ ]:


#find the best parameter
rs.best_params_


# In[ ]:


#now use the best model
best_model=rs.best_estimator_


# In[ ]:


train_rf_predictions = best_model.predict(train)
train_rf_probs = best_model.predict_proba(train)[:, 1]

rf_predictions = best_model.predict(test)
rf_probs = best_model.predict_proba(test)[:, 1]


# In[ ]:


train_rf_predictions = model.predict(train)
train_rf_probs = model.predict_proba(train)[:, 1]

rf_predictions = model.predict(test)
rf_probs = model.predict_proba(test)[:, 1]


# In[ ]:



n_nodes = []
max_depths = []

for ind_tree in best_model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
    
print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')


# In[ ]:


evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)


# In[ ]:


test_data['Survived']=best_model.predict(test_data)


# In[ ]:


test_data['PassengerID']=test_df['PassengerId']


# In[ ]:


submission=test_data[['PassengerID','Survived']]


# In[ ]:


submission.to_csv("submission.csv", index=False)


# In[ ]:




