#!/usr/bin/env python
# coding: utf-8

# # TITANIC DATA ANALYTICS_Python

# Hello,<br>
# We are beginner in Kaggle. <br> 
# So We refer to yassine ghouzam's Kernel(https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling) <br>
# We are not good at Machine Learning. Therefore, welcome any opinion about our Kernel.<br>
# 
# Thank you!<br>

# 

# <ul>
# 	<li>0. Preparation</li>
# 	<li>1. Data Exploration</li>
# 		<ul>
# 			<li>1.1 Age</li>
# 			<li>1.2 Sex</li>
# 			<li>1.3 Embarked</li>
# 			<li>1.4 Pclass</li>
# 			<li>1.5 Parch</li>
# 			<li>1.6 SibSp</li>
# 			<li>1.7 Fare</li>
# 			<li>1.8 Relations among Variables</li>
# 		</ul>
# 	<li>2. Feature Engineering</li>
# 		<ul>
# 			<li>2.1 Name</li>
# 			<li>2.2 Parch+SibSp->FamilySize</li>
# 		</ul>
#     <li>3. Filling Missing Values</li>
#         <ul>
#             <li>3.1 Fare</li>
#             <li>3.2 Embarked</li>
#             <li>3.3 Age</li>
#         </ul>
#     <li>4. Data Categorization & Dummy</li>
#         <ul>
#             <li>4.1 Sex</li>
#             <li>4.2 Embarked</li>
#             <li>4.3 Pclass</li>
#             <li>4.4 Title</li>
#             <li>4.5 FamilySize</li>
#         </ul>
# 	<li>5. Modeling</li>
# 		<ul>
# 			<li>5.1 Cross-Validation</li>
# 			<li>5.2 Tunning Parameters</li>
# 			<li>5.3 Learning Curve</li>
# 			<li>5.4 Model Feature Importance</li>
# 			<li>5.5 Ensemble Modeling</li>
# 		</ul>
# 	<li>6. Prediction</li>
# <ul>

# # 0. Preparation

# ## Import Packages

# In[521]:


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Visualisation
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns


#Modeling
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier,  VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import  cross_val_score, KFold, learning_curve
from sklearn.model_selection import GridSearchCV

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading Dataset

# In[522]:


#get titanic data
train = pd.read_csv("../input/train.csv") #(891,12)
test = pd.read_csv("../input/test.csv") #(418, 11)

#combine train & test
total = train.append(test, ignore_index = True)
testID = test['PassengerId']

del train, test
train=total[:891]


# # 1. Data Exploration

# ## Info

# In[523]:


total.info() 


# In[524]:


total.head()


# In[525]:


total.describe(include='all')


# In[526]:


#unique value
print('col_name'.center(15),'count','value'.center(20))
for col in total.columns:
    length=len(total[col].dropna().unique())
    if length <=10:
        print('##',col.center(11),':' ,length,' ,',total[col].dropna().unique())
    else:
        print('##',col.center(11),':' ,length)


# ## Number of Null

# As you can see, there are missing values.<br>
# the order based on values having more missing values: Cabin > Age > Embarked > Fare <br>

# In[527]:


total.isnull().sum()


# # 1.0 Survived

# In[528]:


train.Survived.value_counts()/train.Survived.count()


# # 1.1 Age 

# Age distribution looks like normal distribution

# In[529]:


sns.distplot(total.Age.dropna())


# We wonder If Is there difference depending on age interval like 2,3,4..<br>
# We set interval from 2 to 10, and show y_axis: survival percentage ,x_axis: age.<br>
# Find out some age bands are difference.

# In[530]:


#Survival(%) by Age Interval
fig,ax=plt.subplots(3,3)
fig.subplots_adjust(hspace=0.8,wspace=0.4)
for interval in range(2,11):
    age_dict0={(i,i+interval):0 for i in range(0,int(total.Age.max()+interval),interval)}
    age_dict1={(i,i+interval):0 for i in range(0,int(total.Age.max()+interval),interval)}
    
    def survive_age1(age):
        global age_dict0; value=age//interval
        age_dict0[(interval*value,interval*(value+1))]+=1
                                        
    def survive_age2(age):
        global age_dict1; value=age//interval
        age_dict1[(interval*value,interval*(value+1))]+=1
                      
    total["Age"][(total["Survived"] == 0) & (total["Age"].notnull())].apply(survive_age1)
    total["Age"][(total["Survived"] == 1) & (total["Age"].notnull())].apply(survive_age2)
    age_list=[round(age_dict1[i]*100/(age_dict1[i]+age_dict0[i])) for i in age_dict1.keys() if age_dict0[i]+age_dict1[i]!=0]
    print('###interval=%d###'%(interval))
    a,b=divmod(interval-2,3)
    ax[a][b].plot(age_list,marker='.')
    ax[a][b].set_title("interval:{}".format(interval))
plt.xlabel("Age",x=-1,y=0)
plt.show()
    


# # 1.2 Sex

# In[531]:


sns.catplot('Sex',data=total,kind='count',size=6)


# In[532]:


sns.barplot(x="Sex", y="Survived", data=total)


# # 1.3 Embarked

# In[533]:


sns.catplot('Embarked',data=total,kind='count',size=6)


# In[534]:


sns.catplot(x='Embarked',y='Survived',data=total,kind='bar',size=6)


# In[535]:


total[["Embarked","Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# # 1.4 Pclass

# In[536]:


sns.catplot('Pclass',data=train,kind='count',size=6)


# Pclass '1' have higher survival than others.

# In[537]:


sns.catplot(x='Pclass',y='Survived',data=train,kind='bar',size=6)


# In[538]:


grid = sns.FacetGrid(total, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# # 1.5 Parch

# In[539]:


sns.catplot('Parch',data=total,kind='count',size=6)


# People who have 1,2,3 parents or children have high survival

# In[ ]:


sns.catplot(x='Parch',y='Survived',data=total,kind='bar',size=6)


# # 1.6 SibSp

# In[ ]:


sns.catplot('SibSp',data=total,kind='count',size=6)


# People who have 1,2 Siblings or spouse have high survival

# In[ ]:


sns.factorplot(x='SibSp',y='Survived',data=total,kind='bar',size=6)


# # 1.7 Fare

# Fare distribution has high kurtosis

# In[ ]:


sns.distplot(total.Fare.dropna()) 


# In[ ]:


sns.catplot(x='Survived',y='Fare',data=total,kind='box',size=6)


# In[ ]:


#Fare Distribution according to Survived
grid = sns.FacetGrid(total, col='Survived', height=3, aspect=1.6)
grid.map(plt.hist, 'Fare', alpha=.5, bins=20)
grid.add_legend();


# # 1.7 Relations among Variables

# There are some correlation between Pclass and Fare, SibSp and Parch .

# In[ ]:


sns.heatmap(total.corr(),annot=True)


# ## Embark + Age

# In[ ]:



sns.catplot(x='Embarked',y='Age',size=6,kind='box',data=total)
#Conclusion : no special differenciation


# ## Sex + Fare

# In[ ]:



sns.factorplot(x='Sex',y='Fare',size=6,kind='box',data=total)


# # 2. Feature Engineering

# # 2.1 Name
# 
# The reason that we extract the title is for filling missing Age and it will be used for value when we process modeling.<br>
# Name is consist of Title,first name,second name. We classify title of name into 5 titles like Mr, Mrs, Miss, Master, Rare.<br>
# Since the others are < 10 each of them so that we decide it would be better to combine all to Rare category.<br>

# In[ ]:


#Age Distirbution according to Title
total['Title'] = total.Name.str.extract('([A-Za-z]+)\.', expand=True)
print(list(total.Title.unique()))
print(total.Title.value_counts())


# In[ ]:


total['Title'] = total['Title'].replace('Mlle', 'Miss')
total['Title'] = total['Title'].replace(['Capt', 'Col','Countess',
    'Don','Dona', 'Dr', 'Major','Mme','Ms','Lady','Sir', 'Rev', 'Jonkheer' ],'Rare')
print(total.Title.value_counts())


# # 2.2 Parch+SibSp->FamilySize

# We suppose Family size is important for  survival.As a result, 1 ~4 size of family survived more than others.

# In[ ]:


total['FamilySize'] = total['SibSp'] + total['Parch'] + 1


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(15,4))
sns.barplot('FamilySize','Survived',data=total,ax=ax[0])
ax[0].set_title('FamilySize vs Survived')


# In[ ]:


total.loc[total['FamilySize'] == 1, 'FamilySize'] = 0
total.loc[(total['FamilySize'] > 1) & (total['FamilySize'] <= 4), 'FamilySize'] = 1
total.loc[(total['FamilySize'] > 4), 'FamilySize']   = 2


# In[ ]:


total[['FamilySize', 'Survived']].groupby(['FamilySize']).mean().sort_values(by='Survived',ascending=False)


# In[ ]:


sns.heatmap(total.corr(),annot=True)


# In[ ]:


#Parchm SibSp del
total = total.drop(['Parch','SibSp'], axis=1)


# # 3 Filling Missing Values
# 
# These are missing values given below<br>
# 
# Fare: 1 <br>
# Embarked: 2 <br>
# Age: 263 <br>
# Cabin: 1114 <br>
# 

# # 3.1 Fare

# Fill median without outliers, Because Fare distribution has high kurtosis

# In[ ]:


total[total.Fare.isnull()]


# In[ ]:


sns.distplot(total.Fare[(total.Pclass==3) & (total.Fare.notnull())])


# In[ ]:


total['Fare'] = total.Fare.fillna(total.Fare.median())


# # 3.2 Embarked

# Two women has 80 of Fare. So we look Embarked and Fare distribution.<br>
# Among 3 ports , C port has fare range including 80. SO we decide those women's port is 'C'

# In[ ]:


total[total.Embarked.isnull()]


# In[ ]:


sns.catplot(x='Embarked',y='Fare',size=6,kind='box',data=total)


# In[ ]:


total['Embarked'] = total.Embarked.fillna('C')


# # 3.3 Age

# There is difference of age distribution depending on title.<br>
# So we fill age missing value according to mean age of title.

# In[ ]:


TotalAge = total[total.Age.isnull()==False]
grid = sns.FacetGrid(TotalAge, col="Title", hue="Title",col_wrap=4)
grid.map(sns.distplot, "Age")


# In[ ]:


total[['Title', 'Age']].groupby(['Title']).median().sort_values(by='Title',ascending=False)


# In[ ]:


total['Age']=total.groupby('Title').transform(lambda x:x.fillna(x.median()))


# # 4 Data Categorization & Dummy

# We dicide to use Age, Embarked, Fare, Plcass, Sex and FamilySize for feature engineering.<br>
# The reason of dropping the others:<br>
# <pre>
# Cabin: Most of Cabins are missing
# Name: We already extracted title
# Ticket: We weren't able to find out encrypted meaning inside.
# </pre>

# In[ ]:


total = total.drop(['Cabin','Name','Ticket'], axis=1)


# In[ ]:


total.head()


# # 4.1 Sex

# We classify Female to 1 and Male to 0

# In[ ]:


total['Sex'] = total['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
total.head()


# # 4.2 Embarked

# We use get_dummies in Pandas for processing Embarked

# In[ ]:


total["Embarked"] = total["Embarked"].astype("category")
total = pd.get_dummies(total, columns = ["Embarked"],prefix="Embarked")
total.head()


# # 4.3 Pclass

# In[ ]:


total["Pclass"] = total["Pclass"].astype("category")
total = pd.get_dummies(total, columns = ["Pclass"],prefix="Pclass")
total.head()


# # 4.4 Title

# In[ ]:


total["Title"] = total["Title"].astype("category")
total = pd.get_dummies(total, columns = ["Title"],prefix="Title")


# # 4.5 FamilySize

# We already categorized FamilySize, We renamed them as Family_Single,Small and Large

# In[ ]:


total[ 'Family_Single' ] = total[ 'FamilySize' ].map( lambda s : 1 if s == 0 else 0 )
total[ 'Family_Small' ]  = total[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
total[ 'Family_Large' ]  = total[ 'FamilySize' ].map( lambda s : 1 if s == 2 else 0 )

total = total.drop(['FamilySize'], axis=1)
total.head()


# # 5 Modeling

# 
# Almost done!  We've just finised feature engineering, now we need to adjust several models.<br>
# <pre>
# train: train set
# y_train: only Survived values of train set
# x_train: all values except for Survived
# test: test set
# </pre>
# 
# We split the train set into train and test again for testing

# In[ ]:


train = total[ :891] ;y_train=train['Survived'];x_train=train.drop('Survived',1)
test=total[891:];test=test.drop('Survived',1)
#testID = test['PassengerId']
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=0)
del total,train


# # 5.1 Cross-Validation

# In[ ]:


random_state = 0
kfold = KFold(n_splits=8, shuffle=True, random_state=random_state)

classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))


cv_means = [];cv_stds= []
for classifier in classifiers :
    result=cross_val_score(classifier, x_train, y = y_train, scoring = "accuracy", cv = kfold)
    cv_means.append(result.mean());cv_stds.append(result.std())
    
cv_df= pd.DataFrame({"Means":cv_means,"Stds": cv_stds,"Algorithm":["SVC","DecisionTree","RandomForest","KNeighboors","LogisticRegression"]})

g = sns.barplot("Means","Algorithm",data = cv_df,orient = "h",**{'xerr':cv_stds})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# # 5.2 Tunning Parameters

# In[ ]:


#SVM
param_grid_svm = [{'kernel': ['rbf'],
'C': [0.001, 0.01, 0.1, 1, 10, 100],
'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
{'kernel': ['linear'],
'C': [0.001, 0.01, 0.1, 1, 10, 100]}]

grid_search = GridSearchCV(SVC(probability=True), param_grid_svm, cv=5,n_jobs=-1)
grid_search.fit(x_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

model_svm = grid_search.best_estimator_


# In[ ]:


#RamdomForest
param_grid_rf = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)
grid_search.fit(x_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

model_rf = grid_search.best_estimator_


# # 5.3 Learning Curve

# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g = plot_learning_curve(model_svm,"SVM learning curves",x_train,y_train,cv=kfold)
g = plot_learning_curve(model_rf,"RF learning curves",x_train,y_train,cv=kfold)


# # 5.4 Model Feature Importance

# Since RandomForest is tree-based classifer, We can check the feature importance as below

# In[ ]:


classifier = model_rf
indices = np.argsort(classifier.feature_importances_)[::-1][:40]
g = sns.barplot(y=x_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40])
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title('RandomForest' + " feature importance")


# # 5.5 Ensemble Modeling

# We dicided to use SVC and RandomForest for Ensemble Modeling with VotingClassifier.

# In[ ]:


votingC = VotingClassifier(estimators=[('rfc', model_rf),('svc', model_svm)], voting='soft', n_jobs=4)

votingC = votingC.fit(x_train, y_train)


# # 6. Prediction

# 

# In[ ]:


test_Survived = pd.Series(votingC.predict(test), name="Survived").astype(int)

results = pd.concat([testID, test_Survived], axis=1)
results.to_csv("ensemble_python_voting.csv",index=False)
results.head()


# In[ ]:




