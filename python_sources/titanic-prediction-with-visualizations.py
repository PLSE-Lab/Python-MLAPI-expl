#!/usr/bin/env python
# coding: utf-8

# #                          Titanic Tragedy Survivors/Deaths Prediction

# RMS Titanic was a British passenger liner operated by the White Star Line that sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after striking an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making the sinking one of modern history's deadliest peacetime commercial marine disasters. RMS Titanic was the largest ship afloat at the time she entered service and was the second of three Olympic-class ocean liners operated by the White Star Line. She was built by the Harland and Wolff shipyard in Belfast. Thomas Andrews, chief naval architect of the shipyard at the time, died in the disaster.
#                                                                                                  The wreck of Titanic was discovered in 1985 (more than 70 years after the disaster) during a Franco-American expedition and US military mission. The ship was split in two and is gradually disintegrating at a depth of 12,415 feet (2,069.2 fathoms; 3,784 m). Thousands of artefacts have been recovered and displayed at museums around the world. Titanic has become one of the most famous ships in history, depicted in numerous works of popular culture, including books, folk songs, films, exhibits, and memorials. Titanic is the second largest ocean liner wreck in the world, only being surpassed by her sister ship HMHS Britannic, however, she is the largest sunk while in service as a liner, as Britannic was in use as a hospital ship at the time of her sinking. The final survivor of the sinking, Millvina Dean, aged two months at the time, died in 2009 at the age of 97.
#                                                                                                   Let's do some visualisation and predict who have survived the tragedy. Here we are going to do this project in a clear linear way so that everyone could understand the process being undertaken.Here it goes.                                                                                         

# # 1.Importing Library And Dataset
# Let's import the necessary library and the data set

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from collections import Counter
from sklearn.preprocessing import OrdinalEncoder
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_curve
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_data=pd.read_csv('../input/titanic/train.csv')
test_data=pd.read_csv('../input/titanic/test.csv')


# Let us check the dataset for any missing values and also analyse if we have to do any data cleaning process for taking it to prediction

# In[ ]:


train_data.head(10)


# In[ ]:


train_data.describe()


# In[ ]:


train_data.isna().any()


# And yes ofcourse there are missing values and we need a lot of data cleaning to be done to get a good visualization and prediction.

# # 2.Data Preprocessing And Cleaning

# From the data insight we could  see that PassengerId is same as the index(not exactly but it just shows the index number of passengers).So we can remove the PassengerId column from our dataset.
# Next comming to the ticket column we don't see anything unique that contributes for the survival of passengers. So we can delete that column from our dataset.
# Cabin column have lot of missing data and also doesn't contribute to survival. So we can delete the Cabin column from our data

# In[ ]:


train_data.drop(['PassengerId','Ticket','Cabin'],axis=1,inplace=True)


# In[ ]:


train_data['Embarked'].isna().value_counts()


# In[ ]:


train_data.isna().any()


# Still we can see missing data in columns like Age and places where people embarked. We can fill the age of the missing people with the help of mean of remaining people's age and fill the embarked people column with the most frequent embarked place.

# In[ ]:


imp=SimpleImputer(strategy='most_frequent')
train_data[['Embarked']]=imp.fit_transform(train_data[['Embarked']])


# In[ ]:


mean=train_data['Age'].mean()
train_data['Age'].fillna(mean,inplace=True)


# With all the missing data filled we shall now change the varying age of people to a particular category. This we do by categorise the age from 0 to 12 as Child, 13 to 20 as Teen, 21 to 30 as Young, 31 to 40 as 30_adult, 41 to 50 as 40_adult and more than 51 as Elderly. By doing this we could get a clear insight of how many children and adults were saved. First lets round off the age.

# In[ ]:


train_data['Age']=np.ceil(train_data['Age'])


# In[ ]:


def get_age(val):
    age_classification={'Child':range(0,13),
                        'Teen':range(13,21),
                        'Young':range(21,31),
                        '30_adult':range(31,41),
                        '40_adult':range(41,51),
                        'Elderly':range(51,90)}
    for key,value in age_classification.items():
        if val in value:
            return key


# In[ ]:


for i in range(len(train_data)):
    train_data['Age'][i]=('{}'.format(get_age(train_data['Age'][i])))    


# With now the Age column changed we shall see the names so that we could get anything common in names.

# In[ ]:


train_data['Name'].unique()


# We could see that the titles stand common amidst the name. So we can use it for prediction.

# In[ ]:


def get_title(val):
    words=val.split()
    title={'Officer':['Capt.','Col.','Major.','Dr.','Rev.'],
          'Royalty':['Jonkheer.','Don.','Sir.','the Countess.','Lady.'],
          'Mrs':['Mme.','Ms.','Mrs.'],
          'Mr':['Mr.'],
          'Miss':['Mlle.','Miss.'],
          'Master':['Master.']}
    for key,value in title.items():
        for word in words:
            if word in value:
                return str(key)


# In[ ]:


train_data['Title']=np.NAN
for i in range(len(train_data)):
    train_data['Title'][i]=get_title(train_data['Name'][i])


# In[ ]:


train_data['Title'].isna().value_counts()


# We have got a new column called Title that display the title of the passengers. We could see that there is a miising value which means that the title was not given to that particular member. So lets deal with it later.
#                                                                                                 Now lets us get some information from SibSp and Parch. These are nothing but the number of siblings and parents that particular passenger have in the board. This is also important because those who are having small family managed to escape which gives us a good correlation for our prediction. Lets convert it into family member(Fam_mem) which tells the total number of family members on the board. Then we will convert the numbers into three columns stating the single passengers as singleton, 2 to 4 members as small family, more than 4 as large_family 

# In[ ]:


train_data['Fam_mem']=train_data['SibSp']+train_data['Parch']+1


# In[ ]:


def fam_size(val):
    fam={'Single':[1],
        'Small_family':[2,3,4],
        'large_family':[5,6,7,8,9,10,11]}
    for key,value in fam.items():
        if val in value:
            return key


# In[ ]:


for i in range(len(train_data)):
    train_data['Fam_mem'][i]=fam_size(train_data['Fam_mem'][i])


# Now we can see how the data transformation made changes

# In[ ]:


train_data.head(10)


# In[ ]:


train_data.isna().sum()


# And still we have the missing value in title column as we mentioned earlier. We can replace the missing title with the most frequent title in the Title column.

# In[ ]:


def most_common(lst):
    data=Counter(lst)
    return data.most_common(1)[0][0]
frequent=most_common(train_data['Title'])


# In[ ]:


train_data['Title'].fillna(frequent,inplace=True)


# In[ ]:


train_data['Title'].unique()


# In[ ]:


train_data.head(10)


# Now we done with cleaning work let's convert the categorical columns like Sex,Age,Embarked,Title,Fam_mem to Ordinal Values which will be very usefull for going ahead with visualization.

# In[ ]:


oe=OrdinalEncoder()
train_data[['Embarked','Sex','Age','Title','Fam_mem']]=oe.fit_transform(train_data[['Embarked','Sex','Age','Title','Fam_mem']])


# In[ ]:


train_data.head(10)


# Done with the all the preprocessing and cleaning part let's visualize it to get some insights.

# # 3.Visualizing

# Let us first get the histogram of the data.

# In[ ]:


train_data.hist(bins=10,figsize=(20,15))


# Let us check the correlation of each numerical data against the Survived column to see how far each feature contribute to our prediction.

# In[ ]:


corr_mat=train_data.corr()
corr_mat['Survived'].sort_values(ascending=False)


# Seems that the feature Sex contribute a lot than others in negative way. This shows that greater the value lesser the survival rate. In Sex feature male is given as 1 and female is given as 0. When the value of feature 'Sex' is '1' the possibility of survival is '0'(i.e If it is a 'male' passenger the probability that he survived is 'low'). Fare feature has given positive correlation. Let us visualize the important correlators in graphical form.

# In[ ]:


attributes=['Survived','Fare','Embarked','Pclass','Sex']
scatter_matrix(train_data[attributes],figsize=(15,10),alpha=0.1)


# In[ ]:


plt.scatter(train_data.iloc[:,1],train_data.iloc[:,3],c=train_data.iloc[:,0],s=50,cmap='RdBu')


# Since it is a classification typed problem we couldn't get a good insight. But we could make a better visualization with bar and pie charts.

# Let's visualise the ratio of survivability of men to women.

# In[ ]:


men_survived_truth=(((train_data['Sex']==1)&(train_data['Survived']==1)))
men_death_truth=(((train_data['Sex']==1)&(train_data['Survived']==0)))
women_survived_truth=(((train_data['Sex']==0)&(train_data['Survived']==1)))
women_death_truth=(((train_data['Sex']==0)&(train_data['Survived']==0)))
men_survived=men_survived_truth.value_counts()
men_death=men_death_truth.value_counts()
women_survived=women_survived_truth.value_counts()
women_death=women_death_truth.value_counts()
men=[men_survived[1],men_death[1]]
women=[women_survived[1],women_death[1]]


# In[ ]:


men_ratio=[(men[0]/(men[0]+men[1]))*100,(men[1]/(men[0]+men[1]))*100]
print(men_ratio)
women_ratio=[(women[0]/(women[0]+women[1]))*100,(women[1]/(women[0]+women[1]))*100]
print(women_ratio)


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,10))
explode=(0.1,0)
ax[0].pie(men_ratio,explode=explode,labels=['Survivors','Deaths'],autopct='%1.2f%%',shadow=True,startangle=90)
ax[0].set_title('Men Ratio')
ax[1].pie(women_ratio,explode=explode,labels=['Survivors','Deaths'],autopct='%1.2f%%',shadow=True,startangle=90)
ax[1].set_title('Women Ratio')


# It is clear that lot of men lost their life in this tragedy and lot of women suvived this tragedy.

# Let us visualize how many men and women in each class managed to survive.

# In[ ]:


men_survivors=[]
men_death=[]
for i in range(1,4):
    Pclassmen_survived_truth=(((train_data['Sex']==1)&(train_data['Survived']==1)&(train_data['Pclass']==i)))
    Pclassmen_death_truth=(((train_data['Sex']==1)&(train_data['Survived']==0)&(train_data['Pclass']==i)))
    pclassmen_survivors=Pclassmen_survived_truth.value_counts()
    pclassmen_deaths=Pclassmen_death_truth.value_counts()
    men_survivors.append(pclassmen_survivors[1])
    men_death.append(pclassmen_deaths[1])
men=[men_survivors,men_death]


# In[ ]:


women_survivors=[]
women_death=[]
for i in range(1,4):
    Pclasswomen_survived_truth=(((train_data['Sex']==0)&(train_data['Survived']==1)&(train_data['Pclass']==i)))
    Pclasswomen_death_truth=(((train_data['Sex']==0)&(train_data['Survived']==0)&(train_data['Pclass']==i)))
    pclasswomen_survivors=Pclasswomen_survived_truth.value_counts()
    pclasswomen_deaths=Pclasswomen_death_truth.value_counts()
    women_survivors.append(pclasswomen_survivors[1])
    women_death.append(pclasswomen_deaths[1])
women=[women_survivors,women_death]


# In[ ]:


print(men)
print(women)


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,10))
sur_death=['Survived','Death']
width=0.5
for i,axi in enumerate(ax.flat):
    N=3
    ind=[x for x in np.arange(1,N+1)]
    axi.bar(ind,women[i],width,label='Women',bottom=men[i],color='Pink')
    axi.bar(ind,men[i],width,label='Men',color='Blue')
    axi.set_xticklabels(['0','Pclass 1','','Pclass 2','','Pclass 3'])
    axi.set_title(sur_death[i])
    axi.legend()


# It's clearly evident that almost all women from Pclass1 and Pclass2 managed to survive.

# In[ ]:


sns.pairplot(train_data)


# Pair plot doesn't give a clear intution. Let's predict the survivability with respect to titles.

# In[ ]:


survivors=[]
death=[]
for i in range(0,6):
    title_survived_truth=(((train_data['Survived']==1)&(train_data['Title']==i)))
    title_death_truth=(((train_data['Survived']==0)&(train_data['Title']==i)))
    title_survived=title_survived_truth.value_counts()
    title_death=title_death_truth.value_counts()
    survivors.append(title_survived[1])
    death.append(title_death[1])
title_sur_death=[survivors,death]
title_sur_death


# In[ ]:


oe.categories_


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,10))
to_plot=[survivors,death]
name=['Survivors','Deaths']
color=['green','red']
for i,axi in enumerate(ax.flat):
    N=6
    ind=[x for x in np.arange(1,N+1)]
    axi.bar(ind,to_plot[i],width,label=('{}'.format(name[i])),color=color[i])
    axi.set_xticklabels(['0','Master','Miss','Mr','Mrs','Officer','Royalty'])
    axi.set_title('{} on basis of Title'.format(name[i]))
    axi.legend()


# It seems Mr are very unfortunate.

# Let us visualize the survivability based on Age.

# In[ ]:


survivors=[]
death=[]
for i in range(0,6):
    age_survived_truth=(((train_data['Survived']==1)&(train_data['Age']==i)))
    age_death_truth=(((train_data['Survived']==0)&(train_data['Age']==i)))
    age_survived=age_survived_truth.value_counts()
    age_death=age_death_truth.value_counts()
    survivors.append(age_survived[1])
    death.append(age_death[1])
age_sur_death=[survivors,death]
age_sur_death


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,10))
to_plot=[survivors,death]
name=['Survivors','Deaths']
color=['green','red']
for i,axi in enumerate(ax.flat):
    N=6
    ind=[x for x in np.arange(1,N+1)]
    axi.bar(ind,to_plot[i],width,label=('{}'.format(name[i])),color=color[i])
    axi.set_xticklabels(['0','30_adult','40_adult','Child','Elderly','Teen','Young'])
    axi.set_title('{} on basis of Age'.format(name[i]))
    axi.legend()


# We may have heard that most children were saved but the data shows that about 30 children died in this tragedy. So sad....

# Let's visualize on survivability basis of embarked places

# In[ ]:


embarked_survivor=[]
embarked_death=[]
for i in range(3):
    embarked_survived_truth=(((train_data['Survived']==1)&(train_data['Embarked']==i)))
    embarked_death_truth=(((train_data['Survived']==0)&(train_data['Embarked']==i)))
    embarked_survivors=embarked_survived_truth.value_counts()
    embarked_deaths=embarked_death_truth.value_counts()
    embarked_survivor.append(embarked_survivors[1])
    embarked_death.append(embarked_deaths[1])
embarked=[embarked_survivor,embarked_death]
embarked


# In[ ]:


fig=plt.figure(figsize=(15,10))
N=3
ind=[x for x in np.arange(1,N+1)]
plt.bar(ind,embarked_survivor,width,label='Survived',bottom=embarked_death,color='Orange')
plt.bar(ind,embarked_death,width,label='Death',color='cyan')
plt.xticks(ind,['C','Q','S'])
plt.title('Survive/death on basis of Embarked')
plt.legend()


# Seems that people who embarked from Queenstown are unfortunate.

# Let us visualize the ratio of survivability based on the family size.

# In[ ]:


survivors=[]
death=[]
for i in range(0,3):
    fam_survived_truth=(((train_data['Survived']==1)&(train_data['Fam_mem']==i)))
    fam_death_truth=(((train_data['Survived']==0)&(train_data['Fam_mem']==i)))
    fam_survived=fam_survived_truth.value_counts()
    fam_death=fam_death_truth.value_counts()
    survivors.append(fam_survived[1])
    death.append(fam_death[1])
fam_sur_death=[survivors,death]
fam_sur_death


# In[ ]:


sur_ratio=[(survivors[0]/(survivors[0]+survivors[1]+survivors[2]))*100,(survivors[1]/(survivors[0]+survivors[1]+survivors[2]))*100,
          (survivors[2]/(survivors[0]+survivors[1]+survivors[2]))*100]
death_ratio=[(death[0]/(death[0]+death[1]+death[2]))*100,(death[1]/(death[0]+death[1]+death[2]))*100,
            (death[2]/(death[0]+death[1]+death[2]))*100]
print(sur_ratio)
print(death_ratio)


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,10))
explode=(0.1,0.1,0.1)
ax[0].pie(sur_ratio,explode=explode,labels=['Single','Small_family','Large_family'],autopct='%1.2f%%',shadow=True,startangle=90)
ax[0].set_title('Survive Ratio')
ax[1].pie(death_ratio,explode=explode,labels=['Single','Small_family','Large_family'],autopct='%1.2f%%',shadow=True,startangle=90)
ax[1].set_title('Death Ratio')


# Seems that many small member family made their way out,but large family members could not make their way out

# # 4.Preprocessing Part-2.0

# Now that with the data getting ready for prediction we could make some more process so that our classifiers performs well on the dataset. The process includes getting dummies for features like Pclass,Sex,Age,Embarked,Title,Fam_mem. This is done because from our above visualization we could see some values in a feature contibute a lot to survivability. Eg: We can definitly say that a women from Pclass '2' has survivability of 95%. So this gives a good correlation than taking the overall feature of Pclass against survivability. 

# In[ ]:


train_data.head(10)


# In[ ]:


preprocess_data=train_data.copy()


# In[ ]:


preprocess_data=pd.get_dummies(preprocess_data,columns=['Pclass','Sex','Age','Embarked','Title','Fam_mem'])


# In[ ]:


preprocess_data.head(10)


# In[ ]:


corr_mat=preprocess_data.corr()
corr_mat['Survived'].sort_values(ascending=False)


# As I mentioned we can see a lot of good positive and negative correlations are obtained. With the help of this our classifier does good job in predictiong the value.
# Eventhough there are good correlators some doesn't have good correlation. Let's remove those as it does not going to affect our predictions. 

# In[ ]:


preprocess_data.drop(['Age_0.0','Title_5.0','Embarked_1.0','Age_1.0','Age_4.0','Age_3.0','Title_4.0','SibSp','Age_5.0',
                     'Parch','Title_0.0','Pclass_2','Name'],axis=1,inplace=True)


# In[ ]:


preprocess_data.shape


# After removing the unneccasary feature let's scale Fare feature as it has range of higher values.

# In[ ]:


scale=StandardScaler()
preprocess_data[['Fare']]=scale.fit_transform(preprocess_data[['Fare']])


# In[ ]:


preprocess_data.head()


# Now let's seperate the data for training

# In[ ]:


preprocess_feature=preprocess_data.drop('Survived',axis=1)
preprocess_label=preprocess_data['Survived']


# In[ ]:


preprocess_feature


# In[ ]:


preprocess_label


# # 5.Selecting the Classifier
# Don't hurry!!!! For prediction we need a best classifier and good hyperparameters tuned for that particular classifier.
# With the training dataset we are going to split it into train set and test set. The split will be of Stratified Sampling in which the test data will be taken in the correct ratio. With the help of this training and test set we are going to select the best classifier. We will also do grid search to get the fine hyperparameters to do the prediction perfectly.

# In[ ]:


split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(preprocess_data,preprocess_data['Survived']):
    strat_preprocess_data=preprocess_data.loc[train_index]
    strat_preprocess_label=preprocess_data.loc[test_index]


# In[ ]:


X_train=strat_preprocess_data.drop('Survived',axis=1)
y_train=strat_preprocess_data['Survived']
X_test=strat_preprocess_label.drop('Survived',axis=1)
y_test=strat_preprocess_label['Survived']


# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# First lets go ahead with DECISION TREE......

# In[ ]:


tree=DecisionTreeClassifier(random_state=42,criterion='entropy',splitter='best')
tree.fit(X_train,y_train)
y_pred=tree.predict(X_test)
accuracy_score(y_test,y_pred)


# It gives us accuracy of 79%. Lets fine tune hyperparameters.

# In[ ]:


param_grid={'max_depth':[2,3,4,5,6]}
grid_tree=GridSearchCV(tree,param_grid,cv=5)
grid_tree.fit(X_train,y_train)
y_pred=grid_tree.best_estimator_.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(grid_tree.best_params_)


# It gives us 80% accuracy. That's ok. Let us visualize in the form of heat map.

# In[ ]:


mat=confusion_matrix(y_test,y_pred)
sns.heatmap(mat.T,square=True,annot=True,cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')


# The classifier did its best. Lets try the same procedure for all other classifiers.

# Now LOGISTIC REGRESSION......

# In[ ]:


log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
y_pred=log_reg.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:


param_grid={'C':[0.0001,0.001,0.01,0.1,0.5],'penalty':['l1','l2']}
grid_log=GridSearchCV(log_reg,param_grid,cv=5)
grid_log.fit(X_train,y_train)
y_pred=grid_log.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(grid_log.best_params_)


# In[ ]:


mat=confusion_matrix(y_test,y_pred)
sns.heatmap(mat.T,square=True,annot=True,cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')


# This performs even more better than Decision Tree.

# Let's see RANDOM FOREST CLASSIFIER......

# In[ ]:


rfc=RandomForestClassifier(criterion='entropy',random_state=42)
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:


param_grid={'max_depth':[2,3,4,5,6],'n_estimators':[100,200,300,400,500]}
grid_forest=GridSearchCV(rfc,param_grid,cv=5)
grid_forest.fit(X_train,y_train)
y_pred=grid_forest.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(grid_forest.best_params_)


# In[ ]:


mat=confusion_matrix(y_test,y_pred)
sns.heatmap(mat.T,square=True,annot=True,cbar=False,fmt='d')
plt.xlabel('true label')
plt.ylabel('predicted label')


# With the fine tuned hyperparameters random forest does equal to logistic regression

# Let us try SGD Classifier......

# In[ ]:


sgd_clf=SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train)
y_pred=sgd_clf.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:


param_grid={'penalty':['l1','l2'],'alpha':[0.0001,0.001,0.01,0.1],'max_iter':[1000,1500,2000,2500]}
grid_sgd=GridSearchCV(sgd_clf,param_grid,cv=5)
grid_sgd.fit(X_train,y_train)
y_pred=grid_sgd.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(grid_sgd.best_params_)


# In[ ]:


mat=confusion_matrix(y_test,y_pred)
sns.heatmap(mat.T,square=True,annot=True,cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')


# This also does the same as logistic regression and random forest.

# Let us try SUPPORT VECTOR MACHINE........

# In[ ]:


svc_clf=SVC(kernel='rbf',random_state=42)
svc_clf.fit(X_train,y_train)
y_pred=svc_clf.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:


param_grid={'C':[1,2,3,4,5],'degree':[2,3,4,5,6],'gamma':['scale','auto']}
grid_svc=GridSearchCV(svc_clf,param_grid,cv=5)
grid_svc.fit(X_train,y_train)
y_pred=grid_svc.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(grid_svc.best_params_)


# In[ ]:


mat=confusion_matrix(y_test,y_pred)
sns.heatmap(mat.T,square=True,annot=True,cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')


# Above all this gives better results.

# Lets try GAUSSIAN NAIVE BAYES......

# In[ ]:


gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:


mat=confusion_matrix(y_test,y_pred)
sns.heatmap(mat.T,square=True,annot=True,cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')


# Gaussian naive bayes also does same prediction as random forest.

# Let us see KNEIGHBORS......

# In[ ]:


knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:


param_grid={'n_neighbors':[4,5,6,7,8],'algorithm':['ball_tree','kd_tree','brute'],'p':[1,2]}
grid_knn=GridSearchCV(knn,param_grid,cv=5)
grid_knn.fit(X_train,y_train)
y_pred=grid_knn.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(grid_knn.best_params_)


# In[ ]:


mat=confusion_matrix(y_test,y_pred)
sns.heatmap(mat.T,square=True,annot=True,cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')


# It does poor compared to all above done classifier.

# Overall we come to the conclusion that SUPPORT VECTOR MACHINE performs well on the training data with fine tuned hyperparameters.

# # 6.Precision,Recall,F1 score
# Lets calculate the precision,recall,f1 score of our chosen classifier on the performed data

# In[ ]:


def display_scores(scores):
    print('Scores:',scores)
    print('Mean:',scores.mean())    
    print('Standard Deviation:',scores.std()) 


# In[ ]:


svc_clf=SVC(kernel='rbf',random_state=42,C=2,degree=2,gamma='auto')
scores=cross_val_score(svc_clf,X_train,y_train,cv=3,scoring='accuracy')
display_scores(scores)


# We can see that how our classifier have performed in cross validation .
# Let us see precision,recall,f1 score.

# In[ ]:


y_scores=cross_val_predict(svc_clf,X_train,y_train,method='decision_function')


# In[ ]:


print(precision_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(f1_score(y_test,y_pred))


# We got about 80% precision and only 59% recall. Precision here means among the classifier predicted the survivors it tend to correctly predict 80% of them as survived and also predicted 20% as suvived but they are dead. Recall means the actula ratio of how much the classifier must predict that the passenger survived.(i.e here the classifier correctly predicted only 59% of total survivors).
# Let us see what will be the recall score if the threshold is increased so that the precision is 90%.

# In[ ]:


precisions,recalls,thresholds=precision_recall_curve(y_train,y_scores)


# In[ ]:


threshold_90_precision=thresholds[np.argmax(precisions>=0.90)]


# In[ ]:


y_train_pred_90=(y_scores>=threshold_90_precision)


# In[ ]:


precision_score(y_train,y_train_pred_90)


# In[ ]:


recall_score(y_train,y_train_pred_90)


# In[ ]:


threshold_90_precision


# Let's cross validate and see the accuracy for the overall data we made in the preprocessing part-2.0

# In[ ]:


svc_clf=SVC(kernel='rbf',random_state=42,C=2,degree=2,gamma='auto')
scores=cross_val_score(svc_clf,preprocess_feature,preprocess_label,cv=3,scoring='accuracy')
display_scores(scores)


# Good prediction in cross validation.

# # 7.Predicting the test data
# Before doing prediction we need to preprocess the data as we did for training data. 

# In[ ]:


test_data.head()


# In[ ]:


test_data.describe()


# In[ ]:


test_data.isna().any()


# The preprocess can be done simply by defining a class with fit,transform and fit_transform method. Transform the data by passing it to the class.

# In[ ]:


def get_age(val):
    age_classification={'Child':range(0,13),
                        'Teen':range(13,21),
                        'Young':range(21,31),
                        '30_adult':range(31,41),
                        '40_adult':range(41,51),
                        'Elderly':range(51,90)}
    for key,value in age_classification.items():
        if val in value:
            return key
def get_title(val):
    words=val.split()
    title={'Officer':['Capt.','Col.','Major.','Dr.','Rev.'],
          'Royalty':['Jonkheer.','Don.','Sir.','the Countess.','Lady.'],
          'Mrs':['Mme.','Ms.','Mrs.'],
          'Mr':['Mr.'],
          'Miss':['Mlle.','Miss.'],
          'Master':['Master.']}
    for key,value in title.items():
        for word in words:
            if word in value:
                return str(key)
def fam_size(val):
    fam={'Single':[1],
        'Small_family':[2,3,4],
        'large_family':[5,6,7,8,9,10,11]}
    for key,value in fam.items():
        if val in value:
            return key
def most_common(lst):
    data=Counter(lst)
    return data.most_common(1)[0][0]


# In[ ]:


from sklearn.base import BaseEstimator,TransformerMixin
class CombinedWorks(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.preprocessing import StandardScaler
        oe=OrdinalEncoder()
        scale=StandardScaler()
        mean_age=X['Age'].mean()
        mean_fare=X['Fare'].mean()
        X['Age'].fillna(mean_age,inplace=True)
        X['Fare'].fillna(mean_fare,inplace=True)
        PassengerId=X['PassengerId']
        X.drop(['PassengerId','Cabin','Ticket'],axis=1,inplace=True)
        X['Title']=np.NAN
        X['Age']=np.ceil(X['Age'])
        X['Fam_mem']=X['SibSp']+X['Parch']+1
        for i in range(len(X)):
            X['Age'][i]=('{}'.format(get_age(X['Age'][i])))  
            X['Title'][i]=get_title(X['Name'][i])
            X['Fam_mem'][i]=fam_size(X['Fam_mem'][i])
        frequent=most_common(X['Title'])
        X['Title'].fillna(frequent,inplace=True)
        X[['Embarked','Sex','Age','Title','Fam_mem']]=oe.fit_transform(X[['Embarked','Sex','Age','Title','Fam_mem']])
        X=pd.get_dummies(X,columns=['Pclass','Sex','Age','Embarked','Title','Fam_mem'])
        X[['Fare']]=scale.fit_transform(X[['Fare']])
        name=X['Name']
        X.drop(['SibSp','Parch','Name'],axis=1,inplace=True)
        return X,name,oe.categories_,PassengerId


# In[ ]:


cw=CombinedWorks()
test_data,Name,catagories,passenger_id=cw.fit_transform(test_data)


# In[ ]:


test_data.head()


# Now we can see that the test data has been modified and lets see if there are all the feature value that were in train data.

# In[ ]:


catagories


# We can see that the test data has no member related to the title 'Royalty'. Now lets remove the uncorrelated data as in the training data

# In[ ]:


test_data.drop(['Age_0.0','Embarked_1.0','Age_1.0','Age_4.0','Age_3.0','Title_4.0','Age_5.0',
                'Title_0.0','Pclass_2'],axis=1,inplace=True)


# In[ ]:


test_data.shape


# Let's fit the training data to the fine tuned classifier and predict the result

# In[ ]:


svc_clf=SVC(kernel='rbf',random_state=42,C=2,degree=2,gamma='auto')
svc_clf.fit(preprocess_feature,preprocess_label)


# In[ ]:


y_pred=svc_clf.predict(test_data)


# In[ ]:


y_pred


# In[ ]:


prediction=np.c_[Name,y_pred]


# In[ ]:


for i in range(len(test_data)):
    if prediction[i][1]==1:
        prediction[i][1]='Yes'
    else:
        prediction[i][1]='No'


# In[ ]:


for i in range(len(test_data)):
    print('Name:{0}\t\tSurvived:{1}'.format(prediction[i][0],prediction[i][1]))


# And now we got the predicted values of those who were in the board. Atlast we did it. We made a lot way gaining a lot of intution about the data and finally making a good prediction. Thank you for travelling with me. :-).If you like it please upvote it.
