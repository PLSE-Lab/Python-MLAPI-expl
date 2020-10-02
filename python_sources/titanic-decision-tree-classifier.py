
# coding: utf-8

# **Titanic: Data Analysis using Decision Tree classifier**
# 
# **1. Introduction:**
# The data sets used in this analysis are composed of 13 features and 891 instances for training, and 418 instances for testing. The features such as survival, ticket class, sex, age in years, the number of siblings/spouse, the number of parents/children, ticket number, passenger fare, cabin number, and Port of Embarkation are included in the data set.
# 
# This notebook is designed to analyze what sorts of people were likely to survive and predict which passengers survived the tragedy. Before starting the analysis, I hypothesized that female or young passengers were more likely to survive. I also assumed that features such as PassengerID, Name of passengers, ticket number, and cabin number have no effect on survival. 

# **2. Methodology:**
# 
# - Import libraries and Load dataset
# 
# - Reduce data dimension by eliminating irrelevent features
# 
# - Pre-processing
# 
#     - Check and Impute missing cases
# 
#     - Discretization via Binning
# 
#     - Convert Discrete Features into Binary
# 
#     - Convert categorical features into numeric
# 
# - Analyze features with visualization
# 
# - Validation Testing and Prediction
# 
#     - Split data into training and validation sets
# 
#     - Decision Tree Classification
# 
#     - Prediction using Testing set

# **3. Analysis, Results, and Findings**
# 
# **3.1 Import Libraries and Load dataset**

# In[1]:

## Import warnings. 
import warnings
warnings.filterwarnings("ignore") 


# In[2]:

## Import analysis modules
import pandas as pd
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, roc_curve, auc

## Import visualization modules
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns



# In[3]:

## Read in file
train_original = pd.read_csv('../input/train.csv')
test_original = pd.read_csv('../input/test.csv')


# In[4]:

## Basic information about the training set
train_original.info()


# In[5]:

## Basic information about the testing set
test_original.info()


# **3.2 Reduce data dimension by eliminating irrelevent features**
# 
# I assumed that features such as PassengerID, Name of passengers, ticket number, and cabin number are irrelevant or insignificant in this analysis, thus I decided to exclude them.

# In[6]:

# Exclude some features to reduce data dimension
train=train_original.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
test=test_original.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
total = [train,test]

train.shape, test.shape


# **Dependent variable**: Survived
# 
# **Predictors**:
# 
# 1. Numerical
# 
#     - Age: age in years
# 
#     - Fare: Passenger fare
# 
#     - SibSp: # of siblings / spouse aboard the Titanic
# 
#     - Parch: # of parents / children aboard the Titanic
# 
# 3. Categorical
# 
#      - Pclass: Ticket Class (1st=1, 2nd=2, 3rd=3)
# 
#      - Sex: male, female
# 
#      - Embark: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

# **3.3 Preprocessing**
# 
# - Check and Impute missing cases
# 
# - Discretization via binning
# 
# - Convert Discrete Features into Binary
# 
# - Convert categorical features to numeric

# **Training Set: Check and Impute missing cases**

# In[7]:

## Missing cases for training set
train.isnull().sum()


# In[8]:

## Age missing cases
train[train['Age'].isnull()].head()


# In[9]:

## Distribution of Age, condition = Pclass
train[train.Pclass==1].Age.plot(kind='kde', color='r', label='1st class')
train[train.Pclass==2].Age.plot(kind='kde', color='b', label='2nd class')
train[train.Pclass==3].Age.plot(kind='kde', color='g',label='3rd class')
plt.xlabel('Age')
plt.legend(loc='best')
plt.grid()


# According to the distribution for "Age" feature, the median age for passengers deviates depending on three different ticket classes. Thus, we may replace missing cases with the median age value for each ticket class.

# In[10]:

## Create function to replace NaN with the median value for each ticket class
def fill_missing_age(dataset):
    for i in range(1,4):
        median_age=dataset[dataset["Pclass"]==i]["Age"].median()
        dataset["Age"]=dataset["Age"].fillna(median_age)
        return dataset

train = fill_missing_age(train)


# In[11]:

## Embarked missing cases 
train[train['Embarked'].isnull()]


# In[12]:

## Create Barplot 
sns.barplot(x="Embarked", y="Fare", hue="Sex", data=train)


# Considering Sex=female and Fare=80, Ports of Embarkation (Embarked) for two missing cases can be assumed to be Cherbourg (C). 

# In[13]:

## Replace missing cases with C
train["Embarked"] = train["Embarked"].fillna('C')


# **Testing Set: Check and Impute missing cases**

# In[14]:

## Missing cases for testing set
test.isnull().sum()


# In[15]:

## Age missing cases
test[test['Age'].isnull()].head()


# In[16]:

## Distribution of Age, condition = Pclass
test[test.Pclass==1].Age.plot(kind='kde', color='r', label='1st class')
test[test.Pclass==2].Age.plot(kind='kde', color='b', label='2nd class')
test[test.Pclass==3].Age.plot(kind='kde', color='g',label='3rd class')
plt.xlabel('Age')
plt.legend(loc='best')
plt.grid()


# With the same reason above, we may replace missing cases with the median age value for each ticket class.

# In[17]:

## Replace missing cases with the median age for each ticket class.
test = fill_missing_age(test)


# In[18]:

## Fare missing cases 
test[test['Fare'].isnull()]


# In[19]:

## Create function to replace NaN with the median fare with given conditions
def fill_missing_fare(dataset):
    median_fare=dataset[(dataset["Pclass"]==3) & (dataset["Embarked"]=="S")]["Fare"].median()
    dataset["Fare"]=dataset["Fare"].fillna(median_fare)
    return dataset

test = fill_missing_fare(test)


# In[20]:

## Re-Check for missing cases
train.isnull().any()


# In[21]:

test.isnull().any()


# **Discretization via binning**
# 
# In this section, I transferred two continuous features (Age and Fare) into discrete counterparts. 

# In[22]:

## Boxplot for Age
sns.boxplot(x=train["Survived"], y=train["Age"])


# In[23]:

## discretize Age feature
for dataset in total:
    dataset.loc[dataset["Age"] <= 9, "Age"] = 0
    dataset.loc[(dataset["Age"] > 9) & (dataset["Age"] <= 19), "Age"] = 1
    dataset.loc[(dataset["Age"] > 19) & (dataset["Age"] <= 29), "Age"] = 2
    dataset.loc[(dataset["Age"] > 29) & (dataset["Age"] <= 39), "Age"] = 3
    dataset.loc[(dataset["Age"] > 29) & (dataset["Age"] <= 39), "Age"] = 3
    dataset.loc[dataset["Age"] > 39, "Age"] = 4
sns.countplot(x="Age", data=train, hue="Survived")


# In[24]:

## Boxplot for Fare
sns.boxplot(x=train["Survived"], y=train["Fare"])


# The skewness of Fare feature is significantly high. Thus, I discretized the number of bin size based on the third quartile value; if the last bin starts with the third quartile value when bin size = n, then n number of bin will be selected. 

# In[25]:

## discretize Fare
pd.qcut(train["Fare"], 8).value_counts()


# In[26]:

for dataset in total:
    dataset.loc[dataset["Fare"] <= 7.75, "Fare"] = 0
    dataset.loc[(dataset["Fare"] > 7.75) & (dataset["Fare"] <= 7.91), "Fare"] = 1
    dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 9.841), "Fare"] = 2
    dataset.loc[(dataset["Fare"] > 9.841) & (dataset["Fare"] <= 14.454), "Fare"] = 3   
    dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] <= 24.479), "Fare"] = 4
    dataset.loc[(dataset["Fare"] >24.479) & (dataset["Fare"] <= 31), "Fare"] = 5   
    dataset.loc[(dataset["Fare"] > 31) & (dataset["Fare"] <= 69.487), "Fare"] = 6
    dataset.loc[dataset["Fare"] > 69.487, "Fare"] = 7    
    
sns.countplot(x="Fare", data=train, hue="Survived")


# **Convert Discrete Features into Binary**

# In[27]:

## Countplot for the number of siblings/spouse
sns.countplot(x="SibSp", data=train, hue="Survived")


# In[28]:

## Countplot for the number of parents/childrens
sns.countplot(x="Parch", data=train, hue="Survived")


# Since the majority of cases are zero for both discrete features, I converted them into binary format.

# In[29]:

## Convert SibSp into binary feature
for dataset in total:
    dataset.loc[dataset["SibSp"]==0, "SibSp"]=0
    dataset.loc[dataset["SibSp"]!=0, "SibSp"]=1

sns.countplot(x="SibSp", data=train, hue="Survived")


# In[30]:

## Convert Parch into binary feature
for dataset in total:
    dataset.loc[dataset["Parch"]==0, "Parch"]=0
    dataset.loc[dataset["Parch"]!=0, "Parch"]=1
    
sns.countplot(x="Parch", data=train, hue="Survived")


# **Convert categorical features to numeric**

# In[31]:

## Scikit learn estimators require numeric features
sex = {'female':0,'male':1}
embarked = {'C':0,'Q':1,'S':2}   


# In[32]:

## Convert categorical features to numeric using mapping function
for dataset in total:
    dataset['Sex'] = dataset['Sex'].map(sex)
    dataset['Embarked'] = dataset['Embarked'].map(embarked)

train.head()


# **3.4 Analyze features with visualization**
# 
# I calculated and visualized survival rate for each feature to see what sorts of people were likely to survive the tragedy. 

# In[33]:

## total survival rate of train dataset
survived_cases=0
for i in range(891):
    if train.Survived[i]==1:
        survived_cases = survived_cases + 1

total_survival_rate = float(survived_cases)/float(891)

print('%0.4f' % (total_survival_rate))


# Total survival rate was 38.38%

# In[34]:

## Survival rate under each feature condition
def survival_rate(feature):
    rate = train[[feature, 'Survived']].groupby([feature], as_index=False).mean().sort_values(by=[feature], ascending=True)
    sns.factorplot(x=feature, y="Survived", data=rate)


# In[35]:

for feature in ["Age", "Fare", "SibSp", "Parch", "Sex", "Embarked", "Pclass"]:
    survival_rate(feature)


# - **Age**: As I expected from the beginning, young passengers (Age<=9) were more likely to survive than the elder. Age bin for passengers with the lowest survival rate was from 30 to 39, indicating that middle-age people sacrificed themselves when the tragedy occurred. 
# 
# 
# - **Fare / Pclass**: The survival rate significantly increased when passenger belonged to the first class. The bar graph below showed that fare and ticket class were closely related to each other. Therefore, we may explicitly assume that the rich were more likely to survive the tragedy. 
# 
# 
# - SibSp / Parch: The graphs of these two features were very similar; passengers with family were more likely to survive than those who came alone. 
# 
# 
# - **Sex**: As I expected from the beginning, the survival rate for female passengers were significantly higher than the one for male. 
# 
# 
# - Embarked: The survival rate graph showed that passengers who came from Cherbourg were more likely to survive. But I couldn't figure out the key relationship between survival rate and port of Embarkation. Further studies are required.

# In[36]:

## Inter-relationship between Fare and Pclass
sns.countplot(x="Fare", data=train, hue="Pclass")


# In[37]:

## Relationship between Embarked and other features
train.groupby(["Embarked"], as_index=False).mean()


# **3.5 Validation Testing and Prediction**
# 
# - Split data into training and validation sets
# 
# - Decision Tree Classification
# 
# - Prediction using Testing set

# **Split data into testing and validation sets**

# In[38]:

## Seperate input features from target feature
x = train.drop("Survived", axis=1)
y = train["Survived"]


# In[40]:

## Split the data into training and validation sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.25,random_state=1)


# In[41]:

## Take a look at the shape
x_train.shape, y_train.shape


# **Decision Tree Classification**

# In[42]:

# Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=1)


# In[43]:

## Run 10 fold cross validation
cvs = cross_val_score(clf,x,y,cv=5)
print(cvs)


# In[44]:

## Show cross validation score mean and std
print("Accuracy: %0.4f (+/- %0.4f)" % (cvs.mean(), cvs.std()*2))


# The cross validation test result showed that accuracy of the model is 0.8070. Thus, we may conclude that 80.70% of cross-validation set tuples were correctly classified by this model.

# In[45]:

## Fit the model with data
clf.fit(x_train, y_train)


# In[47]:

## Accuracy
acc_decision_tree = round(clf.score(x_train, y_train), 4)
print("Accuracy: %0.4f" % (acc_decision_tree))


# We split the data into 75% training and 25% validation sets, and fitted the model. Based on this split, classifier accuracy of the model turned out to be 0.9027. Thus, we may conclude that 90.27% of validation set tuples were correctly classified by this model.

# In[48]:

## Predict y given validation set
predictions = clf.predict(x_test)


# In[49]:

## Take a look at the confusion matrix ([TN,FN],[FP,TP])
confusion_matrix(y_test,predictions)


# In[50]:

## Precision
print("Precision: %0.4f" % precision_score(y_test, predictions))


# The precision is 0.8310. Thus, we may conclude that 83.10% of tuples that the classifier labeled as positive are actually positive by this model.

# In[51]:

## Recall score
print("Recall: %0.4f" % recall_score(y_test, predictions))


# The recall is 0.6211. Thus, we may conclude that 62.11% of real positive tuples were classified by the decision tree classifier.

# In[52]:

## Print classification report
print(classification_report(y_test, predictions))


# In[53]:

## Get data to plot ROC Curve
fp, tp, th = roc_curve(y_test, predictions)
roc_auc = auc(fp, tp)


# In[54]:

## Plot ROC Curve
plt.title('ROC Curve')
plt.plot(fp, tp, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



# **4. Conclusion:**
# 
# To sum up, passengers had higher chance of survival:
# 
# - if they belonged to the first class (or hold expensive ticket)
# 
# - if they were female
# 
# - if they were young
# 
# - if they had family
# 
# - if they came from Cherbourg
# 
# Among these five conditions, ticket class, sex, and age were the most influential on survival. 
# 
# To test the validity of the classification model, I split the "train" data into 75% of training and 25% of validation sets. And it gave us significantly high accuracy: The classification model predicted 90.27% of validation set tuples correctly. However, the prediction score was not as good as its accuracy or precision. Only 62.11% of true survival cases were detected by the classifier.
# 
# We need further studies or data that would give us insight to improve the predictive model. 
