"""import numpy as np
import pandas as pd
import csv as csv
import re as re
from ggplot import *

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
csv_file_object = csv.reader(open('../input/train.csv', 'r')) 


#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)
next(csv_file_object)
#header = csv_file_object.next()  # The next() command just skips the 
                                 # first line which is a header
data=[]                          # Create a variable called 'data'.
for row in csv_file_object:      # Run through each row in the csv file,
    
    data.append(row)             # adding each row to the data variable
data = np.array(data)            # Then convert from a list to an array
			                     # Be aware that each item is currently
i=0                                 # a string in this format
print (data)
title=[]

for i in range(0,891):
  title.append((re.sub("(.*,)|(\\..*)"," ",data[i,3])))


title=np.array([title])
title=title.transpose()

#x=np.concatenate((data,title),axis=1)
 
#print(x)


#title_sex = pd.crosstab(index=x[:,12],
                           #columns=train["Sex"])
#print(title_sex)

list=['  Dona ', '  Lady ', '  the Countess ','  Capt ', '  Col ', '  Don ', 
                '  Dr ', '  Major ', '  Rev ', '  Sir ', '  Jonkheer ']
                
titlex=[]                
         
for i in range(0,891):
  if title[i] in list:
        titlex.append(['  Rare Title '])
  elif title[i]=='  Mlle ' or title[i]=='  Ms ':
        titlex.append(['  Miss '])
  elif title[i]=='  Mme ':
        titlex.append(['  Mrs '])
  else:
        titlex.append(title[i])
titlex= np.array([titlex])
titlex=titlex.reshape(891,1)
#titlex=titlex.transpose()
print(titlex)
print(len(titlex))
#print(len(title2))
y=np.concatenate((data,titlex),axis=1)
#print(y)  

title_sex = pd.crosstab(index=y[:,12],
                           columns=train["Sex"])
print(title_sex)

surname=[]
for i in range(0,891):               
   surname.append((re.sub("(,.*)"," ",data[i,3])))  
#print(surname)


#do families sink or swim together?
Fsize=[]

for i in range(0,891):
    Fsize.append(int(data[i,6])+int(data[i,7])+1)

print(Fsize)
Fsize=np.array(Fsize)
FsizeD=[]

for i in range(0,891):
    if int(Fsize[i])==1:
      FsizeD.append('singleton')
    elif int(Fsize[i])<5 and int(Fsize[i])>1:
      FsizeD.append('small')
    else:
      FsizeD.append('large')
      
print(FsizeD)  

a= ggplot(train,aes(x="Fare",y="Embarked"))
#print(a)

Embarked=[]
for i in range(0,891):
  if i==61 or i==829:
    Embarked.append("C")  
  else:
    Embarked.append(data[i,11])
    
print(Embarked)
      
count=0
#Predictive imputation 
age=[]
for i in range(0,891):
#if data[i,6].dtype == float and np.isnan(data[i,6]):
    if int(data[i][5])>=0:

        print(9)

print(data[4][5])  """   
      
      
      
#import libraries   
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#read in the two files
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.info()
full=pd.concat([train,test])
#print(full.head())


#manage the passengers' titles
full['Title']=full['Name'].apply(lambda x: x.split(' ')[1])
#print(full.head())
title_sex=pd.crosstab(full['Title'],full['Sex'])
#print(title_sex)
full['Title']=full['Title'].replace(['Don.', 'Lady', 'the','Capt.', 'Col.', 'Don.', 
                'Dr.', 'Major.', 'Rev.', 'Sir', 'Jonkheer.','Billiard,','Brito,','Carlo,',
                'Cruyssen,','Gordon,','Impe,','Khalil,','Melkebeke,','Messemaeker,','Mulder,','Palmquist,','Pelsmaeker,',
                'Planke,','Shawah,','Steen,','Velde,','Walle,','der','y'],'Rare Title')
full['Title']=full['Title'].replace(['Mlle.','Ms.'],'Miss.')
full['Title']=full['Title'].replace('Mme.','Mrs.')
title_sex=pd.crosstab(full['Title'],full['Sex'])
print(title_sex)


#verify survival based on family size
full['Surname']=full['Name'].apply(lambda x: x.split(',')[0])
#print(full.head())
#print(full['Surname'].nunique())
full['Fsize']=full['SibSp']+full['Parch']+1
g=sns.FacetGrid(full,col='Survived')
g.map(plt.hist, 'Fsize', bins=40)
plt.savefig('FamilySurvival.png')
full['Fsize']=full['Fsize'].replace(1,0)
full['Fsize']=full['Fsize'].replace([2,3,4],1)
full['Fsize']=full['Fsize'].replace([5,6,7,8,11],2)
#print(full['Fsize'].value_counts())


#missing values
full=full.drop('Cabin',axis=1)
#print(full[full['Embarked'].isnull()])
g=sns.FacetGrid(full)
sns.boxplot(x=full['Embarked'],y=full['Fare'],hue=full['Pclass'])
plt.savefig('embarkment.png')
full.loc[[61,829],'Embarked']='C'
#print(full[full['Fare'].isnull()])
g=sns.FacetGrid(full[(full['Embarked']=='S') & (full['Pclass']==3)])
g.map(plt.hist, 'Fare', bins=40)
plt.savefig('fare.png')
full=full.reset_index()
full.loc[1043,'Fare']=full[(full['Embarked']=='S') & (full['Pclass']==3)]['Fare'].mean()
#print(full[full['Fare'].isnull()])

#Age
"""for i in range(0,1309):
   if full.loc[i,'Age'] not in range(0,150):
    pclass=full.loc[i,'Pclass']
    sex=full.loc[i,'Sex']
    full.loc[i,'Age']=full[(full['Pclass']==pclass) & (full['Sex']==sex)]['Age'].mean()
print(full.info())"""
null_list=full[full['Age'].isnull()]['PassengerId']
for i in null_list:
    pclass=full.loc[i-1,'Pclass']
    sex=full.loc[i-1,'Sex']
    full.loc[i-1,'Age']=full[(full['Pclass']==pclass) & (full['Sex']==sex)]['Age'].mean()

#verify survival based on age
g=sns.FacetGrid(full,col='Sex',hue='Survived')
g.map(plt.hist,'Age',bins=40)
plt.savefig('agesurvival.png')

full['Child']=full['Age'].apply(lambda x: 'Child' if x<18 else 'Adult')
print(full['Child'].value_counts)
child_survival=pd.crosstab(full['Survived'],full['Child'])
#print(child_survival)

#verify survival for mothers
full['Mother']='Not Mother'
full.loc[((full['Sex']=='female') & (full['Parch'] > 0) & (full['Age'] > 18) & (full['Title'] != 'Miss')),'Mother']='Mother'
mother_survival=pd.crosstab(full['Survived'],full['Mother'])
#print(mother_survival)

title_map = {"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Rare Title": 5}
full['Title'] = full['Title'].map(title_map)
full['Embarked'] = full['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
full['Sex'] = full['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
full['Mother']=full['Mother'].map({'Mother':1,'Not Mother':0}).astype(int)
full['Child']=full['Child'].map({'Child':1,'Adult':0}).astype(int)

full.loc[ full['Fare'] <= 7.91, 'Fare'] = 0
full.loc[(full['Fare'] > 7.91) & (full['Fare'] <= 14.454), 'Fare'] = 1
full.loc[(full['Fare'] > 14.454) & (full['Fare'] <= 31), 'Fare']   = 2
full.loc[ full['Fare'] > 31, 'Fare'] = 3

full.loc[ full['Age'] <= 16, 'Age'] = 0
full.loc[(full['Age'] > 16) & (full['Age'] <= 32), 'Age'] = 1
full.loc[(full['Age'] > 32) & (full['Age'] <= 48), 'Age'] = 2
full.loc[(full['Age'] > 48) & (full['Age'] <= 64), 'Age'] = 3
full.loc[ full['Age'] > 64, 'Age'] = 4

full=full.drop(['Ticket','Name','Surname'],axis=1)


#prediction
train=full[0:891]
test=full[891:1309]
x_train=train.drop('Survived',axis=1)
y_train=train['Survived']
test=test.drop('Survived',axis=1)
from sklearn.ensemble import RandomForestClassifier
random_forest=RandomForestClassifier(n_estimators=3)
random_forest.fit(x_train,y_train)
predictions=random_forest.predict(test).astype(int)
df=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})
print(df)
df.to_csv("titanic_results.csv",index=False,header=True)
acc_random_forest = random_forest.score(x_train, y_train) * 100
print(acc_random_forest)