import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
'''
#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())
'''
#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)

#This analyses the age and gender effect on Survival

#Age variable has 177 missing values. We will try to fix this missing value using other training data and applying a Random Forest on the other fields to get the age of the user

#Percentage of Missing Values in Age
total_rows=train.shape[0]
#print("total No of rows in titanic dataset is %d " %total_rows)
age_missing=float(total_rows-train[['Age']].dropna().shape[0]*100)/float(total_rows)
#print("Percentage of Age values missing is %f " %(age_missing))

#What is the average Age value in the dataset
train_age=train[['Age']].dropna
average=train['Age'].dropna().mean()

#print("Average Age of people (ignoring missing datat) is %f " %average)



#If we had only a small percent of missing value for age, we could have replaced it with the average

#Is there any relation between Pclass and age?

ax=train.boxplot(column='Age',by='Pclass')
fig = ax.get_figure()
fig.savefig('Age vs Pclass Boxplot.png')
plt.show()

#From the Box Plot we can observe that average age of Pclass=1 is arnd 38 and for pclass=2 is arnd 29 and pclass=3 is arnd 23
#This means that from Pclass we can calculate the age of a person, we can say if passenger in in Pclass=1 then his age may be 38

#In the name of the passenger, there is the title also included, add a new col called Title and check how that effects the Age
#Name in format Ln,Title.FN, 


train['Title']=train['Name'].map(lambda x : x.split(",")[1].split(".")[0])


#Is there any realtion between age and Title

ax=train.boxplot(column='Age',by='Title')
fig=ax.get_figure()
fig.savefig('AgevsTitleBoxPlot.png' )


#Higher the title like Captain or Colonel, more is the age.

#Effect of number of Parents,Childern and Siblings and Spouses on Board
ax=train.boxplot(column='Age',by='Parch')
fig=ax.get_figure()
fig.savefig("AgevsNumberofParentsandChildren.png")

ax=train.boxplot(column='Age',by='SibSp')
fig=ax.get_figure()
fig.savefig("AgevsNumberofSiblingandSpouse.png")

#From the boxplots we can see that title,Pclass,SibSp,Parch effect the age of a person
#Let us train the dataset using train.csv for Age and test it using test.csv and the predcit the value for missing Age in train.csvs
le=pp.LabelEncoder()
train_new=train.dropna(subset=['Age'])
train_new['Title']=le.fit_transform(train_new['Title'])
#train_X=train_new[['Pclass','Parch','SibSp','Title']]
train_X=train_new[['Title','Pclass','Parch','SibSp']]

#print(list(train_X.columns.values))
train_Y=train_new['Age']

print(list(train_X.columns.values))
test_new=test.drop('Age',axis=1)
test_new['Title']=test_new['Name'].map(lambda x : x.split(",")[1].split(".")[0])
test_new['Title']=le.fit_transform(test_new['Title'])
#print(test_new.describe())
#print(test.describe())

#Apply linear regression model to train_new

#Create Linear Regression object
lr_obj=lm.LinearRegression()
#train using train_new
lr_obj.fit(train_X,train_Y)

#test_X=test_new[['Pclass','Parch','SibSp','Title']]
test_X=test_new[['Title','Pclass','Parch','SibSp']]
#calculate mean square error
MSE=np.mean((lr_obj.predict(test_X)-test['Age'])**2)

print("Mean Square Error is %.2f" %MSE)
 
#get Rows where Age has no NAN values

#print(age_noNa.describe())
#get rows different in train_new and train. This will give me list of users whose value have not been given
unknownage=train.loc[(train['Age'].isnull())]
unknownage['Title']=le.fit_transform(unknownage['Title'])
#print(type(unknownage))
predictedAges =lr_obj.predict(unknownage[['Title','Pclass','Parch','SibSp']])
train.loc[ (train['Age'].isnull()), 'Age' ] = predictedAges 
train.to_csv('train_data_missingage_fixed.csv', index=False)