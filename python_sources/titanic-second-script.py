import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Print you can execute arbitrary python code
train_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_df = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

####Exploratory Data Analysis
#number of people who survived
survive_count = train_df['Survived'].value_counts(normalize=True)
#plotting the survival count
sns.countplot(train_df['Survived'])
#Pclass
survive_Pclass = train_df['Survived'].groupby(train_df['Pclass']).mean() #array
#survive_Pclass = train_df[['Survived', 'Pclass']].groupby('Pclass').mean() #DataFrame
#print (survive_Pclass)
#plotting the survival count by Pclass
sns.countplot(train_df['Pclass'], hue=train_df['Survived'])
#Name
#print (train_df['Name'].head())
#extraction of information of person's title
train_df['Name Title'] = train_df['Name'].apply(lambda x:x.split(',')[1]).apply(lambda x:x.split()[0])
#print (train_df['Name Title'].value_counts())
survive_name_title = train_df['Survived'].groupby(train_df['Name Title']).mean()
#print (survive_name_title)
#check if name length matters in the survival chances
train_df['Name_Len'] = train_df['Name'].apply(lambda x: len(x))
survive_name_len = train_df['Survived'].groupby(train_df['Name_Len']).mean()
#print (survive_name_len)
bucket_Name_Len = pd.qcut(train_df['Name_Len'], 5).value_counts()
#print (train_df['Sex'].value_counts(normalize = True))
survive_sex = train_df['Survived'].groupby(train_df['Sex']).mean()
#print (survive_sex)
#Age
survive_age_isnull = train_df['Survived'].groupby(train_df['Age'].isnull()).mean()
#print (survive_age_isnull)
survive_age_qcut = train_df['Survived'].groupby(pd.qcut(train_df['Age'], 5)).mean()
#print (survive_age_qcut)
#print (pd.qcut(train_df['Age'], 5).value_counts())
#SibSp
survive_sibsp = train_df['Survived'].groupby(train_df['SibSp']).mean()
#print (survive_sibsp)
print (train_df['SibSp'].value_counts())