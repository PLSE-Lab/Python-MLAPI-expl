import pandas as pd
from pandas import Series,DataFrame

import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style('whitegrid')
#matplotlib inline

titanic_df=pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

titanic_df.head()               #BC YEH KAAM KYON NAHI KARR RHA 

titanic_df.info()
print("------------------------------")
test_df.info()


titanic_df.drop(['PassengerId','Name','Ticket'],axis=1)
test_df.drop(['Name','Ticket'],axis=1)

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

sns.factorplot('Embarked','Survived',data=titanic_df,size=4,aspect=3)
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
sns.countplot(x='Embarked', data=titanic_df, ax=axis1)
sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2)

embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)

embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

embark_dummies_test  = pd.get_dummies(test_df['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)

titanic_df = titanic_df.join(embark_dummies_titanic)
test_df    = test_df.join(embark_dummies_test)

titanic_df.drop(['Embarked'], axis=1,inplace=True)
test_df.drop(['Embarked'], axis=1,inplace=True)