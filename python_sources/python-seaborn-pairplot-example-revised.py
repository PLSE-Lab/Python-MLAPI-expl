import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#import
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64})
# I changed the fill na up a little. The previous didn't work for me.
train["Age"].fillna(train["Age"].median())
train["Survived"][train["Survived"]==1] = "Survived"
train["Survived"][train["Survived"]==0] = "Died"
train["ParentsAndChildren"] = train["Parch"]
train["SiblingsAndSpouses"] = train["SibSp"]

#Narrowed down the dataframe to just the features we are concerned with
train = train[['Fare','Survived','ParentsAndChildren','SiblingsAndSpouses','Pclass']]

plt.figure()
sns.pairplot(data=train, hue="Survived", dropna=True)
plt.savefig("1_seaborn_pair_plot.png")
