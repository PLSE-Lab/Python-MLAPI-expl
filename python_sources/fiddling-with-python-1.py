import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

# Replacing missing ages with median
train["Age"][np.isnan(train["Age"])] = np.median(train["Age"])
train["Survived"][train["Survived"]==1] = "Survived"
train["Survived"][train["Survived"]==0] = "Died"
train["ParentsAndChildren"] = train["Parch"]
train["SiblingsAndSpouses"] = train["SibSp"]

plt.figure()
sns.pairplot(data=train[["Fare","Survived","Age","ParentsAndChildren","SiblingsAndSpouses","Pclass"]],
             hue="Survived", dropna=True)
plt.savefig("1_seaborn_pair_plot.png")
