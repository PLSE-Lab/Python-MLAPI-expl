import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt 
sb.set_style("whitegrid")
#from sklearn.svm import SVM

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

fig, ax1 = plt.subplots(1,1)
ax1 = sb.violinplot(x="Pclass", y="Sex", hue="Survived",
    data=train, palette="Set2", split=True,
    scale="count", inner="stick")


plt.savefig("fig.png")

sex_dict = {'male':1,'female':2}
train_arr = train.loc[:,["Sex","Pclass","Age","SibSp","Parch"]].values
train_target = train["Survived"].values
for l in train_arr:
    l[0] = sex_dict[l[0]]


#ax = sb.factorplot(x="Pclass", y="SibSp", hue="Survived",
#                     data=train, palette="YlGnBu_d", size=6,
#                     aspect=2.0)
#sex = train["Sex"]
##pclass = train["Pclass"]
#for i in  xrange(len(sex):
#print(train_arr)   
#ax.savefig("fig.png")

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)