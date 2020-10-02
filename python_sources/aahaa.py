import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv as csv
import platform,os
import seaborn as sns

#print(os.getcwd())

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

train["Age"][np.isnan(train["Age"])] = np.median(train["Age"])
#train["Survived"][train["Survived"]==1] = "Survived"
#train["Survived"][train["Survived"]==0] = "Died"
train["ParentsAndChildren"] = train["Parch"]
train["SiblingsAndSpouses"] = train["SibSp"]



print(sum(train["Survived"]))
print(train.head())

fig = plt.figure(figsize=(18,6), dpi=1600) 
alpha=alpha_scatterplot = 0.2 
alpha_bar_chart = 0.55
train.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)


#d = []
for row in train.head():
    print(row)
    
#d = np.array(d)
#print(d)
