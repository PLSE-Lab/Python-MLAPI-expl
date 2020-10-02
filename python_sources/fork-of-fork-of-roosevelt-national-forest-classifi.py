# %% [code]
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#1 import libaries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import forest
from matplotlib import pyplot as plt

# %% [markdown]
# ## 2. Load data

# %% [code]
test = '../input/learn-together/test.csv'
train = '../input/learn-together/train.csv'

train = pd.read_csv(train)
test = pd.read_csv(test)

_test = test.drop("Id",axis=1)

x = train.drop(["Id","Cover_Type"],axis=1)
y = train["Cover_Type"]

# Used 20% for testing data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20)


"""
x_plot = []
y_plot = []
for i in range(1,120):
 model = forest.RandomForestClassifier(n_estimators=i)
 model = model.fit(x_train,y_train)
 model = model.score(x_test,y_test)
 y_plot.append(model)
 x_plot.append(i)

plt.plot(x_plot,y_plot)
plt.show()

"""

model = forest.RandomForestClassifier(n_estimators=118)
model = model.fit(x_train,y_train)
#model = model.score(x_test,y_test)
model = model.predict(_test)


id = test["Id"]
cover_type =  model

data = {"id":id,"cover_type":model}

submit = pd.DataFrame(data)

submit = submit.to_csv("submission.csv",index=False)

print(submit)







