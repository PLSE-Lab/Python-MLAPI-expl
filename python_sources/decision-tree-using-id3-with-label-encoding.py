

from sklearn.tree import DecisionTreeClassifier as DTC
import numpy as np
import pandas as pd

dataset = pd.read_csv("cosmetic_shop.csv")
X = dataset.iloc[:,:-1].values
X1=dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1].values
#print(X1,Y)

from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
X1=X1.apply(LabelEncoder().fit_transform)
print(X1)

tree = DTC(criterion='entropy')
model = tree.fit(X1,Y)

tree.fit(X1.iloc[:,1:5],Y)

X_in=np.array([0,1,0,0])
y_pred=tree.predict([X_in])
print("Prediction:", y_pred)

import graphviz as gv
import sklearn.tree as tree

gv_comp_model = tree.export_graphviz(model,feature_names=["Age","Income","Gender","Marital_Status"],class_names=['Yes','No'])
#print(gv_comp_model)
x = gv.Source(gv_comp_model)
#print(x)
x.render("treePDF")