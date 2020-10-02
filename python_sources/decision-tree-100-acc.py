#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


dictNames = {}
for x in df.columns:
    if "-" in x:
        dictNames[x] = x.replace("-","_")
df = df.rename(columns = dictNames)
df.columns


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.bar(df["class"], height=10)
plt.show()


# In[ ]:


pd.crosstab(df["cap_shape"], df["class"])


# In[ ]:


for x in df.columns:
    print(f"{x:{25}} has {df[x].nunique():{5}} levels.")


# In[ ]:


dfTemp = pd.crosstab(df["cap_shape"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] = 500* dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "Cap_Shape vs Class");


# In[ ]:


def carShape(x):
    if x in ["f","x"]:
        return "fx"
    elif x in ["b", "k"]:
        return "ck"
    else:
        return x

df["cap_shape"] = list(map(carShape, df["cap_shape"]))


# In[ ]:


df["cap_shape"].unique()


# In[ ]:


dfTemp = pd.crosstab(df["cap_surface"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] = 500* dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "cap_surface vs Class");


# In[ ]:


def carSurface(x):
    if x in ["s","y"]:
        return "sy"
    else:
        return x

df["cap_surface"] = list(map(carSurface, df["cap_surface"]))


# In[ ]:


df["cap_surface"].unique()


# In[ ]:


dfTemp = pd.crosstab(df["cap_color"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] = 500 * dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "cap_color vs Class");


# In[ ]:


def carColor(x):
    if x in ["c","w"]:
        return "cw"
    elif x in ["e","p","y"]:
        return "epy"
    elif x in ["r","u"]:
        return "ru"
    elif x in ["g","n"]:
        return "gh"
    else:
        return x

df["cap_color"] = list(map(carColor, df["cap_color"]))


# In[ ]:


df["cap_color"].unique()


# In[ ]:


dfByCapShape = pd.crosstab(df["bruises"], df["class"])
dfByCapShape.plot(kind="bar",legend = True, grid=True, title = "bruises vs Class");


# In[ ]:


df.drop(columns = ["veil_type"],inplace=True)


# In[ ]:


dfTemp = pd.crosstab(df["odor"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] =  dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "odor vs Class");


# In[ ]:


def odor(x):
    if x in ["c","f","m","p","s","y"]:
        return "cfmpsy"
    else:
        return "aln"

df["odor"] = list(map(odor, df["odor"]))


# In[ ]:


df["odor"].unique()


# In[ ]:


dfTemp = pd.crosstab(df["gill_attachment"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] = 1000* dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "gill_attachment vs Class");


# In[ ]:


dfTemp = pd.crosstab(df["gill_spacing"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] = 1000* dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "gill_spacing vs Class");


# In[ ]:


dfTemp = pd.crosstab(df["gill_size"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] = 100* dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "gill_size vs Class");


# In[ ]:


dfTemp = pd.crosstab(df["gill_color"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] = 1000 * dfTemp["p"]/dfTemp["e"]
my_list = [ 'e', 'g', 'h', 'k', 'n', 'o', 'p',  'u', 'w', 'y']
dfTemp = dfTemp[dfTemp.index.isin(my_list)]
dfTemp.plot(kind="bar",legend = True, title = "gill_color vs Class");


# In[ ]:


def gillColor(x):
    if x in ["n","u"]:
        return "nu"
    elif x in ["g","h"]:
        return "gh"
    elif x in ["e","o"]:
        return "eo"
    elif x in ["k","w","y"]:
        return "kwy"
    else:
        return x

df["gill_color"] = list(map(gillColor, df["gill_color"]))


# In[ ]:


df["gill_color"].unique()


# In[ ]:


dfTemp = pd.crosstab(df["stalk_root"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] = 1000* dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "stalk_root vs Class");


# In[ ]:


def stalk_root(x):
    if x in ["c","r"]:
        return "cr"
    else:
        return x

df["stalk_root"] = list(map(stalk_root, df["stalk_root"]))


# In[ ]:


df["stalk_root"] = df["stalk_root"].apply(lambda x : "x" if x=="?" else x)
df["stalk_root"].unique()


# In[ ]:


dfTemp = pd.crosstab(df["stalk_surface_above_ring"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] = 1000* dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "stalk_surface_above_ring vs Class");


# In[ ]:


def stalk_surface_above_ring(x):
    if x in ["f","s","y"]:
        return "fsy"
    else:
        return x

df["stalk_surface_above_ring"] = list(map(stalk_surface_above_ring, df["stalk_surface_above_ring"]))


# In[ ]:


df["stalk_surface_above_ring"].unique()


# In[ ]:


dfTemp = pd.crosstab(df["stalk_surface_below_ring"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] = 1000* dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "stalk_surface_below_ring vs Class");


# In[ ]:


def stalk_surface_below_ring(x):
    if x in ["f","s","y"]:
        return "fsy"
    else:
        return x

df["stalk_surface_below_ring"] = list(map(stalk_surface_below_ring, df["stalk_surface_below_ring"]))


# In[ ]:


df["stalk_surface_below_ring"].unique()


# In[ ]:


dfTemp = pd.crosstab(df["stalk_color_above_ring"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] =  100000*dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "stalk_color_above_ring vs Class");


# In[ ]:


def stalk_color_above_ring(x):
    if x in ["e","g","o"]:
        return "ego"
    elif x in ["c","n"]:
        return "cn"
    elif x in ["p","w","y"]:
        return "pwy"
    else:
        return x

df["stalk_color_above_ring"] = list(map(stalk_color_above_ring, df["stalk_color_above_ring"]))


# In[ ]:


df["stalk_color_above_ring"].unique()


# In[ ]:


dfTemp = pd.crosstab(df["stalk_color_below_ring"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] =  100000*dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "stalk_color_below_ring vs Class");


# In[ ]:


def stalk_color_below_ring(x):
    if x in ["e","g","o"]:
        return "ego"
    elif x in ["c","y"]:
        return "cy"
    elif x in ["n","p","w"]:
        return "npw"
    else:
        return x

df["stalk_color_below_ring"] = list(map(stalk_color_below_ring, df["stalk_color_below_ring"]))


# In[ ]:


df["stalk_color_below_ring"].unique()


# In[ ]:


dfTemp = pd.crosstab(df["veil_color"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] =  1000*dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "veil_color vs Class");


# In[ ]:


def veil_color(x):
    if x in ["n","o"]:
        return "no"
    else:
        return x

df["veil_color"] = list(map(veil_color, df["veil_color"]))


# In[ ]:


df["veil_color"].unique()


# In[ ]:


dfTemp = pd.crosstab(df["ring_number"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] =  1000*dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "ring_number vs Class");


# In[ ]:


dfTemp = pd.crosstab(df["ring_type"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] =  1000*dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "ring_type vs Class");


# In[ ]:


def ring_type(x):
    if x in ["e","p"]:
        return "ep"
    else:
        return x

df["ring_type"] = list(map(ring_type, df["ring_type"]))


# In[ ]:


df["ring_type"].unique()


# In[ ]:


dfTemp = pd.crosstab(df["ring_number"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] =  1000*dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "ring_number vs Class");


# In[ ]:


dfTemp = pd.crosstab(df["spore_print_color"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] =  1000*dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "spore_print_color vs Class");


# In[ ]:


def spore_print_color(x):
    if x in ["b","o","u","y"]:
        return "bouy"
    elif x in ["k","n"]:
        return "kn"
    else:
        return x

df["spore_print_color"] = list(map(spore_print_color, df["spore_print_color"]))


# In[ ]:


df["spore_print_color"].unique()


# In[ ]:


dfTemp = pd.crosstab(df["population"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] =  1000*dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "population vs Class");


# In[ ]:


def population(x):
    if x in ["a","n"]:
        return "an"
    elif x in ["s","y"]:
        return "sy"
    else:
        return x

df["population"] = list(map(population, df["population"]))


# In[ ]:


df["population"].unique()


# In[ ]:


dfTemp = pd.crosstab(df["habitat"], df["class"])
dfTemp = dfTemp.replace({0:1})
dfTemp["p_e"] =  1000*dfTemp["p"]/dfTemp["e"]
dfTemp.plot(kind="bar",legend = True, title = "habitat vs Class");


# In[ ]:


def habitat(x):
    if x in ["d","g"]:
        return "dg"
    elif x in ["l","u"]:
        return "lu"
    elif x in ["m","w"]:
        return "mw"
    else:
        return x

df["habitat"] = list(map(habitat, df["habitat"]))


# In[ ]:


df["habitat"].unique()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,confusion_matrix


# In[ ]:


for x in df.columns:
    print(f"{x:{25}} has {df[x].nunique():{5}} levels.")


# In[ ]:


for x in df.columns:
    if x == "class":
        pass
    else:
        dfTemp = pd.get_dummies(df[x], prefix= x, drop_first=True)
        df = pd.concat([df, dfTemp], axis=1, join="inner")


# In[ ]:


df.select_dtypes("object").columns


# In[ ]:


y = df[["class"]]
df.drop(columns = ['class', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
       'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color',
       'stalk_shape', 'stalk_root', 'stalk_surface_above_ring',
       'stalk_surface_below_ring', 'stalk_color_above_ring',
       'stalk_color_below_ring', 'veil_color', 'ring_number', 'ring_type',
       'spore_print_color', 'population', 'habitat'] , inplace = True)


# In[ ]:


y["class"].unique()


# In[ ]:


df1 = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")
y = df1[["class"]]
y["class"] = y["class"].apply(lambda x: 1 if x=="p" else 0)


# In[ ]:


y.nunique()


# In[ ]:


from sklearn.model_selection import train_test_split
X = df
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)


# In[ ]:


modelDTC = DecisionTreeClassifier()
modelDTC.fit(X_train,y_train)
modelDTC.score(X_test,y_test)


# In[ ]:


y_pred = modelDTC.predict(X_test)
print(classification_report(y_test,y_pred))


# In[ ]:


from sklearn.metrics import roc_curve, auc, confusion_matrix
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:




