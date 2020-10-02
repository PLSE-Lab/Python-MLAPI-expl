#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix ,classification_report 
from sklearn.model_selection import cross_val_score,cross_validate
import seaborn as sns
from sklearn import preprocessing



#@title select 
quantile =  10#@param ["2", "4", "10", "5"] {type:"raw", allow-input: true}
beaufort = False #@param {type:"boolean"}

join=pd.read_csv("../input/wind-coron/cortegadaD1res4K.csv")
table_columns=[]
table=[]
table_index=[]
X_var=["mod_NE","mod_SE","mod_SW","mod_NW"]
for var_pred0 in X_var:
  var_obs="spd_o"
  if beaufort:

    #first cut observed variable in Beaufort intervals
    bins_b = pd.IntervalIndex.from_tuples([(-1, 0.5), (.5, 1.5), (1.5, 3.3),(3.3,5.5),
                                     (5.5,8),(8,10.7),(10.7,13.8),(13.8,17.1),
                                     (17.1,20.7),(20.7,24.4),(24.4,28.4),(28.4,32.6),(32.6,60)])
    join[var_obs+"_l"]=pd.cut(join[var_obs], bins=bins_b).astype(str)
    join[var_pred0+"_l"]=pd.cut(join[var_pred0],bins = bins_b).astype(str)

    #transform to Beaufort scale
    bins_b=bins_b.astype(str)
    labels=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12"]
    join[var_obs+"_l"]=join[var_obs+"_l"].map({a:b for a,b in zip(bins_b,labels)})
    join[var_pred0+"_l"]=join[var_pred0+"_l"].map({a:b for a,b in zip(bins_b,labels)})

  else:
    #first q cut observed variable then cut predicted with the bins obtained at qcut
    join[var_obs+"_l"]=pd.qcut(join[var_obs], quantile, retbins = False,precision=1).astype(str)
    interval=pd.qcut(join[var_obs], quantile,retbins = True,precision=1)[0].cat.categories
    join[var_pred0+"_l"]=pd.cut(join[var_pred0],bins = interval).astype(str)
        
 

  #results tables
  res_df=pd.DataFrame({"pred_var":join[var_pred0+"_l"],"obs_var":join[var_obs+"_l"]})
  table.append(pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,))
  table_columns.append(pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,normalize="columns"))
  table_index.append(pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,normalize="index")  )

#plt results tables and classification report
for i in range(0,len(X_var)):
  print("Independent variable:",X_var[i]," Observed result:",var_obs)
  print(classification_report(join[var_obs+"_l"],join[X_var[i]+"_l"]))
  fig, axs = plt.subplots(3,figsize = (8,10))
  sns.heatmap(table[i],annot=True,ax=axs[0],cmap="YlGnBu",fmt='.0f',)
  sns.heatmap(table_columns[i],annot=True,ax=axs[1],cmap="YlGnBu",fmt='.0%')
  sns.heatmap(table_index[i],annot=True,ax=axs[2],cmap="YlGnBu",fmt=".0%")
  plt.show()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn import tree
import graphviz
import pickle

#@title Load or train model. If load, same quantile and var_pred1

X_var = ['mod_NE', 'mod_SE','mod_SW', 'mod_NW',"wind_gust_NE","wind_gust_SE","wind_gust_SW","wind_gust_NW"] #@param ["['mod_NE', 'mod_SE','mod_SW', 'mod_NW']", "['mod_NE', 'mod_SE','mod_SW', 'mod_NW',\"wind_gust_NE\",\"wind_gust_SE\",\"wind_gust_SW\",\"wind_gust_NW\"]"] {type:"raw"}
var_obs = "spd_o" 
max_depth=9#@param ["2", "5", "10", "15"] {type:"raw", allow-input: true}
criterion = "gini" #@param ["gini", "entropy"]
RF = False #@param {type:"boolean"}
n_estimators =  200#@param {type:"integer"}
cross_validation = True #@param {type:"boolean"}
#cut in bins
if beaufort:
  Y=join[var_obs+"_l"]
  filename_out = "treeb.h5"
else:
  Y=pd.qcut(join[var_obs], quantile, retbins = False,precision=1).astype(str)
  labels=pd.qcut(join[var_obs], quantile,retbins = True,precision=1)[0].cat.categories.astype(str)
  filename_out = "treeq.h5"
#independent variables. 
X=join[X_var]


#we do not scale!!
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,random_state=42)

#select classification
#train and save as tree2 or load tree
if RF:
  clf1=RandomForestClassifier(n_estimators=n_estimators).fit(x_train,y_train) 
else:
  clf1 = DecisionTreeClassifier(max_depth=max_depth,criterion=criterion).fit(x_train,y_train) 

pickle.dump(clf1, open(filename_out, 'wb'))

y_pred=clf1.predict(x_test)

#plot results
print(classification_report(y_test,y_pred))


#plot results

res_df=pd.DataFrame({"pred_var":y_pred,"obs_var":y_test})
table=pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,)
table_columns1=pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,normalize="columns")
table_index=pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,normalize="index")






fig, axs = plt.subplots(3,figsize = (8,10))
sns.heatmap(table,annot=True,ax=axs[0],cmap="YlGnBu",fmt='.0f',)
sns.heatmap(table_columns1,annot=True,ax=axs[1],cmap="YlGnBu",fmt='.0%')
sns.heatmap(table_index,annot=True,ax=axs[2],cmap="YlGnBu",fmt=".0%")
plt.show()


print("Features importances:")
fi=["{:.0%}".format(x) for x in clf1.feature_importances_]
print(dict(zip(X.columns,fi )))

if RF==False:
  #tree save file name: tree

  dot_data = tree.export_graphviz(clf1, out_file=None, 
                                    feature_names=X.columns,  
                                    class_names=labels,  
                                    filled=True, rounded=True,  
                                    special_characters=True)  
  graph = graphviz.Source(dot_data) 
  graph.render("tree")
   

#cross validation
if cross_validation:
  print ("***Accuracy score***")
  print(cross_val_score(clf1, X, Y, cv=5,scoring="accuracy"))
  print ("***F1_macro score***")
  print(cross_val_score(clf1, X, Y, cv=5,scoring='f1_macro'))

