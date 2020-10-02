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


import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, AlphaDropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaler = MinMaxScaler()
train= True #@param {type:"boolean"}
filename_in = "/content/drive/My Drive/Colab Notebooks/wind_ria_arousa/neural.h5" #@param ["/content/drive/My Drive/Colab Notebooks/wind_ria_arousa/neural.h5"] {type:"raw", allow-input: true}
X_var = ['mod_NE', 'mod_SE','mod_NW', 'mod_SW',"wind_gust_NE","wind_gust_SE" ,"wind_gust_SW","wind_gust_NW"] #@param ["['mod_NE', 'mod_SE','mod_NW', 'mod_SW',\"wind_gust_NE\",\"wind_gust_SE\" ,\"wind_gust_SW\",\"wind_gust_NW\"]", "['mod_NE', 'mod_SE','mod_NW', 'mod_SW']"] {type:"raw", allow-input: true}
learn_rate = 0.001 #@param ["0.001", "0.01", "0.0001"] {type:"raw", allow-input: true}
batch_size =  48#@param ["48", "96", "124"] {type:"raw", allow-input: true}
epochs = 100 #@param ["10", "100", "1000"] {type:"raw", allow-input: true}


if beaufort:
  file_out="neuralb.h5"
else:
  file_out="neuralq.h5"

#cut Y variables in quantiles

if beaufort:
  Y=join[var_obs+"_l"]
else:
  Y=pd.qcut(join[var_obs], quantile, retbins = False,precision=1).astype(str)
  labels=pd.qcut(join[var_obs], quantile,retbins = True,precision=1)[0].cat.categories.astype(str)


#transform bins_label to label binary array

lb = preprocessing.LabelBinarizer()
lb.fit(labels)
Y=lb.transform(Y)


#independent variables. 
X=scaler.fit_transform(join[X_var])


#we  scale and split


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,random_state=42)



#neural network
if train:

  mlp = Sequential()
  mlp.add(Input(shape=(x_train.shape[1], )))
  mlp.add(Dense(48, activation='relu'))
  mlp.add(Dropout(0.5))
  mlp.add(Dense(48, activation='relu'))
  mlp.add(Dropout(0.5))
  mlp.add(Dense(len(labels), activation='sigmoid'))
  mlp.compile(optimizer=tf.keras.optimizers.Adam(lr=learn_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy',])
             

  history = mlp.fit(x=x_train,
                    y=y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    verbose=0).history
  pd.DataFrame(history).plot(grid=True,figsize=(12,12),)
  

else:
  #model loaded must have same X variables
  mlp = tf.keras.models.load_model(filename_in)


mlp.save(file_out)


mlp.summary()
y_pred=mlp.predict(x_test)


#transform bynary array to label scale 

y_pred=lb.inverse_transform(y_pred)
y_test=lb.inverse_transform(y_test)




#plot results
res_df=pd.DataFrame({"pred_var":y_pred,"obs_var":y_test})
table=pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,)
table_columnsneural=pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,normalize="columns")
table_index=pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,normalize="index")


fig, axs = plt.subplots(3,figsize = (8,10))
sns.heatmap(table,annot=True,ax=axs[0],cmap="YlGnBu",fmt='.0f',)
sns.heatmap(table_columnsneural,annot=True,ax=axs[1],cmap="YlGnBu",fmt='.0%')
sns.heatmap(table_index,annot=True,ax=axs[2],cmap="YlGnBu",fmt=".0%")
plt.show()

#plot results
print(classification_report(y_test,y_pred))

