#!/usr/bin/env python
# coding: utf-8

# # 3D Printer DataSet for Mechanical Engineers

# Import Data

# In[ ]:


import pandas as pd
data = pd.read_csv("../input/3dprinter/data.csv", sep = ",")


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# Good for us that there are no "nan" or "null" values

# In[ ]:


for column in data.columns:
    print("{} : {}".format(column,data[column].unique()))


# Converting categorical variable to Numerical

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df = data.copy()
df["infill_pattern"] = le.fit_transform(data["infill_pattern"])
df["material"] = le.fit_transform(data["material"])

        


# In[ ]:


df.head()


# Now we can see how features affect eachother through correlation matrix

# In[ ]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(3)


# In[ ]:


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})


# Pairplot takes some time, be patient

# In[ ]:


import seaborn as sns

sns.pairplot(df)


# Above matrix and graph shows that **layer height** ,*nozzle temperature*, *bed temperature*, *print speed*, *fan speed* affect the **output variables**, here output variables are* not material type*s.
# 
# Some variables for ex **bed temperature**,**nozzle temperature** and **fan speed** are **linearly dependent** and can be combined but, for using FFN we can allow all variables , later we can try dropping some linearly dependent variables

# In[ ]:


sns.pairplot(df,vars=["material", "nozzle_temperature"])


# - Above plot shows that , **nozzle temperature** would be a deciding factor in predicting material type
# - It means that if material is **abs** then nozzle temperature reaches to higher level in comparision to when material is **pla** 

# In[ ]:


sns.pairplot(data,vars=["fan_speed", "nozzle_temperature"],diag_kind="kde",hue="material")


# -Above plot clearly shows that cooling is not very effective in **abs** because the nozzle_temperature is increasing at a higher rate

# In[ ]:


sns.pairplot(df,vars=["material", "print_speed"])


# In[ ]:


y_data = df.material.values
x_data = df.drop(["material","print_speed","infill_pattern","bed_temperature","layer_height"],axis=1)


# **Dropping all the variables which have 0 correlation with material , it would also reduce the computation time**

# In[ ]:


# normalization 
from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()

x_norm = mm.fit_transform(x_data)

# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_norm,y_data,test_size = 0.3,random_state=1)





from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))

score_list = []
for each in range(1,8):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    print(" {} nn score: {} ".format(each,knn2.score(x_test,y_test)))
    
plt.plot(range(1,8),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()


# In ANN all the input variables as well as output variables can be included (though it's not necessary and should be avoided if data is too big), but it can find out important features by itself so it won't make much difference 

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Input, Dense, Flatten
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization

y = df.material.values
x = df.drop(["material"],axis=1)
x_norm = mm.fit_transform(x)

model = Sequential()
model.add(Dense(32,input_dim=11))
model.add(BatchNormalization(axis = -1))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(16))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_norm,y, epochs=500, batch_size =32, validation_split= 0.20)


# In[ ]:


import numpy as np

a1 = 4 #layer_height*100
a2 = 5 #wall_thickness
a3 = 60 #infill_density
a4 = 0 #infilkk_pattern
a5 = 232 #nozzle_temperature 
a6 = 74 #bed_temperature
a7 = 90 #print_speed
a8 = 100 #fan_speed
a9 = 150 #roughness
a10 = 30 #tension_strenght
a11 = 200 #elangation*100

tahmin = np.array([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11]).reshape(1,11)
print(model.predict_classes(tahmin))

if model.predict_classes(tahmin) == 0: 
    print("Material is ABS")
else:   
    print("Material is PLA.")


# This prediction could have also been easily made looking at the Nozzle temperature only, I don't understand why this problem is being solved with ANN when actually a Decision Tree classifier could have been used easily to predict the material used as input
# 

# In[ ]:




