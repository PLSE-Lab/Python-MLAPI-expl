#!/usr/bin/env python
# coding: utf-8

# This code is inspired by the idea of recommendation systems suggested by darkmatter08. 
# However, the model chosen is completely different and I have emphasized a lot on the idea of dimensionality reduction.

# In[ ]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


# here, we're only importing the essential libraries. We will be reducing the dimensions Using PCA to reduce the dimensions and to avoid overfitting. This will also make the dataprocessing, computationally feasible, fast and yet very reliable.

# In[ ]:


dataframe = pd.read_csv("../input/data.csv")
pca = PCA()
Encoder = LabelEncoder()
dataframe.iloc[:,14] = Encoder.fit_transform(dataframe.iloc[:,14].astype(str))
workrate = dataframe['Work Rate'].str.get_dummies(sep='/ ')
position = dataframe['Position'].str.get_dummies(sep='/ ')


# here, we're only importing the essential libraries. We will be reducing the dimensions Using PCA to reduce the dimensions and to avoid overfitting. This will also make the dataprocessing, computationally feasible, fast and yet very reliable.

# In[ ]:


dataframe1 = dataframe.iloc[:, [55,59,63,64,65,69,71,72,76,77]]
dataframe3 = dataframe.iloc[:,[83,86,87,80,81,75]]
#dataframe3 = dataframe.iloc[:,68:80]
dataframe2 = dataframe.iloc[:,[2,17]]
dataframe = pd.concat([dataframe2, dataframe1], axis =1)
#dataframe = pd.concat([dataframe, workrate], axis =1)
dataframe = pd.concat([dataframe, position], axis =1)
dataframe = pd.concat([dataframe, dataframe3], axis =1)
scaler = StandardScaler()
scaler.fit(dataframe.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,39,40,41,42,43,44]])
dataframe.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,39,40,41,42,43,44]] = scaler.transform(dataframe.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,39,40,41,42,43,44]])
dataframe.fillna(0,inplace = True)
X = dataframe.iloc[:,1:]
df1 = dataframe


# Here, we are basically concerned choosing our parameters and making a final dataframe that consists the parameters that are crucial 
# We are using the following parameters:
# 1. Skill Moves
# 2. Finishing
# 3. Dribbling
# 4. Ball Control
# 5. Acceleration
# 6. Sprint Speed
# 7. Shot Power
# 8. Staina
# 9. Strength
# 10. Positioning
# 11. Vision
# 12. Position
# 13. Goalkeeper Diving
# 14. Goalkeeper Positioning
# 15. Goalkeeper Reflexes
# 16. Marking 
# 17. Standing Tackles
# 18. Interceptions
# 
# The parameters chosen are based purely on intuition and trial and error. It is clear that the variables we have are too many to visualize. Also, it will be a lot more complex, computationally, to perform calculations on such myriad of parameters. Plus, we increased the number of parameters dramatically by using the get_dummies() method. the number is as high as 45. Thus, we will use dimensionality reduction with PCA, to make it computationally feasible.

# In[ ]:


pca = PCA(n_components=2)
pca.fit_transform(X)
X = pca.transform(X)
x = pd.DataFrame(X)
explained_var = pca.explained_variance_ratio_
Names = dataframe.iloc[:,0]
names = pd.DataFrame(Names)
dataframe = pd.concat([names, x], axis =1)
recommendations = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X)
player_indices = recommendations.kneighbors(X)[1]


# In[ ]:


explained_var


# Very Interesting! the variable explained_var actually shows us 'how much of the data features can be conserved in two dimensions'. in this case, it is over 70! that is great because our 2 variable data shows about 72 percent of  what a 45 dimensional data would've shown us. Plus, we are reducing the computational complexity by a great extent and also greatly reducing the chance of overfitting.

# In[ ]:


def get_index(x):
    return dataframe[dataframe['Name']==x].index.tolist()[0]

def recommend_me(player):
    print('Here are 5 players similar to', player, ':' '\n')
    index = get_index(player)
    for i in player_indices[index][1:]:
            print(dataframe.iloc[i]['Name'], '\n')


# We are done writing functions to integrate the name to the attributes. It's time for us to run the code and see if the recommendations, derived merely by 3 columns, actually make sense

# In[ ]:


recommend_me("T. Courtois")


# In[ ]:


recommend_me("L. Messi")


# In[ ]:


recommend_me("Isco")


# In[ ]:


recommend_me("M. de Ligt")


# We have used 4 players who play at 4 different positions, and we have also seen the recommendations. this model can still be improved when compared to human recommendations, but it still seems to have done a great job!!

# In[ ]:


#We can plot a graph between these two parameters, to show the divergence in these two dimensions:
x = dataframe.iloc[:,1]
y = dataframe.iloc[:,2]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x,y, color='blue', marker='.')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
plt.show()


# The two axes in the above displpayed figure, are the eigen value based axes that describe most of the data.These axes are highly sensitive and play a crucial role in clustering the data.Infact, these are the only axes based on which the data is clustered
