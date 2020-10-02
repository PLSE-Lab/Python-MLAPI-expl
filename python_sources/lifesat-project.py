#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import sklearn
import sklearn.linear_model
import sklearn.neighbors


# In[ ]:


lifesat=pd.read_csv("/kaggle/input/lifesat.csv")
lifesat


# In[ ]:


lifesat=lifesat.set_index('Country')


# In[ ]:


lifesat.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.tight_layout()
plt.savefig('scatter-plot.png', dpi=600)
plt.show()


# In[ ]:


lifesat.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
plt.axis([0, 60000, 0, 10])
position_text = {
    "Hungary": (5000, 1),
    "Korea": (15000, 1.7),
    "Italy":(22000,1),
    "France": (29000, 2.4),
    "Australia": (40000, 3.0),
    "United States": (52000, 3.8),
}


# In[ ]:


items = position_text.items()
for country, pos_text in items:
    pos_data_x, pos_data_y = lifesat.loc[country]
    country = "U.S." if country == "United States" else country
    plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text,
            arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=4))
    plt.plot(pos_data_x, pos_data_y, "ro")
plt.xlabel("GDP per capita (USD)")
plt.tight_layout()
plt.savefig('scatter-plot-highlight.png', dpi=500)
plt.show()


# In[ ]:


X = lifesat[["GDP per capita"]]
y = lifesat["Life satisfaction"]


# In[ ]:


model = sklearn.linear_model.LinearRegression()


# In[ ]:


model.fit(X,y)


# In[ ]:


lifesat.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
plt.xlabel("GDP per capita (USD)")
plt.axis([0, 60000, 0, 10])
Xaxis=np.linspace(0, 60000, 1000)
plt.plot(Xaxis, model.intercept_ + model.coef_[0]*Xaxis, "b")
plt.text(5000, 3.1, r"$\theta_0 = 4.85$", fontsize=14, color="red")
plt.text(5000, 2.2, r"$\theta_1 = 4.91 \times 10^{-5}$", fontsize=14, color="red")
plt.tight_layout()
plt.savefig('scatter-plot-regression-line.png', dpi=600)
plt.show()


# In[ ]:


X_new = [[22587]]  # Cyprus' GDP per capita
print("Prediction (Linear Regression) for Cyprus:")
print(model.predict(X_new)) # it outputs [5.96242338]


# In[ ]:


k=1
knnModel = sklearn.neighbors.KNeighborsRegressor(n_neighbors=k)


# In[ ]:


knnModel.fit(X, y)


# In[ ]:


X_new = [[22587]]  # Cyprus' GDP per capita
print("Prediction (KNN) for Cyprus:")


# In[ ]:


print(knnModel.predict(X_new))

