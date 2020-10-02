import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

df = pd.read_excel('https://www.kaggle.com/shahbaz275817/combined-power-plant-dataset/downloads/Folds5x2_pp.xlsx/1')

X = df.iloc[:,:4]
Y = df.iloc[:,4:]

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=24)

#prediction_space = np.linspace(min(x_test), max(x_test))


regg = LinearRegression()
regg.fit(x_train,y_train)

ridge = Ridge(alpha=0.1,normalize=True)
ridge.fit(x_train,y_train)

prediction = regg.predict(x_test)
rig_predict = ridge.predict(x_test)

print(rig_predict)
print("Prediction: {}".format(prediction))

print(regg.score(x_test,y_test))
print(ridge.score(x_test,y_test))

plt.plot(x_train,y_train,color='black')
plt.show()

print("R^2: {}".format(regg.score(x_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,prediction))
print("Root Mean Squared Error: {}".format(rmse))

cv_scores = cross_val_score(regg,X,Y,cv=10)
print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


##############Lasso Model#############
lasso = Lasso(alpha=0.4,normalize=True)
lasso.fit(X,Y)
lasso_coef = lasso.coef_
print(lasso_coef)
