#importing necessary libraries:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
#importing the dataset:
a=pd.read_csv("../input/Folds5x2_pp.csv")
print(a)
#summary statistics:
a.info()
a.describe()
#checking for outliers:
for i in range(len(a.columns)):
    plt.boxplot(a.iloc[:,i])
    plt.show()
#checking for the distribution using violin plot:
for i in range(len(a.columns)):
    plt.violinplot(a.iloc[:,i])
    plt.show()
#checking linearity:
sns.pairplot(a)
a.corr(method="spearman")
print(a)
col=a.columns
print(col)
#np_scaled = min_max_scaler.fit_transform(a)
#a_norm = pd.DataFrame(np_scaled, columns = col)
#a_norm
#normalizing the data:
a_nor=preprocessing.normalize(a)
print(a_nor)
a_nor=pd.DataFrame(a_nor)
print(a_nor)
sns.pairplot(a_nor)
a_nor.corr()
a_nor.columns = a.columns
a_nor.head()
x=a_nor.iloc[:,0:4]
y=a_nor.iloc[:,4]
#splitting train set:
X_train,X_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#fitting linear model:
lm=linear_model.LinearRegression()
model=lm.fit(X_train,y_train)
pred=lm.predict(X_train)
print(pred)
print(model.coef_)
print(model.intercept_)
#checking accuracy:
from sklearn.metrics import r2_score
print(r2_score(pred,y_train))
predd = lm.predict(X_test)
print(r2_score(predd,y_test))