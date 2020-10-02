import pandas as pd # data processing 
#Load the dataset
data = pd.read_csv("../input/kc_house_data.csv") 
data.head() # structure of data
data.shape # no of rows * columns
data.columns # check column names
print(len(data))#to check row no
print(len(data.columns))#to check columns no
data.dtypes # looking for data type
print(data.isnull().any()) # looking for null value
# look out for categorical variable data
print("bedrooms:",sorted(data.bedrooms.unique()))
print("bathrooms:",sorted(data.bathrooms.unique()))
print("floors:",sorted(data.floors.unique()))
print("waterfront:",data.waterfront.unique())
print("view:",sorted(data.view.unique()))
print("condition:",sorted(data.condition.unique()))
print("grade:",sorted(data.grade.unique()))
# Create new categorical variable from looking above 
data["waterfront"]=data["waterfront"].astype("category",ordered=True)
data["view"]=data["view"].astype("category",ordered=True)
data["condition"]=data["condition"].astype("category",ordered=True)
data["grade"]=data["grade"].astype("category",ordered=True)
data.dtypes
data=data.drop(["id","date"],axis=1)
data.head()
data.dtypes
data.get_dtype_counts() # count of diff data types
import seaborn as sns
import matplotlib.pyplot as plt
correlation=data.corr()
# plot the heatmap showing calculated correlations
plt.subplots(figsize=(19,19))
plt.title('Pearson Correlation of features')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
#plt.tight_layout()
sns.heatmap(correlation,annot=True,annot_kws={"size":8},linewidths=0.1,cmap="YlGnBu",square=True);
# conclusion 5 most correlated or important features are 
#sqft_living=0.7,
#grade=0.66,
#sqft_above=0.61,
#sqft_living=0.59
#bathrooms=0.52
 # conclusion most irrelevant features are 
 #lat,long,condition,zipcode,sqft_lot,sqft_lot15,yr_built,yr_renovated,floor
 #waterfront,bedrooms,sqft_basement,view
 # Train a simple linear regression model

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data_new=data.drop(["lat","long","condition","zipcode","sqft_lot","sqft_lot15","yr_built","yr_renovated","floors","waterfront","bedrooms","sqft_basement","view"],axis=1)
data_new.columns
data_new.head()
Y=data_new["price"]
X=data_new.drop("price",axis=1)
from sklearn.preprocessing import PolynomialFeatures
# Create Polynomial Features
poly = PolynomialFeatures(2)
X_poly = poly.fit_transform(X)
# Split train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_poly, Y, test_size=0.20)
# Fit model
model = LinearRegression()
model.fit(X_train, Y_train)
from sklearn.metrics import r2_score
# Return R^2
print('Train Score: {:.2f}'.format(model.score(X_train, Y_train)))
print('Test Score: {:.2f}'.format(model.score(X_test, Y_test)))
