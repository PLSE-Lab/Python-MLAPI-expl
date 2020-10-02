import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


#save filepath to variable for easier access
asteroid_file_path = '../input/prediction-of-asteroid-diameter/Asteroid.csv'

#read the data and store data in DataFrame titled asteroid_data
asteroid_data = pd.read_csv(asteroid_file_path)

#print a summary of the data in Asteroid data
#asteroid_data.describe()

#dropna drops missing values
asteroid_data = asteroid_data.dropna(axis=0)

#prediction target
y = asteroid_data.diameter

# a = semi-major axis(au)
# e = eccentricity
# G = Magnitude slope parameter
# i = Inclination with respect to x-y ecliptic plane(deg)
# om = Longitude of the ascending node
# w = argument of perihelion
# q = perihelion distance(au)
# ad = aphelion distance(au)
# per_y = Orbital period
# data_arc = data arc-span(d)
# condition_code = Orbit condition code
# n_obs_used = number of observations used
# H = Absolute Magnitude parameter
# diameter = Diameter of asteroid(Km)
#extent = Object bi/tri axial ellipsoid dimensions(Km)
#albedo = geometric albedo
#rot_per = Rotation Period(h)
#GM = Standard gravitational parameter, Product of mass and gravitational constant
#BV = Color index B-V magnitude difference
#UB = Color index U-B magnitude difference
#IR = Color index I-R magnitude difference
#spec_B = Spectral taxonomic type(SMASSII)
#spec_T = Spectral taxonomic type(Tholen)
#neo = Near Earth Object
#pha = Physically Hazardous Asteroid
#moid = Earth Minimum orbit Intersection Distance(au)

asteroid_features = ['a','e','G','om','w','q','ad','per_y','albedo','rot_per','GM','BV','UB','moid']

X = asteroid_data[asteroid_features]

#X.describe()
#X.head()

#Define model. Specify a number for random_state to ensure same results each run
asteroid_model = DecisionTreeRegressor(random_state=1)

#Fit model
asteroid_model.fit(X, y)

print("Making predictions for the following 5 asteroids:")
print(X.head())
print("The predictions are")
print(asteroid_model.predict(X.head()))

#predicted_diameters = asteroid_model.predict(X)
#mean_absolute_error(y, predicted_diameters)

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# # Define model
# asteroid_model = DecisionTreeRegressor()
# # Fit model
# asteroid_model.fit(train_X, train_y)

# get predicted prices on validation data
# val_predictions = asteroid_model.predict(val_X)
# print(mean_absolute_error(val_y, val_predictions))

# # def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
#     model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
#     model.fit(train_X, train_y)
#     preds_val = model.predict(val_X)
#     mae = mean_absolute_error(val_y, preds_val)
#     return(mae)

#for max_leaf_nodes in [5, 25, 50, 100, 250, 500, 1000, 5000]:
#     my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
#     print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

# #best_tree_size =

# Fill in argument to make optimal size and uncomment
#final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

# fit the final model and uncomment the next two lines
#final_model.fit(X, y)


# forest_model = RandomForestRegressor(random_state=1)
# forest_model.fit(train_X, train_y)
# dia_preds = forest_model.predict(val_X)
# print(mean_absolute_error(val_y, dia_preds))
